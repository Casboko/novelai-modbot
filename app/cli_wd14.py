from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from dotenv import load_dotenv
from PIL import Image
import yaml

load_dotenv()

from .analyzer_wd14 import WD14Analyzer, WD14Prediction, WD14Session
from .batch_loader import ImageLoadResult, ImageRequest, load_images
from .cache_wd14 import CacheKey, WD14Cache
from .config import get_settings
from .config.rules_dicts import RulesDictError, extract_nsfw_general_tags
from .io.stream import SortMode
from .labelspace import LabelSpace, ensure_local_files, load_labelspace, REPO_ID
from .local_cache import resolve_local_file
from .jsonl_merge import merge_jsonl_records
from .profiles import LEGACY_PROFILE, PartitionPaths


@dataclass(slots=True)
class Entry:
    phash: str
    urls: set[str]
    rows: list[dict[str, str]]


DEFAULT_CACHE_PATH = Path("app/cache_wd14.sqlite")
DEFAULT_INPUT_PATTERN = "p0_*.csv"
LOG_LEVEL_CHOICES = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]


def load_rules_nsfw_tags(path: Path) -> set[str]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return set()
    except yaml.YAMLError:
        return set()
    if not isinstance(data, dict):
        return set()
    try:
        tags = extract_nsfw_general_tags(data, strict=True)
    except RulesDictError as exc:
        print(f"[cli_wd14] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    return set(tags)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WD14 EVA02 batch inference")
    default_provider = os.getenv("WD14_PROVIDER", "cpu")
    log_level_default = os.getenv("LOG_LEVEL")
    if log_level_default:
        log_level_default = log_level_default.upper()
    parser.add_argument("--input", type=Path, help="Input CSV or directory of CSVs")
    parser.add_argument(
        "--input-pattern",
        type=str,
        default=None,
        help="Glob pattern when --input is a directory (default p0_*.csv)",
    )
    parser.add_argument(
        "--input-sort",
        type=str,
        choices=["name", "mtime", "reverse-name", "reverse-mtime"],
        default="name",
        help="Ordering for matched input files",
    )
    parser.add_argument("--out", type=Path, help="Output JSONL path override")
    parser.add_argument("--metrics", type=Path, help="Metrics JSON path override")
    parser.add_argument("--cache", type=Path, help="Cache database override")
    parser.add_argument("--model-id", type=str, default=REPO_ID)
    parser.add_argument("--model-revision", type=str, default="main")
    parser.add_argument(
        "--provider",
        type=str,
        default=default_provider,
        choices=["cpu", "openvino", "cuda", "tensorrt"],
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--general-threshold", type=float, default=0.35)
    parser.add_argument("--character-threshold", type=float, default=0.85)
    parser.add_argument("--raw-general-topk", type=int, default=64)
    parser.add_argument("--rules-config", type=Path, default=Path("configs/rules_v2.yaml"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--qps", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of unique pHashes")
    parser.add_argument(
        "--log-level",
        type=lambda value: value.upper(),
        choices=LOG_LEVEL_CHOICES,
        default=log_level_default,
        help="Optional logging level override",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge results into existing JSONL instead of overwriting",
    )
    parser.add_argument("--profile", type=str, help="Profile name for partition defaults")
    parser.add_argument(
        "--date",
        type=str,
        help="Partition date (YYYY-MM-DD, today, yesterday). Default resolved via profile timezone",
    )
    return parser.parse_args()


def load_entries(csv_paths: Sequence[Path] | Path, limit: int = 0) -> dict[str, Entry]:
    if isinstance(csv_paths, Path):
        paths: list[Path] = [csv_paths]
    else:
        paths = list(csv_paths)
    entries: dict[str, Entry] = {}
    for csv_path in paths:
        with csv_path.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                phash = row.get("phash_hex", "").strip()
                if not phash:
                    continue
                url = row.get("url", "").strip()
                if not url:
                    continue
                entry = entries.setdefault(phash, Entry(phash=phash, urls=set(), rows=[]))
                entry.urls.add(url)
                entry.rows.append(row)
                if limit and len(entries) >= limit:
                    return entries
    return entries


def build_requests(entries: dict[str, Entry], missing: Iterable[str]) -> list[ImageRequest]:
    requests: list[ImageRequest] = []
    for phash in missing:
        entry = entries[phash]
        urls = list(entry.urls)
        local_path = resolve_local_file(phash)
        if local_path:
            local_uri = Path(local_path).as_uri()
            if local_uri in urls:
                urls.remove(local_uri)
            urls.insert(0, local_uri)
        requests.append(ImageRequest(identifier=phash, urls=tuple(urls)))
    return requests


def chunked(sequence: list[str], size: int) -> Iterable[list[str]]:
    for idx in range(0, len(sequence), size):
        yield sequence[idx : idx + size]


def _sort_paths(candidates: Iterable[Path], sort: SortMode) -> list[Path]:
    files = [path for path in candidates if path.is_file()]
    if sort in ("name", "reverse-name"):
        files.sort(key=lambda item: item.name)
    elif sort in ("mtime", "reverse-mtime"):
        files.sort(key=lambda item: item.stat().st_mtime)
    else:
        files.sort(key=lambda item: item.name)
    if sort.startswith("reverse"):
        files.reverse()
    return files


def _resolve_input_paths(args: argparse.Namespace, partitions: PartitionPaths) -> list[Path]:
    pattern = args.input_pattern or DEFAULT_INPUT_PATTERN
    sort_mode: SortMode = args.input_sort  # type: ignore[assignment]
    if args.input:
        if args.input.is_dir():
            resolved = _sort_paths(args.input.glob(pattern), sort_mode)
        else:
            if not args.input.exists():
                raise SystemExit(f"Input path not found: {args.input}")
            resolved = [args.input]
    else:
        candidate = partitions.stage_file("p0")
        if candidate.exists():
            resolved = [candidate]
        else:
            resolved = _sort_paths(partitions.stage_dir("p0").glob(pattern), sort_mode)
    if not resolved:
        raise SystemExit("No input CSV files found for WD14 inference")
    return resolved


def serialize_prediction(
    phash: str,
    entry: Entry,
    payload: dict,
) -> dict:
    message_refs: list[dict[str, str]] = []
    for row in entry.rows:
        message_refs.append(
            {
                "message_link": row.get("message_link", ""),
                "message_id": row.get("message_id", ""),
                "channel_id": row.get("channel_id", ""),
                "guild_id": row.get("guild_id", ""),
                "source": row.get("source", ""),
                "url": row.get("url", ""),
            }
        )
    return {
        "phash": phash,
        "messages": message_refs,
        "wd14": payload,
    }


def build_payload(
    prediction: WD14Prediction,
    *,
    model_id: str,
    model_revision: str,
    input_size: int,
) -> dict:
    return {
        "model": model_id,
        "revision": model_revision,
        "input_size": input_size,
        "rating": prediction.rating,
        "general": prediction.general,
        "general_raw": prediction.general_raw,
        "character": prediction.character,
        "raw": prediction.raw_scores.tolist(),
    }


async def run() -> None:
    args = parse_args()

    if args.log_level:
        logging.basicConfig(level=getattr(logging, args.log_level, logging.WARNING))

    settings = get_settings()
    context = settings.build_profile_context(profile=args.profile, date=args.date)
    partitions = PartitionPaths(context)

    input_paths = _resolve_input_paths(args, partitions)

    if args.out is None:
        args.out = partitions.stage_file("p1")
    if args.metrics is None:
        args.metrics = partitions.metrics_file("p1")

    cache_suffix = os.getenv("WD14_CACHE_SUFFIX", "").strip()
    if args.cache is None:
        if cache_suffix:
            cache_name = f"{DEFAULT_CACHE_PATH.stem}_{cache_suffix}{DEFAULT_CACHE_PATH.suffix}"
            args.cache = DEFAULT_CACHE_PATH.with_name(cache_name)
        elif context.profile == LEGACY_PROFILE:
            legacy_name = f"{DEFAULT_CACHE_PATH.stem}_{LEGACY_PROFILE}{DEFAULT_CACHE_PATH.suffix}"
            args.cache = DEFAULT_CACHE_PATH.with_name(legacy_name)
        else:
            args.cache = DEFAULT_CACHE_PATH
    elif cache_suffix and args.cache == DEFAULT_CACHE_PATH:
        cache_name = f"{DEFAULT_CACHE_PATH.stem}_{cache_suffix}{DEFAULT_CACHE_PATH.suffix}"
        args.cache = DEFAULT_CACHE_PATH.with_name(cache_name)

    entries = load_entries(input_paths, args.limit)
    if not entries:
        raise SystemExit("No entries found in CSV. Did you run p0 scan?")

    nsfw_whitelist = load_rules_nsfw_tags(args.rules_config)
    label_csv, model_path = ensure_local_files(repo_id=args.model_id)
    labelspace = load_labelspace(label_csv)
    session = WD14Session(model_path, provider=args.provider, threads=args.threads)
    analyzer = WD14Analyzer(
        session,
        labelspace,
        general_threshold=args.general_threshold,
        character_threshold=args.character_threshold,
        raw_general_topk=args.raw_general_topk,
        raw_general_whitelist=nsfw_whitelist,
    )

    cache = WD14Cache(Path(args.cache))
    cache_hits = 0
    cache_key_map: dict[str, dict] = {}
    failures: dict[str, str] = {}

    all_phashes = list(entries.keys())

    total_infer_seconds = 0.0
    inferred_images = 0

    for phash in all_phashes:
        key = CacheKey(phash=phash, model=args.model_id, revision=args.model_revision)
        cached = cache.get(key)
        if cached is not None:
            cache_hits += 1
            if "general_raw" not in cached:
                raw_scores = cached.get("raw")
                if isinstance(raw_scores, list) and raw_scores:
                    cached_general_raw = analyzer.general_raw_from_scores(raw_scores)
                    cached["general_raw"] = cached_general_raw
                    cache.set(key, cached)
            cache_key_map[phash] = cached

    missing = [phash for phash in all_phashes if phash not in cache_key_map]

    for batch_phashes in chunked(missing, args.batch_size):
        requests = build_requests(entries, batch_phashes)
        results = await load_images(
            requests,
            qps=args.qps,
            concurrency=args.concurrency,
        )
        successful_results: list[ImageLoadResult] = [r for r in results if r.image is not None]
        images: list[Image.Image] = [r.image for r in successful_results]
        if images:
            start = time.perf_counter()
            predictions = analyzer.predict(images)
            total_infer_seconds += time.perf_counter() - start
            inferred_images += len(images)
        else:
            predictions = []
        for result, prediction in zip(successful_results, predictions):
            payload = build_payload(
                prediction,
                model_id=args.model_id,
                model_revision=args.model_revision,
                input_size=session.size,
            )
            cache_key_map[result.request.identifier] = payload
            cache.set(
                CacheKey(
                    phash=result.request.identifier,
                    model=args.model_id,
                    revision=args.model_revision,
                ),
                payload,
            )
        for result in results:
            if result.image is None:
                reason = result.note or "fetch_failed"
                failures[result.request.identifier] = reason
                cache_key_map.setdefault(
                    result.request.identifier,
                    {
                        "model": args.model_id,
                        "revision": args.model_revision,
                        "error": reason,
                    },
                )

    records: dict[str, dict] = {}
    for phash in all_phashes:
        payload = cache_key_map.get(phash)
        if payload is None:
            continue
        entry = entries[phash]
        records[phash] = serialize_prediction(phash, entry, payload)

    if args.merge_existing:
        merge_jsonl_records(
            base_path=args.out,
            updates=records,
            key_field="phash",
            out_path=args.out,
        )
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fp:
            for phash in all_phashes:
                record = records.get(phash)
                if record is None:
                    continue
                json.dump(record, fp, ensure_ascii=False)
                fp.write("\n")

    success_count = sum(1 for payload in cache_key_map.values() if "error" not in payload)
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_unique": len(all_phashes),
        "from_cache": cache_hits,
        "evaluated": success_count,
        "failed": failures,
        "model": args.model_id,
        "revision": args.model_revision,
        "input_size": session.size,
        "images_processed": inferred_images,
    }
    if inferred_images > 0 and total_infer_seconds > 0.0:
        metrics["infer_ms_avg"] = (total_infer_seconds * 1000.0) / inferred_images
        metrics["img_per_sec"] = inferred_images / total_infer_seconds
    else:
        metrics["infer_ms_avg"] = 0.0
        metrics["img_per_sec"] = 0.0
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
