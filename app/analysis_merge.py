from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

import yaml
from .analyzer_nudenet import NudeNetAnalyzer
from .batch_loader import ImageRequest, load_images
from .cache_nudenet import CacheKey as NudeCacheKey, NudeNetCache
from .schema import MessageRef, NudityDetection, parse_bool


NSFW_GENERAL_TAGS_DEFAULT = {
    "bikini",
    "lingerie",
    "underwear",
    "panties",
    "bra",
    "swimsuit",
    "swimwear",
    "naked",
}

STRONG_PREFIXES = (
    "EXPOSED_GENITALIA",
    "EXPOSED_BREAST_",
    "EXPOSED_BUTTOCKS",
    "EXPOSED_ANUS",
)

WEAK_PREFIXES = (
    "COVERED_BREAST_",
    "COVERED_GENITALIA",
    "COVERED_BUTTOCKS",
    "EXPOSED_BELLY",
)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge WD14 and NudeNet analysis")
    parser.add_argument("--scan", type=Path, default=Path("out/p0_scan.csv"))
    parser.add_argument("--wd14", type=Path, default=Path("out/p1_wd14.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("out/p2_analysis.jsonl"))
    parser.add_argument("--metrics", type=Path, default=Path("out/p2_metrics.json"))
    parser.add_argument("--nudenet-cache", type=Path, default=Path("app/cache_nudenet.sqlite"))
    parser.add_argument("--nudenet-config", type=Path, default=Path("configs/nudenet.yaml"))
    parser.add_argument("--xsignals-config", type=Path, default=Path("configs/xsignals.yaml"))
    parser.add_argument("--qps", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, help="Overrides NudeNet batch size")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def load_scan_metadata(csv_path: Path, limit: int = 0) -> dict[str, dict]:
    entries: dict[str, dict] = {}
    with csv_path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            phash = (row.get("phash_hex") or "").strip()
            if not phash:
                continue
            url = (row.get("url") or "").strip()
            if not url:
                continue
            entry = entries.setdefault(
                phash,
                {
                    "rows": [],
                    "urls": set(),
                },
            )
            entry["rows"].append(row)
            entry["urls"].add(url)
            if limit and len(entries) >= limit:
                break
    return entries


def load_wd14(path: Path, limit: int = 0) -> dict[str, dict]:
    data: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            if not line.strip():
                continue
            payload = json.loads(line)
            phash = payload.get("phash")
            if not phash:
                continue
            data[phash] = payload
            if limit and len(data) >= limit:
                break
    return data


def to_message_ref(row: dict) -> MessageRef:
    return MessageRef(
        message_link=row.get("message_link", ""),
        message_id=row.get("message_id", ""),
        channel_id=row.get("channel_id", ""),
        guild_id=row.get("guild_id", ""),
        source=row.get("source", ""),
        url=row.get("url", ""),
        author_id=row.get("author_id"),
        is_nsfw_channel=parse_bool(row.get("is_nsfw_channel")),
    )


def summarize_messages(rows: Iterable[dict]) -> tuple[MessageRef, list[MessageRef]]:
    refs = [to_message_ref(row) for row in rows]
    primary = refs[0]
    return primary, refs


def keep_topk_detections(detections: list[NudityDetection], topk: int) -> list[NudityDetection]:
    return sorted(detections, key=lambda det: det.score, reverse=True)[:topk]


def compute_exposure_score(
    detections: list[NudityDetection],
    thresholds: dict,
    weights: dict,
) -> float:
    strong_threshold = float(thresholds.get("strong_exposed", 0.6))
    weak_threshold = float(thresholds.get("weak_exposed", 0.5))
    strong_scores = [
        det.score
        for det in detections
        if det.score >= strong_threshold and _matches_prefix(det.cls, STRONG_PREFIXES)
    ]
    weak_scores = [
        det.score
        for det in detections
        if det.score >= weak_threshold and _matches_prefix(det.cls, WEAK_PREFIXES)
    ]
    strong_component = (max(strong_scores) if strong_scores else 0.0) * float(weights.get("strong_weight", 1.0))
    weak_component = (max(weak_scores) if weak_scores else 0.0) * float(weights.get("weak_weight", 0.6))
    return round(strong_component + weak_component, 6)


def compute_placement_risk(
    wd14: dict,
    exposure_score: float,
    cfg: dict,
) -> float:
    rating_weight = float(cfg.get("rating_weight", 0.5))
    general_weight = float(cfg.get("general_weight", 0.3))
    exposure_weight = float(cfg.get("exposure_weight", 0.7))
    topk = int(cfg.get("topk", 3))
    nsfw_tags = set(cfg.get("nsfw_tags", NSFW_GENERAL_TAGS_DEFAULT))

    rating = wd14.get("rating", {})
    rating_component = max(float(rating.get("questionable", 0.0)), float(rating.get("explicit", 0.0))) * rating_weight

    general_tags = wd14.get("general", [])
    scores: List[float] = []
    for item in general_tags:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            tag, score = item
        else:
            continue
        if tag in nsfw_tags:
            scores.append(float(score))
    general_component = 0.0
    if scores:
        general_component = mean(sorted(scores, reverse=True)[:topk]) * general_weight

    return round(rating_component + general_component + exposure_score * exposure_weight, 6)


def _matches_prefix(label: str, prefixes: Iterable[str]) -> bool:
    return any(label.startswith(prefix) for prefix in prefixes)


async def async_main() -> None:
    args = parse_args()
    scan_entries = load_scan_metadata(args.scan, args.limit)
    wd14_entries = load_wd14(args.wd14, args.limit)
    nudenet_cfg = load_yaml(args.nudenet_config)
    xsignals_cfg = load_yaml(args.xsignals_config)

    overlay_thresholds = nudenet_cfg.get("thresholds", {})
    keep_topk = int(nudenet_cfg.get("keep_topk", 10))
    batch_size = args.batch_size or int(nudenet_cfg.get("batch_size", 8))

    exposure_weights = xsignals_cfg.get("exposure_score", {})
    placement_cfg = xsignals_cfg.get("placement_risk_pre", {})

    analyzer = NudeNetAnalyzer()
    cache = NudeNetCache(args.nudenet_cache)

    phashes = [ph for ph in wd14_entries if ph in scan_entries]

    cache_hits = 0
    failures: Dict[str, str] = {}
    total_latency_ms = 0.0
    processed = 0

    records: list[dict] = []

    missing: list[str] = []
    cached_payloads: dict[str, dict] = {}

    for phash in phashes:
        key = NudeCacheKey(phash=phash, model="nudenet", version=analyzer.version)
        cached = cache.get(key)
        if cached is not None:
            cache_hits += 1
            cached_payloads[phash] = cached
        else:
            missing.append(phash)

    async def process_batch(batch_phashes: list[str]) -> None:
        nonlocal total_latency_ms, processed
        requests = [
            ImageRequest(
                identifier=phash,
                urls=tuple(scan_entries[phash]["urls"]),
            )
            for phash in batch_phashes
        ]
        results = await load_images(requests, qps=args.qps, concurrency=args.concurrency)
        ok_results = []
        for result in results:
            if result.image is None:
                failures[result.request.identifier] = result.note or "fetch_failed"
            else:
                ok_results.append(result)
        if not ok_results:
            return
        start = time.perf_counter()
        detections = await asyncio.to_thread(analyzer.detect_batch, [res.image for res in ok_results])
        elapsed = (time.perf_counter() - start) * 1000
        total_latency_ms += elapsed
        processed += len(ok_results)
        for res, dets in zip(ok_results, detections):
            payload = {
                "detections": [
                    {
                        "class": det.label,
                        "score": det.score,
                        "box": det.box,
                    }
                    for det in dets
                ],
                "version": analyzer.version,
            }
            cache.set(
                NudeCacheKey(phash=res.request.identifier, model="nudenet", version=analyzer.version),
                payload,
            )
            cached_payloads[res.request.identifier] = payload

    for i in range(0, len(missing), batch_size):
        await process_batch(missing[i : i + batch_size])

    for phash in phashes:
        scan_entry = scan_entries[phash]
        wd14_entry = wd14_entries[phash]
        primary, messages = summarize_messages(scan_entry["rows"])

        wd14_payload = wd14_entry.get("wd14", {})
        nudity_payload = cached_payloads.get(phash, {"detections": []})
        raw_detections = [
            NudityDetection(
                cls=item.get("class", ""),
                score=float(item.get("score", 0.0)),
                box=item.get("box"),
            )
            for item in nudity_payload.get("detections", [])
        ]
        reduced_detections = keep_topk_detections(raw_detections, keep_topk)
        exposure_score = compute_exposure_score(reduced_detections, overlay_thresholds, exposure_weights)
        placement_risk = compute_placement_risk(wd14_payload, exposure_score, placement_cfg)

        record = {
            "phash": phash,
            "guild_id": primary.guild_id,
            "channel_id": primary.channel_id,
            "message_id": primary.message_id,
            "message_link": primary.message_link,
            "source": primary.source,
            "is_nsfw_channel": primary.is_nsfw_channel,
            "messages": [
                {
                    "message_link": msg.message_link,
                    "message_id": msg.message_id,
                    "channel_id": msg.channel_id,
                    "guild_id": msg.guild_id,
                    "source": msg.source,
                    "url": msg.url,
                    "author_id": msg.author_id,
                    "is_nsfw_channel": msg.is_nsfw_channel,
                }
                for msg in messages
            ],
            "wd14": wd14_payload,
            "nudity_detections": [
                {
                    "class": det.cls,
                    "score": det.score,
                    "box": det.box,
                }
                for det in reduced_detections
            ],
            "xsignals": {
                "exposure_score": exposure_score,
                "placement_risk_pre": placement_risk,
            },
            "meta": {
                "nudenet_version": analyzer.version,
                "wd_model": wd14_payload.get("model"),
                "wd_revision": wd14_payload.get("revision"),
                "wd_input_size": wd14_payload.get("input_size"),
                "nudenet_error": failures.get(phash),
            },
        }
        records.append(record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fp:
        for record in records:
            json.dump(record, fp, ensure_ascii=False)
            fp.write("\n")

    if processed:
        avg_latency = total_latency_ms / processed
    else:
        avg_latency = 0.0

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_records": len(records),
        "from_cache": cache_hits,
        "nudity_failures": failures,
        "nudenet_version": analyzer.version,
        "average_nudenet_latency_ms": round(avg_latency, 2),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
