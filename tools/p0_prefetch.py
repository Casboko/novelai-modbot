from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import mimetypes
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import aiohttp

from app.http_client import ImageFetcher, RateLimiter


MANIFEST_FILENAME = "p0_images.csv"


@dataclass(slots=True)
class PrefetchArgs:
    p0_path: Path
    cache_root: Path
    manifest_path: Path
    qps: float
    concurrency: int
    retry: bool


@dataclass(slots=True)
class PrefetchEntry:
    phash: str
    urls: list[str]


@dataclass(slots=True)
class PrefetchResult:
    phash: str
    status: str
    local_path: str
    content_type: str
    size: int
    sha256: str
    mtime: str
    from_url: str


def determine_cache_root(cli_value: Optional[Path]) -> Path:
    if cli_value is not None:
        return cli_value.resolve()
    env_value = os.getenv("CACHE_ROOT")
    if env_value:
        return Path(env_value).resolve()
    workspace = Path("/workspace/cache")
    if workspace.exists():
        return workspace.resolve()
    return Path("./cache").resolve()


def parse_args() -> PrefetchArgs:
    parser = argparse.ArgumentParser(description="Prefetch original images referenced by p0 scan")
    parser.add_argument("--p0", type=Path, default=Path("out/p0_scan.csv"))
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--manifest", type=Path, help="Override manifest output path")
    parser.add_argument("--qps", type=float, default=8.0, help="Max HTTP requests per second")
    parser.add_argument("--concurrency", type=int, default=16, help="Concurrent fetch workers")
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Retry entries whose manifest status is miss/error",
    )
    ns = parser.parse_args()

    cache_root = determine_cache_root(ns.cache_root)
    manifest_dir = cache_root / "manifest"
    manifest_path = ns.manifest if ns.manifest else manifest_dir / MANIFEST_FILENAME
    return PrefetchArgs(
        p0_path=ns.p0,
        cache_root=cache_root,
        manifest_path=manifest_path,
        qps=max(ns.qps, 1.0),
        concurrency=max(ns.concurrency, 1),
        retry=bool(ns.retry),
    )


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    records: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            phash = (row.get("phash") or "").strip()
            if not phash:
                continue
            records[phash] = row
    return records


def write_manifest(path: Path, records: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phash",
        "local_path",
        "content_type",
        "bytes",
        "sha256",
        "mtime",
        "from_url",
        "status",
    ]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for phash in sorted(records):
            row = records[phash]
            writer.writerow(
                {
                    "phash": phash,
                    "local_path": row.get("local_path", ""),
                    "content_type": row.get("content_type", ""),
                    "bytes": row.get("bytes", ""),
                    "sha256": row.get("sha256", ""),
                    "mtime": row.get("mtime", ""),
                    "from_url": row.get("from_url", ""),
                    "status": row.get("status", ""),
                }
            )
    tmp_path.replace(path)


def load_entries(csv_path: Path) -> dict[str, PrefetchEntry]:
    entries: dict[str, PrefetchEntry] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            phash = (row.get("phash_hex") or "").strip()
            url = (row.get("url") or "").strip()
            if not phash or not url:
                continue
            entry = entries.get(phash)
            if entry is None:
                entry = PrefetchEntry(phash=phash, urls=[])
                entries[phash] = entry
            if url not in entry.urls:
                entry.urls.append(url)
    return entries


def guess_extension(content_type: str | None, url: str) -> str:
    if content_type:
        primary = content_type.split(";", 1)[0].strip().lower()
        ext = mimetypes.guess_extension(primary) if primary else None
        if ext:
            return ext
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix
    if suffix:
        return suffix
    return ".bin"


def file_mtime_iso(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
    except FileNotFoundError:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def should_prefetch(record: Optional[dict[str, str]], retry: bool) -> bool:
    if record is None:
        return True
    status = (record.get("status") or "").strip().lower()
    local_path = (record.get("local_path") or "").strip()
    if status == "ok" and local_path:
        if Path(local_path).is_file():
            return False
        return True
    if status in {"", "ok"}:
        return True
    if status.startswith("error") or status == "miss":
        return retry
    return retry


async def perform_prefetch(args: PrefetchArgs) -> None:
    entries = load_entries(args.p0_path)
    if not entries:
        raise SystemExit("p0 CSV に画像エントリが見つかりませんでした")

    manifest_records = load_manifest(args.manifest_path)

    full_dir = args.cache_root / "full"
    manifest_dir = args.cache_root / "manifest"
    full_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    targets: list[PrefetchEntry] = []
    for phash, entry in entries.items():
        record = manifest_records.get(phash)
        if should_prefetch(record, args.retry):
            targets.append(entry)

    if not targets:
        print("[prefetch] 対象の画像はありませんでした。")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        fetcher = ImageFetcher(session, RateLimiter(args.qps))
        tasks = [
            asyncio.create_task(
                _process_entry(entry, fetcher, semaphore, full_dir, manifest_records.get(entry.phash))
            )
            for entry in targets
        ]
        results = await asyncio.gather(*tasks)

    for result in results:
        manifest_records[result.phash] = {
            "phash": result.phash,
            "local_path": result.local_path,
            "content_type": result.content_type,
            "bytes": str(result.size) if result.size else "",
            "sha256": result.sha256,
            "mtime": result.mtime,
            "from_url": result.from_url,
            "status": result.status,
        }

    write_manifest(args.manifest_path, manifest_records)

    success = sum(1 for r in results if r.status == "ok")
    failures = len(results) - success
    print(f"[prefetch] 完了: {success}件成功, {failures}件失敗")


async def _process_entry(
    entry: PrefetchEntry,
    fetcher: ImageFetcher,
    semaphore: asyncio.Semaphore,
    full_dir: Path,
    previous_record: Optional[dict[str, str]],
) -> PrefetchResult:
    async with semaphore:
        return await _download_entry(entry, fetcher, full_dir, previous_record)


async def _download_entry(
    entry: PrefetchEntry,
    fetcher: ImageFetcher,
    full_dir: Path,
    previous_record: Optional[dict[str, str]],
) -> PrefetchResult:
    urls = list(entry.urls)
    last_note: str | None = None
    for url in urls:
        fetch = await fetcher.fetch(url)
        if fetch.data is None:
            last_note = fetch.note or last_note or "miss"
            continue
        content_type = fetch.content_type or "application/octet-stream"
        data = fetch.data
        dest_path = _determine_destination(entry.phash, url, content_type, full_dir, previous_record)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
        tmp_path.write_bytes(data)
        tmp_path.replace(dest_path)
        return PrefetchResult(
            phash=entry.phash,
            status="ok",
            local_path=str(dest_path),
            content_type=content_type,
            size=len(data),
            sha256=sha256_hex(data),
            mtime=file_mtime_iso(dest_path),
            from_url=url,
        )

    status = "miss" if (last_note or "").startswith("miss") else f"error:{last_note or 'unknown'}"
    return PrefetchResult(
        phash=entry.phash,
        status=status,
        local_path="",
        content_type="",
        size=0,
        sha256="",
        mtime="",
        from_url="",
    )


def _determine_destination(
    phash: str,
    used_url: str,
    content_type: str | None,
    full_dir: Path,
    previous_record: Optional[dict[str, str]],
) -> Path:
    if previous_record:
        existing = (previous_record.get("local_path") or "").strip()
        if existing:
            existing_path = Path(existing)
            if existing_path.exists():
                return existing_path
    ext = guess_extension(content_type, used_url)
    if not ext.startswith("."):
        ext = f".{ext}"
    return full_dir / f"{phash}{ext}"


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(perform_prefetch(args))
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
