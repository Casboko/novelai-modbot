from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split p0 index CSV into shards")
    parser.add_argument("--input", type=Path, default=Path("out/p0_scan.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("out/p0"))
    parser.add_argument("--shards", type=int, default=1, help="Number of shards to produce")
    parser.add_argument("--force", action="store_true", help="Overwrite existing shard files")
    return parser.parse_args()


def ensure_output_paths(out_dir: Path, shard_count: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    width = max(2, int(math.log10(max(shard_count - 1, 1))) + 1)
    shard_paths = [out_dir / f"shard_{index:0{width}d}.csv" for index in range(shard_count)]
    return shard_paths


def detect_existing(shard_paths: Iterable[Path]) -> list[Path]:
    return [path for path in shard_paths if path.exists()]


def load_rows(input_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with input_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise SystemExit("Input CSV has no header")
        rows = [row for row in reader]
    return reader.fieldnames, rows


def split_rows(rows: list[dict[str, str]], shard_count: int) -> list[list[dict[str, str]]]:
    total = len(rows)
    if shard_count <= 0:
        raise SystemExit("--shards must be >= 1")
    if total == 0:
        return [[] for _ in range(shard_count)]
    base = total // shard_count
    remainder = total % shard_count
    result: list[list[dict[str, str]]] = []
    offset = 0
    for index in range(shard_count):
        length = base + (1 if index < remainder else 0)
        shard_rows = rows[offset : offset + length]
        result.append(shard_rows)
        offset += length
    return result


def write_shard(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    shard_paths = ensure_output_paths(args.out_dir, args.shards)
    existing = detect_existing(shard_paths)
    if existing and not args.force:
        existing_list = ", ".join(str(path) for path in existing[:5])
        if len(existing) > 5:
            existing_list += ", ..."
        raise SystemExit(
            "Shard files already exist. Use --force to overwrite: "
            f"{existing_list}"
        )

    header, rows = load_rows(args.input)
    shards = split_rows(rows, args.shards)

    for path, shard_rows in zip(shard_paths, shards):
        write_shard(path, header, shard_rows)


if __name__ == "__main__":
    main()
