from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Range:
    start: datetime | None
    end: datetime | None


def parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def parse_args() -> tuple[Path, Path, Range]:
    parser = argparse.ArgumentParser(
        description="Filter p0 scan CSV by created_at timestamp",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("out/p0_scan.csv"),
        help="Source CSV file from p0 scan",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination CSV to write filtered rows",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Inclusive ISO8601 lower bound (UTC assumed when tz missing)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="Inclusive ISO8601 upper bound (UTC assumed when tz missing)",
    )
    args = parser.parse_args()

    start_dt = parse_iso8601(args.start)
    end_dt = parse_iso8601(args.end)
    if start_dt and end_dt and start_dt > end_dt:
        parser.error("--start must be <= --end")

    return args.input, args.out, Range(start=start_dt, end=end_dt)


def should_keep(created_at: str, interval: Range) -> bool:
    created_at = created_at.strip()
    if not created_at:
        return False
    dt = parse_iso8601(created_at)
    if dt is None:
        return False
    if interval.start and dt < interval.start:
        return False
    if interval.end and dt > interval.end:
        return False
    return True


def filter_csv(source: Path, destination: Path, interval: Range) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", newline="") as src, destination.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        if reader.fieldnames is None:
            return 0, 0
        writer.writeheader()
        total = 0
        kept = 0
        for row in reader:
            total += 1
            created_at = row.get("created_at", "")
            if should_keep(created_at, interval):
                writer.writerow(row)
                kept += 1
        return total, kept


def main() -> None:
    source, destination, interval = parse_args()
    if not source.exists():
        raise SystemExit(f"Input file not found: {source}")
    total, kept = filter_csv(source, destination, interval)
    print(f"Processed {total} rows, kept {kept}")


if __name__ == "__main__":
    main()

