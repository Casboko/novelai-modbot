from __future__ import annotations

import argparse
from pathlib import Path

from .config import get_settings
from .profiles import PartitionPaths
from .triage import generate_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate moderation report CSV")
    parser.add_argument("--findings", type=Path, help="Findings JSONL path")
    parser.add_argument("--out", type=Path, help="Report CSV output path")
    parser.add_argument("--severity", type=str, default="all", help="Severity filter: all/red/orange/yellow/green")
    parser.add_argument("--rules", type=Path, default=Path("configs/rules_v2.yaml"))
    parser.add_argument("--channel", action="append", help="Channel ID to include (repeatable)")
    parser.add_argument(
        "--since",
        type=str,
        help="開始日（YYYY-MM-DD または 7d, 24h などの相対指定）",
    )
    parser.add_argument(
        "--until",
        type=str,
        help="終了日（未指定は当日+1日まで含む。YYYY-MM-DD や 24h 等も可）",
    )
    parser.add_argument("--profile", type=str, help="Profile name for partition defaults")
    parser.add_argument(
        "--date",
        type=str,
        help="Partition date (YYYY-MM-DD, today, yesterday). Default resolved via profile timezone",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    context = settings.build_profile_context(profile=args.profile, date=args.date)
    partitions = PartitionPaths(context)
    if args.findings is None:
        args.findings = partitions.stage_file("p3")
    if args.out is None:
        args.out = partitions.report_path()
    summary = generate_report(
        args.findings,
        args.out,
        args.severity,
        args.rules,
        channel_ids=args.channel,
        since=args.since,
        until=args.until,
    )
    print(f"rows={summary.rows} path={summary.path}")


if __name__ == "__main__":
    main()
