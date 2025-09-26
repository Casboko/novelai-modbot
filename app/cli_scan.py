from __future__ import annotations

import argparse
from pathlib import Path

from .triage import run_scan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rule evaluation on analysis output")
    parser.add_argument("--analysis", type=Path, default=Path("out/p2_analysis.jsonl"))
    parser.add_argument("--findings", type=Path, default=Path("out/p3_findings.jsonl"))
    parser.add_argument("--rules", type=Path, default=Path("configs/rules.yaml"))
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
    parser.add_argument(
        "--severity",
        type=str,
        default="all",
        help="Filter severities (comma separated, e.g. red,orange)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    severity = None
    if args.severity and args.severity.lower() != "all":
        severity = [token.strip().lower() for token in args.severity.split(",") if token.strip()]
    summary = run_scan(
        args.analysis,
        args.findings,
        args.rules,
        channel_ids=args.channel,
        since=args.since,
        until=args.until,
        severity_filter=severity,
    )
    print(summary.format_message())


if __name__ == "__main__":
    main()
