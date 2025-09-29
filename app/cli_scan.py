from __future__ import annotations

import argparse
from pathlib import Path

from .engine.types import DslPolicy
from .rule_engine import RuleEngine
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
    parser.add_argument("--dsl-mode", choices=["warn", "strict"], help="Overrides DSL policy mode")
    parser.add_argument("--metrics", type=Path, help="Path to write metrics JSON summary")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate without writing findings")
    parser.add_argument("--print-config", action="store_true", help="Print DSL configuration summary")
    parser.add_argument("--limit", type=int, default=0, help="Maximum records to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N records before evaluation")
    parser.add_argument("--p0", type=Path, help="Path to p0 scan CSV for attachment metadata join")
    parser.add_argument("--no-attachments", action="store_true", help="Disable attachment enrichment")
    parser.add_argument(
        "--csv-attachments",
        action="store_true",
        help="Write extended CSV with attachment summary (p3_report_ext.csv)",
    )
    parser.add_argument(
        "--drop-attachment-urls",
        action="store_true",
        help="Strip attachment URLs from findings output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    severity = None
    if args.severity and args.severity.lower() != "all":
        severity = [token.strip().lower() for token in args.severity.split(",") if token.strip()]
    policy = DslPolicy.from_mode(args.dsl_mode) if args.dsl_mode else None
    engine = RuleEngine(str(args.rules), policy=policy)
    if args.print_config:
        print(engine.describe_config())
    summary = run_scan(
        args.analysis,
        args.findings,
        args.rules,
        channel_ids=args.channel,
        since=args.since,
        until=args.until,
        severity_filter=severity,
        dry_run=args.dry_run,
        metrics_path=args.metrics,
        limit=args.limit,
        offset=args.offset,
        engine=engine,
        p0_path=args.p0,
        include_attachments=not args.no_attachments,
        drop_attachment_urls=args.drop_attachment_urls,
        attachments_report_path=(args.findings.with_name("p3_report_ext.csv") if args.csv_attachments else None),
    )
    if args.dry_run:
        print("[dry-run] findings not written")
    print(summary.format_message())


if __name__ == "__main__":
    main()
