from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine.loader import load_const_overrides_from_path
from .output_paths import default_analysis_path, default_findings_path
from .mode_resolver import has_version_mismatch, resolve_policy
from .rule_engine import RuleEngine
from .triage import run_scan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rule evaluation on analysis output")
    parser.add_argument("--analysis", type=Path, default=default_analysis_path())
    parser.add_argument("--findings", type=Path, default=default_findings_path())
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
    parser.add_argument(
        "--severity",
        type=str,
        default="all",
        help="Filter severities (comma separated, e.g. red,orange)",
    )
    parser.add_argument(
        "--dsl-mode",
        choices=["warn", "strict"],
        help="Overrides DSL policy mode (CLI > ENV > YAML > warn)",
    )
    parser.add_argument(
        "--lock-mode",
        choices=["warn", "strict"],
        help="Force both loader and evaluator to use the specified DSL mode",
    )
    parser.add_argument("--metrics", type=Path, help="Path to write metrics JSON summary")
    parser.add_argument("--trace-jsonl", type=Path, help="Write per-record trace JSONL (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate without writing findings")
    parser.add_argument("--print-config", action="store_true", help="Print DSL configuration summary")
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="Allow version 1 rules by enabling fallback handling",
    )
    parser.add_argument(
        "--fallback",
        choices=["green", "skip"],
        default="green",
        help="Fallback behaviour when legacy rules are allowed",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress startup banner output")
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
    parser.add_argument(
        "--const",
        type=Path,
        help="Path to const override file (YAML/JSON) applied after rules thresholds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    severity = None
    if args.severity and args.severity.lower() != "all":
        severity = [token.strip().lower() for token in args.severity.split(",") if token.strip()]
    override_mode = args.lock_mode or args.dsl_mode
    policy, load_result, _ = resolve_policy(args.rules, override_mode)
    if load_result.status == "error" and not has_version_mismatch(load_result):
        for issue in load_result.issues:
            print(f"[{issue.code}] {issue.where}: {issue.msg}", file=sys.stderr)
        sys.exit(1)

    is_legacy = has_version_mismatch(load_result)
    if is_legacy and not args.allow_legacy:
        for issue in load_result.issues:
            if issue.code == "R2-V001" and issue.where == "version":
                print(f"[error] {issue.msg}", file=sys.stderr)
                break
        print("Use --allow-legacy to continue with fallback handling.", file=sys.stderr)
        sys.exit(2)

    engine = None
    const_overrides = None
    if args.const:
        loaded = load_const_overrides_from_path(args.const)
        const_overrides = loaded or None
    if not is_legacy:
        engine = RuleEngine(str(args.rules), policy=policy, const_overrides=const_overrides)
        if args.print_config:
            print(engine.describe_config())
        if not args.quiet:
            banner = engine.describe_config(as_one_line=True)
            print(f"{banner}, source=CLI")
    else:
        if args.print_config:
            print("policy.mode={mode}\nlegacy rules detected (version v1)".format(mode=policy.mode))
        if not args.quiet:
            print("policy={mode}, dsl=disabled (rules=v1), source=CLI".format(mode=policy.mode))

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
        trace_jsonl=args.trace_jsonl,
        limit=args.limit,
        offset=args.offset,
        engine=engine,
        policy=policy,
        fallback=args.fallback if is_legacy else None,
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
