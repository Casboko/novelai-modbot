from __future__ import annotations

import argparse
from pathlib import Path

from .triage import generate_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate moderation report CSV")
    parser.add_argument("--findings", type=Path, default=Path("out/p3_findings.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("out/p3_report.csv"))
    parser.add_argument("--severity", type=str, default="all", help="Severity filter: all/red/orange/yellow/green")
    parser.add_argument("--rules", type=Path, default=Path("configs/rules.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_report(args.findings, args.out, args.severity, args.rules)
    print(f"rows={summary.rows} path={summary.path}")


if __name__ == "__main__":
    main()
