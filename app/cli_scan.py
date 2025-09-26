from __future__ import annotations

import argparse
from pathlib import Path

from .triage import run_scan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rule evaluation on analysis output")
    parser.add_argument("--analysis", type=Path, default=Path("out/p2_analysis.jsonl"))
    parser.add_argument("--findings", type=Path, default=Path("out/p3_findings.jsonl"))
    parser.add_argument("--rules", type=Path, default=Path("configs/rules.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_scan(args.analysis, args.findings, args.rules)
    print(summary.format_message())


if __name__ == "__main__":
    main()
