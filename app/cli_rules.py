from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataclasses import asdict

from .engine.loader import load_rules_result
from .engine.types import DslMode

_DISABLED_ERROR_CODES = {"R2-E001", "R2-E002", "R2-G001", "R2-P001"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rules helper commands")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate a rules configuration file")
    validate.add_argument("--rules", type=Path, default=Path("configs/rules_v2.yaml"))
    validate.add_argument("--mode", choices=["warn", "strict"], help="Override DSL mode")
    validate.add_argument("--print-config", action="store_true", help="Print configuration summary")
    validate.add_argument("--json", action="store_true", help="Emit validation result as JSON")
    validate.add_argument(
        "--treat-warnings-as-errors",
        action="store_true",
        help="Return exit code 2 when warnings are present",
    )
    validate.add_argument(
        "--allow-disabled",
        action="store_true",
        help="Return exit code 0 when only disabled items caused invalid status",
    )
    return parser


def _errors_due_to_disabled(status: str, counts: dict[str, int], issues) -> bool:
    if status != "invalid":
        return False
    if not (counts.get("disabled_rules") or counts.get("disabled_features")):
        return False
    error_codes = {issue.code for issue in issues if issue.level == "error"}
    return bool(error_codes) and error_codes.issubset(_DISABLED_ERROR_CODES)


def _handle_validate(args: argparse.Namespace) -> int:
    mode: DslMode | None = args.mode  # type: ignore[assignment]
    result = load_rules_result(args.rules, override_mode=mode)

    if args.print_config and result.summary:
        counts = result.counts
        print(
            f"status={result.status} mode={result.mode} rules={counts['rules']} "
            f"features={counts['features']} groups={counts['groups']}"
        )
        print(
            f"errors={counts['errors']} warnings={counts['warnings']} "
            f"disabled_rules={counts['disabled_rules']} disabled_features={counts['disabled_features']}"
        )
        if result.summary.get("rules_ok"):
            print("rules_ok=" + ",".join(result.summary["rules_ok"]))
        if result.summary.get("rules_disabled"):
            print("rules_disabled=" + ",".join(result.summary["rules_disabled"]))
        if result.summary.get("features_disabled"):
            print("features_disabled=" + ",".join(result.summary["features_disabled"]))

    if args.json:
        payload = {
            "status": result.status,
            "mode": result.mode,
            "counts": result.counts,
            "issues": [asdict(issue) for issue in result.issues],
        }
        if result.summary:
            payload["summary"] = result.summary
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for issue in result.issues:
            line = f"{issue.level.upper():7} {issue.code} {issue.where}: {issue.msg}"
            print(line)
            if issue.hint:
                print(f"        hint: {issue.hint}")

    exit_code = 0
    if result.status == "error":
        exit_code = 2
    elif result.status == "invalid":
        disabled_only = args.allow_disabled and _errors_due_to_disabled(result.status, result.counts, result.issues)
        exit_code = 0 if disabled_only else 2

    if exit_code == 0 and args.treat_warnings_as_errors and result.counts.get("warnings", 0) > 0:
        exit_code = 2

    return exit_code


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "validate":
        exit_code = _handle_validate(args)
    else:  # pragma: no cover - defensive
        parser.error(f"Unknown command: {args.command}")
        return
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
