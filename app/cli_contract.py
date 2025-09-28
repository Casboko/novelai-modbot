from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft7Validator

from .triage import P3_CSV_HEADER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate p3 output contracts")
    sub = parser.add_subparsers(dest="command", required=True)

    findings = sub.add_parser("check-findings", help="Validate findings JSONL against schema")
    findings.add_argument("--path", type=Path, default=Path("out/p3_findings.jsonl"))
    findings.add_argument(
        "--schema",
        type=Path,
        default=Path("docs/contracts/p3_findings.schema.json"),
        help="JSON Schema path (Draft-07)",
    )
    findings.add_argument("--limit", type=int, default=0, help="Maximum number of reported errors")
    findings.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")

    report = sub.add_parser("check-report", help="Validate report CSV header ordering")
    report.add_argument("--path", type=Path, default=Path("out/p3_report.csv"))
    report.add_argument("--json", action="store_true", help="Emit machine-readable JSON output")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "check-findings":
        code = _cmd_check_findings(args.path, args.schema, args.limit, args.json)
    elif args.command == "check-report":
        code = _cmd_check_report(args.path, args.json)
    else:  # pragma: no cover - argparse enforces choices
        code = 1
    raise SystemExit(code)


def _cmd_check_findings(path: Path, schema_path: Path, limit: int, json_output: bool) -> int:
    try:
        schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _print_error(f"schema not found: {schema_path}")
        return 1
    except OSError as exc:
        _print_error(f"failed to read schema: {schema_path}: {exc}")
        return 1
    except json.JSONDecodeError as exc:
        _print_error(f"invalid schema JSON: {schema_path}: {exc}")
        return 1

    validator = Draft7Validator(schema_data)
    reported = []
    checked = 0

    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    reported.append(
                        {
                            "line": line_no,
                            "path": "$",
                            "type": "parse",
                            "error": f"JSON decode error: {exc.msg}",
                        }
                    )
                    if _reached_limit(reported, limit):
                        break
                    continue
                checked += 1
                errors = list(validator.iter_errors(payload))
                if errors:
                    for err in errors:
                        reported.append(
                            {
                                "line": line_no,
                                "path": _format_json_pointer(err.path),
                                "type": "schema",
                                "error": err.message,
                            }
                        )
                        if _reached_limit(reported, limit):
                            break
                if _reached_limit(reported, limit):
                    break
    except FileNotFoundError:
        _print_error(f"findings not found: {path}")
        return 1
    except OSError as exc:
        _print_error(f"failed to read findings: {path}: {exc}")
        return 1

    if json_output:
        output = {
            "ok": not reported,
            "checked": checked,
            "errors": reported,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"checked={checked}")
        if reported:
            for item in reported:
                print(
                    f"[line {item['line']}] {item['error']} (path={item['path']}, type={item['type']})"
                )
        else:
            print("findings schema: ok")

    return 0 if not reported else 2


def _cmd_check_report(path: Path, json_output: bool) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                header = []
    except FileNotFoundError:
        _print_error(f"report not found: {path}")
        return 1
    except OSError as exc:
        _print_error(f"failed to read report: {path}: {exc}")
        return 1

    expected = list(P3_CSV_HEADER)
    ok = header == expected

    if json_output:
        output = {
            "ok": ok,
            "expected": expected,
            "actual": header,
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        if ok:
            print("report header: ok")
        else:
            print("report header mismatch")
            print(f" expected: {', '.join(expected)}")
            print(f" actual:   {', '.join(header)}")

    return 0 if ok else 2


def _format_json_pointer(parts: Iterable[Any]) -> str:
    tokens = [str(part) for part in parts]
    return "/".join(tokens) if tokens else "$"


def _reached_limit(items: list[Any], limit: int) -> bool:
    return limit > 0 and len(items) >= limit


def _print_error(message: str) -> None:
    print(message, file=sys.stderr)


if __name__ == "__main__":
    main()
