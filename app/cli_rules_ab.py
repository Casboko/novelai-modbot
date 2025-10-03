from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

from .io.stream import iter_jsonl
from .engine.loader import load_const_overrides_from_path
from .mode_resolver import has_version_mismatch, resolve_policy
from .rule_engine import RuleEngine

SEVERITY_ORDER = {"green": 0, "yellow": 1, "orange": 2, "red": 3}
URL_RE = re.compile(r"(?i)\bhttps?://[^\s<>\(\)\"']+")
MAX_REASON_ITEMS = 3
MAX_REASON_LENGTH = 80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare rule configurations (A/B)")
    parser.add_argument("--analysis", type=Path, required=True, help="Analysis JSONL file or directory")
    parser.add_argument("--rulesA", type=Path, required=True, help="Rules configuration for variant A")
    parser.add_argument("--rulesB", type=Path, required=True, help="Rules configuration for variant B")
    parser.add_argument("--out-json", type=Path, help="Path to write summary JSON")
    parser.add_argument("--out-csv", type=Path, help="Path to write diff CSV")
    parser.add_argument("--sample-diff", type=int, default=0, help="Export top-N diffs as JSONL when >0")
    parser.add_argument("--export-dir", type=Path, help="Directory for diff samples")
    parser.add_argument("--out-dir", type=Path, help="Directory to write compare/diff outputs")
    parser.add_argument(
        "--dsl-mode",
        choices=["warn", "strict"],
        help="Overrides DSL policy mode (CLI > ENV > YAML > warn)",
    )
    parser.add_argument(
        "--lock-mode",
        choices=["warn", "strict"],
        help="Lock both A/B evaluations to the specified mode (overrides CLI/ENV/YAML)",
    )
    parser.add_argument("--allow-legacy", action="store_true", help="Skip comparison when legacy rules are detected")
    parser.add_argument(
        "--samples-minimal",
        action="store_true",
        help="Write minimal JSONL samples instead of full analysis records",
    )
    parser.add_argument(
        "--samples-redact-urls",
        action="store_true",
        help="Redact URLs in exported samples (message_link/url -> null, inline URLs -> [URL])",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of records to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N records")
    parser.add_argument("--print-config", action="store_true", help="Print DSL configuration summaries")
    parser.add_argument("--constA", type=Path, help="Const override file applied to rulesA")
    parser.add_argument("--constB", type=Path, help="Const override file applied to rulesB")
    return parser.parse_args()


def _yaml_declared_mode(result) -> str | None:
    config = getattr(result, "config", None)
    raw = getattr(config, "raw", None)
    if isinstance(raw, dict):
        value = raw.get("dsl_mode")
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"warn", "strict"}:
                return normalized
    return None


def _mode_source(lock_mode: str | None, cli_mode: str | None, env_mode: str | None, result) -> str:
    if lock_mode:
        return "lock-mode"
    if cli_mode:
        return "CLI"
    if env_mode:
        return "ENV"
    if _yaml_declared_mode(result):
        return "YAML"
    return "default"


def _redact_urls_in_text(value: str) -> str:
    if "[URL]" in value:
        return value
    return URL_RE.sub("[URL]", value)


def _trim_reason(value: str) -> str:
    text = value.strip()
    if len(text) <= MAX_REASON_LENGTH:
        return text
    trimmed = text[: MAX_REASON_LENGTH - 3].rstrip()
    return f"{trimmed}..."


def _prepare_reasons(candidate, *, redact_urls: bool) -> list[str]:
    if not isinstance(candidate, (list, tuple)):
        return []
    results: list[str] = []
    for item in candidate:
        if not isinstance(item, str):
            continue
        text = item
        if redact_urls:
            text = _redact_urls_in_text(text)
        text = _trim_reason(text)
        if text:
            results.append(text)
        if len(results) >= MAX_REASON_ITEMS:
            break
    return results


def _coerce_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _build_minimal_sample(payload: dict, *, redact_urls: bool) -> dict:
    _ = redact_urls  # minimal view is always redacted; argument retained for compatibility
    record = payload.get("record") if isinstance(payload, dict) else None
    record_map = record if isinstance(record, dict) else {}
    phash = record_map.get("phash") or record_map.get("phash_hex")

    wd14 = record_map.get("wd14") if isinstance(record_map.get("wd14"), dict) else {}
    rating = wd14.get("rating", {}) if isinstance(wd14, dict) else {}
    xsignals = record_map.get("xsignals") if isinstance(record_map.get("xsignals"), dict) else {}

    rating_explicit = _coerce_float(rating.get("explicit", 0.0))
    rating_questionable = _coerce_float(rating.get("questionable", 0.0))

    metrics_base = {
        "rating_explicit": rating_explicit,
        "rating_questionable": rating_questionable,
    }
    metrics_optional: dict[str, float] = {}
    for key in ("exposure_score", "nsfw_general_sum"):
        if isinstance(xsignals, dict) and key in xsignals:
            metrics_optional[key] = _coerce_float(xsignals.get(key))

    metrics_a = dict(metrics_base)
    metrics_a.update(metrics_optional)
    metrics_b = dict(metrics_base)
    metrics_b.update(metrics_optional)

    severity = payload.get("severity") if isinstance(payload, dict) else {}
    rules = payload.get("rule") if isinstance(payload, dict) else {}
    reasons = payload.get("reasons") if isinstance(payload, dict) else {}

    minimal = {
        "message_link": None,
        "phash": phash if phash else None,
        "severityA": severity.get("A"),
        "severityB": severity.get("B"),
        "ruleA": rules.get("A"),
        "ruleB": rules.get("B"),
        "reasonsA": _prepare_reasons(reasons.get("A"), redact_urls=True),
        "reasonsB": _prepare_reasons(reasons.get("B"), redact_urls=True),
        "metricsA": metrics_a,
        "metricsB": metrics_b,
    }
    return minimal


def _redact_payload(payload: dict) -> dict:
    def _walk(value):
        if isinstance(value, dict):
            result: dict = {}
            for key, item in value.items():
                if key == "message_link" or key.lower() == "url" or key.lower().endswith("_url"):
                    result[key] = None
                else:
                    result[key] = _walk(item)
            return result
        if isinstance(value, list):
            return [_walk(item) for item in value]
        if isinstance(value, str):
            return _redact_urls_in_text(value)
        return value

    return _walk(deepcopy(payload))


class AbSummary:
    def __init__(self) -> None:
        self.counts_a = {"red": 0, "orange": 0, "yellow": 0, "green": 0}
        self.counts_b = {"red": 0, "orange": 0, "yellow": 0, "green": 0}
        self.confusion: dict[tuple[str, str], int] = {}
        self.rule_a: dict[str, int] = {}
        self.rule_b: dict[str, int] = {}
        self.total = 0

    def update(self, res_a, res_b) -> None:
        self.total += 1
        self.counts_a[res_a.severity] = self.counts_a.get(res_a.severity, 0) + 1
        self.counts_b[res_b.severity] = self.counts_b.get(res_b.severity, 0) + 1
        key = (res_a.severity, res_b.severity)
        self.confusion[key] = self.confusion.get(key, 0) + 1
        rule_a = res_a.rule_id or res_a.metrics.get("winning", {}).get("rule_id")
        rule_b = res_b.rule_id or res_b.metrics.get("winning", {}).get("rule_id")
        if rule_a:
            self.rule_a[rule_a] = self.rule_a.get(rule_a, 0) + 1
        if rule_b:
            self.rule_b[rule_b] = self.rule_b.get(rule_b, 0) + 1

    def as_json(self) -> dict:
        delta = {sev: self.counts_b.get(sev, 0) - self.counts_a.get(sev, 0) for sev in SEVERITY_ORDER}
        confusion = {f"{a}->{b}": count for (a, b), count in sorted(self.confusion.items())}
        rule_delta: list[tuple[str, int, int, int]] = []
        all_rules = set(self.rule_a) | set(self.rule_b)
        for rule in all_rules:
            a = self.rule_a.get(rule, 0)
            b = self.rule_b.get(rule, 0)
            rule_delta.append((rule, a, b, b - a))
        rule_delta.sort(key=lambda item: abs(item[3]), reverse=True)
        return {
            "total": self.total,
            "countsA": self.counts_a,
            "countsB": self.counts_b,
            "delta": delta,
            "confusion": confusion,
            "top_rule_delta": [
                {"id": rid, "A": a, "B": b, "delta": diff}
                for rid, a, b, diff in rule_delta[:10]
            ],
        }


class TopNSampler:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, capacity)
        self._entries: list[tuple[float, int, dict]] = []
        self._counter = 0

    def consider(self, score: float, payload: dict) -> None:
        if self.capacity <= 0:
            return
        entry = (score, self._counter, payload)
        self._counter += 1
        if len(self._entries) < self.capacity:
            import heapq

            heapq.heappush(self._entries, entry)
            return
        import heapq

        if score <= self._entries[0][0]:
            return
        heapq.heapreplace(self._entries, entry)

    def export(self, path: Path, *, minimal: bool = False, redact_urls: bool = False) -> None:
        if self.capacity <= 0 or not self._entries:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        records = sorted(self._entries, key=lambda item: (-item[0], item[1]))
        with path.open("w", encoding="utf-8") as handle:
            for score, _, payload in records:
                if minimal:
                    entry = _build_minimal_sample(payload, redact_urls=redact_urls)
                elif redact_urls:
                    entry = _redact_payload(payload)
                else:
                    entry = dict(payload)
                output = dict(entry)
                output["score"] = score
                handle.write(json.dumps(output, ensure_ascii=False))
                handle.write("\n")


def severity_score(sev_a: str, sev_b: str) -> int:
    return abs(SEVERITY_ORDER.get(sev_a, 0) - SEVERITY_ORDER.get(sev_b, 0))


def build_diff_payload(record: dict, res_a, res_b) -> dict:
    rating = record.get("wd14", {}).get("rating", {}) if isinstance(record.get("wd14"), dict) else {}
    xsignals = record.get("xsignals", {}) if isinstance(record.get("xsignals"), dict) else {}
    rule_a = res_a.rule_id or res_a.metrics.get("winning", {}).get("rule_id")
    rule_b = res_b.rule_id or res_b.metrics.get("winning", {}).get("rule_id")
    return {
        "message_link": record.get("message_link"),
        "severity": {"A": res_a.severity, "B": res_b.severity},
        "rule": {"A": rule_a, "B": rule_b},
        "reasons": {"A": res_a.reasons, "B": res_b.reasons},
        "metrics": {"A": res_a.metrics, "B": res_b.metrics},
        "rating": {"questionable": rating.get("questionable", 0.0), "explicit": rating.get("explicit", 0.0)},
        "xsignals": {
            "exposure_score": xsignals.get("exposure_score", 0.0),
            "nsfw_general_sum": xsignals.get("nsfw_general_sum", 0.0),
        },
        "record": record,
    }


def build_csv_row(record: dict, res_a, res_b) -> list:
    rating = record.get("wd14", {}).get("rating", {}) if isinstance(record.get("wd14"), dict) else {}
    xsignals = record.get("xsignals", {}) if isinstance(record.get("xsignals"), dict) else {}
    metrics_a = res_a.metrics if isinstance(res_a.metrics, dict) else {}
    metrics_b = res_b.metrics if isinstance(res_b.metrics, dict) else {}
    return [
        record.get("message_link", ""),
        res_a.severity,
        res_b.severity,
        res_a.rule_id or metrics_a.get("winning", {}).get("rule_id", ""),
        res_b.rule_id or metrics_b.get("winning", {}).get("rule_id", ""),
        rating.get("questionable", 0.0),
        rating.get("explicit", 0.0),
        xsignals.get("exposure_score", 0.0),
        xsignals.get("nsfw_general_sum", 0.0),
        metrics_a.get("minors_peak", 0.0),
        metrics_a.get("gore_peak", 0.0),
        metrics_b.get("minors_peak", 0.0),
        metrics_b.get("gore_peak", 0.0),
    ]


def main() -> None:
    args = parse_args()

    out_json_path = args.out_json
    out_csv_path = args.out_csv
    export_dir = args.export_dir
    if args.out_dir:
        base = args.out_dir
        if out_json_path is None:
            out_json_path = base / "p3_ab_compare.json"
        if out_csv_path is None:
            out_csv_path = base / "p3_ab_diff.csv"
        if export_dir is None:
            export_dir = base

    if out_json_path is None or out_csv_path is None:
        print("[error] --out-json/--out-csv を指定するか --out-dir で出力先をまとめて指定してください。", file=sys.stderr)
        sys.exit(1)

    if export_dir is None:
        export_dir = Path("out/exports")

    override_mode = args.lock_mode or args.dsl_mode

    policy_a, result_a, env_a = resolve_policy(args.rulesA, override_mode)
    if has_version_mismatch(result_a):
        if not args.allow_legacy:
            print("[error] rulesA is not version 2. Upgrade to DSL v2.", file=sys.stderr)
            print("Use --allow-legacy to skip comparison safely.", file=sys.stderr)
            sys.exit(2)
    if result_a.status == "error":
        for issue in result_a.issues:
            print(f"[{issue.code}] {issue.where}: {issue.msg}", file=sys.stderr)
        sys.exit(1)

    policy_b, result_b, env_b = resolve_policy(args.rulesB, override_mode)
    if has_version_mismatch(result_b):
        if not args.allow_legacy:
            print("[error] rulesB is not version 2. Upgrade to DSL v2.", file=sys.stderr)
            print("Use --allow-legacy to skip comparison safely.", file=sys.stderr)
            sys.exit(2)
    if result_b.status == "error":
        for issue in result_b.issues:
            print(f"[{issue.code}] {issue.where}: {issue.msg}", file=sys.stderr)
        sys.exit(1)

    legacy_detected = has_version_mismatch(result_a) or has_version_mismatch(result_b)
    if args.lock_mode:
        if env_a or env_b or args.dsl_mode:
            print(
                "[info] --lock-mode overrides --dsl-mode / MODBOT_DSL_MODE / YAML dsl_mode for both variants",
                file=sys.stderr,
            )
        policy_a = policy_b = type(policy_a).from_mode(args.lock_mode)

    source_a = _mode_source(args.lock_mode, args.dsl_mode, env_a, result_a)
    source_b = _mode_source(args.lock_mode, args.dsl_mode, env_b, result_b)

    if legacy_detected and args.allow_legacy:
        banner = (
            f"policyA={policy_a.mode}(source={source_a}), "
            f"policyB={policy_b.mode}(source={source_b})"
        )
        print(banner)
        summary = AbSummary()
        payload = summary.as_json()
        payload["note"] = "skipped due to legacy ruleset"
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[info] legacy rules detected. Comparison skipped (see note field).", file=sys.stderr)
        return

    overrides_a = load_const_overrides_from_path(args.constA) if args.constA else None
    overrides_b = load_const_overrides_from_path(args.constB) if args.constB else None
    overrides_a = dict(overrides_a) if overrides_a else None
    overrides_b = dict(overrides_b) if overrides_b else None

    engine_a = RuleEngine(str(args.rulesA), policy=policy_a, const_overrides=overrides_a)
    engine_b = RuleEngine(str(args.rulesB), policy=policy_b, const_overrides=overrides_b)
    banner = (
        f"policyA={engine_a.policy.mode}(source={source_a}), "
        f"policyB={engine_b.policy.mode}(source={source_b})"
    )
    if args.lock_mode:
        banner += f", lock-mode={args.lock_mode}"
    print(banner)
    if args.print_config:
        print("[A]", engine_a.describe_config())
        print("[B]", engine_b.describe_config())
    summary = AbSummary()
    sampler = TopNSampler(args.sample_diff)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    iter_records = iter_jsonl(args.analysis, limit=args.limit, offset=args.offset, policy=engine_a.policy)

    with out_csv_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "message_link",
                "severityA",
                "severityB",
                "ruleA",
                "ruleB",
                "rating_questionable",
                "rating_explicit",
                "exposure_score",
                "nsfw_general_sum",
                "minors_peak_A",
                "gore_peak_A",
                "minors_peak_B",
                "gore_peak_B",
            ]
        )
        for record in iter_records:
            res_a = engine_a.evaluate(record)
            res_b = engine_b.evaluate(record)
            summary.update(res_a, res_b)
            rule_a = res_a.rule_id or res_a.metrics.get("winning", {}).get("rule_id")
            rule_b = res_b.rule_id or res_b.metrics.get("winning", {}).get("rule_id")
            if res_a.severity == res_b.severity and rule_a == rule_b:
                continue
            row = build_csv_row(record, res_a, res_b)
            writer.writerow(row)
            score = severity_score(res_a.severity, res_b.severity)
            if score == 0 and rule_a != rule_b:
                score = 0.5
            payload = build_diff_payload(record, res_a, res_b)
            sampler.consider(score, payload)

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(summary.as_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    if args.sample_diff:
        export_dir.mkdir(parents=True, exist_ok=True)
        export_path = export_dir.joinpath("p3_ab_diff_samples.jsonl")
        sampler.export(
            export_path,
            minimal=args.samples_minimal,
            redact_urls=args.samples_redact_urls,
        )


if __name__ == "__main__":
    main()
