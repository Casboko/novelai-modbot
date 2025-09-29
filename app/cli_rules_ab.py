from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .io.stream import iter_jsonl
from .mode_resolver import has_version_mismatch, resolve_policy
from .rule_engine import RuleEngine

SEVERITY_ORDER = {"green": 0, "yellow": 1, "orange": 2, "red": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare rule configurations (A/B)")
    parser.add_argument("--analysis", type=Path, required=True, help="Analysis JSONL file or directory")
    parser.add_argument("--rulesA", type=Path, required=True, help="Rules configuration for variant A")
    parser.add_argument("--rulesB", type=Path, required=True, help="Rules configuration for variant B")
    parser.add_argument("--out-json", type=Path, required=True, help="Path to write summary JSON")
    parser.add_argument("--out-csv", type=Path, required=True, help="Path to write diff CSV")
    parser.add_argument("--sample-diff", type=int, default=0, help="Export top-N diffs as JSONL when >0")
    parser.add_argument("--export-dir", type=Path, default=Path("out/exports"), help="Directory for diff samples")
    parser.add_argument(
        "--dsl-mode",
        choices=["warn", "strict"],
        help="Overrides DSL policy mode (CLI > ENV > YAML > warn)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of records to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N records")
    parser.add_argument("--print-config", action="store_true", help="Print DSL configuration summaries")
    return parser.parse_args()


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

    def export(self, path: Path) -> None:
        if self.capacity <= 0 or not self._entries:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        records = sorted(self._entries, key=lambda item: (-item[0], item[1]))
        with path.open("w", encoding="utf-8") as handle:
            for score, _, payload in records:
                payload = dict(payload)
                payload["score"] = score
                handle.write(json.dumps(payload, ensure_ascii=False))
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

    policy_a, result_a, _ = resolve_policy(args.rulesA, args.dsl_mode)
    if has_version_mismatch(result_a):
        print("[error] rulesA is not version 2. Upgrade to DSL v2.", file=sys.stderr)
        sys.exit(2)
    if result_a.status == "error":
        for issue in result_a.issues:
            print(f"[{issue.code}] {issue.where}: {issue.msg}", file=sys.stderr)
        sys.exit(1)

    policy_b, result_b, _ = resolve_policy(args.rulesB, args.dsl_mode)
    if has_version_mismatch(result_b):
        print("[error] rulesB is not version 2. Upgrade to DSL v2.", file=sys.stderr)
        sys.exit(2)
    if result_b.status == "error":
        for issue in result_b.issues:
            print(f"[{issue.code}] {issue.where}: {issue.msg}", file=sys.stderr)
        sys.exit(1)

    engine_a = RuleEngine(str(args.rulesA), policy=policy_a)
    engine_b = RuleEngine(str(args.rulesB), policy=policy_b)
    if args.print_config:
        print("[A]", engine_a.describe_config())
        print("[B]", engine_b.describe_config())
    summary = AbSummary()
    sampler = TopNSampler(args.sample_diff)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    iter_records = iter_jsonl(args.analysis, limit=args.limit, offset=args.offset, policy=engine_a.policy)

    with args.out_csv.open("w", encoding="utf-8", newline="") as csvfile:
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

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary.as_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    if args.sample_diff:
        export_path = (args.export_dir or Path("out/exports")).joinpath("p3_ab_diff_samples.jsonl")
        sampler.export(export_path)


if __name__ == "__main__":
    main()
