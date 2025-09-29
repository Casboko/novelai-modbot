from __future__ import annotations

import json
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import random

from .engine.types import DslPolicy
from .rule_engine import EvaluationResult, RuleEngine


class ReservoirSampler:
    def __init__(self, capacity: int = 4096) -> None:
        self.capacity = max(1, capacity)
        self._sample: list[float] = []
        self._count = 0

    def add(self, value: float) -> None:
        self._count += 1
        if len(self._sample) < self.capacity:
            self._sample.append(value)
            return
        index = random.randint(0, self._count - 1)
        if index < self.capacity:
            self._sample[index] = value

    def percentile(self, p: float) -> float:
        if not self._sample:
            return 0.0
        ordered = sorted(self._sample)
        idx = int(round((p / 100.0) * (len(ordered) - 1)))
        return float(ordered[idx])


@dataclass(slots=True)
class MetricsReport:
    total: int
    severity_counts: Counter
    winning_counts: Counter
    winning_ratio: dict[str, float]
    top_rules: list[tuple[str, int]]
    latency_ms_avg: float
    latency_ms_p95: float
    wall_time_s: float
    io_write_ms_total: float

    def to_json(self) -> dict:
        return {
            "total": self.total,
            "severity": dict(self.severity_counts),
            "winning": dict(self.winning_counts),
            "winning_ratio": self.winning_ratio,
            "top_rules": self.top_rules,
            "latency_ms_avg": round(self.latency_ms_avg, 3),
            "latency_ms_p95": round(self.latency_ms_p95, 3),
            "wall_time_s": round(self.wall_time_s, 3),
            "io_write_ms_total": round(self.io_write_ms_total, 3),
        }


class MetricsAggregator:
    def __init__(self, sample_capacity: int = 4096) -> None:
        self.total = 0
        self.severity = Counter({"red": 0, "orange": 0, "yellow": 0, "green": 0})
        self.winning = Counter()
        self.rule_hits = Counter()
        self.latency_sum = 0.0
        self.latency_reservoir = ReservoirSampler(sample_capacity)
        self.io_write_ms_total = 0.0

    def update(self, result: EvaluationResult, latency_ms: float, write_ms: float = 0.0) -> None:
        self.total += 1
        self.severity[result.severity] += 1
        winning_info = result.metrics.get("winning") if isinstance(result.metrics, dict) else None
        origin = "dsl"
        rule_id = result.rule_id
        if isinstance(winning_info, dict):
            origin = str(winning_info.get("origin", origin))
            rule_id = rule_id or winning_info.get("rule_id")
        self.winning[origin] += 1
        if rule_id:
            self.rule_hits[str(rule_id)] += 1
        self.latency_sum += latency_ms
        self.latency_reservoir.add(latency_ms)
        self.io_write_ms_total += write_ms

    def finalize(self, wall_time_s: float) -> MetricsReport:
        avg = (self.latency_sum / self.total) if self.total else 0.0
        p95 = self.latency_reservoir.percentile(95.0)
        total_wins = max(1, sum(self.winning.values()))
        winning_ratio = {
            origin: round(count / total_wins, 6)
            for origin, count in self.winning.items()
        }
        winning_ratio.setdefault("dsl", winning_ratio.get("dsl", 0.0))
        return MetricsReport(
            total=self.total,
            severity_counts=Counter(self.severity),
            winning_counts=Counter(self.winning),
            winning_ratio=winning_ratio,
            top_rules=self.rule_hits.most_common(20),
            latency_ms_avg=avg,
            latency_ms_p95=p95,
            wall_time_s=wall_time_s,
            io_write_ms_total=self.io_write_ms_total,
        )


class FindingsWriter:
    def __init__(self, path: Path) -> None:
        """Write findings JSONL while enforcing the p3 contract."""

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def write(self, record: dict, result: EvaluationResult) -> tuple[dict, float]:
        payload = _build_contract_payload(record, result)
        t0 = time.perf_counter()
        json.dump(payload, self._handle, ensure_ascii=False)
        self._handle.write("\n")
        self._handle.flush()
        write_ms = (time.perf_counter() - t0) * 1e3
        return payload, write_ms

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "FindingsWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()


RecordFilter = Callable[[dict], bool]
ResultFilter = Callable[[dict, EvaluationResult], bool]


def _build_contract_payload(record: dict, result: EvaluationResult) -> dict:
    payload = deepcopy(record)
    payload.setdefault("rule_id", None)
    payload.setdefault("rule_title", None)
    payload.setdefault("metrics", {})
    payload["severity"] = result.severity
    payload["rule_id"] = result.rule_id
    payload["rule_title"] = result.rule_title
    payload["reasons"] = list(result.reasons)
    metrics_data = result.metrics or {}
    if isinstance(metrics_data, dict):
        payload["metrics"] = dict(metrics_data)
    else:
        payload["metrics"] = dict(metrics_data)
    for message in payload.get("messages", []) or []:
        if isinstance(message, dict):
            message.setdefault("attachments", [])
    return payload


def evaluate_stream(
    engine: RuleEngine,
    records: Iterable[dict],
    *,
    writer: FindingsWriter | None = None,
    dry_run: bool = False,
    metrics_path: Path | None = None,
    record_filter: RecordFilter | None = None,
    result_filter: ResultFilter | None = None,
    collector: list[dict] | None = None,
    metrics_sample_capacity: int = 4096,
) -> MetricsReport:
    start = time.perf_counter()
    metrics = MetricsAggregator(sample_capacity=metrics_sample_capacity)
    collected = collector if collector is not None else None

    for record in records:
        if record_filter and not record_filter(record):
            continue
        t0 = time.perf_counter()
        result = engine.evaluate(record)
        latency_ms = (time.perf_counter() - t0) * 1e3
        if result_filter and not result_filter(record, result):
            continue
        write_ms = 0.0
        payload: Optional[dict] = None
        if not dry_run:
            if writer is None:
                payload = _build_contract_payload(record, result)
            else:
                payload, write_ms = writer.write(record, result)
        elif collected is not None:
            payload = _build_contract_payload(record, result)

        metrics.update(result, latency_ms=latency_ms, write_ms=write_ms)
        if payload is not None and collected is not None:
            collected.append(payload)

    report = metrics.finalize(time.perf_counter() - start)
    if metrics_path:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(report.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    return report
