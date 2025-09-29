from __future__ import annotations

import pytest

from app.p3_stream import MetricsAggregator
from app.rule_engine import EvaluationResult


def _make_result(severity: str = "yellow") -> EvaluationResult:
    return EvaluationResult(
        severity=severity,
        rule_id="TEST",
        rule_title="Test",
        reasons=["ok"],
        metrics={"winning": {"origin": "dsl", "rule_id": "TEST", "severity": severity}},
    )


def test_metrics_report_throughput_rps() -> None:
    aggregator = MetricsAggregator()
    result = _make_result()
    for _ in range(5):
        aggregator.update(result, latency_ms=10.0)
    report = aggregator.finalize(wall_time_s=2.0)
    assert report.total == 5
    assert pytest.approx(report.throughput_rps, rel=1e-6) == 2.5


def test_metrics_report_zero_guard() -> None:
    aggregator = MetricsAggregator()
    aggregator.update(_make_result("green"), latency_ms=1.0)
    report = aggregator.finalize(wall_time_s=0.0)
    assert report.total == 1
    assert report.throughput_rps == pytest.approx(1000.0)
