from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import date

import pytest

from app import analysis_merge, cli_scan
from zoneinfo import ZoneInfo

from app.profiles import ProfileContext, set_profiles_root_override, clear_context_cache, PartitionPaths

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PROFILE = "demo"
PARTITION_DATE = date(2025, 10, 1)


class DummyAnalyzer:
    version = "dummy"

    def detect_batch(self, images):  # noqa: D401, ANN001
        return [[] for _ in images]


class DummyCache:
    def __init__(self, path):  # noqa: D401, ANN001
        self._store: dict[str, dict] = {}

    def get(self, key):  # noqa: D401, ANN001
        return self._store.get(getattr(key, "phash", ""))

    def set(self, key, payload):  # noqa: D401, ANN001
        self._store[getattr(key, "phash", "")] = payload


def _make_context() -> ProfileContext:
    return ProfileContext(profile=PROFILE, date=PARTITION_DATE, tzinfo=ZoneInfo("UTC"))


def _run_analysis(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    context = _make_context()
    partitions = PartitionPaths(context)
    out_path = partitions.stage_file("p2", ensure_parent=True)
    metrics_path = partitions.metrics_file("p2", ensure_parent=True)
    cache_root = tmp_path / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / "nudenet_cache.sqlite"
    argv = [
        "analysis_merge",
        "--scan",
        str(FIXTURES / "p0_scan_minimal.csv"),
        "--wd14",
        str(FIXTURES / "p1_wd14_minimal_with_raw.jsonl"),
        "--out",
        str(out_path),
        "--metrics",
        str(metrics_path),
        "--nudenet-cache",
        str(cache_path),
        "--nudenet-config",
        str(FIXTURES / "nudenet_minimal.yaml"),
        "--xsignals-config",
        str(FIXTURES / "xsignals_minimal.yaml"),
        "--rules-config",
        str(FIXTURES / "rules_v2_minimal.yaml"),
        "--nudenet-mode",
        "never",
        "--seed",
        "42",
    ]
    monkeypatch.setattr(analysis_merge, "NudeNetAnalyzer", lambda: DummyAnalyzer())
    monkeypatch.setattr(analysis_merge, "NudeNetCache", DummyCache)
    monkeypatch.setattr(sys, "argv", argv)
    analysis_merge.main()
    return out_path


def _run_scan(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, analysis_path: Path) -> Path:
    context = _make_context()
    partitions = PartitionPaths(context)
    findings_path = partitions.stage_file("p3", ensure_parent=True)
    metrics_path = partitions.metrics_file("p3", ensure_parent=True)
    argv = [
        "cli_scan",
        "--analysis",
        str(analysis_path),
        "--findings",
        str(findings_path),
        "--rules",
        str(FIXTURES / "rules_v2_minimal.yaml"),
        "--metrics",
        str(metrics_path),
        "--dsl-mode",
        "warn",
        "--print-config",
        "--limit",
        "1",
        "--offset",
        "0",
        "--since",
        "2000-01-01",
        "--until",
        "2100-01-01",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    cli_scan.main()
    return findings_path


def test_minimal_pipeline_produces_dsl_findings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    set_profiles_root_override(tmp_path / "profiles")
    clear_context_cache()
    context = _make_context()
    try:
        analysis_path = _run_analysis(monkeypatch, tmp_path)

        analysis_record = json.loads(analysis_path.read_text(encoding="utf-8").strip())
        assert "xsignals" in analysis_record
        assert "nsfw_general_sum" in analysis_record["xsignals"]

        findings_path = _run_scan(monkeypatch, tmp_path, analysis_path)
        findings_text = findings_path.read_text(encoding="utf-8").strip().splitlines()
        assert findings_text
        finding = json.loads(findings_text[0])
        assert finding["severity"] == "orange"
        assert finding["rule_id"] == "TEST-ORANGE"
        assert finding["reasons"]
        assert finding["metrics"]["winning"]["origin"] == "dsl"

        partitions = PartitionPaths(context)
        assert analysis_path == partitions.stage_file("p2")
        assert findings_path == partitions.stage_file("p3")
        metrics_p2 = partitions.metrics_file("p2")
        metrics_p3 = partitions.metrics_file("p3")
        assert metrics_p2.exists()
        assert metrics_p3.exists()
    finally:
        set_profiles_root_override(None)
        clear_context_cache()
    analysis_path = _run_analysis(monkeypatch, tmp_path)

    analysis_record = json.loads(analysis_path.read_text(encoding="utf-8").strip())
    assert "xsignals" in analysis_record
    assert "nsfw_general_sum" in analysis_record["xsignals"]

    findings_path = _run_scan(monkeypatch, tmp_path, analysis_path)
    findings_text = findings_path.read_text(encoding="utf-8").strip().splitlines()
    assert findings_text
    finding = json.loads(findings_text[0])
    assert finding["severity"] == "orange"
    assert finding["rule_id"] == "TEST-ORANGE"
    assert finding["reasons"]
    assert finding["metrics"]["winning"]["origin"] == "dsl"
