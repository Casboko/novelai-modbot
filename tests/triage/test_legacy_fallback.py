from __future__ import annotations

import json
from pathlib import Path

from app.engine.types import DslPolicy
from app.triage import run_scan


def _write_analysis(path: Path) -> None:
    record = {
        "phash": "cafebabe",
        "channel_id": "123",
        "created_at": "2024-01-01T00:00:00+00:00",
        "is_nsfw_channel": False,
        "wd14": {"rating": {"general": 0.9, "explicit": 0.0}},
        "messages": [],
        "xsignals": {},
        "nudity_detections": [],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_run_scan_legacy_fallback_green(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.jsonl"
    findings = tmp_path / "findings.jsonl"
    rules = tmp_path / "rules_v1.yaml"
    _write_analysis(analysis)
    rules.write_text("version: 1\n", encoding="utf-8")

    summary = run_scan(
        analysis,
        findings,
        rules,
        since="2000-01-01",
        until="2100-01-01",
        policy=DslPolicy(),
        fallback="green",
    )

    assert summary.total == 1
    assert summary.severity_counts["green"] == 1
    payload = json.loads(findings.read_text(encoding="utf-8"))
    assert payload["severity"] == "green"
    assert payload["reasons"] == ["legacy_ruleset_unsupported"]


def test_run_scan_legacy_fallback_skip(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.jsonl"
    findings = tmp_path / "findings.jsonl"
    rules = tmp_path / "rules_v1.yaml"
    _write_analysis(analysis)
    rules.write_text("version: 1\n", encoding="utf-8")

    summary = run_scan(
        analysis,
        findings,
        rules,
        since="2000-01-01",
        until="2100-01-01",
        policy=DslPolicy(),
        fallback="skip",
    )

    assert summary.total == 0
    assert findings.exists()
    assert findings.read_text(encoding="utf-8") == ""
