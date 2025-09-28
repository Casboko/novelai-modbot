from __future__ import annotations

import json
from pathlib import Path

from app.triage import run_scan


def _write_analysis(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(rec) + "\n" for rec in records), encoding="utf-8")


def _write_rules(path: Path) -> None:
    path.write_text(
        """
models: {}
thresholds: {}
minor_tags: []
violence_tags: []
nsfw_general_tags: []
animal_abuse_tags: []
rule_titles: {}
        """.strip()
        + "\n",
        encoding="utf-8",
    )


def test_run_scan_dry_run(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.jsonl"
    findings = tmp_path / "findings.jsonl"
    metrics = tmp_path / "metrics.json"
    rules = tmp_path / "rules.yaml"
    _write_rules(rules)
    _write_analysis(
        analysis,
        [
            {
                "channel_id": "123",
                "created_at": "2024-01-01T00:00:00+00:00",
                "wd14": {"rating": {"explicit": 0.1, "questionable": 0.1, "general": 0.8, "sensitive": 0.0}},
                "xsignals": {},
                "nudity_detections": [],
                "is_nsfw_channel": False,
                "messages": [],
            }
        ],
    )

    summary = run_scan(
        analysis,
        findings,
        rules,
        dry_run=True,
        metrics_path=metrics,
        limit=1,
        since="2000-01-01",
        until="2100-01-01",
    )

    assert summary.total >= 1
    assert summary.severity_counts["green"] >= 1
    assert not findings.exists()
    assert metrics.exists()
