from __future__ import annotations

import json
from pathlib import Path
import textwrap

from jsonschema import Draft7Validator

from app.triage import run_scan


def _write_analysis(path: Path) -> None:
    record = {
        "phash": "deadbeefdeadbeef",
        "channel_id": "123",
        "created_at": "2024-01-01T00:00:00+00:00",
        "is_nsfw_channel": False,
        "wd14": {
            "rating": {
                "general": 0.7,
                "sensitive": 0.1,
                "questionable": 0.1,
                "explicit": 0.1,
            },
            "general": [["tag_a", 0.5]],
        },
        "xsignals": {"placement_risk_pre": 0.0},
        "nudity_detections": [],
        "messages": [
            {
                "author_id": "555",
                "is_nsfw_channel": False,
                "attachments": [],
                "content": "テスト",
            }
        ],
    }
    payload = json.dumps(record) + "\n"
    path.write_text(payload, encoding="utf-8")


def _write_rules(path: Path) -> None:
    content = textwrap.dedent(
        """
        models: {}
        thresholds: {}
        minor_tags: []
        violence_tags: []
        nsfw_general_tags: []
        animal_abuse_tags: []
        rule_titles: {}
        """
    ).strip()
    path.write_text(content + "\n", encoding="utf-8")


def test_findings_record_matches_schema(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.jsonl"
    findings = tmp_path / "findings.jsonl"
    rules = tmp_path / "rules.yaml"
    _write_analysis(analysis)
    _write_rules(rules)

    run_scan(analysis, findings, rules, since="2000-01-01", until="2100-01-01")

    data = findings.read_text(encoding="utf-8").strip().splitlines()
    assert data
    record = json.loads(data[0])

    schema_path = Path(__file__).resolve().parents[2] / "docs/contracts/p3_findings.schema.json"
    validator = Draft7Validator(json.loads(schema_path.read_text(encoding="utf-8")))

    errors = list(validator.iter_errors(record))
    assert not errors
    assert set(record.keys()).issuperset({"severity", "rule_id", "rule_title", "reasons", "metrics"})
    assert isinstance(record["reasons"], list)
    assert isinstance(record["metrics"], dict)
