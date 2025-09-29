from __future__ import annotations

import csv
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


def _write_p0_scan(path: Path, *, phash: str, message_id: str) -> None:
    header = (
        "guild_id,channel_id,message_id,message_link,created_at,author_id,is_nsfw_channel,"
        "source,attachment_id,filename,content_type,file_size,url,phash_hex,phash_distance_ref,note\n"
    )
    row = \
        f"1,2,{message_id},https://discord.com/channels/1/2/{message_id},2024-01-01T00:00:00+00:00,42,false,attachment," \
        f"att-1,test.png,image/png,123,https://cdn.example.com/test.png,{phash},,\n"
    path.write_text(header + row, encoding="utf-8")


def _write_rules(path: Path) -> None:
    content = textwrap.dedent(
        """
        version: 2
        rule_titles:
          TEST: "Test"

        groups:
          sample: ["tag_a"]

        features:
          sample_score: "score('tag_a')"

        rules:
          - id: TEST
            severity: yellow
            when: "sample_score >= 0.4"
            reasons:
              - "score={sample_score:.2f}"
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


def test_run_scan_enriches_attachments(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.jsonl"
    findings = tmp_path / "findings.jsonl"
    rules = tmp_path / "rules.yaml"
    p0 = tmp_path / "p0_scan.csv"
    report_ext = tmp_path / "p3_report_ext.csv"

    phash = "feedfacefeedface"
    message_id = "999"

    record = {
        "phash": phash,
        "channel_id": "123",
        "created_at": "2024-01-01T00:00:00+00:00",
        "is_nsfw_channel": False,
        "wd14": {"rating": {"general": 0.7, "sensitive": 0.1, "questionable": 0.1, "explicit": 0.1}},
        "xsignals": {"placement_risk_pre": 0.0},
        "nudity_detections": [],
        "messages": [
            {
                "message_id": message_id,
                "author_id": "555",
                "attachments": [],
            }
        ],
    }
    analysis.write_text(json.dumps(record) + "\n", encoding="utf-8")
    _write_rules(rules)
    _write_p0_scan(p0, phash=phash, message_id=message_id)

    run_scan(
        analysis,
        findings,
        rules,
        since="2000-01-01",
        until="2100-01-01",
        p0_path=p0,
        attachments_report_path=report_ext,
    )

    data = json.loads(findings.read_text(encoding="utf-8").strip())
    message = data["messages"][0]
    attachments = message.get("attachments")
    assert attachments and attachments[0]["id"] == "att-1"
    assert attachments[0]["filename"] == "test.png"
    assert attachments[0]["content_type"] == "image/png"
    assert attachments[0]["file_size"] == 123
    assert attachments[0]["source"] == "p0"

    with report_ext.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["attachments_count"] == "1"
    assert rows[0]["first_attachment_id"] == "att-1"
