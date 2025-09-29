from __future__ import annotations

import json
import textwrap
from pathlib import Path

from app.cli_contract import _cmd_check_findings, _cmd_check_report
from app.triage import generate_report, run_scan


def _write_analysis(path: Path) -> None:
    record = {
        "phash": "cafebabecafebabe",
        "channel_id": "999",
        "created_at": "2024-02-01T00:00:00+00:00",
        "is_nsfw_channel": False,
        "wd14": {
            "rating": {
                "general": 0.8,
                "sensitive": 0.1,
                "questionable": 0.05,
                "explicit": 0.05,
            },
            "general": [["tag_a", 0.6]],
        },
        "xsignals": {"placement_risk_pre": 0.0},
        "nudity_detections": [],
        "messages": [
            {
                "author_id": "42",
                "is_nsfw_channel": False,
                "attachments": [],
                "content": "テストメッセージ",
            }
        ],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


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
            when: "sample_score >= 0.5"
            reasons:
              - "score={sample_score:.2f}"
        """
    ).strip()
    path.write_text(content + "\n", encoding="utf-8")


def test_cli_contract_pipeline(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis.jsonl"
    findings = tmp_path / "findings.jsonl"
    report = tmp_path / "report.csv"
    rules = tmp_path / "rules.yaml"
    _write_analysis(analysis)
    _write_rules(rules)

    metrics = tmp_path / "metrics.json"
    trace = tmp_path / "trace.jsonl"

    run_scan(
        analysis,
        findings,
        rules,
        since="2000-01-01",
        until="2100-01-01",
        metrics_path=metrics,
        trace_jsonl=trace,
    )
    generate_report(findings, report, rules_path=rules)

    schema_path = Path(__file__).resolve().parents[2] / "docs/contracts/p3_findings.schema.json"
    assert (
        _cmd_check_findings(findings, schema_path, limit=0, json_output=False) == 0
    )
    assert _cmd_check_report(report, json_output=False) == 0

    findings_raw = findings.read_bytes()
    report_raw = report.read_bytes()
    assert findings_raw.endswith(b"\n")
    assert report_raw.endswith(b"\n")
    assert b"\r" not in findings_raw
    assert b"\r" not in report_raw
    assert "テストメッセージ" in findings.read_text(encoding="utf-8")

    record = json.loads(findings.read_text(encoding="utf-8").splitlines()[0])
    eval_ms = record.get("metrics", {}).get("eval_ms")
    assert isinstance(eval_ms, (int, float))
    assert eval_ms >= 0

    metrics_data = json.loads(metrics.read_text(encoding="utf-8"))
    assert "throughput_rps" in metrics_data
    assert metrics_data["throughput_rps"] >= 0

    trace_lines = trace.read_text(encoding="utf-8").splitlines()
    finding_lines = findings.read_text(encoding="utf-8").splitlines()
    assert len(trace_lines) == len(finding_lines)
    trace_record = json.loads(trace_lines[0])
    assert trace_record.get("eval_ms") == eval_ms
    assert trace_record.get("winning") in {"dsl", "legacy"}
