from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from app import cli_rules_ab


def _write_rules_v2(
    path: Path,
    *,
    threshold: float,
    reason: str = "exp={rating.explicit:.2f}",
    dsl_mode: str | None = None,
) -> None:
    lines: list[str] = ["version: 2"]
    if dsl_mode:
        lines.append(f"dsl_mode: {dsl_mode}")
    lines.extend(
        [
            "rule_titles:",
            '  RULE-RED: "Red rule"',
            "",
            "groups: {}",
            "features: {}",
            "",
            "rules:",
            "  - id: RULE-RED",
            "    severity: red",
            "    priority: 10",
            f"    when: \"rating.explicit >= {threshold}\"",
            f"    reasons: [\"{reason}\"]",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_rules_v1(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            version: 1
            rule_titles: {}
            rules: []
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _analysis_record() -> dict:
    return {
        "message_link": "https://discord.com/channels/1/2/3",
        "phash": "feedfacefeedface",
        "wd14": {
            "rating": {
                "explicit": 0.6,
                "questionable": 0.2,
                "general": 0.1,
            }
        },
        "xsignals": {
            "exposure_score": 0.75,
            "nsfw_general_sum": 1.25,
        },
        "urls": ["https://cdn.example.com/image.png"],
        "messages": [
            {
                "attachments": [
                    {
                        "url": "https://cdn.example.com/attachment.png",
                        "is_spoiler": False,
                    }
                ],
                "is_spoiler": False,
            }
        ],
        "nudity_detections": [
            {
                "class": "EXPOSED_BREAST",
                "score": 0.5,
            }
        ],
    }


def _write_analysis(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def test_lock_mode_overrides_cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("MODBOT_DSL_MODE", raising=False)
    analysis_path = tmp_path / "analysis.jsonl"
    _write_analysis(analysis_path, [_analysis_record()])

    rules_path = tmp_path / "rules.yaml"
    _write_rules_v2(rules_path, threshold=0.1, dsl_mode="strict")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    monkeypatch.setenv("MODBOT_DSL_MODE", "strict")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli_rules_ab",
            "--analysis",
            str(analysis_path),
            "--rulesA",
            str(rules_path),
            "--rulesB",
            str(rules_path),
            "--out-dir",
            str(out_dir),
            "--lock-mode",
            "warn",
            "--dsl-mode",
            "strict",
        ],
    )

    cli_rules_ab.main()
    captured = capsys.readouterr()
    banner = captured.out.strip().splitlines()[0]
    assert "policyA=warn(source=lock-mode)" in banner
    assert "policyB=warn(source=lock-mode)" in banner
    assert "lock-mode=warn" in banner

    compare_path = out_dir / "p3_ab_compare.json"
    diff_csv = out_dir / "p3_ab_diff.csv"
    assert compare_path.exists()
    assert diff_csv.exists()


def test_allow_legacy_skips_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("MODBOT_DSL_MODE", raising=False)
    analysis_path = tmp_path / "analysis.jsonl"
    _write_analysis(analysis_path, [_analysis_record()])

    rules_v2 = tmp_path / "rules_v2.yaml"
    _write_rules_v2(rules_v2, threshold=0.1)
    rules_v1 = tmp_path / "rules_v1.yaml"
    _write_rules_v1(rules_v1)

    out_dir = tmp_path / "outputs"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli_rules_ab",
            "--analysis",
            str(analysis_path),
            "--rulesA",
            str(rules_v2),
            "--rulesB",
            str(rules_v1),
            "--out-dir",
            str(out_dir),
            "--allow-legacy",
        ],
    )

    cli_rules_ab.main()
    captured = capsys.readouterr()
    banner = captured.out.strip().splitlines()[0]
    assert "policyB=warn(source=default)" in banner

    compare_path = out_dir / "p3_ab_compare.json"
    diff_csv = out_dir / "p3_ab_diff.csv"
    samples_path = out_dir / "p3_ab_diff_samples.jsonl"

    assert compare_path.exists()
    assert not diff_csv.exists()
    assert not samples_path.exists()

    payload = json.loads(compare_path.read_text(encoding="utf-8"))
    assert payload["note"] == "skipped due to legacy ruleset"
    assert payload["total"] == 0


def test_out_dir_autofills_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    analysis_path = tmp_path / "analysis.jsonl"
    _write_analysis(analysis_path, [_analysis_record()])

    rules_a = tmp_path / "rules_a.yaml"
    rules_b = tmp_path / "rules_b.yaml"
    _write_rules_v2(rules_a, threshold=0.4, reason="why https://example.com/diff")
    _write_rules_v2(rules_b, threshold=0.9, reason="why https://example.com/diff")

    out_dir = tmp_path / "autodir"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli_rules_ab",
            "--analysis",
            str(analysis_path),
            "--rulesA",
            str(rules_a),
            "--rulesB",
            str(rules_b),
            "--out-dir",
            str(out_dir),
            "--sample-diff",
            "1",
        ],
    )

    cli_rules_ab.main()

    compare_path = out_dir / "p3_ab_compare.json"
    diff_csv = out_dir / "p3_ab_diff.csv"
    samples_path = out_dir / "p3_ab_diff_samples.jsonl"

    assert compare_path.exists()
    assert diff_csv.exists()
    assert samples_path.exists()

    summary = json.loads(compare_path.read_text(encoding="utf-8"))
    assert summary["total"] == 1
    with diff_csv.open("r", encoding="utf-8") as fp:
        rows = fp.read().strip().splitlines()
    assert len(rows) == 2  # header + one diff row


def test_samples_minimal_with_redaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    analysis_path = tmp_path / "analysis.jsonl"
    _write_analysis(analysis_path, [_analysis_record()])

    rules_a = tmp_path / "rules_a.yaml"
    rules_b = tmp_path / "rules_b.yaml"
    reason = "see https://review.example.com/case"
    _write_rules_v2(rules_a, threshold=0.3, reason=reason)
    _write_rules_v2(rules_b, threshold=0.95, reason=reason)

    out_dir = tmp_path / "minimal"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli_rules_ab",
            "--analysis",
            str(analysis_path),
            "--rulesA",
            str(rules_a),
            "--rulesB",
            str(rules_b),
            "--out-dir",
            str(out_dir),
            "--sample-diff",
            "1",
            "--samples-minimal",
            "--samples-redact-urls",
        ],
    )

    cli_rules_ab.main()

    sample_path = out_dir / "p3_ab_diff_samples.jsonl"
    data = sample_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(data) == 1
    entry = json.loads(data[0])
    assert entry["message_link"] is None
    assert entry["phash"] == "feedfacefeedface"
    assert entry["ruleA"] == "RULE-RED"
    assert entry["ruleB"] is None
    assert entry["reasonsA"], "reasonsA should not be empty"
    assert "[URL]" in entry["reasonsA"][0]
    assert "rating_explicit" in entry["metricsA"]
    assert "exposure_score" in entry["metricsA"]
    assert entry["metricsA"]["rating_explicit"] == pytest.approx(0.6)
    assert entry["metricsB"]["rating_questionable"] == pytest.approx(0.2)


def test_samples_redact_full_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    analysis_path = tmp_path / "analysis.jsonl"
    _write_analysis(analysis_path, [_analysis_record()])

    rules_a = tmp_path / "rules_a.yaml"
    rules_b = tmp_path / "rules_b.yaml"
    reason = "reason https://review.example.com/item"
    _write_rules_v2(rules_a, threshold=0.3, reason=reason)
    _write_rules_v2(rules_b, threshold=0.95, reason=reason)

    out_dir = tmp_path / "redact"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli_rules_ab",
            "--analysis",
            str(analysis_path),
            "--rulesA",
            str(rules_a),
            "--rulesB",
            str(rules_b),
            "--out-dir",
            str(out_dir),
            "--sample-diff",
            "1",
            "--samples-redact-urls",
        ],
    )

    cli_rules_ab.main()

    sample_path = out_dir / "p3_ab_diff_samples.jsonl"
    data = sample_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(data) == 1
    entry = json.loads(data[0])
    assert entry["message_link"] is None
    assert "record" in entry
    record = entry["record"]
    assert record["urls"][0] == "[URL]"
    message = record["messages"][0]
    assert message["attachments"][0]["url"] is None
    assert "[URL]" in entry["reasons"]["A"][0]
