from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from app import cli_rules


def _write_rules(path: Path, content: str) -> None:
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def test_cli_rules_validate_json_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    rules_path = tmp_path / "rules.yaml"
    _write_rules(
        rules_path,
        """
        version: 2
        rule_titles:
          TEST: "Test"
        groups: {}
        features: {}
        rules:
          - id: TEST
            severity: green
            when: "rating.general >= 0"
        """,
    )

    monkeypatch.setattr(sys, "argv", ["cli_rules", "validate", "--rules", str(rules_path), "--json"])
    with pytest.raises(SystemExit) as excinfo:
        cli_rules.main()
    assert excinfo.value.code == 0

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["status"] == "ok"
    assert data["counts"]["rules"] == 1


def test_cli_rules_validate_strict_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    rules_path = tmp_path / "rules.yaml"
    _write_rules(
        rules_path,
        """
        version: 2
        rule_titles:
          TEST: "Test"
        groups: {}
        features: {}
        thresholds: {}
        rules:
          - id: TEST
            severity: red
            when: "rating.explicit >= 0.5"
        """,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["cli_rules", "validate", "--rules", str(rules_path), "--mode", "strict"],
    )
    with pytest.raises(SystemExit) as excinfo:
        cli_rules.main()
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "R2-K001" in captured.out


def test_cli_rules_allow_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rules_path = tmp_path / "rules.yaml"
    _write_rules(
        rules_path,
        """
        version: 2
        rule_titles:
          TEST: "Test"
        groups: {}
        features:
          broken: "missing"
        rules:
          - id: TEST
            severity: red
            when: "broken > 0"
        """,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["cli_rules", "validate", "--rules", str(rules_path), "--allow-disabled"],
    )
    with pytest.raises(SystemExit) as excinfo:
        cli_rules.main()
    assert excinfo.value.code == 0


def test_cli_rules_treat_warnings_as_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rules_path = tmp_path / "rules.yaml"
    _write_rules(
        rules_path,
        """
        version: 2
        rule_titles:
          TEST: "Test"
        groups: {}
        features: {}
        thresholds: {}
        rules:
          - id: TEST
            severity: green
            when: "rating.general >= 0"
        """,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli_rules",
            "validate",
            "--rules",
            str(rules_path),
            "--treat-warnings-as-errors",
        ],
    )
    with pytest.raises(SystemExit) as excinfo:
        cli_rules.main()
    assert excinfo.value.code == 2
