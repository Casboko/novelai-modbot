from __future__ import annotations

import textwrap
from pathlib import Path

from app.engine.loader import load_rules_result


def _write_yaml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return path


def _base_rules_yaml(extra: str = "") -> str:
    return textwrap.dedent(
        f"""
        version: 2
        rule_titles:
          TEST-RED: "Test rule"

        groups:
          sample: ["See Through", "see_through"]

        features:
          boost: "max('sample')"

        rules:
          - id: TEST-RED
            severity: red
            when: "boost >= 0.5"
            reasons:
              - "boost={{boost:.2f}}"
        {extra}
        """
    ).strip()


def test_version_mismatch_warn_invalid(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        _base_rules_yaml().replace("version: 2", "version: 1"),
    )
    result = load_rules_result(path)
    assert result.status == "invalid"
    assert result.counts["errors"] == 1


def test_version_mismatch_strict_error(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        _base_rules_yaml().replace("version: 2", "version: 1"),
    )
    result = load_rules_result(path, override_mode="strict")
    assert result.status == "error"


def test_cli_override_mode_priority(tmp_path: Path) -> None:
    yaml_text = _base_rules_yaml("dsl_mode: strict")
    path = _write_yaml(tmp_path, yaml_text)
    result = load_rules_result(path, override_mode="warn")
    assert result.mode == "warn"


def test_unknown_keys_warn(tmp_path: Path) -> None:
    yaml_text = _base_rules_yaml(
        """
        thresholds:
          sample: 0.5
        """
    )
    path = _write_yaml(tmp_path, yaml_text)
    result = load_rules_result(path)
    assert result.status == "ok"
    assert result.counts["warnings"] >= 1
    assert result.counts["unknown_keys"] == 1


def test_unknown_keys_strict_error(tmp_path: Path) -> None:
    yaml_text = _base_rules_yaml(
        """
        thresholds:
          sample: 0.5
        """
    )
    path = _write_yaml(tmp_path, yaml_text)
    result = load_rules_result(path, override_mode="strict")
    assert result.status == "error"


def test_rule_titles_must_cover_rules(tmp_path: Path) -> None:
    yaml_text = textwrap.dedent(
        """
        version: 2
        rule_titles: {}
        groups:
          sample: ["foo"]
        features: {}
        rules:
          - id: TEST-RED
            severity: red
            when: "rating.explicit >= 0.5"
        """
    )
    path = _write_yaml(tmp_path, yaml_text)
    result = load_rules_result(path)
    assert result.status == "invalid"
    assert any(issue.code == "R2-T001" for issue in result.issues)


def test_duplicate_rule_ids_raise_invalid(tmp_path: Path) -> None:
    yaml_text = textwrap.dedent(
        """
        version: 2
        rule_titles:
          DUP: "Duplicate"
        groups: {}
        features: {}
        rules:
          - id: DUP
            severity: red
            when: "rating.explicit >= 0.5"
          - id: DUP
            severity: orange
            when: "rating.explicit >= 0.2"
        """
    )
    path = _write_yaml(tmp_path, yaml_text)
    result = load_rules_result(path)
    assert result.status == "invalid"
    assert result.counts["disabled_rules"] == 1


def test_group_patterns_are_normalized(tmp_path: Path) -> None:
    yaml_text = textwrap.dedent(
        """
        version: 2
        rule_titles:
          TEST: "Test"
        groups:
          sample: ["See Through", "see_through"]
        features: {}
        rules:
          - id: TEST
            severity: red
            when: "rating.explicit >= 0.5"
        """
    )
    path = _write_yaml(tmp_path, yaml_text)
    result = load_rules_result(path)
    assert result.config is not None
    assert result.config.config.groups["sample"] == ["see_through"]
    assert result.counts["collisions"] == 1


def test_unknown_identifier_disables_feature(tmp_path: Path) -> None:
    yaml_text = textwrap.dedent(
        """
        version: 2
        rule_titles:
          TEST: "Test"
        groups: {}
        features:
          broken: "missing_value + 1"
        rules:
          - id: TEST
            severity: red
            when: "rating.explicit >= 0.5"
        """
    )
    path = _write_yaml(tmp_path, yaml_text)
    result = load_rules_result(path)
    assert result.status == "invalid"
    assert result.counts["disabled_features"] == 1
    assert any(issue.code == "R2-E002" for issue in result.issues)


def test_reason_placeholder_is_validated(tmp_path: Path) -> None:
    yaml_text = textwrap.dedent(
        """
        version: 2
        rule_titles:
          TEST: "Test"
        groups: {}
        features: {}
        rules:
          - id: TEST
            severity: red
            when: "rating.explicit >= 0.5"
            reasons:
              - "missing={missing_value:.2f}"
        """
    )
    path = _write_yaml(tmp_path, yaml_text)
    result = load_rules_result(path)
    assert result.status == "invalid"
    assert result.counts["placeholder_fixes"] == 1
    assert any(issue.code == "R2-P001" for issue in result.issues)


def test_rule_defaults_priority_and_stop(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, _base_rules_yaml())
    result = load_rules_result(path)
    assert result.status == "ok"
    assert result.config is not None
    rule = result.config.config.rules[0]
    assert rule.priority == 0
    assert rule.stop is False
