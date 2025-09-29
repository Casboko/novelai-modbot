from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from app import mode_resolver
from app.mode_resolver import has_version_mismatch, resolve_policy


def _write_rules(tmp_path: Path, *, version: int = 2, dsl_mode: str | None = None) -> Path:
    lines = [f"version: {version}"]
    if dsl_mode:
        lines.append(f"dsl_mode: {dsl_mode}")
    lines.extend(
        [
            "rule_titles:",
            "  TEST: \"Test\"",
            "",
            "groups: {}",
            "features: {}",
            "",
            "rules:",
            "  - id: TEST",
            "    severity: red",
            "    when: \"rating.explicit >= 0.0\"",
            "    reasons: [\"ok\"]",
        ]
    )
    content = "\n".join(lines)
    path = tmp_path / "rules.yaml"
    path.write_text(content + "\n", encoding="utf-8")
    return path


def test_resolve_policy_uses_yaml_mode_when_no_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODBOT_DSL_MODE", raising=False)
    rules_path = _write_rules(tmp_path, dsl_mode="strict")

    policy, result, env_mode = resolve_policy(rules_path, None)

    assert policy.mode == "strict"
    assert result.mode == "strict"
    assert env_mode is None


def test_resolve_policy_env_overrides_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rules_path = _write_rules(tmp_path, dsl_mode="warn")
    monkeypatch.setenv("MODBOT_DSL_MODE", "strict")

    policy, result, env_mode = resolve_policy(rules_path, None)

    assert env_mode == "strict"
    assert policy.mode == "strict"
    assert result.mode == "strict"


def test_resolve_policy_cli_overrides_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rules_path = _write_rules(tmp_path, dsl_mode="warn")
    monkeypatch.setenv("MODBOT_DSL_MODE", "strict")

    policy, result, env_mode = resolve_policy(rules_path, "warn")

    assert env_mode == "strict"
    assert policy.mode == "warn"
    assert result.mode == "warn"


def test_invalid_env_value_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rules_path = _write_rules(tmp_path)
    monkeypatch.setenv("MODBOT_DSL_MODE", "invalid")
    mode_resolver._INVALID_ENV_LOGGED = False

    policy, result, env_mode = resolve_policy(rules_path, None)

    assert env_mode is None
    assert policy.mode == "warn"
    assert result.mode == "warn"


def test_has_version_mismatch_detects_v1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODBOT_DSL_MODE", raising=False)
    rules_path = _write_rules(tmp_path, version=1)

    policy, result, _ = resolve_policy(rules_path, None)

    assert policy.mode == "warn"
    assert has_version_mismatch(result)
