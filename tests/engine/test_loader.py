from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.engine import DslValidationError
from app.engine.loader import load_rule_config
from app.engine.types import RuleConfigV2


def _write_yaml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return path


def test_load_rule_config_v2_success(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
        version: 2
        dsl_mode: warn
        models:
          wd14_repo: repo/example
        thresholds:
          wd14_questionable: 0.3
        groups:
          nsfw_general: ["Bikini", "see through"]
        features:
          combo: "rating.explicit * exposure_peak"
        rules:
          - id: RED-1
            severity: red
            priority: 5
            when: "rating.explicit >= 0.5"
            reasons:
              - "exp={rating.explicit:.2f}"
        """,
    )
    dsl_config, raw, policy = load_rule_config(path)
    assert isinstance(dsl_config, RuleConfigV2)
    assert policy.mode == "warn"
    assert dsl_config.groups["nsfw_general"] == ["bikini", "see_through"]
    assert "combo" in dsl_config.features
    assert len(dsl_config.rules) == 1
    assert raw["version"] == 2


def test_load_rule_config_v2_invalid_rule_warn(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
        version: 2
        rules:
          - id: INVALID
            severity: purple
            when: "rating.explicit > 0.2"
        """,
    )
    with pytest.raises(ValueError):
        load_rule_config(path)


def test_load_rule_config_v2_strict_raises(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
        version: 2
        dsl_mode: strict
        rules:
          - id: STRICT
            severity: red
            when: "rating.explicit >> 0.5"
        """,
    )
    with pytest.raises(DslValidationError):
        load_rule_config(path)


def test_load_rule_config_v1_passthrough(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
        models:
          wd14_repo: repo
        thresholds: {}
        minor_tags: []
        rules:
          - sample: value
        """,
    )
    dsl_config, raw, policy = load_rule_config(path)
    assert dsl_config is None
    assert raw["models"]["wd14_repo"] == "repo"
    assert policy.mode == "warn"
