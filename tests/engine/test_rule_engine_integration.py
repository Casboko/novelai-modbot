from __future__ import annotations

import textwrap
from pathlib import Path

from app.rule_engine import RuleEngine


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return path


BASE_CONFIG = """
version: 2
rule_titles:
  DSL-RED: "DSL Red"
  DSL-GREEN: "DSL Green"
groups:
  nsfw_general: ["bikini"]
  violence: ["blood", "gore"]
features:
  nsfw_margin: "max(rating.explicit, rating.questionable) - max(rating.general, rating.sensitive)"
  nsfw_ratio: "(rating.explicit + rating.questionable) / clamp(rating.explicit + rating.questionable + rating.general + rating.sensitive, 1e-6, 1e6)"
  nsfw_general_sum: "sum('nsfw_general')"
  violence_sum: "sum('violence')"
  violence_max: "max('violence')"
  drug_score: "sum('violence')"
rules:
  - id: DSL-RED
    severity: red
    priority: 10
    when: "rating.explicit >= 0.5"
    reasons:
      - "exp={rating.explicit:.2f}"
      - "violence={violence_max:.2f}"
"""


def test_rule_engine_returns_dsl_result(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, BASE_CONFIG)
    engine = RuleEngine(config_path=str(config_path))

    record = {
        "wd14": {
            "rating": {"explicit": 0.7, "questionable": 0.1, "general": 0.1, "sensitive": 0.1},
            "general": [("blood", 0.4)],
        },
        "xsignals": {"placement_risk_pre": 0.3},
        "nudity_detections": [],
        "is_nsfw_channel": False,
        "messages": [],
    }

    result = engine.evaluate(record)

    assert result.severity == "red"
    assert result.rule_id == "DSL-RED"
    assert result.rule_title == "DSL Red"
    assert result.metrics["winning"]["origin"] == "dsl"
    assert result.metrics["violence_max"] == 0.4
    assert result.metrics["placement_risk"] == 0.3
    assert any(reason.startswith("exp=") for reason in result.reasons)


def test_rule_engine_green_when_no_rule_matches(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, BASE_CONFIG)
    engine = RuleEngine(config_path=str(config_path))

    record = {
        "wd14": {
            "rating": {"explicit": 0.2, "questionable": 0.2, "general": 0.3, "sensitive": 0.3},
            "general": [("gore", 0.6)],
        },
        "xsignals": {"placement_risk_pre": 0.1},
        "nudity_detections": [],
        "is_nsfw_channel": True,
        "messages": [],
    }

    result = engine.evaluate(record)

    assert result.severity == "green"
    assert result.rule_id is None
    assert result.rule_title is None
    assert result.metrics["winning"]["origin"] == "dsl"
    # metrics are still exported even when no rule matches
    assert result.metrics["violence_max"] == 0.6
    assert result.metrics["nsfw_general_sum"] == 0.0


def test_rule_engine_exposes_group_patterns(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, BASE_CONFIG)
    engine = RuleEngine(config_path=str(config_path))

    assert engine.groups["violence"] == ("blood", "gore")
    assert engine.groups["nsfw_general"] == ("bikini",)
