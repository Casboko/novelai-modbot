from __future__ import annotations

import textwrap
from pathlib import Path

from app.rule_engine import RuleEngine


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "rules.yaml"
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return path


def test_rule_engine_with_dsl_merges_result(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
        version: 2
        dsl_mode: warn
        models:
          wd14_repo: repo/example
        thresholds: {}
        minor_tags: []
        violence_tags: []
        nsfw_general_tags: []
        animal_abuse_tags: []
        rule_titles:
          DSL-RED: "DSL Red"
        groups: {}
        features: {}
        rules:
          - id: DSL-RED
            severity: red
            priority: 1
            when: "rating.explicit >= 0.5"
            reasons:
              - "exp={rating.explicit:.2f}"
        """,
    )
    engine = RuleEngine(config_path=str(config_path))
    assert "dsl=" in engine.describe_config()
    record = {
        "wd14": {"rating": {"explicit": 0.7, "questionable": 0.1, "general": 0.1, "sensitive": 0.1}},
        "xsignals": {},
        "nudity_detections": [],
        "is_nsfw_channel": False,
        "messages": [],
    }

    result = engine.evaluate(record)

    assert result.severity == "red"
    assert result.rule_id == "DSL-RED"
    assert result.metrics["winning"]["origin"] == "dsl"
    assert result.metrics["dsl"]["rule_id"] == "DSL-RED"
    assert any(reason.startswith("exp=") for reason in result.reasons)
