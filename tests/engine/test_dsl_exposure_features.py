from __future__ import annotations

from app.engine import DslProgram
from app.engine.dsl import DslEvaluationInput
from app.engine.evaluator import DslEvaluator
from app.engine.types import DslPolicy, DslRule, RuleConfigV2


def _build_program() -> DslProgram:
    config = RuleConfigV2(
        rules=[
            DslRule(
                id="EXPOSURE-AREA",
                severity="orange",
                priority=10,
                when="exposure_area >= 0.2",
            ),
            DslRule(
                id="EXPOSURE-COUNT",
                severity="yellow",
                priority=5,
                when="exposure_count >= 2",
            ),
        ]
    )
    return DslProgram.from_config(config, DslPolicy.from_mode("warn"))


def _analysis_record(area: float, count: float) -> dict:
    return {
        "wd14": {
            "rating": {
                "explicit": 0.1,
                "questionable": 0.2,
                "general": 0.7,
            }
        },
        "xsignals": {
            "nudity_area_ratio": area,
            "nudity_box_count": count,
        },
        "messages": [],
    }


def test_dsl_evaluator_uses_exposure_aliases_from_xsignals() -> None:
    evaluator = DslEvaluator(_build_program())
    result = evaluator.evaluate(_analysis_record(0.25, 3))
    assert result.severity == "orange"
    assert result.rule_id == "EXPOSURE-AREA"


def test_dsl_evaluator_falls_back_when_metrics_missing() -> None:
    evaluator = DslEvaluator(_build_program())
    result = evaluator.evaluate(_analysis_record(0.0, 0.0))
    # area rule not hit -> count rule also 0
    assert result.severity == "green"
    assert result.rule_id is None
