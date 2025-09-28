from __future__ import annotations

import pytest

from app.engine.dsl import (
    DslEvaluationInput,
    DslProgram,
    DslRuntimeError,
    DslValidationError,
    compile_expression,
)


def _build_program(strict: bool = False) -> DslProgram:
    groups = {"nsfw_general": ["bikini", "see_through"]}
    features = {"combo_score": "rating.explicit * exposure_peak"}
    rules = [
        {
            "id": "TEST-RED",
            "severity": "red",
            "priority": 5,
            "when": "(rating.explicit >= 0.4) || sum('nsfw_general') >= 0.30",
            "reasons": ["exp={rating.explicit:.2f}", "sum={sum('nsfw_general'):.2f}"],
        },
        {
            "id": "TEST-ORANGE",
            "severity": "orange",
            "priority": 10,
            "when": "combo_score >= 0.10",
            "reasons": ["combo={combo_score:.3f}"],
        },
    ]
    return DslProgram.from_config(groups=groups, features=features, rules=rules, strict=strict)


def _default_input(program: DslProgram) -> DslEvaluationInput:
    return DslEvaluationInput(
        rating={"explicit": 0.45, "questionable": 0.2, "general": 0.3},
        metrics={"exposure_score": 0.22, "minors_peak": 0.05},
        tag_scores={"bikini": 0.32, "see_through": 0.05},
        group_patterns=program.group_patterns,
        nude_flags=("FEMALE_BREAST_EXPOSED",),
        is_nsfw_channel=False,
        is_spoiler=False,
        attachment_count=1,
    )


def test_dsl_program_matches_high_severity_rule() -> None:
    program = _build_program()
    outcome = program.evaluate(_default_input(program))
    assert outcome is not None
    assert outcome.severity == "red"
    assert outcome.rule_id == "TEST-RED"
    # reasons are formatted with two decimals
    assert any(reason.startswith("exp=0.45") for reason in outcome.reasons)
    assert "sum=0.32" in " ".join(outcome.reasons)
    assert outcome.diagnostics["features"]["combo_score"] == pytest.approx(0.45 * 0.22)


def test_dsl_program_returns_lower_severity_when_high_rule_not_hit() -> None:
    program = _build_program()
    evaluation_input = _default_input(program)
    evaluation_input = evaluation_input.__class__(
        rating={"explicit": 0.10, "questionable": 0.05, "general": 0.85},
        metrics={"exposure_score": 0.45},
        tag_scores={"bikini": 0.05},
        group_patterns=program.group_patterns,
        nude_flags=(),
        is_nsfw_channel=True,
        is_spoiler=False,
        attachment_count=0,
    )
    outcome = program.evaluate(evaluation_input)
    assert outcome is not None
    assert outcome.severity == "orange"
    assert outcome.rule_id == "TEST-ORANGE"
    assert any(reason.startswith("combo=") for reason in outcome.reasons)


def test_dsl_program_returns_none_when_no_rule_matches() -> None:
    program = _build_program()
    evaluation_input = DslEvaluationInput(
        rating={"explicit": 0.05},
        metrics={"exposure_score": 0.05},
        tag_scores={"bikini": 0.01},
        group_patterns=program.group_patterns,
        nude_flags=(),
        is_nsfw_channel=True,
        is_spoiler=False,
        attachment_count=0,
    )
    assert program.evaluate(evaluation_input) is None


@pytest.mark.parametrize(
    "expression",
    ["[1, 2, 3]", "__import__('os')", "(lambda x: x)(1)"],
)
def test_compile_expression_rejects_unsupported_nodes(expression: str) -> None:
    with pytest.raises(DslValidationError):
        compile_expression(expression)


def test_reason_template_respects_strict_mode() -> None:
    program = DslProgram.from_config(
        groups={},
        features={},
        rules=[
            {
                "id": "STRICT-FAIL",
                "severity": "red",
                "when": "missing_value > 0",
                "reasons": ["missing={missing_value}"],
            }
        ],
        strict=False,
    )
    outcome = program.evaluate(
        DslEvaluationInput(
            rating={},
            metrics={},
            tag_scores={},
            group_patterns=program.group_patterns,
            nude_flags=(),
            is_nsfw_channel=False,
            is_spoiler=False,
            attachment_count=0,
        )
    )
    assert outcome is None

    strict_program = DslProgram.from_config(
        groups={},
        features={},
        rules=[
            {
                "id": "STRICT-FAIL",
                "severity": "red",
                "when": "missing_value > 0",
                "reasons": ["missing={missing_value}"],
            }
        ],
        strict=True,
    )
    with pytest.raises(DslRuntimeError):
        strict_program.evaluate(
            DslEvaluationInput(
                rating={},
                metrics={},
                tag_scores={},
                group_patterns=strict_program.group_patterns,
                nude_flags=(),
                is_nsfw_channel=False,
                is_spoiler=False,
                attachment_count=0,
            )
        )
