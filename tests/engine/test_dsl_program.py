from __future__ import annotations

import pytest

from app.engine import DslProgram, DslRuntimeError, DslValidationError, DslPolicy
from app.engine.loader import build_const_map
from app.engine.dsl import DslEvaluationInput
from app.engine.dsl_utils import validate_expr
from app.engine.types import DslRule, RuleConfigV2


def _build_program(strict: bool = False) -> DslProgram:
    config = RuleConfigV2(
        groups={"nsfw_general": ["bikini", "see_through"]},
        features={"combo_score": "rating.explicit * exposure_peak"},
        rules=[
            DslRule(
                id="TEST-RED",
                severity="red",
                priority=5,
                when="(rating.explicit >= 0.4) || sum('nsfw_general') >= 0.30",
                reasons=["exp={rating.explicit:.2f}", "sum={sum('nsfw_general'):.2f}"],
            ),
            DslRule(
                id="TEST-ORANGE",
                severity="orange",
                priority=10,
                when="combo_score >= 0.10",
                reasons=["combo={combo_score:.3f}"],
            ),
        ],
    )
    policy = DslPolicy.from_mode("strict" if strict else "warn")
    return DslProgram.from_config(config, policy)


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
    # グループには正規化された全タグが集計されるため、sum は bikini と see_through の合計になる
    assert "sum=0.37" in " ".join(outcome.reasons)
    assert outcome.diagnostics["features"]["combo_score"] == pytest.approx(0.45 * 0.22)


def test_dsl_program_returns_lower_severity_when_high_rule_not_hit() -> None:
    program = _build_program()
    evaluation_input = _default_input(program)
    evaluation_input = evaluation_input.__class__(
        rating={"explicit": 0.30, "questionable": 0.05, "general": 0.60},
        metrics={"exposure_score": 0.60},
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
        validate_expr(expression)


def test_reason_template_respects_strict_mode() -> None:
    warn_config = RuleConfigV2(
        rules=[
            DslRule(
                id="STRICT-FAIL",
                severity="red",
                when="missing_value > 0",
                reasons=["missing={missing_value}"],
            )
        ]
    )
    program = DslProgram.from_config(warn_config, DslPolicy.from_mode("warn"))
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

    strict_program = DslProgram.from_config(warn_config, DslPolicy.from_mode("strict"))
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


def test_const_identifiers_available_in_strict_mode() -> None:
    const_map = build_const_map()
    config = RuleConfigV2(
        rules=[
            DslRule(
                id="CONST-TEST",
                severity="orange",
                when="rating.explicit >= const_WD14_EXPL",
                reasons=["explicit={rating.explicit:.2f}"]
            )
        ]
    )
    program = DslProgram.from_config(config, DslPolicy.from_mode("strict"), const_map=const_map)
    outcome = program.evaluate(
        DslEvaluationInput(
            rating={"explicit": const_map["const_WD14_EXPL"] + 0.01},
            metrics={},
            tag_scores={},
            group_patterns=program.group_patterns,
            nude_flags=(),
            is_nsfw_channel=False,
            is_spoiler=False,
            attachment_count=0,
        )
    )
    assert outcome is not None
    assert outcome.rule_id == "CONST-TEST"


def test_qe_margin_boundary_behaviour() -> None:
    const_map = build_const_map()
    features = {
        "qe_margin": "max(rating.questionable - const_WD14_QUES, rating.explicit - const_WD14_EXPL)",
        "is_low_margin": "qe_margin < const_LOW_MARGIN",
    }
    rules = [
        DslRule(id="LOW", severity="yellow", when="is_low_margin", reasons=["margin={qe_margin:.3f}"])
    ]
    config = RuleConfigV2(features=features, rules=rules)
    program = DslProgram.from_config(config, DslPolicy.from_mode("warn"), const_map=const_map)

    low_outcome = program.evaluate(
        DslEvaluationInput(
            rating={"explicit": const_map["const_WD14_EXPL"] + 0.02, "questionable": 0.0},
            metrics={},
            tag_scores={},
            group_patterns=program.group_patterns,
            nude_flags=(),
            is_nsfw_channel=False,
            is_spoiler=False,
            attachment_count=0,
        )
    )
    assert low_outcome is not None
    assert low_outcome.rule_id == "LOW"

    high_outcome = program.evaluate(
        DslEvaluationInput(
            rating={"explicit": const_map["const_WD14_EXPL"] + const_map["const_LOW_MARGIN"] + 0.2,
                    "questionable": 0.0},
            metrics={},
            tag_scores={},
            group_patterns=program.group_patterns,
            nude_flags=(),
            is_nsfw_channel=False,
            is_spoiler=False,
            attachment_count=0,
        )
    )
    assert high_outcome is None


def _build_coercion_share_program() -> DslProgram:
    const_map = build_const_map()
    features = {
        "is_qplus": "(rating.explicit >= const_WD14_EXPL) || (rating.questionable >= const_WD14_QUES)",
        "rating_total": "clamp(rating.general + rating.sensitive + rating.questionable + rating.explicit, 0.01, 99.0)",
        "explicit_share": "rating.explicit / rating_total",
        "questionable_share": "rating.questionable / rating_total",
        "qe_share_max": "max(explicit_share, questionable_share)",
        "qe_share_low": "qe_share_max < const_QE_SHARE_MIN",
    }
    rules = [
        DslRule(
            id="STRICT",
            severity="orange",
            priority=10,
            when="is_qplus && !qe_share_low && (coercion_final >= const_COERCION_ORG) && (minor_main < const_MINOR_MAIN) && (minor_suspect < const_SUSPECT_Q)",
            reasons=["final={coercion_final:.2f}", "share={qe_share_max:.2f}"],
        ),
        DslRule(
            id="RESCUE",
            severity="orange",
            priority=5,
            when="is_qplus && qe_share_low && (coercion_final >= const_COERCION_FORCE_ORG) && (minor_main < const_MINOR_MAIN) && (minor_suspect < const_SUSPECT_Q)",
            reasons=["final={coercion_final:.2f}", "share={qe_share_max:.2f}"],
        ),
    ]
    config = RuleConfigV2(features=features, rules=rules)
    return DslProgram.from_config(config, DslPolicy.from_mode("warn"), const_map=const_map)


def test_qe_share_low_blocks_strict_rule() -> None:
    program = _build_coercion_share_program()
    const_map = build_const_map()
    outcome = program.evaluate(
        DslEvaluationInput(
            rating={
                "explicit": const_map["const_WD14_EXPL"] + 0.15,
                "questionable": const_map["const_WD14_QUES"] - 0.10,
                "general": 0.20,
                "sensitive": 0.10,
            },
            metrics={"coercion_final": const_map["const_COERCION_ORG"] + 0.02, "minor_main": 0.10, "minor_suspect": 0.10},
            tag_scores={},
            group_patterns=program.group_patterns,
            nude_flags=(),
            is_nsfw_channel=False,
            is_spoiler=False,
            attachment_count=0,
        )
    )
    assert outcome is None


def test_qe_share_low_allows_rule_when_share_high_enough() -> None:
    program = _build_coercion_share_program()
    const_map = build_const_map()
    outcome = program.evaluate(
        DslEvaluationInput(
            rating={
                "explicit": const_map["const_WD14_EXPL"] + 0.15,
                "questionable": const_map["const_WD14_QUES"] - 0.15,
                "general": 0.02,
                "sensitive": 0.02,
            },
            metrics={"coercion_final": const_map["const_COERCION_ORG"] + 0.02, "minor_main": 0.10, "minor_suspect": 0.10},
            tag_scores={},
            group_patterns=program.group_patterns,
            nude_flags=(),
            is_nsfw_channel=False,
            is_spoiler=False,
            attachment_count=0,
        )
    )
    assert outcome is not None
    assert outcome.rule_id == "STRICT"


def test_coercion_force_rescues_low_share_cases() -> None:
    program = _build_coercion_share_program()
    const_map = build_const_map()
    outcome = program.evaluate(
        DslEvaluationInput(
            rating={
                "explicit": const_map["const_WD14_EXPL"] + 0.15,
                "questionable": const_map["const_WD14_QUES"] - 0.10,
                "general": 0.20,
                "sensitive": 0.10,
            },
            metrics={"coercion_final": const_map["const_COERCION_FORCE_ORG"] + 0.01, "minor_main": 0.10, "minor_suspect": 0.10},
            tag_scores={},
            group_patterns=program.group_patterns,
            nude_flags=(),
            is_nsfw_channel=False,
            is_spoiler=False,
            attachment_count=0,
        )
    )
    assert outcome is not None
    assert outcome.rule_id == "RESCUE"


def _build_animal_penalty_program() -> DslProgram:
    const_map = build_const_map()
    features = {
        "animal_presence": "max('animal_subjects')",
        "animal_fp_peak": "max('animal_fp_subjects')",
        "animal_fp_sum": "topk_sum('animal_fp_subjects', k=3, gt=0.35)",
        "animal_fp_offset": "clamp(max(animal_fp_peak, animal_fp_sum / 1.5), 0, 1)",
        "bestiality_peak": "max('bestiality_direct')",
        "bestiality_effective": "clamp(bestiality_peak - (animal_fp_offset * const_ANIMAL_FP_PENALTY), 0, 1)",
        "animal_presence_effective": "clamp(animal_presence - (animal_fp_offset * const_ANIMAL_FP_PENALTY), 0, 1)",
    }
    rules = [
        DslRule(
            id="BESTIALITY",
            severity="red",
            priority=10,
            when="bestiality_effective >= const_BESTIALITY",
            reasons=["eff={bestiality_effective:.2f}", "raw={bestiality_peak:.2f}", "offset={animal_fp_offset:.2f}"],
        ),
        DslRule(
            id="ANIMAL",
            severity="yellow",
            priority=5,
            when="animal_presence_effective >= const_ANIMAL_PRES",
            reasons=["eff={animal_presence_effective:.2f}"],
        ),
    ]
    config = RuleConfigV2(
        groups={
            "animal_subjects": ["animal"],
            "animal_fp_subjects": ["bug"],
            "bestiality_direct": ["animal_penis"],
        },
        features=features,
        rules=rules,
    )
    return DslProgram.from_config(config, DslPolicy.from_mode("warn"), const_map=const_map)


def test_animal_penalty_blocks_when_insect_present() -> None:
    program = _build_animal_penalty_program()
    const_map = build_const_map()
    outcome = program.evaluate(
        DslEvaluationInput(
            rating={"explicit": const_map["const_WD14_EXPL"] + 0.1, "general": 0.2},
            metrics={},
            tag_scores={"animal": 0.6, "bug": 0.9, "animal_penis": 0.4},
            group_patterns=program.group_patterns,
            nude_flags=(),
            is_nsfw_channel=False,
            is_spoiler=False,
            attachment_count=0,
        )
    )
    assert outcome is None


def test_animal_penalty_keeps_true_positive() -> None:
    program = _build_animal_penalty_program()
    const_map = build_const_map()
    outcome = program.evaluate(
        DslEvaluationInput(
            rating={"explicit": const_map["const_WD14_EXPL"] + 0.1, "general": 0.2},
            metrics={},
            tag_scores={"animal": 0.6, "bug": 0.05, "animal_penis": 0.4},
            group_patterns=program.group_patterns,
            nude_flags=(),
            is_nsfw_channel=False,
            is_spoiler=False,
            attachment_count=0,
        )
    )
    assert outcome is not None
    assert outcome.rule_id == "BESTIALITY"
