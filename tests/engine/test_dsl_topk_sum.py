from __future__ import annotations

from app.engine.dsl import DslEvaluationInput
from app.engine.types import DslPolicy, DslRule, RuleConfigV2
from app.engine import DslProgram


def _build_program() -> DslProgram:
    config = RuleConfigV2(
        groups={
            "sexual_theme": ["tag_a", "tag_b", "tag_c", "tag_d"],
        },
        features={
            "top2": "topk_sum('sexual_theme', 2)",
            "top3_gt": "topk_sum('sexual_theme', 3, gt=0.3)",
            "top0": "topk_sum('sexual_theme', 0)",
        },
        rules=[
            DslRule(
                id="PASS",
                severity="green",
                priority=0,
                when="top2 >= 0",
            )
        ],
    )
    return DslProgram.from_config(config, DslPolicy.from_mode("warn"))


def _evaluation_input(program: DslProgram) -> DslEvaluationInput:
    return DslEvaluationInput(
        rating={},
        metrics={},
        tag_scores={
            "tag_a": 0.9,
            "tag_b": 0.5,
            "tag_c": 0.2,
            "tag_d": 0.05,
        },
        group_patterns=program.group_patterns,
        nude_flags=(),
        is_nsfw_channel=False,
        is_spoiler=False,
        attachment_count=0,
    )


def test_topk_sum_feature_values() -> None:
    program = _build_program()
    outcome = program.evaluate(_evaluation_input(program))
    assert outcome is not None
    features = outcome.diagnostics["features"]
    # 上位2件: 0.9 + 0.5
    assert features["top2"] == 1.4
    # gt=0.3 で下限を超えるのは 0.9 / 0.5 のみ
    assert features["top3_gt"] == 1.4
    # k=0 は 0
    assert features["top0"] == 0.0


def test_topk_sum_filters_by_threshold_and_limit() -> None:
    program = DslProgram.from_config(
        RuleConfigV2(
            groups={"g": ["a", "b", "c"]},
            features={"limited": "topk_sum('g', 5, gt=0.95)"},
            rules=[DslRule(id="P", severity="green", when="limited >= 0")],
        ),
        DslPolicy.from_mode("warn"),
    )
    outcome = program.evaluate(
        DslEvaluationInput(
            rating={},
            metrics={},
            tag_scores={"a": 0.96, "b": 0.94, "c": 0.5},
            group_patterns=program.group_patterns,
            nude_flags=(),
            is_nsfw_channel=False,
            is_spoiler=False,
            attachment_count=0,
        )
    )
    assert outcome is not None
    features = outcome.diagnostics["features"]
    # 閾値に届くのは 0.96 のみ
    assert features["limited"] == 0.96


def test_topk_sum_returns_zero_for_missing_group() -> None:
    program = DslProgram.from_config(
        RuleConfigV2(
            groups={},
            features={"missing": "topk_sum('unknown', 3)"},
            rules=[DslRule(id="P", severity="green", when="missing >= 0")],
        ),
        DslPolicy.from_mode("warn"),
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
    assert outcome is not None
    features = outcome.diagnostics["features"]
    assert features["missing"] == 0.0
