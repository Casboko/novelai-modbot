from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .engine.dsl import DslProgram
from .engine.evaluator import DslEvaluator
from .engine.loader import load_rule_config
from .engine.tag_norm import normalize_tag
from .engine.types import DslPolicy, RuleConfigV2

DEFAULT_RULES_PATH = "configs/rules.yaml"


@dataclass(slots=True)
class EvaluationResult:
    severity: str
    rule_id: Optional[str]
    rule_title: Optional[str]
    reasons: List[str]
    metrics: Dict[str, Any]


class RuleEngine:
    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        policy: DslPolicy | None = None,
        nudenet_config_path: str | None = None,  # legacy互換のため受け取るが未使用
    ) -> None:
        rules_path = Path(config_path or DEFAULT_RULES_PATH)
        _ = nudenet_config_path  # kept for compatibility
        requested_policy = policy or DslPolicy()
        dsl_config, raw_config, resolved_policy = load_rule_config(rules_path, requested_policy)
        if dsl_config is None:
            raise ValueError("rules.yaml must declare version 2 configuration")

        self.policy = resolved_policy
        self.dsl_config: RuleConfigV2 = dsl_config
        self.raw_config = raw_config

        self._program = DslProgram.from_config(dsl_config, self.policy)
        self._evaluator = DslEvaluator(self._program)

        rule_titles = raw_config.get("rule_titles") if isinstance(raw_config.get("rule_titles"), Mapping) else {}
        self._rule_titles: Dict[str, str] = {str(k): str(v) for k, v in rule_titles.items()}

        pattern_map: Dict[str, tuple[str, ...]] = {}
        for group_name, tags in self._program.group_patterns.items():
            canonical_name = normalize_tag(group_name)
            if not canonical_name:
                continue
            normalized = tuple(filter(None, (normalize_tag(tag) for tag in tags)))
            pattern_map[canonical_name] = normalized
        self._group_patterns = pattern_map

    def describe_config(self) -> str:
        groups = len(self._program.group_patterns)
        total_patterns = sum(len(items) for items in self._program.group_patterns.values())
        features = len(self._program.compiled_features)
        rules = len(self._program.compiled_rules)
        return (
            f"policy.mode={self.policy.mode}\n"
            f"dsl=enabled groups={groups} patterns={total_patterns} features={features} rules={rules}"
        )

    def evaluate(self, analysis: Mapping[str, Any]) -> EvaluationResult:
        outcome = self._evaluator.evaluate(analysis)
        rule_title = self._rule_titles.get(outcome.rule_id or "") if outcome.rule_id else None
        return EvaluationResult(
            severity=outcome.severity,
            rule_id=outcome.rule_id,
            rule_title=rule_title,
            reasons=list(outcome.reasons),
            metrics=dict(outcome.metrics),
        )

    @property
    def rule_titles(self) -> Dict[str, str]:
        return dict(self._rule_titles)

    @property
    def groups(self) -> Dict[str, tuple[str, ...]]:
        return {name: tuple(tags) for name, tags in self._group_patterns.items()}
