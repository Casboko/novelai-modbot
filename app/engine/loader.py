from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .dsl_utils import extract_braced_expressions, validate_expr
from .tag_norm import normalize_tag
from .types import DslPolicy, DslRule, RuleConfigV2

__all__ = ["load_rule_config"]


def _ensure_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def load_rule_config(path: str | Path, policy: DslPolicy | None = None) -> tuple[RuleConfigV2 | None, dict[str, Any], DslPolicy]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("rules configuration must be a mapping")

    mode = str(raw.get("dsl_mode", (policy.mode if policy else "warn"))).strip().lower() if raw else "warn"
    resolved_policy = policy or DslPolicy.from_mode(mode)
    if resolved_policy.mode != mode:
        resolved_policy = DslPolicy.from_mode(mode)

    version = int(raw.get("version", 1))
    if version != 2:
        return None, dict(raw), resolved_policy

    groups_cfg = _ensure_mapping(raw.get("groups"))
    groups: dict[str, list[str]] = {}
    for group_name, patterns in groups_cfg.items():
        canonical = normalize_tag(group_name)
        if not canonical:
            resolved_policy.warn_once(f"DSL group name '{group_name}' is empty after normalization", key=f"group:{group_name}")
            continue
        if not isinstance(patterns, (list, tuple)):
            if resolved_policy.strict:
                raise ValueError(f"group '{group_name}' must be a sequence")
            resolved_policy.warn_once(f"DSL group '{group_name}' ignored because value is not a sequence", key=f"group:{group_name}:type")
            continue
        normalized_items: list[str] = []
        for pattern in patterns:
            if not isinstance(pattern, str):
                resolved_policy.warn_once(
                    f"DSL group '{group_name}' contains non-string entry '{pattern}'",
                    key=f"group:{group_name}:entry:{pattern}",
                )
                if resolved_policy.strict:
                    raise ValueError(f"group '{group_name}' contains non-string value")
                continue
            normalized = normalize_tag(pattern)
            if not normalized:
                resolved_policy.warn_once(
                    f"DSL group '{group_name}' contains empty pattern after normalization",
                    key=f"group:{group_name}:empty",
                )
                continue
            normalized_items.append(normalized)
        groups[canonical] = normalized_items

    features_cfg = _ensure_mapping(raw.get("features"))
    features: dict[str, str] = {}
    for feature_name, expression in features_cfg.items():
        if not isinstance(expression, str):
            if resolved_policy.strict:
                raise ValueError(f"feature '{feature_name}' must be a string expression")
            resolved_policy.warn_once(
                f"DSL feature '{feature_name}' ignored because expression is not a string",
                key=f"feature:{feature_name}:type",
            )
            continue
        try:
            validate_expr(expression)
        except Exception as exc:
            if resolved_policy.strict:
                raise
            resolved_policy.warn_once(
                f"DSL feature '{feature_name}' skipped due to invalid expression: {exc}",
                key=f"feature:{feature_name}:expr",
            )
            continue
        features[feature_name] = expression

    rules_cfg = raw.get("rules")
    if not isinstance(rules_cfg, list):
        raise ValueError("rules section must be a list")

    seen_ids: set[str] = set()
    rules: list[DslRule] = []
    for entry in rules_cfg:
        if not isinstance(entry, Mapping):
            if resolved_policy.strict:
                raise ValueError("rule entry must be a mapping")
            resolved_policy.warn_once("DSL rule skipped because entry is not a mapping", key=f"rule:{len(rules)}:type")
            continue
        rid = str(entry.get("id", "")).strip()
        if not rid:
            if resolved_policy.strict:
                raise ValueError("rule id is required")
            resolved_policy.warn_once("DSL rule skipped because id is empty", key="rule:empty-id")
            continue
        if rid in seen_ids:
            if resolved_policy.strict:
                raise ValueError(f"duplicate rule id: {rid}")
            resolved_policy.warn_once(f"DSL rule '{rid}' skipped because id is duplicated", key=f"rule:{rid}:dup")
            continue
        severity = str(entry.get("severity", "")).strip().lower()
        if severity not in {"red", "orange", "yellow", "green"}:
            if resolved_policy.strict:
                raise ValueError(f"unsupported severity for rule '{rid}': {severity}")
            resolved_policy.warn_once(
                f"DSL rule '{rid}' skipped because severity is invalid",
                key=f"rule:{rid}:severity",
            )
            continue
        priority = int(entry.get("priority", 100))
        if not (0 <= priority <= 999):
            if resolved_policy.strict:
                raise ValueError(f"rule '{rid}' priority must be between 0 and 999")
            resolved_policy.warn_once(
                f"DSL rule '{rid}' skipped because priority is out of range",
                key=f"rule:{rid}:priority",
            )
            continue
        when_expr = entry.get("when", "")
        if not isinstance(when_expr, str) or not when_expr.strip():
            if resolved_policy.strict:
                raise ValueError(f"rule '{rid}' requires non-empty 'when'")
            resolved_policy.warn_once(
                f"DSL rule '{rid}' skipped because 'when' expression is empty",
                key=f"rule:{rid}:when-empty",
            )
            continue
        try:
            validate_expr(when_expr)
        except Exception as exc:
            if resolved_policy.strict:
                raise
            resolved_policy.warn_once(
                f"DSL rule '{rid}' skipped due to invalid 'when' expression: {exc}",
                key=f"rule:{rid}:when",
            )
            continue

        reasons_raw = entry.get("reasons") or []
        reasons: list[str] = []
        if not isinstance(reasons_raw, list):
            if resolved_policy.strict:
                raise ValueError(f"rule '{rid}' reasons must be a list")
            resolved_policy.warn_once(
                f"DSL rule '{rid}' ignored reasons because value is not a list",
                key=f"rule:{rid}:reasons",
            )
            reasons_raw = []
        for tpl in reasons_raw:
            if not isinstance(tpl, str):
                resolved_policy.warn_once(
                    f"rule '{rid}' reason entry ignored because it is not a string",
                    key=f"rule:{rid}:reason:type",
                )
                if resolved_policy.strict:
                    raise ValueError(f"rule '{rid}' reason must be a string")
                continue
            valid = True
            for expr in extract_braced_expressions(tpl):
                try:
                    validate_expr(expr)
                except Exception as exc:
                    valid = False
                    if resolved_policy.strict:
                        raise
                    resolved_policy.warn_once(
                        f"rule '{rid}' reason '{tpl}' ignored due to invalid expression: {exc}",
                        key=f"rule:{rid}:reason:{tpl}",
                    )
                    break
            if valid:
                reasons.append(tpl)
        rules.append(DslRule(id=rid, severity=severity, priority=priority, when=when_expr, reasons=reasons))
        seen_ids.add(rid)

    if not rules:
        raise ValueError("no valid DSL rules were loaded")

    cfg = RuleConfigV2(groups=groups, features=features, rules=rules)
    return cfg, dict(raw), resolved_policy
