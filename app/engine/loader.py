from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .dsl_runtime import list_builtin_functions, list_builtin_identifiers
from .dsl_utils import compile_expr, extract_braced_expressions
from .tag_norm import normalize_tag
from .types import (
    DslMode,
    DslPolicy,
    DslRule,
    LoadResult,
    LoadedConfig,
    RuleConfigV2,
    ValidationIssue,
)

__all__ = [
    "load_rules_result",
    "load_rule_config",
    "build_const_map",
    "extract_const_overrides",
    "load_const_overrides_from_path",
]

_ALLOWED_KEYS = {"version", "dsl_mode", "rule_titles", "groups", "features", "rules", "thresholds"}
_GROUP_FUNCS = {"sum", "max", "any", "count", "topk_sum"}
_ALLOWED_FUNCTIONS = list_builtin_functions()
_BASE_IDENTIFIERS = list_builtin_identifiers() | {"rating", "channel", "message", "nude", "attachment_count", "metrics"}


def _load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp) or {}
    except FileNotFoundError:
        return {}
    except OSError:
        return {}
    if isinstance(data, Mapping):
        return dict(data)
    return {}


def _flatten_thresholds(payload: Mapping[str, Any] | None, prefix: str = "") -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    flattened: dict[str, float] = {}
    for key, value in payload.items():
        merged_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_thresholds(value, merged_key))
        elif isinstance(value, (int, float)):
            flattened[merged_key] = float(value)
    return flattened


THRESHOLD_TO_CONST: dict[str, str] = {
    "wd14_questionable": "const_WD14_QUES",
    "wd14_explicit": "const_WD14_EXPL",
    "const.wd14.questionable": "const_WD14_QUES",
    "const.wd14.explicit": "const_WD14_EXPL",
    "exposure.area.mid": "const_EXPOSURE_MID",
    "exposure.area.high": "const_EXPOSURE_HIGH",
    "const.exposure.area.mid": "const_EXPOSURE_MID",
    "const.exposure.area.high": "const_EXPOSURE_HIGH",
    "sexual.main.strong": "const_SEXUAL_MAIN_STRONG",
    "const.sexual.main.strong": "const_SEXUAL_MAIN_STRONG",
    "low_margin": "const_LOW_MARGIN",
    "const.low_margin": "const_LOW_MARGIN",
    "minor.main": "const_MINOR_MAIN",
    "const.minor.main": "const_MINOR_MAIN",
    "minor.sit_enabled": "const_MINOR_SIT_ENABLED",
    "const.minor.sit_enabled": "const_MINOR_SIT_ENABLED",
    "suspect.explicit": "const_SUSPECT_E",
    "suspect.questionable": "const_SUSPECT_Q",
    "const.suspect.explicit": "const_SUSPECT_E",
    "const.suspect.questionable": "const_SUSPECT_Q",
    "coercion.pair": "const_COERCION_PAIR",
    "coercion.org": "const_COERCION_ORG",
    "const.coercion.pair": "const_COERCION_PAIR",
    "const.coercion.org": "const_COERCION_ORG",
    "coercion.main_strict": "const_COERCION_MAIN_STRICT",
    "const.coercion.main_strict": "const_COERCION_MAIN_STRICT",
    "coercion.pain_cnt": "const_COERCION_PAIN_CNT",
    "const.coercion.pain_cnt": "const_COERCION_PAIN_CNT",
    "coercion.sub_min": "const_COERCION_SUB_MIN",
    "const.coercion.sub_min": "const_COERCION_SUB_MIN",
    "coercion.uncon_min": "const_COERCION_UNCON_MIN",
    "const.coercion.uncon_min": "const_COERCION_UNCON_MIN",
    "coercion.force_org": "const_COERCION_FORCE_ORG",
    "const.coercion.force_org": "const_COERCION_FORCE_ORG",
    "qe.share_min": "const_QE_SHARE_MIN",
    "const.qe.share_min": "const_QE_SHARE_MIN",
    "bestiality.direct": "const_BESTIALITY",
    "bestiality.direct_gate": "const_BESTIALITY",
    "const.bestiality.direct": "const_BESTIALITY",
    "animal.presence": "const_ANIMAL_PRES",
    "animal.presence_gate": "const_ANIMAL_PRES",
    "const.animal.presence": "const_ANIMAL_PRES",
    "animal.fp_penalty": "const_ANIMAL_FP_PENALTY",
    "const.animal.fp_penalty": "const_ANIMAL_FP_PENALTY",
    "gore.main": "const_GORE_MAIN",
    "gore.main_gate": "const_GORE_MAIN",
    "const.gore.main": "const_GORE_MAIN",
    "gore.density": "const_GORE_DENSITY",
    "gore.density_gate": "const_GORE_DENSITY",
    "const.gore.density": "const_GORE_DENSITY",
    "gore.topk": "const_GORE_TOPK",
    "gore.topk_gate": "const_GORE_TOPK",
    "const.gore.topk": "const_GORE_TOPK",
    "gore.org": "const_GORE_ORG",
    "gore.org_gate": "const_GORE_ORG",
    "const.gore.org": "const_GORE_ORG",
    "gore.red": "const_GORE_RED",
    "gore.red_gate": "const_GORE_RED",
    "const.gore.red": "const_GORE_RED",
}


DEFAULT_CONSTS: dict[str, float] = {
    "const_WD14_EXPL": 0.20,
    "const_WD14_QUES": 0.35,
    "const_EXPOSURE_MID": 0.15,
    "const_EXPOSURE_HIGH": 0.20,
    "const_SEXUAL_MAIN_STRONG": 0.60,
    "const_LOW_MARGIN": 0.10,
    "const_MINOR_MAIN": 0.30,
    "const_MINOR_SIT_ENABLED": 0.0,
    "const_SUSPECT_E": 0.60,
    "const_SUSPECT_Q": 0.45,
    "const_COERCION_PAIR": 0.55,
    "const_COERCION_ORG": 0.95,
    "const_COERCION_MAIN_STRICT": 0.50,
    "const_COERCION_PAIN_CNT": 1.0,
    "const_COERCION_SUB_MIN": 1.0,
    "const_COERCION_UNCON_MIN": 0.35,
    "const_COERCION_FORCE_ORG": 0.98,
    "const_QE_SHARE_MIN": 0.55,
    "const_BESTIALITY": 0.20,
    "const_ANIMAL_PRES": 0.35,
    "const_ANIMAL_FP_PENALTY": 0.80,
    "const_GORE_MAIN": 0.30,
    "const_GORE_DENSITY": 4.0,
    "const_GORE_TOPK": 0.70,
    "const_GORE_ORG": 0.70,
    "const_GORE_RED": 0.85,
}


def extract_const_overrides(thresholds: Mapping[str, Any] | None) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for dotted_key, numeric in _flatten_thresholds(thresholds).items():
        mapped = THRESHOLD_TO_CONST.get(dotted_key)
        if mapped:
            overrides[mapped] = numeric
    return overrides


def _load_scl_thresholds(path: str | Path) -> Mapping[str, Any]:
    return _load_yaml(path)


def build_const_map(
    *,
    rule_thresholds: Mapping[str, Any] | None = None,
    extra_overrides: Mapping[str, float] | None = None,
    scl_thresholds_path: str | Path = "configs/scl/scl_thresholds.yaml",
) -> dict[str, float]:
    consts = dict(DEFAULT_CONSTS)

    scl_payload = _load_scl_thresholds(scl_thresholds_path)
    consts.update(extract_const_overrides(scl_payload))

    if rule_thresholds:
        consts.update(extract_const_overrides(rule_thresholds))

    if extra_overrides:
        for key, value in extra_overrides.items():
            if key.startswith("const_"):
                try:
                    consts[key] = float(value)
                except (TypeError, ValueError):
                    continue

    return consts


def load_const_overrides_from_path(path: str | Path) -> dict[str, float]:
    payload = _load_yaml(path)
    if not payload:
        return {}

    overrides: dict[str, float] = {}
    for key, value in payload.items():
        if not key.startswith("const_"):
            continue
        try:
            overrides[key] = float(value)
        except (TypeError, ValueError):
            continue

    thresholds_payload = None
    if isinstance(payload.get("thresholds"), Mapping):
        thresholds_payload = payload["thresholds"]  # type: ignore[assignment]
    else:
        thresholds_payload = payload

    overrides.update(extract_const_overrides(thresholds_payload))
    return overrides


@dataclass(slots=True)
class _ExpressionCheck:
    node: ast.Expression
    unknown_identifiers: set[str]
    missing_groups: set[str]


class _ExpressionInspector(ast.NodeVisitor):
    def __init__(self, *, groups: Mapping[str, Sequence[str]]) -> None:
        self.groups = {normalize_tag(name): tuple(values) for name, values in groups.items()}
        self.identifiers: set[str] = set()
        self.missing_groups: set[str] = set()

    def visit_Name(self, node: ast.Name) -> Any:  # noqa: D401, ANN401
        if node.id not in _ALLOWED_FUNCTIONS:
            self.identifiers.add(node.id)
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:  # noqa: D401, ANN401
        if isinstance(node.value, ast.Name):
            base = node.value.id
            self.identifiers.add(base)
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:  # noqa: D401, ANN401
        func = node.func
        if isinstance(func, ast.Name) and func.id in _GROUP_FUNCS:
            if node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    canonical = normalize_tag(first.value)
                    if canonical and canonical not in self.groups:
                        self.missing_groups.add(first.value)
        return self.generic_visit(node)


def _record_issue(
    issues: list[ValidationIssue],
    counts: dict[str, int],
    level: str,
    code: str,
    where: str,
    msg: str,
    hint: str | None = None,
) -> None:
    issues.append(ValidationIssue(level=level, code=code, where=where, msg=msg, hint=hint))
    counts["errors" if level == "error" else "warnings"] += 1


def _ensure_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _normalize_group_patterns(
    name: str,
    patterns: Iterable[Any],
    *,
    strict: bool,
    issues: list[ValidationIssue],
    counts: dict[str, int],
    examples: dict[str, set[str]],
) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    originals: dict[str, set[str]] = {}
    for pattern in patterns:
        if not isinstance(pattern, str):
            level = "error" if strict else "warning"
            _record_issue(
                issues,
                counts,
                level,
                "R2-G001",
                f"groups.{name}",
                f"Group '{name}' contains non-string entry: {pattern!r}",
            )
            continue
        canonical = normalize_tag(pattern)
        if not canonical:
            _record_issue(
                issues,
                counts,
                "warning",
                "R2-G001",
                f"groups.{name}",
                "Empty pattern after normalization was skipped",
            )
            continue
        if canonical in seen:
            counts["collisions"] += 1
            originals.setdefault(canonical, {pattern}).add(pattern)
            continue
        seen.add(canonical)
        originals.setdefault(canonical, set()).add(pattern)
        normalized.append(canonical)
    for canonical, variants in originals.items():
        if len(variants) > 1:
            examples.setdefault(canonical, set()).update(variants)
    return normalized


def _check_expression(
    expr: str,
    *,
    groups: Mapping[str, Sequence[str]],
    allowed_identifiers: set[str],
) -> _ExpressionCheck:
    node = compile_expr(expr)
    inspector = _ExpressionInspector(groups=groups)
    inspector.visit(node)
    unknown = {name for name in inspector.identifiers if name not in allowed_identifiers}
    return _ExpressionCheck(node=node, unknown_identifiers=unknown, missing_groups=inspector.missing_groups)


def load_rules_result(
    path: str | Path,
    *,
    policy: DslPolicy | None = None,
    override_mode: DslMode | None = None,
) -> LoadResult:
    logger = policy.logger if policy else logging.getLogger("app.engine.dsl")
    counts = {
        "rules": 0,
        "features": 0,
        "groups": 0,
        "errors": 0,
        "warnings": 0,
        "disabled_rules": 0,
        "disabled_features": 0,
        "placeholder_fixes": 0,
        "unknown_keys": 0,
        "collisions": 0,
    }
    issues: list[ValidationIssue] = []
    fatal = False
    invalid = False

    try:
        with Path(path).open("r", encoding="utf-8") as fp:
            raw_data = yaml.safe_load(fp) or {}
    except OSError as exc:
        _record_issue(
            issues,
            counts,
            "error",
            "R2-IO",
            "top-level",
            f"Failed to read rules file: {exc}",
        )
        return LoadResult(status="error", mode=override_mode or (policy.mode if policy else "warn"), config=None, issues=issues, counts=counts)
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        _record_issue(
            issues,
            counts,
            "error",
            "R2-YAML",
            "top-level",
            f"Failed to parse YAML: {exc}",
        )
        return LoadResult(status="error", mode=override_mode or (policy.mode if policy else "warn"), config=None, issues=issues, counts=counts)

    if not isinstance(raw_data, Mapping):
        _record_issue(issues, counts, "error", "R2-V001", "top-level", "Rules configuration must be a mapping")
        return LoadResult(status="error", mode=override_mode or (policy.mode if policy else "warn"), config=None, issues=issues, counts=counts)

    yaml_mode = str(raw_data.get("dsl_mode", "")).strip().lower() or None
    if yaml_mode not in {"warn", "strict"}:
        yaml_mode = None
    cli_mode = override_mode or (policy.mode if policy else None)
    mode: DslMode = (cli_mode or yaml_mode or "warn")  # type: ignore[assignment]

    strict = mode == "strict"

    unknown_keys = sorted(set(raw_data.keys()) - _ALLOWED_KEYS)
    if unknown_keys:
        counts["unknown_keys"] += len(unknown_keys)
        msg = ", ".join(unknown_keys)
        level = "error" if strict else "warning"
        hint = "They are ignored in DSL v2. Remove or migrate them under groups/features."
        _record_issue(issues, counts, level, "R2-K001", "top-level", f"Unknown keys detected: {msg}", hint)
        if strict:
            fatal = True

    version_raw = raw_data.get("version")
    if version_raw != 2:
        _record_issue(issues, counts, "error", "R2-V001", "version", "rules.yaml must declare version: 2")
        if strict:
            fatal = True
        else:
            invalid = True

    rule_titles_raw = raw_data.get("rule_titles")
    if not isinstance(rule_titles_raw, Mapping):
        _record_issue(issues, counts, "error", "R2-T000", "rule_titles", "rule_titles must be a mapping of id -> title")
        rule_titles: dict[str, str] = {}
        invalid = True
    else:
        rule_titles = {str(k): str(v) for k, v in rule_titles_raw.items()}

    groups_raw = raw_data.get("groups")
    groups: dict[str, list[str]] = {}
    if groups_raw is None:
        groups_raw = {}
    if not isinstance(groups_raw, Mapping):
        _record_issue(issues, counts, "warning", "R2-G001", "groups", "groups must be a mapping")
        groups_raw = {}
    collision_examples: dict[str, set[str]] = {}
    for group_name, patterns in groups_raw.items():
        canonical = normalize_tag(group_name)
        if not canonical:
            _record_issue(
                issues,
                counts,
                "warning",
                "R2-G001",
                "groups",
                f"Group '{group_name}' is empty after normalization",
            )
            continue
        if canonical in groups:
            _record_issue(
                issues,
                counts,
                "warning",
                "R2-G001",
                "groups",
                f"Group '{group_name}' duplicates canonical name '{canonical}'",
            )
            continue
        if not isinstance(patterns, Iterable) or isinstance(patterns, (str, bytes)):
            _record_issue(
                issues,
                counts,
                "error",
                "R2-G001",
                f"groups.{group_name}",
                "Group entries must be a sequence of strings",
            )
            if strict:
                fatal = True
            else:
                invalid = True
            continue
        normalized_items = _normalize_group_patterns(
            group_name,
            patterns,
            strict=strict,
            issues=issues,
            counts=counts,
            examples=collision_examples,
        )
        groups[canonical] = normalized_items
    counts["groups"] = len(groups)
    if counts["collisions"]:
        samples = "; ".join(
            f"{key} <= [{', '.join(sorted(values))}]" for key, values in list(collision_examples.items())[:5]
        )
        logger.debug("dsl loader: group pattern collisions=%d %s", counts["collisions"], samples)

    const_names = set(DEFAULT_CONSTS.keys()) | set(extract_const_overrides(raw_data.get("thresholds")).keys())

    base_allowed_identifiers = set(_BASE_IDENTIFIERS) | set(_ALLOWED_FUNCTIONS) | const_names

    features_raw = _ensure_mapping(raw_data.get("features"))
    features: dict[str, str] = {}
    disabled_features: set[str] = set()
    all_feature_names = {str(name) for name in features_raw.keys()}
    feature_allowed = base_allowed_identifiers | all_feature_names
    for feature_name, expression in features_raw.items():
        if not isinstance(expression, str):
            _record_issue(
                issues,
                counts,
                "error",
                "R2-E001",
                f"features.{feature_name}",
                "Feature expression must be a string",
            )
            invalid = True
            disabled_features.add(feature_name)
            continue
        try:
            check = _check_expression(expression, groups=groups, allowed_identifiers=feature_allowed)
        except Exception as exc:  # pragma: no cover - compile_expr should raise DslValidationError
            _record_issue(
                issues,
                counts,
                "error",
                "R2-E001",
                f"features.{feature_name}",
                f"Invalid expression: {exc}",
            )
            invalid = True
            disabled_features.add(feature_name)
            continue
        if check.unknown_identifiers:
            ident_list = ", ".join(sorted(check.unknown_identifiers))
            _record_issue(
                issues,
                counts,
                "error",
                "R2-E002",
                f"features.{feature_name}",
                f"Unknown identifiers: {ident_list}",
            )
            invalid = True
            disabled_features.add(feature_name)
            continue
        if check.missing_groups:
            missing = ", ".join(sorted(check.missing_groups))
            _record_issue(
                issues,
                counts,
                "error",
                "R2-G001",
                f"features.{feature_name}",
                f"Unknown group reference(s): {missing}",
            )
            invalid = True
            disabled_features.add(feature_name)
            continue
        features[feature_name] = expression
    counts["features"] = len(features)
    counts["disabled_features"] = len(disabled_features)
    if disabled_features:
        invalid = True

    rules_raw = raw_data.get("rules")
    if not isinstance(rules_raw, Sequence):
        _record_issue(issues, counts, "error", "R2-V001", "rules", "rules must be a sequence")
        return LoadResult(status="error", mode=mode, config=None, issues=issues, counts=counts)

    seen_rule_ids: set[str] = set()
    valid_rules: list[DslRule] = []
    disabled_rules: list[str] = []
    rule_allowed_identifiers = base_allowed_identifiers | set(features.keys())

    for index, entry in enumerate(rules_raw):
        where_prefix = f"rules[{index}]"
        if not isinstance(entry, Mapping):
            _record_issue(issues, counts, "error", "R2-V001", where_prefix, "Rule entry must be a mapping")
            invalid = True
            continue
        rule_id = str(entry.get("id", "")).strip()
        if not rule_id:
            _record_issue(issues, counts, "error", "R2-V001", where_prefix, "Rule requires an id")
            invalid = True
            continue
        if rule_id in seen_rule_ids:
            _record_issue(issues, counts, "error", "R2-D001", where_prefix, f"Duplicate rule id: {rule_id}")
            invalid = True
            disabled_rules.append(rule_id)
            continue
        seen_rule_ids.add(rule_id)

        severity = str(entry.get("severity", "")).strip().lower()
        if severity not in {"red", "orange", "yellow", "green"}:
            _record_issue(issues, counts, "error", "R2-V001", where_prefix, f"Unsupported severity: {severity}")
            invalid = True
            disabled_rules.append(rule_id)
            continue

        priority_value = entry.get("priority", 0)
        try:
            priority = int(priority_value)
        except (TypeError, ValueError):
            _record_issue(
                issues,
                counts,
                "warning",
                "R2-V001",
                f"{where_prefix}.priority",
                "Priority must be an integer; defaulting to 0",
            )
            priority = 0
        when_expr = entry.get("when")
        if not isinstance(when_expr, str) or not when_expr.strip():
            _record_issue(
                issues,
                counts,
                "error",
                "R2-V001",
                f"{where_prefix}.when",
                "Rule requires a non-empty when expression",
            )
            invalid = True
            disabled_rules.append(rule_id)
            continue
        try:
            check = _check_expression(when_expr, groups=groups, allowed_identifiers=rule_allowed_identifiers)
        except Exception as exc:
            _record_issue(
                issues,
                counts,
                "error",
                "R2-E001",
                f"{where_prefix}.when",
                f"Invalid expression: {exc}",
            )
            invalid = True
            disabled_rules.append(rule_id)
            continue
        if check.unknown_identifiers:
            ident_list = ", ".join(sorted(check.unknown_identifiers))
            _record_issue(
                issues,
                counts,
                "error",
                "R2-E002",
                f"{where_prefix}.when",
                f"Unknown identifiers: {ident_list}",
            )
            invalid = True
            disabled_rules.append(rule_id)
            continue
        if check.missing_groups:
            missing = ", ".join(sorted(check.missing_groups))
            _record_issue(
                issues,
                counts,
                "error",
                "R2-G001",
                f"{where_prefix}.when",
                f"Unknown group reference(s): {missing}",
            )
            invalid = True
            disabled_rules.append(rule_id)
            continue

        reasons_raw = entry.get("reasons") or []
        reasons: list[str] = []
        removed_placeholders = 0
        if reasons_raw and not isinstance(reasons_raw, Sequence):
            _record_issue(
                issues,
                counts,
                "warning",
                "R2-P001",
                f"{where_prefix}.reasons",
                "reasons must be a list of strings",
            )
            reasons_raw = []
        for reason in reasons_raw:
            if not isinstance(reason, str):
                _record_issue(
                    issues,
                    counts,
                    "warning",
                    "R2-P001",
                    f"{where_prefix}.reasons",
                    "Reason entries must be strings",
                )
                continue
            valid_reason = True
            for fragment in extract_braced_expressions(reason):
                try:
                    fragment_check = _check_expression(
                        fragment,
                        groups=groups,
                        allowed_identifiers=rule_allowed_identifiers,
                    )
                except Exception as exc:
                    _record_issue(
                        issues,
                        counts,
                        "error",
                        "R2-P001",
                        f"{where_prefix}.reasons",
                        f"Invalid placeholder expression '{fragment}': {exc}",
                    )
                    invalid = True
                    valid_reason = False
                    placeholder_issue = True
                    break
                if fragment_check.unknown_identifiers:
                    ident_list = ", ".join(sorted(fragment_check.unknown_identifiers))
                    _record_issue(
                        issues,
                        counts,
                        "error",
                        "R2-P001",
                        f"{where_prefix}.reasons",
                        f"Unknown identifiers in placeholder '{fragment}': {ident_list}",
                    )
                    invalid = True
                    valid_reason = False
                    placeholder_issue = True
                    break
                if fragment_check.missing_groups:
                    missing = ", ".join(sorted(fragment_check.missing_groups))
                    _record_issue(
                        issues,
                        counts,
                        "error",
                        "R2-P001",
                        f"{where_prefix}.reasons",
                        f"Unknown group reference(s) in placeholder '{fragment}': {missing}",
                    )
                    invalid = True
                    valid_reason = False
                    break
            if valid_reason:
                reasons.append(reason)
            else:
                removed_placeholders += 1
        if removed_placeholders:
            counts["placeholder_fixes"] += removed_placeholders

        stop = bool(entry.get("stop", False))

        rule = DslRule(
            id=rule_id,
            severity=severity,  # type: ignore[arg-type]
            when=when_expr,
            priority=priority,
            reasons=reasons,
            stop=stop,
        )
        valid_rules.append(rule)

    counts["rules"] = len(valid_rules)
    counts["disabled_rules"] = len(disabled_rules)
    if disabled_rules:
        invalid = True

    missing_titles = [rule.id for rule in valid_rules if rule.id not in rule_titles]
    if missing_titles:
        ids = ", ".join(sorted(missing_titles))
        _record_issue(
            issues,
            counts,
            "error",
            "R2-T001",
            "rule_titles",
            f"Missing rule_titles entry for: {ids}",
        )
        invalid = True

    if fatal or (strict and counts["errors"] > 0):
        return LoadResult(status="error", mode=mode, config=None, issues=issues, counts=counts)

    status: str = "invalid" if invalid else "ok"

    raw_snapshot = dict(raw_data)
    rule_config = RuleConfigV2(groups=groups, features=features, rules=valid_rules)
    loaded = LoadedConfig(rule_titles=rule_titles, config=rule_config, raw=raw_snapshot)
    summary = {
        "rules_ok": [rule.id for rule in valid_rules],
        "rules_disabled": disabled_rules,
        "features_disabled": sorted(disabled_features),
    }
    return LoadResult(status=status, mode=mode, config=loaded, issues=issues, counts=counts, summary=summary)


def load_rule_config(path: str | Path, policy: DslPolicy | None = None) -> tuple[RuleConfigV2 | None, dict[str, Any], DslPolicy]:
    result = load_rules_result(path, policy=policy)
    effective_policy = policy
    if effective_policy is None or effective_policy.mode != result.mode:
        if effective_policy is not None:
            logger = effective_policy.logger
        else:
            logger = None
        effective_policy = DslPolicy.from_mode(result.mode)
        if logger is not None:
            effective_policy.logger = logger  # type: ignore[assignment]
    logger = effective_policy.logger
    for issue in result.issues:
        message = f"[{issue.code}] {issue.where}: {issue.msg}"
        if issue.hint:
            message = f"{message} (hint: {issue.hint})"
        if issue.level == "error":
            logger.error(message)
        else:
            logger.warning(message)
    if result.status == "error" or not result.config:
        raise ValueError("Failed to load DSL rule configuration")
    return result.config.config, result.config.raw, effective_policy
