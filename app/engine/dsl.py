from __future__ import annotations

import ast
import string
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .dsl_errors import DslRuntimeError
from .dsl_runtime import build_context
from .dsl_utils import compile_expr
from .tag_norm import normalize_tag
from .types import DslPolicy, DslRule, RuleConfigV2

SEVERITY_ORDER = {"green": 0, "yellow": 1, "orange": 2, "red": 3}


class SafeEvaluator(ast.NodeVisitor):
    def __init__(self, namespace: Mapping[str, Any], policy: DslPolicy) -> None:
        self._namespace = namespace
        self._policy = policy
        self._context = "dsl"

    @property
    def strict(self) -> bool:
        return self._policy.strict

    @contextmanager
    def context(self, value: str) -> Any:
        previous = self._context
        self._context = value
        try:
            yield
        finally:
            self._context = previous

    def warn(self, message: str, *, key_suffix: str | None = None) -> None:
        key = f"{self._context}:{key_suffix}" if key_suffix else self._context
        self._policy.warn_once(message, key=key)

    def evaluate(self, node: ast.AST) -> Any:
        return self.visit(node)

    # --- AST visitors -----------------------------------------------------

    def visit_Expression(self, node: ast.Expression) -> Any:  # noqa: D401
        return self.visit(node.body)

    def visit_BoolOp(self, node: ast.BoolOp) -> bool:
        if isinstance(node.op, ast.And):
            for value in node.values:
                if not self.visit(value):
                    return False
            return True
        if isinstance(node.op, ast.Or):
            for value in node.values:
                if self.visit(value):
                    return True
            return False
        raise DslRuntimeError("unsupported boolean operation")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return not bool(operand)
        if isinstance(node.op, ast.USub):
            return -float(operand)
        if isinstance(node.op, ast.UAdd):
            return float(operand)
        raise DslRuntimeError("unsupported unary operation")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return float(left) + float(right)
        if isinstance(node.op, ast.Sub):
            return float(left) - float(right)
        if isinstance(node.op, ast.Mult):
            return float(left) * float(right)
        if isinstance(node.op, ast.Div):
            try:
                return float(left) / float(right)
            except ZeroDivisionError as exc:
                if self.strict:
                    raise DslRuntimeError("division by zero") from exc
                self.warn("division by zero encountered", key_suffix="divzero")
                return 0.0
        raise DslRuntimeError("unsupported arithmetic operation")

    def visit_Compare(self, node: ast.Compare) -> bool:
        left = self.visit(node.left)
        result = True
        for operator, comparator in zip(node.ops, node.comparators, strict=False):
            right = self.visit(comparator)
            if isinstance(operator, ast.Eq):
                result = left == right
            elif isinstance(operator, ast.NotEq):
                result = left != right
            elif isinstance(operator, ast.Gt):
                result = float(left) > float(right)
            elif isinstance(operator, ast.GtE):
                result = float(left) >= float(right)
            elif isinstance(operator, ast.Lt):
                result = float(left) < float(right)
            elif isinstance(operator, ast.LtE):
                result = float(left) <= float(right)
            else:  # pragma: no cover
                raise DslRuntimeError("unsupported comparison operator")
            if not result:
                return False
            left = right
        return True

    def visit_Call(self, node: ast.Call) -> Any:
        func_obj = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}
        if not callable(func_obj):
            if self.strict:
                raise DslRuntimeError("attempted to call a non-callable object")
            self.warn("callable expected but object is not callable", key_suffix="noncallable")
            return 0.0
        try:
            return func_obj(*args, **kwargs)
        except ZeroDivisionError as exc:
            if self.strict:
                raise DslRuntimeError("division by zero") from exc
            self.warn("division by zero inside function call", key_suffix="call-divzero")
            return 0.0
        except Exception as exc:  # pragma: no cover - defensive fallback
            if self.strict:
                raise DslRuntimeError(str(exc)) from exc
            self.warn(f"function call failed: {exc}", key_suffix="call-error")
            return 0.0

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self._namespace:
            return self._namespace[node.id]
        if self.strict:
            raise DslRuntimeError(f"unknown identifier: {node.id}")
        self.warn(f"unknown identifier '{node.id}'", key_suffix=f"name:{node.id}")
        return 0.0

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        base = self.visit(node.value)
        attr = node.attr
        if isinstance(base, Mapping) and attr in base:
            return base[attr]
        if hasattr(base, attr):
            value = getattr(base, attr)
            if callable(value):
                return value
            return value
        if self.strict:
            raise DslRuntimeError(f"attribute '{attr}' is not available")
        self.warn(f"attribute '{attr}' not available", key_suffix=f"attr:{attr}")
        return 0.0

    def visit_Constant(self, node: ast.Constant) -> Any:  # noqa: D401
        return node.value


@dataclass(slots=True)
class CompiledFeature:
    name: str
    expression: ast.Expression


@dataclass(slots=True)
class ReasonSegment:
    literal: str
    expression: ast.Expression | None
    format_spec: str | None
    conversion: str | None


@dataclass(slots=True)
class ReasonTemplate:
    raw: str
    segments: tuple[ReasonSegment, ...]

    def render(self, evaluator: SafeEvaluator, context: str) -> str:
        parts: list[str] = []
        for index, segment in enumerate(self.segments):
            if segment.expression is None:
                parts.append(segment.literal)
                continue
            with evaluator.context(f"{context}:expr{index}"):
                try:
                    value = evaluator.evaluate(segment.expression)
                except DslRuntimeError as exc:
                    if evaluator.strict:
                        raise
                    evaluator.warn(f"reason expression failed: {exc}", key_suffix=f"expr{index}")
                    value = ""
            conversion = segment.conversion
            if conversion == "s":
                value = str(value)
            elif conversion == "r":
                value = repr(value)
            elif conversion == "a":
                value = ascii(value)
            if segment.format_spec:
                try:
                    formatted = format(value, segment.format_spec)
                except Exception:  # pragma: no cover - formatting errors
                    formatted = str(value)
            else:
                formatted = str(value)
            parts.append(formatted)
        return "".join(parts)


@dataclass(slots=True)
class CompiledRule:
    rule: DslRule
    condition: ast.Expression
    reasons: tuple[ReasonTemplate, ...]
    order: int


@dataclass(slots=True)
class DslEvaluationOutcome:
    severity: str
    rule_id: str
    priority: int
    reasons: list[str]
    diagnostics: dict[str, Any]


@dataclass(slots=True)
class DslEvaluationInput:
    rating: Mapping[str, Any]
    metrics: Mapping[str, Any]
    tag_scores: Mapping[str, float]
    group_patterns: Mapping[str, Sequence[str]]
    nude_flags: Sequence[str]
    is_nsfw_channel: bool
    is_spoiler: bool
    attachment_count: int


class DslProgram:
    def __init__(
        self,
        *,
        features: Sequence[CompiledFeature],
        rules: Sequence[CompiledRule],
        groups: Mapping[str, Sequence[str]],
        policy: DslPolicy,
        const_map: Mapping[str, float] | None = None,
    ) -> None:
        self._features = tuple(features)
        self._rules = tuple(rules)
        self._groups = dict(groups)
        self._policy = policy
        self._const_map = dict(const_map or {})

    @classmethod
    def from_config(
        cls,
        config: RuleConfigV2 | None,
        policy: DslPolicy,
        *,
        const_map: Mapping[str, float] | None = None,
    ) -> "DslProgram":
        if config is None:
            raise ValueError("config must not be None")

        compiled_features: list[CompiledFeature] = []
        seen_features: set[str] = set()
        for name, expression in config.features.items():
            canonical = normalize_tag(name)
            if not canonical:
                policy.warn_once(f"DSL feature '{name}' ignored because name is empty after normalization", key=f"feature:{name}:name")
                continue
            if canonical in seen_features:
                policy.warn_once(f"DSL feature '{name}' ignored because normalized name duplicates another feature", key=f"feature:{canonical}:dup")
                continue
            try:
                compiled = compile_expr(expression)
            except Exception as exc:
                if policy.strict:
                    raise
                policy.warn_once(f"DSL feature '{name}' skipped due to compilation error: {exc}", key=f"feature:{name}:compile")
                continue
            compiled_features.append(CompiledFeature(name=canonical, expression=compiled))
            seen_features.add(canonical)

        compiled_rules: list[CompiledRule] = []
        for order, rule in enumerate(config.rules):
            try:
                compiled_condition = compile_expr(rule.when)
            except Exception as exc:
                if policy.strict:
                    raise
                policy.warn_once(f"DSL rule '{rule.id}' skipped due to compilation error: {exc}", key=f"rule:{rule.id}:compile")
                continue
            reason_templates = tuple(
                _compile_reason(template)
                for template in rule.reasons
            )
            compiled_rules.append(CompiledRule(rule=rule, condition=compiled_condition, reasons=reason_templates, order=order))

        if not compiled_rules:
            raise ValueError("no valid DSL rules were compiled")

        return cls(
            features=compiled_features,
            rules=compiled_rules,
            groups=config.groups,
            policy=policy,
            const_map=const_map,
        )

    def evaluate(self, inputs: DslEvaluationInput) -> DslEvaluationOutcome | None:
        runtime = build_context(
            rating=inputs.rating,
            metrics=inputs.metrics,
            tag_scores=inputs.tag_scores,
            group_patterns=self._groups,
            nude_flags=inputs.nude_flags,
            is_nsfw_channel=inputs.is_nsfw_channel,
            is_spoiler=inputs.is_spoiler,
            attachment_count=inputs.attachment_count,
        )
        namespace = runtime.namespace
        for key, value in self._const_map.items():
            namespace[key] = float(value)
        evaluator = SafeEvaluator(namespace, policy=self._policy)

        feature_values: dict[str, Any] = {}
        for feature in self._features:
            with evaluator.context(f"feature:{feature.name}"):
                try:
                    value = evaluator.evaluate(feature.expression)
                except DslRuntimeError as exc:
                    if evaluator.strict:
                        raise
                    evaluator.warn(f"feature evaluation failed: {exc}", key_suffix="eval")
                    value = 0.0
            namespace[feature.name] = value
            feature_values[feature.name] = value

        hits: list[tuple[CompiledRule, list[str]]] = []
        for compiled_rule in self._rules:
            rule = compiled_rule.rule
            with evaluator.context(f"rule:{rule.id}:when"):
                try:
                    matched = bool(evaluator.evaluate(compiled_rule.condition))
                except DslRuntimeError as exc:
                    if evaluator.strict:
                        raise
                    evaluator.warn(f"rule condition evaluation failed: {exc}", key_suffix="condition")
                    matched = False
            if not matched:
                continue
            reasons: list[str] = []
            if compiled_rule.reasons:
                for idx, template in enumerate(compiled_rule.reasons):
                    context = f"rule:{rule.id}:reason:{idx}"
                    with evaluator.context(context):
                        try:
                            rendered = template.render(evaluator, context)
                        except DslRuntimeError as exc:
                            if evaluator.strict:
                                raise
                            evaluator.warn(f"reason rendering failed: {exc}", key_suffix="render")
                            rendered = ""
                    if rendered:
                        reasons.append(rendered)
            if not reasons:
                reasons.append(rule.id)
            hits.append((compiled_rule, reasons))
            if rule.stop:
                break

        if not hits:
            return None

        def sort_key(item: tuple[CompiledRule, list[str]]) -> tuple[int, int, int]:
            compiled_rule, _ = item
            rule = compiled_rule.rule
            severity_rank = SEVERITY_ORDER.get(rule.severity, -1)
            return (-severity_rank, -(rule.priority or 0), compiled_rule.order)

        best_rule, best_reasons = sorted(hits, key=sort_key)[0]
        rule = best_rule.rule
        diagnostics = {
            "origin": "dsl",
            "features": feature_values,
        }
        return DslEvaluationOutcome(
            severity=rule.severity,
            rule_id=rule.id,
            priority=rule.priority,
            reasons=best_reasons,
            diagnostics=diagnostics,
        )

    @property
    def group_patterns(self) -> Mapping[str, Sequence[str]]:
        return self._groups

    @property
    def compiled_features(self) -> tuple[CompiledFeature, ...]:
        return self._features

    @property
    def compiled_rules(self) -> tuple[CompiledRule, ...]:
        return self._rules

    @property
    def const_map(self) -> Mapping[str, float]:
        return self._const_map


def _compile_reason(template: str) -> ReasonTemplate:
    segments: list[ReasonSegment] = []
    formatter = string.Formatter()
    for literal, field_name, format_spec, conversion in formatter.parse(template):
        if literal:
            segments.append(ReasonSegment(literal=literal, expression=None, format_spec=None, conversion=None))
        if field_name is None:
            continue
        expression = compile_expr(field_name)
        segments.append(
            ReasonSegment(
                literal="",
                expression=expression,
                format_spec=format_spec or None,
                conversion=conversion,
            )
        )
    if not segments:
        segments.append(ReasonSegment(literal=template, expression=None, format_spec=None, conversion=None))
    return ReasonTemplate(raw=template, segments=tuple(segments))
