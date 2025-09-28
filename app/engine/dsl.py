from __future__ import annotations

import ast
import string
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .dsl_errors import DslRuntimeError, DslValidationError
from .dsl_runtime import build_context
from .tag_norm import normalize_tag

SEVERITY_ORDER = {"green": 0, "yellow": 1, "orange": 2, "red": 3}
ALLOWED_CALL_NAMES = {"score", "sum", "max", "min", "any", "count", "clamp"}
ALLOWED_ATTR_CALLS = {("nude", "has"), ("nude", "any")}


def _preprocess_expression(source: str) -> str:
    if not isinstance(source, str):
        raise DslValidationError("expression must be a string")
    text = source.strip()
    if not text:
        return "0"
    text = text.replace("&&", " and ").replace("||", " or ")
    text = text.replace("!=", "__dsl_ne__")
    text = text.replace("!", " not ")
    text = text.replace("__dsl_ne__", "!=")
    return text


class _ExpressionValidator(ast.NodeVisitor):
    _allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div)
    _allowed_boolops = (ast.And, ast.Or)
    _allowed_unary = (ast.Not, ast.USub, ast.UAdd)
    _allowed_compops = (
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    )

    def visit_Expression(self, node: ast.Expression) -> None:  # noqa: D401
        self.visit(node.body)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        if not isinstance(node.op, self._allowed_boolops):
            raise DslValidationError("unsupported boolean operator")
        for value in node.values:
            self.visit(value)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, self._allowed_binops):
            raise DslValidationError("unsupported arithmetic operator")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, self._allowed_unary):
            raise DslValidationError("unsupported unary operator")
        self.visit(node.operand)

    def visit_Compare(self, node: ast.Compare) -> None:
        if not all(isinstance(op, self._allowed_compops) for op in node.ops):
            raise DslValidationError("unsupported comparison operator")
        self.visit(node.left)
        for comparator in node.comparators:
            self.visit(comparator)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id not in ALLOWED_CALL_NAMES:
                raise DslValidationError(f"function '{func.id}' is not allowed")
        elif isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Attribute):
                raise DslValidationError("nested attribute call is not allowed")
            if not isinstance(func.value, ast.Name):
                raise DslValidationError("unsupported call expression")
            pair = (func.value.id, func.attr)
            if pair not in ALLOWED_ATTR_CALLS:
                raise DslValidationError(f"method '{func.value.id}.{func.attr}' is not allowed")
        else:
            raise DslValidationError("unsupported call expression")
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Attribute):
            raise DslValidationError("nested attribute access is not allowed")
        if not isinstance(node.value, (ast.Name, ast.Subscript)):
            raise DslValidationError("unsupported attribute base")
        if isinstance(node.value, ast.Subscript):
            raise DslValidationError("subscript access is not allowed")
        self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            raise DslValidationError("identifier with double underscore is not allowed")

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: D401
        if isinstance(node.value, complex):
            raise DslValidationError("complex numbers are not supported")

    def generic_visit(self, node: ast.AST) -> None:  # noqa: D401
        raise DslValidationError(f"unsupported syntax: {type(node).__name__}")


def compile_expression(source: str) -> ast.Expression:
    processed = _preprocess_expression(source)
    try:
        node = ast.parse(processed, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - syntax errors should be rare
        raise DslValidationError(f"invalid expression: {exc}") from exc
    _ExpressionValidator().visit(node)
    return node


class SafeEvaluator(ast.NodeVisitor):
    def __init__(self, namespace: Mapping[str, Any], *, strict: bool = False) -> None:
        self._namespace = namespace
        self._strict = strict

    def evaluate(self, node: ast.AST) -> Any:
        return self.visit(node)

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
            except ZeroDivisionError:
                if self._strict:
                    raise
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
            else:  # pragma: no cover - guarded by validator
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
            if self._strict:
                raise DslRuntimeError("attempted to call a non-callable object")
            return 0.0
        try:
            return func_obj(*args, **kwargs)
        except ZeroDivisionError:
            if self._strict:
                raise
            return 0.0
        except Exception as exc:  # pragma: no cover - defensive fallback
            if self._strict:
                raise DslRuntimeError(str(exc)) from exc
            return 0.0

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self._namespace:
            return self._namespace[node.id]
        if self._strict:
            raise DslRuntimeError(f"unknown identifier: {node.id}")
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
        if self._strict:
            raise DslRuntimeError(f"attribute '{attr}' is not available")
        return 0.0

    def visit_Constant(self, node: ast.Constant) -> Any:  # noqa: D401
        return node.value

    @property
    def strict(self) -> bool:
        return self._strict


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

    def render(self, evaluator: SafeEvaluator, *, strict: bool) -> str:
        parts: list[str] = []
        for segment in self.segments:
            if segment.expression is None:
                parts.append(segment.literal)
                continue
            try:
                value = evaluator.evaluate(segment.expression)
            except DslRuntimeError:
                if strict:
                    raise
                value = ""
            if segment.conversion == "s":
                value = str(value)
            elif segment.conversion == "r":
                value = repr(value)
            elif segment.conversion == "a":
                value = ascii(value)
            if segment.format_spec:
                try:
                    formatted = format(value, segment.format_spec)
                except Exception:  # pragma: no cover
                    formatted = str(value)
            else:
                formatted = str(value)
            parts.append(formatted)
        return "".join(parts)


@dataclass(slots=True)
class CompiledRule:
    rule_id: str
    severity: str
    priority: int
    order: int
    condition: ast.Expression
    reasons: tuple[ReasonTemplate, ...]


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
        group_patterns: Mapping[str, Sequence[str]],
        strict: bool = False,
    ) -> None:
        self._features = tuple(features)
        self._rules = tuple(rules)
        self._group_patterns = {key: tuple(values) for key, values in group_patterns.items()}
        self._strict = strict

    @classmethod
    def from_config(
        cls,
        *,
        groups: Mapping[str, Sequence[str]] | None,
        features: Mapping[str, str] | None,
        rules: Sequence[Mapping[str, Any]] | None,
        strict: bool = False,
    ) -> DslProgram:
        group_patterns: dict[str, tuple[str, ...]] = {}
        if groups:
            for name, values in groups.items():
                canonical = normalize_tag(name)
                if not canonical:
                    continue
                normalized_values = tuple(
                    normalize_tag(value)
                    for value in values
                    if isinstance(value, str) and value.strip()
                )
                group_patterns[canonical] = normalized_values

        compiled_features: list[CompiledFeature] = []
        if features:
            for name, expression in features.items():
                feature_name = normalize_tag(name)
                compiled_features.append(
                    CompiledFeature(name=feature_name, expression=compile_expression(expression))
                )

        compiled_rules: list[CompiledRule] = []
        if rules:
            for index, rule in enumerate(rules):
                rule_id = str(rule.get("id"))
                severity = normalize_tag(rule.get("severity", ""))
                if severity not in SEVERITY_ORDER:
                    raise DslValidationError(f"unsupported severity '{severity}' in DSL rule {rule_id}")
                priority = int(rule.get("priority", 100))
                condition_expr = compile_expression(str(rule.get("when", "0")))
                reasons_raw = rule.get("reasons", []) or []
                reason_templates = tuple(_compile_reason(str(text)) for text in reasons_raw)
                compiled_rules.append(
                    CompiledRule(
                        rule_id=rule_id or f"dsl-rule-{index}",
                        severity=severity,
                        priority=priority,
                        order=index,
                        condition=condition_expr,
                        reasons=reason_templates,
                    )
                )

        return cls(
            features=compiled_features,
            rules=compiled_rules,
            group_patterns=group_patterns,
            strict=strict,
        )

    def evaluate(self, inputs: DslEvaluationInput) -> DslEvaluationOutcome | None:
        runtime = build_context(
            rating=inputs.rating,
            metrics=inputs.metrics,
            tag_scores=inputs.tag_scores,
            group_patterns=self._group_patterns,
            nude_flags=inputs.nude_flags,
            is_nsfw_channel=inputs.is_nsfw_channel,
            is_spoiler=inputs.is_spoiler,
            attachment_count=inputs.attachment_count,
        )
        namespace = runtime.namespace
        evaluator = SafeEvaluator(namespace, strict=self._strict)

        feature_values: dict[str, Any] = {}
        for feature in self._features:
            try:
                value = evaluator.evaluate(feature.expression)
            except DslRuntimeError:
                if self._strict:
                    raise
                value = 0.0
            namespace[feature.name] = value
            feature_values[feature.name] = value

        hits: list[tuple[CompiledRule, list[str]]] = []
        for rule in self._rules:
            try:
                matched = bool(evaluator.evaluate(rule.condition))
            except DslRuntimeError:
                if self._strict:
                    raise
                matched = False
            if not matched:
                continue
            reasons: list[str] = []
            for template in rule.reasons:
                rendered = template.render(evaluator, strict=self._strict)
                if rendered:
                    reasons.append(rendered)
            if not reasons:
                reasons.append(rule.rule_id)
            hits.append((rule, reasons))

        if not hits:
            return None

        def sort_key(item: tuple[CompiledRule, list[str]]) -> tuple[int, int, int]:
            rule, _ = item
            severity_rank = SEVERITY_ORDER.get(rule.severity, -1)
            return (-severity_rank, rule.priority, rule.order)

        best_rule, best_reasons = sorted(hits, key=sort_key)[0]
        diagnostics = {
            "origin": "dsl",
            "features": feature_values,
        }
        return DslEvaluationOutcome(
            severity=best_rule.severity,
            rule_id=best_rule.rule_id,
            priority=best_rule.priority,
            reasons=best_reasons,
            diagnostics=diagnostics,
        )

    @property
    def group_patterns(self) -> Mapping[str, Sequence[str]]:
        return self._group_patterns


def _compile_reason(template: str) -> ReasonTemplate:
    formatter = string.Formatter()
    segments: list[ReasonSegment] = []
    for literal, field_name, format_spec, conversion in formatter.parse(template):
        if literal:
            segments.append(ReasonSegment(literal=literal, expression=None, format_spec=None, conversion=None))
        if field_name is None:
            continue
        expression = compile_expression(field_name)
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
