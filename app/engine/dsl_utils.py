from __future__ import annotations

import ast
import string
from typing import List

from .dsl_errors import DslValidationError

__all__ = ["compile_expr", "validate_expr", "extract_braced_expressions"]


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
            if func.id not in {"score", "sum", "max", "min", "any", "count", "clamp"}:
                raise DslValidationError(f"function '{func.id}' is not allowed")
        elif isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Attribute):
                raise DslValidationError("nested attribute call is not allowed")
            if not isinstance(func.value, ast.Name):
                raise DslValidationError("unsupported call expression")
            pair = (func.value.id, func.attr)
            if pair not in {("nude", "has"), ("nude", "any")}:
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
        if not isinstance(node.value, ast.Name):
            raise DslValidationError("unsupported attribute base")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__"):
            raise DslValidationError("identifier with double underscore is not allowed")

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: D401
        if isinstance(node.value, complex):
            raise DslValidationError("complex numbers are not supported")

    def generic_visit(self, node: ast.AST) -> None:  # noqa: D401
        raise DslValidationError(f"unsupported syntax: {type(node).__name__}")


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


def compile_expr(source: str) -> ast.Expression:
    processed = _preprocess_expression(source)
    try:
        node = ast.parse(processed, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - syntax errors should be rare
        raise DslValidationError(f"invalid expression: {exc}") from exc
    _ExpressionValidator().visit(node)
    return node


def validate_expr(source: str) -> None:
    compile_expr(source)


def extract_braced_expressions(template: str) -> List[str]:
    expressions: list[str] = []
    formatter = string.Formatter()
    for _, field_name, _, _ in formatter.parse(template):
        if field_name is not None and field_name.strip():
            expressions.append(field_name)
    return expressions
