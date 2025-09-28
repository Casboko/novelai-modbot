from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Severity = Literal["red", "orange", "yellow", "green"]
DslMode = Literal["strict", "warn"]


@dataclass(slots=True)
class DslRule:
    id: str
    severity: Severity
    when: str
    priority: int = 0
    reasons: list[str] = field(default_factory=list)
    stop: bool = False


@dataclass(slots=True)
class RuleConfigV2:
    version: int = 2
    groups: dict[str, list[str]] = field(default_factory=dict)
    features: dict[str, str] = field(default_factory=dict)
    rules: list[DslRule] = field(default_factory=list)


@dataclass(slots=True)
class LoadedConfig:
    rule_titles: dict[str, str]
    config: RuleConfigV2
    raw: dict[str, Any]


@dataclass(slots=True)
class ValidationIssue:
    level: Literal["error", "warning"]
    code: str
    where: str
    msg: str
    hint: str | None = None


@dataclass(slots=True)
class LoadResult:
    status: Literal["ok", "invalid", "error"]
    mode: DslMode
    config: LoadedConfig | None
    issues: list[ValidationIssue]
    counts: dict[str, int]
    summary: dict[str, Any] | None = None


@dataclass(slots=True)
class DslPolicy:
    mode: DslMode = "warn"
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("app.engine.dsl"))
    _warned_keys: set[str] = field(default_factory=set, init=False)

    @property
    def strict(self) -> bool:
        return self.mode == "strict"

    def warn_once(self, message: str, key: Optional[str] = None) -> None:
        if self.strict:
            return
        cache_key = key or message
        if cache_key in self._warned_keys:
            return
        self._warned_keys.add(cache_key)
        self.logger.warning(message)

    @classmethod
    def from_mode(cls, mode: str | None) -> "DslPolicy":
        normalized = (mode or "warn").strip().lower()
        if normalized not in {"strict", "warn"}:
            normalized = "warn"
        return cls(mode=normalized)  # type: ignore[arg-type]
