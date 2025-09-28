from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from .tag_norm import normalize_tag

_WILDCARD_CHARS = set("*?[")

_BUILTIN_FUNCTIONS = {"score", "sum", "max", "min", "any", "count", "clamp"}
_BASE_IDENTIFIERS = {
    "rating",
    "channel",
    "message",
    "attachment_count",
    "nude",
    "exposure_score",
    "placement_risk_pre",
    "exposure_peak",
    "minors_peak",
    "gore_peak",
    "nsfw_margin",
    "nsfw_ratio",
    "nsfw_general_sum",
    "violence_sum",
    "violence_max",
    "animals_peak",
    "animal_context_peak",
    "sexual_explicit_sum",
    "sexual_modifier_sum",
    "dismember_peak",
    "gore_sum",
    "drug_score",
}


def clamp(value: float, lo: float, hi: float) -> float:
    lo = float(lo)
    hi = float(hi)
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, min(float(value), hi))


def _has_wildcard(pattern: str) -> bool:
    return any(char in _WILDCARD_CHARS for char in pattern)


class NudeAccessor:
    __slots__ = ("_flags",)

    def __init__(self, flags: Iterable[str]) -> None:
        self._flags = frozenset(str(flag).upper() for flag in flags if flag)

    def has(self, flag: str) -> bool:
        return str(flag).upper() in self._flags

    def any(self, *, prefix: str | None = None, min: int = 1) -> bool:
        if not self._flags:
            return False
        if prefix is None:
            return len(self._flags) >= int(min)
        prefix_upper = str(prefix).upper()
        hits = sum(1 for value in self._flags if value.startswith(prefix_upper))
        return hits >= int(min)


class _GroupResolver:
    __slots__ = ("_patterns", "_tag_scores", "_cache")

    def __init__(
        self,
        patterns: Mapping[str, Sequence[str]],
        tag_scores: Mapping[str, float],
    ) -> None:
        self._patterns = {key: tuple(values) for key, values in patterns.items()}
        self._tag_scores = tag_scores
        self._cache: dict[str, tuple[str, ...]] = {}

    def _resolve(self, name: str) -> tuple[str, ...]:
        canonical = normalize_tag(name)
        if not canonical:
            return ()
        cached = self._cache.get(canonical)
        if cached is not None:
            return cached
        patterns = self._patterns.get(canonical, ())
        if not patterns:
            self._cache[canonical] = ()
            return ()
        matches: set[str] = set()
        tag_keys = tuple(self._tag_scores.keys())
        for pattern in patterns:
            if not pattern:
                continue
            if _has_wildcard(pattern):
                for tag in tag_keys:
                    if fnmatch.fnmatchcase(tag, pattern):
                        matches.add(tag)
            else:
                matches.add(pattern)
        self._cache[canonical] = tuple(sorted(matches))
        return self._cache[canonical]

    def tags(self, name: str) -> tuple[str, ...]:
        return self._resolve(name)

    def sum(self, name: str) -> float:
        tags = self._resolve(name)
        return float(sum(self._tag_scores.get(tag, 0.0) for tag in tags))

    def max(self, name: str) -> float:
        tags = self._resolve(name)
        if not tags:
            return 0.0
        return float(max(self._tag_scores.get(tag, 0.0) for tag in tags))

    def any(self, name: str, *, gt: float = 0.35) -> bool:
        threshold = float(gt)
        for tag in self._resolve(name):
            if float(self._tag_scores.get(tag, 0.0)) >= threshold:
                return True
        return False

    def count(self, name: str, *, gt: float = 0.35) -> int:
        threshold = float(gt)
        hits = 0
        for tag in self._resolve(name):
            if float(self._tag_scores.get(tag, 0.0)) >= threshold:
                hits += 1
        return hits


@dataclass(slots=True)
class RuntimeContext:
    namespace: dict[str, Any]
    resolver: _GroupResolver


def build_context(
    *,
    rating: Mapping[str, Any],
    metrics: Mapping[str, Any],
    tag_scores: Mapping[str, float],
    group_patterns: Mapping[str, Sequence[str]],
    nude_flags: Iterable[str],
    is_nsfw_channel: bool,
    is_spoiler: bool,
    attachment_count: int,
) -> RuntimeContext:
    """Assemble the base namespace for DSL evaluation."""

    resolver = _GroupResolver(group_patterns, tag_scores)

    def score_func(tag: str) -> float:
        return float(tag_scores.get(normalize_tag(tag), 0.0))

    def sum_func(value: Any, *rest: Any) -> float:
        if rest:
            numbers = (float(value), *(float(item) for item in rest))
            return float(sum(numbers))
        return resolver.sum(str(value))

    def max_func(*values: Any) -> float:
        if len(values) == 1 and isinstance(values[0], str):
            return float(resolver.max(values[0]))
        return float(max(float(item) for item in values)) if values else 0.0

    def min_func(*values: Any) -> float:
        if len(values) == 1 and isinstance(values[0], str):
            tags = resolver.tags(values[0])
            if not tags:
                return 0.0
            return float(min(tag_scores.get(tag, 0.0) for tag in tags))
        return float(min(float(item) for item in values)) if values else 0.0

    def any_func(group_name: str, *, gt: float = 0.35) -> bool:
        return resolver.any(group_name, gt=gt)

    def count_func(group_name: str, *, gt: float = 0.35) -> int:
        return resolver.count(group_name, gt=gt)

    namespace: dict[str, Any] = {}

    rating_map: dict[str, float] = {
        key: float(value) for key, value in rating.items() if isinstance(value, (int, float))
    }
    for key in ("explicit", "questionable", "general", "sensitive", "safe"):
        rating_map.setdefault(key, 0.0)
    namespace["rating"] = rating_map

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            namespace[normalize_tag(key)] = float(value)

    namespace.setdefault(
        "exposure_peak",
        float(metrics.get("exposure_peak", metrics.get("exposure_score", 0.0)))
    )
    namespace.setdefault("minors_peak", float(metrics.get("minors_peak", 0.0)))
    namespace.setdefault("gore_peak", float(metrics.get("gore_peak", 0.0)))

    namespace["channel"] = {"is_nsfw": bool(is_nsfw_channel)}
    namespace["message"] = {"is_spoiler": bool(is_spoiler)}
    namespace["attachment_count"] = int(attachment_count)
    namespace["nude"] = NudeAccessor(nude_flags)

    namespace["clamp"] = clamp
    namespace["score"] = score_func
    namespace["sum"] = sum_func
    namespace["max"] = max_func
    namespace["min"] = min_func
    namespace["any"] = any_func
    namespace["count"] = count_func

    return RuntimeContext(namespace=namespace, resolver=resolver)


def list_builtin_identifiers() -> set[str]:
    """Return identifiers that the runtime injects into DSL namespaces."""

    return set(_BASE_IDENTIFIERS)


def list_builtin_functions() -> set[str]:
    """Return callable names that are always available in DSL expressions."""

    return set(_BUILTIN_FUNCTIONS)
