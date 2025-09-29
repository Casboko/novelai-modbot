from __future__ import annotations

import fnmatch
from typing import Iterable, Mapping

from .tag_norm import normalize_tag

_WILDCARD_CHARS = set("*?[")


def group_top_tags(
    tag_scores: Mapping[str, float],
    patterns: Iterable[str],
    *,
    k: int = 5,
    min_score: float = 0.0,
) -> list[tuple[str, float]]:
    """Return top-scoring tags within the provided group patterns.

    Args:
        tag_scores: Mapping of normalised tag -> score.
        patterns: Iterable of group patterns (normalised, may include wildcards).
        k: Maximum number of tags to return.
        min_score: Minimum score threshold for inclusion.

    Returns:
        A list of ``(tag, score)`` tuples sorted by descending score.
    """

    threshold = float(min_score)
    normalised_scores = {
        normalize_tag(tag): float(score)
        for tag, score in tag_scores.items()
        if tag
    }

    hits: dict[str, float] = {}
    for pattern in patterns:
        canonical_pattern = normalize_tag(pattern)
        if not canonical_pattern:
            continue
        if _has_wildcard(canonical_pattern):
            for tag, score in normalised_scores.items():
                if score < threshold:
                    continue
                if fnmatch.fnmatchcase(tag, canonical_pattern):
                    hits[tag] = max(hits.get(tag, 0.0), score)
        else:
            score = normalised_scores.get(canonical_pattern)
            if score is not None and score >= threshold:
                hits[canonical_pattern] = max(hits.get(canonical_pattern, 0.0), score)

    ordered = sorted(hits.items(), key=lambda item: (-item[1], item[0]))
    return ordered[: max(0, int(k))]


def _has_wildcard(pattern: str) -> bool:
    return any(char in _WILDCARD_CHARS for char in pattern)
