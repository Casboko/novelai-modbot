from __future__ import annotations

import fnmatch
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

_WHITESPACE_RE = re.compile(r"\s+")
_UNDERSCORE_RE = re.compile(r"_+")
_WILDCARD_CHARS = set("*?[")

__all__ = [
    "normalize_tag",
    "normalize_pair",
    "format_tag_for_display",
    "expand_groups",
]


def normalize_tag(value: str) -> str:
    """Normalize a tag/group token for consistent matching.

    Steps:
        1. Unicode NFKC normalisation
        2. Trim leading/trailing whitespace
        3. Lowercase
        4. Collapse internal whitespace to a single space
        5. Replace spaces and hyphens with underscores
        6. Collapse sequential underscores
    """

    text = unicodedata.normalize("NFKC", str(value))
    text = text.strip().lower()
    if not text:
        return ""
    text = _WHITESPACE_RE.sub(" ", text)
    text = text.replace("-", "_").replace(" ", "_")
    text = _UNDERSCORE_RE.sub("_", text)
    return text


def normalize_pair(item: object) -> tuple[str, float] | None:
    """Coerce an arbitrary item into a normalised ``(tag, score)`` pair."""

    name: Any
    score: Any
    if isinstance(item, (list, tuple)) and len(item) == 2:
        name, score = item
    elif isinstance(item, Mapping):
        name = item.get("name")
        score = item.get("score")
    else:
        return None

    canonical = normalize_tag(name) if name is not None else ""
    if not canonical:
        return None

    try:
        numeric = float(score)
    except (TypeError, ValueError):
        return None

    return canonical, numeric


def format_tag_for_display(value: str) -> str:
    """Return a human-friendly representation of a normalised tag."""

    text = str(value).strip()
    if not text:
        return ""
    return text.replace("_", " ")


def _has_wildcard(pattern: str) -> bool:
    return any(char in _WILDCARD_CHARS for char in pattern)


def expand_groups(
    groups: Mapping[str, Sequence[str]] | None,
    vocabulary: Iterable[str],
) -> dict[str, set[str]]:
    """Expand wildcard patterns against a vocabulary of tags.

    Args:
        groups: Mapping of group name -> iterable of tag patterns.
        vocabulary: Iterable of tag tokens to match against.

    Returns:
        Mapping of normalised group name to a set of normalised tag tokens.
    """

    result: dict[str, set[str]] = {}
    if not groups:
        return result

    normalized_vocab = {normalize_tag(token) for token in vocabulary}

    for group_name, patterns in groups.items():
        canonical_name = normalize_tag(group_name)
        if not canonical_name:
            continue
        bucket = result.setdefault(canonical_name, set())
        for raw_pattern in patterns or []:
            canonical_pattern = normalize_tag(raw_pattern)
            if not canonical_pattern:
                continue
            if _has_wildcard(canonical_pattern):
                matches = {
                    token for token in normalized_vocab if fnmatch.fnmatchcase(token, canonical_pattern)
                }
                if matches:
                    bucket.update(matches)
            else:
                if canonical_pattern in normalized_vocab:
                    bucket.add(canonical_pattern)
    return result
