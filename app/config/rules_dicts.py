from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from app.engine.tag_norm import normalize_tag

__all__ = ["RulesDictError", "extract_nsfw_general_tags"]


class RulesDictError(ValueError):
    """Raised when rule dictionaries fail validation."""


def _coerce_iterable(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _normalize_tags(items: Iterable[object]) -> set[str]:
    result: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        canonical = normalize_tag(item)
        if canonical:
            result.add(canonical)
    return result


def extract_nsfw_general_tags(
    cfg: Mapping[str, object] | None,
    *,
    strict: bool = True,
) -> list[str]:
    """Extract the canonical NSFW general tag list from ``rules.yaml``.

    The primary source of truth is ``groups.nsfw_general``. A future-compatible
    fallback ``nsfw_general_tags`` may also be present. When both are provided,
    their normalized contents must match unless ``strict`` is set to ``False``.
    """

    if not isinstance(cfg, Mapping):
        return []

    group_tags_raw: Sequence[object] = ()
    groups_raw = cfg.get("groups")
    if isinstance(groups_raw, Mapping):
        group_tags_raw = _coerce_iterable(groups_raw.get("nsfw_general"))
    group_tags = _normalize_tags(group_tags_raw)

    top_level_raw = _coerce_iterable(cfg.get("nsfw_general_tags"))
    top_level_tags = _normalize_tags(top_level_raw)

    if top_level_tags and not group_tags:
        return sorted(top_level_tags)
    if group_tags and not top_level_tags:
        return sorted(group_tags)
    if group_tags and top_level_tags:
        if group_tags != top_level_tags:
            if strict:
                raise RulesDictError(
                    "nsfw_general_tags mismatch between groups.nsfw_general and top-level nsfw_general_tags",
                )
        return sorted(group_tags)
    return []

