from __future__ import annotations

import pytest

from app.config.rules_dicts import RulesDictError, extract_nsfw_general_tags


def test_extract_from_groups_only() -> None:
    cfg = {"groups": {"nsfw_general": ["see_through", "Bikini", " see through "]}}
    assert extract_nsfw_general_tags(cfg) == ["bikini", "see_through"]


def test_extract_from_top_level_only() -> None:
    cfg = {"nsfw_general_tags": ["Topless", " lingerie "]}
    assert extract_nsfw_general_tags(cfg) == ["lingerie", "topless"]


def test_extract_matching_sources() -> None:
    cfg = {
        "groups": {"nsfw_general": ["topless", "lingerie"]},
        "nsfw_general_tags": ["LINGERIE", "topless"],
    }
    assert extract_nsfw_general_tags(cfg) == ["lingerie", "topless"]


def test_extract_mismatch_raises() -> None:
    cfg = {
        "groups": {"nsfw_general": ["topless"]},
        "nsfw_general_tags": ["lingerie"],
    }
    with pytest.raises(RulesDictError):
        extract_nsfw_general_tags(cfg)


def test_non_mapping_returns_empty() -> None:
    assert extract_nsfw_general_tags(None) == []
