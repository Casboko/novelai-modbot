from __future__ import annotations

from app.engine.group_utils import group_top_tags


def test_group_top_tags_filters_and_sorts() -> None:
    tag_scores = {
        "blood": 0.4,
        "gore": 0.2,
        "cat": 0.5,
        "dog": 0.3,
    }
    patterns = ("cat", "dog", "blood", "gore")

    hits = group_top_tags(tag_scores, patterns, k=3, min_score=0.25)

    assert hits == [("cat", 0.5), ("blood", 0.4), ("dog", 0.3)]


def test_group_top_tags_supports_wildcards() -> None:
    tag_scores = {
        "animal_wolf": 0.4,
        "animal_fox": 0.3,
        "animal_bird": 0.1,
        "flower": 0.9,
    }
    patterns = ("animal_*",)

    hits = group_top_tags(tag_scores, patterns, k=5, min_score=0.2)

    assert hits == [("animal_wolf", 0.4), ("animal_fox", 0.3)]
