from __future__ import annotations

import pytest

from app.engine.tag_norm import format_tag_for_display, normalize_pair, normalize_tag


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("See Through", "see_through"),
        ("blood-loss", "blood_loss"),
        ("  A__B  ", "a_b"),
        ("", ""),
    ],
)
def test_normalize_tag(value: str, expected: str) -> None:
    assert normalize_tag(value) == expected


@pytest.mark.parametrize(
    "item",
    [
        ("see through", 0.8),
        ("see_through", 0.8),
        {"name": "see through", "score": 0.8},
    ],
)
def test_normalize_pair_success(item: object) -> None:
    result = normalize_pair(item)
    assert result == ("see_through", pytest.approx(0.8))


@pytest.mark.parametrize(
    "item",
    [
        ("tag",),
        {"name": None, "score": 0.1},
        {"name": "", "score": "nan"},
        42,
    ],
)
def test_normalize_pair_invalid(item: object) -> None:
    assert normalize_pair(item) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("see_through", "see through"),
        ("blood_loss", "blood loss"),
        ("", ""),
    ],
)
def test_format_tag_for_display(value: str, expected: str) -> None:
    assert format_tag_for_display(value) == expected
