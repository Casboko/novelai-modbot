import sys
import types

if "imagehash" not in sys.modules:  # pragma: no cover - test shim
    sys.modules["imagehash"] = types.SimpleNamespace(phash=lambda *_args, **_kwargs: None)

if "PIL" not in sys.modules:  # pragma: no cover - test shim
    pil_module = types.ModuleType("PIL")
    image_module = types.ModuleType("PIL.Image")
    image_module.Image = object  # placeholder attribute for type hints
    pil_module.Image = image_module
    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = image_module

from app.util import ellipsize_inline, summarize_exposure_axes


def test_summarize_exposure_axes_normalises_labels() -> None:
    detections = [
        {"class": "FEMALE_BREAST_EXPOSED", "score": 0.35},
        {"class": "EXPOSED_BREAST_M", "score": 0.82},
        {"class": "MALE_GENITALIA_EXPOSED", "score": 0.67},
        {"class": "EXPOSED_GENITALIA_F", "score": 0.41},
        {"class": "ANUS_EXPOSED", "score": 0.58},
        {"class": "FEMALE_GENITALIA_COVERED", "score": 0.95},  # excluded
    ]

    peaks = summarize_exposure_axes(detections)

    assert peaks == {
        "BREAST": 0.82,
        "FG": 0.41,
        "MG": 0.67,
        "AN": 0.58,
    }


def test_summarize_exposure_axes_ignores_invalid_entries() -> None:
    detections = [
        {"class": "", "score": 0.9},
        {"class": None, "score": 0.4},
        {"class": "EXPOSED_GENITALIA", "score": "not-a-float"},
        {"class": "EXPOSED_ANUS", "score": 0.2},
        {"class": "EXPOSED_GENITALIA_F", "score": 0.0},
    ]

    peaks = summarize_exposure_axes(detections)

    assert peaks["AN"] == 0.2
    assert peaks["FG"] == 0.0
    assert peaks["MG"] == 0.0
    assert peaks["BREAST"] == 0.0


def test_ellipsize_inline_applies_ellipsis() -> None:
    assert ellipsize_inline("a" * 1024) == "a" * 1024
    assert ellipsize_inline("a" * 1025) == ("a" * 1023) + "\u2026"
