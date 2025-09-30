from __future__ import annotations

from app.analysis_merge import compute_nudity_exposure_metrics, DEFAULT_EXPOSED_LABELS
from app.schema import NudityDetection


def _make_detection(label: str, score: float, box: list[float]) -> NudityDetection:
    return NudityDetection(cls=label, score=score, box=box)


def test_compute_nudity_exposure_metrics_with_normalized_boxes() -> None:
    detections = [
        _make_detection("FEMALE_BREAST_EXPOSED", 0.9, [0.1, 0.1, 0.4, 0.5]),
        _make_detection("ANUS_EXPOSED", 0.8, [0.2, 0.2, 0.3, 0.4]),
    ]
    area, count = compute_nudity_exposure_metrics(
        detections,
        allowed_labels=set(DEFAULT_EXPOSED_LABELS),
        min_score=0.6,
        image_size=None,
    )
    expected_area = (0.3 * 0.4) + (0.1 * 0.2)
    assert area == expected_area
    assert count == 2


def test_compute_nudity_exposure_metrics_with_pixel_boxes_and_size() -> None:
    detections = [
        _make_detection("MALE_GENITALIA_EXPOSED", 0.7, [10.0, 20.0, 90.0, 70.0]),
    ]
    area, count = compute_nudity_exposure_metrics(
        detections,
        allowed_labels=set(DEFAULT_EXPOSED_LABELS),
        min_score=0.6,
        image_size=(200, 100),
    )
    # Pixel area (80 * 50) / (200 * 100) = 0.2
    assert area == 0.2
    assert count == 1


def test_compute_nudity_exposure_metrics_ignores_below_threshold_and_unlisted_labels() -> None:
    detections = [
        _make_detection("FEMALE_BREAST_EXPOSED", 0.55, [0.0, 0.0, 0.4, 0.6]),
        _make_detection("BELLY_EXPOSED", 0.95, [0.1, 0.1, 0.5, 0.4]),
    ]
    area, count = compute_nudity_exposure_metrics(
        detections,
        allowed_labels=set(DEFAULT_EXPOSED_LABELS),
        min_score=0.6,
        image_size=None,
    )
    assert area == 0.0
    assert count == 0


def test_compute_nudity_exposure_metrics_fallbacks_without_size() -> None:
    detections = [
        _make_detection("ANUS_EXPOSED", 0.9, [10.0, 10.0, 110.0, 60.0]),
    ]
    area, count = compute_nudity_exposure_metrics(
        detections,
        allowed_labels=set(DEFAULT_EXPOSED_LABELS),
        min_score=0.6,
        image_size=None,
    )
    # Pixel coordinates without image size fallback to skip contribution
    assert area == 0.0
    assert count == 0
