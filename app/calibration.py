from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable

EPS = 1e-6


def _logit(p: float) -> float:
    value = min(max(p, EPS), 1.0 - EPS)
    return math.log(value / (1.0 - value))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def apply_temperature(p: float, temperature: float) -> float:
    if temperature <= 0:
        temperature = EPS
    return _sigmoid(_logit(p) / max(temperature, EPS))


def load_calibration(path: Path) -> Dict:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def _coerce_pair(item: Iterable) -> tuple[str, float] | None:
    if isinstance(item, (list, tuple)) and len(item) == 2:
        name, score = item
        return str(name), float(score)
    if isinstance(item, dict):
        name = item.get("name")
        score = item.get("score")
        if name is None or score is None:
            return None
        return str(name), float(score)
    return None


def apply_wd14_calibration(wd14: dict, calib: Dict) -> dict:
    if not calib:
        return wd14

    rating = wd14.get("rating")
    if isinstance(rating, dict):
        adjusted_rating = {}
        rating_calib = calib.get("rating", {})
        for key, value in rating.items():
            T = float(rating_calib.get(key, 1.0))
            adjusted_rating[key] = apply_temperature(float(value), T)
        wd14["rating"] = adjusted_rating

    general_calib = calib.get("general_tags", {})
    for field in ("general_raw", "general"):
        items = wd14.get(field)
        if not items:
            continue
        calibrated: list[tuple[str, float]] = []
        for item in items:
            pair = _coerce_pair(item)
            if pair is None:
                continue
            name, score = pair
            T = float(general_calib.get(name, 1.0))
            calibrated.append((name, apply_temperature(score, T)))
        wd14[field] = calibrated

    return wd14
