from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image

import nudenet
from nudenet import NudeDetector


@dataclass(slots=True)
class NudeNetDetection:
    label: str
    score: float
    box: list[float] | None


class NudeNetAnalyzer:
    def __init__(self, model: str | None = None) -> None:
        self.detector = NudeDetector(model_name=model) if model else NudeDetector()
        self.version = getattr(nudenet, "__version__", "unknown")

    def detect_batch(self, images: Sequence[Image.Image]) -> list[list[NudeNetDetection]]:
        if not images:
            return []
        arrays = [self._to_array(image) for image in images]
        raw_results = self.detector.detect_batch(arrays)
        normalized: list[list[NudeNetDetection]] = []
        for detections in raw_results:
            group: list[NudeNetDetection] = []
            for det in detections or []:
                label = det.get("class") or det.get("label") or ""
                score = float(det.get("score", 0.0))
                box = det.get("box")
                if box is not None:
                    box = [float(x) for x in box]
                group.append(NudeNetDetection(label=label, score=score, box=box))
            normalized.append(group)
        return normalized

    @staticmethod
    def _to_array(image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
        # NudeNet expects BGR when passing arrays
        return arr[..., ::-1]
