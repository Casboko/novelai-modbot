from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import onnxruntime as ort
from PIL import Image

from .labelspace import LabelSpace


@dataclass(slots=True)
class WD14Prediction:
    rating: dict[str, float]
    general: list[tuple[str, float]]
    general_raw: list[tuple[str, float]]
    character: list[tuple[str, float]]
    raw_scores: np.ndarray


class WD14Session:
    def __init__(
        self,
        model_path: str,
        *,
        provider: str = "cpu",
        threads: int | None = None,
    ) -> None:
        self.model_path = model_path
        sess_options = ort.SessionOptions()
        if threads and threads > 0:
            sess_options.intra_op_num_threads = threads
            sess_options.inter_op_num_threads = max(1, threads // 2)
        provider = provider.lower()
        providers = ["CPUExecutionProvider"]
        if provider == "openvino":
            providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        shape = input_meta.shape
        if len(shape) != 4:
            raise ValueError(f"Unexpected input rank {shape}")
        batch, dim1, dim2, dim3 = shape
        if dim1 == 3:
            self.layout = "NCHW"
            height = dim2
            width = dim3
        elif dim3 == 3:
            self.layout = "NHWC"
            height = dim1
            width = dim2
        else:
            # Fallback: assume channels-first if height inferred from dim2
            self.layout = "NCHW"
            height = dim2
            width = dim3
        if height is None or width is None:
            raise ValueError("Model input shape must define height and width")
        self.size = int(height)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        if image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and image.info.get("transparency") is not None
        ):
            rgba_image = image.convert("RGBA")
            background = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
            rgb_image = Image.alpha_composite(background, rgba_image).convert("RGB")
        else:
            rgb_image = image.convert("RGB")

        width, height = rgb_image.size
        side = max(width, height)
        letterboxed = Image.new("RGB", (side, side), (255, 255, 255))
        letterboxed.paste(
            rgb_image,
            ((side - width) // 2, (side - height) // 2),
        )
        resized = letterboxed.resize((self.size, self.size), Image.BICUBIC)

        arr = np.asarray(resized, dtype=np.float32)
        # Ensure BGR channel order
        arr = arr[:, :, ::-1].copy()
        if self.layout == "NCHW":
            arr = np.transpose(arr, (2, 0, 1))
        return np.ascontiguousarray(arr, dtype=np.float32)

    def infer(self, batch: Sequence[Image.Image]) -> np.ndarray:
        if not batch:
            return np.empty((0, 0), dtype=np.float32)
        processed = np.stack([self.preprocess(image) for image in batch], axis=0)
        outputs = self.session.run(None, {self.input_name: processed})
        return outputs[0]


class WD14Analyzer:
    def __init__(
        self,
        session: WD14Session,
        labelspace: LabelSpace,
        *,
        general_threshold: float = 0.35,
        character_threshold: float = 0.85,
        raw_general_topk: int = 64,
        raw_general_whitelist: Iterable[str] | None = None,
    ) -> None:
        self.session = session
        self.labelspace = labelspace
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.raw_general_topk = max(1, int(raw_general_topk))
        self.raw_general_whitelist = {
            tag for tag in (raw_general_whitelist or []) if isinstance(tag, str)
        }
        self._general_index_names = [
            (idx, self.labelspace.names[idx]) for idx in self.labelspace.general_indices
        ]

    def predict(self, images: Sequence[Image.Image]) -> list[WD14Prediction]:
        scores = self.session.infer(images)
        results: list[WD14Prediction] = []
        for row in scores:
            rating = _collect_scores(
                row,
                self.labelspace.rating_indices,
                self.labelspace.names,
            )
            general = _filter_scores(
                row,
                self.labelspace.general_indices,
                self.labelspace.names,
                self.general_threshold,
            )
            character = _filter_scores(
                row,
                self.labelspace.character_indices,
                self.labelspace.names,
                self.character_threshold,
            )
            general_raw = self._build_general_raw(row)
            results.append(
                WD14Prediction(
                    rating=rating,
                    general=general,
                    general_raw=general_raw,
                    character=character,
                    raw_scores=row,
                )
            )
        return results

    def _build_general_raw(self, row: np.ndarray) -> list[tuple[str, float]]:
        if not self._general_index_names:
            return []
        pairs = [(name, float(row[idx])) for idx, name in self._general_index_names]
        sorted_pairs = sorted(pairs, key=lambda item: item[1], reverse=True)
        keep = set(self.raw_general_whitelist)
        keep.update(name for name, _ in sorted_pairs[: self.raw_general_topk])
        filtered = [(name, score) for name, score in pairs if name in keep]
        filtered.sort(key=lambda item: item[1], reverse=True)
        return filtered

    def general_raw_from_scores(self, scores: Sequence[float]) -> list[tuple[str, float]]:
        if not scores:
            return []
        arr = np.asarray(scores, dtype=np.float32)
        return self._build_general_raw(arr)


def _collect_scores(
    row: np.ndarray,
    indices: Sequence[int],
    names: Sequence[str],
) -> dict[str, float]:
    return {names[i]: float(row[i]) for i in indices}


def _filter_scores(
    row: np.ndarray,
    indices: Sequence[int],
    names: Sequence[str],
    threshold: float,
) -> list[tuple[str, float]]:
    tagged: list[tuple[str, float]] = []
    for i in indices:
        score = float(row[i])
        if score >= threshold:
            tagged.append((names[i], score))
    tagged.sort(key=lambda item: item[1], reverse=True)
    return tagged
