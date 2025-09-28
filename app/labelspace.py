from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from shutil import copy2

from .engine.tag_norm import normalize_tag

REPO_ID = "SmilingWolf/wd-eva02-large-tagger-v3"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"


@dataclass(slots=True)
class LabelSpace:
    names: list[str]
    rating_indices: list[int]
    general_indices: list[int]
    character_indices: list[int]

    def subset(self, indices: Sequence[int]) -> list[str]:
        return [self.names[i] for i in indices]


def ensure_local_files(
    *,
    repo_id: str = REPO_ID,
    models_dir: Path | None = None,
) -> tuple[Path, Path]:
    base_dir = models_dir or Path(__file__).resolve().parent.parent / "models" / "wd14"
    base_dir.mkdir(parents=True, exist_ok=True)

    model_path = base_dir / MODEL_FILENAME
    label_path = base_dir / LABEL_FILENAME

    if not model_path.exists():
        model_path = _download_file(repo_id, MODEL_FILENAME, model_path)

    if not label_path.exists():
        label_path = _download_file(repo_id, LABEL_FILENAME, label_path)

    return label_path, model_path


def load_labelspace(csv_path: Path | str) -> LabelSpace:
    df = pd.read_csv(csv_path)
    names = _normalize_names(df["name"].tolist())
    categories = df["category"].to_numpy()

    rating_indices = np.where(categories == 9)[0].astype(int).tolist()
    general_indices = np.where(categories == 0)[0].astype(int).tolist()
    character_indices = np.where(categories == 4)[0].astype(int).tolist()

    return LabelSpace(
        names=names,
        rating_indices=rating_indices,
        general_indices=general_indices,
        character_indices=character_indices,
    )


def _normalize_names(names: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for name in names:
        canonical = normalize_tag(name)
        normalized.append(canonical)
    return normalized


def _download_file(repo_id: str, filename: str, destination: Path) -> Path:
    try:
        downloaded = hf_hub_download(repo_id, filename)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Unable to download '{filename}' from {repo_id}."
            " Place the file manually under models/wd14/."
        ) from exc
    copy2(downloaded, destination)
    return destination
