from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from timm import create_model
from timm.data import create_transform, resolve_data_config
from timm.models import load_state_dict_from_hf

from app.analyzer_wd14 import WD14Session
from app.labelspace import ensure_local_files, load_labelspace

MODEL_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and image.info.get("transparency") is not None
    ):
        rgba_image = image.convert("RGBA")
        background = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
        background.alpha_composite(rgba_image)
        return background.convert("RGB")
    return image.convert("RGB")


def pil_pad_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


@dataclass
class PredictionResult:
    scores: np.ndarray


def run_official(image: Image.Image) -> PredictionResult:
    model = create_model("hf-hub:" + MODEL_REPO, pretrained=False)
    state_dict = load_state_dict_from_hf(MODEL_REPO)
    model.load_state_dict(state_dict)
    model.eval()

    config = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**config)

    rgb = pil_pad_square(pil_ensure_rgb(image))
    tensor = transform(rgb).unsqueeze(0)
    tensor = tensor[:, [2, 1, 0]]

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return PredictionResult(scores=probs)


def run_local(image: Image.Image) -> PredictionResult:
    label_csv, model_path = ensure_local_files()
    session = WD14Session(str(model_path))
    scores = session.infer([image])[0]
    return PredictionResult(scores=scores)


def topk_diff(
    names: Sequence[str],
    official_scores: np.ndarray,
    local_scores: np.ndarray,
    k: int,
) -> list[tuple[str, float, float, float]]:
    diffs = []
    for idx, name in enumerate(names):
        off = float(official_scores[idx])
        loc = float(local_scores[idx])
        diffs.append((name, off, loc, abs(off - loc)))
    diffs.sort(key=lambda item: item[3], reverse=True)
    return diffs[:k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare local WD14 with official timm implementation")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    image = Image.open(args.image)

    label_csv, _ = ensure_local_files()
    labelspace = load_labelspace(label_csv)

    print("Running official timm implementation...")
    official = run_official(image)
    print("Running local ONNX implementation...")
    local = run_local(image)

    print("\nRating comparison:")
    for idx in labelspace.rating_indices:
        tag = labelspace.names[idx]
        print(
            f"  {tag:12s} official={official.scores[idx]:.4f} local={local.scores[idx]:.4f}"
        )

    print("\nTop diffs (absolute):")
    for tag, off, loc, diff in topk_diff(labelspace.names, official.scores, local.scores, args.topk):
        print(f"  {tag:25s} official={off:.4f} local={loc:.4f} diff={diff:.4f}")


if __name__ == "__main__":
    main()
