from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from app.calibration import EPS, apply_temperature

KEEP_UNDERSCORE = {"0_0", "(o)_(o)"}


def normalize_tag(name: str) -> str:
    value = str(name)
    if value in KEEP_UNDERSCORE:
        return value
    return value.replace("_", " ")


def coerce_pair(item: object) -> Tuple[str, float] | None:
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return str(item[0]), float(item[1])
    if isinstance(item, dict):
        name = item.get("name")
        score = item.get("score")
        if name is None or score is None:
            return None
        return str(name), float(score)
    return None


def load_wd14_predictions(path: Path) -> Dict[str, Dict[str, float]]:
    predictions: Dict[str, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            payload = json.loads(line)
            phash = payload.get("phash")
            if not phash:
                continue
            wd14 = payload.get("wd14", {})
            general_tags = wd14.get("general_raw") or wd14.get("general") or []
            tag_scores: Dict[str, float] = {}
            for item in general_tags:
                pair = coerce_pair(item)
                if pair is None:
                    continue
                name, score = pair
                tag_scores[normalize_tag(name)] = float(score)
            predictions[str(phash)] = tag_scores
    return predictions


def load_labels(path: Path) -> List[Tuple[str, str, float]]:
    records: List[Tuple[str, str, float]] = []
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            phash = (row.get("phash") or "").strip()
            tag = (row.get("tag") or "").strip()
            gt_raw = (row.get("gt") or "").strip()
            if not phash or not tag or not gt_raw:
                continue
            try:
                gt = float(gt_raw)
            except ValueError:
                continue
            gt = 1.0 if gt >= 0.5 else 0.0
            records.append((phash, normalize_tag(tag), gt))
    return records


def collect_samples(
    predictions: Dict[str, Dict[str, float]],
    labels: Iterable[Tuple[str, str, float]],
) -> Tuple[Dict[str, List[Tuple[float, float]]], int]:
    samples: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    missing = 0
    for phash, tag, gt in labels:
        tag_scores = predictions.get(phash)
        if not tag_scores:
            missing += 1
            continue
        score = tag_scores.get(tag)
        if score is None:
            missing += 1
            continue
        samples[tag].append((float(score), float(gt)))
    return samples, missing


def cross_entropy_loss(samples: Sequence[Tuple[float, float]], temperature: float) -> float:
    total = 0.0
    for score, label in samples:
        calibrated = apply_temperature(score, temperature)
        calibrated = min(max(calibrated, EPS), 1.0 - EPS)
        total += -(
            label * math.log(calibrated + EPS)
            + (1.0 - label) * math.log(1.0 - calibrated + EPS)
        )
    return total / len(samples)


def optimize_temperature(samples: Sequence[Tuple[float, float]]) -> float:
    best_T = 1.0
    best_loss = cross_entropy_loss(samples, best_T)
    # coarse search on log scale
    for exponent in range(-10, 11):
        candidate = math.exp(exponent / 5.0)
        loss = cross_entropy_loss(samples, candidate)
        if loss < best_loss:
            best_loss = loss
            best_T = candidate
    # local refinement around the best candidate
    for _ in range(3):
        lower = max(best_T / 2.0, 0.05)
        upper = min(best_T * 2.0, 20.0)
        step = (upper - lower) / 20.0
        for i in range(21):
            candidate = lower + step * i
            loss = cross_entropy_loss(samples, candidate)
            if loss < best_loss:
                best_loss = loss
                best_T = candidate
    return max(best_T, EPS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate WD14 temperature scaling per tag")
    parser.add_argument("--wd14", type=Path, default=Path("out/p1_wd14.jsonl"))
    parser.add_argument("--labels", type=Path, required=True, help="CSV with columns phash,tag,gt")
    parser.add_argument("--out", type=Path, default=Path("configs/calibration_wd14.json"))
    parser.add_argument("--min-samples", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_wd14_predictions(args.wd14)
    if not predictions:
        raise SystemExit("WD14 predictions not found. Run cli_wd14 first.")
    labels = load_labels(args.labels)
    if not labels:
        raise SystemExit("No labels found in CSV.")

    samples, missing = collect_samples(predictions, labels)
    if missing:
        print(f"[warn] Missing predictions for {missing} label rows", file=sys.stderr)

    general_calib: Dict[str, float] = {}
    for tag, pairs in samples.items():
        if len(pairs) < args.min_samples:
            continue
        temperature = optimize_temperature(pairs)
        general_calib[tag] = round(temperature, 6)

    result = {"general_tags": general_calib}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    kept = len(general_calib)
    if kept == 0:
        print("No tags met the minimum sample threshold.", file=sys.stderr)
    else:
        print(f"Saved calibration for {kept} tags to {args.out}")


if __name__ == "__main__":
    main()
