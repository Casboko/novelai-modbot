#!/usr/bin/env python3
"""簡易的な不確実域サンプル抽出ツール。

p2 の analysis JSONL から rating/questionable, rating/explicit, exposure_score が
しきい値近傍にあるレコードを抽出し、レビュー用 CSV を生成します。

Usage:
    python tools/export_uncertain.py \
      --analysis out/p2/p2_analysis_all.jsonl \
      --out out/review/uncertain_top200.csv \
      --q-thr 0.35 --e-thr 0.20 --eps 0.02 --topn 200
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def within(value: float, target: float, epsilon: float) -> bool:
    return abs(value - target) <= epsilon


def iter_analysis(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Export near-threshold samples for manual review")
    parser.add_argument("--analysis", type=Path, required=True, help="p2 analysis JSONL")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--q-thr", type=float, default=0.35, help="questionable score threshold")
    parser.add_argument("--e-thr", type=float, default=0.20, help="explicit score threshold")
    parser.add_argument("--exposure-thr", type=float, default=0.30, help="exposure score threshold")
    parser.add_argument("--eps", type=float, default=0.02, help="epsilon for near-threshold condition")
    parser.add_argument("--topn", type=int, default=200, help="Maximum rows to export")
    args = parser.parse_args()

    candidates: list[tuple[float, dict]] = []
    for record in iter_analysis(args.analysis):
        rating = record.get("wd14", {}).get("rating", {}) if isinstance(record.get("wd14"), dict) else {}
        q = float(rating.get("questionable", 0.0) or 0.0)
        e = float(rating.get("explicit", 0.0) or 0.0)
        expo = float(record.get("xsignals", {}).get("exposure_score", 0.0) or 0.0)
        if not (
            within(q, args.q_thr, args.eps)
            or within(e, args.e_thr, args.eps)
            or within(expo, args.exposure_thr, args.eps)
        ):
            continue
        score = max(
            abs(q - args.q_thr),
            abs(e - args.e_thr),
            abs(expo - args.exposure_thr),
        )
        candidates.append((score, {
            "message_link": record.get("message_link", ""),
            "questionable": q,
            "explicit": e,
            "exposure": expo,
        }))

    candidates.sort(key=lambda item: item[0])
    rows = candidates[: args.topn]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["message_link", "questionable", "explicit", "exposure"])
        writer.writeheader()
        for _, row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
