from __future__ import annotations

import csv
from pathlib import Path

from app.triage import P3_CSV_HEADER, write_report_csv


def _sample_record() -> dict:
    return {
        "severity": "red",
        "rule_id": None,
        "rule_title": None,
        "message_link": "https://example.invalid/msg",
        "author_id": "1001",
        "messages": [{"author_id": "1001", "is_nsfw_channel": False, "attachments": []}],
        "wd14": {
            "rating": {
                "general": 0.7,
                "sensitive": 0.1,
                "questionable": 0.1,
                "explicit": 0.1,
            },
            "general": [["tag_a", 0.5], ["tag_b", 0.4]],
        },
        "nudity_detections": [],
        "metrics": {
            "exposure_score": 0.2,
            "placement_risk": 0.3,
            "nsfw_margin": 0.1,
            "nsfw_ratio": 0.2,
            "nsfw_general_sum": 0.3,
            "animals_sum": 0.0,
        },
        "xsignals": {"placement_risk_pre": 0.1},
        "reasons": ["sample"],
    }


def test_write_report_csv_header_freeze(tmp_path: Path) -> None:
    report_path = tmp_path / "report.csv"
    rows = write_report_csv([_sample_record()], report_path, ())
    assert rows == 1

    with report_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)

    assert header == list(P3_CSV_HEADER)
