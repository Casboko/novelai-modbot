from __future__ import annotations

import json
from pathlib import Path

from app.analysis_merge import load_scan_metadata, load_wd14


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_load_scan_metadata_multiple_files(tmp_path: Path) -> None:
    file_a = tmp_path / "scan_a.csv"
    file_b = tmp_path / "scan_b.csv"
    _write_csv(
        file_a,
        [
            {
                "phash_hex": "aaaa",
                "url": "https://example.com/a.png",
                "message_link": "https://discord.com/channels/0/0/1",
            }
        ],
    )
    _write_csv(
        file_b,
        [
            {
                "phash_hex": "bbbb",
                "url": "https://example.com/b.png",
                "message_link": "https://discord.com/channels/0/0/2",
            }
        ],
    )

    entries = load_scan_metadata([file_a, file_b], limit=0)
    assert sorted(entries.keys()) == ["aaaa", "bbbb"]


def test_load_scan_metadata_honours_limit(tmp_path: Path) -> None:
    file_path = tmp_path / "scan.csv"
    _write_csv(
        file_path,
        [
            {
                "phash_hex": "aaaa",
                "url": "https://example.com/a.png",
                "message_link": "link/a",
            },
            {
                "phash_hex": "bbbb",
                "url": "https://example.com/b.png",
                "message_link": "link/b",
            },
        ],
    )

    entries = load_scan_metadata([file_path], limit=1)
    assert len(entries) == 1


def test_load_wd14_multiple_sources(tmp_path: Path) -> None:
    file_a = tmp_path / "wd14_a.jsonl"
    file_b = tmp_path / "wd14_b.jsonl"
    file_a.write_text(json.dumps({"phash": "aaaa", "wd14": {"rating": {"explicit": 0.2}}}) + "\n", encoding="utf-8")
    file_b.write_text(json.dumps({"phash": "bbbb", "wd14": {"rating": {"explicit": 0.5}}}) + "\n", encoding="utf-8")

    entries = load_wd14([file_a, file_b], limit=0)
    assert set(entries.keys()) == {"aaaa", "bbbb"}


def test_load_wd14_limit_and_overwrite(tmp_path: Path) -> None:
    file_a = tmp_path / "wd14_a.jsonl"
    file_b = tmp_path / "wd14_b.jsonl"
    file_a.write_text(json.dumps({"phash": "aaaa", "value": 1}) + "\n", encoding="utf-8")
    file_b.write_text(
        json.dumps({"phash": "aaaa", "value": 2}) + "\n" + json.dumps({"phash": "bbbb", "value": 3}) + "\n",
        encoding="utf-8",
    )

    entries = load_wd14([file_a, file_b], limit=1)
    assert entries == {"aaaa": {"phash": "aaaa", "value": 2}}

