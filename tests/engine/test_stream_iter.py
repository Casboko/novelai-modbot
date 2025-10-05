from __future__ import annotations

import json
from pathlib import Path

from app.engine.types import DslPolicy
from app.io.stream import iter_jsonl


def test_iter_jsonl_reads_directory(tmp_path: Path) -> None:
    file_a = tmp_path / "a.jsonl"
    file_b = tmp_path / "b.jsonl"
    file_a.write_text(json.dumps({"id": 1}) + "\n", encoding="utf-8")
    file_b.write_text(json.dumps({"id": 2}) + "\n", encoding="utf-8")

    records = list(iter_jsonl(tmp_path, sort="name", policy=DslPolicy()))
    assert [record["id"] for record in records] == [1, 2]


def test_iter_jsonl_limit_offset(tmp_path: Path) -> None:
    file_a = tmp_path / "data.jsonl"
    file_a.write_text("".join(json.dumps({"id": idx}) + "\n" for idx in range(5)), encoding="utf-8")

    records = list(iter_jsonl(file_a, limit=2, offset=1))
    assert [record["id"] for record in records] == [1, 2]


def test_iter_jsonl_reverse_sort(tmp_path: Path) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    first.write_text(json.dumps({"id": "first"}) + "\n", encoding="utf-8")
    second.write_text(json.dumps({"id": "second"}) + "\n", encoding="utf-8")

    records = list(iter_jsonl(tmp_path, sort="reverse-name"))
    assert [record["id"] for record in records] == ["second", "first"]
