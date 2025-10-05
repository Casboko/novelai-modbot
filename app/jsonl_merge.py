from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping


def _temporary_path(path: Path) -> Path:
    if path.suffix:
        return path.with_suffix(path.suffix + ".tmp")
    return path.with_name(path.name + ".tmp")


def load_updates_from_jsonl(paths: Iterable[Path], key_field: str) -> dict[str, dict]:
    updates: dict[str, dict] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = str(record[key_field])
                updates[key] = record
    return updates


def merge_jsonl_records(
    base_path: Path,
    updates: Mapping[str, dict],
    *,
    key_field: str = "phash",
    out_path: Path | None = None,
    ensure_ascii: bool = False,
) -> None:
    if not updates:
        return

    destination = out_path or base_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not base_path.exists():
        with destination.open("w", encoding="utf-8") as dst:
            for key, record in updates.items():
                _write_record(dst, record, ensure_ascii)
        return

    tmp_path = _temporary_path(destination)
    seen: set[str] = set()
    with base_path.open("r", encoding="utf-8") as src, tmp_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            existing = json.loads(line)
            key = str(existing.get(key_field))
            replacement = updates.get(key)
            if replacement is not None:
                _write_record(dst, replacement, ensure_ascii)
                seen.add(key)
            else:
                _write_record(dst, existing, ensure_ascii)
        for key, record in updates.items():
            if key in seen:
                continue
            _write_record(dst, record, ensure_ascii)
    tmp_path.replace(destination)


def _write_record(dst, record: dict, ensure_ascii: bool) -> None:
    dst.write(json.dumps(record, ensure_ascii=ensure_ascii))
    dst.write("\n")

