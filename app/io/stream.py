from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Literal

from ..engine.types import DslPolicy

SortMode = Literal["name", "mtime"]


def _expand_paths(path: Path, pattern: str, sort: SortMode) -> list[Path]:
    if path.is_file():
        return [path]
    candidates = path.glob(pattern)
    if sort == "name":
        ordered = sorted(candidates, key=lambda item: item.name)
    else:
        ordered = sorted(candidates, key=lambda item: item.stat().st_mtime)
    return [item for item in ordered if item.is_file()]


def iter_jsonl(
    source: Path | str,
    *,
    pattern: str = "*.jsonl",
    sort: SortMode = "name",
    limit: int = 0,
    offset: int = 0,
    policy: DslPolicy | None = None,
) -> Iterator[dict]:
    path = Path(source)
    files = _expand_paths(path, pattern, sort)
    produced = 0
    consumed = 0
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                consumed += 1
                if offset and consumed <= offset:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    if policy:
                        policy.warn_once(
                            f"JSON decode error in {file}:{line_no}: {exc}",
                            key=f"json:{file}:{line_no}",
                        )
                    continue
                yield record
                produced += 1
                if limit and produced >= limit:
                    return
