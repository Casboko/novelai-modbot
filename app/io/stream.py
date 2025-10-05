from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Literal

from ..engine.types import DslPolicy

SortMode = Literal["name", "mtime", "reverse-name", "reverse-mtime"]


def _expand_paths(path: Path, pattern: str, sort: SortMode) -> list[Path]:
    if path.is_file():
        return [path]
    candidates = [candidate for candidate in path.glob(pattern) if candidate.is_file()]
    if sort in ("name", "reverse-name"):
        ordered = sorted(candidates, key=lambda item: item.name)
        if sort.startswith("reverse"):
            ordered.reverse()
    elif sort in ("mtime", "reverse-mtime"):
        ordered = sorted(candidates, key=lambda item: item.stat().st_mtime)
        if sort.startswith("reverse"):
            ordered.reverse()
    else:  # pragma: no cover - defensive fallback
        ordered = sorted(candidates, key=lambda item: item.name)
    return ordered


def iter_jsonl(
    source: Path | str,
    *,
    pattern: str = "*.jsonl",
    sort: SortMode = "name",
    limit: int = 0,
    offset: int = 0,
    policy: DslPolicy | None = None,
) -> Iterator[dict]:
    """Yield JSONL records from ``source``.

    Parameters
    ----------
    source:
        Path or directory containing JSONL files.
    pattern:
        Glob pattern evaluated when ``source`` is a directory.
    sort:
        Ordering for matched files. ``reverse-*`` variants invert the selection.
    limit:
        Maximum number of records to yield (0 disables the limit).
    offset:
        Number of records to skip across the merged stream before yielding.
    policy:
        Optional DSL policy used for structured warning emission on decode errors.
    """
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
