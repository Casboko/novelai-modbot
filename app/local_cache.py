from __future__ import annotations

import csv
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _resolve_cache_root() -> Path:
    env_value = os.getenv("CACHE_ROOT")
    if env_value:
        return Path(env_value).resolve()
    workspace = Path("/workspace/cache")
    if workspace.exists():
        return workspace.resolve()
    return Path("./cache").resolve()


CACHE_ROOT: Path = _resolve_cache_root()
MANIFEST_PATH: Path = CACHE_ROOT / "manifest" / "p0_images.csv"
FULL_DIR: Path = CACHE_ROOT / "full"


@lru_cache(maxsize=1)
def _manifest_index() -> dict[str, str]:
    index: dict[str, str] = {}
    if not MANIFEST_PATH.is_file():
        return index
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("status") or "").strip().lower() != "ok":
                continue
            phash = (row.get("phash") or "").strip()
            local_path = (row.get("local_path") or "").strip()
            if not phash or not local_path:
                continue
            index[phash] = local_path
    return index


def resolve_local_file(phash: str) -> Optional[str]:
    manifest = _manifest_index()
    candidate = manifest.get(phash)
    if candidate:
        path = Path(candidate)
        if path.is_file():
            return str(path.resolve())
    return _glob_fallback(phash)


def _glob_fallback(phash: str) -> Optional[str]:
    search_root = FULL_DIR
    if not search_root.exists():
        return None
    for candidate in search_root.glob(f"{phash}.*"):
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def clear_cache() -> None:
    _manifest_index.cache_clear()  # type: ignore[attr-defined]
