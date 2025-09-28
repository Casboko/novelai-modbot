from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .engine.tag_norm import normalize_pair


@dataclass(slots=True)
class CacheKey:
    phash: str
    model: str
    revision: str


class WD14Cache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._configured = False
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=60.0)
        if not self._configured:
            self._configure(conn)
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    phash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    revision TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (phash, model, revision)
                )
                """
            )

    def _configure(self, conn: sqlite3.Connection) -> None:
        cursor = conn.execute("PRAGMA journal_mode=WAL")
        cursor.fetchone()
        conn.execute("PRAGMA synchronous=NORMAL")
        self._configured = True

    def get(self, key: CacheKey) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM predictions WHERE phash=? AND model=? AND revision=?",
                (key.phash, key.model, key.revision),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        _normalize_payload(payload)
        return payload

    def set(self, key: CacheKey, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO predictions (phash, model, revision, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(phash, model, revision) DO UPDATE SET payload=excluded.payload
                """,
                (key.phash, key.model, key.revision, data),
            )
            conn.commit()


def _merge_score_map(items: Any) -> list[tuple[str, float]]:
    merged: dict[str, float] = {}
    if not isinstance(items, list):
        return []
    for item in items:
        pair = normalize_pair(item)
        if pair is None:
            continue
        tag, score = pair
        previous = merged.get(tag)
        if previous is None or score > previous:
            merged[tag] = score
    return sorted(merged.items(), key=lambda entry: entry[1], reverse=True)


def _normalize_payload(payload: dict[str, Any]) -> None:
    for field in ("general_raw", "general"):
        items = payload.get(field)
        if not items:
            continue
        payload[field] = _merge_score_map(items)
