from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(slots=True)
class CacheKey:
    phash: str
    model: str
    version: str


class NudeNetCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    phash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    version TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (phash, model, version)
                )
                """
            )

    def get(self, key: CacheKey) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM detections WHERE phash=? AND model=? AND version=?",
                (key.phash, key.model, key.version),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, key: CacheKey, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO detections (phash, model, version, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(phash, model, version)
                DO UPDATE SET payload=excluded.payload
                """,
                (key.phash, key.model, key.version, data),
            )
            conn.commit()
