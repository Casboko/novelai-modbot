from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping

from zoneinfo import ZoneInfo

from app.profiles import PartitionPaths, ProfileContext


def write_partition_jsonl(
    root: Path,
    profile: str,
    date_token: str,
    stage: str,
    records: Iterable[Mapping[str, object]],
) -> Path:
    """Write JSONL records to a partitioned stage file and return its path."""

    context = ProfileContext(profile=profile, date=date.fromisoformat(date_token), tzinfo=ZoneInfo("UTC"))
    target = PartitionPaths(context, root=root).stage_file(stage, ensure_parent=True)
    target.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return target


LEGACY_TICKETS_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
  ticket_id TEXT PRIMARY KEY,
  guild_id INTEGER NOT NULL,
  channel_id INTEGER NOT NULL,
  message_id INTEGER NOT NULL,
  author_id INTEGER NOT NULL,
  severity TEXT NOT NULL,
  rule_id TEXT,
  reason TEXT,
  message_link TEXT NOT NULL,
  due_at TEXT NOT NULL,
  status TEXT NOT NULL,
  executor_id INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ticket_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ticket_id TEXT NOT NULL,
  actor_id INTEGER NOT NULL,
  action TEXT NOT NULL,
  detail TEXT,
  created_at TEXT NOT NULL
);
"""


def create_legacy_ticket_db(path: Path) -> None:
    """Initialise a legacy ticket DB without profile/partition columns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(LEGACY_TICKETS_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def remove_ticketstore_backup(path: Path) -> None:
    """Remove the backup file produced by migration helpers if present."""

    backup = path.with_suffix(path.suffix + ".bak")
    if backup.exists():
        backup.unlink()
