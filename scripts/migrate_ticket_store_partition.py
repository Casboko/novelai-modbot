#!/usr/bin/env python3
"""SQLite migration helper to add profile/partition columns to the ticket store."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path

from app.profiles import DEFAULT_PROFILE
from app.store import TicketStore

SENTINEL = TicketStore.PARTITION_SENTINEL
SCHEMA_VERSION = TicketStore.SCHEMA_VERSION


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    try:
        for row in cursor.fetchall():
            if row[1] == column:
                return True
        return False
    finally:
        cursor.close()


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
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
          profile TEXT NOT NULL DEFAULT 'current',
          partition_date TEXT NOT NULL DEFAULT '1970-01-01',
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
        CREATE INDEX IF NOT EXISTS idx_tickets_due ON tickets(status, due_at);
        """
    )


def add_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> bool:
    if column_exists(conn, table, column):
        return False
    conn.execute(ddl)
    return True


def backfill(conn: sqlite3.Connection, column: str, value: str) -> None:
    escaped = value.replace("'", "''")
    conn.execute(
        f"UPDATE tickets SET {column}='{escaped}' WHERE {column} IS NULL OR {column}=''"
    )


def update_user_version(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(f"PRAGMA user_version={version};")


def get_user_version(conn: sqlite3.Connection) -> int:
    cursor = conn.execute("PRAGMA user_version;")
    try:
        row = cursor.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    finally:
        cursor.close()


def migrate(path: Path, dry_run: bool) -> None:
    conn = sqlite3.connect(path)
    try:
        ensure_tables(conn)
        to_perform: list[str] = []

        profile_default = DEFAULT_PROFILE
        partition_default = SENTINEL

        if not column_exists(conn, "tickets", "profile"):
            escaped = profile_default.replace("'", "''")
            to_perform.append(
                f"ALTER TABLE tickets ADD COLUMN profile TEXT NOT NULL DEFAULT '{escaped}'"
            )
            to_perform.append(
                f"UPDATE tickets SET profile='{escaped}' WHERE profile IS NULL OR profile=''"
            )
        if not column_exists(conn, "tickets", "partition_date"):
            escaped = partition_default.replace("'", "''")
            to_perform.append(
                f"ALTER TABLE tickets ADD COLUMN partition_date TEXT NOT NULL DEFAULT '{escaped}'"
            )
            to_perform.append(
                f"UPDATE tickets SET partition_date='{escaped}' WHERE partition_date IS NULL OR partition_date=''"
            )

        current_version = get_user_version(conn)

        if dry_run:
            if not to_perform:
                if current_version >= SCHEMA_VERSION:
                    print("No migration necessary; schema is already up to date.")
                else:
                    print(
                        "No DDL changes required, but user_version would be updated "
                        f"from {current_version} to {SCHEMA_VERSION}."
                    )
            else:
                print("The following statements would be executed:")
                for stmt in to_perform:
                    print(f"  {stmt}")
            print(f"Current PRAGMA user_version={current_version}")
            return

        if to_perform:
            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup_path)
            print(f"Backup created at {backup_path}")

        for stmt in to_perform:
            conn.execute(stmt)

        if current_version < SCHEMA_VERSION:
            update_user_version(conn, SCHEMA_VERSION)
        conn.commit()
        print("Migration completed successfully.")
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate TicketStore schema.")
    parser.add_argument("path", type=Path, help="Path to ticket SQLite database")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned changes without modifying the database",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path: Path = args.path
    if not path.exists():
        raise SystemExit(f"Database not found: {path}")
    migrate(path, args.dry_run)


if __name__ == "__main__":
    main()
