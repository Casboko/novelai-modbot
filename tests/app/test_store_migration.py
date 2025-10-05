from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.store import TicketStore
from tests.fixtures.profiles_helpers import (
    create_legacy_ticket_db,
    remove_ticketstore_backup,
)
from scripts import migrate_ticket_store_partition as migrate_script


def _insert_legacy_ticket(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO tickets(
              ticket_id, guild_id, channel_id, message_id, author_id,
              severity, rule_id, reason, message_link, due_at,
              status, executor_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "1:2:3",
                1,
                2,
                3,
                4,
                "orange",
                "RULE",
                "legacy",
                "https://discord.com/channels/1/2/3",
                now,
                "notified",
                5,
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _fetch_columns(path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(row[1]) for row in rows}
    finally:
        conn.close()


def _fetch_row(path: Path, query: str) -> tuple:
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(query).fetchone()
        assert row is not None
        return tuple(row)
    finally:
        conn.close()


def test_ticketstore_connect_migrates_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    create_legacy_ticket_db(db_path)
    _insert_legacy_ticket(db_path)

    async def exercise() -> None:
        store = TicketStore(db_path)
        await store.connect()
        await store.close()

    asyncio.run(exercise())

    columns = _fetch_columns(db_path, "tickets")
    assert {"profile", "partition_date"}.issubset(columns)

    profile, partition_date = _fetch_row(db_path, "SELECT profile, partition_date FROM tickets LIMIT 1")
    assert profile == "current"
    assert partition_date == TicketStore.PARTITION_SENTINEL

    version = _fetch_row(db_path, "PRAGMA user_version;")[0]
    assert version == TicketStore.SCHEMA_VERSION


def test_register_notification_persists_profile_and_partition(tmp_path: Path) -> None:
    db_path = tmp_path / "tickets.db"
    create_legacy_ticket_db(db_path)

    async def scenario():
        store = TicketStore(db_path)
        await store.connect()
        ticket, created = await store.register_notification(
            ticket_id="10:20:30",
            guild_id=10,
            channel_id=20,
            message_id=30,
            author_id=40,
            severity="red",
            rule_id="RULE",
            reason="reason",
            message_link="https://discord.com/channels/10/20/30",
            due_at=datetime.now(timezone.utc),
            executor_id=99,
            profile="demo",
            partition_date="2025-10-01",
        )
        await store.close()
        return ticket, created

    ticket, created = asyncio.run(scenario())
    assert created is True
    assert ticket.profile == "demo"
    assert ticket.partition_date == "2025-10-01"

    stored_profile, stored_partition = _fetch_row(
        db_path,
        "SELECT profile, partition_date FROM tickets WHERE ticket_id='10:20:30'",
    )
    assert stored_profile == "demo"
    assert stored_partition == "2025-10-01"


def test_migration_cli_creates_backup(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db_path = tmp_path / "cli.db"
    create_legacy_ticket_db(db_path)
    _insert_legacy_ticket(db_path)

    migrate_script.migrate(db_path, dry_run=True)
    dry_out = capsys.readouterr().out
    assert "would be executed" in dry_out or "No migration" in dry_out

    migrate_script.migrate(db_path, dry_run=False)
    backup = db_path.with_suffix(db_path.suffix + ".bak")
    assert backup.exists()

    columns = _fetch_columns(db_path, "tickets")
    assert {"profile", "partition_date"}.issubset(columns)

    remove_ticketstore_backup(db_path)
    assert not backup.exists()
