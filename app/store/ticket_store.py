from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
from aiosqlite import Connection

from ..profiles import DEFAULT_PROFILE


@dataclass(slots=True)
class Ticket:
    ticket_id: str
    guild_id: int
    channel_id: int
    message_id: int
    author_id: int
    severity: str
    rule_id: Optional[str]
    reason: Optional[str]
    message_link: str
    due_at: datetime
    status: str
    executor_id: int
    profile: str
    partition_date: str
    created_at: datetime
    updated_at: datetime


class TicketStore:
    SCHEMA_VERSION = 2
    PARTITION_SENTINEL = "1970-01-01"

    def __init__(self, path: Path) -> None:
        self.path = path
        self._db: Optional[aiosqlite.Connection] = None

    @staticmethod
    def build_ticket_id(guild_id: int, channel_id: int, message_id: int) -> str:
        return f"{guild_id}:{channel_id}:{message_id}"

    async def connect(self) -> None:
        if self._db is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA foreign_keys=ON;")
        await self._run_migrations()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def get_ticket(self, ticket_id: str) -> Optional[Ticket]:
        db = await self._require_db()
        cursor = await db.execute("SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,))
        row = await cursor.fetchone()
        await cursor.close()
        return self._row_to_ticket(row) if row else None

    async def register_notification(
        self,
        *,
        ticket_id: str,
        guild_id: int,
        channel_id: int,
        message_id: int,
        author_id: int,
        severity: str,
        rule_id: Optional[str],
        reason: Optional[str],
        message_link: str,
        due_at: datetime,
        executor_id: int,
        profile: Optional[str] = None,
        partition_date: Optional[str] = None,
    ) -> tuple[Ticket, bool]:
        db = await self._require_db()
        now = datetime.now(timezone.utc)
        due_value = self._to_iso(due_at)
        now_iso = self._to_iso(now)
        resolved_profile = profile or DEFAULT_PROFILE
        resolved_partition = partition_date or self.PARTITION_SENTINEL
        cursor = await db.execute(
            """
            INSERT INTO tickets (
              ticket_id, guild_id, channel_id, message_id, author_id,
              severity, rule_id, reason, message_link, due_at,
              status, executor_id, profile, partition_date,
              created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                guild_id,
                channel_id,
                message_id,
                author_id,
                severity,
                rule_id,
                reason,
                message_link,
                due_value,
                "notified",
                executor_id,
                resolved_profile,
                resolved_partition,
                now_iso,
                now_iso,
            ),
        )
        await db.commit()
        created_now = cursor.rowcount == 1
        ticket = await self.get_ticket(ticket_id)
        if ticket is None:  # pragma: no cover - defensive
            raise RuntimeError("Failed to persist ticket")
        return ticket, created_now

    async def update_status(
        self,
        ticket_id: str,
        *,
        status: str,
    ) -> Optional[Ticket]:
        db = await self._require_db()
        now_iso = self._to_iso(datetime.now(timezone.utc))
        await db.execute(
            "UPDATE tickets SET status=?, updated_at=? WHERE ticket_id=?",
            (status, now_iso, ticket_id),
        )
        await db.commit()
        return await self.get_ticket(ticket_id)

    async def append_log(
        self,
        *,
        ticket_id: str,
        actor_id: int,
        action: str,
        detail: Optional[str] = None,
    ) -> None:
        db = await self._require_db()
        await db.execute(
            "INSERT INTO ticket_logs(ticket_id, actor_id, action, detail, created_at) VALUES(?,?,?,?,?)",
            (
                ticket_id,
                actor_id,
                action,
                detail,
                self._to_iso(datetime.now(timezone.utc)),
            ),
        )
        await db.commit()

    async def fetch_due_tickets(self, now: datetime | None = None) -> list[Ticket]:
        db = await self._require_db()
        moment = now or datetime.now(timezone.utc)
        cursor = await db.execute(
            "SELECT * FROM tickets WHERE status=? AND due_at <= ?",
            ("notified", self._to_iso(moment)),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [self._row_to_ticket(row) for row in rows]

    async def fetch_active_tickets(self) -> list[Ticket]:
        """List tickets awaiting deletion (status=notified)."""
        db = await self._require_db()
        cursor = await db.execute(
            "SELECT * FROM tickets WHERE status=? ORDER BY due_at ASC",
            ("notified",),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [self._row_to_ticket(row) for row in rows]

    async def cancel_ticket(self, ticket_id: str, actor_id: int) -> Optional[Ticket]:
        """Mark a ticket as cancelled and record the action."""
        ticket = await self.update_status(ticket_id, status="cancelled")
        if ticket is not None:
            await self.append_log(
                ticket_id=ticket_id,
                actor_id=actor_id,
                action="cancel",
                detail="モデレーターがキャンセル",
            )
        return ticket

    async def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            await self.connect()
        assert self._db is not None  # for type checkers
        return self._db

    @staticmethod
    def _to_iso(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat()

    @staticmethod
    def _parse_iso(value: str) -> datetime:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _row_to_ticket(self, row: aiosqlite.Row) -> Ticket:
        return Ticket(
            ticket_id=row["ticket_id"],
            guild_id=row["guild_id"],
            channel_id=row["channel_id"],
            message_id=row["message_id"],
            author_id=row["author_id"],
            severity=row["severity"],
            rule_id=row["rule_id"],
            reason=row["reason"],
            message_link=row["message_link"],
            due_at=self._parse_iso(row["due_at"]),
            status=row["status"],
            executor_id=row["executor_id"],
            profile=row["profile"],
            partition_date=row["partition_date"],
            created_at=self._parse_iso(row["created_at"]),
            updated_at=self._parse_iso(row["updated_at"]),
        )

    async def _run_migrations(self) -> None:
        db = self._db
        assert db is not None
        await db.executescript(
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
        version = await self._get_user_version(db)
        if version < 2:
            await self._ensure_ticket_columns(db)
            await db.execute(f"PRAGMA user_version={self.SCHEMA_VERSION};")
        await db.commit()

    async def _get_user_version(self, db: Connection) -> int:
        cursor = await db.execute("PRAGMA user_version;")
        row = await cursor.fetchone()
        await cursor.close()
        return int(row[0]) if row and row[0] is not None else 0

    async def _ensure_ticket_columns(self, db: Connection) -> None:
        columns = await self._list_columns(db, "tickets")
        if "profile" not in columns:
            profile_default = DEFAULT_PROFILE.replace("'", "''")
            await db.execute(
                f"ALTER TABLE tickets ADD COLUMN profile TEXT NOT NULL DEFAULT '{profile_default}'",
            )
        else:
            profile_default = DEFAULT_PROFILE.replace("'", "''")
        await db.execute(
            f"UPDATE tickets SET profile='{profile_default}' WHERE profile IS NULL OR profile=''",
        )
        if "partition_date" not in columns:
            partition_default = self.PARTITION_SENTINEL.replace("'", "''")
            await db.execute(
                f"ALTER TABLE tickets ADD COLUMN partition_date TEXT NOT NULL DEFAULT '{partition_default}'",
            )
        else:
            partition_default = self.PARTITION_SENTINEL.replace("'", "''")
        await db.execute(
            f"UPDATE tickets SET partition_date='{partition_default}' WHERE partition_date IS NULL OR partition_date=''",
        )

    async def _list_columns(self, db: Connection, table: str) -> set[str]:
        cursor = await db.execute(f"PRAGMA table_info({table})")
        rows = await cursor.fetchall()
        await cursor.close()
        return {str(row[1]) for row in rows}
