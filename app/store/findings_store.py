from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import aiosqlite

from .models import Finding, StoreRun, StoreRunMetrics, UpsertStats


class FindingsStore:
    SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = path
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        if self._db is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA foreign_keys=ON;")
        await self._db.execute("PRAGMA busy_timeout=60000;")
        await self._run_migrations()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def upsert_findings(
        self,
        records: Iterable[dict[str, Any]],
        *,
        profile: str,
        partition_date: str,
        pipeline_run_id: str,
        store_run_id: str,
        logger: Any | None = None,
        dry_run: bool = False,
    ) -> UpsertStats:
        db = await self._require_db()
        stats = UpsertStats()
        ingested_at = _now_iso()
        await db.execute("BEGIN")
        try:
            for index, record in enumerate(records, start=1):
                stats.total += 1
                try:
                    prepared = self._prepare_record(
                        record,
                        profile=profile,
                        partition_date=partition_date,
                        pipeline_run_id=pipeline_run_id,
                        store_run_id=store_run_id,
                        ingested_at=ingested_at,
                    )
                except ValueError as exc:
                    stats.skipped += 1
                    _log_warn(
                        logger,
                        "store_upsert_skip",
                        index=index,
                        reason=str(exc),
                    )
                    continue

                exists = await self._finding_exists(db, prepared["finding_id"])
                await db.execute(
                    """
                    INSERT INTO findings (
                      finding_id, finding_id_hex, profile, partition_date,
                      guild_id, channel_id, message_id, author_id, severity,
                      rule_id, rule_title, phash, message_link, created_at,
                      captured_at, payload_json, metrics_json, eval_ms,
                      winning_origin, notified_at, pipeline_run_id,
                      store_run_id, ingested_at, updated_at
                    ) VALUES (
                      :finding_id, :finding_id_hex, :profile, :partition_date,
                      :guild_id, :channel_id, :message_id, :author_id, :severity,
                      :rule_id, :rule_title, :phash, :message_link, :created_at,
                      :captured_at, :payload_json, :metrics_json, :eval_ms,
                      :winning_origin, :notified_at, :pipeline_run_id,
                      :store_run_id, :ingested_at, :updated_at
                    )
                    ON CONFLICT(finding_id) DO UPDATE SET
                      finding_id_hex=excluded.finding_id_hex,
                      profile=excluded.profile,
                      partition_date=excluded.partition_date,
                      guild_id=excluded.guild_id,
                      channel_id=excluded.channel_id,
                      message_id=excluded.message_id,
                      author_id=excluded.author_id,
                      severity=excluded.severity,
                      rule_id=excluded.rule_id,
                      rule_title=excluded.rule_title,
                      phash=excluded.phash,
                      message_link=excluded.message_link,
                      created_at=excluded.created_at,
                      captured_at=excluded.captured_at,
                      payload_json=excluded.payload_json,
                      metrics_json=excluded.metrics_json,
                      eval_ms=excluded.eval_ms,
                      winning_origin=excluded.winning_origin,
                      pipeline_run_id=excluded.pipeline_run_id,
                      store_run_id=excluded.store_run_id,
                      updated_at=excluded.updated_at,
                      notified_at=CASE
                        WHEN findings.notified_at IS NULL THEN excluded.notified_at
                        ELSE findings.notified_at
                      END
                    """,
                    prepared,
                )
                await db.execute("DELETE FROM finding_reasons WHERE finding_id=?", (prepared["finding_id"],))
                reasons = prepared["reasons"]
                if reasons:
                    await db.executemany(
                        "INSERT INTO finding_reasons (finding_id, ordinal, reason_text) VALUES (?, ?, ?)",
                        [(prepared["finding_id"], ordinal, text) for ordinal, text in enumerate(reasons)],
                    )
                if exists:
                    stats.updated += 1
                else:
                    stats.inserted += 1
            if dry_run:
                await db.rollback()
            else:
                await db.commit()
        except Exception:
            await db.rollback()
            raise
        return stats

    async def mark_notified(self, finding_ids: Sequence[bytes], *, notified_at: datetime) -> int:
        if not finding_ids:
            return 0
        db = await self._require_db()
        notified_iso = notified_at.astimezone(timezone.utc).replace(microsecond=0).isoformat()
        placeholders = ",".join("?" for _ in finding_ids)
        query = f"UPDATE findings SET notified_at=?, updated_at=? WHERE finding_id IN ({placeholders})"
        cursor = await db.execute(query, (notified_iso, notified_iso, *finding_ids))
        await db.commit()
        return cursor.rowcount

    async def fetch_recent(
        self,
        profile: str,
        *,
        limit: int,
        severity: Sequence[str] | None = None,
    ) -> list[Finding]:
        db = await self._require_db()
        params: list[Any] = [profile]
        severity_clause = ""
        if severity:
            placeholders = ",".join("?" for _ in severity)
            severity_clause = f"AND severity IN ({placeholders})"
            params.extend(severity)
        params.append(limit)
        query = f"""
            SELECT * FROM findings
            WHERE profile=? {severity_clause}
            ORDER BY created_at DESC, message_id DESC
            LIMIT ?
        """
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        await cursor.close()
        return [self._row_to_finding(row) for row in rows]

    async def fetch_by_message(self, guild_id: int, channel_id: int, message_id: int) -> list[Finding]:
        db = await self._require_db()
        cursor = await db.execute(
            """
            SELECT * FROM findings
            WHERE guild_id=? AND channel_id=? AND message_id=?
            ORDER BY created_at DESC
            """,
            (guild_id, channel_id, message_id),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [self._row_to_finding(row) for row in rows]

    async def fetch_run(self, run_id: str) -> StoreRun | None:
        db = await self._require_db()
        cursor = await db.execute("SELECT * FROM store_runs WHERE run_id=?", (run_id,))
        row = await cursor.fetchone()
        await cursor.close()
        return self._row_to_store_run(row) if row else None

    async def fetch_recent_runs(self, *, limit: int = 5) -> list[StoreRun]:
        db = await self._require_db()
        cursor = await db.execute(
            "SELECT * FROM store_runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [self._row_to_store_run(row) for row in rows if row is not None]  # type: ignore[arg-type]

    async def record_run(self, metrics: StoreRunMetrics) -> None:
        db = await self._require_db()
        created_at = _now_iso()
        payload = {
            "run_id": metrics.store_run_id,
            "pipeline_run_id": metrics.pipeline_run_id,
            "profile": metrics.profile,
            "partition_date": metrics.partition_date,
            "started_at": metrics.started_at.astimezone(timezone.utc).replace(microsecond=0).isoformat(),
            "finished_at": metrics.finished_at.astimezone(timezone.utc).replace(microsecond=0).isoformat(),
            "status": metrics.status,
            "attempts": metrics.attempts,
            "records_total": metrics.records_total,
            "records_upserted": metrics.records_upserted,
            "records_skipped": metrics.records_skipped,
            "duration_sec": metrics.duration_sec,
            "retry_count": metrics.retry_count,
            "db_size_bytes": metrics.db_size_bytes,
            "log_path": metrics.log_path,
            "metrics_json": json.dumps(metrics.extra_metrics, ensure_ascii=False) if metrics.extra_metrics else None,
            "created_at": created_at,
        }
        await db.execute(
            """
            INSERT INTO store_runs (
              run_id, pipeline_run_id, profile, partition_date,
              started_at, finished_at, status, attempts,
              records_total, records_upserted, records_skipped,
              duration_sec, retry_count, db_size_bytes,
              log_path, metrics_json, created_at
            ) VALUES (
              :run_id, :pipeline_run_id, :profile, :partition_date,
              :started_at, :finished_at, :status, :attempts,
              :records_total, :records_upserted, :records_skipped,
              :duration_sec, :retry_count, :db_size_bytes,
              :log_path, :metrics_json, :created_at
            )
            ON CONFLICT(run_id) DO UPDATE SET
              pipeline_run_id=excluded.pipeline_run_id,
              profile=excluded.profile,
              partition_date=excluded.partition_date,
              started_at=excluded.started_at,
              finished_at=excluded.finished_at,
              status=excluded.status,
              attempts=excluded.attempts,
              records_total=excluded.records_total,
              records_upserted=excluded.records_upserted,
              records_skipped=excluded.records_skipped,
              duration_sec=excluded.duration_sec,
              retry_count=excluded.retry_count,
              db_size_bytes=excluded.db_size_bytes,
              log_path=excluded.log_path,
              metrics_json=excluded.metrics_json,
              created_at=excluded.created_at
            """,
            payload,
        )
        await db.commit()

    async def _finding_exists(self, db: aiosqlite.Connection, finding_id: bytes) -> bool:
        cursor = await db.execute("SELECT 1 FROM findings WHERE finding_id=?", (finding_id,))
        row = await cursor.fetchone()
        await cursor.close()
        return row is not None

    def _prepare_record(
        self,
        record: dict[str, Any],
        *,
        profile: str,
        partition_date: str,
        pipeline_run_id: str,
        store_run_id: str,
        ingested_at: str,
    ) -> dict[str, Any]:
        guild_id = _to_int(record.get("guild_id"))
        channel_id = _to_int(record.get("channel_id"))
        message_id = _to_int(record.get("message_id"))
        if guild_id is None or channel_id is None or message_id is None:
            raise ValueError("missing identifiers")
        severity = _to_str(record.get("severity"))
        if not severity:
            raise ValueError("missing severity")
        author_id = _to_int(record.get("author_id"))
        rule_id = _to_optional_str(record.get("rule_id"))
        rule_title = _to_optional_str(record.get("rule_title"))
        phash = _to_optional_str(record.get("phash"))
        message_link = _to_optional_str(record.get("message_link"))
        created_at = _to_optional_str(record.get("created_at"))
        metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
        eval_ms = _to_float(metrics.get("eval_ms")) if isinstance(metrics, dict) else None
        winning_origin = None
        if isinstance(metrics, dict):
            winning = metrics.get("winning")
            if isinstance(winning, dict):
                winning_origin = _to_optional_str(winning.get("origin"))
        attachments = []
        for message in record.get("messages") or []:
            if isinstance(message, dict):
                attachments = message.get("attachments") or []
                if attachments:
                    break
        attachment_id = None
        if isinstance(attachments, list) and attachments:
            first = attachments[0]
            if isinstance(first, dict):
                attachment_id = _to_optional_str(first.get("id"))
        fingerprint = f"{profile}:{partition_date}:{message_id}:{attachment_id or phash or 'nohash'}"
        digest = hashlib.sha1(fingerprint.encode("utf-8")).digest()
        reasons = []
        for item in record.get("reasons") or []:
            text = _to_optional_str(item)
            if text:
                reasons.append(text)
        payload_json = json.dumps(record, ensure_ascii=False)
        metrics_json = json.dumps(metrics or {}, ensure_ascii=False)
        return {
            "finding_id": digest,
            "finding_id_hex": digest.hex(),
            "profile": profile,
            "partition_date": partition_date,
            "guild_id": guild_id,
            "channel_id": channel_id,
            "message_id": message_id,
            "author_id": author_id,
            "severity": severity,
            "rule_id": rule_id,
            "rule_title": rule_title,
            "phash": phash,
            "message_link": message_link,
            "created_at": created_at,
            "captured_at": pipeline_run_id,
            "payload_json": payload_json,
            "metrics_json": metrics_json,
            "eval_ms": eval_ms,
            "winning_origin": winning_origin,
            "notified_at": None,
            "pipeline_run_id": pipeline_run_id,
            "store_run_id": store_run_id,
            "ingested_at": ingested_at,
            "updated_at": ingested_at,
            "reasons": reasons,
        }

    async def _run_migrations(self) -> None:
        db = self._db
        assert db is not None
        await db.executescript(
            """
            CREATE TABLE IF NOT EXISTS findings (
              finding_id BLOB PRIMARY KEY,
              finding_id_hex TEXT UNIQUE NOT NULL,
              profile TEXT NOT NULL,
              partition_date TEXT NOT NULL,
              guild_id INTEGER NOT NULL,
              channel_id INTEGER NOT NULL,
              message_id INTEGER NOT NULL,
              author_id INTEGER,
              severity TEXT NOT NULL,
              rule_id TEXT,
              rule_title TEXT,
              phash TEXT,
              message_link TEXT,
              created_at TEXT,
              captured_at TEXT,
              payload_json TEXT NOT NULL,
              metrics_json TEXT NOT NULL,
              eval_ms REAL,
              winning_origin TEXT,
              notified_at TEXT,
              pipeline_run_id TEXT NOT NULL,
              store_run_id TEXT NOT NULL,
              ingested_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS finding_reasons (
              finding_id BLOB NOT NULL,
              ordinal INTEGER NOT NULL,
              reason_text TEXT NOT NULL,
              PRIMARY KEY (finding_id, ordinal),
              FOREIGN KEY (finding_id) REFERENCES findings(finding_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS store_runs (
              run_id TEXT PRIMARY KEY,
              pipeline_run_id TEXT NOT NULL,
              profile TEXT NOT NULL,
              partition_date TEXT NOT NULL,
              started_at TEXT NOT NULL,
              finished_at TEXT NOT NULL,
              status TEXT NOT NULL,
              attempts INTEGER NOT NULL,
              records_total INTEGER NOT NULL,
              records_upserted INTEGER NOT NULL,
              records_skipped INTEGER NOT NULL,
              duration_sec REAL NOT NULL,
              retry_count INTEGER NOT NULL,
              db_size_bytes INTEGER NOT NULL,
              log_path TEXT,
              metrics_json TEXT,
              created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_findings_profile_date ON findings(profile, partition_date DESC);
            CREATE INDEX IF NOT EXISTS idx_findings_severity ON findings(severity, partition_date DESC);
            CREATE INDEX IF NOT EXISTS idx_findings_message ON findings(guild_id, channel_id, message_id);
            """
        )
        await db.execute(f"PRAGMA user_version={self.SCHEMA_VERSION};")
        await db.commit()

    async def _require_db(self) -> aiosqlite.Connection:
        if self._db is None:
            await self.connect()
        assert self._db is not None
        return self._db

    def _row_to_finding(self, row: aiosqlite.Row) -> Finding:
        return Finding(
            finding_id=row["finding_id"],
            finding_id_hex=row["finding_id_hex"],
            profile=row["profile"],
            partition_date=row["partition_date"],
            guild_id=row["guild_id"],
            channel_id=row["channel_id"],
            message_id=row["message_id"],
            author_id=row["author_id"],
            severity=row["severity"],
            rule_id=row["rule_id"],
            rule_title=row["rule_title"],
            phash=row["phash"],
            message_link=row["message_link"],
            created_at=row["created_at"],
            captured_at=row["captured_at"],
            payload_json=row["payload_json"],
            metrics_json=row["metrics_json"],
            eval_ms=row["eval_ms"],
            winning_origin=row["winning_origin"],
            notified_at=row["notified_at"],
            pipeline_run_id=row["pipeline_run_id"],
            store_run_id=row["store_run_id"],
            ingested_at=row["ingested_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_store_run(self, row: aiosqlite.Row | None) -> StoreRun | None:
        if row is None:
            return None
        return StoreRun(
            run_id=row["run_id"],
            pipeline_run_id=row["pipeline_run_id"],
            profile=row["profile"],
            partition_date=row["partition_date"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            status=row["status"],
            attempts=row["attempts"],
            records_total=row["records_total"],
            records_upserted=row["records_upserted"],
            records_skipped=row["records_skipped"],
            duration_sec=row["duration_sec"],
            retry_count=row["retry_count"],
            db_size_bytes=row["db_size_bytes"],
            log_path=row["log_path"],
            metrics_json=row["metrics_json"],
            created_at=row["created_at"],
        )


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        return text
    if value is None:
        return ""
    return str(value).strip()


def _to_optional_str(value: Any) -> Optional[str]:
    text = _to_str(value)
    return text if text else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _log_warn(logger: Any | None, message: str, **fields: Any) -> None:
    if logger is None:
        return
    log_method = getattr(logger, "warn", None)
    if callable(log_method):
        log_method(message, **fields)
