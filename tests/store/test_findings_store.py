from __future__ import annotations

import asyncio
import sys
import types

from datetime import datetime, timezone
from pathlib import Path

import pytest

if "discord" not in sys.modules:  # pragma: no cover - provide minimal stub
    discord_stub = types.ModuleType("discord")
    sys.modules["discord"] = discord_stub

pytest.importorskip("aiosqlite")

from app.store import FindingsStore, StoreRunMetrics


def test_upsert_findings_insert_and_update(tmp_path: Path) -> None:
    async def _execute() -> None:
        store_path = tmp_path / "findings.db"
        store = FindingsStore(store_path)
        await store.connect()
        try:
            record = {
                "severity": "red",
                "rule_id": "TEST-RED",
                "rule_title": "Test Rule",
                "author_id": "123",
                "guild_id": "1",
                "channel_id": "2",
                "message_id": "3",
                "message_link": "https://discord.com/channels/1/2/3",
                "phash": "deadbeef",
                "created_at": "2025-10-12T00:00:00Z",
                "reasons": ["exp=0.80"],
                "metrics": {
                    "eval_ms": 12.5,
                    "winning": {"origin": "dsl"},
                },
                "messages": [
                    {
                        "attachments": [{"id": "attach-1"}],
                    }
                ],
            }
            stats = await store.upsert_findings(
                [record],
                profile="current",
                partition_date="2025-10-12",
                pipeline_run_id="2025-10-12T00:15:00Z",
                store_run_id="store-current-2025-10-12T00:16:00+00:00",
            )
            assert stats.inserted == 1
            assert stats.updated == 0

            results = await store.fetch_by_message(guild_id=1, channel_id=2, message_id=3)
            assert len(results) == 1
            finding = results[0]
            assert finding.severity == "red"
            assert finding.rule_id == "TEST-RED"

            updated = dict(record)
            updated["severity"] = "orange"
            updated["reasons"] = ["exp=0.60"]
            stats_second = await store.upsert_findings(
                [updated],
                profile="current",
                partition_date="2025-10-12",
                pipeline_run_id="2025-10-12T00:30:00Z",
                store_run_id="store-current-2025-10-12T00:31:00+00:00",
            )
            assert stats_second.inserted == 0
            assert stats_second.updated == 1

            refreshed = await store.fetch_by_message(guild_id=1, channel_id=2, message_id=3)
            assert len(refreshed) == 1
            assert refreshed[0].severity == "orange"
        finally:
            await store.close()

    asyncio.run(_execute())


def test_record_run_and_fetch_recent_runs(tmp_path: Path) -> None:
    async def _execute() -> None:
        store = FindingsStore(tmp_path / "findings.db")
        await store.connect()
        try:
            metrics = StoreRunMetrics(
                store_run_id="store-current-2025-10-12T00:16:00+00:00",
                pipeline_run_id="2025-10-12T00:15:00Z",
                profile="current",
                partition_date="2025-10-12",
                started_at=datetime(2025, 10, 12, 0, 16, tzinfo=timezone.utc),
                finished_at=datetime(2025, 10, 12, 0, 16, 30, tzinfo=timezone.utc),
                status="success",
                attempts=1,
                records_total=10,
                records_upserted=10,
                records_skipped=0,
                duration_sec=30.5,
                retry_count=0,
                db_size_bytes=1024,
                log_path=None,
                extra_metrics={"findings_hash": "abc123"},
            )
            await store.record_run(metrics)
            runs = await store.fetch_recent_runs(limit=1)
            assert len(runs) == 1
            assert runs[0].run_id == metrics.store_run_id
            assert runs[0].status == "success"
        finally:
            await store.close()

    asyncio.run(_execute())
