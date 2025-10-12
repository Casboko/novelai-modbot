from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass(slots=True)
class UpsertStats:
    total: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0


@dataclass(slots=True)
class Finding:
    finding_id: bytes
    finding_id_hex: str
    profile: str
    partition_date: str
    guild_id: int
    channel_id: int
    message_id: int
    author_id: Optional[int]
    severity: str
    rule_id: Optional[str]
    rule_title: Optional[str]
    phash: Optional[str]
    message_link: Optional[str]
    created_at: Optional[str]
    captured_at: Optional[str]
    payload_json: str
    metrics_json: str
    eval_ms: Optional[float]
    winning_origin: Optional[str]
    notified_at: Optional[str]
    pipeline_run_id: str
    store_run_id: str
    ingested_at: str
    updated_at: str


@dataclass(slots=True)
class StoreRun:
    run_id: str
    pipeline_run_id: str
    profile: str
    partition_date: str
    started_at: str
    finished_at: str
    status: str
    attempts: int
    records_total: int
    records_upserted: int
    records_skipped: int
    duration_sec: float
    retry_count: int
    db_size_bytes: int
    log_path: Optional[str]
    metrics_json: Optional[str]
    created_at: str


@dataclass(slots=True)
class StoreRunMetrics:
    store_run_id: str
    pipeline_run_id: str
    profile: str
    partition_date: str
    started_at: datetime
    finished_at: datetime
    status: str
    attempts: int
    records_total: int
    records_upserted: int
    records_skipped: int
    duration_sec: float
    retry_count: int
    db_size_bytes: int
    log_path: Optional[str] = None
    extra_metrics: dict[str, Any] = field(default_factory=dict)
