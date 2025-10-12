from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from app.config import get_settings
from app.profiles import ContextPaths, ProfileContext, PartitionPaths
from app.store import FindingsStore, StoreRunMetrics


class JsonLogger:
    _LEVELS: dict[str, int] = {"debug": 10, "info": 20, "warn": 30, "error": 40}

    def __init__(
        self,
        *,
        stdout_level: str = "info",
        file_path: Path | None = None,
        file_level: str | None = None,
    ) -> None:
        self.stdout_level = self._LEVELS.get(stdout_level, 20)
        self.file_level = self._LEVELS.get(file_level or stdout_level, 20)
        self.file_handle = None
        if file_path:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_handle = file_path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def _should_emit(self, level: str, target: str) -> bool:
        numeric = self._LEVELS.get(level, 20)
        threshold = self.stdout_level if target == "stdout" else self.file_level
        return numeric >= threshold

    def _serialize(self, payload: dict[str, Any]) -> str:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, datetime):
                return value.astimezone(timezone.utc).isoformat()
            if isinstance(value, Exception):
                return repr(value)
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]
            return value

        converted = {key: convert(val) for key, val in payload.items()}
        return json.dumps(converted, ensure_ascii=False)

    def _emit(self, stream, level: str, message: str, **fields: Any) -> None:
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level.upper(),
            "message": message,
        }
        record.update(fields)
        stream.write(self._serialize(record) + "\n")
        stream.flush()

    def log(self, level: str, message: str, **fields: Any) -> None:
        if self._should_emit(level, "stdout"):
            self._emit(sys.stdout, level, message, **fields)
        if self.file_handle and self._should_emit(level, "file"):
            self._emit(self.file_handle, level, message, **fields)

    def debug(self, message: str, **fields: Any) -> None:
        self.log("debug", message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        self.log("info", message, **fields)

    def warn(self, message: str, **fields: Any) -> None:
        self.log("warn", message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self.log("error", message, **fields)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync p3 findings into persistent store")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run a single iteration and exit")
    mode.add_argument("--loop", action="store_true", help="Run continuously at the specified interval")
    parser.add_argument("--profile", type=str, help="Profile override")
    parser.add_argument("--date", type=str, help="Partition date override (YYYY-MM-DD)")
    parser.add_argument("--pipeline-run-id", type=str, help="Pipeline run identifier (ISO string)")
    parser.add_argument("--metrics-file", type=Path, help="Metrics JSONL output path override")
    parser.add_argument("--state-file", type=Path, default=Path("tmp/store_state.json"))
    parser.add_argument("--lock-file", type=Path, default=Path("/tmp/modbot_store.lock"))
    parser.add_argument("--interval-minutes", type=int, help="Loop interval override in minutes")
    parser.add_argument("--max-runs", type=int, help="Maximum runs when looping")
    parser.add_argument("--dry-run", action="store_true", help="Process without committing DB changes")
    parser.add_argument("--force", action="store_true", help="Process even if findings hash is unchanged")
    parser.add_argument("--log-file", type=Path, help="Optional JSON log output file")
    parser.add_argument("--quiet", action="store_true", help="Only emit WARN/ERROR logs to stdout")
    parser.add_argument("--verbose", action="store_true", help="Emit DEBUG logs to stdout")
    return parser.parse_args()


@dataclass
class StoreState:
    profile: Optional[str] = None
    partition_date: Optional[str] = None
    last_pipeline_run_id: Optional[str] = None
    last_findings_hash: Optional[str] = None
    last_run_id: Optional[str] = None
    failure_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "partition_date": self.partition_date,
            "last_pipeline_run_id": self.last_pipeline_run_id,
            "last_findings_hash": self.last_findings_hash,
            "last_run_id": self.last_run_id,
            "failure_count": self.failure_count,
        }


def load_state(path: Path) -> StoreState:
    if not path.exists():
        return StoreState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        backup = path.with_suffix(path.suffix + ".bak")
        if backup.exists():
            return load_state(backup)
        return StoreState()
    if not isinstance(data, dict):
        return StoreState()
    state = StoreState()
    state.profile = data.get("profile")
    state.partition_date = data.get("partition_date")
    state.last_pipeline_run_id = data.get("last_pipeline_run_id")
    state.last_findings_hash = data.get("last_findings_hash")
    state.last_run_id = data.get("last_run_id")
    try:
        state.failure_count = int(data.get("failure_count", 0))
    except (TypeError, ValueError):
        state.failure_count = 0
    return state


def save_state(path: Path, state: StoreState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    backup = path.with_suffix(path.suffix + ".bak")
    tmp.write_text(payload, encoding="utf-8")
    if path.exists():
        try:
            path.replace(backup)
        except OSError:
            pass
    tmp.replace(path)


class StoreLock:
    def __init__(self, path: Path, logger: JsonLogger) -> None:
        self.path = path
        self.logger = logger
        self._acquired = False

    def acquire(self) -> None:
        if self._acquired:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                existing_pid = int(self.path.read_text(encoding="utf-8").strip())
            except ValueError:
                existing_pid = None
            if existing_pid and os.path.exists(f"/proc/{existing_pid}"):
                raise SystemExit(f"store sync already running (pid={existing_pid})")
            self.logger.warn("stale lock detected; removing", pid=existing_pid, path=self.path)
            try:
                self.path.unlink()
            except OSError:
                pass
        self.path.write_text(str(os.getpid()), encoding="utf-8")
        self._acquired = True

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            self.path.unlink()
        except OSError:
            pass
        self._acquired = False

    def __enter__(self) -> "StoreLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.release()


def _resolve_interval_minutes(args: argparse.Namespace) -> int:
    if args.interval_minutes:
        return max(1, args.interval_minutes)
    env_value = os.getenv("STORE_INTERVAL_MINUTES")
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            pass
    return 15


def _build_context(settings, profile: Optional[str], date: Optional[str]) -> tuple[ProfileContext, ContextPaths]:
    context = settings.build_profile_context(profile=profile, date=date)
    paths = ContextPaths.for_context(context)
    return context, paths


def _compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _stream_findings(path: Path, logger: JsonLogger) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                logger.warn("invalid json record", line=line_no, error=str(exc))


def _append_metrics_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_latest_pipeline_run(paths: PartitionPaths, logger: JsonLogger) -> Optional[str]:
    metrics_path = paths.pipeline_metrics_path()
    if not metrics_path.exists():
        logger.warn("pipeline_metrics_missing", path=metrics_path)
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except OSError as exc:
        logger.warn("pipeline_metrics_unreadable", path=metrics_path, error=str(exc))
        return None
    if not lines:
        return None
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        logger.warn("pipeline_metrics_corrupt", path=metrics_path)
        return None
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return None
    return run_id


async def run_once(
    *,
    args: argparse.Namespace,
    settings,
    state_path: Path,
    state: StoreState,
    logger: JsonLogger,
) -> bool:
    context, paths = _build_context(settings, args.profile, args.date)
    findings_path = paths.stage_file("p3")
    if not findings_path.exists():
        logger.warn("findings_missing", path=findings_path)
        return False

    pipeline_run_id = args.pipeline_run_id or _read_latest_pipeline_run(paths.partition_paths, logger) or state.last_pipeline_run_id
    if not pipeline_run_id:
        logger.error("pipeline_run_id_missing")
        return False

    metrics_path = args.metrics_file or paths.partition_paths.metrics_file("store", ensure_parent=True)
    findings_hash = _compute_sha256(findings_path)
    if findings_hash == state.last_findings_hash and not args.force:
        logger.info("store_skip_no_changes", profile=context.profile, date=context.iso_date, pipeline_run_id=pipeline_run_id)
        return True

    store_run_started = datetime.now(timezone.utc).replace(microsecond=0)
    store_run_id = f"store-{context.profile}-{store_run_started.isoformat()}"
    logger.info(
        "store_sync_start",
        profile=context.profile,
        date=context.iso_date,
        pipeline_run_id=pipeline_run_id,
        store_run_id=store_run_id,
        findings_path=findings_path,
    )

    store = FindingsStore(settings.findings_db_path)
    await store.connect()
    loop = asyncio.get_running_loop()
    start_monotonic = loop.time()
    success = False
    try:
        stats = await store.upsert_findings(
            _stream_findings(findings_path, logger),
            profile=context.profile,
            partition_date=context.iso_date,
            pipeline_run_id=pipeline_run_id,
            store_run_id=store_run_id,
            logger=logger,
            dry_run=args.dry_run,
        )
        finished_at = datetime.now(timezone.utc).replace(microsecond=0)
        duration = max(0.0, loop.time() - start_monotonic)
        db_size = settings.findings_db_path.stat().st_size if settings.findings_db_path.exists() else 0
        metrics = StoreRunMetrics(
            store_run_id=store_run_id,
            pipeline_run_id=pipeline_run_id,
            profile=context.profile,
            partition_date=context.iso_date,
            started_at=store_run_started,
            finished_at=finished_at,
            status="success",
            attempts=1,
            records_total=stats.total,
            records_upserted=stats.inserted + stats.updated,
            records_skipped=stats.skipped,
            duration_sec=round(duration, 3),
            retry_count=state.failure_count if state.failure_count > 0 else 0,
            db_size_bytes=db_size,
            log_path=str(args.log_file) if args.log_file else None,
            extra_metrics={
                "inserted": stats.inserted,
                "updated": stats.updated,
                "findings_hash": findings_hash,
            },
        )

        if not args.dry_run:
            await store.record_run(metrics=metrics)
            _append_metrics_jsonl(
                metrics_path,
                {
                    "store_run_id": store_run_id,
                    "pipeline_run_id": pipeline_run_id,
                    "profile": context.profile,
                    "partition_date": context.iso_date,
                    "started_at": metrics.started_at.astimezone(timezone.utc).isoformat(),
                    "finished_at": metrics.finished_at.astimezone(timezone.utc).isoformat(),
                    "duration_sec": metrics.duration_sec,
                    "records_total": stats.total,
                    "records_inserted": stats.inserted,
                    "records_updated": stats.updated,
                    "records_skipped": stats.skipped,
                    "findings_hash": findings_hash,
                    "db_size_bytes": db_size,
                },
            )
            state.profile = context.profile
            state.partition_date = context.iso_date
            state.last_pipeline_run_id = pipeline_run_id
            state.last_findings_hash = findings_hash
            state.last_run_id = store_run_id
            state.failure_count = 0
            save_state(state_path, state)
        else:
            logger.info(
                "store_dry_run",
                records_total=stats.total,
                inserted=stats.inserted,
                updated=stats.updated,
                skipped=stats.skipped,
            )

        logger.info(
            "store_sync_complete",
            profile=context.profile,
            date=context.iso_date,
            store_run_id=store_run_id,
            inserted=stats.inserted,
            updated=stats.updated,
            skipped=stats.skipped,
            duration_sec=metrics.duration_sec,
            dry_run=args.dry_run,
        )
        success = True
    except Exception as exc:  # noqa: BLE001
        logger.error("store_sync_error", error=str(exc))
        success = False
    finally:
        await store.close()
    return success


async def run_loop(
    *,
    args: argparse.Namespace,
    settings,
    state_path: Path,
    state: StoreState,
    logger: JsonLogger,
) -> None:
    interval_minutes = _resolve_interval_minutes(args)
    runs = 0
    while True:
        success = await run_once(args=args, settings=settings, state_path=state_path, state=state, logger=logger)
        if not success and not args.dry_run:
            state.failure_count += 1
            save_state(state_path, state)
        runs += 1
        if args.max_runs and runs >= args.max_runs:
            break
        sleep_seconds = max(0.0, interval_minutes * 60)
        if sleep_seconds > 0:
            logger.debug("store_loop_sleep", seconds=sleep_seconds)
            await asyncio.sleep(sleep_seconds)


def main() -> None:
    args = parse_args()
    log_level = "info"
    if args.verbose:
        log_level = "debug"
    if args.quiet:
        log_level = "warn"

    logger = JsonLogger(stdout_level=log_level, file_path=args.log_file)
    try:
        settings = get_settings()
        state_path = args.state_file
        state = load_state(state_path)
        lock = StoreLock(args.lock_file, logger)

        def handle_signal(signum, frame) -> None:  # noqa: ANN001
            logger.warn("store_sync_signal", signal=signum)
            lock.release()
            sys.exit(0)

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        async def runner() -> None:
            with lock:
                if args.loop and not args.once:
                    await run_loop(args=args, settings=settings, state_path=state_path, state=state, logger=logger)
                else:
                    success = await run_once(
                        args=args,
                        settings=settings,
                        state_path=state_path,
                        state=state,
                        logger=logger,
                    )
                    if not success and not args.dry_run:
                        state.failure_count += 1
                        save_state(state_path, state)

        asyncio.run(runner())
    finally:
        logger.close()


if __name__ == "__main__":  # pragma: no cover
    main()
