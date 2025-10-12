from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from app.config import get_settings
from app.notify import (
    HttpNotificationTransport,
    NotificationConfig,
    NotificationRunner,
    load_notification_config,
    load_notification_state,
    save_notification_state,
)
from app.profiles import ContextPaths, ProfileContext


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


class NotificationLock:
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
                raise SystemExit(f"notification runner already active (pid={existing_pid})")
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

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send Discord notifications for pipeline findings")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run a single iteration and exit")
    mode.add_argument("--loop", action="store_true", help="Run continuously at the specified interval")
    parser.add_argument("--profile", type=str, help="Profile override")
    parser.add_argument("--date", type=str, help="Partition date override (YYYY-MM-DD)")
    parser.add_argument("--config", type=Path, default=Path("configs/notifications.yaml"))
    parser.add_argument("--state-file", type=Path, default=Path("tmp/notification_state.json"))
    parser.add_argument("--lock-file", type=Path, default=Path("/tmp/modbot_notifications.lock"))
    parser.add_argument("--interval-minutes", type=int, help="Loop interval override in minutes")
    parser.add_argument("--max-runs", type=int, help="Maximum runs when looping")
    parser.add_argument("--dry-run", action="store_true", help="Generate payloads without sending")
    parser.add_argument("--only", action="append", choices=["findings", "alerts", "summary", "store"], help="Limit notification categories")
    parser.add_argument("--log-file", type=Path, help="Optional JSON log output file")
    parser.add_argument("--quiet", action="store_true", help="Only emit WARN/ERROR logs to stdout")
    parser.add_argument("--verbose", action="store_true", help="Emit DEBUG logs to stdout")
    return parser.parse_args()


def _resolve_interval(args: argparse.Namespace, config: NotificationConfig) -> int:
    if args.interval_minutes:
        return max(1, args.interval_minutes)
    env_value = os.getenv("NOTIFY_INTERVAL_MINUTES")
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            pass
    return max(1, int(config.defaults.alert_cooldown_minutes))


def _build_context(settings, profile: Optional[str], date: Optional[str]) -> tuple[ProfileContext, ContextPaths]:
    context = settings.build_profile_context(profile=profile, date=date)
    paths = ContextPaths.for_context(context)
    return context, paths


def run_once(
    *,
    args: argparse.Namespace,
    settings,
    config: NotificationConfig,
    state_path: Path,
    logger: JsonLogger,
) -> bool:
    context, paths = _build_context(settings, args.profile, args.date)
    state = load_notification_state(state_path)
    transport = HttpNotificationTransport(
        bot_token=settings.discord_bot_token,
        max_retries=config.defaults.max_retries,
    )
    runner = NotificationRunner(
        config=config,
        state=state,
        state_path=state_path,
        context=context,
        paths=paths,
        logger=logger,
        transport=transport,
        findings_store_path=settings.findings_db_path,
    )
    try:
        success = runner.run(only=args.only or None, dry_run=args.dry_run)
    finally:
        runner.close()
        if not args.dry_run:
            save_notification_state(state_path, state)
    return success


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
        config = load_notification_config(args.config)
        state_path = args.state_file
        interval = _resolve_interval(args, config)

        lock = NotificationLock(args.lock_file, logger)

        def handle_signal(signum, frame) -> None:  # pragma: no cover
            logger.warn("received termination signal", signal=signum)
            lock.release()
            logger.close()
            sys.exit(0)

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        with lock:
            runs = 0
            if args.once or not args.loop:
                success = run_once(
                    args=args,
                    settings=settings,
                    config=config,
                    state_path=state_path,
                    logger=logger,
                )
                if not success:
                    sys.exit(1)
                return

            while True:
                start = time.monotonic()
                success = run_once(
                    args=args,
                    settings=settings,
                    config=config,
                    state_path=state_path,
                    logger=logger,
                )
                runs += 1
                if args.max_runs and runs >= args.max_runs:
                    break
                elapsed = time.monotonic() - start
                sleep_for = max(0.0, interval * 60 - elapsed)
                time.sleep(sleep_for)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
