from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from app.config import get_settings
from app.profiles import ContextPaths, ProfileContext, PartitionPaths


# --------------------------------------------------------------------------- #
# Logging utilities


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

    def _emit(self, stream: Any, level: str, message: str, **fields: Any) -> None:
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


# --------------------------------------------------------------------------- #
# Configuration structures


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


@dataclass
class StageOptions:
    enabled: bool = True
    retries: int = 2
    timeout_sec: Optional[int] = None
    cli_args: dict[str, Any] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)


@dataclass
class P0Options(StageOptions):
    resume: bool = True
    since_fallback_minutes: int = 1440


@dataclass
class ReportOptions(StageOptions):
    rules: Optional[str] = None


@dataclass
class PipelineConfig:
    interval_minutes: Optional[int] = None
    p0: P0Options = field(default_factory=P0Options)
    p1: StageOptions = field(default_factory=StageOptions)
    p2: StageOptions = field(default_factory=StageOptions)
    p3: StageOptions = field(default_factory=StageOptions)
    store: StageOptions = field(default_factory=StageOptions)
    report: ReportOptions = field(default_factory=ReportOptions)


CONTROL_KEYS = {"enabled", "retries", "timeout_sec", "extra_args"}
P0_CONTROL_KEYS = CONTROL_KEYS | {"resume", "since_fallback_minutes"}
REPORT_CONTROL_KEYS = CONTROL_KEYS | {"rules"}


def _split_stage_options(
    data: dict[str, Any],
    *,
    control_keys: set[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    controls: dict[str, Any] = {}
    cli_args: dict[str, Any] = {}
    for key, value in data.items():
        if key in control_keys:
            controls[key] = value
        else:
            cli_args[key] = value
    return controls, cli_args


def _to_cli_args(options: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in options.items():
        flag = f"--{key.replace('_', '-')}"
        result[flag] = value
    return result


def load_pipeline_config(path: Path, logger: JsonLogger) -> PipelineConfig:
    if not path.exists():
        logger.warn("pipeline config not found; using defaults", path=path)
        return PipelineConfig()

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SystemExit(f"Failed to parse pipeline config: {exc}") from exc

    if raw is None:
        return PipelineConfig()
    if not isinstance(raw, dict):
        raise SystemExit("Pipeline config must be a mapping at top level.")

    config = PipelineConfig()
    unknown_keys = set(raw.keys()) - {"interval_minutes", "p0", "p1", "p2", "p3", "store", "report"}
    if unknown_keys:
        logger.warn("unknown keys in pipeline config", keys=sorted(unknown_keys))

    config.interval_minutes = raw.get("interval_minutes")

    def parse_stage(
        section: str,
        defaults: StageOptions,
        *,
        control_keys: set[str],
    ) -> StageOptions:
        payload = raw.get(section, {})
        if payload is None:
            return defaults
        if not isinstance(payload, dict):
            logger.warn("stage config must be a mapping", stage=section)
            return defaults

        controls, cli_params = _split_stage_options(payload, control_keys=control_keys)
        stage = defaults
        if "enabled" in controls:
            stage.enabled = bool(controls["enabled"])
        if "retries" in controls:
            try:
                stage.retries = int(controls["retries"])
            except (TypeError, ValueError):
                logger.warn("invalid retries value; using default", stage=section)
        if "timeout_sec" in controls:
            try:
                stage.timeout_sec = int(controls["timeout_sec"])
            except (TypeError, ValueError):
                logger.warn("invalid timeout_sec value; ignoring", stage=section)
        if "extra_args" in controls:
            stage.extra_args = _ensure_list(controls["extra_args"])
        stage.cli_args.update(_to_cli_args(cli_params))
        return stage

    config.p0 = parse_stage("p0", config.p0, control_keys=P0_CONTROL_KEYS)
    if "resume" in raw.get("p0", {}):
        config.p0.resume = bool(raw["p0"]["resume"])
    if "since_fallback_minutes" in raw.get("p0", {}):
        try:
            config.p0.since_fallback_minutes = int(raw["p0"]["since_fallback_minutes"])
        except (TypeError, ValueError):
            logger.warn("invalid since_fallback_minutes; keeping default")

    config.p1 = parse_stage("p1", config.p1, control_keys=CONTROL_KEYS)
    config.p2 = parse_stage("p2", config.p2, control_keys=CONTROL_KEYS)
    config.p3 = parse_stage("p3", config.p3, control_keys=CONTROL_KEYS)
    config.store = parse_stage("store", config.store, control_keys=CONTROL_KEYS)

    report_defaults = config.report
    payload_report = raw.get("report", {})
    if payload_report is None:
        payload_report = {}
    if not isinstance(payload_report, dict):
        logger.warn("report config must be a mapping")
        payload_report = {}
    controls_report, cli_params_report = _split_stage_options(
        payload_report,
        control_keys=REPORT_CONTROL_KEYS,
    )
    if "enabled" in controls_report:
        report_defaults.enabled = bool(controls_report["enabled"])
    if "retries" in controls_report:
        try:
            report_defaults.retries = int(controls_report["retries"])
        except (TypeError, ValueError):
            logger.warn("invalid retries for report; using default")
    if "timeout_sec" in controls_report:
        try:
            report_defaults.timeout_sec = int(controls_report["timeout_sec"])
        except (TypeError, ValueError):
            logger.warn("invalid timeout_sec for report; ignoring")
    if "extra_args" in controls_report:
        report_defaults.extra_args = _ensure_list(controls_report["extra_args"])
    if "rules" in controls_report:
        value = controls_report["rules"]
        report_defaults.rules = str(value) if value else None
    report_defaults.cli_args.update(_to_cli_args(cli_params_report))
    config.report = report_defaults

    return config


# --------------------------------------------------------------------------- #
# State helpers


@dataclass
class PipelineState:
    profile: Optional[str] = None
    last_started_at: Optional[datetime] = None
    last_completed_at: Optional[datetime] = None
    last_status: str = "idle"
    last_run_id: Optional[str] = None
    failure_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "last_started_at": _dt_to_iso(self.last_started_at),
            "last_completed_at": _dt_to_iso(self.last_completed_at),
            "last_status": self.last_status,
            "last_run_id": self.last_run_id,
            "failure_count": self.failure_count,
        }


def _state_backup_path(path: Path) -> Path:
    if path.suffix:
        return path.with_suffix(path.suffix + ".bak")
    return path.with_name(path.name + ".bak")


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _dt_to_iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def load_state(path: Path, logger: JsonLogger) -> PipelineState:
    if not path.exists():
        return PipelineState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        backup = _state_backup_path(path)
        if backup.exists():
            logger.warn("state file corrupt; trying backup", path=path, backup=backup)
            return load_state(backup, logger)
        logger.warn("state file corrupt and no backup; starting fresh", path=path)
        return PipelineState()
    if not isinstance(data, dict):
        logger.warn("state file malformed; ignoring", path=path)
        return PipelineState()
    state = PipelineState()
    state.profile = data.get("profile")
    state.last_started_at = _parse_dt(data.get("last_started_at"))
    state.last_completed_at = _parse_dt(data.get("last_completed_at"))
    state.last_status = data.get("last_status") or "idle"
    state.last_run_id = data.get("last_run_id")
    try:
        state.failure_count = int(data.get("failure_count", 0))
    except (TypeError, ValueError):
        state.failure_count = 0
    return state


def save_state(path: Path, state: PipelineState, logger: JsonLogger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = _state_backup_path(path)
        try:
            shutil.copy2(path, backup)
        except OSError as exc:
            logger.warn("failed to write state backup", error=str(exc), path=path, backup=backup)
    payload = json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
    path.write_text(payload, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Lock handling


class PipelineLock:
    def __init__(self, path: Path, logger: JsonLogger) -> None:
        self.path = path
        self.logger = logger
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._acquired = False

    def acquire(self) -> None:
        if self._acquired:
            return
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                if self._is_stale():
                    continue
                raise SystemExit(f"Lock file exists and process is running: {self.path}")
            else:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(str(os.getpid()))
                self._acquired = True
                return

    def _is_stale(self) -> bool:
        try:
            text = self.path.read_text(encoding="utf-8").strip()
            pid = int(text)
        except (OSError, ValueError):
            self.logger.warn("invalid lock file contents; removing", path=self.path)
            self.path.unlink(missing_ok=True)
            return True
        try:
            os.kill(pid, 0)
        except OSError:
            self.logger.warn("stale lock detected; removing", pid=pid, path=self.path)
            self.path.unlink(missing_ok=True)
            return True
        else:
            return False

    def release(self) -> None:
        if self._acquired:
            try:
                self.path.unlink(missing_ok=True)
            finally:
                self._acquired = False

    def __enter__(self) -> "PipelineLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


# --------------------------------------------------------------------------- #
# Runtime helpers


def _isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _build_stage_command(
    *,
    module: str,
    profile: str,
    date_token: str,
    args_map: dict[str, Any],
    extra_args: Iterable[str],
    include_profile: bool = True,
) -> list[str]:
    cmd = [sys.executable, "-m", module]
    if include_profile:
        cmd += ["--profile", profile]
    cmd += ["--date", date_token]
    for flag, value in args_map.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        cmd.append(flag)
        cmd.append(str(value))
    cmd.extend(str(arg) for arg in extra_args)
    return cmd


def count_records(path: Path, *, kind: str) -> Optional[int]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            if kind == "csv-with-header":
                lines = sum(1 for _ in handle)
                return max(lines - 1, 0)
            return sum(1 for _ in handle)
    except OSError:
        return None


def emit_alert(payload: dict[str, Any], logger: JsonLogger) -> None:
    logger.warn("alert placeholder", alert=payload)


# --------------------------------------------------------------------------- #
# Main runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automate p0→p3 pipeline execution")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run a single pipeline iteration and exit")
    mode.add_argument("--loop", action="store_true", help="Run continuously at the specified interval")
    parser.add_argument("--profile", type=str, help="Profile name override")
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline_defaults.yaml"))
    parser.add_argument("--state-file", type=Path, default=Path("tmp/pipeline_state.json"))
    parser.add_argument("--lock-file", type=Path, default=Path("/tmp/modbot_pipeline.lock"))
    parser.add_argument("--interval-minutes", type=int, help="Interval override in minutes")
    parser.add_argument("--max-runs", type=int, help="Maximum iterations when looping")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing")
    parser.add_argument("--log-file", type=Path, help="Optional path to append JSON logs")
    parser.add_argument("--quiet", action="store_true", help="Only emit WARN/ERROR logs to stdout")
    parser.add_argument("--verbose", action="store_true", help="Emit DEBUG logs to stdout")
    parser.add_argument("--date", type=str, help="Optional partition date override (YYYY-MM-DD)")
    return parser.parse_args()


def _resolve_interval_minutes(args: argparse.Namespace, config: PipelineConfig) -> int:
    if args.interval_minutes:
        return max(1, args.interval_minutes)
    env_value = os.getenv("PIPELINE_INTERVAL_MINUTES")
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            pass
    if config.interval_minutes:
        return max(1, int(config.interval_minutes))
    return 15


def _resolve_profile_context(
    *,
    settings,
    profile_arg: Optional[str],
    date_arg: Optional[str],
    last_completed: Optional[datetime],
) -> ProfileContext:
    context = settings.build_profile_context(profile=profile_arg, date=date_arg)
    if last_completed is None:
        return context
    target_date = last_completed.astimezone(context.tzinfo).date()
    return context.with_date(date_token=target_date.isoformat())


def _build_since_value(now: datetime, state: PipelineState, config: PipelineConfig) -> datetime:
    if state.last_completed_at:
        return state.last_completed_at
    minutes = max(1, config.p0.since_fallback_minutes)
    return now - timedelta(minutes=minutes)


def run_once(
    *,
    args: argparse.Namespace,
    settings,
    config: PipelineConfig,
    logger: JsonLogger,
    state_path: Path,
    dry_run: bool,
) -> bool:
    """Execute a single pipeline iteration. Returns True if successful."""

    state = load_state(state_path, logger)
    context = _resolve_profile_context(
        settings=settings,
        profile_arg=args.profile,
        date_arg=args.date,
        last_completed=state.last_completed_at,
    )
    partitions = PartitionPaths(context)
    context_paths = ContextPaths.for_context(context)
    now = datetime.now(timezone.utc)
    run_id = _isoformat_utc(now)
    since_dt = _build_since_value(now, state, config)
    since_iso = _isoformat_utc(since_dt)
    date_token = context.date.isoformat()

    logger.info(
        "pipeline iteration start",
        profile=context.profile,
        date=date_token,
        since=since_iso,
        run_id=run_id,
        dry_run=dry_run,
    )

    if not dry_run:
        state.profile = context.profile
        state.last_started_at = now
        state.last_status = "running"
        state.last_run_id = run_id
        save_state(state_path, state, logger)

    stage_metrics: dict[str, dict[str, Any]] = {}
    stages = [
        ("p0", config.p0),
        ("p1", config.p1),
        ("p2", config.p2),
        ("p3", config.p3),
        ("store", config.store),
        ("report", config.report),
    ]

    def stage_enabled(options: StageOptions) -> bool:
        return getattr(options, "enabled", True)

    success = True
    for stage_name, stage_config in stages:
        if not stage_enabled(stage_config):
            stage_metrics[stage_name] = {
                "status": "skipped",
                "duration_sec": None,
                "records": None,
                "returncode": None,
                "last_error": None,
            }
            logger.info("stage skipped", stage=stage_name)
            continue

        if dry_run:
            # Build command for logging purposes
            cmd, output_path = build_stage_command(
                stage_name=stage_name,
                stage_config=stage_config,
                partitions=partitions,
                context_paths=context_paths,
                profile=context.profile,
                date_token=date_token,
                since_iso=since_iso,
                run_id=run_id,
            )
            logger.info("dry-run stage", stage=stage_name, command=cmd)
            stage_metrics[stage_name] = {
                "status": "dry-run",
                "duration_sec": None,
                "records": None,
                "returncode": None,
                "last_error": None,
            }
            continue

        attempts = 0
        completed = False
        last_error: Optional[str] = None
        last_returncode: Optional[int] = None
        backoff = 30
        start_monotonic = time.monotonic()
        while attempts < max(1, stage_config.retries):
            attempts += 1
            cmd, output_path = build_stage_command(
                stage_name=stage_name,
                stage_config=stage_config,
                partitions=partitions,
                context_paths=context_paths,
                profile=context.profile,
                date_token=date_token,
                since_iso=since_iso,
                run_id=run_id,
            )
            logger.info("stage command start", stage=stage_name, attempt=attempts, command=cmd)
            try:
                completed, attempt_error, attempt_returncode = execute_command(
                    cmd,
                    timeout=stage_config.timeout_sec,
                    logger=logger,
                    stage=stage_name,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("stage execution raised exception", stage=stage_name, error=str(exc))
                completed = False
                attempt_error = str(exc)
                attempt_returncode = None
            last_error = attempt_error
            last_returncode = attempt_returncode
            if completed:
                break
            if attempts < stage_config.retries:
                logger.warn("stage failed; retrying with backoff", stage=stage_name, attempt=attempts, sleep=backoff)
                time.sleep(backoff)
                backoff *= 2

        duration = time.monotonic() - start_monotonic
        if completed:
            records_kind = "csv-with-header" if stage_name in {"p0", "report"} else "jsonl"
            record_count = count_records(output_path, kind=records_kind)
            stage_metrics[stage_name] = {
                "status": "success",
                "duration_sec": round(duration, 3),
                "records": record_count,
                "attempts": attempts,
                "returncode": last_returncode if last_returncode is not None else 0,
                "last_error": None,
            }
            logger.info(
                "stage completed",
                stage=stage_name,
                duration_sec=round(duration, 3),
                records=record_count,
                attempts=attempts,
            )
        else:
            stage_metrics[stage_name] = {
                "status": "failed",
                "duration_sec": round(duration, 3),
                "records": None,
                "attempts": attempts,
                "returncode": last_returncode,
                "last_error": last_error,
            }
            logger.error("stage failed after retries", stage=stage_name, attempts=attempts)
            success = False
            break
    for stage_name, _stage_config in stages:
        stage_metrics.setdefault(
            stage_name,
            {"status": "pending", "duration_sec": None, "records": None, "returncode": None, "last_error": None},
        )

    finished_at = datetime.now(timezone.utc)
    if not dry_run:
        if success:
            state.last_completed_at = finished_at
            state.last_status = "success"
            state.failure_count = 0
        else:
            state.last_status = "failed"
            state.failure_count += 1
        save_state(state_path, state, logger)
        append_run_metrics(
            partitions=partitions,
            date_token=date_token,
            run_id=run_id,
            started_at=now,
            finished_at=finished_at,
            stage_metrics=stage_metrics,
            logger=logger,
        )

    if not success and not dry_run:
        emit_alert(
            {
                "stage_metrics": stage_metrics,
                "profile": context.profile,
                "date": date_token,
                "run_id": run_id,
            },
            logger,
        )
    return success


def build_stage_command(
    *,
    stage_name: str,
    stage_config: StageOptions,
    partitions: PartitionPaths,
    context_paths: ContextPaths,
    profile: str,
    date_token: str,
    since_iso: str,
    run_id: str | None = None,
) -> tuple[list[str], Path]:
    args_map = dict(stage_config.cli_args)
    extra_args = stage_config.extra_args

    if stage_name == "p0":
        args_map.setdefault("--resume", getattr(stage_config, "resume", True))
        args_map["--since"] = since_iso
        cmd = _build_stage_command(
            module="app.p0_scan",
            profile=profile,
            date_token=date_token,
            args_map=args_map,
            extra_args=extra_args,
        )
        output_path = context_paths.stage_file("p0", ensure_parent=True)
        return cmd, output_path

    if stage_name == "p1":
        if "--merge-existing" not in args_map:
            args_map["--merge-existing"] = True
        cmd = _build_stage_command(
            module="app.cli_wd14",
            profile=profile,
            date_token=date_token,
            args_map=args_map,
            extra_args=extra_args,
        )
        output_path = context_paths.stage_file("p1", ensure_parent=True)
        return cmd, output_path

    if stage_name == "p2":
        if "--merge-existing" not in args_map:
            args_map["--merge-existing"] = True
        cmd = _build_stage_command(
            module="app.analysis_merge",
            profile=profile,
            date_token=date_token,
            args_map=args_map,
            extra_args=extra_args,
        )
        output_path = context_paths.stage_file("p2", ensure_parent=True)
        return cmd, output_path

    if stage_name == "p3":
        if "--merge-existing" not in args_map:
            args_map["--merge-existing"] = True
        cmd = _build_stage_command(
            module="app.cli_scan",
            profile=profile,
            date_token=date_token,
            args_map=args_map,
            extra_args=extra_args,
        )
        output_path = context_paths.stage_file("p3", ensure_parent=True)
        return cmd, output_path

    if stage_name == "store":
        if run_id is None:
            raise ValueError("run_id is required for store stage")
        args_map.setdefault("--once", True)
        args_map.setdefault("--pipeline-run-id", run_id)
        metrics_path = partitions.metrics_file("store", ensure_parent=True)
        args_map.setdefault("--metrics-file", metrics_path)
        cmd = _build_stage_command(
            module="scripts.run_store_sync",
            profile=profile,
            date_token=date_token,
            args_map=args_map,
            extra_args=extra_args,
        )
        return cmd, metrics_path

    if stage_name == "report":
        if isinstance(stage_config, ReportOptions) and stage_config.rules:
            args_map.setdefault("--rules", stage_config.rules)
        cmd = _build_stage_command(
            module="app.cli_report",
            profile=profile,
            date_token=date_token,
            args_map=args_map,
            extra_args=extra_args,
        )
        output_path = partitions.report_path(ensure_parent=True)
        return cmd, output_path

    raise ValueError(f"Unknown stage: {stage_name}")


def _format_stderr_tail(data: bytes, limit: int = 1024) -> tuple[Optional[str], str]:
    if not data:
        return None, "utf-8"
    tail = data[-limit:] if limit > 0 else data
    try:
        text = tail.decode("utf-8")
        return text, "utf-8"
    except UnicodeDecodeError:
        encoded = base64.b64encode(tail).decode("ascii")
        return f"base64:{encoded}", "base64"


def execute_command(
    cmd: list[str],
    *,
    timeout: Optional[int],
    logger: JsonLogger,
    stage: str,
) -> tuple[bool, Optional[str], Optional[int]]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stderr_bytes = exc.stderr or b""
        tail, _ = _format_stderr_tail(stderr_bytes)
        logger.error(
            "stage command timed out",
            stage=stage,
            timeout_sec=timeout,
            stderr=tail,
        )
        return False, tail or "timeout", None
    stdout_text = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
    stderr_text = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
    if stdout_text:
        logger.debug("stage stdout", stage=stage, stdout=stdout_text)
    if stderr_text:
        logger.debug("stage stderr", stage=stage, stderr=stderr_text)
    if proc.returncode != 0:
        tail, _ = _format_stderr_tail(proc.stderr or b"")
        logger.warn("stage command failed", stage=stage, returncode=proc.returncode, stderr=tail)
        return False, tail, proc.returncode
    return True, None, proc.returncode


def append_run_metrics(
    *,
    partitions: PartitionPaths,
    date_token: str,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    stage_metrics: dict[str, dict[str, Any]],
    logger: JsonLogger,
) -> None:
    profile_root = partitions.profile_root(ensure=True)
    metrics_dir = profile_root / "metrics" / "pipeline"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"pipeline_{date_token}.jsonl"
    payload = {
        "run_id": run_id,
        "started_at": _isoformat_utc(started_at),
        "finished_at": _isoformat_utc(finished_at),
        "profile": partitions.context.profile,
        "stages": stage_metrics,
    }
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    logger.debug("appended pipeline metrics", path=metrics_path)


def main() -> None:
    args = parse_args()
    logger_level = "info"
    if args.verbose:
        logger_level = "debug"
    if args.quiet:
        logger_level = "warn"

    logger = JsonLogger(stdout_level=logger_level, file_path=args.log_file)
    try:
        config = load_pipeline_config(args.config, logger)
        settings = get_settings()
        interval_minutes = _resolve_interval_minutes(args, config)
        state_path = args.state_file

        lock = PipelineLock(args.lock_file, logger)

        def handle_signal(signum, frame) -> None:  # pragma: no cover - signal handler
            logger.warn("received termination signal", signal=signum)
            lock.release()
            sys.exit(0)

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        with lock:
            if args.once or not args.loop:
                run_once(
                    args=args,
                    settings=settings,
                    config=config,
                    logger=logger,
                    state_path=state_path,
                    dry_run=args.dry_run,
                )
                return

            runs = 0
            while True:
                start_monotonic = time.monotonic()
                run_once(
                    args=args,
                    settings=settings,
                    config=config,
                    logger=logger,
                    state_path=state_path,
                    dry_run=args.dry_run,
                )
                runs += 1
                if args.max_runs and runs >= args.max_runs:
                    break
                elapsed = time.monotonic() - start_monotonic
                sleep_seconds = max(0.0, interval_minutes * 60 - elapsed)
                if sleep_seconds > 0:
                    logger.debug("sleeping until next interval", seconds=round(sleep_seconds, 2))
                    time.sleep(sleep_seconds)
    finally:
        logger.close()


if __name__ == "__main__":  # pragma: no cover
    main()
