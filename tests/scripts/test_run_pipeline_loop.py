from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pytest

if "pydantic" not in sys.modules:  # pragma: no cover - dependency shim for tests
    pydantic_stub = types.ModuleType("pydantic")

    def Field(*args, **kwargs):
        return kwargs.get("default")

    pydantic_stub.Field = Field
    sys.modules["pydantic"] = pydantic_stub

if "pydantic_settings" not in sys.modules:  # pragma: no cover
    settings_stub = types.ModuleType("pydantic_settings")

    class BaseSettings:
        model_config = {}

        def __init__(self, **_kwargs):
            pass

    class SettingsConfigDict(dict):
        pass

    settings_stub.BaseSettings = BaseSettings
    settings_stub.SettingsConfigDict = SettingsConfigDict
    sys.modules["pydantic_settings"] = settings_stub


from scripts.run_pipeline_loop import (  # noqa: E402
    PipelineConfig,
    PipelineState,
    append_run_metrics,
    build_stage_command,
    count_records,
    load_pipeline_config,
    load_state,
    save_state,
)


class DummyLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def warn(self, message: str, **fields):
        self.messages.append(("warn", message))

    def debug(self, message: str, **fields):
        self.messages.append(("debug", message))

    def info(self, message: str, **fields):
        self.messages.append(("info", message))


@pytest.fixture()
def logger() -> DummyLogger:
    return DummyLogger()


def test_load_pipeline_config_defaults(tmp_path: Path, logger: DummyLogger) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "p1:\n  batch_size: 32\n  merge_existing: false\n  retries: 3\n", encoding="utf-8"
    )
    config = load_pipeline_config(config_path, logger)  # type: ignore[arg-type]
    assert isinstance(config, PipelineConfig)
    assert config.p1.cli_args["--batch-size"] == 32
    assert config.p1.cli_args["--merge-existing"] is False
    assert config.p1.retries == 3


def test_build_stage_command_p0_includes_since(tmp_path: Path) -> None:
    context_paths = SimpleNamespace(
        stage_file=lambda stage, ensure_parent=False: tmp_path / f"{stage}.csv",
    )
    partitions = SimpleNamespace(
        report_path=lambda ensure_parent=False: tmp_path / "report.csv",
    )
    p0_options = PipelineConfig().p0
    cmd, output_path = build_stage_command(
        stage_name="p0",
        stage_config=p0_options,
        partitions=partitions,
        context_paths=context_paths,
        profile="current",
        date_token="2025-10-12",
        since_iso="2025-10-12T00:00:00+00:00",
    )
    assert "--since" in cmd
    assert "--resume" in cmd
    assert output_path == tmp_path / "p0.csv"


def test_append_run_metrics_appends_lines(tmp_path: Path, logger: DummyLogger) -> None:
    class DummyPartitions:
        def __init__(self, root: Path) -> None:
            self._root = root
            self.context = SimpleNamespace(profile="current")

        def profile_root(self, ensure: bool = False) -> Path:
            if ensure:
                self._root.mkdir(parents=True, exist_ok=True)
            return self._root

    partitions = DummyPartitions(tmp_path / "profiles" / "current")
    stage_metrics = {"p0": {"status": "success", "duration_sec": 1.0, "records": 5}}
    append_run_metrics(
        partitions=partitions,  # type: ignore[arg-type]
        date_token="2025-10-12",
        run_id="run-1",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        stage_metrics=stage_metrics,
        logger=logger,  # type: ignore[arg-type]
    )
    metrics_file = partitions.profile_root(True) / "metrics" / "pipeline" / "pipeline_2025-10-12.jsonl"
    assert metrics_file.exists()
    contents = metrics_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1
    payload = json.loads(contents[0])
    assert payload["stages"]["p0"]["records"] == 5


def test_state_roundtrip(tmp_path: Path, logger: DummyLogger) -> None:
    path = tmp_path / "pipeline_state.json"
    state = PipelineState(
        profile="current",
        last_started_at=datetime(2025, 10, 12, 3, 0, tzinfo=timezone.utc),
        last_completed_at=datetime(2025, 10, 12, 3, 15, tzinfo=timezone.utc),
        last_status="success",
        last_run_id="test",
        failure_count=0,
    )
    save_state(path, state, logger)  # type: ignore[arg-type]
    loaded = load_state(path, logger)  # type: ignore[arg-type]
    assert loaded.profile == "current"
    assert loaded.last_status == "success"
    assert loaded.last_completed_at == state.last_completed_at
