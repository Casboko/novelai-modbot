from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pytest

import sys
import types

if "pydantic" not in sys.modules:  # pragma: no cover - dependency shim for tests
    pydantic_stub = types.ModuleType("pydantic")

    def Field(*args, **kwargs):
        return kwargs.get("default")

    class BaseModel:
        def __init__(self, *args, **kwargs):
            pass

    pydantic_stub.Field = Field
    pydantic_stub.BaseModel = BaseModel
    sys.modules["pydantic"] = pydantic_stub

if "pydantic_settings" not in sys.modules:  # pragma: no cover - dependency shim for tests
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

if "discord" not in sys.modules:  # pragma: no cover - provide minimal stub
    sys.modules["discord"] = types.ModuleType("discord")

pytest.importorskip("aiosqlite")

from app.notify import NotificationConfig, NotificationRunner, NotificationState, load_notification_config
from app.notify.pipeline_notifier import save_notification_state
from app.profiles import ContextPaths, ProfileContext, clear_context_cache, set_profiles_root_override


class DummyLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict[str, Any]]] = []

    def _store(self, level: str, message: str, **fields: Any) -> None:
        self.records.append((level, message, fields))

    def debug(self, message: str, **fields: Any) -> None:
        self._store("debug", message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        self._store("info", message, **fields)

    def warn(self, message: str, **fields: Any) -> None:
        self._store("warn", message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self._store("error", message, **fields)


class DummyTransport:
    def __init__(self) -> None:
        self.webhook_payloads: list[dict[str, Any]] = []
        self.bot_payloads: list[dict[str, Any]] = []
        self.fail_next_webhook = False

    def close(self) -> None:
        return None

    def send_webhook(self, channel, payload: dict[str, Any]) -> bool:
        if self.fail_next_webhook:
            self.fail_next_webhook = False
            return False
        self.webhook_payloads.append(payload)
        return True

    def send_bot(self, channel, payload: dict[str, Any]) -> bool:
        self.bot_payloads.append(payload)
        return True


@pytest.fixture()
def notifications_config(tmp_path: Path) -> NotificationConfig:
    config_yaml = tmp_path / "notifications.yaml"
    config_yaml.write_text(
        """
defaults:
  message_url_prefix: "https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"
  rate_limit_seconds: 0
  max_retries: 1
  alert_cooldown_minutes: 30

channels:
  findings_red:
    type: webhook
    endpoint: "https://example.com/webhook"
    severity_min: red
    template: findings
  pipeline_alerts:
    type: bot
    channel_id: "12345"
    template: pipeline_error

templates:
  findings:
    title: "[{severity}] {rule_id}"
    description: "{message_url}"
    fields:
      - name: "Reasons"
        value: "{reasons}"
  pipeline_error:
    title: "[Pipeline] {stage} failed"
    description: "Attempts: {attempts}, Run: {run_id}"
    fields:
      - name: "Last stderr"
        value: "{last_error}"
""".strip(),
        encoding="utf-8",
    )
    return load_notification_config(config_yaml)


@pytest.fixture()
def profile_root(tmp_path: Path):
    set_profiles_root_override(tmp_path / "profiles")
    clear_context_cache()
    yield
    set_profiles_root_override(None)
    clear_context_cache()


def _write_findings(path: Path, records: list[dict[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pipeline_metrics(path: Path, payloads: list[dict[str, Any]]) -> None:
    lines = [json.dumps(item, ensure_ascii=False) for item in payloads]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_runner_sends_new_findings_and_updates_state(
    tmp_path: Path,
    notifications_config: NotificationConfig,
    profile_root,
) -> None:
    context = ProfileContext.from_cli(profile_arg="current", date_arg="2025-10-12")
    paths = ContextPaths.for_context(context)
    findings_path = paths.stage_file("p3", ensure_parent=True)

    record = {
        "severity": "red",
        "rule_id": "TEST-RED",
        "reasons": ["exp=0.85"],
        "created_at": "2025-10-12T02:00:00Z",
        "message_link": "https://discord.com/channels/1/2/3",
        "guild_id": "1",
        "channel_id": "2",
        "channel_name": "alerts",
        "author_id": "42",
        "author_name": "Tester",
        "message_id": "3",
        "phash": "deadbeef",
        "messages": [
            {
                "message_id": "3",
                "attachments": [
                    {
                        "id": "att-1",
                        "url": "https://cdn.discordapp.com/test.png",
                        "phash_hex": "deadbeef",
                    }
                ],
            }
        ],
        "wd14": {"rating": {"explicit": 0.85, "questionable": 0.1}},
    }
    _write_findings(findings_path, [record])

    state_path = tmp_path / "notify_state.json"
    state = NotificationState()
    save_notification_state(state_path, state)

    logger = DummyLogger()
    transport = DummyTransport()
    runner = NotificationRunner(
        config=notifications_config,
        state=state,
        state_path=state_path,
        context=context,
        paths=paths,
        logger=logger,
        transport=transport,
        findings_store_path=None,
    )
    try:
        success = runner.run()
    finally:
        runner.close()
    assert success
    assert transport.webhook_payloads, "expected webhook payload to be sent"
    sent = transport.webhook_payloads[0]
    assert sent["embeds"][0]["title"] == "[RED] TEST-RED"
    assert state.last_findings_at is not None
    assert state.last_findings_id is not None


def test_runner_keeps_pending_on_failure(
    tmp_path: Path,
    notifications_config: NotificationConfig,
    profile_root,
) -> None:
    context = ProfileContext.from_cli(profile_arg="current", date_arg="2025-10-12")
    paths = ContextPaths.for_context(context)
    findings_path = paths.stage_file("p3", ensure_parent=True)
    record = {
        "severity": "red",
        "rule_id": "TEST-RED",
        "reasons": ["exp=0.85"],
        "created_at": "2025-10-12T02:00:00Z",
        "message_link": "https://discord.com/channels/1/2/3",
        "guild_id": "1",
        "channel_id": "2",
        "message_id": "3",
        "phash": "deadbeef",
    }
    _write_findings(findings_path, [record])

    state = NotificationState()
    state_path = tmp_path / "notify_state.json"
    save_notification_state(state_path, state)

    transport = DummyTransport()
    transport.fail_next_webhook = True
    logger = DummyLogger()
    runner = NotificationRunner(
        config=notifications_config,
        state=state,
        state_path=state_path,
        context=context,
        paths=paths,
        logger=logger,
        transport=transport,
        findings_store_path=None,
    )
    try:
        success = runner.run()
    finally:
        runner.close()
    assert not success
    assert state.pending_notifications, "failed payload should remain pending"
    assert state.failure_count == 1


def test_pipeline_alert_sent_once_within_cooldown(
    tmp_path: Path,
    notifications_config: NotificationConfig,
    profile_root,
) -> None:
    context = ProfileContext.from_cli(profile_arg="current", date_arg="2025-10-12")
    paths = ContextPaths.for_context(context)
    metrics_path = paths.pipeline_metrics_path(ensure_parent=True)
    payload = {
        "run_id": "run-1",
        "finished_at": "2025-10-12T03:00:00Z",
        "profile": "current",
        "stages": {
            "p2": {
                "status": "failed",
                "attempts": 2,
                "returncode": 1,
                "last_error": "base64:ZmFpbGVk",
            }
        },
    }
    _write_pipeline_metrics(metrics_path, [payload])

    state = NotificationState()
    state_path = tmp_path / "notify_state.json"
    save_notification_state(state_path, state)

    transport = DummyTransport()
    logger = DummyLogger()
    runner = NotificationRunner(
        config=notifications_config,
        state=state,
        state_path=state_path,
        context=context,
        paths=paths,
        logger=logger,
        transport=transport,
        findings_store_path=None,
    )
    try:
        first = runner.run(only=["alerts"])
        second = runner.run(only=["alerts"])
    finally:
        runner.close()
    assert first is True
    assert second is True
    assert len(transport.bot_payloads) == 1
    assert state.last_alert_run_id == "run-1"
    assert state.last_alert_at is not None
    assert state.last_alert_at >= datetime(2025, 10, 12, 3, 0, tzinfo=timezone.utc)
