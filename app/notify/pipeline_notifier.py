from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Protocol, Sequence

import yaml

try:  # pragma: no cover - optional dependency resolution
    import httpx
except ImportError:  # pragma: no cover - allow graceful failure until runtime usage
    httpx = None

if httpx is not None:  # pragma: no cover - preference given to httpx implementations
    HTTPError = httpx.HTTPError
else:  # pragma: no cover - fallback type for type checking
    class HTTPError(Exception):
        ...

from ..config import get_settings
from ..profiles import ContextPaths, PartitionPaths, ProfileContext

SEVERITY_ORDER = {"green": 0, "yellow": 1, "orange": 2, "red": 3}
EMBED_COLORS = {
    "red": 0xFF4D6D,
    "orange": 0xFF9F1C,
    "yellow": 0xFFD166,
    "green": 0x06D6A0,
}


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_iso(dt: datetime | None) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        if value.startswith("${") and value.endswith("}"):
            key = value[2:-1]
            return os.getenv(key, "")
        return value
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    return value


def _normalize_severity(value: str | None) -> str | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token not in SEVERITY_ORDER:
        raise ValueError(f"unsupported severity: {value}")
    return token


@dataclass(slots=True)
class TemplateFieldModel:
    name: str
    value: str

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TemplateFieldModel":
        return TemplateFieldModel(
            name=str(data.get("name", "")),
            value=str(data.get("value", "")),
        )


@dataclass(slots=True)
class TemplateConfig:
    title: str
    description: str
    fields: list[TemplateFieldModel] = field(default_factory=list)
    thumbnail: str | None = None
    footer: str | None = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TemplateConfig":
        field_items = data.get("fields") or []
        fields = [
            TemplateFieldModel.from_dict(item)
            for item in field_items
            if isinstance(item, dict)
        ]
        return TemplateConfig(
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            fields=fields,
            thumbnail=data.get("thumbnail"),
            footer=data.get("footer"),
        )


@dataclass(slots=True)
class ChannelConfigBase:
    template: str = "findings"
    severity_min: str = "yellow"
    severity_max: str | None = None
    mention_roles: list[str] = field(default_factory=list)
    rate_limit_seconds: float | None = None

    @staticmethod
    def _base_kwargs(data: dict[str, Any]) -> dict[str, Any]:
        template = str(data.get("template", "findings"))
        severity_min = _normalize_severity(data.get("severity_min") or "yellow") or "yellow"
        severity_max = _normalize_severity(data.get("severity_max")) if data.get("severity_max") else None
        mention_roles = [
            str(item).strip()
            for item in data.get("mention_roles", [])
            if isinstance(item, (str, int)) and str(item).strip()
        ]
        rate_limit = data.get("rate_limit_seconds")
        rate_limit_val = float(rate_limit) if rate_limit is not None else None
        return {
            "template": template,
            "severity_min": severity_min,
            "severity_max": severity_max,
            "mention_roles": mention_roles,
            "rate_limit_seconds": rate_limit_val,
        }

    def applies_to(self, severity: str) -> bool:
        severity = severity.lower()
        if severity not in SEVERITY_ORDER:
            return False
        minimum = SEVERITY_ORDER[self.severity_min]
        maximum = SEVERITY_ORDER.get(self.severity_max, SEVERITY_ORDER["red"])
        return minimum <= SEVERITY_ORDER[severity] <= maximum


@dataclass(slots=True)
class WebhookChannelConfig(ChannelConfigBase):
    endpoint: str = ""

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "WebhookChannelConfig":
        endpoint = str(data.get("endpoint", "")).strip()
        if not endpoint:
            raise ValueError("webhook endpoint must be provided")
        base = ChannelConfigBase._base_kwargs(data)
        return WebhookChannelConfig(endpoint=endpoint, **base)


@dataclass(slots=True)
class BotChannelConfig(ChannelConfigBase):
    channel_id: int = 0

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "BotChannelConfig":
        raw = data.get("channel_id")
        if raw is None:
            raise ValueError("channel_id must be provided for bot notifications")
        channel_id = int(str(raw).strip())
        base = ChannelConfigBase._base_kwargs(data)
        return BotChannelConfig(channel_id=channel_id, **base)


ChannelConfig = WebhookChannelConfig | BotChannelConfig


@dataclass(slots=True)
class DefaultsConfig:
    timezone: str = "UTC"
    message_url_prefix: str | None = None
    batch_size: int = 1
    rate_limit_seconds: float = 1.0
    max_retries: int = 5
    alert_cooldown_minutes: int = 30

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "DefaultsConfig":
        return DefaultsConfig(
            timezone=str(data.get("timezone", "UTC")),
            message_url_prefix=data.get("message_url_prefix") or None,
            batch_size=int(data.get("batch_size", 1) or 1),
            rate_limit_seconds=float(data.get("rate_limit_seconds", 1.0) or 1.0),
            max_retries=int(data.get("max_retries", 5) or 5),
            alert_cooldown_minutes=int(data.get("alert_cooldown_minutes", 30) or 30),
        )


@dataclass(slots=True)
class NotificationConfig:
    defaults: DefaultsConfig
    channels: dict[str, ChannelConfig]
    templates: dict[str, TemplateConfig]

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "NotificationConfig":
        expanded = _expand_env(data or {})
        defaults = DefaultsConfig.from_dict(expanded.get("defaults", {}))
        templates_raw = expanded.get("templates", {}) or {}
        templates = {
            name: TemplateConfig.from_dict(raw or {})
            for name, raw in templates_raw.items()
            if isinstance(raw, dict)
        }
        channels_raw = expanded.get("channels", {}) or {}
        channels: dict[str, ChannelConfig] = {}
        for name, raw in channels_raw.items():
            if not isinstance(raw, dict):
                continue
            kind = str(raw.get("type", "webhook")).strip().lower()
            if kind == "webhook":
                channels[name] = WebhookChannelConfig.from_dict(raw)
            elif kind == "bot":
                channels[name] = BotChannelConfig.from_dict(raw)
            else:
                raise ValueError(f"unsupported channel type: {kind}")
        return NotificationConfig(defaults=defaults, channels=channels, templates=templates)

    def template_for(self, channel: ChannelConfig) -> TemplateConfig:
        template_name = channel.template
        if template_name not in self.templates:
            raise KeyError(f"template '{template_name}' not defined in configuration")
        return self.templates[template_name]


@dataclass(slots=True)
class NotificationState:
    profile: str | None = None
    last_findings_at: Optional[datetime] = None
    last_findings_id: Optional[str] = None
    last_alert_at: Optional[datetime] = None
    last_alert_run_id: Optional[str] = None
    pending_notifications: list[dict[str, Any]] = field(default_factory=list)
    failure_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "last_findings_at": _format_iso(self.last_findings_at),
            "last_findings_id": self.last_findings_id,
            "last_alert_at": _format_iso(self.last_alert_at),
            "last_alert_run_id": self.last_alert_run_id,
            "pending_notifications": self.pending_notifications,
            "failure_count": self.failure_count,
        }


def load_notification_state(path: Path) -> NotificationState:
    if not path.exists():
        return NotificationState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        backup = path.with_suffix(path.suffix + ".bak")
        if backup.exists():
            return load_notification_state(backup)
        return NotificationState()
    if not isinstance(data, dict):
        return NotificationState()
    state = NotificationState()
    state.profile = data.get("profile")
    state.last_findings_at = _parse_iso(data.get("last_findings_at"))
    state.last_findings_id = data.get("last_findings_id")
    state.last_alert_at = _parse_iso(data.get("last_alert_at"))
    state.last_alert_run_id = data.get("last_alert_run_id")
    pending = data.get("pending_notifications")
    if isinstance(pending, list):
        state.pending_notifications = [item for item in pending if isinstance(item, dict)]
    try:
        state.failure_count = int(data.get("failure_count", 0))
    except (TypeError, ValueError):
        state.failure_count = 0
    return state


def save_notification_state(path: Path, state: NotificationState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state.to_dict(), ensure_ascii=False, indent=2)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        try:
            path.replace(backup)
        except OSError:
            pass
    tmp.replace(path)


class NotificationTransport(Protocol):
    def send_webhook(self, channel: WebhookChannelConfig, payload: dict[str, Any]) -> bool:
        ...

    def send_bot(self, channel: BotChannelConfig, payload: dict[str, Any]) -> bool:
        ...

    def close(self) -> None:
        ...


class HttpNotificationTransport(NotificationTransport):
    def __init__(
        self,
        *,
        bot_token: str | None,
        timeout: float = 10.0,
        max_retries: int = 5,
    ) -> None:
        if httpx is None:
            raise RuntimeError("httpx is required to send notifications; please install the 'httpx' package.")
        self._client = httpx.Client(timeout=timeout)
        self._bot_token = bot_token
        self._max_retries = max(1, max_retries)

    def close(self) -> None:
        self._client.close()

    def _post_with_retries(self, url: str, *, headers: dict[str, str] | None, payload: dict[str, Any]) -> bool:
        delay = 1.0
        for attempt in range(self._max_retries):
            try:
                response = self._client.post(url, json=payload, headers=headers)
            except HTTPError:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            if response.status_code in (200, 201, 202, 204):
                return True
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait_time = float(retry_after) if retry_after else 1.0
                except ValueError:
                    wait_time = 1.0
                time.sleep(wait_time)
                continue
            if 500 <= response.status_code < 600:
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            return False
        return False

    def send_webhook(self, channel: WebhookChannelConfig, payload: dict[str, Any]) -> bool:
        return self._post_with_retries(channel.endpoint, headers=None, payload=payload)

    def send_bot(self, channel: BotChannelConfig, payload: dict[str, Any]) -> bool:
        if not self._bot_token:
            raise RuntimeError("DISCORD_BOT_TOKEN is not configured for bot notifications")
        url = f"https://discord.com/api/v10/channels/{channel.channel_id}/messages"
        headers = {
            "Authorization": f"Bot {self._bot_token}",
        }
        return self._post_with_retries(url, headers=headers, payload=payload)


@dataclass(slots=True)
class NotificationJob:
    channel_key: str
    channel: ChannelConfig
    payload: dict[str, Any]
    category: str
    severity: str | None = None
    identifier: str | None = None


def load_notification_config(path: Path) -> NotificationConfig:
    if not path.exists():
        raise FileNotFoundError(f"notification config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("notification config must be a mapping at the top level")
    return NotificationConfig.from_dict(raw)


def _build_record_ids(record: dict) -> list[str]:
    ids: list[str] = []
    messages = record.get("messages") or []
    for message in messages:
        if not isinstance(message, dict):
            continue
        msg_id = message.get("message_id") or record.get("message_id")
        attachments = message.get("attachments") or []
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            att_phash = attachment.get("phash_hex") or record.get("phash")
            if msg_id and att_phash:
                ids.append(f"{msg_id}:{att_phash}")
    if not ids:
        msg_id = record.get("message_id")
        phash = record.get("phash")
        if msg_id and phash:
            ids.append(f"{msg_id}:{phash}")
        elif msg_id:
            ids.append(str(msg_id))
        elif phash:
            ids.append(str(phash))
    return ids


def _ensure_allowed_mentions(mention_roles: Sequence[str]) -> dict[str, Any]:
    role_ids = [str(role).strip() for role in mention_roles if str(role).strip()]
    if not role_ids:
        return {"parse": []}
    return {"parse": [], "roles": role_ids}


def _render_template_field(value: str, data: Dict[str, Any]) -> str:
    try:
        return value.format(**data)
    except Exception:
        return value


def _build_embed(template: TemplateConfig, data: Dict[str, Any], severity: str) -> dict[str, Any]:
    embed = {
        "title": _render_template_field(template.title, data),
        "description": _render_template_field(template.description, data),
        "color": EMBED_COLORS.get(severity.lower(), EMBED_COLORS["yellow"]),
    }
    fields = []
    for field in template.fields:
        fields.append(
            {
                "name": _render_template_field(field.name, data),
                "value": _render_template_field(field.value, data),
            }
        )
    if fields:
        embed["fields"] = fields
    if template.thumbnail:
        embed["thumbnail"] = {"url": _render_template_field(template.thumbnail, data)}
    if template.footer:
        embed["footer"] = {"text": _render_template_field(template.footer, data)}
    if data.get("timestamp_iso"):
        embed["timestamp"] = data["timestamp_iso"]
    return embed


def _message_url(record: dict, defaults: DefaultsConfig) -> Optional[str]:
    link = record.get("message_link")
    if link:
        return link
    prefix = defaults.message_url_prefix
    if not prefix:
        return None
    guild = record.get("guild_id")
    channel = record.get("channel_id")
    message = record.get("message_id")
    if not (guild and channel and message):
        return None
    try:
        return prefix.format(guild_id=guild, channel_id=channel, message_id=message)
    except KeyError:
        return None


def _record_timestamp(record: dict) -> Optional[datetime]:
    ts = _parse_iso(record.get("created_at"))
    if ts is not None:
        return ts
    metrics = record.get("metrics", {})
    if isinstance(metrics, dict):
        ts_alt = metrics.get("created_at")
        return _parse_iso(ts_alt)
    return None


def _format_reasons(reasons: Sequence[str] | None) -> str:
    if not reasons:
        return "N/A"
    return "\n".join(str(reason) for reason in reasons)


@dataclass(slots=True)
class FindingsWindow:
    last_timestamp: Optional[datetime]
    last_identifier: Optional[str]

    def is_new(self, created_at: datetime | None, identifiers: Sequence[str]) -> bool:
        if created_at is None:
            created_at = datetime.now(timezone.utc)
        max_identifier = max(identifiers) if identifiers else None
        if self.last_timestamp is None:
            return True
        if created_at > self.last_timestamp:
            return True
        if created_at < self.last_timestamp:
            return False
        if max_identifier is None:
            return False
        if self.last_identifier is None:
            return True
        return max_identifier > self.last_identifier

    def update(self, created_at: datetime | None, identifiers: Sequence[str]) -> None:
        created_at = created_at or datetime.now(timezone.utc)
        max_identifier = max(identifiers) if identifiers else None
        if self.last_timestamp is None or created_at > self.last_timestamp:
            self.last_timestamp = created_at
            self.last_identifier = max_identifier
            return
        if created_at == self.last_timestamp and max_identifier is not None:
            if self.last_identifier is None or max_identifier > self.last_identifier:
                self.last_identifier = max_identifier


class NotificationRunner:
    def __init__(
        self,
        *,
        config: NotificationConfig,
        state: NotificationState,
        state_path: Path,
        context: ProfileContext,
        paths: ContextPaths,
        logger,
        transport: NotificationTransport,
    ) -> None:
        self.config = config
        self.state = state
        self.state_path = state_path
        self.context = context
        self.paths = paths
        self.logger = logger
        self.transport = transport
        self.defaults = config.defaults
        self.state.profile = context.profile

    def run(
        self,
        *,
        only: Sequence[str] | None = None,
        dry_run: bool = False,
    ) -> bool:
        categories = {entry.lower() for entry in (only or ["findings", "alerts"])}
        success = True
        if not self._flush_pending(dry_run=dry_run):
            success = False
        if "findings" in categories:
            success = self._process_findings(dry_run=dry_run) and success
        if "alerts" in categories:
            success = self._process_pipeline_alerts(dry_run=dry_run) and success
        if not dry_run:
            save_notification_state(self.state_path, self.state)
        return success

    def close(self) -> None:
        if hasattr(self.transport, "close"):
            self.transport.close()

    def _flush_pending(self, *, dry_run: bool) -> bool:
        if not self.state.pending_notifications:
            return True
        remaining: list[dict[str, Any]] = []
        all_success = True
        for entry in self.state.pending_notifications:
            channel_key = entry.get("channel_key")
            payload = entry.get("payload")
            mode = entry.get("mode")
            category = entry.get("category", "pending")
            severity = entry.get("severity")
            if not channel_key or not isinstance(payload, dict):
                continue
            channel = self.config.channels.get(channel_key)
            if channel is None:
                continue
            if dry_run:
                self.logger.info(
                    "notification_dry_run",
                    category=category,
                    channel=channel_key,
                    mode=mode,
                    severity=severity,
                    pending=True,
                )
                continue
            try:
                sent = self._dispatch(channel_key, channel, payload, category, severity=severity)
            except Exception as exc:  # noqa: BLE001
                self.logger.error(
                    "notification_error",
                    category=category,
                    channel=channel_key,
                    error=str(exc),
                    pending=True,
                )
                sent = False
            if not sent:
                remaining.append(entry)
                all_success = False
        self.state.pending_notifications = remaining
        if all_success and not remaining:
            self.state.failure_count = 0
        return all_success

    def _process_findings(self, *, dry_run: bool) -> bool:
        findings_path = self.paths.stage_file("p3")
        if not findings_path.exists():
            self.logger.warn("findings_file_missing", path=findings_path)
            return True
        window = FindingsWindow(self.state.last_findings_at, self.state.last_findings_id)
        records_to_notify: list[NotificationJob] = []
        latest_timestamp = window.last_timestamp
        latest_identifier = window.last_identifier
        with findings_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                severity = str(record.get("severity", "")).lower()
                if severity not in SEVERITY_ORDER:
                    continue
                created_at = _record_timestamp(record)
                ids = _build_record_ids(record)
                is_new = window.is_new(created_at, ids)
                window.update(created_at, ids)
                latest_timestamp = window.last_timestamp
                latest_identifier = window.last_identifier
                if not is_new:
                    continue
                template_data = self._build_findings_template_data(record, severity)
                for channel_key, channel in self.config.channels.items():
                    if not channel.applies_to(severity):
                        continue
                    template = self.config.template_for(channel)
                    payload = self._build_payload(channel, template, template_data, severity)
                    identifier = max(ids) if ids else record.get("message_id")
                    records_to_notify.append(
                        NotificationJob(
                            channel_key=channel_key,
                            channel=channel,
                            payload=payload,
                            category="findings",
                            severity=severity,
                            identifier=str(identifier) if identifier is not None else None,
                        )
                    )
        if latest_timestamp is not None:
            self.state.last_findings_at = latest_timestamp
            self.state.last_findings_id = latest_identifier
        if not records_to_notify:
            return True
        return self._deliver(records_to_notify, dry_run=dry_run)

    def _process_pipeline_alerts(self, *, dry_run: bool) -> bool:
        metrics_path = self.paths.pipeline_metrics_path(ensure_parent=False)
        if not metrics_path.exists():
            self.logger.debug("pipeline_metrics_missing", path=metrics_path)
            return True
        try:
            last_line = _read_last_line(metrics_path)
            if not last_line:
                return True
            payload = json.loads(last_line)
        except (OSError, json.JSONDecodeError):
            self.logger.warn("pipeline_metrics_corrupt", path=metrics_path)
            return False
        run_id = payload.get("run_id")
        finished_at = _parse_iso(payload.get("finished_at"))
        stages = payload.get("stages", {})
        if not isinstance(stages, dict):
            return True
        cooldown = timedelta(minutes=self.defaults.alert_cooldown_minutes)
        if (
            self.state.last_alert_run_id == run_id
            and self.state.last_alert_at is not None
            and finished_at is not None
            and finished_at - self.state.last_alert_at < cooldown
        ):
            return True
        failing: list[tuple[str, dict[str, Any]]] = [
            (name, info) for name, info in stages.items() if isinstance(info, dict) and info.get("status") == "failed"
        ]
        if not failing:
            return True
        template_data_list = []
        for stage_name, info in failing:
            attempts = info.get("attempts")
            last_error = info.get("last_error") or "<no stderr>"
            returncode = info.get("returncode")
            template_data_list.append(
                {
                    "stage": stage_name,
                    "attempts": attempts or 0,
                    "last_error": last_error,
                    "returncode": returncode,
                    "run_id": run_id,
                    "finished_at": _format_iso(finished_at) or "",
                }
            )
        jobs: list[NotificationJob] = []
        for channel_key, channel in self.config.channels.items():
            if channel.template != "pipeline_error":
                continue
            template = self.config.template_for(channel)
            for data in template_data_list:
                payload = self._build_payload(channel, template, data, severity="red")
                jobs.append(
                    NotificationJob(
                        channel_key=channel_key,
                        channel=channel,
                        payload=payload,
                        category="alerts",
                        severity="red",
                        identifier=str(run_id) if run_id else None,
                    )
                )
        if not jobs:
            return True
        success = self._deliver(jobs, dry_run=dry_run)
        if success and not dry_run:
            self.state.last_alert_at = finished_at or datetime.now(timezone.utc)
            self.state.last_alert_run_id = run_id
        return success

    def _build_findings_template_data(self, record: dict, severity: str) -> Dict[str, Any]:
        reasons = record.get("reasons")
        ratings = record.get("wd14", {}).get("rating", {}) if isinstance(record.get("wd14"), dict) else {}
        qa = ratings.get("questionable") or 0.0
        exp = ratings.get("explicit") or 0.0
        created_at = _record_timestamp(record)
        channel_name = record.get("channel_name") or record.get("channel_id")
        author_name = record.get("author_name") or record.get("author_id")
        message_url = _message_url(record, self.defaults) or record.get("message_link")
        attachment_url = None
        for message in record.get("messages") or []:
            if not isinstance(message, dict):
                continue
            attachments = message.get("attachments") or []
            for attachment in attachments:
                if isinstance(attachment, dict) and attachment.get("url"):
                    attachment_url = attachment.get("url")
                    break
            if attachment_url:
                break
        return {
            "severity": severity.upper(),
            "severity_lower": severity.lower(),
            "rule_id": record.get("rule_id") or "N/A",
            "rule_title": record.get("rule_title") or "N/A",
            "reasons": _format_reasons(reasons),
            "message_url": message_url or "N/A",
            "channel_name": channel_name or "N/A",
            "author_name": author_name or "N/A",
            "rating_explicit": float(exp),
            "rating_questionable": float(qa),
            "timestamp_iso": _format_iso(created_at),
            "created_at": record.get("created_at") or "",
        }

    def _build_payload(
        self,
        channel: ChannelConfig,
        template: TemplateConfig,
        data: Dict[str, Any],
        severity: str,
    ) -> dict[str, Any]:
        mentions = " ".join(f"<@&{role}>" for role in channel.mention_roles if role)
        embed = _build_embed(template, data, severity)
        payload: dict[str, Any] = {
            "embeds": [embed],
            "allowed_mentions": _ensure_allowed_mentions(channel.mention_roles),
        }
        if mentions:
            payload["content"] = mentions
        return payload

    def _deliver(self, jobs: Sequence[NotificationJob], *, dry_run: bool) -> bool:
        all_success = True
        interval = self.defaults.rate_limit_seconds
        for job in jobs:
            if dry_run:
                self.logger.info(
                    "notification_dry_run",
                    category=job.category,
                    channel=job.channel_key,
                    severity=job.severity,
                    template=getattr(job.channel, "template", None),
                )
                continue
            sent = self._dispatch(
                job.channel_key,
                job.channel,
                job.payload,
                job.category,
                severity=job.severity,
            )
            if not sent:
                all_success = False
                pending_entry = {
                    "channel_key": job.channel_key,
                    "payload": job.payload,
                    "mode": getattr(job.channel, "type", "webhook"),
                    "category": job.category,
                    "severity": job.severity,
                }
                self.state.pending_notifications.append(pending_entry)
            if interval > 0:
                time.sleep(interval)
        if all_success:
            self.state.pending_notifications.clear()
            self.state.failure_count = 0
        else:
            self.state.failure_count += 1
        return all_success

    def _dispatch(
        self,
        channel_key: str,
        channel: ChannelConfig,
        payload: dict[str, Any],
        category: str,
        *,
        severity: str | None,
    ) -> bool:
        try:
            if isinstance(channel, WebhookChannelConfig):
                sent = self.transport.send_webhook(channel, payload)
            elif isinstance(channel, BotChannelConfig):
                sent = self.transport.send_bot(channel, payload)
            else:
                raise RuntimeError(f"unsupported channel type: {type(channel)}")
        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "notification_error",
                category=category,
                channel=channel_key,
                error=str(exc),
            )
            return False

        if sent:
            self.logger.info(
                "notification_sent",
                category=category,
                channel=channel_key,
                severity=severity,
            )
        else:
            self.logger.error(
                "notification_error",
                category=category,
                channel=channel_key,
                severity=severity,
                error="transport_failed",
            )
        return sent


def _read_last_line(path: Path) -> Optional[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            position = handle.tell()
            if position == 0:
                return None
            buffer = bytearray()
            while position:
                position -= 1
                handle.seek(position)
                byte = handle.read(1)
                if byte == b"\n" and buffer:
                    break
                buffer.extend(byte)
            buffer.reverse()
            return buffer.decode("utf-8").strip()
    except OSError:
        return None
