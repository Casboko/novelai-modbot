from __future__ import annotations

import asyncio
import fnmatch
import logging
import re
from functools import lru_cache
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, List, Mapping, NamedTuple, Optional, Sequence

import discord
from discord import app_commands

from .config import get_settings
from .discord_client import create_client
from .engine.group_utils import group_top_tags
from .engine.tag_norm import normalize_pair, normalize_tag
from .mode_resolver import has_version_mismatch, resolve_policy
from .rule_engine import RuleEngine
from .triage import iter_findings, load_findings_async, resolve_time_range, run_scan, write_report_csv
from .store import Ticket, TicketStore
from .util import parse_message_link

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))
SEVERITY_COLORS = {
    "red": discord.Color.from_str("#ff4d6d"),
    "orange": discord.Color.from_str("#ff9f1c"),
    "yellow": discord.Color.from_str("#ffd166"),
    "green": discord.Color.from_str("#06d6a0"),
}
SEVERITY_ORDER = ("red", "orange", "yellow", "green")
SEVERITY_BADGES = {
    "red": "【重大】",
    "orange": "【警戒】",
    "yellow": "【注意】",
    "green": "【確認済】",
}
SEVERITY_FIELD_NAMES = {
    "red": "赤",
    "orange": "橙",
    "yellow": "黄",
    "green": "緑",
}
SEVERITY_OPTION_LABELS = {
    "alerts": "警告（緑を除く）",
    "all": "すべて",
    "red": "赤のみ",
    "orange": "橙のみ",
    "yellow": "黄のみ",
    "green": "緑のみ",
}
DEFAULT_RECORD_TITLE = "モデレーション検知"
UTC_MIN = datetime.min.replace(tzinfo=timezone.utc)
UTC_MAX = datetime.max.replace(tzinfo=timezone.utc)


def build_jump_link(gid: int, cid: int, mid: int) -> str:
    return f"https://discord.com/channels/{gid}/{cid}/{mid}"


class PreviewInfo(NamedTuple):
    thumbnail_url: Optional[str]
    image_url: Optional[str]
    original_url: Optional[str]
    is_spoiler: bool = False


class TicketEntry(NamedTuple):
    ticket: Ticket
    record: dict


def _severity_badge(severity: str) -> str:
    return SEVERITY_BADGES.get(severity.lower(), f"[{severity.upper()}]")


def _severity_field_label(severity: str) -> str:
    return SEVERITY_FIELD_NAMES.get(severity.lower(), severity.upper())


def _severity_option_label(value: str) -> str:
    return SEVERITY_OPTION_LABELS.get(value.lower(), value)


def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))

def _format_top_detections(detections: Iterable[dict], limit: int = 3) -> List[str]:
    formatted: List[tuple[str, float]] = []
    for item in detections or []:
        cls = item.get("class")
        score = item.get("score")
        if cls and score is not None:
            formatted.append((str(cls), float(score)))
    formatted.sort(key=lambda entry: entry[1], reverse=True)
    return [f"{name}:{score:.2f}" for name, score in formatted[:limit]]


def _format_timestamp(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


_RULES_PATH = Path(RuleEngine.DEFAULT_RULES_PATH)
_GROUP_FUNCTIONS = {"max", "sum", "count", "topk_sum"}
_GROUP_WILDCARD_CHARS = set("*?[")
_FEATURE_PRIORITY = (
    "explicit_score",
    "sexual_intensity",
    "nsfw_rating_max",
    "minor_peak_conf",
    "minor_body_score",
    "minor_context_score",
    "gore_final_score",
    "gore_main_peak",
    "animal_presence",
    "coercion_final_score",
    "qe_margin",
)
_DISPLAY_GROUPS = ("gore", "violence", "minors", "animal_subjects", "nsfw_general")
_COMPARISON_RE = re.compile(r"([^&|!=<>]+?)\s*(>=|<=|>|<|==)\s*([^&|]+)")


@lru_cache(maxsize=1)
def _rule_engine_cached(signature: str) -> RuleEngine:
    return RuleEngine(str(_RULES_PATH))


def _get_rule_engine() -> RuleEngine:
    try:
        stat = _RULES_PATH.stat()
        signature = f"{_RULES_PATH.resolve()}::{int(stat.st_mtime)}::{stat.st_size}"
    except OSError:
        signature = "missing"
    return _rule_engine_cached(signature)


def _v2_features(record: Mapping[str, Any]) -> dict[str, float]:
    dsl = (record.get("metrics") or {}).get("dsl") or {}
    feats = dsl.get("features") or {}
    result: dict[str, float] = {}
    for key, value in feats.items():
        if isinstance(value, (int, float)):
            result[str(key)] = float(value)
    return result


def _v2_errors(record: Mapping[str, Any]) -> list[str]:
    metrics = record.get("metrics") or {}
    dsl = metrics.get("dsl")
    if not isinstance(dsl, Mapping):
        return ["metrics.dsl が見つかりません（v2未評価）"]
    features = dsl.get("features")
    if not isinstance(features, Mapping) or not features:
        return ["metrics.dsl.features が空です（v2未評価）"]
    return []


def _truncate_block(lines: Sequence[str], *, code: bool = True, max_chars: int = 950) -> str:
    collected: list[str] = []
    total = 0
    for line in lines:
        text = line.rstrip()
        addition = len(text) + 1
        if total + addition > max_chars:
            break
        collected.append(text)
        total += addition
    if not collected:
        return ""
    body = "\n".join(collected)
    return f"```\n{body}\n```" if code else body


def _collect_tag_scores(record: Mapping[str, Any]) -> dict[str, float]:
    general = ((record.get("wd14") or {}).get("general_raw") or (record.get("wd14") or {}).get("general") or [])
    scores: dict[str, float] = {}
    for item in general:
        pair = normalize_pair(item)
        if pair is None:
            continue
        tag, score = pair
        if score < 0:
            continue
        existing = scores.get(tag)
        if existing is None or score > existing:
            scores[tag] = score
    return scores


def _group_values(tag_scores: Mapping[str, float], patterns: Sequence[str]) -> list[float]:
    if not patterns:
        return []
    hits: list[float] = []
    for pattern in patterns:
        canonical = normalize_tag(pattern)
        if not canonical:
            continue
        if any(ch in canonical for ch in _GROUP_WILDCARD_CHARS):
            for tag, score in tag_scores.items():
                if fnmatch.fnmatchcase(tag, canonical):
                    hits.append(float(score))
        else:
            score = tag_scores.get(canonical)
            if score is not None:
                hits.append(float(score))
    return hits


def _parse_group_call(call_args: str) -> tuple[str | None, dict[str, float]]:
    match = re.search(r"'([^']+)'", call_args)
    if not match:
        return None, {}
    group_name = match.group(1)
    params: dict[str, float] = {}
    k_match = re.search(r"\bk\s*=\s*([0-9]+)", call_args)
    if k_match:
        params["k"] = float(k_match.group(1))
    gt_match = re.search(r"\bgt\s*=\s*([0-9]*\.?[0-9]+)", call_args)
    if gt_match:
        params["gt"] = float(gt_match.group(1))
    return group_name, params


def _evaluate_group_function(
    func: str,
    call_args: str,
    *,
    tag_scores: Mapping[str, float],
    groups: Mapping[str, Sequence[str]],
) -> float | None:
    group_name, params = _parse_group_call(call_args)
    if not group_name:
        return None
    patterns = groups.get(group_name) or groups.get(normalize_tag(group_name)) or ()
    values = _group_values(tag_scores, patterns)
    if func == "max":
        return max(values, default=0.0)
    if func == "sum":
        return float(sum(values))
    if func == "count":
        threshold = params.get("gt", 0.0)
        return float(sum(1 for score in values if score > threshold))
    if func == "topk_sum":
        threshold = params.get("gt", 0.0)
        k = int(params.get("k", 5))
        filtered = [score for score in values if score > threshold]
        filtered.sort(reverse=True)
        return float(sum(filtered[:k]))
    return None


def _resolve_condition_value(
    token: str,
    *,
    record: Mapping[str, Any],
    features: Mapping[str, float],
    metrics: Mapping[str, Any],
    consts: Mapping[str, float],
    groups: Mapping[str, Sequence[str]],
    tag_scores: Mapping[str, float],
) -> float | None:
    text = token.strip()
    if not text:
        return None
    if text.startswith("(") and text.endswith(")"):
        return _resolve_condition_value(
            text[1:-1],
            record=record,
            features=features,
            metrics=metrics,
            consts=consts,
            groups=groups,
            tag_scores=tag_scores,
        )
    func_match = re.match(r"([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)", text)
    if func_match:
        func = func_match.group(1)
        args = func_match.group(2)
        if func in _GROUP_FUNCTIONS:
            return _evaluate_group_function(
                func,
                args,
                tag_scores=tag_scores,
                groups=groups,
            )
    if text in features:
        return float(features[text])
    if text in consts:
        return float(consts[text])
    rating = ((record.get("wd14") or {}).get("rating") or {})
    if text.startswith("rating."):
        key = text.split(".", 1)[1]
        value = rating.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        return None
    if text.startswith("metrics."):
        key = text.split(".", 1)[1]
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        return None
    if text in {"exposure_area", "exposure_count"}:
        value = metrics.get(text)
        if isinstance(value, (int, float)):
            return float(value)
    if text in tag_scores:
        return float(tag_scores[text])
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _build_when_summary_lines(
    when_text: str,
    *,
    record: Mapping[str, Any],
    features: Mapping[str, float],
    metrics: Mapping[str, Any],
    consts: Mapping[str, float],
    groups: Mapping[str, Sequence[str]],
    tag_scores: Mapping[str, float],
    limit: int = 6,
) -> list[str]:
    lines: list[str] = []
    for match in _COMPARISON_RE.finditer(when_text):
        left = match.group(1).strip()
        op = match.group(2)
        right = match.group(3).strip()
        left_val = _resolve_condition_value(
            left,
            record=record,
            features=features,
            metrics=metrics,
            consts=consts,
            groups=groups,
            tag_scores=tag_scores,
        )
        right_val = _resolve_condition_value(
            right,
            record=record,
            features=features,
            metrics=metrics,
            consts=consts,
            groups=groups,
            tag_scores=tag_scores,
        )
        result = None
        if left_val is not None and right_val is not None:
            if op == ">=":
                result = left_val >= right_val
            elif op == "<=":
                result = left_val <= right_val
            elif op == ">":
                result = left_val > right_val
            elif op == "<":
                result = left_val < right_val
            elif op == "==":
                result = abs(left_val - right_val) <= 1e-6
        icon = "✅" if result is True else ("❌" if result is False else "−")
        left_fmt = "-" if left_val is None else f"{left_val:.3f}"
        right_fmt = "-" if right_val is None else f"{right_val:.3f}"
        lines.append(f"{icon} {left} {op} {right} (val={left_fmt}, rhs={right_fmt})")
        if len(lines) >= limit:
            break
    return lines


def _format_wd14_rating_block(record: Mapping[str, Any]) -> str:
    rating = ((record.get("wd14") or {}).get("rating") or {})
    ordering = ("general", "sensitive", "questionable", "explicit")
    labels = {"general": "g", "sensitive": "s", "questionable": "q", "explicit": "e"}
    values = [max(float(rating.get(key, 0.0)), 0.0) for key in ordering]
    total = sum(values) or 1.0
    lines = []
    for key, value in zip(ordering, values):
        label = labels[key]
        ratio = value / total if total else 0.0
        filled = int(round(_clamp(ratio, 0.0, 1.0) * 10))
        filled = min(filled, 10)
        bar = "█" * filled + "▁" * (10 - filled)
        lines.append(f"{label}:{value:.2f} {bar}")
    return "```\n" + "\n".join(lines) + "\n```"
async def _fetch_message(client: discord.Client, channel_id: int, message_id: int) -> discord.Message:
    channel = client.get_channel(channel_id)
    if channel is None:
        channel = await client.fetch_channel(channel_id)
    return await channel.fetch_message(message_id)


def _build_allowed_mentions(user: discord.User | discord.Member) -> discord.AllowedMentions:
    return discord.AllowedMentions(
        everyone=False,
        roles=False,
        users=[user],
        replied_user=False,
    )


async def send_notify_message(
    client: discord.Client,
    settings,
    channel_id: int,
    message_id: int,
    due_hours: Optional[int] = None,
    message: Optional[discord.Message] = None,
    due_at: Optional[datetime] = None,
) -> tuple[str, discord.Message, datetime]:
    if message is None:
        message = await _fetch_message(client, channel_id, message_id)
    if due_at is not None:
        due = due_at.astimezone(JST)
    else:
        hours = due_hours if due_hours is not None else settings.due_hours
        due = datetime.now(JST) + timedelta(hours=hours)
    content = (
        f"{message.author.mention} この投稿はサーバールールに抵触している可能性があります。\n"
        "お手数ですが、ご自身で削除をお願いいたします。\n"
        f"対象: {build_jump_link(message.guild.id, message.channel.id, message.id)}\n"
        f"対応期限: {due:%Y-%m-%d %H:%M} JST"
    )
    await message.reply(content=content, allowed_mentions=_build_allowed_mentions(message.author))
    return content, message, due


def _first_reason(record: Optional[dict]) -> Optional[str]:
    if not record:
        return None
    reasons = record.get("reasons")
    if isinstance(reasons, list) and reasons:
        first = reasons[0]
        return str(first) if first is not None else None
    reason = record.get("reason")
    if isinstance(reason, str) and reason:
        return reason
    return None


def _resolve_ticket_details(record: Optional[dict], message: discord.Message) -> tuple[str, Optional[str], Optional[str], int]:
    severity = (record or {}).get("severity") or "yellow"
    severity = severity.lower()
    if severity not in SEVERITY_COLORS:
        severity = "yellow"
    rule_id = (record or {}).get("rule_id")
    reason = _first_reason(record)
    author_id = message.author.id
    return severity, rule_id, reason, author_id


def _record_matches_message(record: dict, channel_id: int, message_id: int) -> bool:
    def _eq(value, target):
        try:
            return int(value) == target
        except (TypeError, ValueError):
            return False

    if _eq(record.get("channel_id"), channel_id) and _eq(record.get("message_id"), message_id):
        return True
    for entry in record.get("messages", []) or []:
        if not isinstance(entry, dict):
            continue
        if _eq(entry.get("channel_id"), channel_id) and _eq(entry.get("message_id"), message_id):
            return True
    return False


def find_record_for_message(channel_id: int, message_id: int) -> Optional[dict]:
    path = Path("out/p3_findings.jsonl")
    if not path.exists():
        return None
    time_range = (UTC_MIN, UTC_MAX)
    for record in iter_findings(path, time_range=time_range):
        if _record_matches_message(record, channel_id, message_id):
            return record
    return None


def load_findings_index() -> dict[tuple[int, int], dict]:
    path = Path("out/p3_findings.jsonl")
    if not path.exists():
        return {}

    def _key(cid: Any, mid: Any) -> Optional[tuple[int, int]]:
        try:
            return int(cid), int(mid)
        except (TypeError, ValueError):
            return None

    index: dict[tuple[int, int], dict] = {}
    time_range = (UTC_MIN, UTC_MAX)
    for record in iter_findings(path, time_range=time_range):
        key = _key(record.get("channel_id"), record.get("message_id"))
        if key is not None:
            index[key] = record
        for msg in record.get("messages", []) or []:
            key = _key(msg.get("channel_id"), msg.get("message_id"))
            if key is not None and key not in index:
                index[key] = record
    return index


async def process_notification(
    client: discord.Client,
    settings,
    ticket_store: TicketStore,
    *,
    guild_id: int,
    channel_id: int,
    message_id: int,
    executor: discord.abc.User,
    record: Optional[dict] = None,
    due_hours: Optional[int] = None,
) -> tuple[Ticket, bool, str]:
    ticket_id = TicketStore.build_ticket_id(guild_id, channel_id, message_id)
    existing = await ticket_store.get_ticket(ticket_id)
    if existing:
        if existing.status == "notified":
            due_local = existing.due_at.astimezone(JST)
            return (
                existing,
                False,
                f"既に通知済みです。期限: {due_local:%Y-%m-%d %H:%M} JST",
            )
        return (
            existing,
            False,
            f"この案件は既に処理済みです (status={existing.status}).",
        )
    message = await _fetch_message(client, channel_id, message_id)
    severity, rule_id, reason, author_id = _resolve_ticket_details(record, message)
    due_hours_value = due_hours if due_hours is not None else settings.due_hours
    due_jst = datetime.now(JST) + timedelta(hours=due_hours_value)
    ticket, created = await ticket_store.register_notification(
        ticket_id=ticket_id,
        guild_id=guild_id,
        channel_id=channel_id,
        message_id=message_id,
        author_id=author_id,
        severity=severity,
        rule_id=rule_id,
        reason=reason,
        message_link=build_jump_link(guild_id, channel_id, message_id),
        due_at=due_jst.astimezone(timezone.utc),
        executor_id=executor.id,
    )
    if not created:
        if ticket.status == "notified":
            due_local = ticket.due_at.astimezone(JST)
            return (
                ticket,
                False,
                f"既に通知済みです。期限: {due_local:%Y-%m-%d %H:%M} JST",
            )
        return (
            ticket,
            False,
            f"この案件は既に処理済みです (status={ticket.status}).",
        )

    content, _, _ = await send_notify_message(
        client,
        settings,
        channel_id,
        message_id,
        due_hours=due_hours,
        message=message,
        due_at=due_jst,
    )
    await ticket_store.append_log(
        ticket_id=ticket.ticket_id,
        actor_id=executor.id,
        action="notify",
        detail=content,
    )
    return (
        ticket,
        created,
        f"通知を送信しました。期限: {due_jst:%Y-%m-%d %H:%M} JST",
    )


def build_audit_reason(ticket: Ticket, action: str) -> str:
    rule = ticket.rule_id or "none"
    return f"ModBot {action}|rule={rule}|ticket={ticket.ticket_id}"


def _ensure_record_defaults(record: Optional[dict], ticket: Ticket, message: Optional[discord.Message]) -> dict:
    base = dict(record or {})
    base.setdefault("severity", ticket.severity)
    base.setdefault("rule_id", ticket.rule_id)
    if ticket.reason and not base.get("reasons"):
        base["reasons"] = [ticket.reason]
    base.setdefault("message_link", ticket.message_link)
    if message:
        base.setdefault("created_at", message.created_at.astimezone(timezone.utc).isoformat())
        base.setdefault(
            "messages",
            [
                {
                    "channel_id": str(message.channel.id),
                    "message_id": str(message.id),
                    "author_id": str(message.author.id),
                }
            ],
        )
    return base


def build_ticket_log_embed(
    ticket: Ticket,
    *,
    action: str,
    result: str,
    record: Optional[dict] = None,
    message: Optional[discord.Message] = None,
) -> discord.Embed:
    record_data = _ensure_record_defaults(record, ticket, message)
    preview = PreviewInfo(None, None, None, False)
    author = message.author if message else None
    embed = build_record_embed(record_data, 0, 1, preview, author)
    embed.add_field(name="処理", value=action, inline=False)
    embed.add_field(name="結果", value=result, inline=False)
    embed.set_footer(text=f"チケット {ticket.ticket_id}")
    return embed


async def _resolve_log_channel(client: discord.Client, settings) -> Optional[discord.abc.Messageable]:
    channel_id = settings.log_channel_id
    if not channel_id:
        return None
    channel = client.get_channel(int(channel_id))
    if channel is None:
        try:
            channel = await client.fetch_channel(int(channel_id))
        except Exception:  # noqa: BLE001
            logger.exception("log channel fetch failed")
            return None
    if isinstance(channel, discord.abc.Messageable):
        return channel
    logger.warning("log channel %s is not messageable", channel_id)
    return None


async def send_ticket_log(
    client: discord.Client,
    settings,
    ticket: Ticket,
    *,
    action: str,
    result: str,
    record: Optional[dict] = None,
    message: Optional[discord.Message] = None,
) -> None:
    channel = await _resolve_log_channel(client, settings)
    if channel is None:
        return
    embed = build_ticket_log_embed(ticket, action=action, result=result, record=record, message=message)
    await channel.send(embed=embed)


async def handle_due_ticket(
    client: discord.Client,
    settings,
    ticket_store: TicketStore,
    ticket: Ticket,
) -> None:
    record = find_record_for_message(ticket.channel_id, ticket.message_id)
    actor_id = client.user.id if client.user else 0
    try:
        message = await _fetch_message(client, ticket.channel_id, ticket.message_id)
    except discord.NotFound:
        await ticket_store.update_status(ticket.ticket_id, status="author_deleted")
        await ticket_store.append_log(
            ticket_id=ticket.ticket_id,
            actor_id=actor_id,
            action="author_deleted",
            detail="投稿者が既に削除済み",
        )
        await send_ticket_log(
            client,
            settings,
            ticket,
            action="自動削除（期限切れ）",
            result="投稿者によって既に削除されています",
            record=record,
        )
        return
    except discord.HTTPException as exc:  # includes Forbidden
        await ticket_store.update_status(ticket.ticket_id, status="failed")
        await ticket_store.append_log(
            ticket_id=ticket.ticket_id,
            actor_id=actor_id,
            action="auto_delete_failed",
            detail=str(exc),
        )
        await send_ticket_log(
            client,
            settings,
            ticket,
            action="自動削除（期限切れ）",
            result=f"メッセージの取得に失敗しました: {exc}",
            record=record,
        )
        return

    delete_reason = build_audit_reason(ticket, "auto_delete")
    try:
        await message.delete(reason=delete_reason)
    except TypeError as exc:
        # discord.PartialMessage.delete などは reason を受け付けないためフォールバックする
        if "reason" not in str(exc):
            raise
        await message.delete()
    except discord.HTTPException as exc:
        await ticket_store.update_status(ticket.ticket_id, status="failed")
        await ticket_store.append_log(
            ticket_id=ticket.ticket_id,
            actor_id=actor_id,
            action="auto_delete_failed",
            detail=str(exc),
        )
        await send_ticket_log(
            client,
            settings,
            ticket,
            action="自動削除（期限切れ）",
            result=f"メッセージの削除に失敗しました: {exc}",
            record=record,
            message=message,
        )
        return

    await ticket_store.update_status(ticket.ticket_id, status="bot_deleted")
    await ticket_store.append_log(
        ticket_id=ticket.ticket_id,
        actor_id=actor_id,
        action="auto_delete",
        detail="ボットが削除を実行",
    )
    await send_ticket_log(
        client,
        settings,
        ticket,
        action="自動削除（期限切れ）",
        result="ボットがメッセージを削除しました",
        record=record,
        message=message,
    )


async def process_ticket_cancel(
    client: discord.Client,
    settings,
    ticket_store: TicketStore,
    *,
    ticket: Ticket,
    executor: discord.abc.User,
    record: Optional[dict] = None,
) -> tuple[str, Optional[Ticket]]:
    if ticket.status != "notified":
        return "このチケットは既に処理済みです。", None
    updated = await ticket_store.cancel_ticket(ticket.ticket_id, actor_id=executor.id)
    if updated is None:
        return "チケットが見つかりませんでした。", None
    record_data = record or find_record_for_message(ticket.channel_id, ticket.message_id)
    record_data = _ensure_record_defaults(record_data, updated, None)
    await send_ticket_log(
        client,
        settings,
        updated,
        action="キャンセル",
        result="モデレーターがキャンセルしました",
        record=record_data,
    )
    return f"チケット {updated.ticket_id} をキャンセルしました。", updated


async def ticket_watcher(
    client: discord.Client,
    settings,
    ticket_store: TicketStore,
) -> None:
    interval = max(60, int(settings.ticket_poll_interval))
    await client.wait_until_ready()
    try:
        while not client.is_closed():
            try:
                due_tickets = await ticket_store.fetch_due_tickets()
                for ticket in due_tickets:
                    await handle_due_ticket(client, settings, ticket_store, ticket)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("ticket watcher cycle failed")
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("ticket watcher cancelled")
        raise


def severity_color(severity: str) -> discord.Color:
    return SEVERITY_COLORS.get(severity, discord.Color.blurple())


def build_summary_embed(
    summary,
    channel: Optional[discord.abc.GuildChannel],
    since: Optional[str],
    until: Optional[str],
    severity: str,
    requester: discord.User,
    time_range: Optional[tuple[datetime, datetime]] = None,
) -> discord.Embed:
    start, end = time_range if time_range is not None else resolve_time_range(since, until)
    dominant = next((s for s in SEVERITY_ORDER if summary.severity_counts.get(s)), "green")
    embed = discord.Embed(
        title="スキャン概要",
        color=severity_color(dominant),
        timestamp=datetime.now(timezone.utc),
    )
    target_name = channel.mention if isinstance(channel, discord.abc.GuildChannel) else "（全チャンネル）"
    embed.add_field(name="対象チャンネル", value=target_name, inline=False)
    embed.add_field(
        name="期間",
        value=f"{start.isoformat()} → {end.isoformat()}",
        inline=False,
    )
    embed.add_field(name="対象深刻度", value=_severity_option_label(severity or "all"), inline=False)
    embed.add_field(name="合計件数", value=str(summary.total), inline=True)
    for sev in SEVERITY_ORDER:
        embed.add_field(
            name=_severity_field_label(sev),
            value=str(summary.severity_counts.get(sev, 0)),
            inline=True,
        )
    embed.set_footer(text=f"実行者: {requester.display_name}")
    return embed


def build_record_embed(
    record: dict,
    index: int,
    total: int,
    preview: PreviewInfo,
    author: Optional[discord.abc.User] = None,
) -> discord.Embed:
    """Render a record embed using rule_title → rule_id → 既定タイトルの順でフォールバックする。"""
    severity = record.get("severity", "green")
    color = severity_color(severity)
    base_title = record.get("rule_title") or record.get("rule_id") or DEFAULT_RECORD_TITLE
    title = f"{_severity_badge(severity)} {base_title}"
    embed = discord.Embed(
        title=title,
        url=record.get("message_link"),
        color=color,
        timestamp=datetime.now(timezone.utc),
    )

    reasons = record.get("reasons") or []
    if reasons:
        reason_line = "; ".join(reasons[:2])
        if len(reasons) > 2:
            reason_line += " …"
        embed.description = reason_line

    author_id = record.get("author_id") or (record.get("messages") or [{}])[0].get("author_id")
    author_name = None
    author_icon = None
    if author_id:
        if author:
            author_name = author.display_name
            author_icon = author.display_avatar.url if author.display_avatar else None
        else:
            author_name = f"ユーザー {author_id}"
    posted_value = _format_timestamp(record.get("created_at"))
    if posted_value:
        embed.add_field(name="投稿日時", value=posted_value, inline=False)
    if author_name and author_id:
        embed.add_field(name="投稿者", value=f"<@{author_id}>", inline=False)
    if author_name:
        author_kwargs: dict[str, str] = {
            "name": f"@{author_name}",
            "url": record.get("message_link") or discord.utils.MISSING,
        }
        if author_icon:
            author_kwargs["icon_url"] = author_icon
        embed.set_author(**author_kwargs)

    metrics = record.get("metrics", {})
    features = _v2_features(record)
    v2_errors = _v2_errors(record)
    tag_scores = _collect_tag_scores(record)

    try:
        engine = _get_rule_engine()
    except Exception:  # noqa: BLE001
        engine = None

    groups_config: dict[str, tuple[str, ...]] = {}
    consts: Mapping[str, float] = {}
    rule_map: dict[str, Mapping[str, Any]] = {}
    if engine is not None:
        raw_config = engine.raw_config or {}
        raw_groups = raw_config.get("groups")
        if isinstance(raw_groups, Mapping):
            groups_config = {
                str(name): tuple(str(item) for item in values if isinstance(item, str))
                for name, values in raw_groups.items()
                if isinstance(values, (list, tuple))
            }
        consts = getattr(engine, "const_map", {}) or {}
        raw_rules = raw_config.get("rules")
        if isinstance(raw_rules, list):
            for entry in raw_rules:
                if isinstance(entry, Mapping) and isinstance(entry.get("id"), str):
                    rule_map[str(entry["id"])] = entry

    selected_features: list[tuple[str, float]] = []
    seen_features: set[str] = set()
    for key in _FEATURE_PRIORITY:
        if key in features and key not in seen_features:
            selected_features.append((key, features[key]))
            seen_features.add(key)
        if len(selected_features) >= 8:
            break
    if len(selected_features) < 8:
        for name in sorted(features, key=features.get, reverse=True):
            if name in seen_features:
                continue
            selected_features.append((name, features[name]))
            seen_features.add(name)
            if len(selected_features) >= 8:
                break
    if selected_features:
        feature_lines = [f"{name}: {value:.3f}" for name, value in selected_features[:8]]
        block = _truncate_block(feature_lines)
        if block:
            embed.add_field(name="V2 features（上位）", value=block, inline=False)

    if groups_config:
        group_lines: list[str] = []
        for group_name in _DISPLAY_GROUPS:
            patterns = groups_config.get(group_name) or groups_config.get(normalize_tag(group_name)) or ()
            hits = group_top_tags(tag_scores, patterns, k=5, min_score=0.10)
            if hits:
                group_lines.append(
                    f"{group_name}: " + ", ".join(f"{tag}:{score:.2f}" for tag, score in hits)
                )
        block = _truncate_block(group_lines, code=False, max_chars=900)
        if block:
            embed.add_field(name="タググループ（top hits）", value=block, inline=False)

    embed.add_field(name="WD14 rating", value=_format_wd14_rating_block(record), inline=False)
    nudity_list = _format_top_detections(record.get("nudity_detections", []), limit=4)
    embed.add_field(
        name="NudeNet 検出",
        value=", ".join(nudity_list) if nudity_list else "なし",
        inline=False,
    )

    winning = metrics.get("winning") or {}
    rule_id = winning.get("rule_id") if isinstance(winning, Mapping) else None
    if rule_id and rule_id in rule_map:
        when_text = rule_map[rule_id].get("when") if isinstance(rule_map[rule_id], Mapping) else None
        if isinstance(when_text, str) and when_text.strip():
            lines = _build_when_summary_lines(
                when_text,
                record=record,
                features=features,
                metrics=metrics,
                consts=consts,
                groups=groups_config,
                tag_scores=tag_scores,
                limit=6,
            )
            block = _truncate_block(lines)
            if block:
                embed.add_field(name=f"when 概要（{rule_id}）", value=block, inline=False)

    if v2_errors:
        embed.add_field(name="⚠️ V2エラー", value=" / ".join(v2_errors), inline=False)

    if preview.image_url:
        embed.set_image(url=preview.image_url)

    embed.set_footer(text=f"アイテム {index + 1}/{total} ・ phash={record.get('phash', '')}")
    return embed


class RecordPaginatorBase(discord.ui.View):
    def __init__(
        self,
        client: discord.Client,
        settings,
        requester: discord.User,
        items: Sequence[Any],
        *,
        timeout: int = 600,
    ) -> None:
        super().__init__(timeout=timeout)
        self.client = client
        self.settings = settings
        self.requester = requester
        self.items = list(items)
        self.index = 0
        self.message: Optional[discord.Message] = None
        self._preview_cache: dict[tuple[int, int, str], PreviewInfo] = {}
        self._member_cache: dict[int, discord.Member] = {}
        self._current_preview: PreviewInfo | None = None
        self._open_message_button = discord.ui.Button(
            label="投稿を開く",
            style=discord.ButtonStyle.link,
            url="https://discord.com",
            row=1,
            disabled=True,
        )
        self._open_original_button = discord.ui.Button(
            label="元画像を開く",
            style=discord.ButtonStyle.link,
            url="https://discord.com",
            disabled=True,
            row=1,
        )
        self.add_item(self._open_message_button)
        self.add_item(self._open_original_button)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.requester.id:
            await interaction.response.send_message(
                "このビューを操作できるのは発行者のみです。",
                ephemeral=True,
            )
            return False
        return True

    async def on_timeout(self) -> None:
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                child.disabled = True
        if self.message:
            success = await self._safe_edit_message(view=self)
            if not success:
                logger.debug("view message became unavailable on timeout")

    def _empty_message(self) -> str:
        raise NotImplementedError

    def _record_from_item(self, item: Any) -> dict:
        raise NotImplementedError

    def _build_embed(
        self,
        record: dict,
        preview: PreviewInfo,
        author: Optional[discord.abc.User],
        *,
        index: int,
        total: int,
    ) -> discord.Embed:
        raise NotImplementedError

    def _update_additional_buttons(self, has_items: bool) -> None:
        return

    def _has_items(self) -> bool:
        return bool(self.items)

    def _current_item(self) -> Any:
        return self.items[self.index]

    def _current_record(self) -> dict:
        return self._record_from_item(self._current_item())

    def _record_key(self, record: dict) -> Optional[tuple[int, int, str]]:
        channel_id = record.get("channel_id") or (record.get("messages") or [{}])[0].get("channel_id")
        message_id = record.get("message_id") or (record.get("messages") or [{}])[0].get("message_id")
        phash = record.get("phash") or str(record.get("message_id") or "")
        if channel_id and message_id:
            try:
                return int(channel_id), int(message_id), str(phash)
            except (TypeError, ValueError):
                return None
        return None

    async def _resolve_member(self, record: dict) -> Optional[discord.Member]:
        author_id = record.get("author_id") or (record.get("messages") or [{}])[0].get("author_id")
        if not author_id:
            return None
        try:
            author_int = int(author_id)
        except (TypeError, ValueError):
            return None
        cached = self._member_cache.get(author_int)
        if cached:
            return cached
        guild = None
        preferred_guild_id = (
            self.settings.guild_id
            or record.get("guild_id")
            or (record.get("messages") or [{}])[0].get("guild_id")
        )
        if preferred_guild_id:
            try:
                gid = int(preferred_guild_id)
                guild = self.client.get_guild(gid)
                if guild is None:
                    guild = await self.client.fetch_guild(gid)
            except Exception:  # noqa: BLE001
                guild = None
        if not guild:
            return None
        try:
            member = guild.get_member(author_int) or await guild.fetch_member(author_int)
        except Exception:  # noqa: BLE001
            return None
        self._member_cache[author_int] = member
        return member

    async def _ensure_preview(self, record: Optional[dict] = None) -> PreviewInfo:
        if record is None:
            record = self._current_record()
        key = self._record_key(record)
        if key and key in self._preview_cache:
            return self._preview_cache[key]
        info = await self._resolve_preview(record)
        if key:
            self._preview_cache[key] = info
        return info

    async def _resolve_preview(self, record: dict) -> PreviewInfo:
        urls: set[str] = set()
        for msg in record.get("messages", []) or []:
            url = msg.get("url")
            if url:
                urls.add(str(url))

        def first_image_from_messages(messages: Iterable[dict]) -> Optional[dict]:
            for msg in messages or []:
                for attachment in msg.get("attachments", []) or []:
                    content_type = attachment.get("content_type") or ""
                    if not content_type.startswith("image/"):
                        continue
                    url = attachment.get("url")
                    if urls and url not in urls:
                        continue
                    return attachment
                url = msg.get("url")
                if url and (not urls or url in urls):
                    return {
                        "url": url,
                        "content_type": "image/unknown",
                        "is_spoiler": bool(msg.get("is_spoiler")),
                    }
            return None

        preview_data = record.get("preview")
        attachment = None
        if isinstance(preview_data, dict):
            url = preview_data.get("source_url") or preview_data.get("url")
            if url:
                attachment = {
                    "url": url,
                    "content_type": "image/unknown",
                    "is_spoiler": bool(preview_data.get("is_spoiler")),
                }
        if attachment is None:
            attachment = first_image_from_messages(record.get("messages", []))
        if attachment is None:
            targets = self._message_targets_for(record)
            if targets:
                try:
                    message = await _fetch_message(self.client, targets[0], targets[1])
                except Exception:  # noqa: BLE001
                    message = None
                if message:
                    for att in message.attachments:
                        if not (att.content_type and att.content_type.startswith("image/")):
                            continue
                        if urls and att.url not in urls:
                            continue
                        attachment = {
                            "url": att.url,
                            "proxy_url": att.proxy_url,
                            "content_type": att.content_type,
                            "is_spoiler": att.is_spoiler(),
                        }
                        break
                    if attachment is None:
                        for emb in message.embeds:
                            if emb.image:
                                if urls and emb.image.url not in urls:
                                    continue
                                attachment = {
                                    "url": emb.image.url,
                                    "content_type": "image/embedded",
                                    "is_spoiler": False,
                                }
                                break
        if attachment is None:
            return PreviewInfo(None, None, None, False)
        thumbnail_url = attachment.get("proxy_url") or attachment.get("url")
        image_url = attachment.get("url")
        original_url = attachment.get("url")
        is_spoiler = bool(attachment.get("is_spoiler"))
        return PreviewInfo(thumbnail_url, image_url, original_url, is_spoiler)

    def _message_targets(self) -> Optional[tuple[int, int]]:
        if not self._has_items():
            return None
        return self._message_targets_for(self._current_record())

    def _message_targets_for(self, record: dict) -> Optional[tuple[int, int]]:
        channel_id = record.get("channel_id")
        message_id = record.get("message_id")
        try:
            if channel_id is not None and message_id is not None:
                return int(channel_id), int(message_id)
        except (TypeError, ValueError):
            pass
        messages = record.get("messages") or []
        if not messages:
            return None
        entry = messages[0]
        try:
            return int(entry.get("channel_id")), int(entry.get("message_id"))
        except (TypeError, ValueError):
            return None

    async def _safe_edit_message(self, **kwargs) -> bool:
        if not self.message:
            return False
        try:
            await self.message.edit(**kwargs)
            return True
        except discord.HTTPException as exc:
            if exc.status in {401, 404}:
                logger.warning("view message edit failed (%s), detaching view", exc.status)
                self.message = None
                return False
            raise

    async def _jump_to_index(self, new_index: int) -> None:
        if not self._has_items():
            return
        clamped = max(0, min(new_index, len(self.items) - 1))
        if clamped != self.index:
            self.index = clamped
            self._current_preview = None
        await self._refresh_message()

    def _update_button_state(self) -> None:
        has_items = self._has_items()
        for child in self.children:
            if isinstance(child, discord.ui.Button):
                if child.custom_id == "prev":
                    child.disabled = not has_items or self.index == 0
                elif child.custom_id == "next":
                    child.disabled = not has_items or self.index >= len(self.items) - 1
        if not has_items:
            self._open_message_button.url = "https://discord.com"
            self._open_message_button.disabled = True
            self._open_original_button.url = "https://discord.com"
            self._open_original_button.disabled = True
            self._update_additional_buttons(has_items)
            return
        record = self._current_record()
        message_url = record.get("message_link")
        if message_url:
            self._open_message_button.url = message_url
            self._open_message_button.disabled = False
        else:
            self._open_message_button.url = "https://discord.com"
            self._open_message_button.disabled = True
        preview = self._current_preview or PreviewInfo(None, None, None, False)
        if preview.original_url:
            self._open_original_button.url = preview.original_url
            self._open_original_button.disabled = False
        else:
            self._open_original_button.url = message_url or "https://discord.com"
            self._open_original_button.disabled = True
        self._update_additional_buttons(has_items)

    @discord.ui.button(label="ページ指定", style=discord.ButtonStyle.secondary, custom_id="jump", row=0)
    async def on_jump(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self._has_items():
            await interaction.response.send_message(self._empty_message(), ephemeral=True)
            return
        total = len(self.items)
        modal = _PageJumpModal(self, total)
        await interaction.response.send_modal(modal)

    async def send_initial(self, interaction: discord.Interaction) -> None:
        if not self._has_items():
            await interaction.followup.send(self._empty_message(), ephemeral=True)
            return
        self.index = min(self.index, len(self.items) - 1)
        record = self._current_record()
        self._current_preview = await self._ensure_preview(record)
        author = await self._resolve_member(record)
        embed = self._build_embed(
            record,
            self._current_preview,
            author,
            index=self.index,
            total=len(self.items),
        )
        self._update_button_state()
        self.message = await interaction.followup.send(embed=embed, view=self, ephemeral=True)

    async def _respond_refresh(self, interaction: discord.Interaction) -> None:
        if not self._has_items():
            await interaction.response.edit_message(content=self._empty_message(), embed=None, view=None)
            return
        record = self._current_record()
        self._current_preview = await self._ensure_preview(record)
        author = await self._resolve_member(record)
        embed = self._build_embed(
            record,
            self._current_preview,
            author,
            index=self.index,
            total=len(self.items),
        )
        self._update_button_state()
        await interaction.response.edit_message(embed=embed, view=self)

    async def _refresh_message(self) -> None:
        if not self.message:
            return
        if not self._has_items():
            self._update_button_state()
            success = await self._safe_edit_message(
                content=self._empty_message(),
                embed=None,
                view=None,
            )
            if not success:
                logger.debug("view message became unavailable while clearing content")
            return
        record = self._current_record()
        self._current_preview = await self._ensure_preview(record)
        author = await self._resolve_member(record)
        embed = self._build_embed(
            record,
            self._current_preview,
            author,
            index=self.index,
            total=len(self.items),
        )
        self._update_button_state()
        success = await self._safe_edit_message(embed=embed, view=self)
        if not success:
            logger.debug("view message became unavailable while refreshing")

    @discord.ui.button(label="◀", style=discord.ButtonStyle.secondary, custom_id="prev", row=0)
    async def on_prev(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self._has_items():
            await interaction.response.send_message(self._empty_message(), ephemeral=True)
            return
        if self.index > 0:
            self.index -= 1
        await self._respond_refresh(interaction)

    @discord.ui.button(label="▶", style=discord.ButtonStyle.secondary, custom_id="next", row=0)
    async def on_next(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self._has_items():
            await interaction.response.send_message(self._empty_message(), ephemeral=True)
            return
        if self.index < len(self.items) - 1:
            self.index += 1
        await self._respond_refresh(interaction)


class _PageJumpModal(discord.ui.Modal):
    def __init__(self, paginator: RecordPaginatorBase, total: int) -> None:
        super().__init__(title="ページ指定")
        self.paginator = paginator
        self.total = total
        placeholder = f"1 〜 {max(total, 1)}"
        self.page_input = discord.ui.TextInput(
            label="ページ番号",
            placeholder=placeholder,
            min_length=1,
            max_length=7,
        )
        self.add_item(self.page_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        total = len(self.paginator.items)
        if total == 0:
            await interaction.response.send_message(
                self.paginator._empty_message(),
                ephemeral=True,
            )
            return
        raw_value = self.page_input.value.strip()
        try:
            page = int(raw_value)
        except ValueError:
            await interaction.response.send_message(
                "整数のページ番号を入力してください。",
                ephemeral=True,
            )
            return
        if page < 1 or page > total:
            await interaction.response.send_message(
                f"1 〜 {total} の範囲で指定してください。",
                ephemeral=True,
            )
            return
        await self.paginator._jump_to_index(page - 1)
        await interaction.response.send_message(
            f"ページ {page}/{total} を表示しました。",
            ephemeral=True,
        )


class ReportPaginator(RecordPaginatorBase):
    def __init__(
        self,
        client: discord.Client,
        settings,
        ticket_store: TicketStore,
        records: Sequence[dict],
        requester: discord.User,
    ) -> None:
        super().__init__(client, settings, requester, records)
        self.ticket_store = ticket_store
        self._busy_actions: set[str] = set()
        self._message_cache: dict[tuple[int, int], discord.Message] = {}

    def _empty_message(self) -> str:
        return "表示可能な検知がありません。"

    def _record_from_item(self, item: Any) -> dict:
        return item

    def _update_additional_buttons(self, has_items: bool) -> None:
        targets_available = has_items and self._message_targets() is not None
        log_channel_available = bool(self.settings.log_channel_id)
        for child in self.children:
            if not isinstance(child, discord.ui.Button):
                continue
            cid = child.custom_id
            if cid == "notify":
                child.disabled = (
                    not has_items
                    or not targets_available
                    or self._is_action_busy("notify")
                )
            elif cid == "log":
                child.disabled = (
                    not has_items
                    or not targets_available
                    or not log_channel_available
                    or self._is_action_busy("log")
                )
            elif cid == "delete":
                child.disabled = (
                    not has_items
                    or not targets_available
                    or self._is_action_busy("delete")
                )

    def _is_action_busy(self, action: str) -> bool:
        return action in self._busy_actions

    def _set_action_busy(self, action: str, value: bool) -> None:
        if value:
            self._busy_actions.add(action)
        else:
            self._busy_actions.discard(action)
        self._update_button_state()

    async def _fetch_current_message(self, *, use_cache: bool = True) -> Optional[discord.Message]:
        targets = self._message_targets()
        if targets is None:
            return None
        channel_id, message_id = targets
        cache_key = (channel_id, message_id)
        if use_cache and cache_key in self._message_cache:
            return self._message_cache[cache_key]
        try:
            message = await _fetch_message(self.client, channel_id, message_id)
        except discord.HTTPException:
            return None
        self._message_cache[cache_key] = message
        return message

    def _record_guild_id(self, record: dict) -> Optional[int]:
        candidates = [record.get("guild_id")]
        for entry in record.get("messages", []) or []:
            candidates.append(entry.get("guild_id"))
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return int(candidate)
            except (TypeError, ValueError):
                continue
        if self.settings.guild_id:
            try:
                return int(self.settings.guild_id)
            except (TypeError, ValueError):
                return None
        return None

    async def _resolve_ticket(self, message: Optional[discord.Message]) -> Optional[Ticket]:
        targets = self._message_targets()
        if targets is None:
            return None
        channel_id, message_id = targets
        guild_id = None
        if message and message.guild:
            guild_id = message.guild.id
        if guild_id is None:
            guild_id = self._record_guild_id(self._current_record())
        if guild_id is None:
            return None
        ticket_id = TicketStore.build_ticket_id(guild_id, channel_id, message_id)
        return await self.ticket_store.get_ticket(ticket_id)

    async def _send_record_only_log(
        self,
        *,
        action: str,
        result: str,
        record: dict,
        message: Optional[discord.Message],
        executor: discord.abc.User,
    ) -> bool:
        channel = await _resolve_log_channel(self.client, self.settings)
        if channel is None:
            return False
        preview = self._current_preview or await self._ensure_preview(record)
        author = message.author if message else None
        embed = build_record_embed(record, self.index, len(self.items) or 1, preview, author)
        embed.add_field(name="処理", value=action, inline=False)
        embed.add_field(name="結果", value=result, inline=False)
        footer_text = embed.footer.text if embed.footer else ""
        footer_parts = [part for part in (footer_text, f"operator={executor.display_name}") if part]
        embed.set_footer(text=" ・ ".join(footer_parts))
        await channel.send(embed=embed)
        return True

    def _build_embed(
        self,
        record: dict,
        preview: PreviewInfo,
        author: Optional[discord.abc.User],
        *,
        index: int,
        total: int,
    ) -> discord.Embed:
        return build_record_embed(record, index, total, preview, author)

    def _clear_message_cache(self) -> None:
        targets = self._message_targets()
        if targets is None:
            return
        self._message_cache.pop(targets, None)

    def _remove_current_item(self) -> None:
        if not self._has_items():
            return
        self.items.pop(self.index)
        if self.items and self.index >= len(self.items):
            self.index = len(self.items) - 1
        self._current_preview = None
        self._clear_message_cache()

    @discord.ui.button(label="通知", style=discord.ButtonStyle.primary, custom_id="notify", row=2)
    async def on_notify(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self._has_items():
            await interaction.response.send_message(self._empty_message(), ephemeral=True)
            return
        if self._message_targets() is None:
            await interaction.response.send_message("対象メッセージを特定できませんでした。", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        self._set_action_busy("notify", True)
        try:
            message = await self._fetch_current_message()
            if message is None:
                await interaction.followup.send("メッセージを取得できませんでした。", ephemeral=True)
                return
            record = self._current_record()
            try:
                guild_id = message.guild.id if message.guild else self._record_guild_id(record)
                if guild_id is None:
                    await interaction.followup.send("ギルド情報を特定できませんでした。", ephemeral=True)
                    return
                ticket, created, response = await process_notification(
                    self.client,
                    self.settings,
                    self.ticket_store,
                    guild_id=guild_id,
                    channel_id=message.channel.id,
                    message_id=message.id,
                    executor=interaction.user,
                    record=record,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("notify failed")
                await interaction.followup.send(f"通知に失敗しました: {exc}", ephemeral=True)
                return
            await interaction.followup.send(response, ephemeral=True)
        finally:
            self._set_action_busy("notify", False)

    @discord.ui.button(label="モデログ転送", style=discord.ButtonStyle.secondary, custom_id="log", row=2)
    async def on_log(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self._has_items():
            await interaction.response.send_message(self._empty_message(), ephemeral=True)
            return
        if not self.settings.log_channel_id:
            await interaction.response.send_message("モデログ用チャンネルが設定されていません。", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        self._set_action_busy("log", True)
        try:
            message = await self._fetch_current_message()
            record = self._current_record()
            ticket = await self._resolve_ticket(message)
            action_label = "手動ログ"
            executor = interaction.user
            if ticket:
                result_text = f"{executor.display_name} が手動でログを送信"
                try:
                    await send_ticket_log(
                        self.client,
                        self.settings,
                        ticket,
                        action=action_label,
                        result=result_text,
                        record=record,
                        message=message,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("manual log failed")
                    await interaction.followup.send(f"モデログへの送信に失敗しました: {exc}", ephemeral=True)
                    return
            else:
                result_text = f"{executor.display_name} が手動でログを送信"
                success = await self._send_record_only_log(
                    action=action_label,
                    result=result_text,
                    record=record,
                    message=message,
                    executor=executor,
                )
                if not success:
                    await interaction.followup.send("モデログチャンネルにアクセスできません。", ephemeral=True)
                    return
            await interaction.followup.send("モデログへ転送しました。", ephemeral=True)
        finally:
            self._set_action_busy("log", False)

    @discord.ui.button(label="削除", style=discord.ButtonStyle.danger, custom_id="delete", row=2)
    async def on_delete(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self._has_items():
            await interaction.response.send_message(self._empty_message(), ephemeral=True)
            return
        if self._message_targets() is None:
            await interaction.response.send_message("対象メッセージを特定できませんでした。", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        self._set_action_busy("delete", True)
        try:
            message = await self._fetch_current_message(use_cache=False)
            if message is None:
                await interaction.followup.send("メッセージを取得できませんでした。", ephemeral=True)
                return
            record = self._current_record()
            ticket = await self._resolve_ticket(message)
            reason = build_audit_reason(ticket, "manual_delete") if ticket is not None else None
            try:
                if reason is not None:
                    await message.delete(reason=reason)
                else:
                    await message.delete()
            except TypeError as exc:
                if "reason" not in str(exc):
                    logger.exception("delete failed")
                    await interaction.followup.send(f"削除に失敗しました: {exc}", ephemeral=True)
                    return
                await message.delete()
            except discord.NotFound:
                logger.info("message already deleted by another actor")
            except discord.HTTPException as exc:
                logger.exception("delete failed")
                await interaction.followup.send(f"削除に失敗しました: {exc}", ephemeral=True)
                return

            executor = interaction.user
            result_text = f"{executor.display_name} が投稿を削除しました"
            if ticket is not None:
                await self.ticket_store.update_status(ticket.ticket_id, status="mod_deleted")
                await self.ticket_store.append_log(
                    ticket_id=ticket.ticket_id,
                    actor_id=executor.id,
                    action="manual_delete",
                    detail="モデレーターが削除",
                )
                await send_ticket_log(
                    self.client,
                    self.settings,
                    ticket,
                    action="モデレーター削除",
                    result=result_text,
                    record=record,
                    message=message,
                )
            else:
                success = await self._send_record_only_log(
                    action="モデレーター削除",
                    result=result_text,
                    record=record,
                    message=message,
                    executor=executor,
                )
                if not success:
                    await interaction.followup.send("モデログチャンネルにアクセスできません。", ephemeral=True)
                    return

            self._remove_current_item()
            await self._refresh_message()
            await interaction.followup.send("投稿を削除しました。", ephemeral=True)
        finally:
            self._set_action_busy("delete", False)


class TicketPaginator(RecordPaginatorBase):
    def __init__(
        self,
        client: discord.Client,
        settings,
        ticket_store: TicketStore,
        entries: Sequence[TicketEntry],
        requester: discord.User,
    ) -> None:
        super().__init__(client, settings, requester, entries)
        self.ticket_store = ticket_store

    def _empty_message(self) -> str:
        return "通知待ちのチケットはありません。"

    def _record_from_item(self, item: Any) -> dict:
        entry: TicketEntry = item
        return entry.record

    def _current_entry(self) -> TicketEntry:
        return self._current_item()

    def _current_ticket(self) -> Ticket:
        return self._current_entry().ticket

    def _build_embed(
        self,
        record: dict,
        preview: PreviewInfo,
        author: Optional[discord.abc.User],
        *,
        index: int,
        total: int,
    ) -> discord.Embed:
        embed = build_record_embed(record, index, total, preview, author)
        ticket = self._current_ticket()
        due_local = ticket.due_at.astimezone(JST)
        embed.add_field(name="通知期限", value=f"{due_local:%Y-%m-%d %H:%M} JST", inline=False)
        embed.add_field(name="チケットID", value=ticket.ticket_id, inline=False)
        footer_text = embed.footer.text if embed.footer else ""
        footer_parts = [part for part in (footer_text, f"ticket={ticket.ticket_id}") if part]
        embed.set_footer(text=" ・ ".join(footer_parts))
        return embed

    def _update_additional_buttons(self, has_items: bool) -> None:
        for child in self.children:
            if not isinstance(child, discord.ui.Button):
                continue
            if child.custom_id == "log":
                child.disabled = not has_items or not bool(self.settings.log_channel_id)
            elif child.custom_id == "cancel":
                child.disabled = not has_items or self._message_targets() is None

    @discord.ui.button(label="キャンセル", style=discord.ButtonStyle.danger, custom_id="cancel", row=2)
    async def on_cancel(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self._has_items():
            await interaction.response.send_message(self._empty_message(), ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        entry = self._current_entry()
        message_text, updated = await process_ticket_cancel(
            self.client,
            self.settings,
            self.ticket_store,
            ticket=entry.ticket,
            executor=interaction.user,
            record=entry.record,
        )
        if updated is not None:
            self.items.pop(self.index)
            if self.items:
                if self.index >= len(self.items):
                    self.index = len(self.items) - 1
                self._current_preview = None
            await self._refresh_message()
        await interaction.followup.send(message_text, ephemeral=True)

    @discord.ui.button(label="モデログ転送", style=discord.ButtonStyle.secondary, custom_id="log", row=2)
    async def on_log(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if not self._has_items():
            await interaction.response.send_message(self._empty_message(), ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        channel_id = self.settings.log_channel_id
        if not channel_id:
            await interaction.followup.send("モデログ用チャンネルが設定されていません。", ephemeral=True)
            return
        channel = self.client.get_channel(int(channel_id))
        if channel is None:
            try:
                channel = await self.client.fetch_channel(int(channel_id))
            except Exception as exc:  # noqa: BLE001
                logger.exception("fetch log channel failed")
                await interaction.followup.send(f"モデログチャンネルにアクセスできません: {exc}", ephemeral=True)
                return
        record = self._current_record()
        preview = self._current_preview or await self._ensure_preview(record)
        author = await self._resolve_member(record)
        embed = self._build_embed(
            record,
            preview,
            author,
            index=self.index,
            total=len(self.items),
        )
        embed.set_footer(text=f"転送者: {interaction.user.display_name}")
        await channel.send(embed=embed)
        await interaction.followup.send("モデログへ転送しました。", ephemeral=True)

def register_commands(
    client: discord.Client,
    tree: discord.app_commands.CommandTree,
    settings,
    ticket_store: TicketStore,
) -> None:
    guild = discord.Object(id=settings.guild_id) if settings.guild_id else None

    @tree.command(name="ping", description="動作確認", guild=guild)
    async def ping(interaction: discord.Interaction) -> None:
        await interaction.response.send_message("ぽん")

    @tree.command(name="notify", description="削除依頼を返信", guild=guild)
    @app_commands.guild_only()
    @app_commands.describe(
        message_link="通知対象のメッセージリンク",
        due_hours="期限（時間）。未指定は設定値。",
    )
    async def notify(
        interaction: discord.Interaction,
        message_link: str,
        due_hours: int | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            guild_id, channel_id, message_id = parse_message_link(message_link)
        except ValueError:
            await interaction.followup.send("メッセージリンクの形式が正しくありません。", ephemeral=True)
            return

        record = find_record_for_message(channel_id, message_id)
        try:
            _, _, response = await process_notification(
                client,
                settings,
                ticket_store,
                guild_id=guild_id,
                channel_id=channel_id,
                message_id=message_id,
                executor=interaction.user,
                record=record,
                due_hours=due_hours,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("notify failed")
            await interaction.followup.send(f"通知に失敗しました: {exc}", ephemeral=True)
            return
        await interaction.followup.send(response, ephemeral=True)

    @tree.command(name="scan", description="WD14/NudeNet の結果をスキャン", guild=guild)
    @app_commands.guild_only()
    @app_commands.describe(
        channel="対象チャンネル（未指定は全チャンネル）",
        since="開始日（未指定は24時間前 / YYYY-MM-DD や 7d, 24h 等に対応）",
        until="終了日（未指定は現在時刻）",
        severity="対象とする深刻度（未指定は全件）",
        post_summary="モデログへサマリを投稿",
    )
    @app_commands.choices(
        severity=[
            app_commands.Choice(name=SEVERITY_OPTION_LABELS["alerts"], value="alerts"),
            app_commands.Choice(name=SEVERITY_OPTION_LABELS["all"], value="all"),
            app_commands.Choice(name=SEVERITY_OPTION_LABELS["red"], value="red"),
            app_commands.Choice(name=SEVERITY_OPTION_LABELS["orange"], value="orange"),
            app_commands.Choice(name=SEVERITY_OPTION_LABELS["yellow"], value="yellow"),
            app_commands.Choice(name=SEVERITY_OPTION_LABELS["green"], value="green"),
        ]
    )
    async def scan_command(
        interaction: discord.Interaction,
        channel: Optional[discord.TextChannel] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        severity: Optional[app_commands.Choice[str]] = None,
        post_summary: bool = False,
    ) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        target_channel = channel
        channel_ids = [str(channel.id)] if channel else None
        since = since or "1d"
        severity_value = severity.value if severity else "all"
        if severity_value == "all":
            severity_filter = None
        elif severity_value == "alerts":
            severity_filter = ["red", "orange", "yellow"]
        else:
            severity_filter = [severity_value]

        default_end_offset = timedelta()
        resolved_range = resolve_time_range(since, until, default_end_offset=default_end_offset)

        policy, load_result, _ = resolve_policy(Path("configs/rules_v2.yaml"), None)
        if has_version_mismatch(load_result):
            await interaction.followup.send(
                "rules.yaml が version: 2 ではありません。設定を更新してください。",
                ephemeral=True,
            )
            return
        if load_result.status == "error":
            message = "\n".join(f"[{issue.code}] {issue.where}: {issue.msg}" for issue in load_result.issues)
            await interaction.followup.send(
                "ルール読み込みに失敗しました:\n" + message,
                ephemeral=True,
            )
            return

        try:
            summary = await asyncio.to_thread(
                run_scan,
                Path("out/p2_analysis.jsonl"),
                Path("out/p3_findings.jsonl"),
                Path("configs/rules_v2.yaml"),
                channel_ids=channel_ids,
                since=since,
                until=until,
                severity_filter=severity_filter,
                time_range=resolved_range,
                default_end_offset=default_end_offset,
                policy=policy,
            )
        except FileNotFoundError:
            await interaction.followup.send("解析データが見つかりません。Day2 の処理を先に実行してください。", ephemeral=True)
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("scan failed")
            await interaction.followup.send(f"スキャンに失敗しました: {exc}", ephemeral=True)
            return

        embed = build_summary_embed(
            summary,
            target_channel,
            since,
            until,
            severity_value,
            interaction.user,
            time_range=resolved_range,
        )
        await interaction.followup.send(embed=embed, ephemeral=True)

        if post_summary and settings.log_channel_id:
            log_channel = client.get_channel(settings.log_channel_id)
            if log_channel is None:
                try:
                    log_channel = await client.fetch_channel(settings.log_channel_id)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("post summary failed")
                    return
            await log_channel.send(embed=embed)

    @tree.command(name="report", description="モデレーションレポートを生成", guild=guild)
    @app_commands.guild_only()
    @app_commands.describe(
        channel="対象チャンネル（未指定は全チャンネル）",
        since="開始日（未指定は24時間前 / YYYY-MM-DD や 7d, 24h 等に対応）",
        until="終了日（未指定は現在時刻）",
        severity="対象とする深刻度（未指定は警告＝緑以外）",
        format="出力形式",
    )
    @app_commands.choices(
        severity=[
            app_commands.Choice(name=SEVERITY_OPTION_LABELS["alerts"], value="alerts"),
            app_commands.Choice(name=SEVERITY_OPTION_LABELS["all"], value="all"),
            app_commands.Choice(name=SEVERITY_OPTION_LABELS["red"], value="red"),
            app_commands.Choice(name=SEVERITY_OPTION_LABELS["orange"], value="orange"),
            app_commands.Choice(name=SEVERITY_OPTION_LABELS["yellow"], value="yellow"),
            app_commands.Choice(name=SEVERITY_OPTION_LABELS["green"], value="green"),
        ],
        format=[
            app_commands.Choice(name="埋め込み", value="embed"),
            app_commands.Choice(name="CSV", value="csv"),
            app_commands.Choice(name="両方", value="both"),
        ],
    )
    async def report_command(
        interaction: discord.Interaction,
        channel: Optional[discord.TextChannel] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        severity: Optional[app_commands.Choice[str]] = None,
        format: Optional[app_commands.Choice[str]] = None,
    ) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        target_channel = channel
        channel_ids = [str(channel.id)] if channel else None
        since = since or "1d"
        severity_value = severity.value if severity else "alerts"
        if severity_value == "all":
            severity_filter = None
        elif severity_value == "alerts":
            severity_filter = ["red", "orange", "yellow"]
        else:
            severity_filter = [severity_value]
        format_value = format.value if format else "embed"
        default_end_offset = timedelta()
        resolved_range = resolve_time_range(since, until, default_end_offset=default_end_offset)

        try:
            records = await load_findings_async(
                Path("out/p3_findings.jsonl"),
                channel_ids=channel_ids,
                since=since,
                until=until,
                severity=severity_filter,
                time_range=resolved_range,
                default_end_offset=default_end_offset,
            )
        except FileNotFoundError:
            await interaction.followup.send("検知データが見つかりません。先に /scan を実行してください。", ephemeral=True)
            return

        if not records:
            await interaction.followup.send("条件に合致する検知はありませんでした。", ephemeral=True)
            return

        send_embed = format_value in {"embed", "both"}
        send_csv = format_value in {"csv", "both"}

        if send_embed:
            view = ReportPaginator(client, settings, ticket_store, records, interaction.user)
            await view.send_initial(interaction)

        if send_csv:
            engine = RuleEngine()
            violence_patterns = engine.groups.get("violence", ())
            csv_path = Path("out/p3_report.csv")
            rows = write_report_csv(records, csv_path, violence_patterns)
            with csv_path.open("rb") as fp:
                severity_display = _severity_option_label(severity_value)
                await interaction.followup.send(
                    content=f"CSV を作成しました（{rows} 行、対象深刻度={severity_display}）。",
                    file=discord.File(fp=fp, filename=csv_path.name),
                    ephemeral=True,
                )

    @tree.command(name="tickets", description="通知待ちチケットを表示", guild=guild)
    @app_commands.guild_only()
    async def tickets_command(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        active_tickets = await ticket_store.fetch_active_tickets()
        if not active_tickets:
            await interaction.followup.send("通知待ちのチケットはありません。", ephemeral=True)
            return
        record_index = load_findings_index()
        entries: list[TicketEntry] = []
        for ticket in active_tickets:
            record = record_index.get((ticket.channel_id, ticket.message_id))
            if record is None:
                record = find_record_for_message(ticket.channel_id, ticket.message_id)
            record_data = _ensure_record_defaults(record, ticket, None)
            entries.append(TicketEntry(ticket=ticket, record=record_data))
        view = TicketPaginator(client, settings, ticket_store, entries, interaction.user)
        await view.send_initial(interaction)

    watcher_task: asyncio.Task | None = None

    @client.event
    async def setup_hook() -> None:  # type: ignore[override]
        nonlocal watcher_task
        await ticket_store.connect()
        if watcher_task is None:
            watcher_task = client.loop.create_task(ticket_watcher(client, settings, ticket_store))


def main() -> None:
    settings = get_settings()
    client, tree = create_client()
    ticket_store = TicketStore(settings.ticket_db_path)
    register_commands(client, tree, settings, ticket_store)

    logger.info("starting bot")
    client.run(settings.discord_bot_token)


if __name__ == "__main__":
    main()
