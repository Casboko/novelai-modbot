from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Sequence

import discord
from discord import app_commands

from .config import get_settings
from .discord_client import create_client
from .rule_engine import RuleEngine
from .triage import load_findings, resolve_time_range, run_scan, write_report_csv
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


def build_jump_link(gid: int, cid: int, mid: int) -> str:
    return f"https://discord.com/channels/{gid}/{cid}/{mid}"


class PreviewInfo(NamedTuple):
    thumbnail_url: Optional[str]
    image_url: Optional[str]
    original_url: Optional[str]
    is_spoiler: bool = False


def _severity_badge(severity: str) -> str:
    return SEVERITY_BADGES.get(severity.lower(), f"[{severity.upper()}]")


def _severity_field_label(severity: str) -> str:
    return SEVERITY_FIELD_NAMES.get(severity.lower(), severity.upper())


def _severity_option_label(value: str) -> str:
    return SEVERITY_OPTION_LABELS.get(value.lower(), value)


def _clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def _bar(value: float, width: int = 8) -> str:
    clamped = _clamp(value)
    filled = int(round(clamped * width))
    filled = min(filled, width)
    return "█" * filled + "▁" * (width - filled)


def _bar_nonnegative(value: float, width: int = 8) -> str:
    if value <= 0:
        return "▁" * width
    clamped = _clamp(value)
    filled = int(round(clamped * width))
    filled = min(max(filled, 1), width)
    return "█" * filled + "▁" * (width - filled)

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
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _record_matches_message(record, channel_id, message_id):
                return record
    return None


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

    try:
        await message.delete(reason=build_audit_reason(ticket, "auto_delete"))
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
    margin = metrics.get("nsfw_margin", 0.0)
    ratio = metrics.get("nsfw_ratio", 0.0)
    nsfw_sum = metrics.get("nsfw_general_sum", 0.0)
    exposure = metrics.get("exposure_peak", record.get("xsignals", {}).get("exposure_score", 0.0))
    violence = metrics.get("violence_max", 0.0)
    minors = metrics.get("minors_sum", 0.0)
    animals = metrics.get("animals_sum", 0.0)
    adult_index = _clamp(
        ratio * 0.5 + max(margin, 0.0) * 0.3 + _clamp(nsfw_sum, 0.0, 1.0) * 0.2
    )
    metrics_block = (
        "```\n"
        f"🔞 erotic   {adult_index:6.2f} { _bar(adult_index) }\n"
        f"🍑 nudity   {exposure:6.2f} { _bar(exposure) }\n"
        f"🩸 gore     {violence:6.2f} { _bar(violence) }\n"
        f"👶 children {minors:6.2f} { _bar(minors) }\n"
        f"🦊 animals  {animals:6.2f} { _bar(animals) }\n"
        "```"
    )
    embed.add_field(name="指標", value=metrics_block, inline=False)

    detections = _format_top_detections(record.get("nudity_detections", []), limit=3)
    if detections:
        embed.add_field(name="🩱 NudeNet 検出", value=", ".join(detections), inline=False)

    if preview.image_url:
        embed.set_image(url=preview.image_url)

    embed.set_footer(text=f"アイテム {index + 1}/{total} ・ phash={record.get('phash', '')}")
    return embed


class ReportPaginator(discord.ui.View):
    def __init__(
        self,
        client: discord.Client,
        settings,
        ticket_store: TicketStore,
        records: Sequence[dict],
        requester: discord.User,
    ) -> None:
        super().__init__(timeout=600)
        self.client = client
        self.settings = settings
        self.ticket_store = ticket_store
        self.records = list(records)
        self.requester = requester
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
            await self.message.edit(view=self)

    def _current_record(self) -> dict:
        return self.records[self.index]

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

    def _build_embed(self, preview: PreviewInfo, author: Optional[discord.abc.User]) -> discord.Embed:
        return build_record_embed(
            self._current_record(),
            self.index,
            len(self.records),
            preview,
            author,
        )

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
        preferred_guild_id = self.settings.guild_id or record.get("guild_id") or (record.get("messages") or [{}])[0].get("guild_id")
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

    async def _ensure_preview(self) -> PreviewInfo:
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

        preview_data = record.get("preview", {})
        url = preview_data.get("source_url") if isinstance(preview_data, dict) else None
        attachment = None
        if url:
            attachment = {
                "url": url,
                "content_type": "image/unknown",
                "is_spoiler": bool(preview_data.get("is_spoiler")) if isinstance(preview_data, dict) else False,
            }
        if attachment is None:
            attachment = first_image_from_messages(record.get("messages", []))
        if attachment is None:
            targets = self._message_targets()
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

    def _update_button_state(self) -> None:
        target_available = self._message_targets() is not None
        for child in self.children:
            if isinstance(child, discord.ui.Button) and child.custom_id == "prev":
                child.disabled = self.index == 0
            if isinstance(child, discord.ui.Button) and child.custom_id == "next":
                child.disabled = self.index >= len(self.records) - 1
            if isinstance(child, discord.ui.Button) and child.custom_id == "notify":
                child.disabled = not target_available
            if isinstance(child, discord.ui.Button) and child.custom_id == "log":
                child.disabled = not bool(self.settings.log_channel_id)
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

    async def send_initial(self, interaction: discord.Interaction) -> None:
        preview = await self._ensure_preview()
        self._current_preview = preview
        author = await self._resolve_member(self._current_record())
        embed = self._build_embed(preview, author)
        self._update_button_state()
        message = await interaction.followup.send(embed=embed, view=self, ephemeral=True)
        self.message = message

    async def _refresh(self, interaction: discord.Interaction) -> None:
        preview = await self._ensure_preview()
        self._current_preview = preview
        author = await self._resolve_member(self._current_record())
        embed = self._build_embed(preview, author)
        self._update_button_state()
        if self.message and self.message.attachments:
            await interaction.response.edit_message(embed=embed, view=self, attachments=self.message.attachments)
        else:
            await interaction.response.edit_message(embed=embed, view=self)

    def _message_targets(self) -> Optional[tuple[int, int]]:
        messages = self._current_record().get("messages") or []
        if not messages:
            return None
        entry = messages[0]
        try:
            return int(entry["channel_id"]), int(entry["message_id"])
        except (KeyError, ValueError, TypeError):
            return None

    @discord.ui.button(label="◀", style=discord.ButtonStyle.secondary, custom_id="prev", row=0)
    async def on_prev(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if self.index > 0:
            self.index -= 1
        await self._refresh(interaction)

    @discord.ui.button(label="通知する", style=discord.ButtonStyle.primary, custom_id="notify", row=2)
    async def on_notify(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.defer(ephemeral=True)
        target = self._message_targets()
        if target is None:
            await interaction.followup.send("対象メッセージの情報が見つかりませんでした。", ephemeral=True)
            return
        guild_id = interaction.guild_id or self.settings.guild_id
        if guild_id is None:
            await interaction.followup.send("サーバー情報が取得できませんでした。", ephemeral=True)
            return
        try:
            _ticket, _, message = await process_notification(
                self.client,
                self.settings,
                self.ticket_store,
                guild_id=int(guild_id),
                channel_id=target[0],
                message_id=target[1],
                executor=interaction.user,
                record=self._current_record(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("notify button failed")
            await interaction.followup.send(f"通知に失敗しました: {exc}", ephemeral=True)
            return
        await interaction.followup.send(message, ephemeral=True)

    @discord.ui.button(label="モデログ転送", style=discord.ButtonStyle.secondary, custom_id="log", row=2)
    async def on_log(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
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
        preview = self._current_preview or await self._ensure_preview()
        author = await self._resolve_member(self._current_record())
        embed = self._build_embed(preview, author)
        embed.set_footer(text=f"転送者: {interaction.user.display_name}")
        await channel.send(embed=embed)
        await interaction.followup.send("モデログへ転送しました。", ephemeral=True)

    @discord.ui.button(label="▶", style=discord.ButtonStyle.secondary, custom_id="next", row=0)
    async def on_next(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if self.index < len(self.records) - 1:
            self.index += 1
        await self._refresh(interaction)


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

        try:
            summary = await asyncio.to_thread(
                run_scan,
                Path("out/p2_analysis.jsonl"),
                Path("out/p3_findings.jsonl"),
                Path("configs/rules.yaml"),
                channel_ids=channel_ids,
                since=since,
                until=until,
                severity_filter=severity_filter,
                time_range=resolved_range,
                default_end_offset=default_end_offset,
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
            records = load_findings(
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
            violence_tags = set(RuleEngine().config.violence_tags)
            csv_path = Path("out/p3_report.csv")
            rows = write_report_csv(records, csv_path, violence_tags)
            with csv_path.open("rb") as fp:
                severity_display = _severity_option_label(severity_value)
                await interaction.followup.send(
                    content=f"CSV を作成しました（{rows} 行、対象深刻度={severity_display}）。",
                    file=discord.File(fp=fp, filename=csv_path.name),
                    ephemeral=True,
                )

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
