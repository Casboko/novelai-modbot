from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Sequence

import discord
from discord import app_commands

from .config import get_settings
from .discord_client import create_client
from .rule_engine import RuleEngine
from .triage import load_findings, resolve_time_range, run_scan, write_report_csv

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


def build_jump_link(gid: int, cid: int, mid: int) -> str:
    return f"https://discord.com/channels/{gid}/{cid}/{mid}"


class PreviewInfo(NamedTuple):
    thumbnail_url: Optional[str]
    image_url: Optional[str]
    original_url: Optional[str]
    is_spoiler: bool = False


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
) -> str:
    message = await _fetch_message(client, channel_id, message_id)
    hours = due_hours if due_hours is not None else settings.due_hours
    due = datetime.now(JST) + timedelta(hours=hours)
    content = (
        f"{message.author.mention} This post may violate the server rules."
        f" Please remove it yourself.\n"
        f"Target: {build_jump_link(message.guild.id, message.channel.id, message.id)}\n"
        f"Deadline: {due:%Y-%m-%d %H:%M} JST"
    )
    await message.reply(content=content, allowed_mentions=_build_allowed_mentions(message.author))
    return content


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
        title="Scan Summary",
        color=severity_color(dominant),
        timestamp=datetime.now(timezone.utc),
    )
    target_name = channel.mention if isinstance(channel, discord.abc.GuildChannel) else "(all channels)"
    embed.add_field(name="Channel", value=target_name, inline=False)
    embed.add_field(
        name="Period",
        value=f"{start.isoformat()} → {end.isoformat()}",
        inline=False,
    )
    embed.add_field(name="Severity Filter", value=severity or "all", inline=False)
    embed.add_field(name="Total", value=str(summary.total), inline=True)
    for sev in SEVERITY_ORDER:
        embed.add_field(name=sev.upper(), value=str(summary.severity_counts.get(sev, 0)), inline=True)
    embed.set_footer(text=f"Requested by {requester.display_name}")
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
    base_title = record.get("rule_title") or record.get("rule_id") or "Moderation Finding"
    title = f"[{severity.upper()}] {base_title}"
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
            author_name = f"User {author_id}"
    posted_value = _format_timestamp(record.get("created_at"))
    if posted_value:
        embed.add_field(name="Posted", value=posted_value, inline=False)
    if author_name and author_id:
        embed.add_field(name="User", value=f"<@{author_id}>", inline=False)
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
    embed.add_field(name="Metrics", value=metrics_block, inline=False)

    detections = _format_top_detections(record.get("nudity_detections", []), limit=3)
    if detections:
        embed.add_field(name="🩱 NudeNet", value=", ".join(detections), inline=False)

    if preview.image_url:
        embed.set_image(url=preview.image_url)

    embed.set_footer(text=f"Item {index + 1}/{total} • phash={record.get('phash', '')}")
    return embed


class ReportPaginator(discord.ui.View):
    def __init__(
        self,
        client: discord.Client,
        settings,
        records: Sequence[dict],
        requester: discord.User,
    ) -> None:
        super().__init__(timeout=600)
        self.client = client
        self.settings = settings
        self.records = list(records)
        self.requester = requester
        self.index = 0
        self.message: Optional[discord.Message] = None
        self._preview_cache: dict[tuple[int, int], PreviewInfo] = {}
        self._member_cache: dict[int, discord.Member] = {}
        self._current_preview: PreviewInfo | None = None
        self._open_message_button = discord.ui.Button(
            label="Open Message",
            style=discord.ButtonStyle.link,
            url="https://discord.com",
            row=1,
            disabled=True,
        )
        self._open_original_button = discord.ui.Button(
            label="Open Original",
            style=discord.ButtonStyle.link,
            url="https://discord.com",
            disabled=True,
            row=1,
        )
        self.add_item(self._open_message_button)
        self.add_item(self._open_original_button)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.requester.id:
            await interaction.response.send_message("You cannot control this view.", ephemeral=True)
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

    def _record_key(self, record: dict) -> Optional[tuple[int, int]]:
        channel_id = record.get("channel_id") or (record.get("messages") or [{}])[0].get("channel_id")
        message_id = record.get("message_id") or (record.get("messages") or [{}])[0].get("message_id")
        if channel_id and message_id:
            try:
                return int(channel_id), int(message_id)
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
        def first_image_from_messages(messages: Iterable[dict]) -> Optional[dict]:
            for msg in messages or []:
                for attachment in msg.get("attachments", []) or []:
                    content_type = attachment.get("content_type") or ""
                    if content_type.startswith("image/"):
                        return attachment
                url = msg.get("url")
                if url:
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
                        if att.content_type and att.content_type.startswith("image/"):
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

    @discord.ui.button(label="Notify", style=discord.ButtonStyle.primary, custom_id="notify", row=2)
    async def on_notify(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.defer(ephemeral=True)
        target = self._message_targets()
        if target is None:
            await interaction.followup.send("Message reference is missing.", ephemeral=True)
            return
        try:
            await send_notify_message(self.client, self.settings, target[0], target[1])
        except Exception as exc:  # noqa: BLE001
            logger.exception("notify button failed")
            await interaction.followup.send(f"Notify failed: {exc}", ephemeral=True)
            return
        await interaction.followup.send("Notification sent.", ephemeral=True)

    @discord.ui.button(label="Log", style=discord.ButtonStyle.secondary, custom_id="log", row=2)
    async def on_log(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        await interaction.response.defer(ephemeral=True)
        channel_id = self.settings.log_channel_id
        if not channel_id:
            await interaction.followup.send("Log channel is not configured.", ephemeral=True)
            return
        channel = self.client.get_channel(int(channel_id))
        if channel is None:
            try:
                channel = await self.client.fetch_channel(int(channel_id))
            except Exception as exc:  # noqa: BLE001
                logger.exception("fetch log channel failed")
                await interaction.followup.send(f"Cannot access log channel: {exc}", ephemeral=True)
                return
        preview = self._current_preview or await self._ensure_preview()
        author = await self._resolve_member(self._current_record())
        embed = self._build_embed(preview, author)
        embed.set_footer(text=f"Logged by {interaction.user.display_name}")
        await channel.send(embed=embed)
        await interaction.followup.send("Logged to moderation channel.", ephemeral=True)

    @discord.ui.button(label="▶", style=discord.ButtonStyle.secondary, custom_id="next", row=0)
    async def on_next(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        if self.index < len(self.records) - 1:
            self.index += 1
        await self._refresh(interaction)


def register_commands(client: discord.Client, tree: discord.app_commands.CommandTree, settings) -> None:
    guild = discord.Object(id=settings.guild_id) if settings.guild_id else None

    @tree.command(name="ping", description="health check", guild=guild)
    async def ping(interaction: discord.Interaction) -> None:
        await interaction.response.send_message("pong")

    @tree.command(name="notify", description="post a removal request reply", guild=guild)
    @app_commands.guild_only()
    async def notify(
        interaction: discord.Interaction,
        channel_id: str,
        message_id: str,
        due_hours: int | None = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            await send_notify_message(
                client,
                settings,
                int(channel_id),
                int(message_id),
                due_hours,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("notify failed")
            await interaction.followup.send(f"Notify failed: {exc}", ephemeral=True)
            return
        await interaction.followup.send("Notification sent.", ephemeral=True)

    @tree.command(name="scan", description="Run WD14/NudeNet triage", guild=guild)
    @app_commands.guild_only()
    @app_commands.describe(
        channel="対象チャンネル（未指定は現在のチャンネル）",
        since="開始日（YYYY-MM-DD または 7d, 24h など）",
        until="終了日（未指定は翌日まで含む）",
        severity="対象とする深刻度",
        post_summary="モデログへサマリを投稿",
    )
    @app_commands.choices(
        severity=[
            app_commands.Choice(name="Alerts (non-green)", value="alerts"),
            app_commands.Choice(name="All", value="all"),
            app_commands.Choice(name="Red", value="red"),
            app_commands.Choice(name="Orange", value="orange"),
            app_commands.Choice(name="Yellow", value="yellow"),
            app_commands.Choice(name="Green", value="green"),
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
        target_channel = channel or (interaction.channel if isinstance(interaction.channel, discord.abc.GuildChannel) else None)
        channel_ids = [str(target_channel.id)] if target_channel else None
        severity_value = severity.value if severity else "alerts"
        if severity_value == "all":
            severity_filter = None
        elif severity_value == "alerts":
            severity_filter = ["red", "orange", "yellow"]
        else:
            severity_filter = [severity_value]

        default_end_offset = timedelta(days=1)
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
            await interaction.followup.send("Analysis data not found. Run Day2 first.", ephemeral=True)
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("scan failed")
            await interaction.followup.send(f"Scan failed: {exc}", ephemeral=True)
            return

        severity_display = "non-green" if severity_value == "alerts" else severity_value
        embed = build_summary_embed(
            summary,
            target_channel,
            since,
            until,
            severity_display,
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

    @tree.command(name="report", description="Generate moderation report", guild=guild)
    @app_commands.guild_only()
    @app_commands.describe(
        channel="対象チャンネル（未指定で現在のチャンネル）",
        since="開始日（YYYY-MM-DD または 7d, 24h など）",
        until="終了日（未指定は翌日まで含む）",
        severity="対象とする深刻度",
        format="出力形式",
    )
    @app_commands.choices(
        severity=[
            app_commands.Choice(name="Alerts (non-green)", value="alerts"),
            app_commands.Choice(name="All", value="all"),
            app_commands.Choice(name="Red", value="red"),
            app_commands.Choice(name="Orange", value="orange"),
            app_commands.Choice(name="Yellow", value="yellow"),
            app_commands.Choice(name="Green", value="green"),
        ],
        format=[
            app_commands.Choice(name="Embed", value="embed"),
            app_commands.Choice(name="CSV", value="csv"),
            app_commands.Choice(name="Both", value="both"),
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
        target_channel = channel or (interaction.channel if isinstance(interaction.channel, discord.abc.GuildChannel) else None)
        channel_ids = [str(target_channel.id)] if target_channel else None
        severity_value = severity.value if severity else "alerts"
        if severity_value == "all":
            severity_filter = None
        elif severity_value == "alerts":
            severity_filter = ["red", "orange", "yellow"]
        else:
            severity_filter = [severity_value]
        format_value = format.value if format else "embed"
        default_end_offset = timedelta(days=1)
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
            await interaction.followup.send("Findings not found. Run /scan first.", ephemeral=True)
            return

        if not records:
            await interaction.followup.send("No findings for the given parameters.", ephemeral=True)
            return

        send_embed = format_value in {"embed", "both"}
        send_csv = format_value in {"csv", "both"}

        if send_embed:
            view = ReportPaginator(client, settings, records, interaction.user)
            await view.send_initial(interaction)

        if send_csv:
            violence_tags = set(RuleEngine().config.violence_tags)
            csv_path = Path("out/p3_report.csv")
            rows = write_report_csv(records, csv_path, violence_tags)
            with csv_path.open("rb") as fp:
                severity_display = "non-green" if severity_value == "alerts" else severity_value
                await interaction.followup.send(
                    content=f"CSV generated ({rows} rows, severity={severity_display}).",
                    file=discord.File(fp=fp, filename=csv_path.name),
                    ephemeral=True,
                )


def main() -> None:
    settings = get_settings()
    client, tree = create_client()
    register_commands(client, tree, settings)

    logger.info("starting bot")
    client.run(settings.discord_bot_token)


if __name__ == "__main__":
    main()
