from __future__ import annotations

from datetime import datetime, timedelta, timezone

import discord

from .config import get_settings
from .rules import RuleDecision

JST = timezone(timedelta(hours=9))


def build_due_timestamp(hours: int) -> datetime:
    return datetime.now(JST) + timedelta(hours=hours)


def build_notification_content(decision: RuleDecision, message_link: str, author: discord.User) -> str:
    settings = get_settings()
    due = build_due_timestamp(settings.due_hours)
    return (
        f"{author.mention} この投稿はDiscordガイドラインに抵触している可能性があります。\n"
        "お手数ですが、内容をご確認のうえ削除をお願いします。\n"
        f"対象: {message_link}\n"
        f"対応期限: {due:%Y-%m-%d %H:%M} JST"
    )
