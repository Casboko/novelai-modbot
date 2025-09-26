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
        f"{author.mention} This post may violate the server rules."
        f" Please review: {message_link}\n"
        f"Deadline: {due:%Y-%m-%d %H:%M} JST"
    )
