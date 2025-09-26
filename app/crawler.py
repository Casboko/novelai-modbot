from __future__ import annotations

from typing import AsyncIterator

import discord


class MessageCrawler:
    """Traverse message history and yield attachments for analysis."""

    def __init__(self, client: discord.Client) -> None:
        self._client = client

    async def iter_attachments(self, channel: discord.abc.GuildChannel) -> AsyncIterator[discord.Attachment]:
        raise NotImplementedError("MessageCrawler.iter_attachments must be implemented")
