from __future__ import annotations

import logging

import discord
from discord import app_commands

from .config import get_settings

logger = logging.getLogger(__name__)


def create_client() -> tuple[discord.Client, app_commands.CommandTree]:
    settings = get_settings()
    intents = discord.Intents.default()
    intents.message_content = True

    client = discord.Client(intents=intents)
    tree = app_commands.CommandTree(client)

    @client.event
    async def on_ready() -> None:
        logger.info("ready %s", client.user)
        await tree.sync()
        guild_id = settings.guild_id
        if guild_id is not None:
            guild = discord.Object(id=guild_id)
            await tree.sync(guild=guild)

    return client, tree
