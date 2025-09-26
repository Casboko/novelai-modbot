from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import discord

from .config import get_settings
from .discord_client import create_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))


def build_jump_link(gid: int, cid: int, mid: int) -> str:
    return f"https://discord.com/channels/{gid}/{cid}/{mid}"


def register_commands(client: discord.Client, tree: discord.app_commands.CommandTree) -> None:
    @tree.command(name="ping", description="health check")
    async def ping(interaction: discord.Interaction) -> None:
        await interaction.response.send_message("pong")

    @tree.command(name="notify", description="post a removal request reply")
    async def notify(
        interaction: discord.Interaction,
        channel_id: str,
        message_id: str,
        due_hours: int | None = None,
    ) -> None:
        settings = get_settings()
        channel = await client.fetch_channel(int(channel_id))
        message = await channel.fetch_message(int(message_id))

        hours = due_hours if due_hours is not None else settings.due_hours
        due = datetime.now(JST) + timedelta(hours=hours)
        allowed = discord.AllowedMentions(everyone=False, roles=False, users=[message.author], replied_user=False)
        content = (
            f"{message.author.mention} This post may violate the server rules."
            f" Please remove it yourself.\n"
            f"Target: {build_jump_link(message.guild.id, message.channel.id, message.id)}\n"
            f"Deadline: {due:%Y-%m-%d %H:%M} JST"
        )
        await message.reply(content=content, allowed_mentions=allowed)
        await interaction.response.send_message("Notification sent.", ephemeral=True)


def main() -> None:
    settings = get_settings()
    client, tree = create_client()
    register_commands(client, tree)

    logger.info("starting bot")
    client.run(settings.discord_bot_token)


if __name__ == "__main__":
    main()
