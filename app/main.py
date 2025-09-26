from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import discord
from discord import app_commands

from .config import get_settings
from .discord_client import create_client
from .triage import generate_report, run_scan

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

    @tree.command(name="scan", description="Run WD14/NudeNet triage")
    async def scan_command(interaction: discord.Interaction) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)

        try:
            summary = await asyncio.to_thread(run_scan)
        except FileNotFoundError:
            await interaction.followup.send(
                "Analysis data not found. Run the Day2 pipeline first.", ephemeral=True
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("scan failed")
            await interaction.followup.send(f"Scan failed: {exc}", ephemeral=True)
            return
        message = summary.format_message()
        await interaction.followup.send(f"Scan completed: {message}", ephemeral=True)

    @tree.command(name="report", description="Generate moderation report CSV")
    @app_commands.describe(severity="Filter by severity")
    @app_commands.choices(
        severity=[
            app_commands.Choice(name="All", value="all"),
            app_commands.Choice(name="Red", value="red"),
            app_commands.Choice(name="Orange", value="orange"),
            app_commands.Choice(name="Yellow", value="yellow"),
            app_commands.Choice(name="Green", value="green"),
        ]
    )
    async def report_command(
        interaction: discord.Interaction,
        severity: app_commands.Choice[str] | None = None,
    ) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)

        severity_value = severity.value if severity else "all"
        try:
            summary = await asyncio.to_thread(generate_report, severity=severity_value)
        except FileNotFoundError:
            await interaction.followup.send(
                "Findings not found. Run /scan first.", ephemeral=True
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("report failed")
            await interaction.followup.send(f"Report failed: {exc}", ephemeral=True)
            return
        file_path = summary.path
        if not file_path.exists():
            await interaction.followup.send("Report generation failed.", ephemeral=True)
            return

        with file_path.open("rb") as fp:
            discord_file = discord.File(fp=fp, filename=file_path.name)
            await interaction.followup.send(
                content=f"Generated report ({summary.rows} rows, severity={severity_value}).",
                file=discord_file,
                ephemeral=True,
            )


def main() -> None:
    settings = get_settings()
    client, tree = create_client()
    register_commands(client, tree)

    logger.info("starting bot")
    client.run(settings.discord_bot_token)


if __name__ == "__main__":
    main()
