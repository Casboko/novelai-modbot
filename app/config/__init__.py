"""Configuration utilities and settings helpers for Modbot."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment/.env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Discord bot credentials / behaviour
    discord_bot_token: str
    guild_id: Optional[int] = None
    log_channel_id: Optional[int] = None

    # Notification / ticket handling defaults
    due_hours: int = 72
    ticket_db_path: Path = Field(default=Path("data/modbot.db"), alias="TICKET_DB_PATH")
    ticket_poll_interval: int = Field(default=300, ge=60, alias="TICKET_POLL_INTERVAL")

    # Locale defaults
    timezone: str = Field(default="Asia/Tokyo", alias="TZ")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()


__all__ = ["Settings", "get_settings"]
