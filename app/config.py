from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    discord_bot_token: str
    guild_id: Optional[int] = None
    log_channel_id: Optional[int] = None
    due_hours: int = 72
    ticket_db_path: Path = Field(default=Path("data/modbot.db"), alias="TICKET_DB_PATH")
    ticket_poll_interval: int = Field(default=300, ge=60, alias="TICKET_POLL_INTERVAL")
    timezone: str = Field(default="Asia/Tokyo", alias="TZ")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
