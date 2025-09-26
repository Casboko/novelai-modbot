from __future__ import annotations

from functools import lru_cache
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
    timezone: str = Field(default="Asia/Tokyo", alias="TZ")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
