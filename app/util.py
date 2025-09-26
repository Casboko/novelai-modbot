from __future__ import annotations

from io import BytesIO
import re
from typing import Iterable, Tuple

import imagehash
from PIL import Image


def chunked(iterable: Iterable, size: int):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == size:
            yield tuple(chunk)
            chunk.clear()
    if chunk:
        yield tuple(chunk)


def compute_phash(data: bytes, hash_size: int = 8) -> str:
    with Image.open(BytesIO(data)) as image:
        phash = imagehash.phash(image.convert("RGB"), hash_size=hash_size)
    return str(phash)


_MESSAGE_LINK_PATTERN = re.compile(
    r"^https://(?:(?:canary|ptb)\.)?discord\.com/channels/(?P<guild_id>\d+)/(?P<channel_id>\d+)/(?P<message_id>\d+)$"
)


def parse_message_link(message_link: str) -> Tuple[int, int, int]:
    """Parse a Discord message jump link.

    Raises:
        ValueError: If the link format is unsupported or identifiers are missing.
    """

    match = _MESSAGE_LINK_PATTERN.match(message_link.strip())
    if not match:
        raise ValueError("Unsupported message link format")
    try:
        guild_id = int(match.group("guild_id"))
        channel_id = int(match.group("channel_id"))
        message_id = int(match.group("message_id"))
    except (TypeError, ValueError) as exc:  # noqa: BLE001
        raise ValueError("Message link contains invalid identifiers") from exc
    return guild_id, channel_id, message_id
