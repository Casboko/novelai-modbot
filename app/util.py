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


EXPOSED_TOKEN = "EXPOSED"
COVERED_TOKEN = "COVERED"

EXPOSURE_AXIS_LABEL_JP = {
    "BREAST": "胸",
    "FG": "陰♀",
    "MG": "陰♂",
    "AN": "肛",
}


def summarize_exposure_axes(detections: list[dict]) -> dict[str, float]:
    """Return per-axis NudeNet exposure peaks for UI rendering.

    Only detections whose class names contain ``EXPOSED`` and do not contain
    ``COVERED`` are considered. Various label patterns (e.g. FEMALE_/MALE_,
    *_EXPOSED, EXPOSED_*_F/M) are normalised onto four axes: BREAST, FG, MG, AN.
    """

    peaks = {"BREAST": 0.0, "FG": 0.0, "MG": 0.0, "AN": 0.0}
    for det in detections or []:
        label = str(det.get("class", "")).upper()
        if not label or EXPOSED_TOKEN not in label or COVERED_TOKEN in label:
            continue
        try:
            score = float(det.get("score", 0.0))
        except (TypeError, ValueError):
            continue

        axis: str | None = None
        if "BREAST" in label:
            axis = "BREAST"
        elif "GENITALIA" in label or "GENITAL" in label or "GENITALS" in label:
            tokens = {token for token in re.split(r"[^A-Z]", label) if token}
            if "FEMALE" in tokens or "F" in tokens:
                axis = "FG"
            elif "MALE" in tokens or "M" in tokens:
                axis = "MG"
        elif "ANUS" in label:
            axis = "AN"

        if axis and score > peaks[axis]:
            peaks[axis] = score
    return peaks


def ellipsize_inline(text: str, max_len: int = 1024) -> str:
    """Truncate inline text with an ellipsis if it exceeds ``max_len``."""

    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 1)] + "\u2026"
