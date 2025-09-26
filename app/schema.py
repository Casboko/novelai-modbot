from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass(slots=True)
class MessageRef:
    message_link: str
    message_id: str
    channel_id: str
    guild_id: str
    source: str
    url: str
    is_nsfw_channel: Optional[bool] = None


@dataclass(slots=True)
class WD14Record:
    model: str
    revision: str
    input_size: Optional[int]
    rating: Mapping[str, float]
    general: List[tuple[str, float]]
    character: List[tuple[str, float]]
    raw: List[float] = field(default_factory=list)
    error: Optional[str] = None


@dataclass(slots=True)
class NudityDetection:
    cls: str
    score: float
    box: Optional[List[float]]


@dataclass(slots=True)
class AnalysisRecord:
    phash: str
    primary_message: MessageRef
    messages: List[MessageRef]
    wd14: WD14Record
    nudity: List[NudityDetection]
    xsignals: Dict[str, float]
    meta: Dict[str, Any]


def parse_bool(value: str | bool | None) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def format_detection(detection: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "class": detection["cls"],
        "score": detection["score"],
        "box": detection.get("box"),
    }
