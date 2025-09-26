from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import discord


@dataclass
class AnalysisResult:
    tags: Dict[str, float]
    nsfw_score: float
    reason: str


class Analyzer:
    """Run WD14 and NudeNet inference on attachments."""

    async def analyze_attachment(self, attachment: discord.Attachment) -> AnalysisResult:
        raise NotImplementedError("Analyzer.analyze_attachment must be implemented")
