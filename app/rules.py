from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .analyzer import AnalysisResult


class Verdict(str, Enum):
    RED = "red"
    ORANGE = "orange"
    YELLOW = "yellow"
    GREEN = "green"


@dataclass
class RuleDecision:
    verdict: Verdict
    rationale: str


class RuleEngine:
    """Evaluate analyzer results against moderation policy."""

    def classify(self, result: AnalysisResult) -> RuleDecision:
        raise NotImplementedError("RuleEngine.classify must be implemented")
