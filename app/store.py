from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .rules import RuleDecision


@dataclass
class Record:
    message_link: str
    author_id: int
    decision: RuleDecision


class Store:
    """Persist scan results to CSV or SQLite."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, records: Iterable[Record]) -> None:
        raise NotImplementedError("Store.save must be implemented")
