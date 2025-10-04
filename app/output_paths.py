from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


P2_ANALYSIS_CANDIDATES: tuple[Path, ...] = (
    Path("out/p2/p2_analysis_all.jsonl"),
    Path("out/p2_analysis_all.jsonl"),
    Path("out/p2/p2_analysis.jsonl"),
    Path("out/p2_analysis.jsonl"),
)

P3_FINDINGS_CANDIDATES: tuple[Path, ...] = (
    Path("out/p3/p3_decision_all.jsonl"),
    Path("out/p3/p3_findings_all.jsonl"),
    Path("out/p3/p3_decision.jsonl"),
    Path("out/p3/p3_findings.jsonl"),
    Path("out/p3_findings.jsonl"),
)

P3_REPORT_CANDIDATES: tuple[Path, ...] = (
    Path("out/p3/p3_report.csv"),
    Path("out/p3_report.csv"),
)


def _resolve_existing(candidates: Sequence[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def default_analysis_path() -> Path:
    """Preferred location for merged analysis records."""

    return P2_ANALYSIS_CANDIDATES[0]


def default_findings_path() -> Path:
    """Preferred location for DSL evaluation output."""

    return P3_FINDINGS_CANDIDATES[0]


def default_report_path() -> Path:
    """Preferred location for CSV reports derived from findings."""

    return P3_REPORT_CANDIDATES[0]


def resolve_analysis_path() -> Path | None:
    """Return the first existing analysis file, if any."""

    return _resolve_existing(P2_ANALYSIS_CANDIDATES)


def resolve_findings_path() -> Path | None:
    """Return the first existing findings file, if any."""

    return _resolve_existing(P3_FINDINGS_CANDIDATES)


def resolve_report_path() -> Path | None:
    """Return the first existing findings CSV, if any."""

    return _resolve_existing(P3_REPORT_CANDIDATES)


def ensure_parent(path: Path) -> Path:
    """Create the parent directory of ``path`` and return the path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    return path

