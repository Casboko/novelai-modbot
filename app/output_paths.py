"""Output path helpers with profile-aware fallbacks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

from .profiles import (
    DEFAULT_PROFILE,
    PROFILES_ROOT,
    PartitionPaths,
    ProfileContext,
    StageName,
)

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


def _partition_candidate(context: ProfileContext, stage: StageName) -> Path:
    return PartitionPaths(context).stage_file(stage)


def _context_from_env() -> ProfileContext | None:
    if "MODBOT_PROFILE" not in os.environ and "MODBOT_PARTITION_DATE" not in os.environ:
        return None
    try:
        return ProfileContext.from_env()
    except Exception:  # pragma: no cover - fallback when env is invalid
        return None


def partition_paths(context: ProfileContext) -> PartitionPaths:
    """Return ``PartitionPaths`` for the provided context."""

    return PartitionPaths(context)


def default_analysis_path(context: ProfileContext | None = None) -> Path:
    """Preferred location for merged analysis records.

    When ``context`` is ``None`` the legacy path is returned to preserve
    backwards compatibility. New callers should supply a ``ProfileContext``.
    """

    if context is None:
        return P2_ANALYSIS_CANDIDATES[0]
    return _partition_candidate(context, "p2")


def default_findings_path(context: ProfileContext | None = None) -> Path:
    if context is None:
        return P3_FINDINGS_CANDIDATES[0]
    return _partition_candidate(context, "p3")


def default_report_path(context: ProfileContext | None = None) -> Path:
    if context is None:
        return P3_REPORT_CANDIDATES[0]
    return PartitionPaths(context).report_path()


def resolve_analysis_path(context: ProfileContext | None = None) -> Path | None:
    if context is not None:
        candidate = _partition_candidate(context, "p2")
        if candidate.exists():
            return candidate
    else:
        env_context = _context_from_env()
        if env_context is not None:
            candidate = _partition_candidate(env_context, "p2")
            if candidate.exists():
                return candidate
    return _resolve_existing(P2_ANALYSIS_CANDIDATES)


def resolve_findings_path(context: ProfileContext | None = None) -> Path | None:
    if context is not None:
        candidate = _partition_candidate(context, "p3")
        if candidate.exists():
            return candidate
    else:
        env_context = _context_from_env()
        if env_context is not None:
            candidate = _partition_candidate(env_context, "p3")
            if candidate.exists():
                return candidate
    return _resolve_existing(P3_FINDINGS_CANDIDATES)


def resolve_report_path(context: ProfileContext | None = None) -> Path | None:
    if context is not None:
        candidate = PartitionPaths(context).report_path()
        if candidate.exists():
            return candidate
    else:
        env_context = _context_from_env()
        if env_context is not None:
            candidate = PartitionPaths(env_context).report_path()
            if candidate.exists():
                return candidate
    return _resolve_existing(P3_REPORT_CANDIDATES)


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def profile_root(profile: str | None = None, *, ensure: bool = False) -> Path:
    profile_name = profile or DEFAULT_PROFILE
    path = PROFILES_ROOT / profile_name
    if ensure:
        path.mkdir(parents=True, exist_ok=True)
    return path


def find_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


__all__ = [
    "P2_ANALYSIS_CANDIDATES",
    "P3_FINDINGS_CANDIDATES",
    "P3_REPORT_CANDIDATES",
    "default_analysis_path",
    "default_findings_path",
    "default_report_path",
    "ensure_parent",
    "find_existing",
    "partition_paths",
    "profile_root",
    "resolve_analysis_path",
    "resolve_findings_path",
    "resolve_report_path",
]
