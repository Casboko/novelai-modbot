"""Profile-aware output path helpers for partitioned pipelines."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, Literal, TYPE_CHECKING, Optional, Tuple

from zoneinfo import ZoneInfo

if os.name == "nt":  # pragma: no cover - zoneinfo fallback
    try:
        from backports.zoneinfo import ZoneInfo  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - only on legacy python
        pass

if TYPE_CHECKING:  # pragma: no cover - circular import guard
    from .config import Settings


DEFAULT_PROFILE: str = "current"
LEGACY_PROFILE: str = "legacy"
PROFILES_ROOT: Path = Path("out/profiles")


_PROFILES_ROOT_OVERRIDE: Path | None = None


def set_profiles_root_override(path: Path | None) -> None:
    """Override the profiles root (testing and tooling support)."""

    global _PROFILES_ROOT_OVERRIDE
    _PROFILES_ROOT_OVERRIDE = Path(path) if path is not None else None
    clear_context_cache()


def get_profiles_root() -> Path:
    """Return the active profiles root, considering overrides."""

    return _PROFILES_ROOT_OVERRIDE or PROFILES_ROOT

StageName = Literal["p0", "p1", "p2", "p3"]


class ProfileError(RuntimeError):
    """Raised when profile configuration or path resolution fails."""


_STAGE_FILE_TEMPLATES: dict[str, dict[str, str | Path]] = {
    "p0": {
        "dir": Path("p0"),
        "filename": "p0_{date}.csv",
        "glob": "p0_*.csv",
        "prefix": "p0_",
        "suffix": ".csv",
    },
    "p1": {
        "dir": Path("p1"),
        "filename": "p1_{date}.jsonl",
        "glob": "p1_*.jsonl",
        "prefix": "p1_",
        "suffix": ".jsonl",
    },
    "p2": {
        "dir": Path("p2"),
        "filename": "p2_{date}.jsonl",
        "glob": "p2_*.jsonl",
        "prefix": "p2_",
        "suffix": ".jsonl",
    },
    "p3": {
        "dir": Path("p3"),
        "filename": "findings_{date}.jsonl",
        "glob": "findings_*.jsonl",
        "prefix": "findings_",
        "suffix": ".jsonl",
    },
}


def _normalize_profile(value: str | None) -> str:
    if value is None:
        return DEFAULT_PROFILE
    normalized = value.strip()
    return normalized if normalized else DEFAULT_PROFILE


def _sanitize_timezone(name: str | None) -> str | None:
    if name is None:
        return None
    candidate = name.strip()
    if not candidate:
        return None
    if candidate.startswith("<") and candidate.endswith(">"):
        return None
    return candidate


def _timezone_from_name(name: str | None) -> ZoneInfo | UTC:
    normalized = _sanitize_timezone(name)
    if not normalized:
        return UTC
    try:
        return ZoneInfo(normalized)
    except Exception as exc:  # noqa: BLE001
        raise ProfileError(f"invalid timezone: {normalized}") from exc


def _resolve_date(date_token: str | None, tzinfo: ZoneInfo | UTC) -> date:
    now = datetime.now(tzinfo)
    if not date_token or date_token.lower() == "today":
        return now.date()
    if date_token.lower() == "yesterday":
        return (now - timedelta(days=1)).date()
    try:
        parsed = datetime.fromisoformat(date_token)
    except ValueError as exc:  # noqa: B904
        raise ProfileError(f"invalid partition date: {date_token}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tzinfo)
    return parsed.astimezone(tzinfo).date()


@dataclass(slots=True)
class ProfileContext:
    """Active profile information used to resolve output paths."""

    profile: str
    date: date
    tzinfo: ZoneInfo | UTC

    @classmethod
    def from_env(
        cls,
        *,
        date_token: str | None = None,
        default_profile: str | None = None,
        default_timezone: str | None = None,
    ) -> "ProfileContext":
        profile_env = os.getenv("MODBOT_PROFILE")
        tz_env = _sanitize_timezone(os.getenv("MODBOT_TZ"))
        tz_fallback = _sanitize_timezone(default_timezone)
        tz_name = tz_env or tz_fallback
        profile = _normalize_profile(profile_env or default_profile)
        tzinfo = _timezone_from_name(tz_name)
        date_value = _resolve_date(date_token or os.getenv("MODBOT_PARTITION_DATE"), tzinfo)
        return cls(profile=profile, date=date_value, tzinfo=tzinfo)

    @classmethod
    def from_cli(
        cls,
        profile_arg: str | None,
        date_arg: str | None,
        *,
        default_profile: str | None = None,
        timezone_hint: str | None = None,
        default_timezone: str | None = None,
    ) -> "ProfileContext":
        profile = _normalize_profile(profile_arg) if profile_arg else None
        env_profile = os.getenv("MODBOT_PROFILE")
        final_profile = _normalize_profile(profile or env_profile or default_profile)
        tz_env = _sanitize_timezone(os.getenv("MODBOT_TZ"))
        tz_hint = _sanitize_timezone(timezone_hint)
        tz_default = _sanitize_timezone(default_timezone)
        tz_name = tz_env or tz_hint or tz_default
        tzinfo = _timezone_from_name(tz_name)
        date_value = _resolve_date(date_arg, tzinfo)
        return cls(profile=final_profile, date=date_value, tzinfo=tzinfo)

    @classmethod
    def from_settings(
        cls,
        settings: "Settings",
        *,
        profile: str | None = None,
        date_arg: str | None = None,
    ) -> "ProfileContext":
        default_profile = getattr(settings, "profile_default", DEFAULT_PROFILE)
        timezone_hint = getattr(settings, "profile_timezone", None) or settings.timezone
        return cls.from_cli(
            profile_arg=profile,
            date_arg=date_arg,
            default_profile=default_profile,
            timezone_hint=timezone_hint,
            default_timezone=settings.timezone,
        )

    @property
    def iso_date(self) -> str:
        return self.date.isoformat()

    def with_date(self, *, date_token: str | None) -> "ProfileContext":
        new_date = _resolve_date(date_token, self.tzinfo)
        return ProfileContext(profile=self.profile, date=new_date, tzinfo=self.tzinfo)


@dataclass(slots=True)
class PartitionPaths:
    """Resolve partition-specific paths for a profile context."""

    context: ProfileContext
    root: Path = field(default_factory=get_profiles_root)

    def profile_root(self, *, ensure: bool = False) -> Path:
        path = self.root / self.context.profile
        if ensure:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _stage_config(self, stage: StageName) -> dict[str, str | Path]:
        try:
            return _STAGE_FILE_TEMPLATES[stage]
        except KeyError as exc:
            raise ProfileError(f"unknown stage: {stage}") from exc

    def stage_dir(self, stage: StageName, *, ensure: bool = False) -> Path:
        cfg = self._stage_config(stage)
        path = self.profile_root(ensure=ensure) / Path(cfg["dir"])
        if ensure:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def stage_file(self, stage: StageName, *, ensure_parent: bool = False) -> Path:
        cfg = self._stage_config(stage)
        filename = str(cfg["filename"]).format(date=self.context.iso_date)
        path = self.stage_dir(stage, ensure=ensure_parent) / filename
        if ensure_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def p0_state_path(self, *, ensure_parent: bool = False) -> Path:
        path = self.profile_root(ensure=ensure_parent) / "status" / f"p0_state_{self.context.profile}.json"
        if ensure_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def p0_cache_path(self, *, ensure_parent: bool = False) -> Path:
        path = self.profile_root(ensure=ensure_parent) / "cache" / f"p0_phash_{self.context.profile}.csv"
        if ensure_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def metrics_file(self, stage: str, *, ensure_parent: bool = False) -> Path:
        path = (
            self.profile_root(ensure=ensure_parent)
            / "metrics"
            / stage
            / f"{stage}_metrics_{self.context.iso_date}.json"
        )
        if ensure_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def status_manifest(self, stage: str, *, ensure_parent: bool = False) -> Path:
        path = self.profile_root(ensure=ensure_parent) / "status" / f"{stage}_manifest_{self.context.profile}.json"
        if ensure_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def attachments_index(self, *, ensure_parent: bool = False) -> Path:
        path = (
            self.profile_root(ensure=ensure_parent)
            / "attachments"
            / "p0"
            / f"p0_{self.context.iso_date}_index.json"
        )
        if ensure_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def report_path(self, *, ensure_parent: bool = False) -> Path:
        path = self.profile_root(ensure=ensure_parent) / "p3" / f"report_{self.context.iso_date}.csv"
        if ensure_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def report_ext_path(self, *, ensure_parent: bool = False) -> Path:
        path = self.profile_root(ensure=ensure_parent) / "p3" / "report_ext" / f"report_ext_{self.context.iso_date}.csv"
        if ensure_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def attachments_dir(self, *, ensure: bool = False) -> Path:
        path = self.profile_root(ensure=ensure) / "attachments"
        if ensure:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def pipeline_metrics_path(self, date: str | None = None, *, ensure_parent: bool = False) -> Path:
        token_source = date if date else self.context.iso_date
        token = token_source.strip() if isinstance(token_source, str) and token_source.strip() else self.context.iso_date
        path = (
            self.profile_root(ensure=ensure_parent)
            / "metrics"
            / "pipeline"
            / f"pipeline_{token}.jsonl"
        )
        if ensure_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path


@dataclass(slots=True)
class ContextPaths:
    """Cached path helpers derived from a profile context."""

    context: ProfileContext
    partition_paths: PartitionPaths
    _stage_cache: dict[StageName, Path] = field(default_factory=dict)

    def stage_file(self, stage: StageName, *, ensure_parent: bool = False) -> Path:
        if ensure_parent:
            return self.partition_paths.stage_file(stage, ensure_parent=True)
        cached = self._stage_cache.get(stage)
        if cached is not None:
            return cached
        path = self.partition_paths.stage_file(stage, ensure_parent=False)
        self._stage_cache[stage] = path
        return path

    def report_path(self, *, ensure_parent: bool = False) -> Path:
        return self.partition_paths.report_path(ensure_parent=ensure_parent)

    def report_ext_path(self, *, ensure_parent: bool = False) -> Path:
        return self.partition_paths.report_ext_path(ensure_parent=ensure_parent)

    def metrics_path(self, stage: str, *, ensure_parent: bool = False) -> Path:
        return self.partition_paths.metrics_file(stage, ensure_parent=ensure_parent)

    @classmethod
    def for_context(cls, context: ProfileContext) -> ContextPaths:
        return _cached_context_paths(context)

    def pipeline_metrics_path(self, *, ensure_parent: bool = False) -> Path:
        return self.partition_paths.pipeline_metrics_path(ensure_parent=ensure_parent)


_CONTEXT_PATHS_CACHE: dict[Tuple[str, str, str], ContextPaths] = {}


def clear_context_cache() -> None:
    """Clear cached ContextPaths entries (used when overriding roots)."""

    _CONTEXT_PATHS_CACHE.clear()


def _context_cache_key(context: ProfileContext) -> Tuple[str, str, str]:
    tz = getattr(context.tzinfo, "key", str(context.tzinfo))
    return context.profile, context.iso_date, tz


def _cached_context_paths(context: ProfileContext) -> ContextPaths:
    key = _context_cache_key(context)
    cached = _CONTEXT_PATHS_CACHE.get(key)
    if cached is not None:
        return cached
    paths = ContextPaths(context=context, partition_paths=PartitionPaths(context))
    _CONTEXT_PATHS_CACHE[key] = paths
    return paths


@dataclass(slots=True)
class ContextResolveResult:
    """Outcome of resolving a context for command execution."""

    context: ProfileContext
    paths: ContextPaths
    fallback_reason: Optional[str] = None

    @property
    def iso_date(self) -> str:
        return self.context.iso_date


def _extract_partition_date(path: Path, *, stage: StageName) -> date | None:
    cfg = _STAGE_FILE_TEMPLATES[stage]
    prefix = str(cfg["prefix"])
    suffix = str(cfg["suffix"])
    stem = path.name
    if not stem.startswith(prefix) or not stem.endswith(suffix):
        return None
    token = stem[len(prefix) : len(stem) - len(suffix)]
    try:
        return date.fromisoformat(token)
    except ValueError:
        return None


def list_partitions(
    profile: str,
    stage: StageName,
    *,
    limit: int | None = None,
    order: Literal["asc", "desc"] = "desc",
    root: Path | None = None,
) -> list[date]:
    cfg = _STAGE_FILE_TEMPLATES[stage]
    base_root = root or get_profiles_root()
    directory = base_root / profile / Path(cfg["dir"])
    if not directory.exists():
        return []
    results: list[date] = []
    for path in sorted(directory.glob(str(cfg["glob"]))):
        parsed = _extract_partition_date(path, stage=stage)
        if parsed is not None:
            results.append(parsed)
    results.sort(reverse=(order == "desc"))
    if limit is not None:
        return results[:limit]
    return results


def iter_partitions(
    profile: str,
    stage: StageName,
    *,
    order: Literal["asc", "desc"] = "desc",
    root: Path | None = None,
) -> Iterator[date]:
    for value in list_partitions(profile, stage, order=order, root=root):
        yield value


def latest_partition_date(
    profile: str,
    stage: StageName,
    *,
    root: Path | None = None,
) -> Optional[date]:
    items = list_partitions(profile, stage, limit=1, root=root)
    return items[0] if items else None


def resolve_latest_partition(
    context: ProfileContext,
    stage: StageName,
    *,
    fallback_to_legacy: bool = True,
    legacy_candidates: Iterable[Path] | None = None,
) -> tuple[Path, Optional[str]]:
    paths = ContextPaths.for_context(context)
    stage_path = paths.stage_file(stage, ensure_parent=False)
    fallback_reason: Optional[str] = None
    if stage_path.exists():
        return stage_path, fallback_reason
    legacy_paths = list(legacy_candidates or ()) if fallback_to_legacy else []
    for candidate in legacy_paths:
        if candidate.exists():
            fallback_reason = "fallback=legacy"
            return candidate, fallback_reason
    fallback_reason = fallback_reason or "fallback=missing"
    return stage_path, fallback_reason


def iter_recent_findings(
    context: ProfileContext,
    *,
    days_back: int = 7,
    stage: StageName = "p3",
    root: Path | None = None,
) -> Iterator[tuple[date, Path]]:
    target_root = root or get_profiles_root()
    for partition_date in list_partitions(
        context.profile,
        stage,
        limit=days_back,
        root=target_root,
    ):
        date_token = partition_date.isoformat()
        partition_context = context.with_date(date_token=date_token)
        yield partition_date, PartitionPaths(partition_context, root=target_root).stage_file(stage)


__all__ = [
    "DEFAULT_PROFILE",
    "LEGACY_PROFILE",
    "PROFILES_ROOT",
    "clear_context_cache",
    "ContextPaths",
    "ContextResolveResult",
    "get_profiles_root",
    "PartitionPaths",
    "ProfileContext",
    "ProfileError",
    "StageName",
    "iter_recent_findings",
    "iter_partitions",
    "latest_partition_date",
    "list_partitions",
    "resolve_latest_partition",
    "set_profiles_root_override",
]
