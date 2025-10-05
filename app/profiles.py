"""Profile-aware output path helpers for partitioned pipelines."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, Literal, TYPE_CHECKING

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


def _timezone_from_name(name: str | None) -> ZoneInfo | UTC:
    if not name:
        return UTC
    try:
        return ZoneInfo(name)
    except Exception as exc:  # noqa: BLE001
        raise ProfileError(f"invalid timezone: {name}") from exc


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
        tz_env = os.getenv("MODBOT_TZ") or default_timezone
        profile = _normalize_profile(profile_env or default_profile)
        tzinfo = _timezone_from_name(tz_env)
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
    ) -> "ProfileContext":
        profile = _normalize_profile(profile_arg) if profile_arg else None
        env_profile = os.getenv("MODBOT_PROFILE")
        final_profile = _normalize_profile(profile or env_profile or default_profile)
        tz_name = os.getenv("MODBOT_TZ") or timezone_hint
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
    root: Path = PROFILES_ROOT

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
    root: Path = PROFILES_ROOT,
) -> list[date]:
    cfg = _STAGE_FILE_TEMPLATES[stage]
    directory = root / profile / Path(cfg["dir"])
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
    root: Path = PROFILES_ROOT,
) -> Iterator[date]:
    for value in list_partitions(profile, stage, order=order, root=root):
        yield value


__all__ = [
    "DEFAULT_PROFILE",
    "LEGACY_PROFILE",
    "PROFILES_ROOT",
    "PartitionPaths",
    "ProfileContext",
    "ProfileError",
    "StageName",
    "iter_partitions",
    "list_partitions",
]
