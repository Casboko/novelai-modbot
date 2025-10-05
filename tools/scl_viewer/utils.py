from __future__ import annotations

import csv
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from collections.abc import Iterable, Mapping, Sequence

import pandas as pd
import yaml
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.io.stream import iter_jsonl
from app.profiles import (
    ContextPaths,
    ContextResolveResult,
    PartitionPaths,
    PROFILES_ROOT,
    list_partitions,
    DEFAULT_PROFILE,
)
from app.triage import P3_CSV_HEADER

P0_HEADER = [
    "guild_id",
    "channel_id",
    "message_id",
    "message_link",
    "created_at",
    "author_id",
    "is_nsfw_channel",
    "source",
    "attachment_id",
    "filename",
    "content_type",
    "file_size",
    "url",
    "phash_hex",
    "phash_distance_ref",
    "note",
]

THRESHOLD_PREFIX = "T_"
FEATURE_REF_PATTERN = re.compile(r"features\.([A-Z0-9_]+)")
REQUIRED_FINDING_KEYS = {"severity", "rule_id", "rule_title", "reasons", "metrics"}


class UtilsError(RuntimeError):
    """Base error for viewer utilities."""


class P0MergeError(UtilsError):
    """Raised when merging p0 CSV shards fails."""


class ThresholdsError(UtilsError):
    """Raised when threshold compilation fails."""


class CliCommandError(UtilsError):
    """Raised when a CLI command returns a non-zero exit code."""

    def __init__(self, command: Sequence[str], result: subprocess.CompletedProcess[str]):
        self.command = tuple(command)
        self.returncode = result.returncode
        self.stdout = result.stdout
        self.stderr = result.stderr
        super().__init__(
            "Command failed: %s (exit code %s)" % (" ".join(self.command), self.returncode)
        )


@dataclass(slots=True)
class ThresholdCompileResult:
    rules_path: Path
    derived_symbols: dict[str, Any]


@dataclass(slots=True)
class ThresholdScanResult:
    referenced: set[str]
    defined: set[str]


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for record in iter_jsonl(path):
        if not REQUIRED_FINDING_KEYS.issubset(record):
            missing = REQUIRED_FINDING_KEYS.difference(record)
            raise UtilsError(
                f"p3 findings record missing required keys {sorted(missing)} in {path}"
            )
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            raise UtilsError(f"record metrics must be an object: {path}")
        if not isinstance(record.get("reasons"), list):
            raise UtilsError(f"record reasons must be a list: {path}")
        records.append(record)
    return records


def read_report_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise UtilsError(f"report CSV not found: {path}") from exc
    except Exception as exc:  # noqa: BLE001
        raise UtilsError(f"failed to read report CSV: {path}: {exc}") from exc
    header = list(df.columns)
    if header != list(P3_CSV_HEADER):
        raise UtilsError(
            "report CSV header mismatch. Expected %s, got %s"
            % (",".join(P3_CSV_HEADER), ",".join(header))
        )
    return df


def available_profiles(default: str | None = None) -> list[str]:
    if PROFILES_ROOT.exists():
        entries = sorted(p.name for p in PROFILES_ROOT.iterdir() if p.is_dir())
    else:
        entries = []
    if default and default not in entries:
        entries.insert(0, default)
    if not entries:
        entries = [DEFAULT_PROFILE]
    return entries


def available_partition_dates(profile: str, *, stage: str = "p3", limit: int = 30) -> list[str]:
    dates = [d.isoformat() for d in list_partitions(profile, stage, limit=limit)]
    return dates


def build_context(
    *,
    profile: str | None = None,
    date: str | None = None,
) -> ContextResolveResult:
    settings = get_settings()
    context = settings.build_profile_context(profile=profile, date=date)
    return ContextResolveResult(
        context=context,
        paths=ContextPaths.for_context(context),
    )


def format_context_notice(context_result: ContextResolveResult) -> str:
    parts = [
        f"profile={context_result.context.profile}",
        f"date={context_result.context.iso_date}",
    ]
    if context_result.fallback_reason:
        parts.append(f"fallback={context_result.fallback_reason}")
    return ", ".join(parts)


def merge_p0_sources(dest: Path, *, partitions: PartitionPaths | None = None) -> Path | None:
    dest = dest.resolve()
    single_candidates: list[Path] = []
    shard_candidates: list[Path] = []
    if partitions is not None:
        single_candidates.append(partitions.stage_file("p0"))
        shard_candidates.append(partitions.stage_dir("p0") / "shards")
    single_candidates.append(Path("out/p0_scan.csv"))
    shard_candidates.append(Path("out/p0"))

    single: Path | None = None
    for candidate in single_candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            single = candidate
            break

    shards_dir: Path | None = None
    for shard_root in shard_candidates:
        shard_root = shard_root.resolve()
        if shard_root.exists():
            shards_dir = shard_root
            break

    shard_paths: list[Path] = []
    if shards_dir is not None and shards_dir.exists():
        shard_paths = sorted(shards_dir.glob("shard_*.csv"))
    if single is not None and single.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(single, dest)
        return dest
    if not shard_paths:
        return None
    rows: list[dict[str, str]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    fallback_counter = 0
    for shard in shard_paths:
        with shard.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise P0MergeError(f"p0 shard missing header: {shard}")
            if reader.fieldnames != P0_HEADER:
                raise P0MergeError(
                    "unexpected p0 header in %s: %s" % (shard, ",".join(reader.fieldnames))
                )
            for row in reader:
                key = _dedupe_key(row)
                if key[0] == 3:
                    fallback_counter += 1
                    key = (3, fallback_counter)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                normalized = {column: (row.get(column) or "") for column in P0_HEADER}
                rows.append(normalized)
    if not rows:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=P0_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return dest


def _dedupe_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    message_id = str(row.get("message_id", "")).strip()
    attachment_id = str(row.get("attachment_id", "")).strip()
    phash = str(row.get("phash_hex", "")).strip().lower()
    url = str(row.get("url", "")).strip()
    if message_id and attachment_id:
        return 0, message_id, attachment_id
    if message_id and phash:
        return 1, message_id, phash
    if url:
        return 2, url
    return 3, 0


def compile_rules_with_thresholds(
    rules_path: Path,
    *,
    thresholds_path: Path | None,
    output_path: Path,
) -> ThresholdCompileResult:
    raw_rules = _load_yaml(rules_path)
    if not isinstance(raw_rules, dict):
        raise ThresholdsError(f"rules file must be a mapping: {rules_path}")
    base_features = raw_rules.get("features")
    if base_features is None:
        base_features = {}
    if not isinstance(base_features, dict):
        raise ThresholdsError("rules.features must be a mapping")
    merged_features = dict(base_features)
    derived: dict[str, Any] = {}
    if thresholds_path is not None and thresholds_path.exists():
        thresholds_data = _load_yaml(thresholds_path)
        if thresholds_data is None:
            thresholds_data = {}
        if not isinstance(thresholds_data, Mapping):
            raise ThresholdsError("thresholds YAML must be a mapping")
        derived = _flatten_thresholds(thresholds_data)
        for name, value in derived.items():
            if name in merged_features:
                raise ThresholdsError(
                    f"threshold constant {name} already defined in features"
                )
            merged_features[name] = value
    raw_rules["features"] = merged_features
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_rules, handle, sort_keys=False, allow_unicode=True)
    return ThresholdCompileResult(rules_path=output_path, derived_symbols=derived)


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _flatten_thresholds(data: Mapping[str, Any], prefix: str | None = None) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        token = str(key)
        path = f"{prefix}.{token}" if prefix else token
        if isinstance(value, Mapping):
            flattened.update(_flatten_thresholds(value, path))
            continue
        if isinstance(value, bool):
            flattened[_feature_name(path)] = int(value)
        elif isinstance(value, (int, float)):
            flattened[_feature_name(path)] = value
        else:
            raise ThresholdsError(
                f"threshold {path} must be int/float/bool, got {type(value).__name__}"
            )
    return flattened


def _feature_name(path: str) -> str:
    normalized = re.sub(r"[^0-9A-Z]+", "_", path.strip().upper())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ThresholdsError("invalid threshold key")
    return f"{THRESHOLD_PREFIX}{normalized}"


def scan_threshold_symbols(rules_path: Path) -> ThresholdScanResult:
    raw_rules = _load_yaml(rules_path)
    if not isinstance(raw_rules, Mapping):
        raise ThresholdsError(f"rules file must be a mapping: {rules_path}")
    features = raw_rules.get("features") if isinstance(raw_rules.get("features"), Mapping) else {}
    defined = {
        key
        for key in features.keys()
        if isinstance(key, str) and key.startswith(THRESHOLD_PREFIX)
    }
    referenced = _collect_feature_references(raw_rules)
    return ThresholdScanResult(referenced=referenced, defined=defined)


def _collect_feature_references(node: Any) -> set[str]:
    referenced: set[str] = set()
    if isinstance(node, str):
        for match in FEATURE_REF_PATTERN.findall(node):
            if match.startswith(THRESHOLD_PREFIX):
                referenced.add(match)
        return referenced
    if isinstance(node, Mapping):
        for value in node.values():
            referenced.update(_collect_feature_references(value))
    elif isinstance(node, Iterable):
        for item in node:
            referenced.update(_collect_feature_references(item))
    return referenced


def ensure_thumbnail(source: Path, dest: Path, *, size: int = 256) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resample = Image.Resampling.LANCZOS  # Pillow >= 9.1
    except AttributeError:  # pragma: no cover - fallback for older Pillow
        resample = Image.LANCZOS
    with Image.open(source) as img:
        converted = img.convert("RGB")
        converted.thumbnail((size, size), resample)
        converted.save(dest, format="JPEG", quality=90)
    return dest


def run_cli_scan(
    *,
    analysis: Path,
    findings: Path,
    rules: Path,
    p0: Path | None,
    limit: int = 0,
    offset: int = 0,
    dsl_mode: str = "warn",
    profile: str,
    date: str | None,
    extra_args: Sequence[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "app.cli_scan",
        "--analysis",
        str(analysis),
        "--findings",
        str(findings),
        "--rules",
        str(rules),
        "--dsl-mode",
        dsl_mode,
        "--limit",
        str(limit),
        "--offset",
        str(offset),
        "--profile",
        profile,
    ]
    if date:
        command.extend(["--date", date])
    if p0 is not None:
        command.extend(["--p0", str(p0)])
    if extra_args:
        command.extend(extra_args)
    return _run_cli(command)


def run_cli_report(
    *,
    findings: Path,
    out_path: Path,
    severity: str | None = None,
    profile: str,
    date: str | None,
    extra_args: Sequence[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "app.cli_report",
        "--findings",
        str(findings),
        "--out",
        str(out_path),
        "--profile",
        profile,
    ]
    if date:
        command.extend(["--date", date])
    if severity and severity.lower() != "all":
        command.extend(["--severity", severity])
    if extra_args:
        command.extend(extra_args)
    return _run_cli(command)


def run_cli_ab(
    *,
    analysis: Path,
    rules_a: Path,
    rules_b: Path,
    out_dir: Path,
    sample_diff: int = 0,
    profile: str,
    date: str | None,
    extra_args: Sequence[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "app.cli_rules_ab",
        "--analysis",
        str(analysis),
        "--rulesA",
        str(rules_a),
        "--rulesB",
        str(rules_b),
        "--out-dir",
        str(out_dir),
        "--profile",
        profile,
    ]
    if date:
        command.extend(["--date", date])
    if sample_diff:
        command.extend(["--sample-diff", str(sample_diff)])
    if extra_args:
        command.extend(extra_args)
    return _run_cli(command)


def run_cli_contract_report(
    *,
    report_path: Path,
    profile: str,
    date: str | None,
    extra_args: Sequence[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "app.cli_contract",
        "check-report",
        "--path",
        str(report_path),
        "--profile",
        profile,
    ]
    if date:
        command.extend(["--date", date])
    if extra_args:
        command.extend(extra_args)
    return _run_cli(command)


def _run_cli(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise CliCommandError(command, result)
    return result
