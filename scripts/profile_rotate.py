from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

from app.config import get_settings
from app.profiles import (
    LEGACY_PROFILE,
    PartitionPaths,
    ProfileContext,
    list_partitions,
)

StageList = Tuple[str, ...]

PRIMARY_STAGES: StageList = ("p0", "p1", "p2", "p3")
METRIC_STAGES: StageList = ("p1", "p2", "p3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rotate profile partitions into legacy storage")
    parser.add_argument("--profile", type=str, help="Source profile (default: settings profile)")
    parser.add_argument(
        "--dest-profile",
        type=str,
        default=LEGACY_PROFILE,
        help="Destination profile (default: legacy)",
    )
    parser.add_argument("--date", type=str, help="Reference date for retention window")
    parser.add_argument("--retention-days", type=int, help="Number of days to retain")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without moving files")
    parser.add_argument("--force", action="store_true", help="Overwrite destination files if they exist")
    return parser.parse_args()


def cutoff_date(reference: datetime, retention_days: int) -> datetime:
    if retention_days <= 0:
        raise SystemExit("--retention-days must be >= 1")
    return reference - timedelta(days=retention_days - 1)


def list_eligible_dates(profile: str, stage: str, cutoff: datetime) -> List[datetime.date]:
    candidates = list_partitions(profile, stage)
    return [item for item in candidates if item < cutoff.date()]


def ensure_directory(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def move_file(source: Path, destination: Path, *, dry_run: bool, force: bool) -> bool:
    if not source.exists():
        return False
    if destination.exists():
        if not force:
            print(f"[skip] {destination} already exists (use --force)")
            return False
        if not dry_run:
            destination.unlink()
    ensure_directory(destination.parent, dry_run=dry_run)
    action = "Would move" if dry_run else "Moving"
    print(f"{action} {source} -> {destination}")
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
    _cleanup_if_empty(source.parent, dry_run=dry_run)
    return True


def _cleanup_if_empty(path: Path, *, dry_run: bool) -> None:
    if not path.exists():
        return
    try:
        next(path.iterdir())
    except StopIteration:
        if dry_run:
            print(f"[dry-run] would remove empty directory {path}")
        else:
            path.rmdir()


def history_path(partitions: PartitionPaths) -> Path:
    return partitions.profile_root(ensure=True) / "status" / "rotation_history.json"


def append_history(path: Path, payload: dict, *, dry_run: bool) -> None:
    if dry_run:
        return
    history: List[dict] = []
    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []
    history.append(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    settings = get_settings()
    source_context = settings.build_profile_context(profile=args.profile, date=args.date)
    retention_days = args.retention_days or settings.profile_retention_days
    reference_dt = datetime.combine(source_context.date, datetime.min.time(), tzinfo=source_context.tzinfo)
    cutoff = cutoff_date(reference_dt, retention_days)

    dest_context = ProfileContext(profile=args.dest_profile, date=source_context.date, tzinfo=source_context.tzinfo)

    source_partitions = PartitionPaths(source_context)
    dest_partitions = PartitionPaths(dest_context)

    movements: List[dict] = []

    for stage in PRIMARY_STAGES:
        dates = list_eligible_dates(source_context.profile, stage, cutoff)
        for day in dates:
            src_ctx = source_context.with_date(date_token=day.isoformat())
            dst_ctx = ProfileContext(profile=dest_context.profile, date=day, tzinfo=dest_context.tzinfo)
            src_path = PartitionPaths(src_ctx).stage_file(stage)
            dst_path = PartitionPaths(dst_ctx).stage_file(stage)
            if move_file(src_path, dst_path, dry_run=args.dry_run, force=args.force):
                movements.append({"stage": stage, "date": day.isoformat(), "source": str(src_path), "dest": str(dst_path)})

            # move report assets for p3
            if stage == "p3":
                src_report = PartitionPaths(src_ctx).report_path()
                dst_report = PartitionPaths(dst_ctx).report_path()
                if src_report.exists():
                    move_file(src_report, dst_report, dry_run=args.dry_run, force=args.force)
                src_report_ext = PartitionPaths(src_ctx).report_ext_path()
                dst_report_ext = PartitionPaths(dst_ctx).report_ext_path()
                if src_report_ext.exists():
                    move_file(src_report_ext, dst_report_ext, dry_run=args.dry_run, force=args.force)

            if stage == "p0":
                src_attach = PartitionPaths(src_ctx).attachments_index()
                dst_attach = PartitionPaths(dst_ctx).attachments_index()
                if move_file(src_attach, dst_attach, dry_run=args.dry_run, force=args.force):
                    movements.append({
                        "stage": "attachments",
                        "date": day.isoformat(),
                        "source": str(src_attach),
                        "dest": str(dst_attach),
                    })

    for stage in METRIC_STAGES:
        dates = list_eligible_dates(source_context.profile, stage, cutoff)
        for day in dates:
            src_ctx = source_context.with_date(date_token=day.isoformat())
            dst_ctx = ProfileContext(profile=dest_context.profile, date=day, tzinfo=dest_context.tzinfo)
            src_metrics = PartitionPaths(src_ctx).metrics_file(stage)
            dst_metrics = PartitionPaths(dst_ctx).metrics_file(stage)
            if move_file(src_metrics, dst_metrics, dry_run=args.dry_run, force=args.force):
                movements.append({
                    "stage": f"metrics:{stage}",
                    "date": day.isoformat(),
                    "source": str(src_metrics),
                    "dest": str(dst_metrics),
                })

    if args.dry_run:
        print(f"[dry-run] rotations planned: {len(movements)}")
        return

    if not movements:
        print("No partitions eligible for rotation")
        return

    history_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_profile": source_context.profile,
        "destination_profile": dest_context.profile,
        "retention_days": retention_days,
        "cutoff": cutoff.date().isoformat(),
        "moved": movements,
    }
    append_history(history_path(source_partitions), history_entry, dry_run=False)
    print(f"Rotation complete. Moved {len(movements)} items.")


if __name__ == "__main__":
    main()

