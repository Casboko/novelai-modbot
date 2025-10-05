from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from app.config import get_settings
from app.profiles import PartitionPaths, ProfileContext

OLD_STAGE_FILES: Dict[str, Tuple[Path, ...]] = {
    "p0": (Path("p0_scan.csv"),),
    "p1": (Path("p1_wd14.jsonl"),),
    "p2": (
        Path("p2/p2_analysis_all.jsonl"),
        Path("p2_analysis_all.jsonl"),
        Path("p2/p2_analysis.jsonl"),
        Path("p2_analysis.jsonl"),
    ),
    "p3": (
        Path("p3/p3_findings_all.jsonl"),
        Path("p3/p3_decision_all.jsonl"),
        Path("p3/p3_findings.jsonl"),
        Path("p3_findings.jsonl"),
    ),
}

OLD_METRICS: Dict[str, Tuple[Path, ...]] = {
    "p1": (Path("p1_wd14_metrics.json"),),
    "p2": (Path("p2_metrics.json"), Path("metrics/p2_metrics.json")),
    "p3": (Path("metrics/p3_run.json"), Path("p3/metrics/p3_run.json")),
}

OLD_REPORTS: Tuple[Path, ...] = (Path("p3/p3_report.csv"), Path("p3_report.csv"))
OLD_REPORT_EXT: Tuple[Path, ...] = (Path("p3/p3_report_ext.csv"),)
OLD_ATTACHMENTS: Tuple[Path, ...] = (Path("p0/p0_attachments.json"), Path("p0_attachments.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate legacy out/ files into profile partitions")
    parser.add_argument("--source", type=Path, default=Path("out"), help="Legacy out/ root")
    parser.add_argument("--profile", type=str, help="Destination profile (default: legacy)")
    parser.add_argument("--date", type=str, help="Partition date to assign")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without moving")
    parser.add_argument("--force", action="store_true", help="Overwrite destination if it exists")
    return parser.parse_args()


def find_first_existing(root: Path, candidates: Iterable[Path]) -> Optional[Path]:
    for relative in candidates:
        candidate = root / relative
        if candidate.exists():
            return candidate
    return None


def move_file(source: Path, destination: Path, *, dry_run: bool, force: bool) -> bool:
    if destination.exists():
        if not force:
            print(f"[skip] {destination} exists (use --force)" )
            return False
        if not dry_run:
            destination.unlink()
    action = "Would move" if dry_run else "Moving"
    print(f"{action} {source} -> {destination}")
    if dry_run:
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    return True


def main() -> None:
    args = parse_args()
    settings = get_settings()
    context = settings.build_profile_context(profile=args.profile, date=args.date)
    partitions = PartitionPaths(context)

    moved = 0

    partitions = PartitionPaths(context)

    for stage, candidates in OLD_STAGE_FILES.items():
        source_path = find_first_existing(args.source, candidates)
        if not source_path:
            continue
        dest_path = partitions.stage_file(stage)
        if move_file(source_path, dest_path, dry_run=args.dry_run, force=args.force):
            moved += 1

    for stage, candidates in OLD_METRICS.items():
        source_path = find_first_existing(args.source, candidates)
        if not source_path:
            continue
        dest_path = partitions.metrics_file(stage)
        if move_file(source_path, dest_path, dry_run=args.dry_run, force=args.force):
            moved += 1

    report_source = find_first_existing(args.source, OLD_REPORTS)
    if report_source:
        if move_file(report_source, partitions.report_path(), dry_run=args.dry_run, force=args.force):
            moved += 1

    report_ext_source = find_first_existing(args.source, OLD_REPORT_EXT)
    if report_ext_source:
        if move_file(report_ext_source, partitions.report_ext_path(), dry_run=args.dry_run, force=args.force):
            moved += 1

    attachments_source = find_first_existing(args.source, OLD_ATTACHMENTS)
    if attachments_source:
        if move_file(attachments_source, partitions.attachments_index(), dry_run=args.dry_run, force=args.force):
            moved += 1

    if args.dry_run:
        print(f"[dry-run] {moved} moves planned")
    else:
        print(f"Migration complete. {moved} items moved.")


if __name__ == "__main__":
    main()
