from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from app.config import get_settings
from app.profiles import LEGACY_PROFILE, PartitionPaths, ProfileContext

WD14_CACHE_DEFAULT = Path("app/cache_wd14.sqlite")
NUDENET_CACHE_DEFAULT = Path("app/cache_nudenet.sqlite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone shared cache DBs for profile-specific runs")
    parser.add_argument("--profile", type=str, help="Destination profile (default: settings/legacy)")
    parser.add_argument("--date", type=str, help="Partition date used for derived paths")
    parser.add_argument(
        "--wd14-source",
        type=Path,
        default=WD14_CACHE_DEFAULT,
        help="Source WD14 cache database",
    )
    parser.add_argument(
        "--nudenet-source",
        type=Path,
        default=NUDENET_CACHE_DEFAULT,
        help="Source NudeNet cache database",
    )
    parser.add_argument("--wd14-dest", type=Path, help="Destination WD14 cache path override")
    parser.add_argument("--nudenet-dest", type=Path, help="Destination NudeNet cache path override")
    parser.add_argument("--force", action="store_true", help="Overwrite destination if it exists")
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without copying")
    return parser.parse_args()


def ensure_dest_path(path: Path, *, force: bool, dry_run: bool) -> bool:
    if path.exists() and not force:
        print(f"[skip] {path} already exists (use --force to overwrite)")
        return False
    if path.exists() and force and not dry_run:
        path.unlink()
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
    return True


def clone_cache(src: Path, dest: Path, *, force: bool, dry_run: bool) -> None:
    if not src.exists():
        print(f"[warn] source cache not found: {src}")
        return
    if not ensure_dest_path(dest, force=force, dry_run=dry_run):
        return
    action = "Would copy" if dry_run else "Copying"
    print(f"{action} {src} -> {dest}")
    if not dry_run:
        shutil.copy2(src, dest)


def default_dest(base: Path, profile: str) -> Path:
    return base.with_name(f"{base.stem}_{profile}{base.suffix}")


def main() -> None:
    args = parse_args()
    settings = get_settings()
    profile_arg = args.profile or LEGACY_PROFILE
    context = settings.build_profile_context(profile=profile_arg, date=args.date)
    partitions = PartitionPaths(context)

    dest_wd14 = args.wd14_dest or default_dest(WD14_CACHE_DEFAULT, context.profile)
    dest_nudenet = args.nudenet_dest or default_dest(NUDENET_CACHE_DEFAULT, context.profile)

    clone_cache(args.wd14_source, dest_wd14, force=args.force, dry_run=args.dry_run)
    clone_cache(args.nudenet_source, dest_nudenet, force=args.force, dry_run=args.dry_run)

    if not args.dry_run:
        manifest_path = partitions.status_manifest("cache", ensure_parent=True)
        payload = {
            "profile": context.profile,
            "wd14": str(dest_wd14.resolve()),
            "nudenet": str(dest_nudenet.resolve()),
            "source_wd14": str(args.wd14_source.resolve()),
            "source_nudenet": str(args.nudenet_source.resolve()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path.write_text(json_dumps(payload), encoding="utf-8")


def json_dumps(data: dict) -> str:
    import json

    return json.dumps(data, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

