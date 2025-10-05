from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.config import get_settings
from app.profiles import PartitionPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WD14 inference shards with orchestration")
    parser.add_argument("--shard-glob", required=True, help="Glob pattern to locate shard CSV files")
    parser.add_argument("--out-dir", type=Path, help="Output directory for shard artifacts")
    parser.add_argument("--provider", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--qps", type=float, default=5.0)
    parser.add_argument("--parallel", type=int, default=1, help="Maximum concurrent shard executions")
    parser.add_argument("--resume", action="store_true", help="Skip shards whose outputs already exist")
    parser.add_argument("--status-file", type=Path, help="Path to manifest JSON for status tracking")
    parser.add_argument("--retries", type=int, default=1, help="Number of retries for failed shards")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments appended to each cli_wd14 invocation",
    )
    parser.add_argument("--profile", type=str, help="Profile name for partition defaults")
    parser.add_argument(
        "--date",
        type=str,
        help="Partition date (YYYY-MM-DD, today, yesterday). Default resolved via profile timezone",
    )
    return parser.parse_args()


@dataclass
class ShardTask:
    csv_path: Path
    label: str
    out_path: Path
    metrics_path: Path
    tmp_out: Path
    tmp_metrics: Path


class Manifest:
    def __init__(self, stage: str, path: Optional[Path]) -> None:
        self.stage = stage
        self.path = path
        self.data: Dict[str, Any] = {"stage": stage, "updated_at": None, "shards": {}}
        if path and path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data = loaded
            except json.JSONDecodeError:
                pass
        self._dirty = False

    def mark_dirty(self) -> None:
        self._dirty = True

    def update_shard(self, label: str, payload: Dict[str, Any]) -> None:
        self.data.setdefault("shards", {})[label] = payload
        self.mark_dirty()

    def get_shard(self, label: str) -> Dict[str, Any]:
        return self.data.setdefault("shards", {}).setdefault(label, {})

    def save(self) -> None:
        if not self._dirty:
            return
        if not self.path:
            self._dirty = False
            return
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        self._dirty = False


def natural_key(path: Path) -> List[Any]:
    import re

    parts = re.split(r"(\d+)", path.stem)
    key: List[Any] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        elif part:
            key.append(part.lower())
    key.append(path.suffix.lower())
    return key


def derive_label(shard_path: Path) -> str:
    return shard_path.stem


def prepare_tasks(args: argparse.Namespace) -> list[ShardTask]:
    shard_files = sorted((Path(p) for p in glob.glob(args.shard_glob)), key=natural_key)
    if not shard_files:
        raise SystemExit(f"No shard files matched: {args.shard_glob}")
    out_dir = args.out_dir
    p1_dir = out_dir / "p1"
    metrics_dir = out_dir / "metrics" / "p1"
    p1_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[ShardTask] = []
    for shard_path in shard_files:
        label = derive_label(shard_path)
        out_path = p1_dir / f"p1_wd14_{label}.jsonl"
        metrics_path = metrics_dir / f"p1_{label}.json"
        tmp_out = Path(f"{out_path}.tmp")
        tmp_metrics = Path(f"{metrics_path}.tmp")
        tasks.append(
            ShardTask(
                csv_path=shard_path,
                label=label,
                out_path=out_path,
                metrics_path=metrics_path,
                tmp_out=tmp_out,
                tmp_metrics=tmp_metrics,
            )
        )
    return tasks


async def execute_task(
    task: ShardTask,
    args: argparse.Namespace,
    manifest: Manifest,
    attempt: int,
    semaphore: asyncio.Semaphore,
) -> bool:
    async with semaphore:
        entry = manifest.get_shard(task.label)
        entry.update(
            {
                "state": "running",
                "attempt": attempt,
                "tmp": str(task.tmp_out),
            }
        )
        manifest.mark_dirty()
        manifest.save()

        if task.tmp_out.exists():
            task.tmp_out.unlink()
        if task.tmp_metrics.exists():
            task.tmp_metrics.unlink()

        cmd = [
            str(args.python),
            "-m",
            "app.cli_wd14",
            "--input",
            str(task.csv_path),
            "--out",
            str(task.tmp_out),
            "--metrics",
            str(task.tmp_metrics),
            "--provider",
            str(args.provider),
            "--batch-size",
            str(args.batch_size),
            "--concurrency",
            str(args.concurrency),
            "--qps",
            str(args.qps),
        ]

        if args.extra_args:
            cmd.extend(args.extra_args)

        process = await asyncio.create_subprocess_exec(*cmd)
        entry["pid"] = process.pid
        manifest.mark_dirty()
        manifest.save()

        returncode = await process.wait()

        entry["exit_code"] = returncode
        manifest.mark_dirty()

        if returncode == 0 and task.tmp_out.exists():
            os.replace(task.tmp_out, task.out_path)
            if task.tmp_metrics.exists():
                os.replace(task.tmp_metrics, task.metrics_path)
            else:
                entry["state"] = "failed"
                entry["reason"] = "metrics_missing"
                manifest.mark_dirty()
                manifest.save()
                return False
            entry.update(
                {
                    "state": "done",
                    "retries": attempt,
                    "out": str(task.out_path),
                    "metrics": str(task.metrics_path),
                }
            )
            manifest.mark_dirty()
            manifest.save()
            return True

        # failure path
        entry.update(
            {
                "state": "failed",
                "retries": attempt,
            }
        )
        manifest.mark_dirty()
        manifest.save()
        return False


def filter_tasks(tasks: Iterable[ShardTask], args: argparse.Namespace, manifest: Manifest) -> deque[ShardTask]:
    queue: deque[ShardTask] = deque()
    for task in tasks:
        if args.resume and task.out_path.exists():
            entry = manifest.get_shard(task.label)
            entry.update(
                {
                    "state": "done",
                    "exit_code": 0,
                    "retries": entry.get("retries", 0),
                    "out": str(task.out_path),
                    "metrics": str(task.metrics_path),
                    "skipped": True,
                }
            )
            manifest.mark_dirty()
            continue
        entry = manifest.get_shard(task.label)
        if entry.get("state") not in {"running", "done"}:
            entry.update({"state": "queued", "out": str(task.out_path), "metrics": str(task.metrics_path)})
            manifest.mark_dirty()
        queue.append(task)
    manifest.save()
    return queue


async def run_all(tasks: list[ShardTask], args: argparse.Namespace, manifest: Manifest) -> int:
    remaining = filter_tasks(tasks, args, manifest)
    if not remaining:
        manifest.save()
        return 0

    parallel = max(1, args.parallel)
    semaphore = asyncio.Semaphore(parallel)
    attempt = 0
    failures: list[ShardTask] = []

    while remaining:
        tasks_in_round = list(remaining)
        remaining.clear()
        results = await asyncio.gather(
            *[execute_task(task, args, manifest, attempt, semaphore) for task in tasks_in_round]
        )
        for task, success in zip(tasks_in_round, results):
            if not success:
                failures.append(task)
        attempt += 1
        if failures and attempt <= args.retries:
            remaining.extend(failures)
            failures = []

    manifest.save()
    if failures:
        return 1
    return 0


def main() -> None:
    args = parse_args()
    settings = get_settings()
    context = settings.build_profile_context(profile=args.profile, date=args.date)
    partitions = PartitionPaths(context)
    if args.out_dir is None:
        args.out_dir = partitions.profile_root(ensure=True)
    if args.status_file is None:
        args.status_file = partitions.status_manifest("p1", ensure_parent=True)
    tasks = prepare_tasks(args)
    manifest = Manifest(stage="p1", path=args.status_file)
    exit_code = asyncio.run(run_all(tasks, args, manifest))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
