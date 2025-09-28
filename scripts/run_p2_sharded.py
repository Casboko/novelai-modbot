from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import glob
from typing import Any, Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run analysis merge shards with orchestration")
    parser.add_argument("--shard-glob", required=True, help="Glob pattern to locate shard CSV files")
    parser.add_argument("--wd14-dir", type=Path, required=True, help="Directory containing p1 shard outputs")
    parser.add_argument("--out-dir", type=Path, default=Path("out"))
    parser.add_argument("--qps", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Skip shards whose outputs already exist")
    parser.add_argument("--status-file", type=Path, help="Path to manifest JSON for status tracking")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments appended to each analysis_merge invocation",
    )
    return parser.parse_args()


@dataclass
class ShardTask:
    csv_path: Path
    label: str
    wd14_path: Path
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


def derive_label(path: Path) -> str:
    return path.stem


def prepare_tasks(args: argparse.Namespace) -> list[ShardTask]:
    shard_files = sorted((Path(p) for p in glob.glob(args.shard_glob)), key=natural_key)
    if not shard_files:
        raise SystemExit(f"No shard files matched: {args.shard_glob}")

    wd14_dir = args.wd14_dir
    out_dir = args.out_dir
    p2_dir = out_dir / "p2"
    metrics_dir = out_dir / "metrics"
    p2_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[ShardTask] = []
    for shard_path in shard_files:
        label = derive_label(shard_path)
        wd14_path = wd14_dir / f"p1_wd14_{label}.jsonl"
        out_path = p2_dir / f"p2_analysis_{label}.jsonl"
        metrics_path = metrics_dir / f"p2_{label}.json"
        tmp_out = Path(f"{out_path}.tmp")
        tmp_metrics = Path(f"{metrics_path}.tmp")
        tasks.append(
            ShardTask(
                csv_path=shard_path,
                label=label,
                wd14_path=wd14_path,
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
        if not task.wd14_path.exists():
            entry = manifest.get_shard(task.label)
            entry.update(
                {
                    "state": "failed",
                    "exit_code": -1,
                    "retries": attempt,
                    "reason": f"Missing WD14 shard: {task.wd14_path}",
                }
            )
            manifest.mark_dirty()
            manifest.save()
            return False

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
            "app.analysis_merge",
            "--scan",
            str(task.csv_path),
            "--wd14",
            str(task.wd14_path),
            "--out",
            str(task.tmp_out),
            "--metrics",
            str(task.tmp_metrics),
            "--qps",
            str(args.qps),
            "--concurrency",
            str(args.concurrency),
        ]

        if args.batch_size:
            cmd.extend(["--batch-size", str(args.batch_size)])

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
    tasks = prepare_tasks(args)
    manifest = Manifest(stage="p2", path=args.status_file)
    exit_code = asyncio.run(run_all(tasks, args, manifest))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
