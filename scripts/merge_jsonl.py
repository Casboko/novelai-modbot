from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple JSONL files into one")
    parser.add_argument("--glob", required=True, help="Glob pattern for JSONL files")
    parser.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--append-newline", action="store_true", help="Ensure trailing newline at EOF")
    return parser.parse_args()


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


def iter_files(pattern: str) -> list[Path]:
    paths = [Path(p) for p in glob.glob(pattern)]
    filtered = [path for path in paths if not path.name.endswith(".tmp")]
    return sorted(filtered, key=natural_key)


def merge(files: Iterable[Path], output_path: Path, append_newline: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_fp:
        first = True
        for file_path in files:
            with file_path.open("r", encoding="utf-8") as in_fp:
                for line in in_fp:
                    if first:
                        first = False
                    out_fp.write(line)
        if append_newline and not first:
            out_fp.write("\n")


def main() -> None:
    args = parse_args()
    files = iter_files(args.glob)
    if not files:
        raise SystemExit(f"No files matched pattern: {args.glob}")
    merge(files, args.out, args.append_newline)


if __name__ == "__main__":
    main()
