from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.jsonl_merge import load_updates_from_jsonl, merge_jsonl_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge JSONL files by key without loading the base file into memory",
    )
    parser.add_argument("--base", type=Path, required=True, help="Existing JSONL file")
    parser.add_argument(
        "--updates",
        type=Path,
        nargs="+",
        required=True,
        help="One or more JSONL files containing new or updated records",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output JSONL path (defaults to overwriting --base)",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="phash",
        help="Field name used as the unique key",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = args.base
    if not base_path.exists():
        raise SystemExit(f"Base file not found: {base_path}")
    update_paths = [path for path in args.updates if path.exists()]
    if not update_paths:
        raise SystemExit("No update files found")

    updates = load_updates_from_jsonl(update_paths, args.key)

    out_path = args.out or base_path
    merge_jsonl_records(
        base_path=base_path,
        updates=updates,
        key_field=args.key,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()

