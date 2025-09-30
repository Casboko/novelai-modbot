from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence


@dataclass
class TagAggregate:
    tag_name: str
    primary_categories: set[str] = field(default_factory=set)
    detailed_categories: set[str] = field(default_factory=set)
    max_d_count: int | None = None
    max_n_count: int | None = None
    source_files: set[str] = field(default_factory=set)

    def add_entry(self, entry: dict[str, object], source: Path) -> None:
        primary = entry.get("primaryCategory")
        if isinstance(primary, str):
            self.primary_categories.add(primary)
        categories = entry.get("detailedCategories")
        if isinstance(categories, list):
            for category in categories:
                if isinstance(category, str):
                    self.detailed_categories.add(category)
        d_count = entry.get("dCount")
        if isinstance(d_count, int):
            if self.max_d_count is None or d_count > self.max_d_count:
                self.max_d_count = d_count
        n_count = entry.get("nCount")
        if isinstance(n_count, int):
            if self.max_n_count is None or n_count > self.max_n_count:
                self.max_n_count = n_count
        self.source_files.add(str(source))


@dataclass
class Wd14Tag:
    tag_id: int
    name: str
    category: int
    count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate classified tag JSON files, filter by detailed categories, "
            "and intersect with WD14 selected tags."
        )
    )
    parser.add_argument(
        "--tags-dir",
        type=Path,
        default=Path("tmp/tags"),
        help="Directory containing classified tag JSON files.",
    )
    parser.add_argument(
        "--wd14-csv",
        type=Path,
        default=Path("models/wd14/selected_tags.csv"),
        help="Path to WD14 selected_tags.csv file.",
    )
    parser.add_argument(
        "--detailed-prefix",
        action="append",
        default=[],
        help="Keep tags that have at least one detailed category starting with the given prefix."
        " May be passed multiple times.",
    )
    parser.add_argument(
        "--detailed-exact",
        action="append",
        default=[],
        help="Keep tags that have at least one detailed category exactly matching the given value."
        " May be passed multiple times.",
    )
    parser.add_argument(
        "--primary-category",
        action="append",
        default=[],
        help="Only keep tags whose primary categories include one of the provided values.",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "tsv", "json"],
        default="csv",
        help="Output format. Defaults to csv.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output file path. Defaults to stdout when not provided.",
    )
    parser.add_argument(
        "--include-sources",
        action="store_true",
        help="Include source JSON filenames in the output.",
    )
    parser.add_argument(
        "--include-counts",
        action="store_true",
        help="Include maximum observed dCount/nCount columns in the output.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of rows in the output after sorting.",
    )
    return parser.parse_args()


def load_wd14_tags(path: Path) -> dict[str, Wd14Tag]:
    if not path.exists():
        raise FileNotFoundError(f"WD14 CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        mapping: dict[str, Wd14Tag] = {}
        for row in reader:
            name = row.get("name")
            if not name:
                continue
            try:
                tag = Wd14Tag(
                    tag_id=int(row["tag_id"]),
                    name=name,
                    category=int(row["category"]),
                    count=int(row["count"]),
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid WD14 row: {row}") from exc
            mapping[tag.name] = tag
    return mapping


def load_tag_catalog(tags_dir: Path) -> dict[str, TagAggregate]:
    if not tags_dir.is_dir():
        raise NotADirectoryError(f"Tag directory not found: {tags_dir}")
    catalog: dict[str, TagAggregate] = {}
    for file_path in sorted(tags_dir.glob("*.json")):
        raw = file_path.read_text(encoding="utf-8")
        try:
            entries = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON: {file_path}") from exc
        if not isinstance(entries, list):
            raise ValueError(f"Unexpected JSON structure (expected list): {file_path}")
        relative = file_path.relative_to(tags_dir)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            tag_name = entry.get("tagName")
            if not isinstance(tag_name, str):
                continue
            aggregate = catalog.get(tag_name)
            if aggregate is None:
                aggregate = TagAggregate(tag_name=tag_name)
                catalog[tag_name] = aggregate
            aggregate.add_entry(entry, relative)
    return catalog


def filter_entries(
    entries: Iterable[TagAggregate],
    detailed_prefixes: Sequence[str],
    detailed_exacts: Sequence[str],
    primary_categories: Sequence[str],
) -> Iterator[TagAggregate]:
    prefixes = tuple(prefix for prefix in detailed_prefixes if prefix)
    exacts = set(value for value in detailed_exacts if value)
    primary = set(value for value in primary_categories if value)

    for entry in entries:
        if prefixes:
            if not any(
                category.startswith(prefix) for prefix in prefixes for category in entry.detailed_categories
            ):
                continue
        if exacts:
            if not entry.detailed_categories.intersection(exacts):
                continue
        if primary:
            if not entry.primary_categories.intersection(primary):
                continue
        yield entry


def build_rows(
    catalog: dict[str, TagAggregate],
    wd14_map: dict[str, Wd14Tag],
    detailed_prefixes: Sequence[str],
    detailed_exacts: Sequence[str],
    primary_categories: Sequence[str],
    limit: int | None,
) -> list[tuple[TagAggregate, Wd14Tag]]:
    matched: list[tuple[TagAggregate, Wd14Tag]] = []
    for entry in filter_entries(catalog.values(), detailed_prefixes, detailed_exacts, primary_categories):
        wd14_tag = wd14_map.get(entry.tag_name)
        if wd14_tag is None:
            continue
        matched.append((entry, wd14_tag))
    matched.sort(key=lambda pair: pair[0].tag_name)
    if limit is not None:
        return matched[:limit]
    return matched


def write_csv(
    rows: list[tuple[TagAggregate, Wd14Tag]],
    out_path: Path | None,
    include_sources: bool,
    include_counts: bool,
    delimiter: str,
) -> None:
    fieldnames = ["tagName", "wd14Category", "wd14Count", "primaryCategories", "detailedCategories"]
    if include_sources:
        fieldnames.append("sourceFiles")
    if include_counts:
        fieldnames.extend(["maxDCount", "maxNCount"])

    def entry_to_row(entry: TagAggregate, wd14_tag: Wd14Tag) -> dict[str, object]:
        row: dict[str, object] = {
            "tagName": entry.tag_name,
            "wd14Category": wd14_tag.category,
            "wd14Count": wd14_tag.count,
            "primaryCategories": "|".join(sorted(entry.primary_categories)) if entry.primary_categories else "",
            "detailedCategories": "|".join(sorted(entry.detailed_categories)) if entry.detailed_categories else "",
        }
        if include_sources:
            row["sourceFiles"] = "|".join(sorted(entry.source_files))
        if include_counts:
            row["maxDCount"] = entry.max_d_count if entry.max_d_count is not None else ""
            row["maxNCount"] = entry.max_n_count if entry.max_n_count is not None else ""
        return row

    output_file = None
    writer_fp = sys.stdout
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer_fp = out_path.open("w", encoding="utf-8", newline="")
        output_file = writer_fp
    try:
        writer = csv.DictWriter(writer_fp, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for entry, wd14_tag in rows:
            writer.writerow(entry_to_row(entry, wd14_tag))
    finally:
        if output_file is not None:
            output_file.close()


def write_json(
    rows: list[tuple[TagAggregate, Wd14Tag]],
    out_path: Path | None,
    include_sources: bool,
    include_counts: bool,
) -> None:
    serialised: list[dict[str, object]] = []
    for entry, wd14_tag in rows:
        item: dict[str, object] = {
            "tagName": entry.tag_name,
            "wd14": {
                "tagId": wd14_tag.tag_id,
                "category": wd14_tag.category,
                "count": wd14_tag.count,
            },
            "primaryCategories": sorted(entry.primary_categories),
            "detailedCategories": sorted(entry.detailed_categories),
        }
        if include_sources:
            item["sourceFiles"] = sorted(entry.source_files)
        if include_counts:
            item["maxDCount"] = entry.max_d_count
            item["maxNCount"] = entry.max_n_count
        serialised.append(item)
    target = sys.stdout
    output_file = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = out_path.open("w", encoding="utf-8")
        target = output_file
    try:
        json.dump(serialised, target, ensure_ascii=False, indent=2)
        if target is sys.stdout:
            target.write("\n")
    finally:
        if output_file is not None:
            output_file.close()


def main() -> None:
    args = parse_args()
    wd14_map = load_wd14_tags(args.wd14_csv)
    tag_catalog = load_tag_catalog(args.tags_dir)
    rows = build_rows(
        tag_catalog,
        wd14_map,
        args.detailed_prefix,
        args.detailed_exact,
        args.primary_category,
        args.limit,
    )

    if args.format == "json":
        write_json(rows, args.out, args.include_sources, args.include_counts)
    else:
        delimiter = "," if args.format == "csv" else "\t"
        write_csv(rows, args.out, args.include_sources, args.include_counts, delimiter)


if __name__ == "__main__":
    main()
