from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

from app.profiles import PartitionPaths, ProfileContext, list_partitions


def make_context(profile: str = "demo", day: date | None = None) -> ProfileContext:
    actual_date = day or date(2025, 10, 1)
    return ProfileContext(profile=profile, date=actual_date, tzinfo=ZoneInfo("UTC"))


def test_stage_file_and_metrics_paths(tmp_path):
    context = make_context()
    paths = PartitionPaths(context, root=tmp_path)

    analysis_path = paths.stage_file("p2", ensure_parent=True)
    metrics_path = paths.metrics_file("p2", ensure_parent=True)

    assert analysis_path == tmp_path / "demo" / "p2" / "p2_2025-10-01.jsonl"
    assert metrics_path == tmp_path / "demo" / "metrics" / "p2" / "p2_metrics_2025-10-01.json"
    assert analysis_path.parent.is_dir()
    assert metrics_path.parent.is_dir()


def test_list_partitions_respects_order(tmp_path):
    base = tmp_path / "demo" / "p2"
    base.mkdir(parents=True, exist_ok=True)
    for token in ("2025-10-01", "2025-10-03", "2025-10-02"):
        (base / f"p2_{token}.jsonl").write_text("{}\n", encoding="utf-8")

    descending = list_partitions("demo", "p2", root=tmp_path)
    assert descending[:3] == [
        date(2025, 10, 3),
        date(2025, 10, 2),
        date(2025, 10, 1),
    ]

    ascending = list_partitions("demo", "p2", order="asc", root=tmp_path)
    assert ascending[:3] == [
        date(2025, 10, 1),
        date(2025, 10, 2),
        date(2025, 10, 3),
    ]


def test_partition_paths_for_attachments(tmp_path):
    context = make_context(day=date(2025, 11, 5))
    paths = PartitionPaths(context, root=tmp_path)
    idx_path = paths.attachments_index(ensure_parent=True)

    assert idx_path == tmp_path / "demo" / "attachments" / "p0" / "p0_2025-11-05_index.json"
    assert idx_path.parent.is_dir()

