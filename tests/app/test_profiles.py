from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

from app.profiles import (
    PartitionPaths,
    ProfileContext,
    clear_context_cache,
    get_profiles_root,
    iter_recent_findings,
    latest_partition_date,
    list_partitions,
    resolve_latest_partition,
    set_profiles_root_override,
)


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


def test_latest_partition_date_and_resolve(tmp_path):
    set_profiles_root_override(tmp_path)
    clear_context_cache()
    try:
        ctx = make_context()
        paths = PartitionPaths(ctx)
        p3_dir = paths.stage_dir("p3", ensure=True)
        (p3_dir / "findings_2025-09-29.jsonl").write_text("{}\n", encoding="utf-8")
        (p3_dir / "findings_2025-10-01.jsonl").write_text("{}\n", encoding="utf-8")

        latest = latest_partition_date(ctx.profile, "p3")
        assert latest == date(2025, 10, 1)

        resolved_path, reason = resolve_latest_partition(ctx, "p3")
        assert resolved_path.name == "findings_2025-10-01.jsonl"
        assert reason is None

        (p3_dir / "findings_2025-10-01.jsonl").unlink()
        legacy_candidate = tmp_path / "findings_legacy.jsonl"
        legacy_candidate.write_text("{}\n", encoding="utf-8")
        resolved_path, reason = resolve_latest_partition(
            ctx,
            "p3",
            legacy_candidates=[legacy_candidate],
        )
        assert resolved_path == legacy_candidate
        assert reason == "fallback=legacy"

        legacy_candidate.unlink()
        resolved_path, reason = resolve_latest_partition(ctx, "p3", legacy_candidates=[])
        assert resolved_path.name == "findings_2025-10-01.jsonl"
        assert reason == "fallback=missing"
    finally:
        set_profiles_root_override(None)
        clear_context_cache()


def test_iter_recent_findings_respects_days_back(tmp_path):
    set_profiles_root_override(tmp_path)
    clear_context_cache()
    try:
        ctx = make_context()
        root = get_profiles_root()
        base = root / ctx.profile / "p3"
        base.mkdir(parents=True, exist_ok=True)
        for token in ("2025-09-28", "2025-09-29", "2025-10-01"):
            (base / f"findings_{token}.jsonl").write_text("{}\n", encoding="utf-8")

        results = list(iter_recent_findings(ctx, days_back=2))
        # Should include newest two partitions
        assert [item[0] for item in results] == [date(2025, 10, 1), date(2025, 9, 29)]
        assert all(path.parent.parent == base.parent for _, path in results)

        other_root = tmp_path / "alt_profiles"
        other_root.mkdir()
        other_ctx = make_context(profile="other")
        (other_root / "other" / "p3").mkdir(parents=True, exist_ok=True)
        (other_root / "other" / "p3" / "findings_2025-10-05.jsonl").write_text("{}\n", encoding="utf-8")

        results = list(iter_recent_findings(other_ctx, days_back=1, root=other_root))
        assert [item[0] for item in results] == [date(2025, 10, 5)]
        assert results[0][1].parent.parent == other_root / "other"
    finally:
        set_profiles_root_override(None)
        clear_context_cache()
