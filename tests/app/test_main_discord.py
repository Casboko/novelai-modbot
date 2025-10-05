from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import app.main as main
from app.main import (
    ContextPaths,
    _format_context_notice,
    _iter_finding_candidates,
    _make_context_result,
    _merge_fallback_reason,
    _ticket_context_result,
    create_context_factory,
    find_record_for_message,
    load_findings_index,
)
from app.profiles import ContextResolveResult, ProfileContext, clear_context_cache
from app.store import Ticket
from tests.fixtures.profiles_helpers import write_partition_jsonl


class DummySettings:
    def __init__(self, *, base_profile: str = "demo", base_date: date = date(2025, 10, 1)) -> None:
        self.profile_default = base_profile
        self.profile_timezone = None
        self.timezone = "UTC"
        self._base_date = base_date

    def build_profile_context(self, profile: str | None = None, date: str | None = None) -> ProfileContext:
        date_token = date or self._base_date.isoformat()
        return ProfileContext.from_cli(
            profile_arg=profile,
            date_arg=date_token,
            default_profile=self.profile_default,
            timezone_hint=self.profile_timezone,
            default_timezone=self.timezone,
        )


def test_create_context_factory_uses_latest_partition(profiles_root_override: Path) -> None:
    settings = DummySettings(base_date=date(2025, 10, 1))
    base_context, context_factory = create_context_factory(settings)
    assert base_context.iso_date == "2025-10-01"

    write_partition_jsonl(profiles_root_override, base_context.profile, "2025-10-03", "p2", [{}])

    result = context_factory()
    assert result.context.iso_date == "2025-10-03"
    assert result.fallback_reason is None

    for path in (profiles_root_override / base_context.profile / "p2").glob("*"):
        if path.is_file():
            path.unlink()

    result_missing = context_factory()
    assert result_missing.context.iso_date == base_context.iso_date
    assert result_missing.fallback_reason == "fallback=missing"


def test_create_context_factory_respects_explicit_date(monkeypatch, profiles_root_override: Path) -> None:
    settings = DummySettings(base_date=date(2025, 10, 1))
    _, context_factory = create_context_factory(settings)

    def _raise(*args, **kwargs):  # pragma: no cover - ensures no implicit call
        raise AssertionError("latest_partition_date should not be called")

    monkeypatch.setattr(main, "latest_partition_date", _raise)

    result = context_factory(date="2025-09-15")
    assert result.context.iso_date == "2025-09-15"
    assert result.fallback_reason == "fallback=missing"


def test_iter_finding_candidates_orders_recent_and_legacy(profiles_root_override: Path) -> None:
    settings = DummySettings(base_date=date(2025, 10, 2))
    _, context_factory = create_context_factory(settings)
    context_result = context_factory(stage="p3")
    context = context_result.context

    write_partition_jsonl(profiles_root_override, context.profile, context.iso_date, "p3", [{}])
    write_partition_jsonl(profiles_root_override, context.profile, "2025-09-30", "p3", [{}])
    legacy_candidate = profiles_root_override / "legacy.jsonl"
    legacy_candidate.write_text("{}\n", encoding="utf-8")

    candidates = list(
        _iter_finding_candidates(
            context,
            days_back=2,
            stage="p3",
            legacy_candidates=[legacy_candidate],
        )
    )

    assert [ctx.iso_date for ctx, *_ in candidates] == [
        "2025-10-02",
        "2025-09-30",
        "2025-10-02",
    ]
    assert [reason for *_, reason in candidates] == [None, "fallback=recent", "fallback=legacy"]


def test_find_record_for_message_combines_fallback_reasons(profiles_root_override: Path) -> None:
    settings = DummySettings(base_date=date(2025, 10, 3))
    _, context_factory = create_context_factory(settings)
    base_result = context_factory(stage="p3")
    context = base_result.context

    write_partition_jsonl(
        profiles_root_override,
        context.profile,
        context.iso_date,
        "p3",
        [
            {
                "channel_id": "111",
                "message_id": "222",
                "rule_id": "LATEST",
            }
        ],
    )
    write_partition_jsonl(
        profiles_root_override,
        context.profile,
        "2025-10-01",
        "p3",
        [
            {
                "channel_id": "111",
                "message_id": "333",
                "rule_id": "RECENT",
            }
        ],
    )

    context_result = ContextResolveResult(
        context=context,
        paths=ContextPaths.for_context(context),
        fallback_reason="fallback=missing",
    )

    match = find_record_for_message(
        111,
        333,
        context_result=context_result,
        days_back=3,
        legacy_candidates=(),
    )

    assert match is not None
    assert match.record["rule_id"] == "RECENT"
    assert match.context.iso_date == "2025-10-01"
    assert match.fallback_reason == "fallback=missing, fallback=recent"


def test_load_findings_index_includes_thread_messages(profiles_root_override: Path) -> None:
    settings = DummySettings(base_date=date(2025, 10, 2))
    _, context_factory = create_context_factory(settings)
    context_result = context_factory(stage="p3")
    context = context_result.context

    write_partition_jsonl(
        profiles_root_override,
        context.profile,
        "2025-10-01",
        "p3",
        [
            {
                "channel_id": "123",
                "message_id": "456",
                "messages": [
                    {"channel_id": "123", "message_id": "789"},
                ],
            }
        ],
    )

    index = load_findings_index(
        _make_context_result(context),
        days_back=2,
        legacy_candidates=(),
    )

    assert (123, 456) in index
    assert (123, 789) in index
    assert index[(123, 456)].fallback_reason in (None, "fallback=recent")


def test_merge_and_format_helpers() -> None:
    context = ProfileContext(profile="demo", date=date(2025, 10, 1), tzinfo=ZoneInfo("UTC"))
    assert _merge_fallback_reason(None, "fallback=legacy", "fallback=legacy", "fallback=recent") == "fallback=legacy, fallback=recent"
    assert _format_context_notice(context, None) == "プロファイル: demo / 対象日: 2025-10-01"
    assert _format_context_notice(context, "fallback=legacy").endswith("fallback=legacy")


def test_ticket_context_result(monkeypatch, profiles_root_override: Path) -> None:
    settings = DummySettings(base_date=date(2025, 10, 1))
    _, context_factory = create_context_factory(settings)

    ticket = Ticket(
        ticket_id="1:2:3",
        guild_id=1,
        channel_id=2,
        message_id=3,
        author_id=4,
        severity="orange",
        rule_id="R-1",
        reason="reason",
        message_link="https://discord.com/channels/1/2/3",
        due_at=datetime.now(timezone.utc),
        status="notified",
        executor_id=5,
        profile="demo",
        partition_date="2025-10-01",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    resolved = _ticket_context_result(context_factory, ticket)
    assert resolved.context.iso_date == "2025-10-01"
    assert resolved.context.profile == "demo"


@pytest.fixture(autouse=True)
def _clear_profile_cache():
    yield
    clear_context_cache()
