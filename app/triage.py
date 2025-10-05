from __future__ import annotations

import asyncio
import csv
import json
import logging
import re
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Literal, Optional, Sequence

from .engine.group_utils import group_top_tags
from .engine.tag_norm import normalize_pair
from .engine.types import DslPolicy
from .io.stream import iter_jsonl
from .output_paths import default_analysis_path, default_findings_path, default_report_path
from .p3_stream import FindingsWriter, evaluate_stream, _build_contract_payload
from .rule_engine import EvaluationResult, RuleEngine
from .triage_attachments import P0Index
from .jsonl_merge import merge_jsonl_records


FINDINGS_PATH = default_findings_path()
REPORT_PATH = default_report_path()
ANALYSIS_PATH = default_analysis_path()

# CSV 列順の契約。順序変更は CLI/テストで検知する。
P3_CSV_HEADER: tuple[str, ...] = (
    "severity",
    "rule_id",
    "rule_title",
    "message_link",
    "author_id",
    "is_nsfw_channel",
    "wd14_rating_general",
    "wd14_rating_sensitive",
    "wd14_rating_questionable",
    "wd14_rating_explicit",
    "top_tags",
    "nudity_tops",
    "exposure_score",
    "placement_risk_pre",
    "nsfw_margin",
    "nsfw_ratio",
    "nsfw_general_sum",
    "violence_tags",
    "animals_sum",
    "reasons",
)


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScanSummary:
    total: int
    severity_counts: Counter
    output_path: Path
    records: List[dict]

    def format_message(self) -> str:
        parts = [f"total={self.total}"]
        for level in ("red", "orange", "yellow", "green"):
            count = self.severity_counts.get(level, 0)
            parts.append(f"{level}={count}")
        return ", ".join(parts)


@dataclass(slots=True)
class ReportSummary:
    rows: int
    path: Path
    severity_filter: Optional[str]


@dataclass(slots=True)
class AttachmentStats:
    index_rows: int = 0
    records_with_index: int = 0
    records_enriched: int = 0
    attachments_added: int = 0
    degraded_records: int = 0
    source_path: Optional[Path] = None


def run_scan(
    analysis_path: Path | str = ANALYSIS_PATH,
    findings_path: Path | str = FINDINGS_PATH,
    rules_path: Path | str = Path("configs/rules_v2.yaml"),
    channel_ids: Optional[Sequence[str]] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    severity_filter: Optional[Sequence[str]] = None,
    time_range: Optional[tuple[datetime, datetime]] = None,
    default_end_offset: Optional[timedelta] = None,
    *,
    dry_run: bool = False,
    metrics_path: Path | None = None,
    dsl_mode: str | None = None,
    policy: DslPolicy | None = None,
    trace_jsonl: Path | None = None,
    limit: int = 0,
    offset: int = 0,
    engine: RuleEngine | None = None,
    fallback: Literal["green", "skip"] | None = None,
    p0_path: Path | str | None = None,
    include_attachments: bool = True,
    drop_attachment_urls: bool = False,
    attachments_report_path: Path | None = None,
    merge_existing: bool = False,
) -> ScanSummary:
    """Evaluate analysis records and write findings adhering to the p3 contract.

    FindingsWriter で出力する各レコードには少なくとも
    `severity` / `rule_id` / `rule_title` / `reasons` / `metrics`
    を含める必要があり、`metrics` はオブジェクト（将来拡張可）として扱う。
    """
    analysis_path = Path(analysis_path)
    findings_path = Path(findings_path)
    rules_path = Path(rules_path)

    effective_policy = policy
    if engine is not None:
        effective_policy = engine.policy
    if effective_policy is None and dsl_mode:
        effective_policy = DslPolicy.from_mode(dsl_mode)
    if engine is None and fallback is None:
        engine = RuleEngine(str(rules_path), policy=effective_policy)
        effective_policy = engine.policy
    if effective_policy is None:
        effective_policy = DslPolicy()

    channel_set = set(channel_ids) if channel_ids else None
    severity_allowed = set(severity_filter) if severity_filter else None
    start_at, end_at = (
        time_range
        if time_range is not None
        else resolve_time_range(since, until, default_end_offset=default_end_offset)
    )

    analysis_iter = iter_jsonl(
        analysis_path,
        limit=limit,
        offset=offset,
        policy=effective_policy,
    )

    attachment_index: Optional[P0Index]
    attachment_stats: AttachmentStats = AttachmentStats()
    attachment_index = None
    if include_attachments:
        resolved_path = _resolve_p0_scan_path(p0_path)
        if resolved_path:
            try:
                attachment_index = P0Index.from_csv(resolved_path)
                attachment_stats.index_rows = attachment_index.total_rows
                attachment_stats.source_path = resolved_path
            except Exception as exc:  # noqa: BLE001
                logger.warning("failed to load p0 index for attachments: %s", exc)
        else:
            logger.info("p0 scan csv not found; skipping attachment enrichment")

    enriched_iter: Iterable[dict]
    if attachment_index is None:
        enriched_iter = analysis_iter
    else:
        enriched_iter = _enrich_records_with_attachments(
            analysis_iter,
            attachment_index,
            attachment_stats,
            drop_urls=drop_attachment_urls,
        )

    def record_filter(record: dict) -> bool:
        if channel_set and record.get("channel_id") not in channel_set:
            return False
        created_at = parse_created_at(record.get("created_at"))
        if created_at is not None and not (start_at <= created_at <= end_at):
            return False
        return True

    def result_filter(record: dict, result) -> bool:  # noqa: ANN202
        if severity_allowed and result.severity not in severity_allowed:
            return False
        return True

    collector: List[dict] = []
    metrics_path = Path(metrics_path) if metrics_path else None

    writer_context = nullcontext(None)
    if not dry_run and not merge_existing:
        writer_context = FindingsWriter(findings_path)

    with writer_context as writer:
        if fallback is None:
            report = evaluate_stream(
                engine,
                enriched_iter,
                writer=writer,
                dry_run=dry_run,
                metrics_path=metrics_path,
                record_filter=record_filter,
                result_filter=result_filter,
                collector=collector,
                trace_path=trace_jsonl,
            )
        else:
            report = _run_legacy_fallback(
                enriched_iter,
                writer if not dry_run else None,
                collector,
                record_filter,
                result_filter,
                metrics_path,
                fallback,
            )

    if merge_existing and not dry_run:
        updates = {}
        for payload in collector:
            key = payload.get("phash")
            if key is None:
                continue
            updates[str(key)] = payload
        merge_jsonl_records(
            base_path=findings_path,
            updates=updates,
            key_field="phash",
            out_path=findings_path,
        )

    if attachment_index is not None:
        logger.info(
            "attachments_joined source=%s index_rows=%d records_with_index=%d records_enriched=%d added=%d degraded=%d",
            attachment_stats.source_path,
            attachment_stats.index_rows,
            attachment_stats.records_with_index,
            attachment_stats.records_enriched,
            attachment_stats.attachments_added,
            attachment_stats.degraded_records,
        )

    if attachments_report_path and collector:
        _write_attachment_report(collector, attachments_report_path)

    return ScanSummary(
        total=report.total,
        severity_counts=report.severity_counts,
        output_path=findings_path,
        records=collector,
    )


@dataclass(slots=True)
class _FallbackReport:
    total: int
    severity_counts: Counter


def _run_legacy_fallback(
    records: Iterable[dict],
    writer: FindingsWriter | None,
    collector: list[dict],
    record_filter: RecordFilter | None,
    result_filter: ResultFilter | None,
    metrics_path: Path | None,
    fallback: Literal["green", "skip"],
) -> _FallbackReport:
    severity = Counter({"red": 0, "orange": 0, "yellow": 0, "green": 0})
    produced = 0
    processed = 0
    fallback_reason = "legacy_ruleset_unsupported"

    for record in records:
        if record_filter and not record_filter(record):
            continue
        processed += 1
        if fallback == "skip":
            continue

        result = EvaluationResult(
            severity="green",
            rule_id=None,
            rule_title=None,
            reasons=[fallback_reason],
            metrics={
                "winning": {
                    "origin": "legacy",
                    "severity": "green",
                    "rule_id": None,
                }
            },
        )
        if result_filter and not result_filter(record, result):
            continue

        payload: Optional[dict]
        if writer is not None:
            payload, _ = writer.write(record, result, eval_ms=0.0)
        else:
            payload = _build_contract_payload(record, result)
            payload.setdefault("metrics", {})["eval_ms"] = 0.0

        severity["green"] += 1
        produced += 1
        if payload is not None:
            collector.append(payload)

    if metrics_path:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_snapshot = {
            "total": produced,
            "processed": processed,
            "severity": dict(severity),
            "fallback": {
                "mode": fallback,
                "reason": fallback_reason,
            },
        }
        metrics_path.write_text(json.dumps(metrics_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.warning(
        "legacy_fallback applied mode=%s processed=%d produced=%d",
        fallback,
        processed,
        produced,
    )

    return _FallbackReport(total=produced, severity_counts=severity)


def _resolve_p0_scan_path(requested: Path | str | None) -> Optional[Path]:
    candidates: list[Path] = []
    if requested:
        candidates.append(Path(requested))
    else:
        candidates.extend([Path("out/p0_scan.csv"), Path("out/out/p0_scan.csv")])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _enrich_records_with_attachments(
    records: Iterable[dict],
    index: P0Index,
    stats: AttachmentStats,
    *,
    drop_urls: bool = False,
) -> Iterator[dict]:
    for record in records:
        phash = record.get("phash")
        attachments_by_msg = index.get(phash) if isinstance(phash, str) else {}
        if attachments_by_msg:
            stats.records_with_index += 1
            added, degraded = _apply_attachments_to_record(record, attachments_by_msg, drop_urls=drop_urls)
            stats.attachments_added += added
            if added:
                stats.records_enriched += 1
            if degraded:
                stats.degraded_records += 1
        yield record


def _apply_attachments_to_record(
    record: dict,
    attachments_by_msg: dict[str, tuple[dict[str, object], ...]],
    *,
    drop_urls: bool = False,
) -> tuple[int, bool]:
    messages_raw = record.get("messages")
    messages = messages_raw if isinstance(messages_raw, list) else []
    message_index: dict[str, dict] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        msg_id = message.get("message_id")
        if msg_id is None:
            continue
        message_index[str(msg_id)] = message

    added = 0
    unmatched: list[dict[str, object]] = []
    seen_keys: set[tuple[str, object]] = set()

    for msg_id, attachments in attachments_by_msg.items():
        target_message = message_index.get(msg_id) if msg_id else None
        if target_message is None and msg_id and str(msg_id) not in message_index:
            unmatched.extend(attachments)
            continue
        destination = target_message.setdefault("attachments", []) if target_message else None
        if destination is None:
            unmatched.extend(attachments)
            continue
        existing_ids = {
            item.get("id")
            for item in destination
            if isinstance(item, dict) and item.get("id") is not None
        }
        for attachment in attachments:
            key = (msg_id, attachment.get("id"))
            if key in seen_keys:
                continue
            copy = dict(attachment)
            if drop_urls:
                copy["url"] = None
            if copy.get("id") in existing_ids:
                continue
            destination.append(copy)
            seen_keys.add(key)
            if copy.get("id") is not None:
                existing_ids.add(copy["id"])
            added += 1

    degraded = False
    if unmatched:
        degraded = True
        fallback_message: Optional[dict]
        if messages:
            fallback_message = messages[0]
            fallback_attachments = fallback_message.setdefault("attachments", [])
            fallback_existing = {
                item.get("id")
                for item in fallback_attachments
                if isinstance(item, dict) and item.get("id") is not None
            }
            for attachment in unmatched:
                key = ("", attachment.get("id"))
                if key in seen_keys:
                    continue
                copy = dict(attachment)
                if drop_urls:
                    copy["url"] = None
                if copy.get("id") in fallback_existing:
                    continue
                fallback_attachments.append(copy)
                seen_keys.add(key)
                if copy.get("id") is not None:
                    fallback_existing.add(copy["id"])
                added += 1
        else:
            fallback_message = None
    return added, degraded


def _write_attachment_report(records: Sequence[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phash",
        "severity",
        "rule_id",
        "message_link",
        "attachments_count",
        "first_attachment_id",
        "first_attachment_filename",
        "first_attachment_content_type",
        "first_attachment_url",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            messages = record.get("messages") or []
            attachments: list[dict] = []
            for message in messages:
                if isinstance(message, dict):
                    attachments.extend(message.get("attachments") or [])
            if not attachments:
                continue
            first = attachments[0]
            writer.writerow(
                {
                    "phash": record.get("phash"),
                    "severity": record.get("severity"),
                    "rule_id": record.get("rule_id"),
                    "message_link": record.get("message_link"),
                    "attachments_count": len(attachments),
                    "first_attachment_id": first.get("id"),
                    "first_attachment_filename": first.get("filename"),
                    "first_attachment_content_type": first.get("content_type"),
                    "first_attachment_url": first.get("url"),
                }
            )

def generate_report(
    findings_path: Path | str = FINDINGS_PATH,
    report_path: Path | str = REPORT_PATH,
    severity: Optional[str] = None,
    rules_path: Path | str = Path("configs/rules_v2.yaml"),
    channel_ids: Optional[Sequence[str]] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    time_range: Optional[tuple[datetime, datetime]] = None,
    default_end_offset: Optional[timedelta] = None,
) -> ReportSummary:
    report_path = Path(report_path)
    rules_path = Path(rules_path)

    engine = RuleEngine(str(rules_path))
    violence_patterns = engine.groups.get("violence", ())

    allowed = {"red", "orange", "yellow", "green"}
    severity_filter: Optional[List[str]]
    if severity is None or severity == "all":
        severity_filter = None
    else:
        severity_filter = [token.strip() for token in severity.split(",") if token.strip()]
        invalid = [token for token in severity_filter if token not in allowed]
        if invalid:
            raise ValueError(f"Unsupported severity filter: {', '.join(invalid)}")
    records = load_findings(
        findings_path,
        channel_ids=channel_ids,
        since=since,
        until=until,
        severity=severity_filter,
        time_range=time_range,
        default_end_offset=default_end_offset,
    )

    rows = write_report_csv(records, report_path, violence_patterns)

    selected = ",".join(severity_filter) if severity_filter else None
    return ReportSummary(rows=rows, path=report_path, severity_filter=selected)


def iter_findings(
    findings_path: Path | str = FINDINGS_PATH,
    channel_ids: Optional[Sequence[str]] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    severity: Optional[Sequence[str]] = None,
    time_range: Optional[tuple[datetime, datetime]] = None,
    default_end_offset: Optional[timedelta] = None,
    limit: Optional[int] = None,
) -> Iterator[dict]:
    findings_path = Path(findings_path)
    channel_set = set(channel_ids) if channel_ids else None
    severity_set = set(severity) if severity else None
    start_at, end_at = (
        time_range
        if time_range is not None
        else resolve_time_range(since, until, default_end_offset=default_end_offset)
    )

    yielded = 0
    with findings_path.open("r", encoding="utf-8") as src:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if severity_set and payload.get("severity") not in severity_set:
                continue
            if channel_set and payload.get("channel_id") not in channel_set:
                continue
            created_at = parse_created_at(payload.get("created_at"))
            if created_at is not None and not (start_at <= created_at <= end_at):
                continue
            yield payload
            if limit is not None:
                yielded += 1
                if yielded >= limit:
                    break


def load_findings(
    findings_path: Path | str = FINDINGS_PATH,
    channel_ids: Optional[Sequence[str]] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    severity: Optional[Sequence[str]] = None,
    time_range: Optional[tuple[datetime, datetime]] = None,
    default_end_offset: Optional[timedelta] = None,
    limit: Optional[int] = None,
) -> List[dict]:
    return list(
        iter_findings(
            findings_path,
            channel_ids=channel_ids,
            since=since,
            until=until,
            severity=severity,
            time_range=time_range,
            default_end_offset=default_end_offset,
            limit=limit,
        )
    )


async def load_findings_async(
    findings_path: Path | str = FINDINGS_PATH,
    channel_ids: Optional[Sequence[str]] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    severity: Optional[Sequence[str]] = None,
    time_range: Optional[tuple[datetime, datetime]] = None,
    default_end_offset: Optional[timedelta] = None,
    limit: Optional[int] = None,
) -> List[dict]:
    args = dict(
        findings_path=findings_path,
        channel_ids=channel_ids,
        since=since,
        until=until,
        severity=severity,
        time_range=time_range,
        default_end_offset=default_end_offset,
        limit=limit,
    )
    return await asyncio.to_thread(lambda: load_findings(**args))


def write_report_csv(
    records: Sequence[dict],
    report_path: Path,
    violence_patterns: Sequence[str],
) -> int:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=list(P3_CSV_HEADER), lineterminator="\n")
        assert list(writer.fieldnames or []) == list(P3_CSV_HEADER)
        writer.writeheader()
        rows = 0
        for payload in records:
            sev = payload.get("severity", "green")
            message = payload.get("messages", [{}])
            primary = message[0] if message else {}
            rating = payload.get("wd14", {}).get("rating", {})
            general = payload.get("wd14", {}).get("general", [])
            formatted_tags = _format_top_tags(general)
            top_tags = ", ".join(formatted_tags)
            nudity = ", ".join(_format_top_detections(payload.get("nudity_detections", [])))
            tag_scores = _build_tag_scores(general)
            violence_hits = group_top_tags(tag_scores, violence_patterns, k=5, min_score=0.1)
            violence = ", ".join(name for name, _ in violence_hits)
            metrics = payload.get("metrics", {})
            writer.writerow(
                {
                    "severity": sev,
                    "rule_id": payload.get("rule_id", ""),
                    "rule_title": payload.get("rule_title", ""),
                    "message_link": payload.get("message_link", ""),
                    "author_id": primary.get("author_id", ""),
                    "is_nsfw_channel": primary.get("is_nsfw_channel"),
                    "wd14_rating_general": rating.get("general", 0.0),
                    "wd14_rating_sensitive": rating.get("sensitive", 0.0),
                    "wd14_rating_questionable": rating.get("questionable", 0.0),
                    "wd14_rating_explicit": rating.get("explicit", 0.0),
                    "top_tags": top_tags,
                    "nudity_tops": nudity,
                    "exposure_score": metrics.get("exposure_peak", payload.get("xsignals", {}).get("exposure_score", 0.0)),
                    "placement_risk_pre": metrics.get("placement_risk", payload.get("xsignals", {}).get("placement_risk_pre", 0.0)),
                    "nsfw_margin": metrics.get("nsfw_margin", 0.0),
                    "nsfw_ratio": metrics.get("nsfw_ratio", 0.0),
                    "nsfw_general_sum": metrics.get("nsfw_general_sum", 0.0),
                    "violence_tags": violence,
                    "animals_sum": metrics.get("animals_sum", 0.0),
                    "reasons": "; ".join(payload.get("reasons", [])),
                }
            )
            rows += 1

    return rows


def _build_tag_scores(general: Iterable) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for item in general:
        pair = normalize_pair(item)
        if pair is None:
            continue
        tag, score = pair
        current = scores.get(tag)
        if current is None or score > current:
            scores[tag] = score
    return scores


def _format_top_tags(general: Iterable) -> List[str]:
    pairs: List[tuple[str, float]] = []
    for item in general:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append((item[0], float(item[1])))
        elif isinstance(item, dict):
            name = item.get("name")
            score = item.get("score")
            if name is not None and score is not None:
                pairs.append((name, float(score)))
    pairs.sort(key=lambda t: t[1], reverse=True)
    return [f"{name}:{score:.2f}" for name, score in pairs[:8]]


def _format_top_detections(detections: Iterable[dict]) -> List[str]:
    formatted: List[str] = []
    for det in detections:
        cls = det.get("class", "")
        score = det.get("score", 0.0)
        if cls:
            formatted.append(f"{cls}:{float(score):.2f}")
    return formatted[:8]


RELATIVE_PATTERN = re.compile(
    r"^(?:(?P<years>\d+)y)?(?:(?P<weeks>\d+)w)?(?:(?P<days>\d+)d)?"
    r"(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?:(?P<seconds>\d+)s)?$"
)


def resolve_time_range(
    since: Optional[str],
    until: Optional[str],
    default_days: int = 7,
    default_end_offset: Optional[timedelta] = None,
) -> tuple[datetime, datetime]:
    now = datetime.now(timezone.utc)
    if until:
        end = parse_time_argument(until, now)
    else:
        end = now + (default_end_offset or timedelta())
    start = parse_time_argument(since, now) if since else end - timedelta(days=default_days)
    if start > end:
        start, end = end, start
    return start, end


def parse_time_argument(value: Optional[str], now: datetime) -> datetime:
    if value is None:
        return now
    text = value.strip()
    if not text:
        return now
    lowered = text.lower()
    if lowered in {"now", "today"}:
        return now
    match = RELATIVE_PATTERN.fullmatch(lowered)
    if match and any(match.groupdict().values()):
        years = int(match.group("years") or 0)
        weeks = int(match.group("weeks") or 0)
        days = int(match.group("days") or 0)
        hours = int(match.group("hours") or 0)
        minutes = int(match.group("minutes") or 0)
        seconds = int(match.group("seconds") or 0)
        total_days = years * 365 + weeks * 7 + days
        delta = timedelta(days=total_days, hours=hours, minutes=minutes, seconds=seconds)
        return now - delta
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:  # noqa: BLE001
        raise ValueError(f"Invalid datetime format: {value}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def parse_created_at(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt
