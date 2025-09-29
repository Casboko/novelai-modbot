from __future__ import annotations

import asyncio
import csv
import json
import re
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from .engine.group_utils import group_top_tags
from .engine.tag_norm import normalize_pair
from .engine.types import DslPolicy
from .io.stream import iter_jsonl
from .p3_stream import FindingsWriter, evaluate_stream
from .rule_engine import RuleEngine


FINDINGS_PATH = Path("out/p3_findings.jsonl")
REPORT_PATH = Path("out/p3_report.csv")
ANALYSIS_PATH = Path("out/p2_analysis.jsonl")

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


def run_scan(
    analysis_path: Path | str = ANALYSIS_PATH,
    findings_path: Path | str = FINDINGS_PATH,
    rules_path: Path | str = Path("configs/rules.yaml"),
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
    limit: int = 0,
    offset: int = 0,
    engine: RuleEngine | None = None,
) -> ScanSummary:
    """Evaluate analysis records and write findings adhering to the p3 contract.

    FindingsWriter で出力する各レコードには少なくとも
    `severity` / `rule_id` / `rule_title` / `reasons` / `metrics`
    を含める必要があり、`metrics` はオブジェクト（将来拡張可）として扱う。
    """
    analysis_path = Path(analysis_path)
    findings_path = Path(findings_path)
    rules_path = Path(rules_path)

    policy = DslPolicy.from_mode(dsl_mode) if dsl_mode else None
    engine = engine or RuleEngine(str(rules_path), policy=policy)

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
        policy=engine.policy,
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

    context = FindingsWriter(findings_path) if not dry_run else nullcontext(None)
    with context as writer:
        report = evaluate_stream(
            engine,
            analysis_iter,
            writer=writer,
            dry_run=dry_run,
            metrics_path=metrics_path,
            record_filter=record_filter,
            result_filter=result_filter,
            collector=collector,
        )

    return ScanSummary(
        total=report.total,
        severity_counts=report.severity_counts,
        output_path=findings_path,
        records=collector,
    )


def generate_report(
    findings_path: Path | str = FINDINGS_PATH,
    report_path: Path | str = REPORT_PATH,
    severity: Optional[str] = None,
    rules_path: Path | str = Path("configs/rules.yaml"),
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
