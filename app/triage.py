from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from .rule_engine import RuleEngine


FINDINGS_PATH = Path("out/p3_findings.jsonl")
REPORT_PATH = Path("out/p3_report.csv")
ANALYSIS_PATH = Path("out/p2_analysis.jsonl")


@dataclass(slots=True)
class ScanSummary:
    total: int
    severity_counts: Counter
    output_path: Path

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
) -> ScanSummary:
    analysis_path = Path(analysis_path)
    findings_path = Path(findings_path)
    rules_path = Path(rules_path)

    engine = RuleEngine(str(rules_path))
    findings_path.parent.mkdir(parents=True, exist_ok=True)

    severity_counts: Counter = Counter()
    total = 0

    with analysis_path.open("r", encoding="utf-8") as src, findings_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            payload = json.loads(line)
            result = engine.evaluate(payload)
            payload["severity"] = result.severity
            payload["reasons"] = result.reasons
            json.dump(payload, dst, ensure_ascii=False)
            dst.write("\n")
            severity_counts[result.severity] += 1
            total += 1

    return ScanSummary(total=total, severity_counts=severity_counts, output_path=findings_path)


def generate_report(
    findings_path: Path | str = FINDINGS_PATH,
    report_path: Path | str = REPORT_PATH,
    severity: Optional[str] = None,
    rules_path: Path | str = Path("configs/rules.yaml"),
) -> ReportSummary:
    findings_path = Path(findings_path)
    report_path = Path(report_path)
    rules_path = Path(rules_path)

    engine = RuleEngine(str(rules_path))
    violence_tags = set(engine.config.violence_tags)

    allowed = {"red", "orange", "yellow", "green"}
    if severity is not None and severity != "all" and severity not in allowed:
        raise ValueError(f"Unsupported severity filter: {severity}")

    severity_filter = None if severity in (None, "all") else severity
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with findings_path.open("r", encoding="utf-8") as src, report_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=[
                "severity",
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
                "violence_tags",
                "reasons",
            ],
        )
        writer.writeheader()
        rows = 0
        for line in src:
            if not line.strip():
                continue
            payload = json.loads(line)
            sev = payload.get("severity", "green")
            if severity_filter and sev != severity_filter:
                continue
            message = payload.get("messages", [{}])
            primary = message[0] if message else {}
            rating = payload.get("wd14", {}).get("rating", {})
            general = payload.get("wd14", {}).get("general", [])
            formatted_tags = _format_top_tags(general)
            top_tags = ", ".join(formatted_tags)
            nudity = ", ".join(_format_top_detections(payload.get("nudity_detections", [])))
            violence = ", ".join(
                tag.split(":")[0]
                for tag in formatted_tags
                if tag.split(":")[0] in violence_tags
            )
            writer.writerow(
                {
                    "severity": sev,
                    "message_link": payload.get("message_link", ""),
                    "author_id": primary.get("author_id", ""),
                    "is_nsfw_channel": primary.get("is_nsfw_channel"),
                    "wd14_rating_general": rating.get("general", 0.0),
                    "wd14_rating_sensitive": rating.get("sensitive", 0.0),
                    "wd14_rating_questionable": rating.get("questionable", 0.0),
                    "wd14_rating_explicit": rating.get("explicit", 0.0),
                    "top_tags": top_tags,
                    "nudity_tops": nudity,
                    "exposure_score": payload.get("xsignals", {}).get("exposure_score", 0.0),
                    "placement_risk_pre": payload.get("xsignals", {}).get("placement_risk_pre", 0.0),
                    "violence_tags": violence,
                    "reasons": "; ".join(payload.get("reasons", [])),
                }
            )
            rows += 1

    return ReportSummary(rows=rows, path=report_path, severity_filter=severity_filter)


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
