from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml


@dataclass(slots=True)
class RuleConfig:
    wd14_repo: str
    thresholds: Dict[str, float]
    minor_tags: List[str]
    violence_tags: List[str]
    nsfw_tags: List[str]
    xsignal_weights: Dict[str, float]


@dataclass(slots=True)
class EvaluationResult:
    severity: str
    reasons: List[str]


DEFAULT_SEVERITY = "green"


class RuleEngine:
    def __init__(self, config_path: str | None = None) -> None:
        path = config_path or "configs/rules.yaml"
        self.config = self._load_config(path)

    @staticmethod
    def _load_config(path: str) -> RuleConfig:
        with open(path, "r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)
        return RuleConfig(
            wd14_repo=data.get("models", {}).get("wd14_repo", ""),
            thresholds=data.get("thresholds", {}),
            minor_tags=data.get("minor_tags", []),
            violence_tags=data.get("violence_tags", []),
            nsfw_tags=data.get("nsfw_general_tags", []),
            xsignal_weights=data.get("xsignals_weights", {}),
        )

    def evaluate(self, analysis: Mapping[str, Any]) -> EvaluationResult:
        reasons: List[str] = []

        wd14 = analysis.get("wd14", {})
        rating = wd14.get("rating", {})
        general_tags = wd14.get("general", [])
        xsignals = analysis.get("xsignals", {})
        nudity = analysis.get("nudity_detections", [])
        is_nsfw_channel = analysis.get("is_nsfw_channel")

        thresholds = self.config.thresholds

        # RED: Violence/Gore
        gore_scores = [
            float(score)
            for tag, score in self._iter_general_tags(general_tags)
            if tag in self.config.violence_tags
        ]
        if gore_scores:
            max_gore = max(gore_scores)
            sum_gore = sum(gore_scores)
            if max_gore >= thresholds.get("gore_any", 0.3):
                reasons.append(f"gore_any={max_gore:.2f}")
            if sum_gore >= thresholds.get("gore_sum", 0.4):
                reasons.append(f"gore_sum={sum_gore:.2f}")
            if reasons:
                return EvaluationResult("red", reasons)

        # RED: minors sum
        minor_scores = [
            float(score)
            for tag, score in self._iter_general_tags(general_tags)
            if tag in self.config.minor_tags
        ]
        if minor_scores:
            minors_sum = sum(minor_scores)
            if minors_sum >= thresholds.get("wd14_minors_sum", 0.4):
                reasons.append(f"minors_sum={minors_sum:.2f}")
                return EvaluationResult("red", reasons)

        # ORANGE: adult content outside NSFW channels
        questionable = float(rating.get("questionable", 0.0))
        explicit = float(rating.get("explicit", 0.0))
        exposure_score = float(xsignals.get("exposure_score", 0.0))
        placement_risk = float(xsignals.get("placement_risk_pre", 0.0))

        exposure_threshold = thresholds.get("exposure_strong", 0.6)
        placement_threshold = thresholds.get("exposure_risk", 0.6)

        adult_flag = (
            questionable >= thresholds.get("wd14_questionable", 0.35)
            or explicit >= thresholds.get("wd14_explicit", 0.2)
            or exposure_score >= exposure_threshold
        )

        if adult_flag and (is_nsfw_channel is False):
            reasons.append(
                f"adult_content=questionable:{questionable:.2f},explicit:{explicit:.2f},exposure:{exposure_score:.2f}"
            )
            return EvaluationResult("orange", reasons)

        # YELLOW: weak minors hint + exposure
        if minor_scores:
            minors_peak = max(minor_scores)
            if minors_peak >= thresholds.get("yellow_minor_hint", 0.2) or exposure_score > 0:
                reasons.append(f"minor_hint={minors_peak:.2f}")
                if exposure_score > 0:
                    reasons.append(f"exposure={exposure_score:.2f}")
                return EvaluationResult("yellow", reasons)

        if placement_risk >= placement_threshold and (is_nsfw_channel is False):
            reasons.append(f"placement_risk={placement_risk:.2f}")
            return EvaluationResult("yellow", reasons)

        return EvaluationResult(DEFAULT_SEVERITY, reasons)

    @staticmethod
    def _iter_general_tags(general_tags: Iterable) -> Iterable[tuple[str, float]]:
        for item in general_tags:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                yield item[0], float(item[1])
            elif isinstance(item, Mapping):
                name = item.get("name")
                score = item.get("score")
                if name is not None and score is not None:
                    yield name, float(score)


