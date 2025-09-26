from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

DEFAULT_SEVERITY = "green"
KEEP_UNDERSCORE = {"0_0", "(o)_(o)"}


def _normalize_tag(name: str) -> str:
    value = str(name)
    if value in KEEP_UNDERSCORE:
        return value
    return value.replace("_", " ")


@dataclass(slots=True)
class RuleConfig:
    wd14_repo: str
    thresholds: Dict[str, float]
    minor_tags: List[str]
    violence_tags: List[str]
    nsfw_tags: List[str]
    animal_tags: List[str]
    xsignal_weights: Dict[str, float]
    rule_titles: Dict[str, str]


@dataclass(slots=True)
class EvaluationResult:
    severity: str
    rule_id: Optional[str]
    rule_title: Optional[str]
    reasons: List[str]
    metrics: Dict[str, float]


class RuleEngine:
    def __init__(self, config_path: str | None = None) -> None:
        path = config_path or "configs/rules.yaml"
        self.config = self._load_config(path)

    @staticmethod
    def _load_config(path: str) -> RuleConfig:
        with open(path, "r", encoding="utf-8") as fp:
            data = yaml.safe_load(fp)
        def normalize_list(items: Iterable[str]) -> List[str]:
            return [_normalize_tag(item) for item in items if isinstance(item, str)]
        return RuleConfig(
            wd14_repo=data.get("models", {}).get("wd14_repo", ""),
            thresholds=data.get("thresholds", {}),
            minor_tags=normalize_list(data.get("minor_tags", [])),
            violence_tags=normalize_list(data.get("violence_tags", [])),
            nsfw_tags=normalize_list(data.get("nsfw_general_tags", [])),
            animal_tags=normalize_list(data.get("animal_abuse_tags", [])),
            xsignal_weights=data.get("xsignals_weights", {}),
            rule_titles=data.get("rule_titles", {}),
        )

    def evaluate(self, analysis: Mapping[str, Any]) -> EvaluationResult:
        wd14 = analysis.get("wd14", {})
        rating = wd14.get("rating", {})
        general_tags = wd14.get("general", [])
        general_map = {tag: score for tag, score in self._iter_general_tags(general_tags)}
        xsignals = analysis.get("xsignals", {})
        nudity = analysis.get("nudity_detections", [])
        is_nsfw_channel = bool(analysis.get("is_nsfw_channel"))

        thresholds = self.config.thresholds

        questionable = float(rating.get("questionable", 0.0))
        explicit = float(rating.get("explicit", 0.0))
        general_rating = float(rating.get("general", 0.0))
        sensitive_rating = float(rating.get("sensitive", 0.0))
        exposure_score = float(xsignals.get("exposure_score", 0.0))
        placement_risk = float(xsignals.get("placement_risk_pre", 0.0))

        nsfw_general_sum = sum(float(general_map.get(tag, 0.0)) for tag in self.config.nsfw_tags)
        nsfw_margin = max(questionable, explicit) - max(general_rating, sensitive_rating)
        rating_total = questionable + explicit + general_rating + sensitive_rating + 1e-6
        nsfw_ratio = (questionable + explicit) / rating_total

        violence_scores = [float(general_map.get(tag, 0.0)) for tag in self.config.violence_tags]
        violence_max = max(violence_scores) if violence_scores else 0.0
        violence_sum = sum(violence_scores)

        minor_scores = [float(general_map.get(tag, 0.0)) for tag in self.config.minor_tags]
        minors_sum = sum(minor_scores)
        minors_peak = max(minor_scores) if minor_scores else 0.0

        animal_scores = [float(general_map.get(tag, 0.0)) for tag in self.config.animal_tags]
        animals_sum = sum(animal_scores)
        animals_max = max(animal_scores) if animal_scores else 0.0

        exposure_det = self._max_exposed_detection(nudity)
        exposure_peak = max(exposure_score, exposure_det)

        metrics = {
            "questionable": questionable,
            "explicit": explicit,
            "general_rating": general_rating,
            "sensitive_rating": sensitive_rating,
            "nsfw_margin": nsfw_margin,
            "nsfw_ratio": nsfw_ratio,
            "nsfw_general_sum": nsfw_general_sum,
            "exposure_score": exposure_score,
            "exposure_detection": exposure_det,
            "exposure_peak": exposure_peak,
            "placement_risk": placement_risk,
            "violence_max": violence_max,
            "violence_sum": violence_sum,
            "minors_sum": minors_sum,
            "minors_peak": minors_peak,
            "animals_sum": animals_sum,
            "animals_max": animals_max,
        }

        # RED: Violence/Gore
        if violence_max >= thresholds.get("gore_any", 0.3) or violence_sum >= thresholds.get("gore_sum", 0.4):
            reasons = []
            if violence_max >= thresholds.get("gore_any", 0.3):
                reasons.append(f"gore_any={violence_max:.2f}")
            if violence_sum >= thresholds.get("gore_sum", 0.4):
                reasons.append(f"gore_sum={violence_sum:.2f}")
            return self._result("red", "RED-201", reasons, metrics)

        # RED: Minors
        if minors_sum >= thresholds.get("wd14_minors_sum", 0.4):
            reasons = [f"minors_sum={minors_sum:.2f}"]
            return self._result("red", "RED-301", reasons, metrics)

        # ORANGE: Adult content outside NSFW
        exposure_strong = thresholds.get("exposure_strong", 0.6)
        exposure_mid = thresholds.get("exposure_mid", 0.3)
        nsfw_margin_min = thresholds.get("nsfw_margin_min", 0.0)
        nsfw_ratio_min = thresholds.get("nsfw_ratio_min", 0.45)
        nsfw_general_min = thresholds.get("nsfw_general_sum", 0.2)

        adult_core = questionable >= thresholds.get("wd14_questionable", 0.35)
        adult_balance = nsfw_margin >= nsfw_margin_min or nsfw_ratio >= nsfw_ratio_min
        adult_context = nsfw_general_sum >= nsfw_general_min or exposure_peak >= exposure_mid
        strong_exposure = exposure_peak >= exposure_strong

        if not is_nsfw_channel and ((adult_core and adult_balance and adult_context) or strong_exposure):
            reasons = [
                f"q={questionable:.2f}",
                f"margin={nsfw_margin:.2f}",
                f"ratio={nsfw_ratio:.2f}",
                f"nsfw_sum={nsfw_general_sum:.2f}",
                f"exposure={exposure_peak:.2f}",
            ]
            return self._result("orange", "ORANGE-101", reasons, metrics)

        # YELLOW: Minor hint or moderate exposure
        if minors_peak >= thresholds.get("yellow_minor_hint", 0.2):
            reasons = [f"minor_hint={minors_peak:.2f}"]
            if exposure_peak > 0:
                reasons.append(f"exposure={exposure_peak:.2f}")
            return self._result("yellow", "YELLOW-201", reasons, metrics)

        if not is_nsfw_channel and (exposure_peak >= exposure_mid or placement_risk >= thresholds.get("exposure_risk", 0.6)):
            reasons = [f"exposure={exposure_peak:.2f}", f"placement={placement_risk:.2f}"]
            return self._result("yellow", "YELLOW-101", reasons, metrics)

        return self._result(DEFAULT_SEVERITY, None, [], metrics)

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

    @staticmethod
    def _max_exposed_detection(nudity: Iterable[Mapping[str, Any]]) -> float:
        """Expose variants include FEMALE_BREAST_EXPOSED / EXPOSED_BREAST_F etc."""
        max_score = 0.0
        for det in nudity or []:
            label = str(det.get("class", "") or "").upper()
            if "EXPOSED" in label and "COVERED" not in label:
                score = float(det.get("score", 0.0))
                if score > max_score:
                    max_score = score
        return max_score

    def _result(
        self,
        severity: str,
        rule_id: Optional[str],
        reasons: List[str],
        metrics: Dict[str, float],
    ) -> EvaluationResult:
        title = self.config.rule_titles.get(rule_id) if rule_id else None
        return EvaluationResult(
            severity=severity,
            rule_id=rule_id,
            rule_title=title,
            reasons=reasons,
            metrics=metrics,
        )
