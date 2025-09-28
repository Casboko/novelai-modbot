from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .engine.dsl import DslEvaluationInput, DslEvaluationOutcome, DslProgram, SEVERITY_ORDER
from .engine.loader import load_rule_config
from .engine.tag_norm import normalize_tag as dsl_normalize_tag
from .engine.types import DslPolicy

DEFAULT_SEVERITY = "green"
DEFAULT_RULES_PATH = "configs/rules.yaml"
DEFAULT_NUDENET_CONFIG_PATH = "configs/nudenet.yaml"
KEEP_UNDERSCORE = {"0_0", "(o)_(o)"}

# ---------------------------------------------------------------------------
# 静的サブ集合（全て小文字、スペース区切り）。
# WD14 の一般タグは `_normalize_tag` の後に lower() してマッチさせる。
# ---------------------------------------------------------------------------
SEX_MODIFIER_TAGS = {
    "cameltoe",
    "collar",
    "leash",
    "lactation",
    "nipples",
    "pregnant",
    "pubic hair",
    "slave",
    "spanking",
}

ANIMAL_ABUSE_CONTEXT_TAGS = {
    "animal abuse",
    "animal cruelty",
    "animal harm",
    "animal torture",
    "animal blood",
    "animal gore",
    "animal corpse",
    "animal trap",
    "animal fight",
    "animal starvation",
    "animal neglect",
    "feral abuse",
    "yiff",
    "bestiality",
    "zoophilia",
    "zoosadism",
    "furry bestiality",
}

GORE_TAGS = {
    "blood",
    "blood on face",
    "blood on arm",
    "blood stain",
    "gore",
    "guts",
    "injury",
    "scar",
    "scar on face",
    "scar on arm",
    "scar across eye",
    "wound",
    "bruise",
    "corpse",
}

DISMEMBER_TAGS = {
    "amputation",
    "amputee",
    "dismember",
    "dismemberment",
    "decapitation",
    "severed arm",
    "severed leg",
    "severed head",
}

DRUG_KEYWORDS = (
    "drug",
    "drugs",
    "drugged",
    "pill",
    "syringe",
    "injection",
    "marijuana",
    "cannabis",
    "weed",
    "cocaine",
    "heroin",
    "meth",
    "lsd",
    "addiction",
    "overdose",
)

MILD_EXPOSURE_DEFAULT_PATTERNS = (
    "ARMPIT",
    "BELLY",
    "MIDRIFF",
    "STOMACH",
    "ABDOMEN",
)
MILD_EXPOSURE_DEFAULT_THRESHOLD = 0.30

THREAT_IGNORED = True  # Violence threat 系タグは今回参照しない（誤検知抑止）


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
class PreparedContext:
    rating: Dict[str, float]
    tag_scores: Dict[str, float]
    nude_flags: Tuple[str, ...]
    is_nsfw_channel: bool
    is_spoiler: bool
    attachment_count: int
    metrics: Dict[str, Any]


@dataclass(slots=True)
class EvaluationResult:
    severity: str
    rule_id: Optional[str]
    rule_title: Optional[str]
    reasons: List[str]
    metrics: Dict[str, Any]


class RuleEngine:
    def __init__(
        self,
        config_path: str | None = None,
        *,
        nudenet_config_path: str | None = None,
    ) -> None:
        rules_path = config_path or DEFAULT_RULES_PATH
        dsl_config, raw_config, policy = load_rule_config(rules_path)
        self._policy: DslPolicy = policy
        self.raw_config = raw_config
        self.config = self._build_legacy_config(raw_config)
        self._minor_tags = [tag.lower() for tag in self.config.minor_tags]
        self._violence_tags = [tag.lower() for tag in self.config.violence_tags]
        self._nsfw_tags = [tag.lower() for tag in self.config.nsfw_tags]
        # animal_tags から文脈語と主語語を分離
        context_tags: set[str] = set()
        subject_tags: set[str] = set()
        for tag in self.config.animal_tags:
            lowered = tag.lower()
            if lowered in ANIMAL_ABUSE_CONTEXT_TAGS:
                context_tags.add(lowered)
            else:
                subject_tags.add(lowered)
        self._animal_context_tags = sorted(context_tags)
        self._animal_subject_tags = sorted(subject_tags)

        # NudeNet の軽微露出パターンのロード
        self._mild_patterns, self._mild_threshold = self._load_mild_exposure_settings(
            nudenet_config_path or DEFAULT_NUDENET_CONFIG_PATH
        )
        self._dsl_program: DslProgram | None = None
        if dsl_config is not None:
            self._dsl_program = DslProgram.from_config(dsl_config, policy)

    # ------------------------------------------------------------------
    # 設定ロード
    # ------------------------------------------------------------------
    @staticmethod
    def _build_legacy_config(data: Mapping[str, Any]) -> RuleConfig:
        def normalize_list(items: Any) -> List[str]:
            if isinstance(items, (list, tuple, set)):
                return [_normalize_tag(item) for item in items if isinstance(item, str)]
            return []

        models = data.get("models") if isinstance(data.get("models"), Mapping) else {}
        thresholds = data.get("thresholds") if isinstance(data.get("thresholds"), Mapping) else {}
        xsignals_weights = data.get("xsignals_weights") if isinstance(data.get("xsignals_weights"), Mapping) else {}
        rule_titles = data.get("rule_titles") if isinstance(data.get("rule_titles"), Mapping) else {}

        return RuleConfig(
            wd14_repo=str(models.get("wd14_repo", "")) if isinstance(models, Mapping) else "",
            thresholds=dict(thresholds),
            minor_tags=normalize_list(data.get("minor_tags")),
            violence_tags=normalize_list(data.get("violence_tags")),
            nsfw_tags=normalize_list(data.get("nsfw_general_tags")),
            animal_tags=normalize_list(data.get("animal_abuse_tags")),
            xsignal_weights=dict(xsignals_weights),
            rule_titles={str(k): str(v) for k, v in rule_titles.items()},
        )

    @staticmethod
    def _load_mild_exposure_settings(path: str) -> tuple[tuple[str, ...], float]:
        config_path = Path(path)
        if not config_path.is_file():
            return MILD_EXPOSURE_DEFAULT_PATTERNS, MILD_EXPOSURE_DEFAULT_THRESHOLD
        try:
            with config_path.open("r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
        except Exception:  # noqa: BLE001 - 設定読み込み失敗時はデフォルトへフォールバック
            return MILD_EXPOSURE_DEFAULT_PATTERNS, MILD_EXPOSURE_DEFAULT_THRESHOLD
        patterns = data.get("mild_exposure_label_patterns")
        threshold = data.get("mild_exposure_threshold", MILD_EXPOSURE_DEFAULT_THRESHOLD)
        if isinstance(patterns, Sequence) and patterns:
            normalized = tuple(str(item).upper() for item in patterns if item)
            return normalized, float(threshold)
        return MILD_EXPOSURE_DEFAULT_PATTERNS, float(threshold)

    # ------------------------------------------------------------------
    # メイン評価
    # ------------------------------------------------------------------
    def evaluate(self, analysis: Mapping[str, Any]) -> EvaluationResult:
        legacy_result, context = self._evaluate_legacy(analysis)
        if not self._dsl_program:
            return legacy_result
        dsl_outcome = self._evaluate_dsl(context)
        if not dsl_outcome:
            return legacy_result
        return self._merge_results(legacy_result, dsl_outcome)

    def _evaluate_legacy(self, analysis: Mapping[str, Any]) -> tuple[EvaluationResult, PreparedContext]:
        wd14 = analysis.get("wd14") or {}
        rating = wd14.get("rating") or {}
        general_tags = wd14.get("general") or []
        general_entries = list(self._iter_general_tags(general_tags))
        general_map = {
            _normalize_tag(tag).lower(): score for tag, score in general_entries
        }
        normalized_tag_scores = {}
        for tag, score in general_entries:
            normalized = dsl_normalize_tag(tag)
            if not normalized:
                continue
            normalized_tag_scores[normalized] = float(score)
        xsignals = analysis.get("xsignals", {})
        nudity = analysis.get("nudity_detections", [])
        is_nsfw_channel = bool(analysis.get("is_nsfw_channel"))

        questionable = float(rating.get("questionable", 0.0))
        explicit = float(rating.get("explicit", 0.0))
        general_rating = float(rating.get("general", 0.0))
        sensitive_rating = float(rating.get("sensitive", 0.0))

        exposure_score = float(xsignals.get("exposure_score", 0.0))
        placement_risk = float(xsignals.get("placement_risk_pre", 0.0))
        nsfw_margin = float(xsignals.get("nsfw_margin", max(questionable, explicit) - max(general_rating, sensitive_rating)))
        nsfw_ratio = float(xsignals.get("nsfw_ratio", 0.0))
        nsfw_general_sum = float(xsignals.get("nsfw_general_sum", 0.0))
        if not nsfw_general_sum:
            nsfw_general_sum = self._sum_by_tags(general_map, self._nsfw_tags)

        exposure_detection = self._max_exposed_detection(nudity)
        exposure_peak = max(exposure_score, exposure_detection)
        mild_exposure_peak = self._mild_exposure_peak(nudity)

        minor_scores = [float(general_map.get(tag, 0.0)) for tag in self._minor_tags]
        minors_sum = sum(minor_scores)
        minors_peak = max(minor_scores) if minor_scores else 0.0

        violence_scores = [float(general_map.get(tag, 0.0)) for tag in self._violence_tags]
        violence_sum = sum(violence_scores)
        violence_max = max(violence_scores) if violence_scores else 0.0

        animal_subject_scores = [float(general_map.get(tag, 0.0)) for tag in self._animal_subject_tags]
        animal_context_scores = [float(general_map.get(tag, 0.0)) for tag in self._animal_context_tags]
        animal_peak = max(animal_subject_scores) if animal_subject_scores else 0.0
        animal_context_peak = max(animal_context_scores) if animal_context_scores else 0.0

        sexual_explicit_sum = 0.0
        sexual_modifier_sum = 0.0
        sexual_modifier_hits: List[tuple[str, float]] = []
        for tag in self._nsfw_tags:
            score = float(general_map.get(tag, 0.0))
            if score <= 0.0:
                continue
            if tag in SEX_MODIFIER_TAGS:
                sexual_modifier_sum += score
                sexual_modifier_hits.append((tag, score))
            else:
                sexual_explicit_sum += score

        gore_sum = self._sum_by_tags(general_map, GORE_TAGS)
        gore_peak = self._peak_by_tags(general_map, GORE_TAGS)
        dismember_peak = self._peak_by_tags(general_map, DISMEMBER_TAGS)

        drug_score = self._max_substring_score(general_map, DRUG_KEYWORDS)

        thresholds = self.config.thresholds
        minor_peak_min = float(thresholds.get("minor_peak_min", 0.20))
        sexual_explicit_sum_med = float(thresholds.get("sexual_explicit_sum_med", 0.10))
        sexual_explicit_sum_high = float(thresholds.get("sexual_explicit_sum_high", 0.15))
        sexual_modifier_sum_min = float(thresholds.get("sexual_modifier_sum_min", 0.10))
        animal_subject_min = float(thresholds.get("animal_subject_min", 0.35))
        gore_peak_min = float(thresholds.get("gore_peak_min", 0.30))
        gore_sum_min = float(thresholds.get("gore_sum_min", 0.40))
        dismember_peak_min = float(thresholds.get("dismember_peak_min", 0.20))
        mild_exposure_peak_min = float(thresholds.get("mild_exposure_peak_min", self._mild_threshold))
        drug_any_min = float(thresholds.get("drug_any_min", 0.15))

        sexual_high = (
            exposure_peak >= float(thresholds.get("exposure_strong", 0.60))
            or explicit >= float(thresholds.get("wd14_explicit", 0.20))
            or sexual_explicit_sum >= sexual_explicit_sum_high
        )
        sexual_med = (
            exposure_peak >= float(thresholds.get("exposure_mid", 0.30))
            or questionable >= float(thresholds.get("wd14_questionable", 0.35))
            or sexual_explicit_sum >= sexual_explicit_sum_med
        )
        sexual_with_modifiers = sexual_modifier_sum >= sexual_modifier_sum_min and (
            sexual_med or exposure_peak >= float(thresholds.get("exposure_mid", 0.30))
        )
        gore_any = gore_peak >= gore_peak_min or gore_sum >= gore_sum_min
        dismember_hit = dismember_peak >= dismember_peak_min
        mild_exposure = mild_exposure_peak >= mild_exposure_peak_min or (
            mild_exposure_peak == 0.0 and questionable >= float(thresholds.get("wd14_questionable", 0.35))
        )
        drug_any = drug_score >= drug_any_min

        metrics = {
            "questionable": questionable,
            "explicit": explicit,
            "general_rating": general_rating,
            "sensitive_rating": sensitive_rating,
            "nsfw_margin": nsfw_margin,
            "nsfw_ratio": nsfw_ratio,
            "nsfw_general_sum": nsfw_general_sum,
            "exposure_score": exposure_score,
            "exposure_detection": exposure_detection,
            "exposure_peak": exposure_peak,
            "mild_exposure_peak": mild_exposure_peak,
            "placement_risk": placement_risk,
            "violence_max": violence_max,
            "violence_sum": violence_sum,
            "minors_sum": minors_sum,
            "minors_peak": minors_peak,
            "animals_peak": animal_peak,
            "animal_context_peak": animal_context_peak,
            "sexual_explicit_sum": sexual_explicit_sum,
            "sexual_modifier_sum": sexual_modifier_sum,
            "gore_peak": gore_peak,
            "gore_sum": gore_sum,
            "dismember_peak": dismember_peak,
            "drug_score": drug_score,
        }

        messages = analysis.get("messages", []) or []
        attachment_count = 0
        is_spoiler = bool(analysis.get("is_spoiler", False))
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            attachment_count += len(message.get("attachments", []) or [])
            if message.get("is_spoiler"):
                is_spoiler = True
            for attachment in message.get("attachments", []) or []:
                if isinstance(attachment, Mapping) and attachment.get("is_spoiler"):
                    is_spoiler = True

        nude_flags: tuple[str, ...] = tuple(
            str(det.get("class", "")).upper()
            for det in nudity
            if isinstance(det, Mapping) and det.get("class")
        )

        rating_map: Dict[str, float] = {
            str(key): float(value)
            for key, value in rating.items()
            if isinstance(value, (int, float))
        }
        for key in ("explicit", "questionable", "general", "sensitive", "safe"):
            rating_map.setdefault(key, 0.0)

        prepared = PreparedContext(
            rating=rating_map,
            tag_scores=normalized_tag_scores,
            nude_flags=nude_flags,
            is_nsfw_channel=is_nsfw_channel,
            is_spoiler=is_spoiler,
            attachment_count=attachment_count,
            metrics=metrics,
        )

        reasons_base: list[str] = []
        if wd14.get("error") or not rating:
            reasons_base.append("wd14_missing")
        # 1. 非NSFWチャンネル × 性的
        if not is_nsfw_channel and (sexual_high or sexual_med or sexual_with_modifiers):
            reasons = reasons_base + [
                "channel=non-nsfw",
                f"sexual_high={sexual_high}",
                f"sexual_med={sexual_med}",
                f"mod_sum={sexual_modifier_sum:.2f}",
            ]
            return self._result("red", "RED-NSFW-101", reasons, metrics), prepared

        # 2. 未成年 × 性的
        if minors_peak >= minor_peak_min and (sexual_high or sexual_with_modifiers):
            reasons = reasons_base + [
                f"minor_peak={minors_peak:.2f}",
                f"sexual_high={sexual_high}",
                f"mod_sum={sexual_modifier_sum:.2f}",
            ]
            if sexual_modifier_hits:
                mods = ",".join(tag for tag, _ in sexual_modifier_hits)
                reasons.append(f"mods={mods}")
            return self._result("red", "RED-MINOR-SEX-201", reasons, metrics), prepared

        # 3. 未成年 × ゴア
        if minors_peak >= minor_peak_min and gore_any:
            reasons = reasons_base + [
                f"minor_peak={minors_peak:.2f}",
                f"gore_peak={gore_peak:.2f}",
                f"gore_sum={gore_sum:.2f}",
            ]
            return self._result("red", "RED-MINOR-GORE-202", reasons, metrics), prepared

        # 4. 動物 × 性的
        bestiality_hit = animal_context_peak > 0.0 and any(
            tag in ANIMAL_ABUSE_CONTEXT_TAGS
            for tag, score in general_map.items()
            if score > 0.0 and tag in ANIMAL_ABUSE_CONTEXT_TAGS
        )
        if animal_peak >= animal_subject_min and (
            bestiality_hit or (sexual_high and sexual_modifier_sum >= sexual_modifier_sum_min)
        ):
            reasons = reasons_base + [
                f"animal_peak={animal_peak:.2f}",
                f"bestiality={bestiality_hit}",
                f"sexual_high={sexual_high}",
                f"mod_sum={sexual_modifier_sum:.2f}",
            ]
            return self._result("red", "RED-ANIMAL-SEX-301", reasons, metrics), prepared

        # 5. 動物 × 虐待/流血
        if animal_peak >= animal_subject_min and (animal_context_peak > 0.0 or gore_any):
            reasons = reasons_base + [
                f"animal_peak={animal_peak:.2f}",
                f"context_peak={animal_context_peak:.2f}",
                f"gore_any={gore_any}",
            ]
            return self._result("red", "RED-ANIMAL-GORE-302", reasons, metrics), prepared

        # 6. 欠損 × 出血
        if dismember_hit and gore_any:
            reasons = reasons_base + [
                f"dismember={dismember_peak:.2f}",
                f"gore_peak={gore_peak:.2f}",
                f"gore_sum={gore_sum:.2f}",
            ]
            return self._result("red", "RED-DISMEMBER-BLOOD-401", reasons, metrics), prepared

        # 7. 成人 × 性的 × 薬物（NSFWのみ）
        if minors_peak < minor_peak_min and (sexual_high or sexual_med or sexual_with_modifiers) and drug_any:
            reasons = reasons_base + [
                f"sexual_state={'high' if sexual_high else 'med'}",
                f"drug_score={drug_score:.2f}",
            ]
            if not is_nsfw_channel:
                # 非NSFWの場合は規約REDが優先しているが、保険で reason を残す。
                reasons.append("channel=non-nsfw")
            return self._result("orange", "ORANGE-ADULT-SEX-DRUG-501", reasons, metrics), prepared

        # 8. 未成年 × 軽微露出
        if minors_peak >= minor_peak_min and mild_exposure and not sexual_high:
            reasons = reasons_base + [
                f"minor_peak={minors_peak:.2f}",
                f"mild_exposure={mild_exposure_peak:.2f}",
            ]
            return self._result("orange", "ORANGE-MINOR-MILD-601", reasons, metrics), prepared

        return self._result(DEFAULT_SEVERITY, None, reasons_base, metrics), prepared

    def _evaluate_dsl(self, context: PreparedContext) -> DslEvaluationOutcome | None:
        if not self._dsl_program:
            return None
        inputs = DslEvaluationInput(
            rating=context.rating,
            metrics=context.metrics,
            tag_scores=context.tag_scores,
            group_patterns=self._dsl_program.group_patterns,
            nude_flags=context.nude_flags,
            is_nsfw_channel=context.is_nsfw_channel,
            is_spoiler=context.is_spoiler,
            attachment_count=context.attachment_count,
        )
        return self._dsl_program.evaluate(inputs)

    def _merge_results(
        self,
        legacy: EvaluationResult,
        dsl_outcome: DslEvaluationOutcome,
    ) -> EvaluationResult:
        severity_rank = SEVERITY_ORDER
        legacy_rank = severity_rank.get(legacy.severity, 0)
        dsl_rank = severity_rank.get(dsl_outcome.severity, 0)

        legacy_priority = 100
        dsl_priority = dsl_outcome.priority

        winner = "legacy"
        if dsl_rank > legacy_rank:
            winner = "dsl"
        elif dsl_rank == legacy_rank and dsl_priority < legacy_priority:
            winner = "dsl"

        metrics = dict(legacy.metrics)
        metrics["dsl"] = {
            "rule_id": dsl_outcome.rule_id,
            "severity": dsl_outcome.severity,
            "priority": dsl_outcome.priority,
            **dsl_outcome.diagnostics,
            "reasons": dsl_outcome.reasons,
            "matched": True,
        }

        if winner == "dsl":
            primary_reasons = list(dsl_outcome.reasons)
            secondary_reasons = legacy.reasons
            severity = dsl_outcome.severity
            rule_id = dsl_outcome.rule_id
            rule_title = self.config.rule_titles.get(rule_id)
            priority = dsl_priority
        else:
            primary_reasons = list(legacy.reasons)
            secondary_reasons = dsl_outcome.reasons
            severity = legacy.severity
            rule_id = legacy.rule_id
            rule_title = legacy.rule_title
            priority = legacy_priority

        extras = [reason for reason in secondary_reasons if reason not in primary_reasons][:2]
        merged_reasons = primary_reasons + extras

        metrics["winning"] = {
            "origin": winner,
            "severity": severity,
            "rule_id": rule_id,
            "priority": priority,
        }

        return EvaluationResult(
            severity=severity,
            rule_id=rule_id,
            rule_title=rule_title,
            reasons=merged_reasons,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # ユーティリティ
    # ------------------------------------------------------------------
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

    def _mild_exposure_peak(self, nudity: Iterable[Mapping[str, Any]]) -> float:
        patterns = self._mild_patterns
        max_score = 0.0
        for det in nudity or []:
            label = str(det.get("class", "") or "").upper()
            if "EXPOSED" not in label or "COVERED" in label:
                continue
            if any(pattern in label for pattern in patterns):
                score = float(det.get("score", 0.0))
                if score > max_score:
                    max_score = score
        return max_score

    @staticmethod
    def _sum_by_tags(score_map: Mapping[str, float], tags: Iterable[str]) -> float:
        return sum(float(score_map.get(tag, 0.0)) for tag in tags)

    @staticmethod
    def _peak_by_tags(score_map: Mapping[str, float], tags: Iterable[str]) -> float:
        scores = [float(score_map.get(tag, 0.0)) for tag in tags]
        return max(scores) if scores else 0.0

    @staticmethod
    def _max_substring_score(score_map: Mapping[str, float], patterns: Sequence[str]) -> float:
        max_score = 0.0
        for tag, score in score_map.items():
            if score <= 0.0:
                continue
            for keyword in patterns:
                if keyword in tag:
                    if score > max_score:
                        max_score = float(score)
        return max_score

    def _result(
        self,
        severity: str,
        rule_id: Optional[str],
        reasons: List[str],
        metrics: Dict[str, Any],
    ) -> EvaluationResult:
        title = self.config.rule_titles.get(rule_id) if rule_id else None
        return EvaluationResult(
            severity=severity,
            rule_id=rule_id,
            rule_title=title,
            reasons=reasons,
            metrics=metrics,
        )


def _normalize_tag(name: str) -> str:
    value = str(name)
    if value in KEEP_UNDERSCORE:
        return value
    return value.replace("_", " ")
