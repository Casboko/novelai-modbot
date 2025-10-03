from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

from .dsl import (
    SEVERITY_ORDER,
    CompiledFeature,
    CompiledRule,
    DslProgram,
    SafeEvaluator,
)
from .dsl_errors import DslRuntimeError
from .dsl_runtime import build_context
from .tag_norm import normalize_pair



def _clip_unit_interval(value: float) -> float:
    """Clamp the provided value into the inclusive range [0.0, 1.0]."""

    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


FEATURE_TO_EXPORT: Mapping[str, str] = {
    "nsfw_margin": "nsfw_margin",
    "nsfw_ratio": "nsfw_ratio",
    "nsfw_general_sum": "nsfw_general_sum",
    "violence_sum": "violence_sum",
    "violence_peak": "violence_max",
    "violence_max": "violence_max",
    "minors_sum": "minors_sum",
    "minors_peak": "minors_max",
    "minors_max": "minors_max",
    "animals_sum": "animals_sum",
    "animals_peak": "animals_max",
    "animals_max": "animals_max",
    "gore_sum": "gore_sum",
    "gore_peak": "gore_max",
    "gore_max": "gore_max",
    "drug_score": "drug_score",
    "qe_margin": "qe_margin",
}


@dataclass(slots=True)
class PreparedContext:
    rating: Mapping[str, float]
    tag_scores: Mapping[str, float]
    metrics: Dict[str, float]
    nude_flags: Sequence[str]
    is_nsfw_channel: bool
    is_spoiler: bool
    attachment_count: int


@dataclass(slots=True)
class EvaluatorResult:
    severity: str
    rule_id: str | None
    priority: int
    reasons: list[str]
    metrics: Dict[str, Any]


class DslEvaluator:
    def __init__(self, program: DslProgram) -> None:
        self._program = program

    def build_context(self, analysis: Mapping[str, Any]) -> PreparedContext:
        wd14 = analysis.get("wd14") or {}
        rating_raw = wd14.get("rating") or {}
        rating: Dict[str, float] = {
            str(key): float(value)
            for key, value in rating_raw.items()
            if isinstance(value, (int, float))
        }
        for key in ("explicit", "questionable", "general", "sensitive", "safe"):
            rating.setdefault(key, 0.0)

        general_entries = wd14.get("general_raw") or wd14.get("general") or []
        tag_scores: Dict[str, float] = {}
        for item in general_entries:
            pair = normalize_pair(item)
            if pair is None:
                continue
            tag, score = pair
            existing = tag_scores.get(tag)
            if existing is None or score > existing:
                tag_scores[tag] = score

        xsignals = analysis.get("xsignals") or {}
        metrics: Dict[str, float] = {}
        for key in (
            "exposure_score",
            "placement_risk_pre",
            "nsfw_margin",
            "nsfw_ratio",
            "nsfw_general_sum",
            "nudity_area_ratio",
            "nudity_box_count",
        ):
            value = xsignals.get(key)
            if isinstance(value, (int, float)):
                metrics[key] = float(value)

        area_ratio = metrics.get("nudity_area_ratio", 0.0)
        metrics["exposure_area"] = float(area_ratio)
        box_count = metrics.get("nudity_box_count", 0.0)
        metrics["exposure_count"] = float(box_count)

        messages = analysis.get("messages", []) or []
        attachment_count = 0
        is_spoiler = bool(analysis.get("is_spoiler", False))
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            attachments = message.get("attachments", []) or []
            attachment_count += len(attachments)
            if message.get("is_spoiler"):
                is_spoiler = True
            for attachment in attachments:
                if isinstance(attachment, Mapping) and attachment.get("is_spoiler"):
                    is_spoiler = True

        nudity = analysis.get("nudity_detections", []) or []
        nude_flags: list[str] = []
        exposed_max = 0.0
        exposed_prod = 1.0
        for det in nudity:
            if not isinstance(det, Mapping):
                continue
            label = det.get("class")
            if not label:
                continue
            canonical_label = str(label).upper()
            nude_flags.append(canonical_label)

            if "EXPOSED" not in canonical_label or "COVERED" in canonical_label:
                continue

            score_raw = det.get("score", 0.0)
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                continue
            clipped = _clip_unit_interval(score)
            if clipped > exposed_max:
                exposed_max = clipped
            exposed_prod *= 1.0 - clipped

        metrics.setdefault("exposure_peak", _clip_unit_interval(exposed_max))
        metrics.setdefault("exposure_score", _clip_unit_interval(1.0 - exposed_prod))

        return PreparedContext(
            rating=rating,
            tag_scores=tag_scores,
            metrics=metrics,
            nude_flags=tuple(nude_flags),
            is_nsfw_channel=bool(analysis.get("is_nsfw_channel")),
            is_spoiler=is_spoiler,
            attachment_count=attachment_count,
        )

    def evaluate(self, analysis: Mapping[str, Any]) -> EvaluatorResult:
        context = self.build_context(analysis)

        runtime = build_context(
            rating=context.rating,
            metrics=context.metrics,
            tag_scores=context.tag_scores,
            group_patterns=self._program.group_patterns,
            nude_flags=context.nude_flags,
            is_nsfw_channel=context.is_nsfw_channel,
            is_spoiler=context.is_spoiler,
            attachment_count=context.attachment_count,
        )

        for key, value in self._program.const_map.items():
            runtime.namespace[key] = float(value)

        evaluator = SafeEvaluator(runtime.namespace, policy=self._program._policy)

        feature_values = self._evaluate_features(self._program.compiled_features, evaluator)

        for name, value in feature_values.items():
            runtime.namespace[name] = value

        hits = self._evaluate_rules(self._program.compiled_rules, evaluator)

        if hits:
            compiled_rule, reasons = self._select_winner(hits)
            rule = compiled_rule.rule
            severity = rule.severity
            rule_id = rule.id
            priority = rule.priority or 0
        else:
            compiled_rule = None
            reasons = []
            severity = "green"
            rule_id = None
            priority = 0

        metrics = self._export_metrics(context.metrics, feature_values)
        metrics["winning"] = {
            "origin": "dsl",
            "severity": severity,
            "rule_id": rule_id,
            "priority": priority,
        }
        metrics["dsl"] = {
            "origin": "dsl",
            "matched": bool(compiled_rule),
            "reasons": reasons,
            "features": feature_values,
        }

        return EvaluatorResult(
            severity=severity,
            rule_id=rule_id,
            priority=priority,
            reasons=reasons,
            metrics=metrics,
        )

    def _evaluate_features(
        self,
        features: Sequence[CompiledFeature],
        evaluator: SafeEvaluator,
    ) -> Dict[str, float]:
        values: Dict[str, float] = {}
        namespace = evaluator._namespace  # underlying dict shared with runtime
        for feature in features:
            with evaluator.context(f"feature:{feature.name}"):
                try:
                    value = evaluator.evaluate(feature.expression)
                except DslRuntimeError:
                    if evaluator.strict:
                        raise
                    value = 0.0
            if isinstance(value, (int, float)):
                numeric = float(value)
            else:
                numeric = 0.0
            values[feature.name] = numeric
            try:
                namespace[feature.name] = numeric
            except TypeError:
                # namespace may be an immutable mapping; skip in that case
                pass
        return values

    def _evaluate_rules(
        self,
        rules: Sequence[CompiledRule],
        evaluator: SafeEvaluator,
    ) -> list[tuple[CompiledRule, list[str]]]:
        hits: list[tuple[CompiledRule, list[str]]] = []
        for compiled_rule in rules:
            rule = compiled_rule.rule
            matched = False
            with evaluator.context(f"rule:{rule.id}:when"):
                try:
                    matched = bool(evaluator.evaluate(compiled_rule.condition))
                except DslRuntimeError:
                    if evaluator.strict:
                        raise
                    matched = False
            if not matched:
                continue
            reasons: list[str] = []
            if compiled_rule.reasons:
                for idx, template in enumerate(compiled_rule.reasons):
                    context = f"rule:{rule.id}:reason:{idx}"
                    with evaluator.context(context):
                        try:
                            rendered = template.render(evaluator, context)
                        except DslRuntimeError:
                            if evaluator.strict:
                                raise
                            rendered = ""
                    if rendered:
                        reasons.append(rendered)
            if not reasons:
                reasons.append(rule.id)
            hits.append((compiled_rule, reasons))
            if rule.stop:
                break
        return hits

    def _select_winner(
        self,
        hits: Iterable[tuple[CompiledRule, list[str]]],
    ) -> tuple[CompiledRule, list[str]]:
        return min(hits, key=self._sort_key)

    def _sort_key(self, item: tuple[CompiledRule, list[str]]) -> tuple[int, int, int]:
        compiled_rule, _ = item
        rule = compiled_rule.rule
        severity_rank = SEVERITY_ORDER.get(rule.severity, 0)
        priority = int(rule.priority or 0)
        return (-severity_rank, -priority, compiled_rule.order)

    def _export_metrics(
        self,
        base_metrics: Mapping[str, float],
        features: Mapping[str, float],
    ) -> Dict[str, Any]:
        metrics = {key: float(value) for key, value in base_metrics.items()}
        for feature_name, export_name in FEATURE_TO_EXPORT.items():
            value = features.get(feature_name)
            if isinstance(value, (int, float)):
                metrics[export_name] = float(value)
        if "exposure_score" in base_metrics and "exposure_score" not in metrics:
            metrics["exposure_score"] = float(base_metrics["exposure_score"])
        placement = base_metrics.get("placement_risk_pre")
        if isinstance(placement, (int, float)):
            metrics["placement_risk"] = float(placement)
            metrics.setdefault("placement_risk_pre", float(placement))
        return metrics
