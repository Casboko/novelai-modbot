from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import altair as alt
import pandas as pd
import streamlit as st
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.batch_loader import ImageRequest, load_images
from app.local_cache import resolve_local_file
from tools.scl_viewer import utils

TMP_DIR = Path("tools/scl_viewer/tmp")
THUMBS_DIR = Path(os.getenv("SCL_CACHE_THUMBS", "tools/scl_viewer/thumbs"))
CACHE_ROOT = Path(os.getenv("CACHE_ROOT", "cache"))
CACHE_DIR = Path(os.getenv("SCL_CACHE_FULL", CACHE_ROOT / "imgs"))
SCL_NO_FETCH = os.getenv("SCL_NO_FETCH", "0") == "1"

FINDINGS_PATH = TMP_DIR / "findings.jsonl"
REPORT_PATH = TMP_DIR / "report.csv"
RULES_COMPILED_PATH = TMP_DIR / "rules_compiled.yaml"
P0_MERGED_PATH = TMP_DIR / "p0_merged.csv"
AB_DIR = TMP_DIR / "ab"

SEVERITY_ORDER = ["red", "orange", "yellow", "green"]
DEFAULT_RULES = Path("configs/rules_v2.yaml")
DEFAULT_THRESHOLDS = Path("configs/scl/scl_thresholds.yaml")
DEFAULT_ANALYSIS = Path("out/p2/p2_analysis_all.jsonl")
DEFAULT_RULES_PREV = Path("configs/rules_v2_prev.yaml")

COLUMN_PRESETS: dict[str, list[str] | None] = {
    "Moderation Core": [
        "phash",
        "severity",
        "rule_id",
        "rule_title",
        "wd14_rating_g",
        "wd14_rating_s",
        "wd14_rating_q",
        "wd14_rating_e",
        "exposure_area",
        "exposure_count",
        "exposure_score",
        "coercion_score",
    ],
    "v2 Features": [
        "phash",
        "severity",
        "rule_id",
        "rule_title",
        "explicit_score",
        "sexual_intensity",
        "nsfw_rating_max",
        "minor_peak_conf",
        "minor_body_score",
        "minor_context_score",
        "minor_special_score",
        "gore_peak_conf",
        "gore_density",
        "gore_intensity",
        "animal_presence",
        "coercion_score",
    ],
    "All": None,
}


def main() -> None:
    st.set_page_config(page_title="SCL Viewer", layout="wide")
    st.title("SCL ルールチューニングビューア")
    ensure_directories()

    sidebar_state = render_sidebar()

    action_placeholder = st.empty()

    if sidebar_state.run_clicked:
        handle_run(action_placeholder, sidebar_state)
    if sidebar_state.report_clicked:
        handle_report(action_placeholder)
    if sidebar_state.ab_clicked:
        handle_ab(action_placeholder, sidebar_state)
    if sidebar_state.contract_clicked:
        handle_contract(action_placeholder)

    findings_records = st.session_state.get("findings_records")
    if findings_records:
        render_findings(findings_records, sidebar_state)
    else:
        st.info("Run を実行すると最新の findings を表示します。")

    if "report_df" in st.session_state:
        st.subheader("p3 レポート (CSV)")
        st.dataframe(st.session_state["report_df"], width="stretch")

    if "ab_compare" in st.session_state:
        render_ab_outputs()
@dataclass
class SidebarState:
    rules_path: Path
    thresholds_path: Path | None
    analysis_path: Path
    rules_prev_path: Path | None
    limit: int
    offset: int
    severity_filter: list[str]
    rule_query: str
    exposure_min: float
    since: str
    until: str
    run_clicked: bool
    report_clicked: bool
    ab_clicked: bool
    contract_clicked: bool


def render_sidebar() -> SidebarState:
    with st.sidebar:
        st.header("パラメータ")
        rules_path = Path(
            st.text_input("Rules (B)", value=str(DEFAULT_RULES), help="チューニング対象のルール定義")
        )
        thresholds_value = st.text_input(
            "閾値ファイル", value=str(DEFAULT_THRESHOLDS), help="configs/scl/scl_thresholds.yaml"
        )
        thresholds_path = Path(thresholds_value)
        if not thresholds_path.exists():
            thresholds_path = None
        analysis_path = Path(
            st.text_input("Analysis JSONL", value=str(DEFAULT_ANALYSIS), help="out/p2 の統合 JSONL")
        )
        rules_prev_input = st.text_input(
            "Rules (A)", value=str(DEFAULT_RULES_PREV), help="比較用の旧ルール"
        )
        rules_prev_path = Path(rules_prev_input) if rules_prev_input else None
        limit = int(st.number_input("Limit", min_value=0, value=800, step=100))
        offset = int(st.number_input("Offset", min_value=0, value=0, step=50))
        severity_filter = st.multiselect(
            "表示する severity",
            options=SEVERITY_ORDER,
            default=SEVERITY_ORDER,
        )
        rule_query = st.text_input("rule_id 部分一致", value="")
        exposure_min = st.slider(
            "exposure_area の下限",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
        )
        since = st.text_input("since", value="2000-01-01", help="YYYY-MM-DD ないし相対指定")
        until = st.text_input("until", value="2100-01-01", help="YYYY-MM-DD ないし相対指定")
        st.divider()
        run_clicked = st.button("Run (cli_scan)")
        report_clicked = st.button("Report (cli_report)")
        ab_clicked = st.button("A/B diff (cli_rules_ab)")
        contract_clicked = st.button("契約チェック (cli_contract)")
    return SidebarState(
        rules_path=rules_path,
        thresholds_path=thresholds_path,
        analysis_path=analysis_path,
        rules_prev_path=rules_prev_path,
        limit=limit,
        offset=offset,
        severity_filter=severity_filter,
        rule_query=rule_query,
        exposure_min=exposure_min,
        since=since,
        until=until,
        run_clicked=run_clicked,
        report_clicked=report_clicked,
        ab_clicked=ab_clicked,
        contract_clicked=contract_clicked,
    )


def handle_run(placeholder, state: SidebarState) -> None:
    with placeholder.container():
        st.write("### Run 実行ログ")
        try:
            merged = utils.merge_p0_sources(P0_MERGED_PATH)
        except utils.P0MergeError as exc:
            st.error(str(exc))
            return
        if merged is None:
            st.warning("p0 CSV が見つかりませんでした。添付メタは付かないまま実行します。")

        try:
            compiled = prepare_rules(state.rules_path, state.thresholds_path)
        except utils.ThresholdsError as exc:
            st.error(str(exc))
            return

        extra_args: list[str] = []
        if state.since.strip():
            extra_args.extend(["--since", state.since.strip()])
        if state.until.strip():
            extra_args.extend(["--until", state.until.strip()])
        if merged is None:
            p0_path = None
        else:
            p0_path = merged
        try:
            result = utils.run_cli_scan(
                analysis=state.analysis_path,
                findings=FINDINGS_PATH,
                rules=compiled.rules_path,
                p0=p0_path,
                limit=state.limit,
                offset=state.offset,
                extra_args=extra_args,
            )
        except (utils.CliCommandError, FileNotFoundError) as exc:
            st.error(str(exc))
            if isinstance(exc, utils.CliCommandError):
                st.code(exc.stderr or exc.stdout)
            return
        st.success("cli_scan 完了")
        st.code(result.stdout)
        st.session_state["rules_effective_path"] = str(compiled.rules_path)
        load_findings_into_state()


def handle_report(placeholder) -> None:
    with placeholder.container():
        st.write("### Report 実行ログ")
        if not FINDINGS_PATH.exists():
            st.warning("先に Run を実行し findings を生成してください。")
            return
        try:
            result = utils.run_cli_report(findings=FINDINGS_PATH, out_path=REPORT_PATH)
        except utils.CliCommandError as exc:
            st.error(str(exc))
            st.code(exc.stderr or exc.stdout)
            return
        st.success("cli_report 完了")
        st.code(result.stdout)
        try:
            st.session_state["report_df"] = utils.read_report_csv(REPORT_PATH)
        except utils.UtilsError as exc:
            st.error(str(exc))


def handle_ab(placeholder, state: SidebarState) -> None:
    with placeholder.container():
        st.write("### A/B diff 実行ログ")
        if state.rules_prev_path is None or not state.rules_prev_path.exists():
            st.error("比較対象 Rules (A) が見つかりません")
            return
        try:
            compiled = prepare_rules(state.rules_path, state.thresholds_path)
        except utils.ThresholdsError as exc:
            st.error(str(exc))
            return
        AB_DIR.mkdir(parents=True, exist_ok=True)
        try:
            result = utils.run_cli_ab(
                analysis=state.analysis_path,
                rules_a=state.rules_prev_path,
                rules_b=compiled.rules_path,
                out_dir=AB_DIR,
                sample_diff=200,
                extra_args=("--samples-minimal", "--samples-redact-urls"),
            )
        except utils.CliCommandError as exc:
            st.error(str(exc))
            st.code(exc.stderr or exc.stdout)
            return
        st.success("cli_rules_ab 完了")
        st.code(result.stdout)
        load_ab_outputs()


def handle_contract(placeholder) -> None:
    with placeholder.container():
        st.write("### 契約チェック")
        if not REPORT_PATH.exists():
            st.warning("レポート CSV が見つかりません。先に Report を実行してください。")
            return
        try:
            result = utils.run_cli_contract_report(report_path=REPORT_PATH)
        except utils.CliCommandError as exc:
            st.error(str(exc))
            st.code(exc.stderr or exc.stdout)
            return
        st.success("report header: ok")
        st.code(result.stdout or "report header: ok")


def prepare_rules(rules_path: Path, thresholds_path: Path | None) -> utils.ThresholdCompileResult:
    scan = utils.scan_threshold_symbols(rules_path)
    if thresholds_path is None:
        missing = scan.referenced - scan.defined
        if missing:
            formatted = ", ".join(sorted(missing))
            raise utils.ThresholdsError(
                "thresholds YAML が存在しません。以下の定数に値を設定してください: %s" % formatted
            )
    return utils.compile_rules_with_thresholds(
        rules_path,
        thresholds_path=thresholds_path,
        output_path=RULES_COMPILED_PATH,
    )


def load_findings_into_state() -> None:
    try:
        records = utils.read_jsonl(FINDINGS_PATH)
    except utils.UtilsError as exc:
        st.error(str(exc))
        return
    st.session_state["findings_records"] = records
    rules_source = st.session_state.get("rules_effective_path")
    rules_path = Path(rules_source) if rules_source else DEFAULT_RULES
    st.session_state["findings_df"] = build_findings_dataframe(records, rules_path=rules_path)


def load_ab_outputs() -> None:
    compare_path = AB_DIR / "p3_ab_compare.json"
    diff_csv = AB_DIR / "p3_ab_diff.csv"
    samples_path = AB_DIR / "p3_ab_diff_samples.jsonl"
    payload = None
    if compare_path.exists():
        payload = json.loads(compare_path.read_text(encoding="utf-8"))
    st.session_state["ab_compare"] = payload
    if diff_csv.exists():
        st.session_state["ab_diff_df"] = pd.read_csv(diff_csv)
    else:
        st.session_state.pop("ab_diff_df", None)
    if samples_path.exists():
        st.session_state["ab_samples"] = list(_iter_jsonl(samples_path, limit=200))
    else:
        st.session_state.pop("ab_samples", None)


def render_findings(records: list[dict], state: SidebarState) -> None:
    rules_source = st.session_state.get("rules_effective_path")
    rules_path = Path(rules_source) if rules_source else state.rules_path

    df = st.session_state.get("findings_df")
    if df is None:
        df = build_findings_dataframe(records, rules_path=rules_path)
    df = df.copy()

    filtered = df
    if state.severity_filter and "severity" in filtered.columns:
        filtered = filtered[filtered["severity"].isin(state.severity_filter)]
    if state.rule_query and "rule_id" in filtered.columns:
        filtered = filtered[filtered["rule_id"].str.contains(state.rule_query, case=False, na=False)]
    if state.exposure_min > 0 and "exposure_area" in filtered.columns:
        filtered = filtered[filtered["exposure_area"].fillna(0.0) >= state.exposure_min]
    filtered = filtered.copy()
    if "severity" in filtered.columns:
        filtered = filtered.sort_values("severity", key=_severity_sort_key)

    st.session_state["findings_df"] = df

    st.subheader("Findings")
    left_col, right_col = st.columns([5, 3], gap="large")

    with left_col:
        preset = st.radio(
            "列プリセット",
            list(COLUMN_PRESETS.keys()),
            horizontal=True,
            key="scl_column_preset",
        )

        available_columns = list(filtered.columns)
        default_columns = _default_visible_columns(available_columns, preset)

        previous_preset = st.session_state.get("_scl_prev_preset")
        if previous_preset != preset:
            st.session_state["_scl_prev_preset"] = preset
            st.session_state["scl_visible_columns"] = default_columns

        visible_columns = st.multiselect(
            "表示列",
            options=available_columns,
            default=st.session_state.get("scl_visible_columns", default_columns),
            key="scl_visible_columns",
        )
        visible_columns = [col for col in visible_columns if col in filtered.columns]
        if not visible_columns:
            visible_columns = default_columns or available_columns
        st.session_state["scl_visible_columns"] = visible_columns
        df_show = filtered[visible_columns]

        table_column_config: dict[str, Any] = {}
        if "thumbnail" in df_show.columns:
            table_column_config["thumbnail"] = st.column_config.ImageColumn(
                "thumb",
                help="p0 添付から生成したサムネイル",
                width=120,
            )

        st.dataframe(
            df_show.reset_index(drop=True),
            width="stretch",
            column_config=table_column_config,
        )

        selected_phash = None
        if not filtered.empty and "phash" in filtered.columns:
            options = filtered["phash"].tolist()
            default_index = 0
            stored = st.session_state.get("last_selected_phash")
            if stored in options:
                default_index = options.index(stored)
            selected_phash = st.selectbox(
                "詳細表示するレコード",
                options,
                index=default_index,
                key="scl_selected_phash",
            )
            st.session_state["last_selected_phash"] = selected_phash
        else:
            st.session_state["last_selected_phash"] = None

        st.session_state["findings_df"] = df

    with right_col:
        selected_record = None
        selected_phash = st.session_state.get("last_selected_phash")
        if selected_phash:
            selected_record = next((rec for rec in records if rec.get("phash") == selected_phash), None)
        render_detail_panel(selected_record, rules_path=rules_path)
def get_thumbnail_for_record(record: dict) -> Path | None:
    phash = record.get("phash")
    if not phash:
        return None
    local = resolve_local_file(str(phash))
    image_path = Path(local) if local else (CACHE_DIR / f"{phash}.jpg")
    if not image_path.exists():
        if SCL_NO_FETCH:
            return None
        urls = list(_attachment_urls(record))
        if not urls:
            return None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        downloaded = download_image(phash, urls, image_path)
        if downloaded is None:
            return None
    thumb_path = THUMBS_DIR / f"{phash}.jpg"
    if not thumb_path.exists() or thumb_path.stat().st_mtime < image_path.stat().st_mtime:
        utils.ensure_thumbnail(image_path, thumb_path)
    return thumb_path


def download_image(identifier: str, urls: Iterable[str], dest: Path) -> Path | None:
    request = ImageRequest(identifier=identifier, urls=tuple(urls))
    try:
        results = asyncio.run(load_images([request], qps=3.0, concurrency=2))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(load_images([request], qps=3.0, concurrency=2))
        finally:
            asyncio.set_event_loop(None)
            loop.close()
    result = results[0]
    if result.image is None:
        return None
    original = result.image
    image = original.convert("RGB")
    original.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, format="JPEG", quality=95)
    return dest


def render_ab_outputs() -> None:
    compare = st.session_state.get("ab_compare")
    if compare:
        st.subheader("A/B サマリー")
        st.json(compare)
    diff_df = st.session_state.get("ab_diff_df")
    if diff_df is not None:
        st.subheader("A/B 差分 CSV")
        st.dataframe(diff_df, width="stretch")
    samples = st.session_state.get("ab_samples")
    if samples:
        st.subheader("A/B サンプル")
        st.json(samples[:50])


def ensure_directories() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)


def build_findings_dataframe(records: list[dict], *, rules_path: Path | None = None) -> pd.DataFrame:
    feature_ids = _discover_feature_ids(rules_path) if rules_path else []
    default_features = [
        "nsfw_rating_max",
        "explicit_score",
        "sexual_intensity",
        "minor_peak_conf",
        "minor_body_score",
        "minor_context_score",
        "minor_special_score",
        "gore_peak_conf",
        "gore_density",
        "gore_intensity",
        "animal_presence",
        "coercion_score",
    ]
    present: set[str] = set()
    for record in records[:50]:
        metrics = record.get("metrics", {}) or {}
        dsl = metrics.get("dsl", {}) or {}
        feats = (dsl.get("features", {}) or {})
        if isinstance(feats, dict):
            present.update(key for key in feats.keys() if isinstance(key, str))

    col_features: list[str] = []
    seen: set[str] = set()
    for key in default_features + feature_ids + sorted(present):
        if key and key not in seen:
            seen.add(key)
            col_features.append(key)

    rows = []
    for record in records:
        metrics = record.get("metrics", {}) or {}
        dsl = metrics.get("dsl", {}) or {}
        feats = (dsl.get("features", {}) or {})
        thumb_path = get_thumbnail_for_record(record)
        ratings = _extract_wd14_rating(record)
        row = {
            "thumbnail": str(thumb_path) if thumb_path else None,
            "phash": record.get("phash"),
            "severity": record.get("severity"),
            "rule_id": record.get("rule_id"),
            "rule_title": record.get("rule_title"),
            "message_link": record.get("message_link"),
            "reasons": "; ".join(record.get("reasons", [])),
            "exposure_area": metrics.get("exposure_area"),
            "exposure_count": metrics.get("exposure_count"),
            "exposure_score": metrics.get("exposure_score"),
            "nsfw_general_sum": metrics.get("nsfw_general_sum"),
            "placement_risk": metrics.get("placement_risk") or metrics.get("placement_risk_pre"),
            "winning_rule": (metrics.get("winning", {}) or {}).get("rule_id"),
            "wd14_rating_g": ratings.get("g"),
            "wd14_rating_s": ratings.get("s"),
            "wd14_rating_q": ratings.get("q"),
            "wd14_rating_e": ratings.get("e"),
        }
        for feature_name in col_features:
            row[feature_name] = feats.get(feature_name)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def render_detail_panel(record: dict | None, *, rules_path: Path) -> None:
    st.subheader("詳細")
    if not record:
        st.info("左の表で行を選択すると詳細を表示します。")
        return

    severity = record.get("severity") or "-"
    rule_id = record.get("rule_id") or "-"
    rule_title = record.get("rule_title")
    summary_parts = [f"Severity: {severity}", f"Rule: {rule_id}"]
    if rule_title:
        summary_parts.append(rule_title)
    st.markdown(" | ".join(summary_parts))

    reasons = record.get("reasons") or []
    if reasons:
        st.write("Reasons: " + "; ".join(reasons))

    message_link = record.get("message_link")
    if message_link:
        st.markdown(f"[Message Link]({message_link})")

    thumb_path = get_thumbnail_for_record(record)
    if thumb_path and thumb_path.exists():
        st.image(str(thumb_path), width="stretch")

    metrics = record.get("metrics", {}) or {}
    ratings = _extract_wd14_rating(record)
    nude = _extract_nudenet_metrics(record)

    rating_cols = st.columns(4)
    rating_labels = {"g": "wd14:g", "s": "wd14:s", "q": "wd14:q", "e": "wd14:e"}
    for col, key in zip(rating_cols, ("g", "s", "q", "e")):
        value = ratings.get(key)
        if value is None:
            col.metric(rating_labels[key], "-")
        else:
            col.metric(rating_labels[key], f"{value:.2f}")

    nude_cols = st.columns(3)
    nude_cols[0].metric("exposure_area", f"{nude['area']:.3f}")
    nude_cols[1].metric("exposure_count", str(nude["count"]))
    nude_cols[2].metric("exposure_score", f"{nude['score']:.3f}")

    if nude["by_class"]:
        with st.expander("NudeNet クラス別スコア", expanded=False):
            nude_df = pd.DataFrame(
                sorted(nude["by_class"].items(), key=lambda item: item[1], reverse=True),
                columns=["class", "score"],
            )
            st.dataframe(nude_df, width="stretch", hide_index=True)

    dsl = metrics.get("dsl", {}) or {}
    feature_values = (dsl.get("features", {}) or {})
    rules_data = _load_rules_for_detail(rules_path)
    group_hits = compute_group_hits(record, rules_data.get("groups") or {}, topk=256)
    winning_rule_id = (metrics.get("winning", {}) or {}).get("rule_id")
    if winning_rule_id:
        rule_entry = (rules_data.get("rules") or {}).get(winning_rule_id)
        if rule_entry and rule_entry.get("when"):
            with st.expander(f"ルール {winning_rule_id} の when 式", expanded=True):
                when_clause = rule_entry.get("when")
                st.code(when_clause, language="text")
                when_summary = _build_when_summary(
                    when_clause,
                    record,
                    feature_values,
                    rules_data.get("features") or {},
                )
                if when_summary:
                    summary_df = pd.DataFrame(when_summary)
                    if "result" in summary_df.columns:
                        summary_df["result"] = summary_df["result"].map(
                            lambda flag: "✅" if flag else ("❌" if flag is False else "-")
                        )
                    st.dataframe(summary_df, width="stretch", hide_index=True)

    if feature_values:
        st.subheader("features スコア")
        feature_defs = rules_data.get("features") or {}
        sorted_features = sorted(
            feature_values.items(),
            key=lambda item: (item[1] is None, -float(item[1]) if isinstance(item[1], (int, float)) else 0.0),
        )
        for name, value in sorted_features:
            label = _format_feature_value(name, value)
            related_groups = _related_groups_for_feature(name, feature_defs, group_hits)
            with st.expander(label, expanded=False):
                if related_groups:
                    for group_name in related_groups:
                        items = group_hits.get(group_name) or []
                        if not items:
                            continue
                        chart_data = pd.DataFrame(items, columns=["tag", "score"]).head(20)
                        chart = (
                            alt.Chart(chart_data)
                            .mark_bar()
                            .encode(
                                x=alt.X("score:Q", title="score"),
                                y=alt.Y("tag:N", sort="-x", title="tag"),
                                tooltip=["tag", alt.Tooltip("score:Q", format=".4f")],
                            )
                        )
                        st.altair_chart(chart, width="stretch")
                else:
                    st.caption("関連タグのヒットはありません。")
    else:
        st.info("metrics.dsl.features が見つかりません。v2 ルールで再スキャンすると詳細が表示されます。")


def _default_visible_columns(columns: list[str], preset: str) -> list[str]:
    preset_columns = COLUMN_PRESETS.get(preset)
    if preset_columns is None:
        return [col for col in columns if col != "thumbnail"]
    resolved = [col for col in preset_columns if col in columns]
    return resolved or [col for col in columns if col != "thumbnail"]


def _format_feature_value(name: str, value: Any) -> str:
    if isinstance(value, (int, float)):
        icon = "⚪"
        if value >= 0.9:
            icon = "🔴"
        elif value >= 0.7:
            icon = "🟠"
        return f"{icon} {name}: {value:.3f}"
    return f"⚪ {name}: {value}"


def _extract_wd14_rating(record: dict) -> dict[str, float | None]:
    metrics = record.get("metrics", {}) or {}
    potential_sources: list[dict[str, Any]] = []

    for candidate in (
        metrics.get("wd14_rating"),
        (metrics.get("wd14") or {}).get("rating") if isinstance(metrics.get("wd14"), dict) else None,
        (record.get("wd14") or {}).get("rating") if isinstance(record.get("wd14"), dict) else None,
    ):
        if isinstance(candidate, dict):
            potential_sources.append(candidate)

    key_aliases = {
        "g": ["g", "general", "general_rating"],
        "s": ["s", "sensitive", "sensitive_rating"],
        "q": ["q", "questionable", "rating_questionable"],
        "e": ["e", "explicit", "rating_explicit"],
    }

    result: dict[str, float | None] = {}
    for target, aliases in key_aliases.items():
        value: float | None = None
        for source in potential_sources:
            for alias in aliases:
                if alias in source:
                    value = _safe_float(source.get(alias))
                    if value is not None:
                        break
            if value is not None:
                break
        if value is None:
            for alias in aliases:
                candidate = metrics.get(alias)
                value = _safe_float(candidate)
                if value is not None:
                    break
        result[target] = value
    return result


def _extract_nudenet_metrics(record: dict) -> dict[str, Any]:
    metrics = record.get("metrics", {}) or {}
    result: dict[str, Any] = {
        "area": float(metrics.get("exposure_area") or 0.0),
        "count": int(metrics.get("exposure_count") or 0),
        "score": float(metrics.get("exposure_score") or 0.0),
        "by_class": {},
    }
    nudenet = metrics.get("nudenet") or {}
    exposure_scores = nudenet.get("exposure_scores") or {}
    if isinstance(exposure_scores, dict):
        result["by_class"] = {
            name: float(score)
            for name, score in exposure_scores.items()
            if isinstance(score, (int, float)) and score > 0
        }
    return result


def _load_rules_for_detail(rules_path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(Path(rules_path).read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}
    groups = data.get("groups") if isinstance(data.get("groups"), dict) else {}
    raw_features = data.get("features") if isinstance(data.get("features"), dict) else {}
    rules_map: dict[str, dict[str, Any]] = {}
    for rule in data.get("rules", []) or []:
        if not isinstance(rule, dict):
            continue
        rule_id = rule.get("id")
        if not isinstance(rule_id, str):
            continue
        rules_map[rule_id] = {
            "when": rule.get("when"),
            "severity": rule.get("severity"),
        }
    return {"groups": groups, "features": raw_features, "rules": rules_map}


COMPARISON_PATTERN = re.compile(r"([A-Za-z0-9_.]+)\s*(>=|<=|>|<|==)\s*(T_[A-Za-z0-9_]+|\d+(?:\.\d+)?)")
GROUP_TOKEN_PATTERN = re.compile(r"'([^']+)'")


def _build_when_summary(
    when_clause: str,
    record: dict,
    feature_values: dict[str, Any],
    feature_defs: dict[str, Any],
) -> list[dict[str, Any]]:
    if not when_clause:
        return []
    numeric_values: dict[str, float] = {}
    for name, value in feature_values.items():
        numeric = _safe_float(value)
        if numeric is not None:
            numeric_values[name] = numeric
    metrics = record.get("metrics", {}) or {}
    for name, value in metrics.items():
        numeric = _safe_float(value)
        if numeric is not None:
            numeric_values.setdefault(name, numeric)

    threshold_values: dict[str, float] = {}
    for name, value in feature_defs.items():
        numeric = _safe_float(value)
        if numeric is not None:
            threshold_values[name] = numeric

    rows: list[dict[str, Any]] = []
    for lhs, operator, rhs in COMPARISON_PATTERN.findall(when_clause):
        metric_name = lhs.split(".")[-1]
        current_value = numeric_values.get(metric_name)
        threshold_value: float | None
        threshold_label = rhs
        if rhs.startswith("T_"):
            threshold_value = threshold_values.get(rhs)
        else:
            threshold_value = _safe_float(rhs)
        result = None
        if current_value is not None and threshold_value is not None:
            result = _evaluate_comparison(current_value, operator, threshold_value)
        rows.append(
            {
                "metric": metric_name,
                "value": current_value,
                "operator": operator,
                "threshold_label": threshold_label,
                "threshold": threshold_value,
                "result": result,
            }
        )
    return rows


def _evaluate_comparison(value: float, operator: str, threshold: float) -> bool:
    if operator == ">=":
        return value >= threshold
    if operator == ">":
        return value > threshold
    if operator == "<=":
        return value <= threshold
    if operator == "<":
        return value < threshold
    if operator == "==":
        return value == threshold
    return False


def compute_group_hits(
    record: dict,
    groups: dict[str, list[str]],
    *,
    topk: int = 256,
) -> dict[str, list[tuple[str, float]]]:
    wd14 = record.get("wd14") or {}
    pairs: list[tuple[str, float]] = []
    general_topk = wd14.get("general_topk")
    if isinstance(general_topk, list):
        for item in general_topk:
            if isinstance(item, dict):
                tag = item.get("name")
                score = _safe_float(item.get("score"))
                if isinstance(tag, str) and score is not None:
                    pairs.append((tag, score))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                tag = item[0]
                score = _safe_float(item[1])
                if isinstance(tag, str) and score is not None:
                    pairs.append((tag, score))
    if not pairs:
        general_raw = wd14.get("general_raw")
        if isinstance(general_raw, list):
            for tag, score in general_raw[:topk]:
                value = _safe_float(score)
                if isinstance(tag, str) and value is not None:
                    pairs.append((tag, value))
    if not pairs:
        return {}
    pairs = sorted(pairs, key=lambda item: item[1], reverse=True)[:topk]
    scores = {tag: score for tag, score in pairs}
    hits: dict[str, list[tuple[str, float]]] = {}
    for group_name, group_tags in (groups or {}).items():
        if not isinstance(group_tags, list):
            continue
        matched = [(tag, scores[tag]) for tag in group_tags if tag in scores]
        if matched:
            hits[group_name] = sorted(matched, key=lambda item: item[1], reverse=True)
    return hits


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _related_groups_for_feature(
    feature_name: str,
    feature_defs: dict[str, Any],
    group_hits: dict[str, list[tuple[str, float]]],
) -> list[str]:
    expr = feature_defs.get(feature_name)
    if expr is None:
        return []
    if isinstance(expr, str):
        text = expr
    else:
        try:
            text = json.dumps(expr, ensure_ascii=False)
        except TypeError:
            text = str(expr)
    related: list[str] = []
    seen: set[str] = set()
    for token in GROUP_TOKEN_PATTERN.findall(text or ""):
        if token in group_hits and token not in seen:
            seen.add(token)
            related.append(token)
    return related


def _discover_feature_ids(rules_path: Path | None) -> list[str]:
    if rules_path is None:
        return []
    try:
        data = yaml.safe_load(Path(rules_path).read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return []
    features = data.get("features")
    found: list[str] = []
    if isinstance(features, dict):
        for key in features.keys():
            if isinstance(key, str) and not key.startswith("T_"):
                found.append(key)
    elif isinstance(features, list):
        for item in features:
            if isinstance(item, dict):
                identifier = item.get("id")
                if isinstance(identifier, str) and not identifier.startswith("T_"):
                    found.append(identifier)
    return found


def _severity_sort_key(series: pd.Series) -> pd.Series:
    order = {sev: idx for idx, sev in enumerate(SEVERITY_ORDER)}
    return series.map(lambda sev: order.get(sev, len(order)))


def _iter_jsonl(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            if not line.strip():
                continue
            yield json.loads(line)


def _attachment_urls(record: dict) -> Iterable[str]:
    for message in record.get("messages", []) or []:
        attachments = message.get("attachments") or []
        for attachment in attachments:
            url = attachment.get("url")
            if url:
                yield url


if __name__ == "__main__":
    main()
