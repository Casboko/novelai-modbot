from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import altair as alt
import pandas as pd
import streamlit as st
import yaml
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# このファイルが置かれているディレクトリ（tools/scl_viewer）が先頭に
# 残っていると、`import app` が同名ファイルを再帰的に読み込み続ける。
# ルートパスを追加済みなので、衝突を避けるために該当エントリを除外する。
SCRIPT_DIR = Path(__file__).resolve().parent
script_dir_candidates = {str(SCRIPT_DIR), str(SCRIPT_DIR.resolve())}
for idx, entry in list(enumerate(sys.path)):
    normalized = entry if entry else "."
    try:
        resolved = str(Path(normalized).resolve())
    except (OSError, RuntimeError):
        continue
    if resolved in script_dir_candidates:
        del sys.path[idx]

# Streamlit からこのファイルを実行すると、モジュール名 `app` として
# `sys.modules` に登録されるため、ルートパッケージ `app` と衝突する。
# 衝突が発生している場合のみエイリアスを付け替え、以降の
# `from app import ...` が本来のパッケージを参照できるようにする。
maybe_script_module = sys.modules.get("app")
if maybe_script_module is not None:
    module_path = getattr(maybe_script_module, "__file__", None)
    if module_path and Path(module_path).resolve() == Path(__file__).resolve():
        sys.modules.setdefault("tools.scl_viewer.app", maybe_script_module)
        sys.modules.pop("app", None)

from app.batch_loader import ImageRequest, load_images
from app.config import get_settings
from app.local_cache import resolve_local_file
from app.profiles import PartitionPaths
from tools.scl_viewer import utils

TMP_DIR = Path("tools/scl_viewer/tmp")
THUMBS_DIR = Path(os.getenv("SCL_CACHE_THUMBS", "tools/scl_viewer/thumbs"))
CACHE_ROOT = Path(os.getenv("CACHE_ROOT", "cache"))
CACHE_DIR = Path(os.getenv("SCL_CACHE_FULL", CACHE_ROOT / "imgs"))
SCL_NO_FETCH = os.getenv("SCL_NO_FETCH", "0") == "1"

PLACEHOLDER_THUMB = THUMBS_DIR / "_placeholder.jpg"

FINDINGS_PATH = TMP_DIR / "findings.jsonl"
REPORT_PATH = TMP_DIR / "report.csv"
RULES_COMPILED_PATH = TMP_DIR / "rules_compiled.yaml"
P0_MERGED_PATH = TMP_DIR / "p0_merged.csv"
AB_DIR = TMP_DIR / "ab"

SETTINGS = get_settings()
BASE_CONTEXT = SETTINGS.build_profile_context()

SEVERITY_ORDER = ["red", "orange", "yellow", "green"]
SEVERITY_BADGES = {
    "red": "🟥 RED",
    "orange": "🟧 ORANGE",
    "yellow": "🟨 YELLOW",
    "green": "🟩 GREEN",
}

LAYOUT_MODES = ("classic", "stacked")
DEFAULT_LAYOUT_MODE = "stacked"
DEFAULT_LIST_HEIGHT_PX = 216
CHART_TOPK_MAX = 10

CATEGORY_GROUPS_DEFAULT: dict[str, set[str]] = {
    "minor": {
        "minor_peak",
        "minor_body_high",
        "minor_body_medium",
        "minor_body_low",
        "minor_face_like",
        "minor_context",
        "minor_uniforms",
        "minor_school_context",
        "minor_tags",
        "underage_context",
        "youth_context",
    },
    "animal": {
        "animal_subjects",
        "animal_context",
        "bestiality_direct",
        "bestiality_context",
        "beastlike_traits",
        "feral_subjects",
    },
    "nonconsent": {
        "coercion_main",
        "coercion_context",
        "rape_tags",
        "abuse_tags",
        "unconscious_states",
        "restraint_gear",
        "bondage_context",
        "pain_expressions",
    },
    "gore": {
        "gore_peak",
        "gore_context",
        "blood_tags",
        "injury_tags",
        "wound_exposed",
        "mutilation_tags",
        "death_markers",
        "shock_tags",
    },
}

CATEGORY_PREFIXES: dict[str, tuple[str, ...]] = {
    "minor": ("minor_", "underage_", "youth_"),
    "animal": ("animal_", "bestiality_", "beast_", "feral_"),
    "nonconsent": ("coercion_", "rape_", "abuse_", "nonconsent_", "restraint_", "pain_", "kidnap_"),
    "gore": ("gore_", "injury_", "blood_", "wound_", "mutilation_", "death_", "shock_"),
}

CATEGORY_LABELS: dict[str, str] = {
    "minor": "マイナー系",
    "animal": "アニマル系",
    "nonconsent": "非合意系",
    "gore": "ゴア・ショッキング系",
}


def _cache_key_for_path(path: Path) -> tuple[str, float]:
    try:
        return str(path), path.stat().st_mtime
    except FileNotFoundError:
        return str(path), 0.0


@st.cache_data(show_spinner=False)
def _load_findings_cached(
    findings_path: str,
    findings_mtime: float,
    rules_path: str | None,
    rules_mtime: float,
) -> tuple[list[dict], pd.DataFrame]:
    path = Path(findings_path)
    records = utils.read_jsonl(path)
    rules = Path(rules_path) if rules_path else None
    df = _build_findings_dataframe(records, rules_path=rules)
    return records, df


def _fmt_time_iso_to_min(iso_like: object) -> str | None:
    if not iso_like:
        return None
    value = str(iso_like)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%d %H:%M")


def _short_id(value: object, prefix: str) -> str:
    if value is None:
        return "-"
    text = str(value)
    if not text:
        return "-"
    return f"{prefix}{text[-6:]}"


def _count_attachments(record: Mapping[str, Any]) -> int:
    total = 0
    for message in record.get("messages", []) or []:
        if not isinstance(message, Mapping):
            continue
        attachments = message.get("attachments") or []
        total += len(attachments)
    return total


def _dedup_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def resolve_category_groups(available_groups: set[str]) -> dict[str, set[str]]:
    resolved: dict[str, set[str]] = {}
    for category, candidates in CATEGORY_GROUPS_DEFAULT.items():
        resolved[category] = {candidate for candidate in candidates if candidate in available_groups}
    return resolved


def _match_category_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def _category_feature_entries(
    feature_values: Mapping[str, Any],
    prefixes: tuple[str, ...],
    limit: int = 6,
) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    for key, value in feature_values.items():
        numeric = _safe_float(value)
        if numeric is None or numeric <= 0:
            continue
        if _match_category_prefix(key, prefixes):
            items.append((key, numeric))
    items.sort(key=lambda item: item[1], reverse=True)
    return items[:limit]


def _category_topk_dataframe(
    group_hits: Mapping[str, list[tuple[str, float]]],
    explicit_groups: set[str],
    prefixes: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, hits in (group_hits or {}).items():
        if group_name in explicit_groups or _match_category_prefix(group_name, prefixes):
            for tag, score in hits:
                numeric = _safe_float(score)
                if numeric is None:
                    continue
                rows.append({"group": group_name, "tag": str(tag), "score": numeric})
    if not rows:
        return pd.DataFrame(columns=["tag", "score"])
    df = pd.DataFrame(rows)
    df = df.sort_values("score", ascending=False).head(CHART_TOPK_MAX)
    return df[["tag", "score"]]


def _render_topk_chart(df: pd.DataFrame) -> None:
    if df.empty:
        st.caption("関連タグのヒットはありません。")
        return
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("score:Q", title="score"),
            y=alt.Y("tag:N", sort="-x", title="tag"),
            tooltip=[alt.Tooltip("tag:N"), alt.Tooltip("score:Q", format=".3f")],
        )
    )
    st.altair_chart(chart, use_container_width=True)


def summarize_reasons(record: Mapping[str, Any]) -> tuple[str, list[str]]:
    raw_reasons = record.get("reasons") or []
    processed: list[str] = []
    seen: set[str] = set()
    for reason in raw_reasons:
        text = _format_reason_text(str(reason))
        if not text or text in seen:
            continue
        seen.add(text)
        processed.append(text)
        if len(processed) >= 5:
            break
    if not processed:
        return "—", []
    return processed[0], processed


def _format_reason_text(text: str) -> str:
    cleaned = text.strip().lstrip("-•")
    if not cleaned:
        return ""
    cleaned = cleaned.replace("_", " ")

    def _repl(match: re.Match[str]) -> str:
        value = _safe_float(match.group(0))
        if value is None:
            return match.group(0)
        return f"{value:.3f}"

    formatted = REASON_NUMBER_PATTERN.sub(_repl, cleaned)
    return formatted


def _ensure_placeholder_thumb() -> Path:
    PLACEHOLDER_THUMB.parent.mkdir(parents=True, exist_ok=True)
    if PLACEHOLDER_THUMB.exists():
        return PLACEHOLDER_THUMB
    image = Image.new("RGB", (256, 256), (230, 230, 230))
    draw = ImageDraw.Draw(image)
    draw.text((40, 116), "no image", fill=(120, 120, 120))
    image.save(PLACEHOLDER_THUMB, format="JPEG", quality=85)
    return PLACEHOLDER_THUMB
DEFAULT_RULES = Path("configs/rules_v2.yaml")
DEFAULT_THRESHOLDS = Path("configs/scl/scl_thresholds.yaml")
DEFAULT_RULES_PREV = Path("configs/rules_v2_prev.yaml")

COLUMN_PRESETS: dict[str, list[str] | None] = {
    "Core": [
        "sev_badge",
        "severity",
        "rule_id",
        "wd14_rating_e",
        "exposure_area",
        "message_time",
        "author_short",
        "channel_short",
        "is_nsfw_channel",
        "message_link",
    ],
    "Moderation Core": [
        "sev_badge",
        "phash",
        "severity",
        "rule_id",
        "rule_title",
        "rating_explicit",
        "rating_questionable",
        "qe_margin",
        "wd14_rating_g",
        "wd14_rating_s",
        "wd14_rating_q",
        "wd14_rating_e",
        "exposure_area",
        "exposure_count",
        "exposure_score",
        "coercion_main_peak",
        "sec_peak",
        "sub_cnt",
        "rest_topk",
        "coercion_final_score",
        "bestiality_peak",
        "animal_presence",
        "sexual_main",
        "is_explicit_exposed",
        "is_nsfw_channel",
        "gore_main_peak",
        "gore_density_cnt",
        "injury_topk",
        "blood_topk",
        "gore_base_score",
        "gore_offset",
        "gore_final_score",
        "coercion_score",
    ],
    "v2 Features": [
        "sev_badge",
        "phash",
        "severity",
        "rule_id",
        "rule_title",
        "rating_explicit",
        "rating_questionable",
        "qe_margin",
        "explicit_score",
        "sexual_intensity",
        "nsfw_rating_max",
        "is_E",
        "is_Q",
        "minor_peak_conf",
        "minor_main_peak",
        "minor_sit_high_peak",
        "minor_sit_other_peak",
        "minor_suspect_score",
        "sexual_main",
        "exposure_prominent",
        "is_explicit_exposed",
        "bestiality_peak",
        "animal_presence",
        "gore_main_peak",
        "gore_density_cnt",
        "injury_topk",
        "blood_topk",
        "gore_base_score",
    "gore_offset",
    "gore_final_score",
    "coercion_main_peak",
    "sec_peak",
    "sub_cnt",
    "rest_topk",
    "coercion_final_score",
    "coercion_score",
    "qe_margin",
    ],
    "All": None,
}


FEATURE_FALLBACKS: dict[str, tuple[str, ...]] = {
    "is_E": ("is_E", "rating_is_explicit"),
    "is_Q": ("is_Q", "rating_is_questionable"),
    "minor_main_peak": ("minor_main_peak", "minor_peak_conf"),
    "minor_sit_high_peak": ("minor_sit_high_peak", "minor_context_score"),
    "minor_sit_other_peak": ("minor_sit_other_peak", "minor_context_score"),
    "sexual_main": ("sexual_main", "sexual_intensity"),
    "exposure_prominent": ("exposure_prominent",),
    "minor_suspect_score": ("minor_suspect_score", "minor_body_score"),
    "is_explicit_exposed": ("is_explicit_exposed",),
    "coercion_main_peak": ("coercion_main_peak",),
    "sec_peak": ("sec_peak",),
    "sub_cnt": ("sub_cnt",),
    "rest_topk": ("rest_topk",),
    "coercion_final_score": ("coercion_final_score", "coercion_score"),
    "bestiality_peak": ("bestiality_peak", "bestiality"),
    "animal_presence": ("animal_presence", "animal_presence_peak"),
    "gore_main_peak": ("gore_main_peak", "gore_peak_conf"),
    "gore_density_cnt": ("gore_density_cnt", "gore_density"),
    "injury_topk": ("injury_topk",),
    "blood_topk": ("blood_topk",),
    "gore_base_score": ("gore_base_score", "gore_intensity"),
    "gore_offset": ("gore_offset",),
    "gore_final_score": ("gore_final_score", "gore_intensity"),
}


def _feature_value_with_fallback(feats: Mapping[str, Any], name: str) -> Any:
    for candidate in FEATURE_FALLBACKS.get(name, (name,)):
        value = feats.get(candidate)
        if value is not None:
            return value
    return feats.get(name)


def _trigger_run(source: str = "main") -> None:
    st.session_state["__run_trigger_ts__"] = time.time()
    st.session_state["__run_source__"] = source


def should_run(state: SidebarState) -> bool:
    run_clicked = getattr(state, "run_clicked", False)
    if run_clicked:
        _trigger_run("sidebar")
        try:
            state.run_clicked = False
        except Exception:
            pass

    trigger_ts = float(st.session_state.get("__run_trigger_ts__", 0.0))
    handled_ts = float(st.session_state.get("__run_handled_ts__", 0.0))
    return trigger_ts > handled_ts


def mark_run_handled() -> None:
    st.session_state["__run_handled_ts__"] = float(
        st.session_state.get("__run_trigger_ts__", 0.0)
    )


def _get_state_value(
    state: SidebarState,
    *keys: str,
    default: str = "-",
) -> str:
    for key in keys:
        value = getattr(state, key, None)
        if value:
            return str(value)
    for key in keys:
        value = st.session_state.get(key)
        if value:
            return str(value)
    for key in keys:
        value = os.environ.get(key)
        if value:
            return str(value)
    return default


def _update_selection_from_event(event, df_for_selection: pd.DataFrame) -> None:
    rows = getattr(getattr(event, "selection", None), "rows", []) or []
    if rows:
        idx = int(rows[0])
        if 0 <= idx < len(df_for_selection):
            st.session_state["last_selected_phash"] = df_for_selection.loc[idx, "phash"]
    elif df_for_selection.empty:
        st.session_state["last_selected_phash"] = None


def _get_selected_record(records: list[dict]) -> dict | None:
    phash = st.session_state.get("last_selected_phash")
    if phash is None:
        return None
    return next((rec for rec in records if rec.get("phash") == phash), None)


def _where_list(layout_mode: str) -> str:
    return "上段" if layout_mode == "stacked" else "左側"


def _where_detail(layout_mode: str) -> str:
    return "下段" if layout_mode == "stacked" else "右側"


def main() -> None:
    st.set_page_config(page_title="SCL Viewer", layout="wide")
    st.title("SCL ルールチューニングビューア")
    ensure_directories()

    sidebar_state = render_sidebar()

    current_layout = st.session_state.get("layout_mode", DEFAULT_LAYOUT_MODE)
    try:
        layout_index = list(LAYOUT_MODES).index(current_layout)
    except ValueError:
        layout_index = list(LAYOUT_MODES).index(DEFAULT_LAYOUT_MODE)
    layout_mode = st.radio(
        "Layout",
        options=list(LAYOUT_MODES),
        index=layout_index,
        horizontal=True,
        key="layout_mode",
    )

    action_placeholder = st.empty()

    if should_run(sidebar_state):
        try:
            handle_run(action_placeholder, sidebar_state)
        finally:
            mark_run_handled()
    if sidebar_state.report_clicked:
        handle_report(action_placeholder, sidebar_state)
    if sidebar_state.ab_clicked:
        handle_ab(action_placeholder, sidebar_state)
    if sidebar_state.contract_clicked:
        handle_contract(action_placeholder, sidebar_state)

    findings_records = st.session_state.get("findings_records")
    if findings_records:
        render_findings(findings_records, sidebar_state, layout_mode)
    else:
        st.info("Run を実行すると最新の findings を表示します。")

    if "report_df" in st.session_state:
        st.subheader("p3 レポート (CSV)")
        st.dataframe(st.session_state["report_df"], width="stretch")

    if "ab_compare" in st.session_state:
        render_ab_outputs()
@dataclass
class SidebarState:
    profile: str
    date: str
    context_result: utils.ContextResolveResult
    context_notice: str
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

        base_profile = BASE_CONTEXT.profile
        profiles = utils.available_profiles(base_profile)
        profile_index = profiles.index(base_profile) if base_profile in profiles else 0
        profile = st.selectbox("Profile", options=profiles, index=profile_index)

        partition_dates = utils.available_partition_dates(profile)
        default_date = partition_dates[0] if partition_dates else SETTINGS.build_profile_context(profile=profile).iso_date
        date_value = st.text_input(
            "Partition Date",
            value=default_date,
            help="YYYY-MM-DD / today / yesterday。空欄で既定値",
        ).strip()
        context_result = utils.build_context(profile=profile, date=date_value or None)
        context_notice = utils.format_context_notice(context_result)
        st.caption(context_notice)

        rules_path = Path(
            st.text_input("Rules (B)", value=str(DEFAULT_RULES), help="チューニング対象のルール定義")
        ).expanduser()
        thresholds_value = st.text_input(
            "閾値ファイル", value=str(DEFAULT_THRESHOLDS), help="configs/scl/scl_thresholds.yaml"
        )
        thresholds_path = Path(thresholds_value).expanduser()
        if not thresholds_path.exists():
            thresholds_path = None
        default_analysis = context_result.paths.stage_file("p2")
        analysis_input = st.text_input(
            "Analysis JSONL",
            value=str(default_analysis),
            help="プロファイルの p2 JSONL。必要に応じて上書き",
        ).strip()
        analysis_path = Path(analysis_input).expanduser() if analysis_input else default_analysis
        rules_prev_input = st.text_input(
            "Rules (A)", value=str(DEFAULT_RULES_PREV), help="比較用の旧ルール"
        )
        rules_prev_path = Path(rules_prev_input).expanduser() if rules_prev_input else None
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
        run_clicked = st.button("Run (cli_scan)", key="run_btn_sidebar")
        report_clicked = st.button("Report (cli_report)")
        ab_clicked = st.button("A/B diff (cli_rules_ab)")
        contract_clicked = st.button("契約チェック (cli_contract)")

    return SidebarState(
        profile=profile,
        date=date_value or None,
        context_result=context_result,
        context_notice=context_notice,
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
            partitions = PartitionPaths(state.context_result.context)
            merged = utils.merge_p0_sources(P0_MERGED_PATH, partitions=partitions)
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
        if state.thresholds_path is not None:
            extra_args.extend(["--const", str(state.thresholds_path)])
        if merged is None:
            p0_path = None
        else:
            p0_path = merged
        if not state.analysis_path.exists():
            st.error(f"Analysis path not found: {state.analysis_path}")
            return
        try:
            result = utils.run_cli_scan(
                analysis=state.analysis_path,
                findings=FINDINGS_PATH,
                rules=compiled.rules_path,
                p0=p0_path,
                limit=state.limit,
                offset=state.offset,
                profile=state.context_result.context.profile,
                date=state.date,
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
        st.session_state["context_notice"] = state.context_notice
        st.cache_data.clear()
        load_findings_into_state(state.context_result)


def handle_report(placeholder, state: SidebarState) -> None:
    with placeholder.container():
        st.write("### Report 実行ログ")
        if not FINDINGS_PATH.exists():
            st.warning("先に Run を実行し findings を生成してください。")
            return
        try:
            result = utils.run_cli_report(
                findings=FINDINGS_PATH,
                out_path=REPORT_PATH,
                profile=state.context_result.context.profile,
                date=state.date,
            )
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
        extra_args = ["--samples-minimal", "--samples-redact-urls"]
        if state.thresholds_path is not None:
            extra_args.extend(["--constB", str(state.thresholds_path)])
        try:
            result = utils.run_cli_ab(
                analysis=state.analysis_path,
                rules_a=state.rules_prev_path,
                rules_b=compiled.rules_path,
                out_dir=AB_DIR,
                sample_diff=200,
                profile=state.context_result.context.profile,
                date=state.date,
                extra_args=extra_args,
            )
        except utils.CliCommandError as exc:
            st.error(str(exc))
            st.code(exc.stderr or exc.stdout)
            return
        st.success("cli_rules_ab 完了")
        st.code(result.stdout)
        load_ab_outputs()


def handle_contract(placeholder, state: SidebarState) -> None:
    with placeholder.container():
        st.write("### 契約チェック")
        if not REPORT_PATH.exists():
            st.warning("レポート CSV が見つかりません。先に Report を実行してください。")
            return
        try:
            result = utils.run_cli_contract_report(
                report_path=REPORT_PATH,
                profile=state.context_result.context.profile,
                date=state.date,
            )
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


def load_findings_into_state(context_result: utils.ContextResolveResult) -> None:
    rules_source = st.session_state.get("rules_effective_path")
    effective_rules_path = Path(rules_source) if rules_source else DEFAULT_RULES
    findings_key = _cache_key_for_path(FINDINGS_PATH)
    rules_key = _cache_key_for_path(effective_rules_path)
    try:
        records, df = _load_findings_cached(
            findings_key[0],
            findings_key[1],
            str(effective_rules_path) if effective_rules_path else None,
            rules_key[1],
        )
    except utils.UtilsError as exc:
        st.error(str(exc))
        return
    st.session_state["findings_records"] = records
    st.session_state["findings_df"] = df
    st.session_state["context_notice"] = utils.format_context_notice(context_result)


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


def render_findings(records: list[dict], state: SidebarState, layout_mode: str) -> None:
    rules_source = st.session_state.get("rules_effective_path")
    rules_path = Path(rules_source) if rules_source else state.rules_path

    context_notice = st.session_state.get("context_notice", state.context_notice)
    if context_notice:
        st.caption(context_notice)

    df = st.session_state.get("findings_df")
    if df is None:
        df = _build_findings_dataframe(records, rules_path=rules_path)
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
    if "severity" in filtered.columns and "sev_badge" not in filtered.columns:
        filtered.insert(
            0,
            "sev_badge",
            filtered["severity"].map(lambda sev: SEVERITY_BADGES.get(str(sev), "⬜ -")),
        )

    st.session_state["findings_df"] = df

    if "list_height_px" not in st.session_state:
        st.session_state["list_height_px"] = DEFAULT_LIST_HEIGHT_PX

    mode = layout_mode if layout_mode in LAYOUT_MODES else DEFAULT_LAYOUT_MODE

    if mode == "stacked":
        table_container = st.container()
        detail_container = st.container()
    else:
        left_col, right_col = st.columns([3, 7], gap="large")
        table_container = left_col
        detail_container = right_col
    filtered_reset = filtered.reset_index(drop=True)

    def render_table_area() -> None:
        loc_table = _where_list(mode)
        loc_detail = _where_detail(mode)

        st.subheader("Findings")
        st.session_state["list_height_px"] = DEFAULT_LIST_HEIGHT_PX

        available_columns = list(filtered.columns)
        df_display = filtered_reset[available_columns].copy()

        table_column_config: dict[str, Any] = {}
        if "message_link" in df_display.columns:
            table_column_config["message_link"] = st.column_config.LinkColumn("link")

        event = st.dataframe(
            df_display,
            key="findings_table",
            width="stretch",
            column_config=table_column_config,
            on_select="rerun",
            selection_mode="single-row",
            height=st.session_state["list_height_px"],
            row_height=32,
        )

        _update_selection_from_event(event, filtered_reset)

        st.caption(
            f"💡 行をクリックすると{loc_detail}に詳細を表示します。列をソートすると選択はクリアされますが、{loc_detail}は直前の表示を維持します。"
        )

    if mode == "stacked":
        with table_container:
            with st.container(border=True):
                render_table_area()
        st.divider()
        with detail_container:
            with st.container(border=True):
                render_detail_panel(
                    _get_selected_record(records),
                    rules_path=rules_path,
                    layout_mode=mode,
                )
    else:
        with table_container:
            render_table_area()
        with detail_container:
            render_detail_panel(
                _get_selected_record(records),
                rules_path=rules_path,
                layout_mode=mode,
            )
def get_thumbnail_for_record(record: dict) -> Path:
    phash = record.get("phash")
    if not phash:
        return _ensure_placeholder_thumb()

    local = resolve_local_file(str(phash))
    image_path = Path(local) if local else (CACHE_DIR / f"{phash}.jpg")

    if not image_path.exists():
        if SCL_NO_FETCH:
            return _ensure_placeholder_thumb()
        urls = list(_attachment_urls(record))
        if not urls:
            return _ensure_placeholder_thumb()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        downloaded = download_image(phash, urls, image_path)
        if downloaded is None:
            return _ensure_placeholder_thumb()

    thumb_path = THUMBS_DIR / f"{phash}.jpg"
    try:
        if (not thumb_path.exists()) or (thumb_path.stat().st_mtime < image_path.stat().st_mtime):
            utils.ensure_thumbnail(image_path, thumb_path)
        return thumb_path
    except Exception:
        return _ensure_placeholder_thumb()


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
    _ensure_placeholder_thumb()


def _build_findings_dataframe(records: list[dict], *, rules_path: Path | None = None) -> pd.DataFrame:
    feature_ids = _discover_feature_ids(rules_path) if rules_path else []
    default_features = [
        "nsfw_rating_max",
        "explicit_score",
        "sexual_intensity",
        "is_E",
        "is_Q",
        "minor_peak_conf",
        "minor_main_peak",
        "minor_sit_high_peak",
        "minor_sit_other_peak",
        "minor_suspect_score",
        "sexual_main",
        "exposure_prominent",
        "is_explicit_exposed",
        "bestiality_peak",
        "animal_presence",
        "gore_main_peak",
        "gore_density_cnt",
        "injury_topk",
        "blood_topk",
        "gore_base_score",
        "gore_offset",
        "gore_final_score",
        "coercion_main_peak",
        "sec_peak",
        "sub_cnt",
        "rest_topk",
        "coercion_final_score",
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
        ratings = _extract_wd14_rating(record)
        qe_margin_value = metrics.get("qe_margin")
        if qe_margin_value is None and isinstance(feats, Mapping):
            qe_margin_value = feats.get("qe_margin")
        is_nsfw_flag = record.get("is_nsfw_channel")
        if is_nsfw_flag is None:
            channel_info = record.get("channel") or {}
            is_nsfw_flag = channel_info.get("is_nsfw")
        row = {
            "phash": record.get("phash"),
            "severity": record.get("severity"),
            "rule_id": record.get("rule_id"),
            "rule_title": record.get("rule_title"),
            "message_link": record.get("message_link"),
            "reasons": "; ".join(record.get("reasons", [])),
            "rating_questionable": ratings.get("q"),
            "rating_explicit": ratings.get("e"),
            "qe_margin": qe_margin_value,
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
            "is_nsfw_channel": None if is_nsfw_flag is None else bool(is_nsfw_flag),
        }
        row.update(
            {
                "message_time": _fmt_time_iso_to_min(record.get("created_at")),
                "author_short": _short_id(record.get("author_id"), "a"),
                "channel_short": _short_id(record.get("channel_id"), "c"),
                "att_count": _count_attachments(record),
            }
        )
        for feature_name in col_features:
            row[feature_name] = _feature_value_with_fallback(feats, feature_name)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def render_detail_panel(
    record: dict | None,
    *,
    rules_path: Path,
    layout_mode: str = DEFAULT_LAYOUT_MODE,
) -> None:
    st.subheader("詳細")
    mode = layout_mode if layout_mode in LAYOUT_MODES else DEFAULT_LAYOUT_MODE
    if not record:
        st.info(f"{_where_list(mode)}の一覧で行を選択すると詳細を表示します。")
        return

    metrics = record.get("metrics", {}) or {}
    ratings = _extract_wd14_rating(record)
    dsl_block = (metrics.get("dsl") or {}) if isinstance(metrics.get("dsl"), Mapping) else {}
    feature_values = (
        (dsl_block.get("features") or {}) if isinstance(dsl_block, Mapping) else {}
    ) or {}

    severity_key = str(record.get("severity") or "").lower()
    sev_badge = SEVERITY_BADGES.get(severity_key, "⬜ -")
    rule_title = record.get("rule_title") or "-"
    rule_id = record.get("rule_id") or "-"
    summary_text, reason_lines = summarize_reasons(record)

    is_nsfw_flag = record.get("is_nsfw_channel")
    if is_nsfw_flag is None:
        channel_info = record.get("channel") or {}
        is_nsfw_flag = channel_info.get("is_nsfw")

    if isinstance(is_nsfw_flag, bool):
        nsfw_value: bool | None = is_nsfw_flag
    elif isinstance(is_nsfw_flag, str):
        lowered = is_nsfw_flag.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            nsfw_value = True
        elif lowered in {"false", "0", "no", "n"}:
            nsfw_value = False
        else:
            nsfw_value = None
    elif is_nsfw_flag is None:
        nsfw_value = None
    else:
        nsfw_value = bool(is_nsfw_flag)

    q_value = _safe_float(ratings.get("q"))
    e_value = _safe_float(ratings.get("e"))
    qe_margin_value = _safe_float(metrics.get("qe_margin"))
    if qe_margin_value is None:
        qe_margin_value = _safe_float(feature_values.get("qe_margin"))
    if qe_margin_value is None and q_value is not None and e_value is not None:
        qe_margin_value = max(0.0, e_value - q_value)

    exposure_area = _safe_float(metrics.get("exposure_area"))
    exposure_count_raw = metrics.get("exposure_count")
    exposure_count = (
        exposure_count_raw
        if isinstance(exposure_count_raw, int)
        else int(_safe_float(exposure_count_raw) or 0)
    )
    exposure_score = _safe_float(metrics.get("exposure_score"))
    nsfw_general_sum = _safe_float(metrics.get("nsfw_general_sum"))
    placement_risk = _safe_float(
        metrics.get("placement_risk") or metrics.get("placement_risk_pre")
    )

    def fmt_float(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f}"

    def fmt_bool(value: bool | None) -> str:
        if value is None:
            return "—"
        return "true" if bool(value) else "false"

    thumb_col, info_col = st.columns([1, 2], gap="large")

    with thumb_col:
        thumb_path = get_thumbnail_for_record(record)
        if thumb_path.exists():
            st.image(str(thumb_path), width=260)
        else:
            st.image(str(thumb_path))

    with info_col:
        message_link = record.get("message_link")
        if message_link:
            st.markdown(f"[message_link]({message_link})")
        header_line = f"**{sev_badge}** · **{rule_title}**"
        st.markdown(header_line)
        if rule_id and rule_id != "-":
            st.caption(f"rule_id: {rule_id}")
        st.write(summary_text)
        st.write(f"is_nsfw_channel: {fmt_bool(nsfw_value)}")
        st.write(
            "rating_questionable: "
            f"{fmt_float(q_value)} / rating_explicit: {fmt_float(e_value)} / qe_margin: {fmt_float(qe_margin_value)}"
        )

        kpi_cols = st.columns(3, gap="small")
        with kpi_cols[0]:
            st.markdown("**Exposure**")
            st.write(f"count: {exposure_count}")
            st.write(f"area: {fmt_float(exposure_area)}")
            st.write(f"score: {fmt_float(exposure_score)}")
        with kpi_cols[1]:
            st.markdown("**Context**")
            st.write(f"nsfw_general_sum: {fmt_float(nsfw_general_sum)}")
            st.write(f"placement_risk: {fmt_float(placement_risk)}")
        with kpi_cols[2]:
            st.markdown("**wd14 ratings**")
            st.write(f"g: {fmt_float(_safe_float(ratings.get('g')))}")
            st.write(f"s: {fmt_float(_safe_float(ratings.get('s')))}")
            st.write(f"q: {fmt_float(q_value)}")
            st.write(f"e: {fmt_float(e_value)}")

    rules_data = _load_rules_for_detail(rules_path)
    groups_map = rules_data.get("groups") or {}
    group_hits = compute_group_hits(record, groups_map, topk=256)
    available_groups = set(groups_map.keys()) if isinstance(groups_map, dict) else set()
    resolved_groups = resolve_category_groups(available_groups)

    st.markdown("---")
    category_cols = st.columns(4, gap="small")
    for idx, category in enumerate(("minor", "animal", "nonconsent", "gore")):
        with category_cols[idx]:
            st.markdown(f"**{CATEGORY_LABELS.get(category, category)}**")
            feature_items = _category_feature_entries(
                feature_values, CATEGORY_PREFIXES.get(category, tuple())
            )
            if feature_items:
                st.caption("features (>0)")
                for name, value in feature_items:
                    label = name.replace("_", " ")
                    st.write(f"- {label}: {value:.3f}")
            else:
                st.caption("features (>0): —")
            st.caption("topk タグ")
            df_hits = _category_topk_dataframe(
                group_hits,
                resolved_groups.get(category, set()),
                CATEGORY_PREFIXES.get(category, tuple()),
            )
            _render_topk_chart(df_hits)

    if reason_lines:
        st.markdown("---")
        st.markdown("**reasons（詳細）**")
        for line in reason_lines:
            st.write(f"- {line}")


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


REASON_NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+")


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
