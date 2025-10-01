from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.batch_loader import ImageRequest, load_images
from tools.scl_viewer import utils

TMP_DIR = Path("tools/scl_viewer/tmp")
THUMBS_DIR = Path("tools/scl_viewer/thumbs")
CACHE_DIR = Path("cache/imgs")
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
        st.dataframe(st.session_state["report_df"], use_container_width=True)

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
    st.session_state["findings_df"] = build_findings_dataframe(records)


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
    df = st.session_state.get("findings_df")
    if df is None:
        df = build_findings_dataframe(records)
    filtered = df
    if state.severity_filter:
        filtered = filtered[filtered["severity"].isin(state.severity_filter)]
    if state.rule_query:
        filtered = filtered[filtered["rule_id"].str.contains(state.rule_query, case=False, na=False)]
    if state.exposure_min > 0:
        filtered = filtered[filtered["exposure_area"].fillna(0.0) >= state.exposure_min]
    filtered = filtered.sort_values("severity", key=_severity_sort_key)
    st.subheader("Findings")
    column_config = {
        "thumbnail": st.column_config.ImageColumn(
            "thumb",
            help="p0 添付から生成したサムネイル",
            width="small",
        )
    }
    st.dataframe(filtered, use_container_width=True, column_config=column_config)
    render_gallery(filtered, records)


def render_gallery(df: pd.DataFrame, records: list[dict], *, columns: int = 3) -> None:
    st.markdown("#### サムネイル")
    if df.empty:
        st.info("フィルタに一致するレコードがありません。")
        return
    record_map = {record.get("phash"): record for record in records}
    rows = []
    for phash in df["phash"].dropna().head(12):
        record = record_map.get(phash)
        if not record:
            continue
        thumb = get_thumbnail_for_record(record)
        rows.append((record, thumb))
    if not rows:
        st.info("表示できるサムネイルがありません。")
        return
    for start in range(0, len(rows), columns):
        cols = st.columns(columns)
        for col, (record, thumb_path) in zip(cols, rows[start : start + columns]):
            with col:
                if thumb_path and thumb_path.exists():
                    col.image(str(thumb_path), use_column_width=True)
                else:
                    col.write("(画像なし)")
                col.caption(
                    f"{record.get('severity', '').upper()} | {record.get('rule_id') or ''}\n"
                    f"{'; '.join(record.get('reasons', []))}"
                )


def get_thumbnail_for_record(record: dict) -> Path | None:
    phash = record.get("phash")
    if not phash:
        return None
    urls = list(_attachment_urls(record))
    if not urls:
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    image_path = CACHE_DIR / f"{phash}.jpg"
    if not image_path.exists():
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
        st.dataframe(diff_df, use_container_width=True)
    samples = st.session_state.get("ab_samples")
    if samples:
        st.subheader("A/B サンプル")
        st.json(samples[:50])


def ensure_directories() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    THUMBS_DIR.mkdir(parents=True, exist_ok=True)


def build_findings_dataframe(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        metrics = record.get("metrics", {}) or {}
        thumb_path = get_thumbnail_for_record(record)
        rows.append(
            {
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
                "placement_risk": metrics.get("placement_risk")
                or metrics.get("placement_risk_pre"),
                "winning_rule": (metrics.get("winning", {}) or {}).get("rule_id"),
            }
        )
    df = pd.DataFrame(rows)
    return df


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
