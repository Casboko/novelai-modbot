from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from app import analysis_merge

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class DummyAnalyzer:
    version = "dummy"

    def detect_batch(self, images):  # noqa: D401, ANN001
        return [[] for _ in images]


class DummyCache:
    def __init__(self, path):  # noqa: D401, ANN001
        self._store: dict[str, dict] = {}

    def get(self, key):  # noqa: D401, ANN001
        return self._store.get(getattr(key, "phash", ""))

    def set(self, key, payload):  # noqa: D401, ANN001
        self._store[getattr(key, "phash", "")] = payload


def _common_args(tmp_path: Path, *, out_name: str = "p2.jsonl") -> list[str]:
    out_path = tmp_path / out_name
    metrics_path = tmp_path / f"{out_path.stem}_metrics.json"
    cache_path = tmp_path / "nudenet_cache.sqlite"
    args = [
        "analysis_merge",
        "--scan",
        str(FIXTURES / "p0_scan_minimal.csv"),
        "--wd14",
        str(FIXTURES / "p1_wd14_minimal.jsonl"),
        "--out",
        str(out_path),
        "--metrics",
        str(metrics_path),
        "--nudenet-cache",
        str(cache_path),
        "--nudenet-config",
        str(FIXTURES / "nudenet_minimal.yaml"),
        "--xsignals-config",
        str(FIXTURES / "xsignals_minimal.yaml"),
        "--nudenet-mode",
        "never",
        "--seed",
        "123",
    ]
    return args


def _run_analysis_merge(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(analysis_merge, "NudeNetAnalyzer", lambda: DummyAnalyzer())
    monkeypatch.setattr(analysis_merge, "NudeNetCache", DummyCache)
    monkeypatch.setattr(sys, "argv", argv)
    analysis_merge.main()


def _load_first_record(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").strip()
    assert text
    return json.loads(text.splitlines()[0])


def _arg_path(argv: list[str], flag: str) -> Path:
    idx = argv.index(flag)
    return Path(argv[idx + 1])


def test_analysis_merge_uses_rules_dictionary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    argv = _common_args(tmp_path) + [
        "--rules-config",
        str(FIXTURES / "rules_v2_minimal.yaml"),
    ]
    _run_analysis_merge(monkeypatch, argv)

    out_path = _arg_path(argv, "--out")
    record = _load_first_record(out_path)
    nsfw_sum = record["xsignals"]["nsfw_general_sum"]
    assert pytest.approx(nsfw_sum, rel=1e-6) == 1.0
    metrics_path = _arg_path(argv, "--metrics")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["total_records"] == 1


def test_analysis_merge_fallback_to_xsignals(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    argv = _common_args(tmp_path, out_name="p2_fallback.jsonl") + [
        "--rules-config",
        str(tmp_path / "missing_rules.yaml"),
    ]
    _run_analysis_merge(monkeypatch, argv)

    out_path = _arg_path(argv, "--out")
    record = _load_first_record(out_path)
    # fallback_tag comes from xsignals fixture and should be counted
    assert pytest.approx(record["xsignals"]["nsfw_general_sum"], rel=1e-6) == 0.2


def test_analysis_merge_fails_on_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    argv = _common_args(tmp_path, out_name="p2_mismatch.jsonl") + [
        "--rules-config",
        str(FIXTURES / "rules_v2_mismatch.yaml"),
    ]
    monkeypatch.setattr(analysis_merge, "NudeNetAnalyzer", lambda: DummyAnalyzer())
    monkeypatch.setattr(analysis_merge, "NudeNetCache", DummyCache)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        analysis_merge.main()
    assert excinfo.value.code == 2


def test_analysis_merge_prefers_general_raw(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base_args = _common_args(tmp_path, out_name="p2_base.jsonl") + [
        "--rules-config",
        str(FIXTURES / "rules_v2_minimal.yaml"),
    ]
    _run_analysis_merge(monkeypatch, base_args)
    base_record = _load_first_record(_arg_path(base_args, "--out"))

    enriched_args = _common_args(tmp_path, out_name="p2_raw.jsonl") + [
        "--rules-config",
        str(FIXTURES / "rules_v2_minimal.yaml"),
        "--wd14",
        str(FIXTURES / "p1_wd14_minimal_with_raw.jsonl"),
    ]
    _run_analysis_merge(monkeypatch, enriched_args)
    enriched_record = _load_first_record(_arg_path(enriched_args, "--out"))

    assert (
        enriched_record["xsignals"]["nsfw_general_sum"]
        >= base_record["xsignals"]["nsfw_general_sum"]
    )


def test_analysis_merge_streaming_matches_buffered(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    stream_args = _common_args(tmp_path, out_name="p2_stream.jsonl") + [
        "--rules-config",
        str(FIXTURES / "rules_v2_minimal.yaml"),
    ]
    _run_analysis_merge(monkeypatch, stream_args)
    stream_output = _arg_path(stream_args, "--out").read_text(encoding="utf-8")

    buffered_args = _common_args(tmp_path, out_name="p2_buffered.jsonl") + [
        "--rules-config",
        str(FIXTURES / "rules_v2_minimal.yaml"),
        "--buffered",
    ]
    _run_analysis_merge(monkeypatch, buffered_args)
    buffered_output = _arg_path(buffered_args, "--out").read_text(encoding="utf-8")

    assert stream_output == buffered_output
