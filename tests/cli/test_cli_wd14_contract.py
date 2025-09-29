from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import cli_wd14

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class DummySession:
    def __init__(self, *args, **kwargs):  # noqa: D401, ANN001
        self.size = 512


class DummyPrediction:
    def __init__(self) -> None:
        self.rating = {"general": 0.8, "explicit": 0.1, "questionable": 0.05, "sensitive": 0.05}
        self.general = [("bikini", 0.6)]
        self.general_raw = [("bikini", 0.65)]
        self.character = []
        self.raw_scores = SimpleNamespace(tolist=lambda: [0.1, 0.2])


class DummyAnalyzer:
    def __init__(self, session, labelspace, *, raw_general_whitelist, **kwargs):  # noqa: D401, ANN001
        self.session = session
        self.labelspace = labelspace
        self.raw_general_whitelist = tuple(sorted(raw_general_whitelist))

    def predict(self, images):  # noqa: D401, ANN001
        return [DummyPrediction() for _ in images]

    @staticmethod
    def general_raw_from_scores(scores):  # noqa: D401, ANN001
        return [("from_cache", 0.9) for _ in scores[:1]]


class DummyCache:
    def __init__(self, path):  # noqa: D401, ANN001
        self._store: dict[tuple[str, str, str], dict] = {}

    def get(self, key):  # noqa: D401, ANN001
        handle = (key.phash, key.model, key.revision)
        return self._store.get(handle)

    def set(self, key, payload):  # noqa: D401, ANN001
        handle = (key.phash, key.model, key.revision)
        self._store[handle] = payload


def _base_args(tmp_path: Path, *, rules: Path) -> list[str]:
    out_path = tmp_path / "wd14.jsonl"
    metrics_path = tmp_path / "wd14_metrics.json"
    cache_path = tmp_path / "wd14_cache.sqlite"
    return [
        "cli_wd14",
        "--input",
        str(FIXTURES / "p0_scan_minimal.csv"),
        "--out",
        str(out_path),
        "--metrics",
        str(metrics_path),
        "--cache",
        str(cache_path),
        "--rules-config",
        str(rules),
        "--batch-size",
        "1",
    ]


def _run_cli(monkeypatch: pytest.MonkeyPatch, argv: list[str], tmp_path: Path, created_analyzers: list[DummyAnalyzer]) -> None:
    monkeypatch.setattr(cli_wd14, "ensure_local_files", lambda repo_id: (tmp_path / "labels.csv", tmp_path / "model.onnx"))
    monkeypatch.setattr(cli_wd14, "load_labelspace", lambda *_: {})
    monkeypatch.setattr(cli_wd14, "WD14Session", DummySession)

    def _factory(*args, **kwargs):
        analyzer = DummyAnalyzer(*args, **kwargs)
        created_analyzers.append(analyzer)
        return analyzer

    monkeypatch.setattr(cli_wd14, "WD14Analyzer", _factory)
    monkeypatch.setattr(cli_wd14, "WD14Cache", DummyCache)

    def _fake_load_images(requests, **kwargs):  # noqa: ANN001
        return [SimpleNamespace(image="dummy", request=req, note=None) for req in requests]

    async def _fake_load_images_async(requests, **kwargs):  # noqa: ANN001
        return _fake_load_images(requests, **kwargs)

    monkeypatch.setattr(cli_wd14, "load_images", _fake_load_images_async)
    monkeypatch.setattr(sys, "argv", argv)
    cli_wd14.main()


def _arg_path(argv: list[str], flag: str) -> Path:
    idx = argv.index(flag)
    return Path(argv[idx + 1])


def test_cli_wd14_generates_predictions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    created_analyzers: list[DummyAnalyzer] = []
    argv = _base_args(tmp_path, rules=FIXTURES / "rules_v2_minimal.yaml")
    _run_cli(monkeypatch, argv, tmp_path, created_analyzers)

    out_path = _arg_path(argv, "--out")
    record = json.loads(out_path.read_text(encoding="utf-8").strip())
    assert record["phash"] == "feedfacefeedface"
    wd14_payload = record["wd14"]
    assert "general" in wd14_payload and wd14_payload["general"]
    assert "general_raw" in wd14_payload and wd14_payload["general_raw"]
    # whitelistは rules_v2_minimal.yaml の正規化済みタグが渡される
    assert created_analyzers
    assert set(created_analyzers[0].raw_general_whitelist) == {"bikini", "see_through", "topless"}


def test_cli_wd14_rules_mismatch_exits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    created_analyzers: list[DummyAnalyzer] = []
    argv = _base_args(tmp_path, rules=FIXTURES / "rules_v2_mismatch.yaml")
    monkeypatch.setattr(cli_wd14, "ensure_local_files", lambda repo_id: (tmp_path / "labels.csv", tmp_path / "model.onnx"))
    monkeypatch.setattr(cli_wd14, "load_labelspace", lambda *_: {})
    monkeypatch.setattr(cli_wd14, "WD14Session", DummySession)
    monkeypatch.setattr(cli_wd14, "WD14Analyzer", lambda *a, **k: DummyAnalyzer(*a, **k))
    monkeypatch.setattr(cli_wd14, "WD14Cache", DummyCache)
    monkeypatch.setattr(cli_wd14, "load_images", lambda *a, **k: [])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        cli_wd14.main()
    assert excinfo.value.code == 2
    assert not created_analyzers
