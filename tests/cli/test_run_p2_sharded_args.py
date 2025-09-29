from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts import run_p2_sharded


class DummyProcess:
    def __init__(self, args: tuple[str, ...]) -> None:
        self.args = args
        self.pid = 9999

    async def wait(self) -> int:
        return 0

class _Patch:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, captured: list[tuple[str, ...]]):
        self.monkeypatch = monkeypatch
        self.captured = captured

    async def fake_exec(self, *cmd, **kwargs):  # noqa: ANN001
        self.captured.append(tuple(cmd))
        try:
            out_idx = cmd.index("--out") + 1
            metrics_idx = cmd.index("--metrics") + 1
        except ValueError:
            out_idx = metrics_idx = None
        if out_idx is not None:
            out_path = Path(cmd[out_idx])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("{}\n", encoding="utf-8")
        if metrics_idx is not None:
            metrics_path = Path(cmd[metrics_idx])
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text("{}", encoding="utf-8")
        return DummyProcess(cmd)


def _invoke(monkeypatch: pytest.MonkeyPatch, argv: list[str], captured: list[tuple[str, ...]]) -> None:
    patch = _Patch(monkeypatch, captured)
    monkeypatch.setattr(run_p2_sharded.asyncio, "create_subprocess_exec", patch.fake_exec)
    monkeypatch.setattr(sys, "argv", argv)
    try:
        run_p2_sharded.main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise


def test_run_p2_sharded_injects_rules_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    shard_dir = tmp_path / "p0"
    shard_dir.mkdir()
    shard_file = shard_dir / "shard_000.csv"
    shard_file.write_text("id\n1\n", encoding="utf-8")

    wd14_dir = tmp_path / "p1"
    wd14_dir.mkdir()
    wd14_file = wd14_dir / "p1_wd14_shard_000.jsonl"
    wd14_file.write_text("{}\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    status_file = tmp_path / "status.json"

    argv = [
        "run_p2_sharded",
        "--shard-glob",
        str(shard_file),
        "--wd14-dir",
        str(wd14_dir),
        "--out-dir",
        str(out_dir),
        "--status-file",
        str(status_file),
        "--parallel",
        "1",
    ]

    captured: list[tuple[str, ...]] = []
    _invoke(monkeypatch, argv, captured)

    assert captured, "no subprocess command recorded"
    injected = [cmd for cmd in captured if "--rules-config" in cmd]
    assert injected, "--rules-config not injected"
    for cmd in injected:
        assert "configs/rules.yaml" in cmd
