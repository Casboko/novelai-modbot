from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app import cli_scan

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _base_args(tmp_path: Path) -> list[str]:
    return [
        "cli_scan",
        "--analysis",
        str(FIXTURES / "p2_analysis_minimal.jsonl"),
        "--findings",
        str(tmp_path / "findings.jsonl"),
        "--rules",
        str(FIXTURES / "rules_v2_minimal.yaml"),
        "--metrics",
        str(tmp_path / "metrics.json"),
        "--print-config",
    ]


def _run_cli(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: list[str]) -> str:
    monkeypatch.setattr(sys, "argv", argv)
    cli_scan.main()
    captured = capsys.readouterr()
    return captured.out


def _extract_banner(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("policy="):
            return line
    raise AssertionError("banner with policy line not found")


def test_cli_scan_uses_env_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("MODBOT_DSL_MODE", "strict")
    argv = _base_args(tmp_path)
    output = _run_cli(monkeypatch, capsys, argv)
    banner = _extract_banner(output)
    assert "policy=strict" in banner


def test_cli_scan_cli_overrides_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("MODBOT_DSL_MODE", "strict")
    argv = _base_args(tmp_path) + ["--dsl-mode", "warn"]
    output = _run_cli(monkeypatch, capsys, argv)
    banner = _extract_banner(output)
    assert "policy=warn" in banner


def test_cli_scan_lock_mode_forces_policy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("MODBOT_DSL_MODE", "warn")
    argv = _base_args(tmp_path) + ["--dsl-mode", "warn", "--lock-mode", "strict"]
    output = _run_cli(monkeypatch, capsys, argv)
    banner = _extract_banner(output)
    assert "policy=strict" in banner
