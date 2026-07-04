"""CLI 风场与地形参数测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ballistic_sim.cli import main


def _run_cli(monkeypatch, args: list[str], cwd: Path) -> None:
    """Invoke ``ballistic_sim.cli.main`` in-process with ``args``."""
    monkeypatch.setattr(sys, "argv", ["ballistic-sim"] + args)
    monkeypatch.chdir(cwd)
    main()


def test_cli_projectile_with_wind_model(tmp_path: Path, monkeypatch) -> None:
    """--wind-model 参数可正确注入配置并运行。"""
    _run_cli(
        monkeypatch,
        [
            "--mission",
            "projectile",
            "--preset",
            "M107",
            "--qe",
            "800",
            "--az",
            "45",
            "--wind-model",
            "log",
            "--no-viz",
        ],
        tmp_path,
    )


def test_cli_projectile_with_terrain_hilly(tmp_path: Path, monkeypatch) -> None:
    """--terrain-model hilly 可正确注入配置并运行。"""
    _run_cli(
        monkeypatch,
        [
            "--mission",
            "projectile",
            "--preset",
            "M107",
            "--qe",
            "800",
            "--az",
            "45",
            "--terrain-model",
            "hilly",
            "--terrain-extent",
            "0,1,0,1",
            "--no-viz",
        ],
        tmp_path,
    )


def test_cli_projectile_with_wind_profile(tmp_path: Path, monkeypatch) -> None:
    """--wind-profile 参数从文件加载廓线。"""
    profile_path = tmp_path / "wind.txt"
    profile_path.write_text("# h e n u\n0 0 0 0\n1000 5 0 0\n", encoding="utf-8")
    _run_cli(
        monkeypatch,
        [
            "--mission",
            "projectile",
            "--preset",
            "M107",
            "--qe",
            "800",
            "--az",
            "45",
            "--wind-model",
            "profile",
            "--wind-profile",
            str(profile_path),
            "--no-viz",
        ],
        tmp_path,
    )


def test_cli_terrain_extent_invalid_raises(tmp_path: Path, monkeypatch) -> None:
    """--terrain-extent 格式错误时抛出异常。"""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ballistic-sim",
            "--mission",
            "projectile",
            "--preset",
            "M107",
            "--qe",
            "800",
            "--terrain-model",
            "hilly",
            "--terrain-extent",
            "0,1,0",
            "--no-viz",
        ],
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        main()
