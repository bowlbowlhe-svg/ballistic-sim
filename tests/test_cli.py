"""CLI end-to-end tests for stage 4."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest  # noqa: E402

pytest.importorskip("ballistic_sim.cli", reason="CLI module not implemented")

from ballistic_sim.cli import main  # noqa: E402


def _run_cli(monkeypatch, args: list[str], cwd: Path) -> None:
    """Invoke ``ballistic_sim.cli.main`` in-process with ``args``."""
    monkeypatch.setattr(sys, "argv", ["ballistic-sim"] + args)
    monkeypatch.chdir(cwd)
    main()


def _pngs_in(path: Path) -> list[Path]:
    """Return all PNG files found recursively under ``path``."""
    return list(path.rglob("*.png"))


def test_cli_projectile_generates_png(tmp_path: Path, monkeypatch) -> None:
    """``--mission projectile --preset M107 --qe 800 --az 45`` should produce PNGs."""
    _run_cli(
        monkeypatch,
        ["--mission", "projectile", "--preset", "M107", "--qe", "800", "--az", "45"],
        tmp_path,
    )
    assert _pngs_in(tmp_path), "No PNG generated"


def test_cli_rocket_generates_png(tmp_path: Path, monkeypatch) -> None:
    """``--mission rocket --rocket CZ-2F`` should produce PNGs."""
    _run_cli(monkeypatch, ["--mission", "rocket", "--rocket", "CZ-2F"], tmp_path)
    assert _pngs_in(tmp_path), "No PNG generated"


def test_cli_no_viz_does_not_generate_png(tmp_path: Path, monkeypatch) -> None:
    """``--no-viz`` must suppress PNG output."""
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
            "--no-viz",
        ],
        tmp_path,
    )
    assert not _pngs_in(tmp_path), "Unexpected PNG generated with --no-viz"


def test_cli_missile_generates_png(tmp_path: Path, monkeypatch) -> None:
    """``--mission missile --missile SRBM_600`` should run without error."""
    _run_cli(monkeypatch, ["--mission", "missile", "--missile", "SRBM_600", "--no-viz"], tmp_path)


def test_cli_icbm_generates_png(tmp_path: Path, monkeypatch) -> None:
    """``--mission icbm`` should run without error."""
    _run_cli(monkeypatch, ["--mission", "icbm", "--no-viz"], tmp_path)


def test_cli_suborbital_generates_png(tmp_path: Path, monkeypatch) -> None:
    """``--mission suborbital`` should run without error."""
    _run_cli(monkeypatch, ["--mission", "suborbital", "--no-viz"], tmp_path)


def test_cli_help_returns_zero(monkeypatch) -> None:
    """``--help`` should exit cleanly."""
    monkeypatch.setattr(sys, "argv", ["ballistic-sim", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
