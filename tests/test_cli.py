"""CLI end-to-end tests for stage 4."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("ballistic_sim.cli", reason="CLI module not implemented")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Invoke ``ballistic_sim.cli.main`` in a fresh interpreter with ``args``."""
    script = (
        "import sys\n"
        f"sys.argv = ['ballistic-sim'] + {args!r}\n"
        "from ballistic_sim.cli import main\n"
        "main()\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


def _pngs_in(path: Path) -> list[Path]:
    """Return all PNG files found recursively under ``path``."""
    return list(path.rglob("*.png"))


def test_cli_projectile_generates_png(tmp_path: Path) -> None:
    """``--mission projectile --preset M107 --qe 800 --az 45`` should produce PNGs."""
    result = _run_cli(
        ["--mission", "projectile", "--preset", "M107", "--qe", "800", "--az", "45"],
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    pngs = _pngs_in(tmp_path)
    assert pngs, f"No PNG generated; stdout={result.stdout}\nstderr={result.stderr}"


def test_cli_rocket_generates_png(tmp_path: Path) -> None:
    """``--mission rocket --rocket CZ-2F`` should produce PNGs."""
    result = _run_cli(
        ["--mission", "rocket", "--rocket", "CZ-2F"],
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    pngs = _pngs_in(tmp_path)
    assert pngs, f"No PNG generated; stdout={result.stdout}\nstderr={result.stderr}"


def test_cli_no_viz_does_not_generate_png(tmp_path: Path) -> None:
    """``--no-viz`` must suppress PNG output."""
    result = _run_cli(
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
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    pngs = _pngs_in(tmp_path)
    assert not pngs, f"Unexpected PNG generated with --no-viz: {pngs}"
