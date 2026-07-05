"""Shared pytest fixtures and helpers for ballistic-sim."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import matplotlib
import pytest

matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def _reset_matplotlib_backend():
    """Reset matplotlib backend to Agg after each test to isolate GUI tests."""
    yield
    matplotlib.use("Agg", force=True)
    plt = matplotlib.pyplot
    if hasattr(plt, "close"):
        plt.close("all")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden"


def load_golden(name: str) -> Dict[str, Any]:
    """Load a golden reference JSON by file name."""
    path = GOLDEN_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing golden file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def golden_dir() -> Path:
    """Return the directory containing golden reference files."""
    return GOLDEN_DIR


@pytest.fixture
def golden_m107() -> Dict[str, Any]:
    """Return the M107 atmospheric golden reference."""
    return load_golden("atmospheric_m107.json")


@pytest.fixture
def golden_cz2f() -> Dict[str, Any]:
    """Return the CZ-2F exo-atmospheric golden reference."""
    return load_golden("exo_cz2f.json")
