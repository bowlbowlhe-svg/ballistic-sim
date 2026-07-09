"""版本号一致性测试。"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient  # noqa: E402

from ballistic_sim import __version__  # noqa: E402
from ballistic_sim.api import app  # noqa: E402
from ballistic_sim.api.main import create_app  # noqa: E402

TARGET_VERSION = "0.6.1"


def test_version_matches_expected() -> None:
    """``ballistic_sim.__version__`` 与目标版本一致。"""
    assert __version__ == TARGET_VERSION


def test_health_endpoint_reports_same_version() -> None:
    """API /health 返回的版本号与包版本一致。"""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["version"] == __version__


def test_fastapi_version_matches() -> None:
    """FastAPI 应用版本号与包版本一致。"""
    test_app = create_app()
    assert test_app.version == TARGET_VERSION


def test_pyproject_version_matches() -> None:
    """pyproject.toml 中的 version 与目标版本一致。"""
    try:
        import tomllib  # type: ignore
    except ImportError:
        import tomli as tomllib  # type: ignore
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    assert data["project"]["version"] == TARGET_VERSION


def test_changelog_contains_version_section() -> None:
    """CHANGELOG.md 包含目标版本章节。"""
    from pathlib import Path

    changelog = Path(__file__).resolve().parent.parent / "CHANGELOG.md"
    text = changelog.read_text(encoding="utf-8")
    assert f"## [{TARGET_VERSION}]" in text
