"""版本号一致性测试。"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient  # noqa: E402

from ballistic_sim import __version__  # noqa: E402
from ballistic_sim.api import app  # noqa: E402


def test_version_matches_expected() -> None:
    """``ballistic_sim.__version__`` 与目标版本一致。"""
    assert __version__ == "0.3.0"


def test_health_endpoint_reports_same_version() -> None:
    """API /health 返回的版本号与包版本一致。"""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["version"] == __version__
