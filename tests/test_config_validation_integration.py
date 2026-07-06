"""配置校验集成测试：覆盖 simulate、CLI、API 入口。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ballistic_sim.config import (
    GuidanceConfig,
    LaunchConfig,
    SimConfig,
    VehicleConfig,
    validate_config,
)
from ballistic_sim.simulator import simulate

pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi.testclient import TestClient  # noqa: E402

from ballistic_sim.api import app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    """Return a TestClient for the API app."""
    return TestClient(app)


def test_simulate_raises_valueerror_for_low_thrust():
    """T/W <= 1 的 rocket 配置应使 simulate() 抛出 ValueError。"""
    cfg = SimConfig(
        mission="rocket",
        vehicle=VehicleConfig(mass_kg=1000.0, thrust_N=5000.0, burn_time_s=60.0),
        launch=LaunchConfig(),
    )
    with pytest.raises(ValueError, match="T/W"):
        simulate(cfg)


def test_validate_config_error_for_missing_proportional_target():
    """proportional 制导缺少 target_lat/target_lon 时 validate_config 返回 ERROR。"""
    cfg = SimConfig(
        mission="missile",
        vehicle=VehicleConfig(mass_kg=1000.0, thrust_N=20000.0, burn_time_s=60.0),
        launch=LaunchConfig(),
        guidance=GuidanceConfig(guidance_law="proportional"),
    )
    issues = validate_config(cfg)
    errors = [i for i in issues if i.severity == "ERROR"]
    assert any("target_lat_deg" in i.message or "target_lon_deg" in i.message for i in errors)


def test_simulate_runs_with_warnings_only():
    """仅含 WARNING 的配置 simulate() 仍可正常运行。"""
    cfg = SimConfig(
        mission="projectile",
        launch=LaunchConfig(elevation_deg=88.0),
    )
    issues = validate_config(cfg)
    assert not any(i.severity == "ERROR" for i in issues)
    result = simulate(cfg)
    assert result.stop_reason != ""


def test_cli_exits_nonzero_for_invalid_config(tmp_path: Path, monkeypatch) -> None:
    """CLI 对非法配置应返回非零退出码。"""
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "invalid_rocket.yaml"
    config_path.write_text(
        "mission: rocket\n"
        "vehicle:\n"
        "  mass_kg: 1000.0\n"
        "  thrust_N: 5000.0\n"
        "  burn_time_s: 60.0\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ballistic_sim",
            "--config",
            str(config_path),
            "--no-viz",
            "--no-summary",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "T/W" in result.stderr


def test_api_returns_400_for_invalid_config(client: TestClient) -> None:
    """API 对非法配置返回 400。"""
    response = client.post(
        "/simulate/rocket",
        json={
            "vehicle": {
                "mass_kg": 1000.0,
                "thrust_N": 5000.0,
                "burn_time_s": 60.0,
                "stages": None,
            },
            "include_trajectory": False,
        },
    )
    assert response.status_code == 400
    assert "T/W" in response.text
