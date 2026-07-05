"""Tests for the ballistic_sim Web API."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient  # noqa: E402

from ballistic_sim.api import app, create_app  # noqa: E402
from ballistic_sim.api.dependencies import require_fastapi, require_uvicorn  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    """Return a TestClient for the API app."""
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    """GET /health returns ok and version."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    from ballistic_sim import __version__

    assert data["version"] == __version__


def test_root_serves_index_html(client: TestClient) -> None:
    """GET / returns the static frontend."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Ballistic Sim Web API" in response.text


def test_simulate_projectile_default(client: TestClient) -> None:
    """POST /simulate/projectile with defaults returns a valid response."""
    response = client.post("/simulate/projectile", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["mission"] == "projectile"
    assert data["stop_reason"] != ""
    assert data["range_m"] is not None
    assert data["apogee_m"] is not None
    assert data["impact_lat"] is not None
    assert data["impact_lon"] is not None
    assert data["trajectory"] is not None
    assert len(data["trajectory"]["t"]) > 0


def test_simulate_projectile_custom_launch(client: TestClient) -> None:
    """Custom launch angles are respected."""
    response = client.post(
        "/simulate/projectile",
        json={"launch": {"elevation_deg": 30, "azimuth_deg": 45}, "include_trajectory": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mission"] == "projectile"
    assert data["trajectory"] is None
    assert data["range_m"] > 0


def test_simulate_unknown_mission_returns_400(client: TestClient) -> None:
    """An unsupported mission yields a 400 error."""
    response = client.post("/simulate/ufo", json={})
    assert response.status_code == 400


def test_simulate_rocket_default(client: TestClient) -> None:
    """POST /simulate/rocket with defaults runs without error."""
    response = client.post("/simulate/rocket", json={"include_trajectory": False})
    assert response.status_code == 200
    data = response.json()
    assert data["mission"] == "rocket"
    assert data["stop_reason"] != ""


def test_simulate_missile_default(client: TestClient) -> None:
    """POST /simulate/missile with defaults runs without error."""
    response = client.post("/simulate/missile", json={"include_trajectory": False})
    assert response.status_code == 200
    data = response.json()
    assert data["mission"] == "missile"
    assert data["stop_reason"] != ""


def test_simulate_icbm_default(client: TestClient) -> None:
    """POST /simulate/icbm with defaults runs without error."""
    response = client.post("/simulate/icbm", json={"include_trajectory": False})
    assert response.status_code == 200
    data = response.json()
    assert data["mission"] == "icbm"


def test_simulate_suborbital_default(client: TestClient) -> None:
    """POST /simulate/suborbital with defaults runs without error."""
    response = client.post("/simulate/suborbital", json={"include_trajectory": False})
    assert response.status_code == 200
    data = response.json()
    assert data["mission"] == "suborbital"


def test_simulate_with_environment_override(client: TestClient) -> None:
    """Environment overrides are merged into the base config."""
    response = client.post(
        "/simulate/projectile",
        json={
            "environment": {"wind_model": "none"},
            "options": {"rtol": 1e-5},
            "include_trajectory": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mission"] == "projectile"
    assert data["range_m"] is not None


def test_monte_carlo_endpoint(client: TestClient) -> None:
    """POST /simulate/monte-carlo returns dispersion statistics."""
    response = client.post(
        "/simulate/monte-carlo",
        json={
            "mission": "projectile",
            "launch": {"elevation_deg": 45, "azimuth_deg": 90},
            "n_samples": 20,
            "backend": "batch",
            "seed": 42,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["samples"] > 0
    assert data["cep50_m"] >= 0
    assert data["cep90_m"] >= data["cep50_m"]


def test_firecontrol_projectile(client: TestClient) -> None:
    """POST /firecontrol/solve for projectile returns a firing solution."""
    response = client.post(
        "/firecontrol/solve",
        json={
            "mission": "projectile",
            "launch": {"lat_deg": 0.0, "lon_deg": 0.0},
            "target_lat_deg": 0.05,
            "target_lon_deg": 0.05,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["mission"] == "projectile"
    assert data["qe_deg"] is not None
    assert data["az_deg"] is not None
    assert data["converged"] in (True, False)


def test_firecontrol_unsupported_mission_returns_501(client: TestClient) -> None:
    """Fire control for rocket/icbm/suborbital returns 501."""
    response = client.post(
        "/firecontrol/solve",
        json={
            "mission": "rocket",
            "target_lat_deg": 0.0,
            "target_lon_deg": 0.0,
        },
    )
    assert response.status_code == 501


def test_create_app_returns_distinct_instances() -> None:
    """Each call to create_app returns a new FastAPI instance."""
    app1 = create_app()
    app2 = create_app()
    assert app1 is not app2


def test_dependency_guards_raise_friendly_error(monkeypatch) -> None:
    """Missing fastapi/uvicorn guards raise friendly ImportError messages."""
    import sys

    monkeypatch.setitem(sys.modules, "fastapi", None)
    with pytest.raises(ImportError, match="pip install ballistic_sim\\[web\\]"):
        require_fastapi()

    monkeypatch.setitem(sys.modules, "uvicorn", None)
    with pytest.raises(ImportError, match="pip install ballistic_sim\\[web\\]"):
        require_uvicorn()
