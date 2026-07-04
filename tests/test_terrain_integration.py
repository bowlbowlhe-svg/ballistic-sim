"""地形集成端到端冒烟测试。"""

from __future__ import annotations

from ballistic_sim.config import (
    EnvironmentConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    VehicleConfig,
)
from ballistic_sim.phases.builder import build_phases
from ballistic_sim.simulator import simulate


def test_projectile_simulation_with_hilly_terrain() -> None:
    """projectile 任务在 hilly 地形下可完整仿真并落地。"""
    cfg = SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(mass_kg=10.0, diameter_m=0.1, cd=0.3),
        launch=LaunchConfig(
            lat_deg=0.0,
            lon_deg=0.0,
            alt_m=10.0,
            azimuth_deg=45.0,
            elevation_deg=45.0,
            v0_m_s=100.0,
        ),
        environment=EnvironmentConfig(
            atmosphere="isa",
            terrain_model="hilly",
            terrain_extent=(0.0, 1.0, 0.0, 1.0),
            terrain_max_height=50.0,
        ),
        options=OptionsConfig(integrator="RK45", rtol=1e-6, atol=1e-9, max_step=1.0),
    )
    phases = build_phases(cfg)
    result = simulate(cfg, phases=phases)
    assert result.y.size > 0
    assert result.t[-1] > 0.0
    assert result.stop_reason != "integration_failed@无动力弹道"


def test_rocket_simulation_with_null_terrain() -> None:
    """rocket 任务在 null 地形下仍可构建 phase 并运行。"""
    cfg = SimConfig(
        mission="rocket",
        vehicle=VehicleConfig(mass_kg=1000.0, diameter_m=0.5, cd=0.3, area_ref_m2=1.0),
        launch=LaunchConfig(
            lat_deg=0.0,
            lon_deg=0.0,
            alt_m=1.0,
            azimuth_deg=90.0,
            elevation_deg=90.0,
            v0_m_s=0.0,
        ),
        environment=EnvironmentConfig(
            atmosphere="isa",
            gravity_model="j2",
            terrain_model="null",
        ),
        options=OptionsConfig(integrator="RK45", rtol=1e-6, atol=1e-9, max_step=1.0),
    )
    phases = build_phases(cfg)
    result = simulate(cfg, phases=phases)
    assert result.y.size > 0
    assert result.t[-1] >= 0.0
