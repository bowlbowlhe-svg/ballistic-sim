"""风场与地形集成边界测试。

覆盖阶段 2.3 真实风场与地形集成的关键异常路径与行为差异，
防止配置解析与仿真主循环出现回归。
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from ballistic_sim.config import (
    EnvironmentConfig,
    GuidanceConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    VehicleConfig,
)
from ballistic_sim.dynamics.common import DynamicContext
from ballistic_sim.dynamics.point_mass import PointMassDynamics
from ballistic_sim.dynamics.powered_eci import PoweredECIDynamics
from ballistic_sim.frames import ecef_to_eci, ecef_to_geodetic, eci_to_ecef
from ballistic_sim.models.aerodynamics import ConstantAeroModel
from ballistic_sim.models.atmosphere import StandardAtmosphere
from ballistic_sim.models.terrain import NullTerrainModel, TerrainExtent, TerrainModel
from ballistic_sim.models.wind import LogarithmicWind, UniformWind
from ballistic_sim.phases.coasting import CoastingPhase
from ballistic_sim.phases.events import make_ground_event
from ballistic_sim.phases.powered import PoweredPhase
from ballistic_sim.phases.terminal import TerminalPhase
from ballistic_sim.context import _resolve_terrain, _resolve_wind
from ballistic_sim.simulator import simulate


def _make_cfg(env: EnvironmentConfig | None = None) -> SimConfig:
    return SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(mass_kg=10.0, diameter_m=0.1, cd=0.3),
        launch=LaunchConfig(),
        environment=env or EnvironmentConfig(),
        options=OptionsConfig(),
    )


def test_profile_wind_missing_source_raises() -> None:
    """profile 风模型在 wind_profile_path/text 均缺失时抛出 ValueError。"""
    cfg = _make_cfg(EnvironmentConfig(wind_model="profile"))
    with pytest.raises(ValueError, match="wind_profile_text|wind_profile_path"):
        _resolve_wind(cfg)


@pytest.mark.parametrize(
    "terrain_model",
    ["numpy", "image", "geotiff", "srtm_dir", "srtm_files"],
)
def test_file_terrain_missing_path_raises(terrain_model: str) -> None:
    """需要文件路径的地形模型在 terrain_path 缺失时抛出 ValueError。"""
    cfg = _make_cfg(EnvironmentConfig(terrain_model=terrain_model))
    with pytest.raises(ValueError, match="terrain_path"):
        _resolve_terrain(cfg)


def test_dryden_wind_projectile_simulation_runs() -> None:
    """dryden 风模型可正常构造并完成 projectile 仿真。"""
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
            wind_model="dryden",
            wind_m_s=[1.0, 0.0, 0.0],
            wind_dryden_seed=42,
        ),
        options=OptionsConfig(integrator="RK45", rtol=1e-6, atol=1e-9, max_step=1.0),
    )
    result = simulate(cfg)
    assert result.y.size > 0
    assert result.t[-1] > 0.0
    assert result.stop_reason != "integration_failed@无动力弹道"


def test_rocket_hilly_terrain_ground_event_depends_on_terrain() -> None:
    """rocket 任务在 hilly 地形下落地事件触发高度受地形高程影响。"""
    lat0, lon0 = 0.0, 0.0
    # 构造一个 100 m 高、覆盖范围足够大的水平地形，确保火箭回落期间不越界
    elev = np.full((4, 4), 100.0, dtype=float)
    extent = TerrainExtent(lat_min=-5.0, lat_max=5.0, lon_min=-5.0, lon_max=5.0)
    terrain = TerrainModel(elev, extent)

    # 小型单级火箭，主动段后无动力回落
    mass = 1000.0
    m_prop = 50.0
    m_dry = mass - m_prop
    Aref = 0.196
    thrust = 10.0e3
    isp = 290.0
    stage: dict[str, Any] = dict(
        name="small-rocket",
        thrust_sl=thrust,
        thrust_vac=thrust * 1.1,
        isp_vac=isp,
        m_prop=m_prop,
        m_dry=m_dry,
        Aref=Aref,
    )
    guid = dict(
        lat_deg=lat0,
        lon_deg=lon0,
        azimuth_deg=90.0,
        t_pitchover=5.0,
        kick_deg=85.0,
        t_kick_end=15.0,
    )
    dyn_boost = PoweredECIDynamics(stage=stage, guidance=guid)
    t_burn = float(stage["m_prop"]) / dyn_boost.prop.mdot

    ph_boost = PoweredPhase(
        name="动力上升",
        t_span=(0.0, t_burn * 1.5),
        dynamics=dyn_boost,
        guidance=guid,
        m_dry=m_dry,
        sep_name="燃尽",
        terrain=terrain,
        lat0=lat0,
        lon0=lon0,
    )
    coast_stage = dict(
        name="coast",
        thrust_sl=0.0,
        thrust_vac=0.0,
        isp_vac=1.0,
        m_prop=0.0,
        m_dry=m_dry,
        Aref=Aref,
    )
    dyn_coast = PoweredECIDynamics(
        stage=coast_stage,
        guidance=guid,
        modes={"thrust": False, "drag": True, "j2": True},
    )
    ph_coast = CoastingPhase(
        name="无动力回落",
        t_span=(0.0, 3600.0),
        dynamics=dyn_coast,
        guidance=guid,
        terrain=terrain,
        lat0=lat0,
        lon0=lon0,
    )
    ph_terminal = TerminalPhase(
        name="终点",
        t_span=(0.0, 3600.0),
        dynamics=dyn_coast,
        terrain=terrain,
        lat0=lat0,
        lon0=lon0,
    )

    cfg = SimConfig(
        mission="rocket",
        vehicle=VehicleConfig(
            mass_kg=mass,
            diameter_m=0.5,
            cd=0.3,
            area_ref_m2=Aref,
        ),
        launch=LaunchConfig(
            lat_deg=lat0,
            lon_deg=lon0,
            alt_m=110.0,
            azimuth_deg=90.0,
            elevation_deg=90.0,
            v0_m_s=0.0,
        ),
        environment=EnvironmentConfig(atmosphere="isa", gravity_model="j2"),
        guidance=GuidanceConfig(),
        options=OptionsConfig(
            integrator="RK45",
            rtol=1e-6,
            atol=1e-9,
            max_step=1.0,
            terminate_impact=True,
        ),
    )
    # 注入自定义 HillyTerrainModel，属于 phase 显式传参的合法例外。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = simulate(cfg, phases=[ph_boost, ph_coast, ph_terminal])
    assert result.y.size > 0
    # 应触发落地事件
    landing_events = [ev for ev in result.event_log if "落地" in (ev.get("name") or "")]
    assert landing_events, f"未触发落地事件: {result.event_log}"

    # 落地时刻 ECEF 高度应接近地形高程 100 m（允许数值容差）
    r_end = result.y[-1, 0:3]
    r_ecef = eci_to_ecef(r_end, float(result.t[-1]))
    _, _, alt_end = ecef_to_geodetic(r_ecef)
    assert 90.0 < alt_end < 115.0

    # 与同一初始条件但 NullTerrainModel 的落地高度不同
    ph_boost_nt = PoweredPhase(
        name="动力上升",
        t_span=(0.0, t_burn * 1.5),
        dynamics=dyn_boost,
        guidance=guid,
        m_dry=m_dry,
        sep_name="燃尽",
        terrain=NullTerrainModel(),
        lat0=lat0,
        lon0=lon0,
    )
    ph_coast_nt = CoastingPhase(
        name="无动力回落",
        t_span=(0.0, 3600.0),
        dynamics=dyn_coast,
        guidance=guid,
        terrain=NullTerrainModel(),
        lat0=lat0,
        lon0=lon0,
    )
    ph_terminal_nt = TerminalPhase(
        name="终点",
        t_span=(0.0, 3600.0),
        dynamics=dyn_coast,
        terrain=NullTerrainModel(),
        lat0=lat0,
        lon0=lon0,
    )
    # 注入 NullTerrainModel 做对比，属于 phase 显式传参的合法例外。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result_nt = simulate(cfg, phases=[ph_boost_nt, ph_coast_nt, ph_terminal_nt])
    landing_events_nt = [ev for ev in result_nt.event_log if "落地" in (ev.get("name") or "")]
    assert landing_events_nt, f"未触发落地事件 (null terrain): {result_nt.event_log}"
    r_end_nt = result_nt.y[-1, 0:3]
    r_ecef_nt = eci_to_ecef(r_end_nt, float(result_nt.t[-1]))
    _, _, alt_end_nt = ecef_to_geodetic(r_ecef_nt)
    assert abs(alt_end - 100.0) < abs(alt_end_nt - 100.0)


def test_point_mass_eci_logarithmic_wind_differs_from_uniform() -> None:
    """PointMassDynamics ECI 模式下，LogarithmicWind 与 UniformWind 产生不同轨迹。"""
    lat0, lon0 = 0.0, 0.0
    mass = 10.0
    Aref = 0.1

    def _integrate(wind: UniformWind | LogarithmicWind) -> np.ndarray:
        cfg = SimConfig(
            mission="rocket",
            vehicle=VehicleConfig(mass_kg=mass, diameter_m=0.1, cd=0.3),
            launch=LaunchConfig(lat_deg=lat0, lon_deg=lon0),
            environment=EnvironmentConfig(),
            guidance=GuidanceConfig(),
            options=OptionsConfig(),
        )
        ctx = DynamicContext(
            cfg=cfg,
            atmosphere=StandardAtmosphere(),
            wind=wind,
            aero=ConstantAeroModel(cd=0.3),
            terrain=None,
            gravity_model="j2",
            options={},
        )
        dyn = PointMassDynamics(frame="ECI", mass=mass, Aref=Aref, options={"drag": True})
        r_ecef = np.array([6378137.0 + 1000.0, 0.0, 0.0], dtype=float)
        r_eci = ecef_to_eci(r_ecef, 0.0)
        v_eci = np.array([0.0, 0.0, 100.0], dtype=float)
        y0 = np.concatenate([r_eci, v_eci, [mass]])
        sol = solve_ivp(
            lambda t, y: dyn.rhs(t, y, ctx),
            (0.0, 10.0),
            y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
            max_step=0.5,
        )
        return np.asarray(sol.y[:, -1], dtype=float)

    y_uniform = _integrate(UniformWind(e=0.0, n=0.0, u=0.0))
    # 参考高度 10 m、风速 10 m/s 的对数风，在 1000 m 处风速更大
    y_log = _integrate(LogarithmicWind(u_ref=10.0, z_ref=10.0, z0=0.03, direction_deg=90.0))
    assert not np.allclose(y_uniform[0:3], y_log[0:3], atol=1e-3)


def test_make_ground_event_eci_uses_terrain_height_at_lat_lon() -> None:
    """make_ground_event ECI 分支在 terrain 存在时按地形高程判定。"""
    elev = np.array([[200.0]], dtype=float)
    terrain = TerrainModel(elev, TerrainExtent(0.0, 1.0, 0.0, 1.0))
    ev = make_ground_event(frame="ECI", terrain=terrain)

    r_ecef = np.array([6378137.0 + 250.0, 0.0, 0.0], dtype=float)
    r_eci = eci_to_ecef(r_ecef, 0.0)
    y = np.concatenate([r_eci, np.zeros(3)])
    # 250 m 高度 - 200 m 地形 = 50 m，事件值为正
    assert ev(0.0, y) > 0.0

    r_ecef2 = np.array([6378137.0 + 150.0, 0.0, 0.0], dtype=float)
    r_eci2 = eci_to_ecef(r_ecef2, 0.0)
    y2 = np.concatenate([r_eci2, np.zeros(3)])
    # 150 m 高度 - 200 m 地形 = -50 m，事件值为负
    assert ev(0.0, y2) < 0.0
