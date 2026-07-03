"""弹丸/火炮预设。

兼容层: 数据优先从 ``presets/projectiles.yaml`` 加载, 保留旧版 ``m107_config``
函数签名不变。
"""

from __future__ import annotations

import numpy as np

from ballistic_sim.config import (
    EnvironmentConfig,
    GuidanceConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    VehicleConfig,
)
from ballistic_sim.dynamics.mpm import MPMOptions, MPMDynamics
from ballistic_sim.models.aerodynamics import DRAG_G1, DRAG_G7
from ballistic_sim.phases.powered import PoweredPhase
from ballistic_sim.phases.terminal import TerminalPhase
from ballistic_sim.presets.loader import get_projectile, list_projectiles, make_aero_tables


def _projectile_config_from_preset(name: str) -> SimConfig:
    """由 YAML 预设构造 SimConfig。"""
    p = get_projectile(name)
    aero = make_aero_tables(p)
    drag_name = p.get("drag", "G1")
    drag_law = DRAG_G7 if drag_name == "G7" else DRAG_G1
    return SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(
            mass_kg=float(p["m"]),
            diameter_m=float(p["d"]),
            cd=float(p["i"]),
            area_ref_m2=None,
        ),
        launch=LaunchConfig(
            lat_deg=float(p["lat"]),
            lon_deg=float(p.get("lon", 0.0)),
            alt_m=float(p["h0"]),
            azimuth_deg=float(p["az0"]),
            elevation_deg=float(p["theta0"]),
            v0_m_s=float(p["v0"]),
            t0_s=0.0,
        ),
        environment=EnvironmentConfig(
            atmosphere="isa",
            wind_m_s=[0.0, 0.0, 0.0],
            gravity_model="wgs84",
        ),
        guidance=GuidanceConfig(),
        options=OptionsConfig(
            integrator="DOP853",
            rtol=1e-6,
            atol=1e-9,
            max_step=1.0,
            terminate_impact=True,
        ),
    )


def _projectile_dynamics_from_preset(name: str) -> MPMDynamics:
    """由 YAML 预设构造 MPM 动力学模块。"""
    p = get_projectile(name)
    aero = make_aero_tables(p)
    drag_name = p.get("drag", "G1")
    drag_law = DRAG_G7 if drag_name == "G7" else DRAG_G1
    return MPMDynamics(
        mass_kg=float(p["m"]),
        diameter_m=float(p["d"]),
        form_factor=float(p["i"]),
        drag_law=drag_law,
        twist_cal=float(p["twist_cal"]),
        Ix=float(p["Ix"]),
        It=float(p["It"]),
        CMa_table=aero["CMa_table"],
        CLa_table=aero["CLa_table"],
        Clp_table=aero["Clp_table"],
        options=MPMOptions(
            use_drag=True,
            use_wind=False,
            use_coriolis=True,
            use_spin=True,
            use_dynamic_alpha=False,
        ),
        lat_deg=float(p["lat"]),
    )


def m107_config() -> SimConfig:
    """M107 155mm 榴弹典型配置。"""
    return _projectile_config_from_preset("M107")


def projectile_phases(name: str) -> list:
    """由弹丸预设构造 [动力/无动力段, 终点] phase 列表。"""
    p = get_projectile(name)
    cfg = _projectile_config_from_preset(name)
    dyn = _projectile_dynamics_from_preset(name)
    return [
        PoweredPhase(
            name="无动力弹道",
            t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 3000.0),
            dynamics=dyn,
            guidance=None,
            m_dry=float(p["m"]),
            sep_name="落地",
        ),
        TerminalPhase(
            name="终点",
            t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 3000.0),
            dynamics=dyn,
        ),
    ]


def m107_phases() -> list:
    """M107 对应的 phase 列表。"""
    return projectile_phases("M107")


__all__ = [
    "m107_config",
    "m107_phases",
    "projectile_phases",
    "_projectile_config_from_preset",
    "_projectile_dynamics_from_preset",
    "list_projectiles",
]
