"""弹丸/火炮预设。

兼容层: 数据优先从 ``presets/projectiles.yaml`` 加载, 保留旧版 ``m107_config``
函数签名不变。
"""

from __future__ import annotations


from ballistic_sim.config import (
    EnvironmentConfig,
    GuidanceConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    VehicleConfig,
)
from ballistic_sim.phases.builder import build_phases
from ballistic_sim.presets.loader import get_projectile, list_projectiles


def _projectile_config_from_preset(name: str) -> SimConfig:
    """由 YAML 预设构造 SimConfig。"""
    from typing import Literal

    p = get_projectile(name)
    drag_name = p.get("drag", "G1")
    drag_law: Literal["G1", "G7"] = "G7" if drag_name == "G7" else "G1"
    return SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(
            mass_kg=float(p["m"]),
            diameter_m=float(p["d"]),
            cd=float(p["i"]),
            area_ref_m2=None,
            drag_law=drag_law,
            twist_cal=float(p.get("twist_cal", 20.0)),
            Ix=float(p.get("Ix", 0.1)),
            It=float(p.get("It", 1.0)),
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


def m107_config() -> SimConfig:
    """M107 155mm 榴弹典型配置。"""
    return _projectile_config_from_preset("M107")


def projectile_phases(name: str) -> list:
    """由弹丸预设构造 [动力/无动力段, 终点] phase 列表。

    .. deprecated:: v0.4.0
        后请使用 ``build_phases(cfg)``。
    """
    cfg = _projectile_config_from_preset(name)
    return build_phases(cfg)


def m107_phases() -> list:
    """M107 对应的 phase 列表。

    .. deprecated:: v0.4.0
        后请使用 ``build_phases(cfg)``。
    """
    return projectile_phases("M107")


__all__ = [
    "m107_config",
    "m107_phases",
    "projectile_phases",
    "_projectile_config_from_preset",
    "list_projectiles",
]
