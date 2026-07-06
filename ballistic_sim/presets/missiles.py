"""弹道导弹预设。

兼容层: 数据优先从 ``presets/missiles.yaml`` 加载。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ballistic_sim.config import (
    EnvironmentConfig,
    GuidanceConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    StageConfig,
    VehicleConfig,
)
from ballistic_sim.presets.loader import get_missile, list_missiles


def _mass_flow(stage: Dict[str, Any]) -> float:
    from ballistic_sim.constants import G0_STANDARD

    return float(stage["thrust_vac"]) / (float(stage["isp_vac"]) * G0_STANDARD)


def _stages_from_missile(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    """由导弹预设构造串联记账的 stage dict 列表。"""
    raw = m.get("stages", [])
    payload = float(m["payload_mass"])
    n = len(raw)
    stages: List[Dict[str, Any]] = []
    for i, s in enumerate(raw):
        m_above = sum(float(raw[j]["m_dry"]) + float(raw[j]["m_prop"]) for j in range(i + 1, n))
        m_dry_phase = float(s["m_dry"]) + m_above + payload
        m_prop = float(s["m_prop"])
        stages.append(
            dict(
                name=str(s["name"]),
                thrust_vac=float(s["thrust_vac"]),
                thrust_sl=float(s["thrust_sl"]),
                isp_vac=float(s["isp_vac"]),
                m_prop=m_prop,
                m_dry=m_dry_phase,
                Aref=float(np.pi * (float(s["diameter_m"]) / 2.0) ** 2),
                sep_dmass=0.0 if i == n - 1 else float(s["m_dry"]),
                m_after_sep=m_dry_phase - (0.0 if i == n - 1 else float(s["m_dry"])),
                burn_time=m_prop / _mass_flow(s),
            )
        )
    return stages


def _stage_configs_from_missile(m: Dict[str, Any]) -> List[StageConfig]:
    """将导弹预设的各级转换为 ``StageConfig`` 列表。"""
    stages: List[StageConfig] = []
    for s in m.get("stages", []):
        d = float(s["diameter_m"])
        stages.append(
            StageConfig(
                name=str(s["name"]),
                thrust_sl=float(s["thrust_sl"]),
                thrust_vac=float(s["thrust_vac"]),
                isp_vac=float(s["isp_vac"]),
                m_prop=float(s["m_prop"]),
                m_dry=float(s["m_dry"]),
                diameter_m=d,
                Aref=float(np.pi * (d / 2.0) ** 2),
                burn_time=float(s["m_prop"]) / _mass_flow(s),
            )
        )
    return stages


def _warhead_mass(m: Dict[str, Any]) -> float:
    """再入飞行器质量: 优先使用 ``warhead.mass``, 否则 ``payload_mass``。"""
    warhead = m.get("warhead")
    if isinstance(warhead, dict):
        return float(warhead.get("mass", m["payload_mass"]))
    return float(m["payload_mass"])


def _fairing_from_missile(m: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """返回整流罩配置字典, 若未配置则返回 None。"""
    fairing = m.get("fairing")
    if isinstance(fairing, dict):
        return dict(fairing)
    return None


def missile_config(name: str) -> SimConfig:
    """由 YAML 导弹预设构造 SimConfig。"""
    m = get_missile(name)
    site = m["_site"]
    guid = m["guidance"]
    stages = _stages_from_missile(m)
    liftoff_mass = float(stages[0]["m_dry"]) + float(stages[0]["m_prop"])
    return SimConfig(
        mission="missile",
        vehicle=VehicleConfig(
            mass_kg=liftoff_mass,
            diameter_m=float(stages[0].get("diameter_m", 1.0)),
            cd=0.3,
        ),
        launch=LaunchConfig(
            lat_deg=float(site["lat"]),
            lon_deg=float(site["lon"]),
            alt_m=float(site["alt"]),
            azimuth_deg=float(guid["azimuth_deg"]),
            elevation_deg=90.0,
            v0_m_s=0.0,
            t0_s=0.0,
        ),
        environment=EnvironmentConfig(
            atmosphere="isa",
            gravity_model="j2",
        ),
        guidance=GuidanceConfig(
            guidance_law="none",
            kick_deg=float(guid["kick_deg"]),
            t_pitchover=float(guid.get("t_pitchover", 10.0)),
            t_kick_end=float(guid.get("t_kick_end", 25.0)),
        ),
        options=OptionsConfig(
            integrator="DOP853",
            rtol=1e-6,
            atol=1e-9,
            max_step=1.0,
            terminate_impact=True,
        ),
    )


def missile_full_config(name: str) -> SimConfig:
    """由 YAML 导弹预设构造完整任务链可用的 ``SimConfig`` (含多级 stages)。"""
    m = get_missile(name)
    site = m["_site"]
    guid = m["guidance"]
    stage_configs = _stage_configs_from_missile(m)
    payload = _warhead_mass(m)
    liftoff_mass = sum(s.m_dry + s.m_prop for s in stage_configs) + payload
    mission = str(m.get("mission", "missile"))
    return SimConfig(
        mission=mission,
        vehicle=VehicleConfig(
            mass_kg=liftoff_mass,
            diameter_m=stage_configs[0].diameter_m or 1.0,
            cd=0.3,
            area_ref_m2=stage_configs[0].Aref,
            stages=stage_configs,
        ),
        launch=LaunchConfig(
            lat_deg=float(site["lat"]),
            lon_deg=float(site["lon"]),
            alt_m=float(site["alt"]),
            azimuth_deg=float(guid["azimuth_deg"]),
            elevation_deg=90.0,
            v0_m_s=0.0,
            t0_s=0.0,
        ),
        environment=EnvironmentConfig(
            atmosphere="isa",
            gravity_model="j2",
        ),
        guidance=GuidanceConfig(
            guidance_law="none",
            kick_deg=float(guid["kick_deg"]),
            t_pitchover=float(guid.get("t_pitchover", 10.0)),
            t_kick_end=float(guid.get("t_kick_end", 25.0)),
        ),
        options=OptionsConfig(
            integrator="DOP853",
            rtol=1e-6,
            atol=1e-9,
            max_step=1.0,
            terminate_impact=True,
        ),
    )


__all__ = [
    "missile_config",
    "missile_full_config",
    "list_missiles",
]
