"""运载火箭预设（CZ-2F/CZ-3B 等公开近似参数）。

兼容层: 数据优先从 ``presets/rockets.yaml`` 加载, 保留旧版 ``cz2f_config``
函数签名不变。
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
from ballistic_sim.constants import G0_STANDARD
from ballistic_sim.phases.base import Phase
from ballistic_sim.phases.terminal import TerminalPhase
from ballistic_sim.presets.loader import get_rocket, list_rockets

_G0 = G0_STANDARD


def _mass_flow(stage: Dict[str, Any]) -> float:
    """由真空推力/比冲计算质量流率。"""
    return float(stage["thrust_vac"]) / (float(stage["isp_vac"]) * _G0)


def _build_parallel_stages(
    booster: Dict[str, Any],
    core1: Dict[str, Any],
    core2: Dict[str, Any],
    fairing: Dict[str, Any],
    aref_core: float,
    aref_parallel: float,
    payload_mass: float,
    core3: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], float, float]:
    """构造并联助推火箭的三段等效级。

    额外返回 ``m_core1_in_P1`` 与 ``m_core1_in_P2``，供 ``rocket_stages`` 转换为
    ``StageConfig`` 序列时使用。
    """
    m_pl = float(payload_mass)
    m_core3 = 0.0
    if core3 is not None:
        m_core3 = float(core3.get("m_prop", 0.0)) + float(core3.get("m_dry", 0.0))

    mdot_boost = _mass_flow(booster)
    mdot_core1 = _mass_flow(core1)
    mdot_core2 = _mass_flow(core2)

    t_boost = float(booster["m_prop"]) / mdot_boost
    m_core1_in_P1 = mdot_core1 * t_boost
    m_core1_in_P2 = float(core1["m_prop"]) - m_core1_in_P1

    m_above_P3_burnout = float(core2["m_dry"]) + m_core3 + m_pl
    m_above_P2_burnout = (
        float(core1["m_dry"])
        + float(core2["m_prop"])
        + float(core2["m_dry"])
        + float(fairing["m"])
        + m_core3
        + m_pl
    )
    m_above_P1_burnout = (
        float(booster["m_dry"])
        + float(core1["m_dry"])
        + m_core1_in_P2
        + float(core2["m_prop"])
        + float(core2["m_dry"])
        + float(fairing["m"])
        + m_core3
        + m_pl
    )

    thr_sl_P1 = float(booster["thrust_sl"]) + float(core1["thrust_sl"])
    thr_vac_P1 = float(booster["thrust_vac"]) + float(core1["thrust_vac"])
    mdot_P1 = mdot_boost + mdot_core1
    isp_vac_P1 = thr_vac_P1 / (mdot_P1 * _G0)

    s1 = dict(
        name=f"P1 起飞(并联: {booster.get('name', '助推')}+{core1.get('name', '芯一级')})",
        thrust_sl=thr_sl_P1,
        thrust_vac=thr_vac_P1,
        isp_vac=isp_vac_P1,
        m_prop=float(booster["m_prop"]) + m_core1_in_P1,
        m_dry=m_above_P1_burnout,
        Aref=float(aref_parallel),
        burn_time=t_boost,
    )
    s2 = dict(
        name=f"P2 {core1.get('name', '芯一级续飞')}",
        thrust_sl=float(core1["thrust_sl"]),
        thrust_vac=float(core1["thrust_vac"]),
        isp_vac=float(core1["isp_vac"]),
        m_prop=m_core1_in_P2,
        m_dry=m_above_P2_burnout,
        Aref=float(aref_core),
        burn_time=m_core1_in_P2 / mdot_core1,
    )
    s3 = dict(
        name=f"P3 {core2.get('name', '芯二级')}",
        thrust_sl=float(core2["thrust_sl"]),
        thrust_vac=float(core2["thrust_vac"]),
        isp_vac=float(core2["isp_vac"]),
        m_prop=float(core2["m_prop"]),
        m_dry=m_above_P3_burnout,
        Aref=float(aref_core),
        burn_time=float(core2["m_prop"]) / mdot_core2,
    )
    return s1, s2, s3, m_core1_in_P1, m_core1_in_P2


def _azimuth_for_inclination(lat_deg: float, inc_deg: float) -> float:
    """球面三角反算发射方位角。"""
    ratio = np.cos(np.deg2rad(inc_deg)) / np.cos(np.deg2rad(lat_deg))
    return float(np.degrees(np.arcsin(np.clip(ratio, -1.0, 1.0))))


def rocket_stages(
    name: str = "CZ2F",
    payload_mass_kg: Optional[float] = None,
) -> List[StageConfig]:
    """由火箭预设构造与 ``StageConfig`` / ``build_phases`` 兼容的 stage 列表。

    将并联助推 + 芯一级续飞等效为串行 StageConfig：
    stage1 为 ``助推+芯一级并联段`` (推力/质量为两者之和),
    stage2 为 ``芯一级续飞段``,
    stage3 为 ``芯二级``,
    若配置 ``core3_ignition: true`` 则追加 stage4 ``三级``。
    """
    r = get_rocket(name)
    payload = (
        payload_mass_kg if payload_mass_kg is not None else float(r.get("payload_mass_kg", 0.0))
    )
    booster = r["booster"]
    core1 = r["core1"]
    core2 = r["core2"]
    fairing = r["fairing"]
    core3 = r.get("core3")
    aref_core = float(r["Aref_core_m2"])
    aref_parallel = float(r["Aref_parallel_m2"])

    s1, s2, s3, m_core1_in_P1, m_core1_in_P2 = _build_parallel_stages(
        booster, core1, core2, fairing, aref_core, aref_parallel, payload, core3
    )

    def _stage(name: str, s: Dict[str, Any], dry: float, diameter: float) -> StageConfig:
        return StageConfig(
            name=name,
            thrust_sl=float(s["thrust_sl"]),
            thrust_vac=float(s["thrust_vac"]),
            isp_vac=float(s["isp_vac"]),
            m_prop=float(s["m_prop"]),
            m_dry=float(dry),
            diameter_m=float(diameter),
            Aref=float(np.pi * (diameter / 2.0) ** 2),
            burn_time=float(s["burn_time"]),
        )

    stages: List[StageConfig] = [
        _stage(
            f"P1 {booster.get('name', '助推')}+{core1.get('name', '芯一级')}",
            s1,
            float(booster["m_dry"]),
            float(booster.get("diameter_m", r["diameter_core_m"])),
        ),
        _stage(
            f"P2 {core1.get('name', '芯一级续飞')}",
            s2,
            float(core1["m_dry"]),
            float(core1.get("diameter_m", r["diameter_core_m"])),
        ),
        _stage(
            f"P3 {core2.get('name', '芯二级')}",
            s3,
            float(core2["m_dry"]),
            float(core2.get("diameter_m", r["diameter_core_m"])),
        ),
    ]

    if core3 is not None and r.get("core3_ignition"):
        stages.append(
            StageConfig(
                name=f"P4 {core3.get('name', '三级')}",
                thrust_sl=float(core3.get("thrust_sl", 0.0)),
                thrust_vac=float(core3["thrust_vac"]),
                isp_vac=float(core3["isp_vac"]),
                m_prop=float(core3["m_prop"]),
                m_dry=float(core3["m_dry"]),
                diameter_m=float(core3.get("diameter_m", r["diameter_core_m"])),
                Aref=float(
                    np.pi * (float(core3.get("diameter_m", r["diameter_core_m"])) / 2.0) ** 2
                ),
                burn_time=float(core3["m_prop"]) / _mass_flow(core3),
            )
        )

    return stages


def rocket_full_config(
    name: str = "CZ2F",
    payload_mass_kg: Optional[float] = None,
) -> SimConfig:
    """由 YAML 火箭预设构造完整任务链可用的 ``SimConfig`` (含多级 stages)。"""
    r = get_rocket(name)
    launch = r.get("launch", {})
    target = r.get("target", {})
    payload = (
        payload_mass_kg if payload_mass_kg is not None else float(r.get("payload_mass_kg", 0.0))
    )
    stages = rocket_stages(name, payload_mass_kg=payload)
    liftoff_mass = sum(s.m_dry + s.m_prop for s in stages) + payload
    inc = float(target.get("target_inc_deg", 42.0))
    lat = float(launch.get("lat_deg", 0.0))
    lon = float(launch.get("lon_deg", 0.0))
    az = _azimuth_for_inclination(lat, inc)
    guid = r.get("guidance", {})
    return SimConfig(
        mission="rocket",
        vehicle=VehicleConfig(
            mass_kg=liftoff_mass,
            diameter_m=float(r["diameter_core_m"]),
            cd=0.3,
            area_ref_m2=float(r["Aref_core_m2"]),
            stages=stages,
        ),
        launch=LaunchConfig(
            lat_deg=lat,
            lon_deg=lon,
            alt_m=float(launch.get("alt_m", 0.0)),
            azimuth_deg=az,
            elevation_deg=90.0,
            v0_m_s=0.0,
            t0_s=0.0,
        ),
        environment=EnvironmentConfig(
            atmosphere="isa",
            wind_m_s=[0.0, 0.0, 0.0],
            gravity_model="j2",
        ),
        guidance=GuidanceConfig(
            guidance_law="none",
            kick_deg=float(guid.get("kick_deg", 4.0)),
            t_pitchover=float(guid.get("t_pitchover", 10.0)),
            t_kick_end=float(guid.get("t_kick_end", 25.0)),
        ),
        options=OptionsConfig(
            integrator="DOP853",
            rtol=1e-9,
            atol=1e-6,
            max_step=1.0,
            terminate_apogee=False,
            terminate_impact=True,
        ),
    )


def rocket_config(
    name: str = "CZ2F",
    payload_mass_kg: Optional[float] = None,
    target_peri_km: Optional[float] = None,
    target_apo_km: Optional[float] = None,
    target_inc_deg: Optional[float] = None,
) -> SimConfig:
    """由 YAML 火箭预设构造 SimConfig。"""
    r = get_rocket(name)
    launch = r.get("launch", {})
    target = r.get("target", {})
    payload = (
        payload_mass_kg if payload_mass_kg is not None else float(r.get("payload_mass_kg", 0.0))
    )
    inc = (
        target_inc_deg if target_inc_deg is not None else float(target.get("target_inc_deg", 42.0))
    )

    lat = float(launch.get("lat_deg", 0.0))
    lon = float(launch.get("lon_deg", 0.0))
    az = _azimuth_for_inclination(lat, inc)

    booster = r["booster"]
    core1 = r["core1"]
    core2 = r["core2"]
    fairing = r["fairing"]
    core3 = r.get("core3")
    liftoff_mass = (
        float(booster["m_prop"])
        + float(booster["m_dry"])
        + float(core1["m_prop"])
        + float(core1["m_dry"])
        + float(core2["m_prop"])
        + float(core2["m_dry"])
        + float(fairing["m"])
        + float(payload)
    )
    if core3 is not None:
        liftoff_mass += float(core3.get("m_prop", 0.0)) + float(core3.get("m_dry", 0.0))

    return SimConfig(
        mission="rocket",
        vehicle=VehicleConfig(
            mass_kg=liftoff_mass,
            diameter_m=float(r["diameter_core_m"]),
            cd=0.3,
            area_ref_m2=float(r["Aref_core_m2"]),
        ),
        launch=LaunchConfig(
            lat_deg=lat,
            lon_deg=lon,
            alt_m=float(launch.get("alt_m", 0.0)),
            azimuth_deg=az,
            elevation_deg=90.0,
            v0_m_s=0.0,
            t0_s=0.0,
        ),
        environment=EnvironmentConfig(
            atmosphere="isa",
            wind_m_s=[0.0, 0.0, 0.0],
            gravity_model="j2",
        ),
        guidance=GuidanceConfig(
            guidance_law="none",
            kick_deg=float(r.get("guidance", {}).get("kick_deg", 4.0)),
        ),
        options=OptionsConfig(
            integrator="DOP853",
            rtol=1e-9,
            atol=1e-6,
            max_step=1.0,
            terminate_apogee=False,
            terminate_impact=True,
        ),
    )


def _set_terminal_target(phases: list[Phase], target: Optional[Dict[str, Any]]) -> list[Phase]:
    """将轨道目标写入最后一个 TerminalPhase（如果存在）。"""
    if phases and target is not None and isinstance(phases[-1], TerminalPhase):
        phases[-1].target = {
            "peri_km": float(target.get("target_peri_km", 200.0)),
            "apo_km": float(target.get("target_apo_km", 350.0)),
        }
    return phases


def cz2f_config(
    payload_mass_kg: float = 8000.0,
    target_peri_km: float = 200.0,
    target_apo_km: float = 350.0,
    target_inc_deg: float = 42.0,
) -> SimConfig:
    """CZ-2F + 酒泉 + 神舟 LEO 的 SimConfig。"""
    return rocket_config(
        name="CZ2F",
        payload_mass_kg=payload_mass_kg,
        target_peri_km=target_peri_km,
        target_apo_km=target_apo_km,
        target_inc_deg=target_inc_deg,
    )


__all__ = [
    "cz2f_config",
    "rocket_config",
    "rocket_full_config",
    "rocket_stages",
    "list_rockets",
]
