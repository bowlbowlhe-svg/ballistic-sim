"""运载火箭预设（CZ-2F/CZ-3B 等公开近似参数）。

兼容层: 数据优先从 ``presets/rockets.yaml`` 加载, 保留旧版 ``cz2f_config`` /
``cz2f_phases`` 函数签名不变。
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
from ballistic_sim.dynamics.powered_eci import PoweredECIDynamics
from ballistic_sim.models.propulsion import PropulsionModel
from ballistic_sim.phases.base import Phase
from ballistic_sim.phases.builder import build_phases
from ballistic_sim.phases.events import make_fairing_jettison_event
from ballistic_sim.phases.powered import PoweredPhase
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


def _guidance_from_rocket(r: Dict[str, Any]) -> Dict[str, Any]:
    """由火箭预设构造制导初值。"""
    launch = r.get("launch", {})
    guid = dict(r.get("guidance", {}))
    guid.setdefault("lat_deg", float(launch.get("lat_deg", 0.0)))
    guid.setdefault("lon_deg", float(launch.get("lon_deg", 0.0)))
    guid.setdefault("azimuth_deg", float(launch.get("azimuth_deg", 90.0)))
    return guid


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


def rocket_full_chain(
    name: str = "CZ2F",
    payload_mass_kg: Optional[float] = None,
) -> list[Phase]:
    """由 YAML 火箭预设构造完整多级 Phase 链。"""
    r = get_rocket(name)
    return _set_terminal_target(
        build_phases(rocket_full_config(name, payload_mass_kg=payload_mass_kg)),
        r.get("target"),
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


def rocket_phases(
    cfg: SimConfig,
    name: str = "CZ2F",
    payload_mass_kg: Optional[float] = None,
) -> list[Phase]:
    """由 YAML 火箭预设构造 Phase 列表。

    若 ``cfg.vehicle.stages`` 已提供，则复用 ``build_phases`` 自动构建完整链，
    以演示与 ``StageConfig`` 的兼容性；否则使用传统手动三段/四段装配。
    """
    if cfg.vehicle.stages is not None:
        r = get_rocket(name)
        return _set_terminal_target(build_phases(cfg), r.get("target"))

    r = get_rocket(name)
    payload = (
        payload_mass_kg if payload_mass_kg is not None else float(r.get("payload_mass_kg", 0.0))
    )
    s1, s2, s3, _m_p1, _m_p2 = _build_parallel_stages(
        r["booster"],
        r["core1"],
        r["core2"],
        r["fairing"],
        float(r["Aref_core_m2"]),
        float(r["Aref_parallel_m2"]),
        payload,
        r.get("core3"),
    )
    base_guid = _guidance_from_rocket(r)
    base_guid["azimuth_deg"] = cfg.launch.azimuth_deg
    base_guid["lat_deg"] = cfg.launch.lat_deg
    base_guid["lon_deg"] = cfg.launch.lon_deg

    guid_P1 = dict(base_guid)
    dyn_P1 = PoweredECIDynamics(stage=s1, guidance=guid_P1)
    t_P1_est = float(s1["m_prop"]) / dyn_P1.prop.mdot
    ph_P1 = PoweredPhase(
        name=s1["name"],
        t_span=(cfg.launch.t0_s, cfg.launch.t0_s + t_P1_est * 1.5),
        dynamics=dyn_P1,
        guidance=guid_P1,
        m_dry=float(s1["m_dry"]),
        m_after_separation=float(s2["m_dry"] + s2["m_prop"]),
        sep_name="助推分离",
    )

    guid_P2 = dict(base_guid)
    dyn_P2 = PoweredECIDynamics(stage=s2, guidance=guid_P2)
    t_P2_est = float(s2["m_prop"]) / dyn_P2.prop.mdot
    ph_P2 = PoweredPhase(
        name=s2["name"],
        t_span=(cfg.launch.t0_s, cfg.launch.t0_s + t_P2_est * 1.5),
        dynamics=dyn_P2,
        guidance=guid_P2,
        m_dry=float(s2["m_dry"]),
        m_after_separation=float(s3["m_dry"] + s3["m_prop"]),
        sep_name="一二级分离",
    )

    t_P1_and_P2 = t_P1_est + t_P2_est
    t_P3_est = float(s3["m_prop"]) / PropulsionModel.from_stage(s3).mdot
    guid_P3 = _upperstage_guidance(base_guid, t_P1_and_P2, t_P3_est)
    dyn_P3 = PoweredECIDynamics(stage=s3, guidance=guid_P3, use_upperstage=True)
    fairing_h_m = float(r["fairing"].get("jettison_h_km", 110.0)) * 1e3
    fairing_ev = make_fairing_jettison_event(
        mode="altitude",
        h_m=fairing_h_m,
        frame="ECI",
    )
    ph_P3 = PoweredPhase(
        name=s3["name"],
        t_span=(cfg.launch.t0_s, cfg.launch.t0_s + t_P3_est * 1.5),
        dynamics=dyn_P3,
        guidance=guid_P3,
        m_dry=float(s3["m_dry"]),
        sep_name="SECO",
    )
    ph_P3.events.append(fairing_ev)

    phases: list[Phase] = [ph_P1, ph_P2, ph_P3]
    t_accum = t_P1_and_P2 + t_P3_est
    dyn_last = dyn_P3

    core3 = r.get("core3")
    if core3 is not None and r.get("core3_ignition"):
        mdot_core3 = _mass_flow(core3)
        t_P4_est = float(core3["m_prop"]) / mdot_core3
        s4 = dict(
            name=f"P4 {core3.get('name', '三级')}",
            thrust_sl=float(core3.get("thrust_sl", 0.0)),
            thrust_vac=float(core3["thrust_vac"]),
            isp_vac=float(core3["isp_vac"]),
            m_prop=float(core3["m_prop"]),
            m_dry=float(core3["m_dry"]) + payload,
            Aref=float(np.pi * (float(core3.get("diameter_m", r["diameter_core_m"])) / 2.0) ** 2),
            burn_time=t_P4_est,
        )
        guid_P4 = _upperstage_guidance(base_guid, t_accum, t_P4_est)
        dyn_P4 = PoweredECIDynamics(stage=s4, guidance=guid_P4, use_upperstage=True)
        ph_P4 = PoweredPhase(
            name=str(s4["name"]),
            t_span=(cfg.launch.t0_s, cfg.launch.t0_s + t_P4_est * 1.5),
            dynamics=dyn_P4,
            guidance=guid_P4,
            m_dry=float(s4.get("m_dry", 0.0)),  # type: ignore[arg-type]
            sep_name="三级分离",
        )
        phases.append(ph_P4)
        t_accum += t_P4_est
        dyn_last = dyn_P4

    target = r.get("target")
    terminal_target = None
    if isinstance(target, dict):
        terminal_target = {
            "peri_km": float(target.get("target_peri_km", 200.0)),
            "apo_km": float(target.get("target_apo_km", 350.0)),
        }
    terminal = TerminalPhase(
        name="轨道插入",
        t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 3600.0),
        dynamics=dyn_last,
        target=terminal_target,
    )
    phases.append(terminal)
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


def cz2f_phases(
    cfg: SimConfig,
    payload_mass_kg: float = 8000.0,
) -> list[Phase]:
    """由 SimConfig 构造 CZ-2F 三段 Phase 列表。"""
    return rocket_phases(cfg, name="CZ2F", payload_mass_kg=payload_mass_kg)


def _upperstage_guidance(
    base_guid: Dict[str, Any], t_us_start: float, t_us_dur: float
) -> Dict[str, Any]:
    """上面级线性俯仰制导。"""
    g = dict(base_guid)
    g.update(
        dict(
            t_us_start=float(t_us_start),
            gamma_end_deg=0.0,
            gamma0_deg=None,
            pitch_rate_dps=None,
            t_us_dur=float(t_us_dur),
        )
    )
    return g


__all__ = [
    "cz2f_config",
    "cz2f_phases",
    "rocket_config",
    "rocket_full_config",
    "rocket_full_chain",
    "rocket_phases",
    "rocket_stages",
    "list_rockets",
]
