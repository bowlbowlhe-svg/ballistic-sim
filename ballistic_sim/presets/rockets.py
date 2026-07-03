"""运载火箭预设（CZ-2F 公开近似参数）。"""

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
from ballistic_sim.constants import G0_STANDARD
from ballistic_sim.dynamics.powered_eci import PoweredECIDynamics
from ballistic_sim.models.propulsion import PropulsionModel
from ballistic_sim.phases.base import Phase
from ballistic_sim.phases.coasting import CoastingPhase
from ballistic_sim.phases.events import make_fairing_event_h
from ballistic_sim.phases.powered import PoweredPhase
from ballistic_sim.phases.terminal import TerminalPhase

# CZ-2F 组件级参数（公开近似）
_D_CORE = 3.35
_AREF_CORE = float(np.pi * (_D_CORE / 2.0) ** 2)
_AREF_PARALLEL = 13.0

_BOOSTER = dict(
    name="CZ2F-助推×4",
    thrust_sl=4 * 740.0e3,
    thrust_vac=4 * 816.0e3,
    isp_vac=289.0,
    burn_time=127.0,
    m_prop=4 * 37700.0,
    m_dry=4 * 3000.0,
    Aref=_AREF_PARALLEL,
)

_CORE1 = dict(
    name="CZ2F-芯一级",
    thrust_sl=4 * 740.0e3,
    thrust_vac=4 * 816.0e3,
    isp_vac=289.0,
    burn_time=160.0,
    m_prop=187000.0,
    m_dry=9500.0,
    Aref=_AREF_CORE,
)

_CORE2 = dict(
    name="CZ2F-芯二级",
    thrust_vac=740.0e3 + 4 * 47.1e3,
    thrust_sl=820.0e3,
    isp_vac=298.0,
    isp_sl=289.0,
    burn_time=350.0,
    m_prop=86000.0,
    m_dry=4500.0,
    Aref=_AREF_CORE,
)

_FAIRING = dict(m=3000.0, jettison_q_pa=100.0, jettison_h_km=110.0)

_PAYLOAD = dict(m=8000.0)


def _mass_flow(stage: dict) -> float:
    return float(stage["thrust_vac"]) / (float(stage["isp_vac"]) * G0_STANDARD)


def _build_cz2f_stages(payload_mass: float = 8000.0):
    """构造 CZ-2F 三段等效级。"""
    m_pl = float(payload_mass)
    mdot_boost = _mass_flow(_BOOSTER)
    mdot_core1 = _mass_flow(_CORE1)
    mdot_core2 = _mass_flow(_CORE2)

    t_boost = _BOOSTER["m_prop"] / mdot_boost
    m_core1_in_P1 = mdot_core1 * t_boost
    m_core1_in_P2 = _CORE1["m_prop"] - m_core1_in_P1

    m_above_P3_burnout = _CORE2["m_dry"] + m_pl
    m_above_P2_burnout = _CORE1["m_dry"] + _CORE2["m_prop"] + _CORE2["m_dry"] + _FAIRING["m"] + m_pl
    m_above_P1_burnout = (
        _BOOSTER["m_dry"]
        + _CORE1["m_dry"]
        + m_core1_in_P2
        + _CORE2["m_prop"]
        + _CORE2["m_dry"]
        + _FAIRING["m"]
        + m_pl
    )

    thr_sl_P1 = _BOOSTER["thrust_sl"] + _CORE1["thrust_sl"]
    thr_vac_P1 = _BOOSTER["thrust_vac"] + _CORE1["thrust_vac"]
    mdot_P1 = mdot_boost + mdot_core1
    isp_vac_P1 = thr_vac_P1 / (mdot_P1 * G0_STANDARD)

    stage_P1 = dict(
        name="P1 起飞(并联)",
        thrust_sl=thr_sl_P1,
        thrust_vac=thr_vac_P1,
        isp_vac=isp_vac_P1,
        m_prop=_BOOSTER["m_prop"] + m_core1_in_P1,
        m_dry=m_above_P1_burnout,
        Aref=_AREF_PARALLEL,
        burn_time=t_boost,
    )
    stage_P2 = dict(
        name="P2 芯一级续飞",
        thrust_sl=_CORE1["thrust_sl"],
        thrust_vac=_CORE1["thrust_vac"],
        isp_vac=_CORE1["isp_vac"],
        m_prop=m_core1_in_P2,
        m_dry=m_above_P2_burnout,
        Aref=_AREF_CORE,
        burn_time=m_core1_in_P2 / mdot_core1,
    )
    stage_P3 = dict(
        name="P3 芯二级",
        thrust_sl=_CORE2["thrust_sl"],
        thrust_vac=_CORE2["thrust_vac"],
        isp_vac=_CORE2["isp_vac"],
        m_prop=_CORE2["m_prop"],
        m_dry=m_above_P3_burnout,
        Aref=_AREF_CORE,
        burn_time=_CORE2["m_prop"] / mdot_core2,
    )
    return stage_P1, stage_P2, stage_P3


def _guidance(
    lat_deg: float,
    lon_deg: float,
    azimuth_deg: float,
) -> dict:
    return dict(
        lat_deg=float(lat_deg),
        lon_deg=float(lon_deg),
        azimuth_deg=float(azimuth_deg),
        t_pitchover=12.0,
        kick_deg=4.0,
        t_kick_end=30.0,
    )


def cz2f_config(
    payload_mass_kg: float = 8000.0,
    target_peri_km: float = 200.0,
    target_apo_km: float = 350.0,
    target_inc_deg: float = 42.0,
) -> SimConfig:
    """CZ-2F + 酒泉 + 神舟 LEO 的 SimConfig。"""
    lat = 40.96
    lon = 100.29
    # 球面三角反算方位角
    ratio = np.cos(np.deg2rad(target_inc_deg)) / np.cos(np.deg2rad(lat))
    az = float(np.degrees(np.arcsin(np.clip(ratio, -1.0, 1.0))))
    return SimConfig(
        mission="rocket",
        vehicle=VehicleConfig(
            mass_kg=sum(
                [
                    _BOOSTER["m_prop"] + _BOOSTER["m_dry"],
                    _CORE1["m_prop"] + _CORE1["m_dry"],
                    _CORE2["m_prop"] + _CORE2["m_dry"],
                    _FAIRING["m"],
                    payload_mass_kg,
                ]
            ),
            diameter_m=_D_CORE,
            cd=0.3,
            area_ref_m2=_AREF_CORE,
        ),
        launch=LaunchConfig(
            lat_deg=lat,
            lon_deg=lon,
            alt_m=1000.0,
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
            kick_deg=4.0,
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


def cz2f_phases(
    cfg: SimConfig,
    payload_mass_kg: float = 8000.0,
) -> list[Phase]:
    """由 SimConfig 构造 CZ-2F 三段 Phase 列表。"""
    s1, s2, s3 = _build_cz2f_stages(payload_mass_kg)
    base_guid = _guidance(
        cfg.launch.lat_deg,
        cfg.launch.lon_deg,
        cfg.launch.azimuth_deg,
    )

    # P1 起飞（并联）
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

    # P2 芯一级续飞
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

    # P3 芯二级（抛罩后入轨）
    t_P3_est = float(s3["m_prop"]) / PropulsionModel.from_stage(s3).mdot
    guid_P3 = _upperstage_guidance(base_guid, t_P1_est + t_P2_est, t_P3_est)
    dyn_P3 = PoweredECIDynamics(stage=s3, guidance=guid_P3, use_upperstage=True)
    fairing_ev = make_fairing_event_h(_FAIRING["jettison_h_km"] * 1e3, frame="ECI")
    ph_P3 = PoweredPhase(
        name=s3["name"],
        t_span=(cfg.launch.t0_s, cfg.launch.t0_s + t_P3_est * 1.5),
        dynamics=dyn_P3,
        guidance=guid_P3,
        m_dry=float(s3["m_dry"]),
        sep_name="SECO",
    )
    ph_P3.events.append(fairing_ev)

    terminal = TerminalPhase(
        name="轨道插入",
        t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 3600.0),
        dynamics=dyn_P3,
    )
    return [ph_P1, ph_P2, ph_P3, terminal]


def _upperstage_guidance(base_guid: dict, t_us_start: float, t_us_dur: float) -> dict:
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
