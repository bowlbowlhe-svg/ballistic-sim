"""弹道导弹预设。

兼容层: 数据优先从 ``presets/missiles.yaml`` 加载。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ballistic_sim.config import (
    EnvironmentConfig,
    GuidanceConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    VehicleConfig,
)
from ballistic_sim.constants import WGS84_A
from ballistic_sim.dynamics.powered_eci import PoweredECIDynamics
from ballistic_sim.frames import ecef_to_geodetic
from ballistic_sim.phases.base import Phase
from ballistic_sim.phases.coasting import CoastingPhase
from ballistic_sim.phases.powered import PoweredPhase
from ballistic_sim.phases.reentry import ReentryPhase
from ballistic_sim.phases.terminal import TerminalPhase
from ballistic_sim.presets.loader import get_missile, list_missiles

H_ENTRY = 100e3


def _make_altitude_event(h_m: float, direction: int, terminal: bool = False):
    def ev(t: float, y: np.ndarray) -> float:
        r = np.asarray(y[0:3], dtype=float)
        return float(np.linalg.norm(r) - WGS84_A - h_m)

    ev.terminal = terminal  # type: ignore[attr-defined]
    ev.direction = direction  # type: ignore[attr-defined]
    return ev


def _make_apogee_event():
    def ev(t: float, y: np.ndarray) -> float:
        r = np.asarray(y[0:3], dtype=float)
        v = np.asarray(y[3:6], dtype=float)
        return float(np.dot(r, v))

    ev.terminal = False  # type: ignore[attr-defined]
    ev.direction = -1  # type: ignore[attr-defined]
    return ev


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
        ),
        options=OptionsConfig(
            integrator="DOP853",
            rtol=1e-6,
            atol=1e-9,
            max_step=1.0,
            terminate_impact=True,
        ),
    )


def missile_phases(name: str) -> list[Phase]:
    """由 YAML 导弹预设构造 Phase 列表。"""
    m = get_missile(name)
    site = m["_site"]
    guid = dict(m["guidance"])
    guid["lat_deg"] = float(site["lat"])
    guid["lon_deg"] = float(site["lon"])
    stages = _stages_from_missile(m)

    phases: List[Phase] = []
    for i, s in enumerate(stages):
        dyn = PoweredECIDynamics(stage=s, guidance=guid)
        t_burn = float(s["m_prop"]) / dyn.prop.mdot
        phases.append(
            PoweredPhase(
                name=f"{s['name']} 动力",
                t_span=(0.0, t_burn * 1.5),
                dynamics=dyn,
                guidance=guid,
                m_dry=float(s["m_dry"]),
                m_after_separation=float(s["m_after_sep"]),
                sep_name=f"{s['name']} 分离",
            )
        )

    # 无动力滑行 + 再入 (沿用末级动力学, 推力关闭)
    last_stage = dict(stages[-1])
    last_stage.update(thrust_vac=0.0, thrust_sl=0.0, isp_vac=1.0, m_prop=0.0)
    dyn_coast = PoweredECIDynamics(
        stage=last_stage,
        guidance=guid,
        modes={"thrust": False, "drag": True, "j2": True},
    )
    exit_ev = _make_altitude_event(H_ENTRY, +1, terminal=False)
    reentry_ev = _make_altitude_event(H_ENTRY, -1, terminal=True)
    phases.append(
        CoastingPhase(
            name="中段",
            t_span=(0.0, 7200.0),
            dynamics=dyn_coast,
            guidance=guid,
            events=[_make_apogee_event(), exit_ev, reentry_ev],
        )
    )
    phases.append(
        ReentryPhase(
            name="再入段",
            t_span=(0.0, 7200.0),
            dynamics=dyn_coast,
            events=[_make_altitude_event(0.0, -1, terminal=True)],
        )
    )
    phases.append(
        TerminalPhase(
            name="终点",
            t_span=(0.0, 7200.0),
            dynamics=dyn_coast,
        )
    )
    return phases


__all__ = [
    "missile_config",
    "missile_phases",
    "list_missiles",
]
