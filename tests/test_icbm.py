"""ICBM 跨大气层端到端测试（阶段 3）。

使用占位单级 ICBM 配置，运行主动段 → 中段 → 再入段：
- 验证全段事件链（ burnout / apogee / 出大气 / 再入 / 落地）；
- 验证全程能量在主动段单调增、无动力段单调降；
- 验证落点距离、再入速度、最大高度在合理量级。
"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.config import (
    EnvironmentConfig,
    GuidanceConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    VehicleConfig,
)
from ballistic_sim.constants import GM_EARTH, WGS84_A
from ballistic_sim.dynamics.powered_eci import PoweredECIDynamics
from ballistic_sim.frames import ecef_to_geodetic, eci_to_ecef, haversine_distance
from ballistic_sim.phases.coasting import CoastingPhase
from ballistic_sim.phases.powered import PoweredPhase
from ballistic_sim.phases.reentry import ReentryPhase
from ballistic_sim.phases.terminal import TerminalPhase
from ballistic_sim.simulator import simulate

H_ENTRY = 100e3  # 大气边界，用于出/再入事件


def _make_altitude_event(h_m: float, direction: int, terminal: bool = False):
    """基于球近似高的高度穿越事件。"""

    def ev(t: float, y: np.ndarray) -> float:
        r = np.asarray(y[0:3], dtype=float)
        return float(np.linalg.norm(r) - WGS84_A - h_m)

    ev.terminal = terminal  # type: ignore[attr-defined]
    ev.direction = direction  # type: ignore[attr-defined]
    return ev


def _make_apogee_event():
    """ECI 远地点事件：径向速度由正变负。"""

    def ev(t: float, y: np.ndarray) -> float:
        r = np.asarray(y[0:3], dtype=float)
        v = np.asarray(y[3:6], dtype=float)
        return float(np.dot(r, v))

    ev.terminal = False  # type: ignore[attr-defined]
    ev.direction = -1  # type: ignore[attr-defined]
    return ev


def _placeholder_icbm_phases():
    """构造占位 ICBM 阶段链。"""
    lat = 0.0
    lon = 0.0
    azimuth = 45.0
    warhead_mass = 1000.0

    # 主动段：单级大推力， burnout 速度约 8 km/s
    stage = dict(
        name="ICBM-boost",
        thrust_vac=1.2e6,
        thrust_sl=1.0e6,
        isp_vac=300.0,
        m_prop=20000.0,
        m_dry=warhead_mass,
        Aref=1.0,
    )
    guid = dict(
        lat_deg=lat,
        lon_deg=lon,
        azimuth_deg=azimuth,
        t_pitchover=10.0,
        kick_deg=20.0,
        t_kick_end=60.0,
    )

    dyn_boost = PoweredECIDynamics(stage=stage, guidance=guid)
    t_burn = float(stage["m_prop"]) / dyn_boost.prop.mdot
    ph_boost = PoweredPhase(
        name="主动段",
        t_span=(0.0, t_burn * 1.2),
        dynamics=dyn_boost,
        guidance=guid,
        m_dry=warhead_mass,
        sep_name="关机",
    )

    # 无动力级：滑行 + 再入共用，推力关闭
    coast_stage = dict(
        name="ICBM-coast",
        thrust_vac=0.0,
        thrust_sl=0.0,
        isp_vac=1.0,
        m_prop=0.0,
        m_dry=warhead_mass,
        Aref=0.5,
    )
    dyn_coast = PoweredECIDynamics(
        stage=coast_stage,
        guidance=guid,
        modes={"thrust": False, "drag": True, "j2": True},
    )

    exit_ev = _make_altitude_event(H_ENTRY, +1, terminal=False)
    reentry_ev = _make_altitude_event(H_ENTRY, -1, terminal=True)
    ph_coast = CoastingPhase(
        name="中段",
        t_span=(0.0, 7200.0),
        dynamics=dyn_coast,
        guidance=guid,
        events=[_make_apogee_event(), exit_ev, reentry_ev],
    )

    ground_ev = _make_altitude_event(0.0, -1, terminal=True)
    ph_reentry = ReentryPhase(
        name="再入段",
        t_span=(0.0, 7200.0),
        dynamics=dyn_coast,
        events=[ground_ev],
    )

    ph_terminal = TerminalPhase(
        name="终点",
        t_span=(0.0, 7200.0),
        dynamics=dyn_coast,
    )

    cfg = SimConfig(
        mission="icbm",
        vehicle=VehicleConfig(
            mass_kg=float(stage["m_prop"] + stage["m_dry"]),
            diameter_m=0.5,
            cd=0.3,
        ),
        launch=LaunchConfig(
            lat_deg=lat,
            lon_deg=lon,
            alt_m=1.0,  # 避免地面事件在 t=0 触发
            azimuth_deg=azimuth,
            elevation_deg=90.0,
            v0_m_s=0.0,
        ),
        environment=EnvironmentConfig(
            atmosphere="isa",
            gravity_model="j2",
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
    return cfg, [ph_boost, ph_coast, ph_reentry, ph_terminal]


@pytest.mark.slow
def test_icbm_end_to_end() -> None:
    """占位 ICBM：主动段→中段→再入段端到端检查。"""
    cfg, phases = _placeholder_icbm_phases()
    result = simulate(cfg, phases=phases)

    assert result.stop_reason == "completed"
    assert result.y.size > 0
    assert len(result.t) > 2

    # ---- 事件链检查 ----
    assert any("主动段" in ev["phase"] and ev["event_index"] == 0 for ev in result.event_log)
    assert any("中段" in ev["phase"] and ev["event_index"] == 0 for ev in result.event_log)  # apogee
    assert any("中段" in ev["phase"] and ev["event_index"] == 1 for ev in result.event_log)  # exit
    assert any(
        "中段" in ev["phase"] and ev["event_index"] == 2 for ev in result.event_log
    )  # entry
    assert any("再入段" in ev["phase"] for ev in result.event_log)

    # ---- 全程能量曲线 ----
    r = result.y[:, 0:3]
    v = result.y[:, 3:6]
    energy = 0.5 * np.sum(v * v, axis=1) - GM_EARTH / np.linalg.norm(r, axis=1)

    # 找到 burnout 时刻索引
    burnout_ev = next(
        ev for ev in result.event_log if "主动段" in ev["phase"] and ev["event_index"] == 0
    )
    idx_boost = int(np.searchsorted(result.t, burnout_ev["t"]))

    # 主动段能量增加
    assert energy[idx_boost] > energy[0]
    assert np.all(np.diff(energy[: idx_boost + 1]) >= -1.0)  # 允许数值噪声

    # 无动力段能量单调不增（J2+气动耗散；允许数值噪声 1 kJ/kg）
    assert np.all(np.diff(energy[idx_boost:]) <= 1e3)
    assert energy[-1] < energy[idx_boost]

    # ---- 出/再入检查 ----
    alt = np.linalg.norm(r, axis=1) - WGS84_A
    exits = np.where((alt[:-1] < H_ENTRY) & (alt[1:] >= H_ENTRY))[0]
    entries = np.where((alt[:-1] > H_ENTRY) & (alt[1:] <= H_ENTRY))[0]
    assert exits.size >= 1, "未检测到出大气事件"
    assert entries.size >= 1, "未检测到再入事件"
    assert entries[0] > exits[0]

    # ---- 落点合理性 ----
    r_eci_end = result.y[-1, 0:3]
    r_ecef_end = eci_to_ecef(r_eci_end, float(result.t[-1]))
    lat_f, lon_f, _ = ecef_to_geodetic(r_ecef_end)
    range_m = haversine_distance(cfg.launch.lat_deg, cfg.launch.lon_deg, lat_f, lon_f)

    assert 1e5 < range_m < 5e6, f"射程 {range_m/1e3:.0f} km 不在合理 ICBM 区间"
    assert alt.max() > H_ENTRY, f"最大高度 {alt.max()/1e3:.0f} km 未出大气"
    assert alt.max() < 2e7, f"最大高度 {alt.max()/1e3:.0f} km 过大"
    v_impact = float(np.linalg.norm(result.y[-1, 3:6]))
    assert 1000.0 < v_impact < 10000.0, f"落地速度 {v_impact:.0f} m/s 不合理"
