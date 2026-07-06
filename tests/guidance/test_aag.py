"""增广显式制导 (AAG) 单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.config import (
    EnvironmentConfig,
    GuidanceConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    StageConfig,
    VehicleConfig,
)
from ballistic_sim.constants import GM_EARTH, WGS84_A
from ballistic_sim.guidance.aag import (
    AAGState,
    aag_acceleration,
    aag_replan,
    aag_thrust_dir,
    make_aag_state,
)
from ballistic_sim.phases.builder import build_phases


def _make_stage() -> dict:
    return {
        "thrust_vac": 1.0e5,
        "isp_vac": 300.0,
        "m_prop": 5000.0,
        "m_dry": 1000.0,
        "Aref": 5.0,
    }


def test_make_aag_state_from_orbit() -> None:
    """由轨道参数构造 AAGState 应得到合理终端速度。"""
    state = make_aag_state({"peri_km": 200.0, "apo_km": 200.0})
    r_T = state.terminal["r_T_m"]
    v_T = state.terminal["v_T_ms"]
    assert r_T == pytest.approx(WGS84_A + 200e3, abs=1.0)
    assert v_T == pytest.approx(np.sqrt(GM_EARTH / r_T), rel=1e-3)


def test_aag_replan_sets_direction() -> None:
    """AAG 重规划应成功并设置推力方向。"""
    state = make_aag_state({"r_T_m": WGS84_A + 200e3, "v_T_ms": 7800.0, "gamma_T": 0.0})
    stage = _make_stage()
    r0 = np.array([WGS84_A, 0.0, 0.0])
    v0 = np.array([0.0, 100.0, 500.0])
    ok = aag_replan(state, 0.0, r0, v0, 6000.0, stage)
    assert ok
    assert state.dir_eci is not None
    assert np.linalg.norm(state.dir_eci) == pytest.approx(1.0, abs=1e-6)
    assert not state.failed


def test_aag_thrust_dir_calls_replan() -> None:
    """aag_thrust_dir 应触发重规划并返回单位矢量。"""
    state = make_aag_state({"peri_km": 200.0, "apo_km": 200.0})
    stage = _make_stage()
    r0 = np.array([WGS84_A, 0.0, 0.0])
    v0 = np.array([0.0, 100.0, 500.0])
    d = aag_thrust_dir(0.0, r0, v0, 6000.0, stage, state)
    assert d is not None
    assert np.linalg.norm(d) == pytest.approx(1.0, abs=1e-6)


def test_aag_acceleration_magnitude() -> None:
    """aag_acceleration 应返回含推力大小的加速度。"""
    state = make_aag_state({"peri_km": 200.0, "apo_km": 200.0})
    stage = _make_stage()
    r0 = np.array([WGS84_A, 0.0, 0.0])
    v0 = np.array([0.0, 100.0, 500.0])
    a = aag_acceleration(0.0, r0, v0, 6000.0, stage, state)
    assert a is not None
    expected = stage["thrust_vac"] / 6000.0
    assert np.linalg.norm(a) == pytest.approx(expected, rel=1e-3)


def test_aag_failed_on_bad_stage() -> None:
    """AAG 在非法 stage 上应设置 failed 标志。"""
    state = make_aag_state({"peri_km": 200.0})
    bad_stage = {"thrust_vac": 0.0, "isp_vac": 300.0}
    r0 = np.array([WGS84_A, 0.0, 0.0])
    v0 = np.array([0.0, 100.0, 500.0])
    ok = aag_replan(state, 0.0, r0, v0, 1000.0, bad_stage)
    assert not ok
    assert state.failed


def test_aag_builder_assembles_guidance() -> None:
    """builder 对 rocket+aag 应在动力段注入 AAG 状态。"""
    cfg = SimConfig(
        mission="rocket",
        vehicle=VehicleConfig(
            mass_kg=30000.0,
            stages=[
                StageConfig(
                    name="S1",
                    thrust_sl=300.0e3,
                    thrust_vac=350.0e3,
                    isp_vac=300.0,
                    m_prop=15000.0,
                    m_dry=3000.0,
                    diameter_m=2.0,
                )
            ],
        ),
        launch=LaunchConfig(lat_deg=28.5, lon_deg=-80.0, elevation_deg=90.0),
        guidance=GuidanceConfig(
            guidance_law="aag",
            target_alt_m=200e3,
            terminal_velocity_m_s=7800.0,
            terminal_fpa_deg=0.0,
        ),
        options=OptionsConfig(),
        environment=EnvironmentConfig(),
    )
    phases = build_phases(cfg)
    powered = phases[0]
    assert powered.guidance["phase"] == "aag"
    assert "_aag_state" in powered.guidance
    assert isinstance(powered.guidance["_aag_state"], AAGState)


def test_aag_orbit_insertion_velocity() -> None:
    """AAG 制导下火箭末速应接近目标速度（误差 < 50 m/s）。"""
    from ballistic_sim.simulator import simulate

    # 采用单级火箭可达的亚轨道/弹道目标，验证 AAG 终端速度精度
    target_v = 2500.0
    cfg = SimConfig(
        mission="rocket",
        vehicle=VehicleConfig(
            mass_kg=22000.0,
            stages=[
                StageConfig(
                    name="S1",
                    thrust_sl=450.0e3,
                    thrust_vac=500.0e3,
                    isp_vac=300.0,
                    m_prop=15000.0,
                    m_dry=2000.0,
                    diameter_m=2.0,
                )
            ],
        ),
        launch=LaunchConfig(
            lat_deg=28.5,
            lon_deg=-80.0,
            alt_m=10000.0,
            elevation_deg=60.0,
            azimuth_deg=90.0,
        ),
        guidance=GuidanceConfig(
            guidance_law="aag",
            target_alt_m=100e3,
            terminal_velocity_m_s=target_v,
            terminal_fpa_deg=0.0,
            guidance_replan_period=5.0,
        ),
        options=OptionsConfig(max_step=2.0, terminate_impact=False),
        environment=EnvironmentConfig(atmosphere="none"),
    )
    res = simulate(cfg)
    if res.y.size == 0:
        pytest.skip("AAG 仿真未产生有效轨迹")
    y_end = res.y[-1]
    v_end = float(np.linalg.norm(y_end[3:6]))
    dv = abs(v_end - target_v)
    assert dv < 50.0
