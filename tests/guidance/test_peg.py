"""PEG (Powered Explicit Guidance) 核心路径单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.constants import GM_EARTH, WGS84_A
from ballistic_sim.guidance import peg

_TARGET = {"peri_km": 200.0, "apo_km": 350.0, "inc_deg": 42.0}


def test_make_target_terminal_basic() -> None:
    """make_target_terminal 返回合理的终端约束。"""
    tt = peg.make_target_terminal(_TARGET)
    assert tt["r_T_m"] == pytest.approx(WGS84_A + 200e3, rel=1e-6)
    assert tt["gamma_T"] == 0.0
    assert tt["inc_deg"] == 42.0
    assert tt["v_T_ms"] > 0.0


def test_make_peg_state_periapsis_mode() -> None:
    """periapsis 模式构造的 PegState 字段正确。"""
    state = peg.make_peg_state(_TARGET)
    assert state.peg_mode == "periapsis"
    assert state.target_terminal["r_T_m"] == pytest.approx(WGS84_A + 200e3, rel=1e-6)


def test_make_peg_state_apogee_circ_mode() -> None:
    """apogee_circ 模式构造的 PegState 目标为圆轨道速度。"""
    state = peg.make_peg_state(_TARGET, peg_mode="apogee_circ")
    assert state.peg_mode == "apogee_circ"
    r_apo = WGS84_A + 350e3
    assert state.target_terminal["r_T_m"] == pytest.approx(r_apo, rel=1e-6)
    assert state.target_terminal["v_T_ms"] == pytest.approx(np.sqrt(GM_EARTH / r_apo), rel=1e-6)


def test_effective_exhaust_velocity() -> None:
    """真空比冲换算有效排气速度。"""
    ve = peg._effective_exhaust_velocity({"isp_vac": 300.0})
    assert ve == pytest.approx(300.0 * 9.80665, rel=1e-6)


def test_estimate_tgo_typical() -> None:
    """典型参数下 t_go 为正且有限。"""
    tgo = peg._estimate_tgo(vgo_mag=1000.0, m_now=5000.0, thrust=1e5, ve_eff=3000.0)
    assert np.isfinite(tgo)
    assert tgo > 0.0


def test_estimate_tgo_invalid_returns_nan() -> None:
    """非法参数返回 nan。"""
    assert np.isnan(peg._estimate_tgo(1000.0, 5000.0, 0.0, 3000.0))
    assert np.isnan(peg._estimate_tgo(1000.0, 5000.0, 1e5, 0.0))


def _circular_orbit_state(alt_m: float):
    """构造圆轨道 (r, v) 状态。"""
    r = np.array([WGS84_A + alt_m, 0.0, 0.0], dtype=float)
    v = np.array([0.0, np.sqrt(GM_EARTH / (WGS84_A + alt_m)), 0.0], dtype=float)
    return r, v


def test_replan_converges_near_circular() -> None:
    """在接近目标圆轨道状态下，replan 成功且 vgo 很小。"""
    state = peg.make_peg_state(_TARGET)
    r, v = _circular_orbit_state(200e3)
    stage = {"thrust_vac": 1e5, "isp_vac": 320.0}
    ok = peg.replan(state, 0.0, r, v, 5000.0, stage)
    assert ok
    assert not state.failed
    # 200 km 圆轨道速度与目标椭圆近地点速度差异约 44 m/s
    assert state.vgo_mag < 100.0
    assert state.t_go > 0.0
    assert np.isclose(np.linalg.norm(state.dir_eci), 1.0)


def test_peg_thrust_dir_returns_unit_vector() -> None:
    """peg_thrust_dir 返回单位矢量。"""
    state = peg.make_peg_state(_TARGET)
    r, v = _circular_orbit_state(200e3)
    stage = {"thrust_vac": 1e5, "isp_vac": 320.0}
    d = peg.peg_thrust_dir(0.0, r, v, 5000.0, stage, state)
    assert d is not None
    assert np.isclose(np.linalg.norm(d), 1.0)


def test_peg_thrust_dir_replans_periodically() -> None:
    """两次调用间隔超过 replan_period 会触发重规划。"""
    state = peg.make_peg_state(_TARGET, replan_period=2.0)
    r, v = _circular_orbit_state(200e3)
    stage = {"thrust_vac": 1e5, "isp_vac": 320.0}
    peg.peg_thrust_dir(0.0, r, v, 5000.0, stage, state)
    n1 = state.n_replans
    peg.peg_thrust_dir(3.0, r, v, 5000.0, stage, state)
    assert state.n_replans > n1


def test_replan_apogee_circularization_success() -> None:
    """远地点圆化重规划在圆轨道状态下成功。"""
    state = peg.make_peg_state(_TARGET, peg_mode="apogee_circ", r_apo_circ_m=WGS84_A + 350e3)
    r, v = _circular_orbit_state(350e3)
    stage = {"thrust_vac": 1e4, "isp_vac": 320.0}
    ok = peg.replan_apogee_circularization(state, 0.0, r, v, 5000.0, stage)
    assert ok
    assert state.vgo_mag < 1.0


def test_make_orbit_reached_event_triggers() -> None:
    """能量达标事件在目标轨道附近触发。"""
    ev = peg.make_orbit_reached_event(_TARGET)
    tt = peg.make_target_terminal(_TARGET)
    r = np.array([tt["r_T_m"], 0.0, 0.0], dtype=float)
    v = np.array([0.0, tt["v_T_ms"], 0.0], dtype=float)
    y = np.concatenate([r, v])
    val = ev(0.0, y)
    # 恰好满足目标终端能量时事件值应接近 0 或 ≤ 0（允许浮点噪声）
    assert val <= 1e-6


def test_make_apogee_circularization_event_triggers_at_apogee() -> None:
    """远地点圆化事件在远地点 γ=0 处触发。"""
    ev = peg.make_apogee_circularization_event(WGS84_A + 350e3)
    r, v = _circular_orbit_state(350e3)
    y = np.concatenate([r, v])
    val = ev(0.0, y)
    assert abs(val) < 1.0
