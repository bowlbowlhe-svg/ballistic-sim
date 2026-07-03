"""补充 PEG 边界与失败分支测试以提升覆盖率。"""

from __future__ import annotations

import numpy as np

from ballistic_sim.constants import GM_EARTH, WGS84_A
from ballistic_sim.guidance import peg


def _circular_orbit_state(alt_m: float):
    r = np.array([WGS84_A + alt_m, 0.0, 0.0], dtype=float)
    v = np.array([0.0, np.sqrt(GM_EARTH / (WGS84_A + alt_m)), 0.0], dtype=float)
    return r, v


def test_replan_fails_with_zero_thrust() -> None:
    """推力为零时 replan 失败。"""
    state = peg.make_peg_state({"peri_km": 200.0})
    r, v = _circular_orbit_state(200e3)
    ok = peg.replan(state, 0.0, r, v, 5000.0, {"thrust_vac": 0.0, "isp_vac": 300.0})
    assert not ok
    assert state.failed


def test_replan_fails_with_nan_inputs() -> None:
    """非法输入时 replan 失败。"""
    state = peg.make_peg_state({"peri_km": 200.0})
    r = np.array([np.nan, 0.0, 0.0])
    v = np.array([0.0, 1000.0, 0.0])
    ok = peg.replan(state, 0.0, r, v, 5000.0, {"thrust_vac": 1e5, "isp_vac": 300.0})
    assert not ok
    assert state.failed


def test_replan_apogee_circ_fails_zero_velocity() -> None:
    """远地点圆化在水平速度为零时失败。"""
    state = peg.make_peg_state({"peri_km": 200.0, "apo_km": 350.0}, peg_mode="apogee_circ")
    r = np.array([WGS84_A + 350e3, 0.0, 0.0])
    v = np.zeros(3)
    ok = peg.replan_apogee_circularization(
        state, 0.0, r, v, 5000.0, {"thrust_vac": 1e4, "isp_vac": 300.0}
    )
    assert not ok
    assert state.failed


def test_peg_thrust_dir_failed_state_returns_none() -> None:
    """state.failed 且未触发重规划时 peg_thrust_dir 返回 None。"""
    state = peg.make_peg_state({"peri_km": 200.0}, replan_period=10.0)
    r, v = _circular_orbit_state(200e3)
    stage = {"thrust_vac": 1e5, "isp_vac": 300.0}
    # 先做一次成功规划，获得 dir_eci
    peg.peg_thrust_dir(0.0, r, v, 5000.0, stage, state)
    # 在重规划周期内设置 failed，下一次返回 None
    state.failed = True
    d = peg.peg_thrust_dir(1.0, r, v, 5000.0, stage, state)
    assert d is None


def test_peg_thrust_dir_no_replan_within_period() -> None:
    """两次调用间隔小于 replan_period 不触发重规划。"""
    state = peg.make_peg_state({"peri_km": 200.0}, replan_period=10.0)
    r, v = _circular_orbit_state(200e3)
    stage = {"thrust_vac": 1e5, "isp_vac": 300.0}
    peg.peg_thrust_dir(0.0, r, v, 5000.0, stage, state)
    n1 = state.n_replans
    peg.peg_thrust_dir(1.0, r, v, 5000.0, stage, state)
    assert state.n_replans == n1


def test_make_apogee_circ_event_zero_radius() -> None:
    """远地点圆化事件在 r=0 时返回安全值。"""
    ev = peg.make_apogee_circularization_event(WGS84_A + 350e3)
    y = np.zeros(6)
    assert ev(0.0, y) == 1.0


def test_make_apogee_circ_event_first_gamma_negative() -> None:
    """远地点圆化事件在首次 γ<0 时立即触发。"""
    ev = peg.make_apogee_circularization_event(WGS84_A + 350e3)
    r = np.array([WGS84_A + 350e3, 0.0, 0.0])
    v = np.array([-100.0, 100.0, 0.0])  # 径向分量负 => gamma < 0
    y = np.concatenate([r, v])
    val = ev(0.0, y)
    # 首次 γ<0 直接设置 t_cut
    assert abs(val) <= 1e-3
