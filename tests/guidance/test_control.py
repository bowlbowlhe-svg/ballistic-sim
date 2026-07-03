"""6-DOF 控制律接口单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.guidance.control import (
    ControlMoment,
    NoAutoDimensionUpgrade,
    SixDOFControl,
)


def test_control_moment_vector() -> None:
    """ControlMoment.vector 返回 [L, M, N]。"""
    m = ControlMoment(L=1.0, M=2.0, N=3.0)
    assert np.allclose(m.vector(), np.array([1.0, 2.0, 3.0]))


def test_require_initial_attitude_rejects_3dof() -> None:
    """3-DOF 状态禁止启用 6-DOF 控制。"""
    ctrl = SixDOFControl()
    y3 = np.zeros(7)
    with pytest.raises(NoAutoDimensionUpgrade):
        ctrl.require_initial_attitude(y3)


def test_require_initial_attitude_rejects_zero_quaternion() -> None:
    """13 维状态但零四元数仍视为未提供姿态。"""
    ctrl = SixDOFControl()
    y13 = np.zeros(13)
    y13[6:10] = 0.0
    with pytest.raises(NoAutoDimensionUpgrade):
        ctrl.require_initial_attitude(y13)


def test_require_initial_attitude_accepts_valid() -> None:
    """13 维状态且四元数非零时通过检查。"""
    ctrl = SixDOFControl()
    y13 = np.zeros(13)
    y13[6:10] = np.array([1.0, 0.0, 0.0, 0.0])
    ctrl.require_initial_attitude(y13)


def test_canard_moment_signs() -> None:
    """鸭舵力矩符号与攻角/侧滑角误差相反。"""
    ctrl = SixDOFControl(canard_gain=2.0)
    m = ctrl.canard_moment(np.radians(10.0), np.radians(5.0))
    assert m.L == 0.0
    assert m.M == pytest.approx(-2.0 * np.radians(10.0), rel=1e-6)
    assert m.N == pytest.approx(2.0 * np.radians(5.0), rel=1e-6)


def test_pulse_moment_below_deadband() -> None:
    """姿态误差低于死区时脉冲力矩为零。"""
    ctrl = SixDOFControl(pulse_magnitude=100.0, pulse_deadband_rad=0.1)
    m = ctrl.pulse_moment(np.array([0.01, 0.0, 0.0]))
    assert m.L == 0.0
    assert m.M == 0.0
    assert m.N == 0.0


def test_pulse_moment_above_deadband() -> None:
    """姿态误差超过死区时输出固定幅值脉冲。"""
    ctrl = SixDOFControl(pulse_magnitude=100.0, pulse_deadband_rad=0.01)
    err = np.array([0.1, 0.0, 0.0])
    m = ctrl.pulse_moment(err)
    assert np.isclose(np.linalg.norm(m.vector()), 100.0)
    assert np.allclose(m.vector() / np.linalg.norm(m.vector()), err / np.linalg.norm(err))


def test_control_moment_combined() -> None:
    """综合力矩入口返回鸭舵 + 脉冲的矢量和。"""
    ctrl = SixDOFControl(canard_gain=1.0, pulse_magnitude=10.0, pulse_deadband_rad=0.01)
    y13 = np.zeros(13)
    y13[6:10] = np.array([1.0, 0.0, 0.0, 0.0])
    err = np.array([0.1, 0.0, 0.0])
    m = ctrl.control_moment(
        y13,
        alpha_rad=np.radians(5.0),
        beta_rad=0.0,
        attitude_error=err,
    )
    expected_m = -1.0 * np.radians(5.0)
    expected_l = 10.0
    assert m.M == pytest.approx(expected_m, abs=1e-12)
    assert m.L == pytest.approx(expected_l, abs=1e-12)
