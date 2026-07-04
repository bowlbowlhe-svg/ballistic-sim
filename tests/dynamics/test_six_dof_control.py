"""6-DOF 控制律与状态维度检查（动力学视角）。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.guidance.control import (
    ControlMoment,
    NoAutoDimensionUpgrade,
    SixDOFControl,
)


def test_sixdof_control_returns_moment_for_13dim() -> None:
    """SixDOFControl 对 13 维状态返回非空控制力矩。"""
    ctrl = SixDOFControl(canard_gain=2.0)
    y13 = np.zeros(13)
    y13[6:10] = np.array([0.0, 0.0, 0.0, 1.0])
    m = ctrl.control_moment(y13, alpha_rad=np.radians(5.0), beta_rad=np.radians(3.0))
    assert isinstance(m, ControlMoment)
    assert m.L == pytest.approx(0.0, abs=1e-12)
    assert m.M == pytest.approx(-2.0 * np.radians(5.0), rel=1e-12)
    assert m.N == pytest.approx(2.0 * np.radians(3.0), rel=1e-12)


def test_sixdof_control_rejects_7dim() -> None:
    """SixDOFControl 对 7 维状态抛 NoAutoDimensionUpgrade。"""
    ctrl = SixDOFControl()
    y7 = np.zeros(7)
    with pytest.raises(NoAutoDimensionUpgrade):
        ctrl.control_moment(y7, alpha_rad=0.0, beta_rad=0.0)


def test_sixdof_control_rejects_zero_quaternion() -> None:
    """13 维状态但零四元数仍视为未提供姿态。"""
    ctrl = SixDOFControl()
    y13 = np.zeros(13)
    with pytest.raises(NoAutoDimensionUpgrade):
        ctrl.control_moment(y13, alpha_rad=0.0, beta_rad=0.0)


def test_sixdof_canard_moment_signs() -> None:
    """鸭舵力矩符号：M = -gain*alpha, N = gain*beta。"""
    ctrl = SixDOFControl(canard_gain=1.5)
    alpha = np.radians(8.0)
    beta = np.radians(-4.0)
    m = ctrl.canard_moment(alpha, beta)
    assert m.L == pytest.approx(0.0, abs=1e-12)
    assert m.M == pytest.approx(-1.5 * alpha, rel=1e-12)
    assert m.N == pytest.approx(1.5 * beta, rel=1e-12)
