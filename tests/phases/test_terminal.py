"""终点 Phase 单元测试。"""

from __future__ import annotations

import numpy as np

from ballistic_sim.dynamics.point_mass import PointMassDynamics
from ballistic_sim.phases.terminal import TerminalPhase


def test_terminal_phase_default_ground_event() -> None:
    """TerminalPhase 默认创建落地事件。"""
    dyn = PointMassDynamics(frame="ECI")
    ph = TerminalPhase(name="终点", t_span=(0.0, 100.0), dynamics=dyn)
    assert ph.is_terminal is True
    assert len(ph.events) == 1


def test_terminal_phase_orbit_insertion_event() -> None:
    """TerminalPhase 给定 target 时创建轨道插入事件。"""
    dyn = PointMassDynamics(frame="ECI")
    ph = TerminalPhase(
        name="终点",
        t_span=(0.0, 100.0),
        dynamics=dyn,
        target={"peri_km": 200.0, "apo_km": 350.0},
    )
    assert len(ph.events) == 1


def test_terminal_phase_rhs_zero() -> None:
    """TerminalPhase rhs 返回零导数。"""
    dyn = PointMassDynamics(frame="ECI")
    ph = TerminalPhase(name="终点", t_span=(0.0, 100.0), dynamics=dyn)
    y = np.ones(7)
    dy = ph.rhs(0.0, y, None, ph)
    assert np.allclose(dy, 0.0)


def test_terminal_phase_state_dim_and_frame() -> None:
    """TerminalPhase 状态维度/坐标系与动力学一致。"""
    dyn = PointMassDynamics(frame="ECI")
    ph = TerminalPhase(name="终点", t_span=(0.0, 100.0), dynamics=dyn)
    assert ph.state_dim() == 7
    assert ph.native_frame() == "ECI"
