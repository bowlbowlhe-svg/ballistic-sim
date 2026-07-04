"""6-DOF 刚体动力学冒烟测试。"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from ballistic_sim.config import (
    EnvironmentConfig,
    GuidanceConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    VehicleConfig,
)
from ballistic_sim.dynamics.common import DynamicContext
from ballistic_sim.dynamics.six_dof import SixDOFDynamics
from ballistic_sim.models.atmosphere import StandardAtmosphere
from ballistic_sim.models.wind import UniformWind


def _make_ctx() -> DynamicContext:
    cfg = SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(mass_kg=1.0, diameter_m=0.1, cd=0.3),
        launch=LaunchConfig(
            lat_deg=0.0,
            lon_deg=0.0,
            alt_m=0.0,
            azimuth_deg=90.0,
            elevation_deg=45.0,
            v0_m_s=100.0,
        ),
        environment=EnvironmentConfig(atmosphere="isa"),
        guidance=GuidanceConfig(),
        options=OptionsConfig(),
    )
    return DynamicContext(
        cfg=cfg,
        atmosphere=StandardAtmosphere(),
        wind=UniformWind(),
        aero=None,
        options={},
    )


def test_state_dim() -> None:
    """状态维度为 13。"""
    dyn = SixDOFDynamics()
    assert dyn.state_dim() == 13


def test_initial_state_shape_and_contents() -> None:
    """initial_state 返回 13 维向量，四元数单位化，横向角速度为 0。"""
    dyn = SixDOFDynamics()
    y0 = dyn.initial_state(v0=100.0, theta_deg=45.0, az_deg=90.0)
    assert y0.size == 13
    assert np.linalg.norm(y0[6:10]) == pytest.approx(1.0)
    assert y0[10] == pytest.approx(0.0)
    assert y0[11] == pytest.approx(0.0)
    assert y0[12] > 0.0  # 自转率由 twist_cal 计算


def test_rhs_shape() -> None:
    """rhs 返回 13 维导数。"""
    dyn = SixDOFDynamics()
    ctx = _make_ctx()
    y0 = dyn.initial_state(v0=100.0, theta_deg=45.0, az_deg=90.0)
    dy = dyn.rhs(0.0, y0, ctx)
    assert dy.shape == (13,)


def test_free_space_energy_conservation() -> None:
    """关闭 drag/gravity/coriolis 后，自由空间积分 1 秒能量守恒。"""
    dyn = SixDOFDynamics(
        mass_kg=2.0,
        Ix=0.1,
        It=1.5,
        options={
            "drag": False,
            "gravity": False,
            "coriolis": False,
            "thrust": False,
        },
    )
    ctx = _make_ctx()
    y0 = dyn.initial_state(v0=200.0, theta_deg=30.0, az_deg=45.0)
    # 给非零横向角速度，使姿态运动学也被积分
    y0[10] = 0.5
    y0[11] = -0.3

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return dyn.rhs(t, y, ctx)

    sol = solve_ivp(rhs, [0.0, 1.0], y0, rtol=1e-9, atol=1e-12)
    assert sol.success

    def kinetic_energy(state: np.ndarray) -> float:
        v = state[3:6]
        wy = state[10]
        wz = state[11]
        p = state[12]
        return (
            0.5 * dyn.mass_kg * np.dot(v, v)
            + 0.5 * dyn.It * (wy**2 + wz**2)
            + 0.5 * dyn.Ix * p**2
        )

    e0 = kinetic_energy(y0)
    e1 = kinetic_energy(sol.y[:, -1])
    assert abs(e1 - e0) / abs(e0) < 1e-6


def test_telemetry_keys() -> None:
    """telemetry 返回要求的派生量键。"""
    dyn = SixDOFDynamics()
    ctx = _make_ctx()
    y0 = dyn.initial_state(v0=100.0, theta_deg=45.0, az_deg=90.0)
    tel = dyn.telemetry(0.0, y0, ctx)
    required = {"h", "v_inertial", "alpha_deg", "beta_deg", "Ma", "q", "rho", "spin_rpm"}
    assert required.issubset(tel.keys())
    assert tel["h"] == pytest.approx(0.0)
    assert tel["spin_rpm"] > 0.0


def test_with_control_moment() -> None:
    """控制力矩被正确叠加到体轴力矩。"""
    from ballistic_sim.guidance.control import SixDOFControl

    control = SixDOFControl(canard_gain=10.0)
    dyn = SixDOFDynamics(control=control)
    ctx = _make_ctx()
    y0 = dyn.initial_state(v0=100.0, theta_deg=10.0, az_deg=0.0)
    # 人为制造攻角/侧滑，使鸭舵输出非零力矩
    y0[4] += 5.0  # 侧向速度产生 beta
    y0[5] += 3.0  # 垂直速度产生 alpha
    dy = dyn.rhs(0.0, y0, ctx)
    assert dy.shape == (13,)
    # 鸭舵力矩使俯仰/偏航角加速度非零
    assert dy[10] != 0.0 or dy[11] != 0.0
