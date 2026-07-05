"""6-DOF 刚体动力学单元测试（D30 参数）。"""

from __future__ import annotations

import warnings

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
from ballistic_sim.dynamics.six_dof import (
    SixDOFDynamics,
    _axis_from_quat,
    _normalize_quat,
)
from ballistic_sim.guidance.control import ControlMoment
from ballistic_sim.models.atmosphere import StandardAtmosphere
from ballistic_sim.models.wind import UniformWind


def _make_ctx(wind_e: float = 0.0) -> DynamicContext:
    cfg = SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(mass_kg=21.76, diameter_m=0.122, cd=0.844),
        launch=LaunchConfig(
            lat_deg=39.9,
            lon_deg=0.0,
            alt_m=0.0,
            azimuth_deg=90.0,
            elevation_deg=45.0,
            v0_m_s=690.0,
        ),
        environment=EnvironmentConfig(atmosphere="isa"),
        guidance=GuidanceConfig(),
        options=OptionsConfig(),
    )
    return DynamicContext(
        cfg=cfg,
        atmosphere=StandardAtmosphere(),
        wind=UniformWind(e=wind_e, n=0.0, u=0.0),
        aero=None,
        gravity_model="wgs84",
        options={},
    )


def _d30_coeff_tables() -> dict[str, np.ndarray]:
    ma = np.linspace(0, 3, 10)
    return {
        "CD0_table": np.column_stack([ma, np.full(10, 0.25)]),
        "CDa2_table": np.column_stack([ma, np.full(10, 0.05)]),
        "CNa_table": np.column_stack([ma, np.full(10, 1.6)]),
        "CMa_table": np.column_stack([ma, np.full(10, -3.5)]),
        "CMq_table": np.column_stack([ma, np.full(10, -1.5)]),
        "Clp_table": np.column_stack([ma, np.full(10, -0.025)]),
    }


def _make_d30(**overrides) -> SixDOFDynamics:
    params = {
        "mass_kg": 21.76,
        "diameter_m": 0.122,
        "form_factor": 0.844,
        "twist_cal": 20.0,
        "Ix": 0.045,
        "It": 0.38,
        "x_cp_cg": 0.05,
        "lat_deg": 39.9,
        **_d30_coeff_tables(),
    }
    params.update(overrides)
    return SixDOFDynamics(**params)


def test_sixdof_state_dim_and_frame() -> None:
    """SixDOFDynamics 状态维度为 13，原生坐标系为 ENU。"""
    dyn = _make_d30()
    assert dyn.state_dim() == 13
    assert dyn.native_frame() == "ENU"


def test_sixdof_initial_state_shape_and_spin() -> None:
    """initial_state 返回 13 维向量，横向角速度为 0，自旋为正。"""
    dyn = _make_d30()
    y0 = dyn.initial_state(v0=690.0, theta_deg=45.0, az_deg=90.0)
    assert y0.shape == (13,)
    assert y0[10] == pytest.approx(0.0)
    assert y0[11] == pytest.approx(0.0)
    assert y0[12] > 0.0


def test_sixdof_initial_state_q_aligns_velocity() -> None:
    """初始四元数使弹轴与初速方向对齐。"""
    dyn = _make_d30()
    y0 = dyn.initial_state(v0=690.0, theta_deg=45.0, az_deg=90.0)
    v = y0[3:6]
    s = _axis_from_quat(_normalize_quat(y0[6:10]))
    assert np.allclose(s, v / np.linalg.norm(v), atol=1e-12)


def test_sixdof_rhs_free_space_energy_conservation() -> None:
    """关闭 drag/gravity/coriolis 后，自由空间积分 1s 动能相对误差 <1e-6。"""
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
    y0[10] = 0.5
    y0[11] = -0.3

    sol = solve_ivp(lambda t, y: dyn.rhs(t, y, ctx), [0.0, 1.0], y0, rtol=1e-9, atol=1e-12)
    assert sol.success

    def kinetic_energy(state: np.ndarray) -> float:
        v = state[3:6]
        wy, wz, p = state[10], state[11], state[12]
        return (
            0.5 * dyn.mass_kg * np.dot(v, v) + 0.5 * dyn.It * (wy**2 + wz**2) + 0.5 * dyn.Ix * p**2
        )

    e0 = kinetic_energy(y0)
    e1 = kinetic_energy(sol.y[:, -1])
    assert abs(e1 - e0) / abs(e0) < 1e-6


def test_sixdof_rhs_quaternion_norm_preservation() -> None:
    """四元数模长在自由空间积分中保持为 1。"""
    dyn = SixDOFDynamics(
        options={
            "drag": False,
            "gravity": False,
            "coriolis": False,
            "thrust": False,
        },
    )
    ctx = _make_ctx()
    y0 = dyn.initial_state(v0=200.0, theta_deg=30.0, az_deg=45.0)
    y0[10] = 0.5
    y0[11] = -0.3

    sol = solve_ivp(lambda t, y: dyn.rhs(t, y, ctx), [0.0, 1.0], y0, rtol=1e-9, atol=1e-12)
    assert sol.success
    q_final = _normalize_quat(sol.y[6:10, -1])
    assert np.linalg.norm(q_final) == pytest.approx(1.0, abs=1e-8)


class _StubControl:
    """输出固定控制力矩的 stub。"""

    def __init__(self, moment: ControlMoment) -> None:
        self.moment = moment

    def control_moment(self, y, alpha, beta):
        return self.moment


def test_sixdof_rhs_with_control_moment() -> None:
    """注入 stub 控制力矩后，角加速度与自旋导数应体现该力矩。"""
    moment = ControlMoment(L=2.0, M=-4.0, N=6.0)
    dyn = SixDOFDynamics(
        mass_kg=2.0,
        Ix=0.1,
        It=1.5,
        control=_StubControl(moment),
        options={
            "drag": False,
            "gravity": False,
            "coriolis": False,
            "thrust": False,
        },
    )
    ctx = _make_ctx()
    y0 = dyn.initial_state(v0=100.0, theta_deg=0.0, az_deg=90.0)
    dy = dyn.rhs(0.0, y0, ctx)

    assert dy[10] == pytest.approx(moment.M / dyn.It, abs=1e-12)
    assert dy[11] == pytest.approx(moment.N / dyn.It, abs=1e-12)
    assert dy[12] == pytest.approx(moment.L / dyn.Ix, abs=1e-12)


def test_sixdof_telemetry_keys() -> None:
    """telemetry 返回要求的派生量键。"""
    dyn = _make_d30()
    ctx = _make_ctx()
    y0 = dyn.initial_state(v0=690.0, theta_deg=45.0, az_deg=90.0)
    tel = dyn.telemetry(0.0, y0, ctx)
    required = {"h", "v_inertial", "alpha_deg", "beta_deg", "Ma", "q", "rho", "spin_rpm"}
    assert required.issubset(tel.keys())
    assert tel["h"] == pytest.approx(0.0, abs=1e-12)
    assert tel["spin_rpm"] > 0.0


def test_sixdof_xcp_equal_xcg_warning() -> None:
    """静稳定距为 0 时应发出警告并回退到默认 0.05m。"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dyn = _make_d30(x_cp_cg=0.0)
    assert dyn.x_cp_cg == pytest.approx(0.05, abs=1e-12)
    assert any("x_cp_cg" in str(warning.message) for warning in w)


def test_sixdof_thrust_accelerates_along_body_axis() -> None:
    """开启推力选项后，轴向推力沿弹轴产生加速度。"""
    thrust = 2000.0
    mass = 10.0
    burn = 2.0
    dyn = SixDOFDynamics(
        mass_kg=mass,
        diameter_m=0.1,
        Ix=0.1,
        It=1.0,
        thrust_N=thrust,
        burn_time_s=burn,
        options={"drag": False, "gravity": True, "coriolis": False, "thrust": True},
    )
    ctx = _make_ctx()
    # 垂直发射（用极小初速保证弹轴朝上）
    y0 = dyn.initial_state(v0=1.0, theta_deg=90.0, az_deg=0.0)
    sol = solve_ivp(lambda t, y: dyn.rhs(t, y, ctx), [0.0, 1.0], y0, rtol=1e-9, atol=1e-12)
    assert sol.success
    v_u = sol.y[5, -1]
    # 净加速度 ≈ 推力/mass - g
    g0 = 9.80665
    expected = 1.0 + (thrust / mass - g0) * 1.0
    assert v_u == pytest.approx(expected, rel=0.02)

    tel = dyn.telemetry(0.5, sol.y[:, -1], ctx)
    assert tel["thrust_N"] == pytest.approx(thrust, rel=1e-12)


def test_sixdof_thrust_shuts_off_after_burn_time() -> None:
    """超过 burn_time_s 后推力应为 0。"""
    dyn = SixDOFDynamics(
        mass_kg=10.0,
        thrust_N=1000.0,
        burn_time_s=0.5,
        options={"drag": False, "gravity": False, "coriolis": False, "thrust": True},
    )
    ctx = _make_ctx()
    y0 = dyn.initial_state(v0=0.0, theta_deg=90.0, az_deg=0.0)
    dy = dyn.rhs(1.0, y0, ctx)
    # 关闭 drag/gravity/thrust_off 后加速度应为 0
    assert np.allclose(dy[3:6], 0.0, atol=1e-12)


def test_sixdof_piecewise_thrust_schedule() -> None:
    """thrust_N / burn_time_s 为列表时按分段推力工作。"""
    dyn = SixDOFDynamics(
        mass_kg=10.0,
        thrust_N=[2000.0, 1000.0],
        burn_time_s=[1.0, 1.0],
        options={"drag": False, "gravity": False, "coriolis": False, "thrust": True},
    )
    ctx = _make_ctx()
    y0 = dyn.initial_state(v0=0.0, theta_deg=90.0, az_deg=0.0)

    # 第一段推力（零初速时弹轴沿 E，推力产生 E 向加速度）
    dy = dyn.rhs(0.5, y0, ctx)
    assert dy[3] == pytest.approx(2000.0 / 10.0, abs=1e-9)
    tel = dyn.telemetry(0.5, y0, ctx)
    assert tel["thrust_N"] == pytest.approx(2000.0, rel=1e-12)

    # 第二段推力
    dy = dyn.rhs(1.5, y0, ctx)
    assert dy[3] == pytest.approx(1000.0 / 10.0, abs=1e-9)
    tel = dyn.telemetry(1.5, y0, ctx)
    assert tel["thrust_N"] == pytest.approx(1000.0, rel=1e-12)

    # 关机后
    dy = dyn.rhs(2.5, y0, ctx)
    assert np.allclose(dy[3:6], 0.0, atol=1e-12)
    tel = dyn.telemetry(2.5, y0, ctx)
    assert tel["thrust_N"] == pytest.approx(0.0, abs=1e-12)


def test_sixdof_piecewise_thrust_schedule_relative_to_t_start() -> None:
    """分段推力时间表相对于 ``t_start_s`` 生效。"""
    dyn = SixDOFDynamics(
        mass_kg=10.0,
        thrust_N=[1500.0],
        burn_time_s=[1.0],
        t_start_s=5.0,
        options={"drag": False, "gravity": False, "coriolis": False, "thrust": True},
    )
    ctx = _make_ctx()
    y0 = dyn.initial_state(v0=0.0, theta_deg=90.0, az_deg=0.0)

    assert dyn._thrust_at(5.5) == pytest.approx(1500.0, rel=1e-12)
    assert dyn._thrust_at(4.5) == pytest.approx(0.0, abs=1e-12)
    assert dyn._thrust_at(6.5) == pytest.approx(0.0, abs=1e-12)

    dy = dyn.rhs(5.5, y0, ctx)
    assert dy[3] == pytest.approx(1500.0 / 10.0, abs=1e-9)


def test_sixdof_piecewise_thrust_length_mismatch_raises() -> None:
    """thrust_N 与 burn_time_s 长度不一致时抛出 ValueError。"""
    with pytest.raises(ValueError, match="长度"):
        SixDOFDynamics(
            mass_kg=10.0,
            thrust_N=[1000.0, 2000.0],
            burn_time_s=[1.0],
            options={"thrust": True},
        )
