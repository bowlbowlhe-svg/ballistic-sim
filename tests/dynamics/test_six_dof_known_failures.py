"""6-DOF 已知故障回归测试（Known Failures）。

KF-001：高射角 + 侧风落地稳定性。
KF-002/003/004：3-DOF 自动升维、零四元数、鸭舵力矩符号，
    由 tests/dynamics/test_six_dof_control.py 覆盖，本文件仅做交叉引用。
"""

from __future__ import annotations

import pytest
from scipy.integrate import solve_ivp

from ballistic_sim.dynamics.six_dof import SixDOFDynamics
from ballistic_sim.phases.events import make_ground_event
from tests.dynamics.test_six_dof import _d30_coeff_tables, _make_ctx


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


def _simulate_sixdof(dyn: SixDOFDynamics, ctx, theta: float, az: float, t_max: float = 300.0):
    """积分 6-DOF 直到落地，返回 sol 对象。"""
    y0 = dyn.initial_state(v0=690.0, theta_deg=theta, az_deg=az, h0=0.0)
    ground = make_ground_event(frame="ENU")
    sol = solve_ivp(
        lambda t, y: dyn.rhs(t, y, ctx),
        (0.0, t_max),
        y0,
        method="DOP853",
        events=ground,
        rtol=1e-6,
        atol=1e-9,
        max_step=1.0,
    )
    return sol


@pytest.mark.divergence
def test_kf_001_d30_high_qe_crosswind_lands() -> None:
    """KF-001：D30 高射角 50° + 东向侧风 8m/s 能落地且不抛 RuntimeError。"""
    dyn = _make_d30()
    ctx = _make_ctx(wind_e=8.0)
    sol = _simulate_sixdof(dyn, ctx, theta=50.0, az=90.0)
    assert sol.success, f"积分失败: {sol.message}"
    assert sol.t_events is not None
    assert len(sol.t_events) == 1
    assert sol.t_events[0] is not None
    assert len(sol.t_events[0]) > 0, "未触发落地事件"
    y_end = sol.y[:, -1]
    assert abs(float(y_end[2])) < 1.0  # 落地高度接近 0


def test_kf_002_003_004_covered_by_control_tests() -> None:
    """KF-002/003/004 在 tests/dynamics/test_six_dof_control.py 中覆盖。

    - KF-002：7 维状态禁止自动升维 -> test_sixdof_control_rejects_7dim
    - KF-003：零四元数禁止启用 6-DOF 控制 -> test_sixdof_control_rejects_zero_quaternion
    - KF-004：鸭舵力矩符号 -> test_sixdof_canard_moment_signs
    """
    from tests.dynamics import test_six_dof_control

    test_six_dof_control.test_sixdof_control_rejects_7dim()
    test_six_dof_control.test_sixdof_control_rejects_zero_quaternion()
    test_six_dof_control.test_sixdof_canard_moment_signs()
