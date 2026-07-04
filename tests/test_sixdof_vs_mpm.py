"""6-DOF 与 MPM 对拍测试（D30，零攻角发射）。"""

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
from ballistic_sim.dynamics.mpm import MPMOptions, MPMDynamics
from ballistic_sim.dynamics.six_dof import SixDOFDynamics
from ballistic_sim.models.aerodynamics import DragLaw
from ballistic_sim.models.atmosphere import StandardAtmosphere
from ballistic_sim.models.wind import UniformWind
from ballistic_sim.phases.events import make_ground_event
from tests.dynamics.test_six_dof import _d30_coeff_tables


def _make_ctx() -> DynamicContext:
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
        wind=UniformWind(),
        aero=None,
        gravity_model="wgs84",
        options={},
    )


def _make_sixdof() -> SixDOFDynamics:
    return SixDOFDynamics(
        mass_kg=21.76,
        diameter_m=0.122,
        form_factor=0.844,
        twist_cal=20.0,
        Ix=0.045,
        It=0.38,
        x_cp_cg=0.05,
        lat_deg=39.9,
        **_d30_coeff_tables(),
    )


def _make_mpm() -> MPMDynamics:
    # 自定义阻力律，使 MPM 与 6-DOF 的零攻角阻力系数一致：Cd0=0.25，Cd_alpha2=0.05。
    drag_law = DragLaw(
        name="custom",
        cd0_fn=lambda _ma: 0.25,
        cd_alpha2_fn=lambda _ma: 0.05,
    )
    return MPMDynamics(
        mass_kg=21.76,
        diameter_m=0.122,
        form_factor=0.844,
        drag_law=drag_law,
        twist_cal=20.0,
        Ix=0.045,
        It=0.38,
        lat_deg=39.9,
        options=MPMOptions(
            use_drag=True,
            use_wind=False,
            use_coriolis=True,
            use_spin=True,
            use_dynamic_alpha=False,
            method="DOP853",
            rtol=1e-6,
            atol=1e-9,
        ),
    )


def _fly_to_ground(dyn, ctx, t_max: float = 300.0) -> dict:
    """积分到落地并返回射程、TOF、最大高度。"""
    y0 = dyn.initial_state(v0=690.0, theta_deg=45.0, az_deg=90.0, h0=0.0)
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
    if not sol.success:
        raise RuntimeError(f"{type(dyn).__name__} 积分失败: {sol.message}")
    if sol.t_events is None or sol.t_events[0] is None or len(sol.t_events[0]) == 0:
        raise RuntimeError(f"{type(dyn).__name__} 未触发落地事件")
    t_end = float(sol.t_events[0][0])
    idx = int(np.argmin(np.abs(sol.t - t_end)))
    r_end = sol.y[0:3, idx]
    range_m = float(np.linalg.norm(r_end[0:2]))
    max_h = float(np.max(sol.y[2, :]))
    return {"range_m": range_m, "tof_s": t_end, "max_h_m": max_h}


@pytest.mark.slow
@pytest.mark.golden
def test_sixdof_vs_mpm_d30_zero_alpha() -> None:
    """D30 零攻角发射：6-DOF 与 MPM 的射程、TOF、最大高度相对误差 <2%。"""
    ctx = _make_ctx()
    six = _fly_to_ground(_make_sixdof(), ctx)
    mpm = _fly_to_ground(_make_mpm(), ctx)

    for key in ("range_m", "tof_s", "max_h_m"):
        ref = max(abs(six[key]), abs(mpm[key]), 1e-6)
        err = abs(six[key] - mpm[key]) / ref
        assert err < 0.02, f"{key} 相对误差 {err:.3%} 超过 2% (6-DOF={six[key]}, MPM={mpm[key]})"
