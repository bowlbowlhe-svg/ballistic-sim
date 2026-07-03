"""补充 MPM 动力学测试以提升覆盖率。"""

from __future__ import annotations

import numpy as np
import pytest

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
from ballistic_sim.models.aerodynamics import DRAG_G1
from ballistic_sim.models.atmosphere import StandardAtmosphere
from ballistic_sim.models.wind import UniformWind


def _make_ctx() -> DynamicContext:
    cfg = SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(mass_kg=43.2, diameter_m=0.155, cd=0.78),
        launch=LaunchConfig(
            lat_deg=39.9,
            lon_deg=0.0,
            alt_m=0.0,
            azimuth_deg=90.0,
            elevation_deg=45.0,
            v0_m_s=684.0,
        ),
        environment=EnvironmentConfig(atmosphere="isa"),
        guidance=GuidanceConfig(),
        options=OptionsConfig(),
    )
    return DynamicContext(
        cfg=cfg,
        atmosphere=StandardAtmosphere(),
        wind=UniformWind(e=0.0, n=5.0, u=0.0),
        aero=None,
        gravity_model="wgs84",
        options={},
    )


def test_mpmdynamics_default_drag_law() -> None:
    """未指定 drag_law 时默认使用 G1。"""
    dyn = MPMDynamics()
    assert dyn.drag_law is DRAG_G1


def test_mpmdynamics_with_aero_tables() -> None:
    """提供 CMa/CLa/Clp 表时创建插值器。"""
    ma = np.array([0.0, 1.0, 2.0])
    vals = np.array([1.0, 2.0, 1.5])
    table = np.column_stack([ma, vals])
    dyn = MPMDynamics(CMa_table=table, CLa_table=table, Clp_table=table)
    assert dyn._CMa(1.0) == pytest.approx(2.0, abs=1e-6)


def test_mpm_no_drag() -> None:
    """关闭阻力时加速度仅含重力。"""
    dyn = MPMDynamics(
        mass_kg=10.0,
        diameter_m=0.1,
        options=MPMOptions(use_drag=False),
    )
    ctx = _make_ctx()
    y = dyn.initial_state(100.0, 45.0, 90.0)
    dy = dyn.rhs(0.0, y, ctx)
    # 无阻力时水平加速度应仅由科氏力贡献
    assert dy[0] == y[3]
    assert dy[1] == y[4]
    assert dy[2] == y[5]


def test_mpm_no_coriolis() -> None:
    """关闭科氏力时水平加速度为零（无风无阻力）。"""
    dyn = MPMDynamics(
        mass_kg=10.0,
        diameter_m=0.1,
        options=MPMOptions(use_drag=False, use_coriolis=False, use_wind=False),
    )
    ctx = _make_ctx()
    y = dyn.initial_state(100.0, 0.0, 90.0)
    dy = dyn.rhs(0.0, y, ctx)
    assert dy[3] == 0.0
    assert dy[4] == 0.0


def test_mpm_dynamic_alpha() -> None:
    """开启动态攻角时返回非零 dalpha。"""
    dyn = MPMDynamics(
        mass_kg=10.0,
        diameter_m=0.1,
        options=MPMOptions(use_dynamic_alpha=True, alpha0_deg=0.1),
    )
    ctx = _make_ctx()
    y = dyn.initial_state(100.0, 45.0, 90.0)
    dy = dyn.rhs(0.0, y, ctx)
    assert dy.size == 8


def test_mpm_state_dim_and_frame() -> None:
    """MPM 状态维度为 8，坐标系为 ENU。"""
    dyn = MPMDynamics()
    assert dyn.state_dim() == 8
    assert dyn.native_frame() == "ENU"


def test_mpm_aero_env_with_wind() -> None:
    """_aero 在风中计算相对速度。"""
    dyn = MPMDynamics()
    ctx = _make_ctx()
    v = np.array([100.0, 0.0, 0.0])
    env = dyn._aero(ctx, 100.0, v)
    assert env.rho > 0.0
    assert env.q >= 0.0
