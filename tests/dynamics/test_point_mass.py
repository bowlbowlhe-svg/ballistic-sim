"""3-DOF 质点动力学单元测试。"""

from __future__ import annotations

import numpy as np

from ballistic_sim.config import (
    EnvironmentConfig,
    GuidanceConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    VehicleConfig,
)
from ballistic_sim.dynamics.common import AeroEnv, DynamicContext
from ballistic_sim.dynamics.point_mass import PointMassDynamics
from ballistic_sim.frames import ecef_to_eci, geodetic_to_ecef
from ballistic_sim.models.aerodynamics import ConstantAeroModel
from ballistic_sim.models.atmosphere import StandardAtmosphere
from ballistic_sim.models.wind import UniformWind


def _make_ctx(frame: str = "ECI", wind: UniformWind | None = None) -> DynamicContext:
    cfg = SimConfig(
        mission="projectile" if frame == "ENU" else "rocket",
        vehicle=VehicleConfig(mass_kg=100.0, diameter_m=0.5, cd=0.3),
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
        wind=wind or UniformWind(),
        aero=ConstantAeroModel(cd=0.3),
        gravity_model="j2",
        options={},
    )


def test_point_mass_dynamics_initialization() -> None:
    """PointMassDynamics 构造参数被正确保存。"""
    dyn = PointMassDynamics(frame="ENU", mass=10.0, Aref=0.2)
    assert dyn.frame == "ENU"
    assert dyn.mass == 10.0
    assert dyn.Aref == 0.2
    assert dyn.options.get("drag") is True


def test_rhs_enu_basic_shape() -> None:
    """ENU 模式下 rhs 返回 7 维导数。"""
    dyn = PointMassDynamics(frame="ENU", mass=10.0, Aref=0.1)
    ctx = _make_ctx(frame="ENU")
    y = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0, 10.0])
    dy = dyn.rhs(0.0, y, ctx)
    assert dy.shape == (7,)
    assert dy[6] == 0.0


def test_rhs_eci_basic_shape() -> None:
    """ECI 模式下 rhs 返回 7 维导数。"""
    dyn = PointMassDynamics(frame="ECI", mass=10.0, Aref=0.1)
    ctx = _make_ctx(frame="ECI")
    r_ecef = geodetic_to_ecef(0.0, 0.0, 100.0)
    r_eci = ecef_to_eci(r_ecef, 0.0)
    v_eci = np.array([10.0, 0.0, 0.0])
    y = np.concatenate([r_eci, v_eci, [10.0]])
    dy = dyn.rhs(0.0, y, ctx)
    assert dy.shape == (7,)


def test_drag_accel_with_zero_density() -> None:
    """密度为 0 时阻力加速度为零。"""
    dyn = PointMassDynamics(frame="ECI", mass=10.0, Aref=0.1)
    env = AeroEnv(rho=0.0, c=300.0, p=0.0, T=200.0, q=0.0, Ma=0.0)
    a_drag = dyn._drag_accel(_make_ctx(), np.array([100.0, 0.0, 0.0]), env, 10.0)
    assert np.allclose(a_drag, 0.0)


def test_drag_accel_opposes_velocity() -> None:
    """阻力加速度方向与相对速度相反。"""
    dyn = PointMassDynamics(frame="ECI", mass=10.0, Aref=1.0)
    env = AeroEnv(rho=1.225, c=340.0, p=101325.0, T=288.0, q=0.0, Ma=0.0)
    v_rel = np.array([100.0, 0.0, 0.0])
    a_drag = dyn._drag_accel(_make_ctx(), v_rel, env, 10.0)
    assert a_drag[0] < 0.0
    assert np.isclose(a_drag[1], 0.0, atol=1e-12)
    assert np.isclose(a_drag[2], 0.0, atol=1e-12)


def test_drag_disabled_option() -> None:
    """options drag=False 时阻力加速度为零。"""
    dyn = PointMassDynamics(frame="ECI", mass=10.0, Aref=1.0, options={"drag": False})
    env = AeroEnv(rho=1.225, c=340.0, p=101325.0, T=288.0, q=0.0, Ma=0.0)
    a_drag = dyn._drag_accel(_make_ctx(), np.array([100.0, 0.0, 0.0]), env, 10.0)
    assert np.allclose(a_drag, 0.0)


def test_telemetry_returns_expected_keys() -> None:
    """telemetry 返回高度、速度、动压、马赫数、密度。"""
    dyn = PointMassDynamics(frame="ENU", mass=10.0)
    ctx = _make_ctx(frame="ENU")
    y = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0, 10.0])
    tel = dyn.telemetry(0.0, y, ctx)
    assert set(tel.keys()) == {"h", "v_inertial", "q", "Ma", "rho"}
    assert tel["h"] == 100.0


def test_coriolis_option_enu() -> None:
    """ENU 模式下启用科氏力会产生非零侧向加速度。"""
    dyn = PointMassDynamics(
        frame="ENU", mass=10.0, Aref=0.1, options={"drag": False, "coriolis": True}
    )
    ctx = _make_ctx(frame="ENU")
    y = np.array([0.0, 0.0, 100.0, 100.0, 0.0, 0.0, 10.0])
    dy = dyn.rhs(0.0, y, ctx)
    assert abs(dy[5]) > 0.0


def test_rhs_eci_uses_height_for_wind() -> None:
    """ECI 模式下 wind(h) 按几何高度调用而非固定 0。"""
    from ballistic_sim.models.wind import UniformWind

    class LoggingWind(UniformWind):
        """记录被查询高度的风模型。"""

        def __init__(self):
            super().__init__()
            self.heights: list[float] = []

        def __call__(self, h, s=None):
            self.heights.append(float(h))
            return super().__call__(h, s)

    wind = LoggingWind()
    ctx = _make_ctx(frame="ECI", wind=wind)
    dyn = PointMassDynamics(frame="ECI", mass=10.0, Aref=0.1)
    r_ecef = geodetic_to_ecef(0.0, 0.0, 100.0)
    r_eci = ecef_to_eci(r_ecef, 0.0)
    v_eci = np.array([10.0, 0.0, 0.0])
    y = np.concatenate([r_eci, v_eci, [10.0]])
    dyn.rhs(0.0, y, ctx)
    assert len(wind.heights) >= 1
    assert all(h > 0.0 for h in wind.heights)
