"""补充 PoweredECIDynamics 测试以提升覆盖率。"""

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
from ballistic_sim.dynamics.common import DynamicContext
from ballistic_sim.constants import WGS84_A
from ballistic_sim.dynamics.powered_eci import PoweredECIDynamics
from ballistic_sim.models.atmosphere import StandardAtmosphere
from ballistic_sim.models.wind import UniformWind


def _make_ctx() -> DynamicContext:
    cfg = SimConfig(
        mission="rocket",
        vehicle=VehicleConfig(mass_kg=1000.0, diameter_m=0.5, cd=0.3),
        launch=LaunchConfig(
            lat_deg=0.0,
            lon_deg=0.0,
            alt_m=0.0,
            azimuth_deg=90.0,
            elevation_deg=90.0,
            v0_m_s=0.0,
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
        gravity_model="j2",
        options={},
    )


def test_powered_eci_initial_state() -> None:
    """initial_state 返回 7 维状态。"""
    stage = {
        "name": "test",
        "thrust_vac": 1e5,
        "thrust_sl": 1e5,
        "isp_vac": 300.0,
        "m_prop": 100.0,
        "m_dry": 50.0,
        "Aref": 1.0,
    }
    dyn = PoweredECIDynamics(stage=stage, guidance={"lat_deg": 0.0, "lon_deg": 0.0})
    y0 = dyn.initial_state(0.0, 90.0, 90.0)
    assert y0.size == 7
    assert y0[6] == 150.0


def test_powered_eci_rhs_no_thrust() -> None:
    """关闭推力时仅受重力。"""
    stage = {
        "name": "coast",
        "thrust_vac": 0.0,
        "thrust_sl": 0.0,
        "isp_vac": 300.0,
        "m_prop": 0.0,
        "m_dry": 100.0,
        "Aref": 1.0,
    }
    dyn = PoweredECIDynamics(
        stage=stage,
        guidance={"lat_deg": 0.0, "lon_deg": 0.0},
        modes={"thrust": False, "drag": False, "j2": False},
    )
    ctx = _make_ctx()
    r = np.array([WGS84_A, 0.0, 0.0])
    v = np.array([0.0, 1000.0, 0.0])
    y = np.concatenate([r, v, [100.0]])
    dy = dyn.rhs(0.0, y, ctx)
    # 无推力无阻力无 J2 时点质量重力
    assert np.linalg.norm(dy[3:6]) > 0.0
    assert dy[6] == 0.0


def test_powered_eci_telemetry() -> None:
    """telemetry 返回派生量字典。"""
    stage = {
        "name": "test",
        "thrust_vac": 1e5,
        "thrust_sl": 1e5,
        "isp_vac": 300.0,
        "m_prop": 100.0,
        "m_dry": 50.0,
        "Aref": 1.0,
    }
    dyn = PoweredECIDynamics(
        stage=stage,
        guidance={
            "lat_deg": 0.0,
            "lon_deg": 0.0,
            "azimuth_deg": 90.0,
            "t_pitchover": 10.0,
            "t_kick_end": 30.0,
            "kick_deg": 15.0,
        },
    )
    ctx = _make_ctx()
    y = dyn.initial_state(0.0, 90.0, 90.0)
    tel = dyn.telemetry(0.0, y, ctx)
    assert "h" in tel
    assert "v_inertial" in tel
    assert "thrust" in tel
    assert "accel_g" in tel


def test_powered_eci_state_dim_and_frame() -> None:
    """状态维度为 7，坐标系为 ECI。"""
    dyn = PoweredECIDynamics(
        stage={"thrust_sl": 0.0, "thrust_vac": 0.0, "isp_vac": 300.0, "Aref": 1.0},
        guidance={},
    )
    assert dyn.state_dim() == 7
    assert dyn.native_frame() == "ECI"


def test_powered_eci_upperstage_guidance() -> None:
    """use_upperstage=True 时使用上面级线性俯仰律。"""
    stage = {
        "name": "test",
        "thrust_vac": 1e5,
        "thrust_sl": 1e5,
        "isp_vac": 300.0,
        "m_prop": 100.0,
        "m_dry": 50.0,
        "Aref": 1.0,
    }
    guid = {
        "lat_deg": 0.0,
        "lon_deg": 0.0,
        "azimuth_deg": 90.0,
        "phase": "upperstage",
        "t_us_start": 0.0,
        "t_us_dur": 50.0,
        "gamma_end_deg": 0.0,
    }
    dyn = PoweredECIDynamics(stage=stage, guidance=guid, use_upperstage=True)
    ctx = _make_ctx()
    y = dyn.initial_state(0.0, 90.0, 90.0)
    dy = dyn.rhs(0.0, y, ctx)
    assert dy.size == 7
