"""PointMassDynamics RHS 重构后的回归与 smoke 测试。"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

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
from ballistic_sim.context import _resolve_dynamics_context
from ballistic_sim.dynamics.common import AeroEnv, DynamicContext
from ballistic_sim.dynamics.point_mass import PointMassDynamics
from ballistic_sim.models.aerodynamics import ConstantAeroModel
from ballistic_sim.models.atmosphere import StandardAtmosphere
from ballistic_sim.models.wind import UniformWind
from ballistic_sim.phases.base import PhaseContext
from ballistic_sim.phases.coasting import CoastingPhase
from ballistic_sim.simulator import simulate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = PROJECT_ROOT / "tests" / "golden" / "point_mass_rhs_baseline.json"


def _make_ctx(frame: str = "ECI", wind: UniformWind | None = None) -> DynamicContext:
    cfg = SimConfig(
        mission="projectile" if frame == "ENU" else "rocket",
        vehicle=VehicleConfig(mass_kg=100.0, diameter_m=0.5, cd=0.3),
        launch=LaunchConfig(
            lat_deg=30.0,
            lon_deg=45.0,
            alt_m=100.0,
            azimuth_deg=90.0,
            elevation_deg=45.0,
            v0_m_s=500.0,
        ),
        environment=EnvironmentConfig(atmosphere="isa", wind_m_s=[2.0, 1.0, 0.0]),
        guidance=GuidanceConfig(),
        options=OptionsConfig(),
    )
    return DynamicContext(
        cfg=cfg,
        atmosphere=StandardAtmosphere(),
        wind=wind or UniformWind(e=2.0, n=1.0, u=0.0),
        aero=ConstantAeroModel(cd=0.3),
        gravity_model="j2",
        options={},
    )


def _make_dyn(frame: str, ctx: DynamicContext) -> PointMassDynamics:
    return PointMassDynamics(
        frame=frame,  # type: ignore[arg-type]
        mass=100.0,
        Aref=0.25,
        lat0=ctx.cfg.launch.lat_deg,
        lon0=ctx.cfg.launch.lon_deg,
    )


def test_rhs_numerical_regression() -> None:
    """重构后 RHS 积分结果与 golden 基线一致。"""
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden: dict[str, list[dict[str, float]]] = json.load(f)

    for frame, t_end in [("ENU", 30.0), ("ECI", 60.0)]:
        ctx = _make_ctx(frame=frame)
        dyn = _make_dyn(frame, ctx)
        y0 = dyn.initial_state(
            v0=ctx.cfg.launch.v0_m_s,
            theta_deg=ctx.cfg.launch.elevation_deg,
            az_deg=ctx.cfg.launch.azimuth_deg,
            h0=ctx.cfg.launch.alt_m,
        )
        sol = solve_ivp(
            lambda t, y: dyn.rhs(t, y, ctx),
            (0.0, t_end),
            y0,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
            max_step=1.0,
            dense_output=True,
        )
        expected = golden[frame]
        for row in expected:
            t = row["t"]
            y = sol.sol(t)
            assert y[0] == pytest.approx(row["x"], rel=1e-9)
            assert y[1] == pytest.approx(row["y"], rel=1e-9)
            assert y[2] == pytest.approx(row["z"], rel=1e-9)
            assert y[3] == pytest.approx(row["vx"], rel=1e-9)
            assert y[4] == pytest.approx(row["vy"], rel=1e-9)
            assert y[5] == pytest.approx(row["vz"], rel=1e-9)


def test_aero_env_full_reuses_wind_and_atm() -> None:
    """rhs 对 atmosphere/wind 各只调用一次。"""
    ctx = _make_ctx(frame="ECI")
    atm_call_count = [0]
    wind_call_count = [0]

    real_atm = ctx.atmosphere

    def counted_atm(h: float) -> Any:
        atm_call_count[0] += 1
        return real_atm(h)

    real_wind = ctx.wind

    def counted_wind(h: float, s: float | None = None) -> Any:
        wind_call_count[0] += 1
        return real_wind(h, s)

    ctx.atmosphere = counted_atm
    ctx.wind = counted_wind
    dyn = _make_dyn("ECI", ctx)
    y0 = dyn.initial_state(
        v0=ctx.cfg.launch.v0_m_s,
        theta_deg=ctx.cfg.launch.elevation_deg,
        az_deg=ctx.cfg.launch.azimuth_deg,
        h0=ctx.cfg.launch.alt_m,
    )
    dyn.rhs(0.0, y0, ctx)
    assert atm_call_count[0] == 1
    assert wind_call_count[0] == 1


def test_rhs_enu_smoke_via_simulate() -> None:
    """ENU 框架下使用 PointMassDynamics 的 simulate smoke 测试。"""
    cfg = SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(mass_kg=10.0, diameter_m=0.1, cd=0.3),
        launch=LaunchConfig(
            lat_deg=0.0,
            lon_deg=0.0,
            alt_m=0.0,
            azimuth_deg=0.0,
            elevation_deg=45.0,
            v0_m_s=700.0,
        ),
        environment=EnvironmentConfig(wind_m_s=[2.0, 0.0, 0.0]),
        options=OptionsConfig(max_step=1.0),
    )
    ctx = _resolve_dynamics_context(cfg)
    cfg._dynamics_context = ctx  # type: ignore[attr-defined]
    dyn = PointMassDynamics(
        frame="ENU",
        mass=cfg.vehicle.mass_kg,
        Aref=np.pi * (cfg.vehicle.diameter_m / 2.0) ** 2,
        lat0=cfg.launch.lat_deg,
        lon0=cfg.launch.lon_deg,
    )
    phase = CoastingPhase(
        name="无动力",
        t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 300.0),
        dynamics=dyn,
        terrain=ctx.terrain,
        lat0=cfg.launch.lat_deg,
        lon0=cfg.launch.lon_deg,
    )
    # 直接构造 PointMassDynamics/Phase 进行 smoke 测试，属于 phase 显式传参的合法例外。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # 直接构造 PointMassDynamics/Phase 进行 smoke 测试，属于 phase 显式传参的合法例外。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = simulate(cfg, phases=[phase])
    assert result.stop_reason == "completed"
    assert len(result.t) > 2
    # 落地事件应触发
    assert result.y[-1, 2] < 10.0


def test_rhs_eci_smoke_via_simulate() -> None:
    """ECI 框架下使用 PointMassDynamics 的 simulate smoke 测试。"""
    cfg = SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(mass_kg=10.0, diameter_m=0.1, cd=0.3),
        launch=LaunchConfig(
            lat_deg=30.0,
            lon_deg=45.0,
            alt_m=100.0,
            azimuth_deg=90.0,
            elevation_deg=45.0,
            v0_m_s=700.0,
        ),
        environment=EnvironmentConfig(atmosphere="isa"),
        options=OptionsConfig(max_step=1.0),
    )
    ctx = _resolve_dynamics_context(cfg)
    cfg._dynamics_context = ctx  # type: ignore[attr-defined]
    dyn = PointMassDynamics(
        frame="ECI",
        mass=cfg.vehicle.mass_kg,
        Aref=np.pi * (cfg.vehicle.diameter_m / 2.0) ** 2,
        lat0=cfg.launch.lat_deg,
        lon0=cfg.launch.lon_deg,
    )
    phase = CoastingPhase(
        name="无动力",
        t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 300.0),
        dynamics=dyn,
        terrain=ctx.terrain,
        lat0=cfg.launch.lat_deg,
        lon0=cfg.launch.lon_deg,
    )
    # 直接构造 PointMassDynamics/Phase 进行 smoke 测试，属于 phase 显式传参的合法例外。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = simulate(cfg, phases=[phase])
    assert result.stop_reason == "completed"
    assert len(result.t) > 2


def test_aero_env_full_via_phase_context() -> None:
    """rhs 通过 PhaseContext 正确解析 DynamicContext。"""
    ctx = _make_ctx(frame="ENU")
    dyn = _make_dyn("ENU", ctx)
    y = np.array([0.0, 0.0, 100.0, 10.0, 0.0, 0.0, 10.0])
    phase = CoastingPhase(
        name="test",
        t_span=(0.0, 100.0),
        dynamics=dyn,
    )
    phase.cfg = ctx.cfg  # type: ignore[attr-defined]
    phase.cfg._dynamics_context = ctx  # type: ignore[attr-defined]
    pctx = PhaseContext(cfg=phase.cfg, phase=phase, t0=0.0)
    dy = dyn.rhs(0.0, y, pctx)
    assert dy.shape == (7,)


def test_aero_env_lightweight_wrapper_signature_unchanged() -> None:
    """_aero_env 保持原有签名并返回 AeroEnv。"""
    ctx = _make_ctx(frame="ENU")
    dyn = _make_dyn("ENU", ctx)
    env = dyn._aero_env(ctx, np.array([0.0, 0.0, 100.0]), np.array([10.0, 0.0, 0.0]))
    assert isinstance(env, AeroEnv)
    assert env.rho > 0.0
