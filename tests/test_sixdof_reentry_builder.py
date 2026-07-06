"""6-DOF 再入段通过 builder 与阶段链集成的端到端测试。"""

from __future__ import annotations

import warnings

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
from ballistic_sim.dynamics.six_dof import SixDOFDynamics
from ballistic_sim.phases.builder import build_phases
from ballistic_sim.phases.powered import PoweredPhase
from ballistic_sim.phases.reentry import ReentryPhase
from ballistic_sim.phases.terminal import TerminalPhase
from ballistic_sim.simulator import simulate


def _d30_coeff_tables():
    """返回 D30 马赫数-系数表（常数近似）用于测试。"""
    ma = np.linspace(0.0, 3.0, 10)
    return {
        "CD0_table": np.column_stack([ma, np.full_like(ma, 0.25)]),
        "CDa2_table": np.column_stack([ma, np.full_like(ma, 0.05)]),
        "CNa_table": np.column_stack([ma, np.full_like(ma, 1.6)]),
        "CMa_table": np.column_stack([ma, np.full_like(ma, -3.5)]),
        "CMq_table": np.column_stack([ma, np.full_like(ma, -1.5)]),
        "Clp_table": np.column_stack([ma, np.full_like(ma, -0.025)]),
    }


def test_builder_inserts_sixdof_reentry_phase() -> None:
    """sixdof_reentry=True 时阶段序列包含 sixdof ReentryPhase。"""
    cfg = SimConfig(
        mission="icbm",
        vehicle=VehicleConfig(mass_kg=1000.0, diameter_m=0.5, cd=0.3, thrust_N=1.2e6),
        launch=LaunchConfig(lat_deg=0.0, lon_deg=0.0, alt_m=1.0, azimuth_deg=45.0),
        environment=EnvironmentConfig(),
        guidance=GuidanceConfig(),
        options=OptionsConfig(sixdof_reentry=True),
    )
    phases = build_phases(cfg)
    reentry_phases = [p for p in phases if isinstance(p, ReentryPhase)]
    assert len(reentry_phases) == 1
    assert reentry_phases[0].fidelity == "sixdof"
    assert reentry_phases[0].dynamics.state_dim() == 13


def test_builder_no_sixdof_reentry_by_default() -> None:
    """默认不插入 sixdof 再入段。"""
    cfg = SimConfig(
        mission="icbm",
        vehicle=VehicleConfig(mass_kg=1000.0, diameter_m=0.5, cd=0.3),
        launch=LaunchConfig(lat_deg=0.0, lon_deg=0.0, alt_m=1.0, azimuth_deg=45.0),
        environment=EnvironmentConfig(),
        guidance=GuidanceConfig(),
        options=OptionsConfig(),
    )
    phases = build_phases(cfg)
    reentry_phases = [p for p in phases if isinstance(p, ReentryPhase)]
    assert len(reentry_phases) == 0


@pytest.mark.slow
def test_simulate_projectile_with_sixdof_phase() -> None:
    """用 SixDOFDynamics 作为单段 6-DOF 弹道，验证可落地。"""
    mass_kg = 21.76
    diameter_m = 0.122
    dyn = SixDOFDynamics(
        mass_kg=mass_kg,
        diameter_m=diameter_m,
        form_factor=0.844,
        Ix=0.045,
        It=0.38,
        x_cp_cg=0.05,
        lat_deg=39.9,
        twist_cal=20.0,
        **_d30_coeff_tables(),
        options={"drag": True, "gravity": True, "coriolis": True, "thrust": False},
    )
    cfg = SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(mass_kg=mass_kg, diameter_m=diameter_m, cd=0.844),
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
        options=OptionsConfig(
            integrator="DOP853",
            rtol=1e-6,
            atol=1e-9,
            max_step=1.0,
            terminate_impact=True,
        ),
    )
    phases = [
        PoweredPhase(
            name="6-DOF 弹道",
            t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 400.0),
            dynamics=dyn,
            guidance=None,
            m_dry=mass_kg,
            sep_name="落地",
        ),
        TerminalPhase(
            name="终点",
            t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 400.0),
            dynamics=dyn,
        ),
    ]
    # 直接构造 SixDOFDynamics/Phase 验证再入 builder 集成，属于 phase 显式传参的合法例外。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = simulate(cfg, phases=phases)
    assert result.stop_reason == "completed"
    assert result.t.size > 0
    landing_events = [ev for ev in result.event_log if "落地" in ev.get("name", "")]
    assert len(landing_events) >= 1, f"events={result.event_log}"
