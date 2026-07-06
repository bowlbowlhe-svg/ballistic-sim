"""弹丸 preset 经 build_phases 构造 MPM 动力学回归测试。"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from ballistic_sim.dynamics.mpm import MPMDynamics
from ballistic_sim.phases.builder import build_phases
from ballistic_sim.presets.projectiles import m107_config
from ballistic_sim.simulator import simulate


def test_m107_build_phases_mpm_dynamics_matches_preset() -> None:
    """build_phases(m107_config()) 应把 YAML 中的 twist_cal/Ix/It 传入 MPMDynamics。"""
    cfg = m107_config()
    phases = build_phases(cfg)
    assert len(phases) == 2
    dyn = phases[0].dynamics
    assert isinstance(dyn, MPMDynamics)
    assert dyn.twist_cal == pytest.approx(20.0)
    assert dyn.Ix == pytest.approx(0.144)
    assert dyn.It == pytest.approx(1.22)
    assert dyn.mass_kg == pytest.approx(43.2)
    assert dyn.diameter_m == pytest.approx(0.155)
    assert dyn.form_factor == pytest.approx(0.78)


def test_m107_trajectory_regression(golden_m107: Dict[str, Any]) -> None:
    """m107 preset 走 build_phases 后的 trajectory 与 Stage 1 前 golden 一致（容差稍宽）。"""
    cfg = m107_config()
    phases = build_phases(cfg)
    result = simulate(cfg, phases)

    y_end = result.y[-1]
    scalars = golden_m107["scalars"]
    range_m = float(np.linalg.norm(y_end[0:2]))
    v_impact = float(np.linalg.norm(y_end[3:6]))
    tof_s = float(result.t[-1])
    max_alt_m = float(np.max(result.y[:, 2]))

    assert range_m == pytest.approx(scalars["range_m"], rel=1e-3)
    assert v_impact == pytest.approx(scalars["v_impact_m_s"], rel=1e-3)
    assert tof_s == pytest.approx(scalars["tof_s"], rel=1e-3)
    assert max_alt_m == pytest.approx(scalars["max_alt_m"], rel=1e-3)


def test_default_vehicle_config_explicit_equals_implicit() -> None:
    """VehicleConfig 显式默认值与隐式默认值应产生一致的 trajectory。"""
    from ballistic_sim.config import SimConfig, VehicleConfig

    cfg_implicit = SimConfig(mission="projectile", vehicle=VehicleConfig())
    cfg_explicit = SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(
            twist_cal=20.0,
            Ix=0.1,
            It=1.0,
            x_cp_cg=0.05,
        ),
    )
    res_implicit = simulate(cfg_implicit, build_phases(cfg_implicit))
    res_explicit = simulate(cfg_explicit, build_phases(cfg_explicit))

    assert res_implicit.t[-1] == pytest.approx(res_explicit.t[-1], rel=1e-12)
    assert float(np.linalg.norm(res_implicit.y[-1, 0:2])) == pytest.approx(
        float(np.linalg.norm(res_explicit.y[-1, 0:2])), rel=1e-12
    )


def test_m107_default_vehicle_config_values() -> None:
    """未显式覆盖时 VehicleConfig 默认值应走 builder 进入 MPMDynamics。"""
    from ballistic_sim.config import SimConfig, VehicleConfig

    cfg = SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(),
    )
    phases = build_phases(cfg)
    dyn = phases[0].dynamics
    assert isinstance(dyn, MPMDynamics)
    assert dyn.twist_cal == pytest.approx(20.0)
    assert dyn.Ix == pytest.approx(0.1)
    assert dyn.It == pytest.approx(1.0)
