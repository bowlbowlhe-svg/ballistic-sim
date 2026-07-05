"""能量管理制导单元测试。"""

from __future__ import annotations

import numpy as np

from ballistic_sim.config import (
    EnvironmentConfig,
    GuidanceConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    StageConfig,
    VehicleConfig,
)
from ballistic_sim.constants import WGS84_A
from ballistic_sim.guidance.control import ControlMoment
from ballistic_sim.guidance.energy_management import (
    EnergyManagementGuidance,
    specific_energy,
)
from ballistic_sim.phases.builder import build_phases


def test_energy_specific_energy() -> None:
    """比机械能公式符号与量级正确。"""
    r = np.array([WGS84_A + 100e3, 0.0, 0.0])
    v = np.array([0.0, 7800.0, 0.0])
    E = specific_energy(r, v)
    assert E < 0.0
    # v 降为 0 时势能为负但绝对值更大
    E2 = specific_energy(r, np.zeros(3))
    assert E2 < E


def test_energy_command_against_profile() -> None:
    """能量高于目标剖面时应输出正法向加速度（拉起）。"""
    em = EnergyManagementGuidance(
        energy_target_j_kg=-50e6,
        energy_slope_j_kg_m=-1.0,
        kp=0.1,
    )
    r = np.array([WGS84_A + 50e3, 0.0, 0.0])
    v = np.array([0.0, 5000.0, 0.0])
    cmd = em.command(0.0, r, v)
    E_now = specific_energy(r, v)
    E_target = em._target_energy_at_height(50e3)
    if E_now > E_target:
        assert cmd["normal_accel"] > 0.0
    assert abs(cmd["normal_accel"]) <= em.max_normal_accel + 1e-6


def test_energy_command_below_profile() -> None:
    """能量低于目标剖面时应输出负法向加速度（俯冲）。"""
    em = EnergyManagementGuidance(
        energy_target_j_kg=-10e6,
        energy_slope_j_kg_m=-1.0,
        kp=0.1,
    )
    r = np.array([WGS84_A + 50e3, 0.0, 0.0])
    v = np.array([0.0, 500.0, 0.0])
    cmd = em.command(0.0, r, v)
    assert cmd["normal_accel"] < 0.0


def test_energy_control_moment() -> None:
    """能量管理制导应返回非零控制力矩。"""
    em = EnergyManagementGuidance(
        energy_target_j_kg=-50e6,
        kp=0.1,
        moment_gain=1.0e3,
    )
    y = np.zeros(13)
    y[0:3] = np.array([WGS84_A + 50e3, 0.0, 0.0])
    y[3:6] = np.array([0.0, 5000.0, 0.0])
    y[6:10] = np.array([0.0, 0.0, 0.0, 1.0])
    cm = em.control_moment(y, alpha_rad=0.2, beta_rad=0.05)
    assert isinstance(cm, ControlMoment)
    assert cm.M != 0.0


def test_energy_set_target_profile() -> None:
    """set_target_profile 应更新剖面参数。"""
    em = EnergyManagementGuidance()
    em.set_target_profile(-40e6, -2.0)
    assert em.energy_target_j_kg == -40e6
    assert em.energy_slope_j_kg_m == -2.0


def test_energy_builder_with_sixdof() -> None:
    """builder 对 suborbital+energy+sixdof_reentry 应注入能量管理制导对象。"""
    cfg = SimConfig(
        mission="suborbital",
        vehicle=VehicleConfig(
            mass_kg=3000.0,
            stages=[
                StageConfig(
                    name="S1",
                    thrust_sl=80.0e3,
                    thrust_vac=100.0e3,
                    isp_vac=280.0,
                    m_prop=2000.0,
                    m_dry=500.0,
                    diameter_m=1.0,
                )
            ],
        ),
        launch=LaunchConfig(lat_deg=28.5, lon_deg=-80.0, elevation_deg=90.0),
        guidance=GuidanceConfig(
            guidance_law="energy",
            target_lat_deg=28.5,
            target_lon_deg=-79.0,
            energy_target_j_kg=-55e6,
        ),
        options=OptionsConfig(sixdof_reentry=True),
        environment=EnvironmentConfig(),
    )
    phases = build_phases(cfg)
    reentry = [ph for ph in phases if ph.name == "再入段"]
    assert len(reentry) == 1
    assert isinstance(reentry[0].dynamics.guidance, EnergyManagementGuidance)
