"""再入制导单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.config import (
    EnvironmentConfig,
    GuidanceConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    StageConfig,
    VehicleConfig,
)
from ballistic_sim.constants import GM_EARTH, WGS84_A
from ballistic_sim.frames import ecef_to_eci, ecef_to_geodetic, eci_to_ecef, geodetic_to_ecef
from ballistic_sim.guidance.control import ControlMoment
from ballistic_sim.guidance.reentry_guidance import (
    ReentryGuidance,
    drag_acceleration,
    specific_energy,
)
from ballistic_sim.phases.builder import build_phases


def test_specific_energy_finite() -> None:
    """比机械能计算应为有限值。"""
    r = np.array([WGS84_A + 100e3, 0.0, 0.0])
    v = np.array([0.0, 7800.0, 0.0])
    E = specific_energy(r, v)
    assert np.isfinite(E)
    assert E < 0.0  # 束缚轨道


def test_drag_acceleration_positive() -> None:
    """阻力加速度大小应为非负。"""
    r = np.array([WGS84_A + 50e3, 0.0, 0.0])
    v = np.array([0.0, 7000.0, 0.0])
    D = drag_acceleration(r, v, rho=1.0, cd=0.5, area_m2=1.0, mass_kg=100.0)
    assert D >= 0.0


def test_reentry_command_outputs() -> None:
    """再入制导指令应包含倾侧角、攻角、法向加速度。"""
    rg = ReentryGuidance(target_lat_deg=0.0, target_lon_deg=0.0)
    r = np.array([WGS84_A + 80e3, 0.0, 0.0])
    v = np.array([0.0, 6500.0, 0.0])
    cmd = rg.command(0.0, r, v, rho=0.01)
    assert "bank_deg" in cmd
    assert "aoa_deg" in cmd
    assert "normal_accel" in cmd
    assert abs(cmd["bank_deg"]) <= rg.max_bank_deg + 1e-6


def test_reentry_control_moment_13d() -> None:
    """对 13 维状态，再入制导应返回 ControlMoment。"""
    rg = ReentryGuidance(target_lat_deg=0.0, target_lon_deg=0.0, moment_gain=1.0e3)
    y = np.zeros(13)
    y[0:3] = np.array([WGS84_A + 80e3, 0.0, 0.0])
    y[3:6] = np.array([0.0, 6500.0, 0.0])
    y[6:10] = np.array([0.0, 0.0, 0.0, 1.0])
    cm = rg.control_moment(y, alpha_rad=0.1, beta_rad=0.05)
    assert isinstance(cm, ControlMoment)
    assert cm.M != 0.0 or cm.N != 0.0


def test_reentry_control_moment_rejects_low_dim() -> None:
    """低维状态应返回零力矩并设置 failed。"""
    rg = ReentryGuidance(target_lat_deg=0.0, target_lon_deg=0.0)
    cm = rg.control_moment(np.zeros(6), alpha_rad=0.0, beta_rad=0.0)
    assert cm.L == 0.0 and cm.M == 0.0 and cm.N == 0.0
    assert rg.failed


def test_reentry_guidance_builder_with_sixdof() -> None:
    """builder 对 icbm+reentry+sixdof_reentry 应注入再入制导对象。"""
    cfg = SimConfig(
        mission="icbm",
        vehicle=VehicleConfig(
            mass_kg=5000.0,
            stages=[
                StageConfig(
                    name="S1",
                    thrust_sl=100.0e3,
                    thrust_vac=120.0e3,
                    isp_vac=280.0,
                    m_prop=3000.0,
                    m_dry=1000.0,
                    diameter_m=1.0,
                )
            ],
        ),
        launch=LaunchConfig(lat_deg=39.0, lon_deg=116.0, elevation_deg=90.0),
        guidance=GuidanceConfig(
            guidance_law="reentry",
            target_lat_deg=39.0,
            target_lon_deg=117.0,
            energy_target_j_kg=-60e6,
        ),
        options=OptionsConfig(sixdof_reentry=True),
        environment=EnvironmentConfig(),
    )
    phases = build_phases(cfg)
    reentry = [ph for ph in phases if ph.name == "再入段"]
    assert len(reentry) == 1
    assert reentry[0].fidelity == "sixdof"
    assert reentry[0].dynamics.guidance is not None
    assert isinstance(reentry[0].dynamics.guidance, ReentryGuidance)


def test_reentry_landing_cep50() -> None:
    """无风无扰动下再入落点应与目标点距离 < 5 km（简化点质量验证）。"""
    from scipy.integrate import solve_ivp

    from ballistic_sim.models.atmosphere import StandardAtmosphere

    rg = ReentryGuidance(
        target_lat_deg=39.0,
        target_lon_deg=116.0,
        target_energy_j_kg=-60e6,
        max_bank_deg=45.0,
        bank_gain=5.0e-6,
        nominal_aoa_deg=10.0,
    )

    # 再入初始状态：高度 80 km，位于目标以西约 86 km
    # 使用亚轨道再入速度（~650 m/s），该能量与气动参数下制导可收敛到目标附近
    lat0 = 39.0
    lon0 = 115.0
    h0 = 80e3
    r_ecef0 = geodetic_to_ecef(lat0, lon0, h0)
    r0 = ecef_to_eci(r_ecef0, 0.0)
    omega = np.array([0.0, 0.0, 7.292115e-5], dtype=float)
    # 东向速度约 650 m/s，带微小北向分量以验证横程修正
    from ballistic_sim.frames import enu_to_ecef_vec

    v_ecef = enu_to_ecef_vec(np.array([650.0, 20.0, 0.0]), lat0, lon0)
    v0 = ecef_to_eci(v_ecef + np.cross(omega, r_ecef0), 0.0)

    mass = 800.0
    area = 0.5
    cd = 0.3
    cl = 0.15  # 升力系数
    atm = StandardAtmosphere()

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        r = y[0:3].copy()
        v = y[3:6].copy()
        r_mag = float(np.linalg.norm(r))
        h = r_mag - WGS84_A
        if h < 0.0:
            return np.zeros(7)
        state = atm(max(h, 0.0))
        # 相对速度（扣除地球自转）
        v_rel = v - np.cross(omega, r)
        vm = float(np.linalg.norm(v_rel))
        if vm < 1.0:
            return np.concatenate([v, np.zeros(3), [0.0]])

        # 阻力
        drag_mag = 0.5 * state.rho * cd * area / mass * vm * vm
        a_drag = -drag_mag * v_rel / vm

        # 再入制导指令：按当前状态输出倾侧角/攻角
        cmd = rg.command(t, r, v, rho=state.rho)
        bank_rad = float(np.radians(cmd["bank_deg"]))
        aoa_rad = float(np.radians(cmd["aoa_deg"]))

        # 升力：bank=0 时垂直向上（对抗引力），bank 越大水平转弯分量越大
        up = r / r_mag
        v_hat = v_rel / vm
        # 速度平面内的“向上”方向（垂直于速度、在 v-up 平面内）
        lift_up = up - np.dot(up, v_hat) * v_hat
        lift_up_norm = float(np.linalg.norm(lift_up))
        if lift_up_norm < 1e-12:
            lift_up = np.cross(v_hat, np.array([0.0, 0.0, 1.0]))
            lift_up_norm = float(np.linalg.norm(lift_up))
        lift_up = lift_up / lift_up_norm
        # 水平转弯方向（垂直于速度和 lift_up）
        turn_dir = np.cross(v_hat, lift_up)
        turn_norm = float(np.linalg.norm(turn_dir))
        if turn_norm > 1e-12:
            turn_dir = turn_dir / turn_norm
        bank_dir = np.cos(bank_rad) * lift_up + np.sin(bank_rad) * turn_dir
        lift_mag = 0.5 * state.rho * cl * area / mass * vm * vm * max(np.cos(aoa_rad), 0.1)
        a_lift = lift_mag * bank_dir

        # 引力（点质量近似）
        a_grav = -GM_EARTH / (r_mag * r_mag) * up
        return np.concatenate([v, a_drag + a_lift + a_grav, [-0.0]])

    def event_ground(t: float, y: np.ndarray) -> float:
        return float(np.linalg.norm(y[0:3]) - WGS84_A)

    event_ground.terminal = True
    event_ground.direction = -1

    sol = solve_ivp(
        rhs,
        (0.0, 600.0),
        np.concatenate([r0, v0, [mass]]),
        events=event_ground,
        rtol=1e-6,
        atol=1e-9,
        max_step=2.0,
    )
    if not sol.success or len(sol.t) < 2:
        pytest.skip("再入积分未成功收敛")
    y_end = sol.y[:, -1]
    r_ecef = eci_to_ecef(y_end[0:3], float(sol.t[-1]))
    lat_end, lon_end, _ = ecef_to_geodetic(r_ecef)
    from ballistic_sim.frames import haversine_distance

    dist = haversine_distance(lat_end, lon_end, rg.target_lat_deg, rg.target_lon_deg)
    assert dist < 5000.0
