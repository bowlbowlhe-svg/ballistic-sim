"""再入制导补充单元测试：边界条件、纯算法与失败回退。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ballistic_sim.constants import WGS84_A
from ballistic_sim.frames import ecef_to_eci, enu_to_ecef_vec, geodetic_to_ecef
from ballistic_sim.guidance.reentry_guidance import (
    ReentryGuidance,
    drag_acceleration,
    specific_energy,
)


def test_specific_energy_zero_radius_returns_zero() -> None:
    """r 接近零时比机械能应返回 0，避免除零。"""
    assert specific_energy(np.zeros(3), np.array([100.0, 0.0, 0.0])) == 0.0


def test_specific_energy_scalar_and_list_input() -> None:
    """specific_energy 应接受标量/列表输入并 reshape 为三维。"""
    E = specific_energy([WGS84_A + 50e3, 0.0, 0.0], [0.0, 7000.0, 0.0])
    assert np.isfinite(E)


def test_drag_acceleration_zero_density() -> None:
    """密度 <= 0 时阻力加速度应为 0。"""
    r = np.array([WGS84_A + 50e3, 0.0, 0.0])
    v = np.array([0.0, 7000.0, 0.0])
    assert drag_acceleration(r, v, rho=0.0, cd=0.5, area_m2=1.0, mass_kg=100.0) == 0.0
    assert drag_acceleration(r, v, rho=-1.0, cd=0.5, area_m2=1.0, mass_kg=100.0) == 0.0


def test_drag_acceleration_zero_or_negative_mass() -> None:
    """质量 <= 0 时阻力加速度应为 0。"""
    r = np.array([WGS84_A + 50e3, 0.0, 0.0])
    v = np.array([0.0, 7000.0, 0.0])
    assert drag_acceleration(r, v, rho=1.0, cd=0.5, area_m2=1.0, mass_kg=0.0) == 0.0
    assert drag_acceleration(r, v, rho=1.0, cd=0.5, area_m2=1.0, mass_kg=-10.0) == 0.0


def test_drag_acceleration_zero_relative_velocity() -> None:
    """相对速度接近零时阻力加速度应为 0。"""
    r = np.array([WGS84_A + 50e3, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 7.292115e-5], dtype=float)
    # 使惯性速度与地球自转同步，从而相对速度近似为 0
    v = np.cross(omega, r)
    D = drag_acceleration(r, v, rho=1.0, cd=0.5, area_m2=1.0, mass_kg=100.0)
    assert D == pytest.approx(0.0, abs=1.0)


def test_drag_acceleration_finite_and_non_negative() -> None:
    """正常输入下阻力加速度应为有限非负值。"""
    r = np.array([WGS84_A + 50e3, 0.0, 0.0])
    v = np.array([0.0, 7000.0, 0.0])
    D = drag_acceleration(r, v, rho=1.225, cd=0.5, area_m2=2.0, mass_kg=1000.0)
    assert math.isfinite(D)
    assert D >= 0.0


def test_drag_acceleration_uses_relative_velocity() -> None:
    """drag_acceleration 扣除地球自转后的相对速度。"""
    r = np.array([WGS84_A + 50e3, 0.0, 0.0])
    # 与地球自转同步的 ECI 速度，相对速度近似为 0
    v_sync = np.cross(np.array([0.0, 0.0, 7.292115e-5]), r)
    D = drag_acceleration(r, v_sync, rho=1.0, cd=1.0, area_m2=1.0, mass_kg=1.0)
    assert D == pytest.approx(0.0, abs=1.0)


def test_reentry_guidance_default_target_energy() -> None:
    """未指定 target_energy_j_kg 时应使用默认海平面估算。"""
    rg = ReentryGuidance()
    expected = -0.5 * 9.80665 * WGS84_A
    assert rg._target_energy() == pytest.approx(expected, rel=1e-6)


def test_reentry_guidance_set_target_chain() -> None:
    """set_target 应返回 self 并更新经纬度与能量。"""
    rg = ReentryGuidance()
    assert rg.set_target(10.0, 20.0, -50e6) is rg
    assert rg.target_lat_deg == 10.0
    assert rg.target_lon_deg == 20.0
    assert rg.target_energy_j_kg == -50e6


def test_reentry_guidance_set_target_preserves_energy_when_none() -> None:
    """set_target 不传入 energy 时应保留原有能量设置。"""
    rg = ReentryGuidance(target_energy_j_kg=-55e6)
    rg.set_target(1.0, 2.0)
    assert rg.target_energy_j_kg == -55e6


def test_reentry_guidance_command_with_nan_input() -> None:
    """输入 NaN 时应设置 failed 标志并返回安全默认指令。"""
    rg = ReentryGuidance()
    r = np.array([np.nan, 0.0, 0.0])
    v = np.array([0.0, 1000.0, 0.0])
    cmd = rg.command(0.0, r, v, rho=0.1)
    assert rg.failed
    assert cmd["failed"]
    assert cmd["bank_deg"] == 0.0
    assert cmd["aoa_deg"] == rg.nominal_aoa_deg


def test_reentry_guidance_command_with_inf_input() -> None:
    """输入 inf 时应设置 failed 标志。"""
    rg = ReentryGuidance()
    r = np.array([np.inf, 0.0, 0.0])
    v = np.array([0.0, 1000.0, 0.0])
    rg.command(0.0, r, v, rho=0.1)
    assert rg.failed


def test_reentry_guidance_command_zero_relative_velocity() -> None:
    """相对速度为零时 command 应返回有限指令且不抛异常。"""
    rg = ReentryGuidance()
    r = np.array([WGS84_A + 50e3, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 7.292115e-5], dtype=float)
    v = np.cross(omega, r)
    cmd = rg.command(0.0, r, v, rho=0.1)
    assert math.isfinite(cmd["bank_deg"])
    assert math.isfinite(cmd["normal_accel"])
    assert not cmd["failed"]


def test_reentry_guidance_command_aoa_clipped_to_minimum() -> None:
    """能量远低于目标剖面时攻角应被裁剪到下限 1°。"""
    rg = ReentryGuidance(target_energy_j_kg=-50e6, nominal_aoa_deg=10.0)
    # 低能量 + 小射程 -> aoa 计算值远小于 1°
    r = np.array([WGS84_A + 1e3, 0.0, 0.0])
    v = np.array([0.0, 10.0, 0.0])
    cmd = rg.command(0.0, r, v, rho=0.1)
    assert cmd["aoa_deg"] == pytest.approx(1.0, abs=1e-6)


def test_reentry_guidance_command_aoa_clipped_to_maximum() -> None:
    """能量远高于目标剖面时攻角应被裁剪到上限 30°。"""
    rg = ReentryGuidance(target_energy_j_kg=-100e6, nominal_aoa_deg=10.0)
    # 高能量 + 小射程 -> aoa 计算值远大于 30°
    r = np.array([WGS84_A + 1e3, 0.0, 0.0])
    v = np.array([0.0, 8000.0, 0.0])
    cmd = rg.command(0.0, r, v, rho=0.1)
    assert cmd["aoa_deg"] == pytest.approx(30.0, abs=1e-6)


def test_reentry_guidance_command_bank_clipped_to_max() -> None:
    """能量误差极大、射程极小时倾侧角应被裁剪到最大值。"""
    rg = ReentryGuidance(target_energy_j_kg=-100e6, max_bank_deg=45.0, bank_gain=1.0e-3)
    r = np.array([WGS84_A + 1e3, 0.0, 0.0])
    v = np.array([0.0, 8000.0, 0.0])
    cmd = rg.command(0.0, r, v, rho=0.1)
    assert abs(cmd["bank_deg"]) <= 45.0 + 1e-6
    assert abs(cmd["bank_deg"]) == pytest.approx(45.0, abs=1e-6)


def test_reentry_guidance_command_normal_accel_nonzero_when_bank() -> None:
    """存在非零倾侧角时法向加速度应为正。"""
    rg = ReentryGuidance(target_energy_j_kg=-100e6, max_bank_deg=45.0, bank_gain=1.0e-3)
    r = np.array([WGS84_A + 1e3, 0.0, 0.0])
    v = np.array([0.0, 8000.0, 0.0])
    cmd = rg.command(0.0, r, v, rho=0.1)
    assert cmd["normal_accel"] > 0.0


def test_reentry_guidance_command_normal_accel_zero_when_stationary() -> None:
    """速度为零时法向加速度应为 0。"""
    rg = ReentryGuidance()
    r = np.array([WGS84_A + 50e3, 0.0, 0.0])
    cmd = rg.command(0.0, r, np.zeros(3), rho=0.1)
    assert cmd["normal_accel"] == 0.0


def test_reentry_guidance_range_to_go_at_target_zero() -> None:
    """当前位置正好在目标点时剩余射程为 0。"""
    rg = ReentryGuidance(target_lat_deg=39.0, target_lon_deg=116.0)
    r_ecef = geodetic_to_ecef(39.0, 116.0, 80e3)
    r = ecef_to_eci(r_ecef, 0.0)
    assert rg._range_to_go(r) == pytest.approx(0.0, abs=1.0)


def test_reentry_guidance_range_to_go_antipode() -> None:
    """当前位置在目标对跖点时剩余射程约为半圆周（使用 haversine 平均半径）。"""
    rg = ReentryGuidance(target_lat_deg=0.0, target_lon_deg=0.0)
    r_ecef = geodetic_to_ecef(0.0, 180.0, 0.0)
    r = ecef_to_eci(r_ecef, 0.0)
    rng = rg._range_to_go(r)
    assert rng == pytest.approx(np.pi * 6371000.0, rel=1e-3)


def test_reentry_guidance_cross_track_error_left_turn() -> None:
    """目标在速度方向左侧时横程误差符号应使倾侧角为正。"""
    rg = ReentryGuidance(target_lat_deg=39.0, target_lon_deg=116.0)
    # 位于目标以西，速度朝正北；目标在右侧（东侧）
    r_ecef = geodetic_to_ecef(39.0, 115.0, 80e3)
    r = ecef_to_eci(r_ecef, 0.0)
    # ENU 北向速度
    v = ecef_to_eci(np.array([0.0, 700.0, 0.0]), 0.0)
    err = rg._cross_track_error(r, v)
    cmd = rg.command(0.0, r, v, rho=0.1)
    # 目标在东（右侧），err < 0，bank_sign = 1，bank_deg > 0
    assert err < 0.0
    assert cmd["bank_deg"] > 0.0


def test_reentry_guidance_cross_track_error_right_turn() -> None:
    """目标在速度方向右侧时横程误差符号应使倾侧角为负。"""
    rg = ReentryGuidance(target_lat_deg=39.0, target_lon_deg=116.0)
    r_ecef = geodetic_to_ecef(39.0, 117.0, 80e3)
    r = ecef_to_eci(r_ecef, 0.0)
    # 朝正北，目标在西（左侧）
    v = ecef_to_eci(np.array([0.0, 700.0, 0.0]), 0.0)
    err = rg._cross_track_error(r, v)
    cmd = rg.command(0.0, r, v, rho=0.1)
    assert err > 0.0
    assert cmd["bank_deg"] < 0.0


def test_reentry_guidance_cross_track_error_north_heading() -> None:
    """正北朝向、目标在正北时横程误差接近 0。"""
    rg = ReentryGuidance(target_lat_deg=40.0, target_lon_deg=116.0)
    lat0 = 39.0
    lon0 = 116.0
    r_ecef = geodetic_to_ecef(lat0, lon0, 80e3)
    r = ecef_to_eci(r_ecef, 0.0)
    omega = np.array([0.0, 0.0, 7.292115e-5], dtype=float)
    v_ecef = enu_to_ecef_vec(np.array([0.0, 1000.0, 0.0]), lat0, lon0) + np.cross(omega, r_ecef)
    v = ecef_to_eci(v_ecef, 0.0)
    err = rg._cross_track_error(r, v)
    assert abs(err) < 1.0


def test_reentry_guidance_cross_track_error_vertical_velocity() -> None:
    """速度垂直时横程误差计算不应抛异常。"""
    rg = ReentryGuidance(target_lat_deg=39.0, target_lon_deg=116.0)
    r_ecef = geodetic_to_ecef(39.0, 115.0, 80e3)
    r = ecef_to_eci(r_ecef, 0.0)
    v = ecef_to_eci(np.array([0.0, 0.0, 500.0]), 0.0)
    err = rg._cross_track_error(r, v)
    assert math.isfinite(err)


def test_reentry_guidance_control_moment_zero_error() -> None:
    """攻角等于制导指令攻角时控制力矩应为零。"""
    rg = ReentryGuidance(moment_gain=1.0e3)
    y = np.zeros(13)
    y[0:3] = np.array([WGS84_A + 50e3, 0.0, 0.0])
    y[3:6] = np.array([0.0, 5000.0, 0.0])
    y[6:10] = np.array([0.0, 0.0, 0.0, 1.0])
    aoa_cmd = rg.command(0.0, y[0:3], y[3:6])["aoa_deg"]
    cm = rg.control_moment(y, alpha_rad=float(np.radians(aoa_cmd)), beta_rad=0.0)
    assert cm.L == pytest.approx(0.0, abs=1e-9)
    assert cm.M == pytest.approx(0.0, abs=1e-9)
    assert cm.N == pytest.approx(0.0, abs=1e-9)


def test_reentry_guidance_control_moment_sign() -> None:
    """control_moment 力矩方向应抵消攻角/侧滑角误差。"""
    rg = ReentryGuidance(moment_gain=1.0e3, nominal_aoa_deg=0.0)
    y = np.zeros(13)
    y[0:3] = np.array([WGS84_A + 50e3, 0.0, 0.0])
    y[3:6] = np.array([0.0, 5000.0, 0.0])
    y[6:10] = np.array([0.0, 0.0, 0.0, 1.0])
    cm = rg.control_moment(y, alpha_rad=0.1, beta_rad=0.05)
    # 正攻角误差产生负俯仰力矩；正侧滑角误差产生负偏航力矩
    assert cm.M < 0.0
    assert cm.N < 0.0


def test_reentry_guidance_control_moment_with_negative_gain() -> None:
    """负力矩增益会反向输出力矩。"""
    rg = ReentryGuidance(moment_gain=-1.0e3, nominal_aoa_deg=0.0)
    y = np.zeros(13)
    y[0:3] = np.array([WGS84_A + 50e3, 0.0, 0.0])
    y[3:6] = np.array([0.0, 5000.0, 0.0])
    y[6:10] = np.array([0.0, 0.0, 0.0, 1.0])
    cm = rg.control_moment(y, alpha_rad=0.1, beta_rad=0.0)
    assert cm.M > 0.0


def test_reentry_guidance_failed_flag_set_on_nan() -> None:
    """异常输入应设置 failed 标志。"""
    rg = ReentryGuidance()
    rg.command(0.0, np.array([np.nan, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), rho=0.1)
    assert rg.failed


def test_reentry_guidance_command_accepts_dyn_ctx() -> None:
    """command 应接受任意的 dyn_ctx 参数而不抛异常。"""
    rg = ReentryGuidance()
    r = np.array([WGS84_A + 50e3, 0.0, 0.0])
    v = np.array([0.0, 1000.0, 0.0])
    cmd = rg.command(0.0, r, v, rho=0.1, dyn_ctx={"foo": "bar"})
    assert "bank_deg" in cmd


def test_reentry_guidance_bank_gain_affects_magnitude() -> None:
    """更大的 bank_gain 在同样能量误差下产生更大的倾侧角。"""
    r = np.array([WGS84_A + 1e3, 0.0, 0.0])
    v = np.array([0.0, 8000.0, 0.0])
    rg1 = ReentryGuidance(target_energy_j_kg=-100e6, bank_gain=1.0e-6)
    rg2 = ReentryGuidance(target_energy_j_kg=-100e6, bank_gain=1.0e-3)
    cmd1 = rg1.command(0.0, r, v, rho=0.1)
    cmd2 = rg2.command(0.0, r, v, rho=0.1)
    assert abs(cmd2["bank_deg"]) >= abs(cmd1["bank_deg"])


def test_reentry_guidance_bank_sign_matches_cross_track() -> None:
    """倾侧角符号应与横程误差符号相反。"""
    rg = ReentryGuidance(target_lat_deg=39.0, target_lon_deg=116.0)
    r_ecef = geodetic_to_ecef(39.0, 117.0, 80e3)
    r = ecef_to_eci(r_ecef, 0.0)
    v = ecef_to_eci(np.array([0.0, 700.0, 0.0]), 0.0)
    err = rg._cross_track_error(r, v)
    cmd = rg.command(0.0, r, v, rho=0.1)
    assert np.sign(cmd["bank_deg"]) == -np.sign(err)


def test_reentry_guidance_command_bank_within_bounds() -> None:
    """command 输出的倾侧角绝对值不应超过 max_bank_deg。"""
    rg = ReentryGuidance(max_bank_deg=30.0, target_energy_j_kg=-100e6, bank_gain=1.0e-3)
    r = np.array([WGS84_A + 1e3, 0.0, 0.0])
    v = np.array([0.0, 8000.0, 0.0])
    cmd = rg.command(0.0, r, v, rho=0.1)
    assert abs(cmd["bank_deg"]) <= 30.0 + 1e-6


def test_reentry_guidance_repr() -> None:
    """ReentryGuidance 数据类 repr 应包含关键字段。"""
    rg = ReentryGuidance(target_lat_deg=10.0, target_lon_deg=20.0)
    text = repr(rg)
    assert "ReentryGuidance" in text
    assert "10.0" in text
    assert "20.0" in text


def test_reentry_guidance_target_energy_override() -> None:
    """显式 target_energy_j_kg 应覆盖默认能量。"""
    rg = ReentryGuidance(target_energy_j_kg=-30e6)
    assert rg._target_energy() == -30e6


def test_reentry_guidance_haversine_target_cross_track_at_equator() -> None:
    """赤道上横程误差计算应给出预期符号。"""
    rg = ReentryGuidance(target_lat_deg=0.0, target_lon_deg=10.0)
    # 位于目标以西，相对速度朝北；目标在东侧（右侧）
    lat0 = 0.0
    lon0 = 9.0
    r_ecef = geodetic_to_ecef(lat0, lon0, 80e3)
    r = ecef_to_eci(r_ecef, 0.0)
    omega = np.array([0.0, 0.0, 7.292115e-5], dtype=float)
    v_ecef = enu_to_ecef_vec(np.array([0.0, 500.0, 0.0]), lat0, lon0) + np.cross(omega, r_ecef)
    v = ecef_to_eci(v_ecef, 0.0)
    err = rg._cross_track_error(r, v)
    # 目标在东（右侧），err > 0，bank_sign = -1
    assert err > 0.0
