"""比例导引制导律单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.constants import WGS84_A
from ballistic_sim.guidance.proportional_navigation import (
    ProNavGuidance,
    make_static_target_provider,
    pro_nav_acceleration,
)


def _simple_pn_intercept(
    r0: np.ndarray,
    v0: np.ndarray,
    target_provider,
    nav_constant: float = 3.0,
    max_accel: float = 100.0,
    dt: float = 0.02,
    t_max: float = 120.0,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """简单点质量积分：仅受比例导引加速度（局部惯性系）。"""
    r = np.asarray(r0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    pn = ProNavGuidance(nav_constant=nav_constant, max_accel_m_s2=max_accel)
    pn.target_provider = target_provider
    t = 0.0
    min_miss = float("inf")
    while t < t_max:
        r_t, v_t = target_provider(t)
        miss_now = float(np.linalg.norm(r_t - r))
        if miss_now < min_miss:
            min_miss = miss_now
        a_cmd = pn.acceleration(r, v, r_t, v_t)
        # 二阶 Runge-Kutta（中点法）
        v_half = v + 0.5 * a_cmd * dt
        r_t_half, v_t_half = target_provider(t + 0.5 * dt)
        a_half = pn.acceleration(r + 0.5 * v * dt, v_half, r_t_half, v_t_half)
        v = v + a_half * dt
        r = r + v_half * dt
        t += dt
        if min_miss < 1.0:
            break
    return t, r, v, min_miss


def test_pro_nav_static_target_miss_distance() -> None:
    """对静态目标的末端比例导引脱靶量应 < 100 m。"""
    # 局部惯性系：目标在 x 轴 8 km 处，导弹初速带 y 向偏移
    r0 = np.array([0.0, 0.0, 0.0])

    def provider(t: float) -> tuple[np.ndarray, np.ndarray]:
        return np.array([8000.0, 0.0, 0.0]), np.zeros(3)

    v0 = np.array([600.0, 80.0, 0.0])

    _, _, _, miss = _simple_pn_intercept(
        r0, v0, provider, nav_constant=3.5, max_accel=150.0, t_max=60.0
    )
    assert miss < 100.0


def test_pro_nav_constant_velocity_target() -> None:
    """对匀速运动目标的拦截脱靶量应 < 100 m。"""
    r0 = np.array([0.0, 0.0, 0.0])
    # 目标以 100 m/s 沿 y 方向运动，导弹朝 x 方向发射
    target_v = np.array([0.0, 100.0, 0.0])

    def provider(t: float) -> tuple[np.ndarray, np.ndarray]:
        return np.array([6000.0, 100.0 * t, 0.0]), target_v

    v0 = np.array([700.0, 20.0, 0.0])

    _, _, _, miss = _simple_pn_intercept(
        r0, v0, provider, nav_constant=3.5, max_accel=200.0, t_max=80.0
    )
    assert miss < 100.0


def test_pro_nav_acceleration_perpendicular_to_los() -> None:
    """真比例导引加速度应垂直于视线。"""
    r_rel = np.array([1000.0, 0.0, 0.0])
    v_rel = np.array([-100.0, 50.0, 0.0])
    a = pro_nav_acceleration(r_rel, v_rel, nav_constant=3.0, mode="true")
    assert abs(np.dot(a, r_rel)) < 1.0


def test_pro_nav_max_accel_clipping() -> None:
    """最大加速度裁剪应生效。"""
    pn = ProNavGuidance(nav_constant=5.0, max_accel_m_s2=10.0)
    pn.set_target(0.0, 0.001, 0.0)
    r_own = np.array([WGS84_A, 0.0, 0.0])
    v_own = np.array([0.0, 100.0, 0.0])
    a = pn.acceleration(r_own, v_own)
    assert np.linalg.norm(a) <= 10.0 + 1e-6


def test_pro_nav_set_methods_chain() -> None:
    """链式 setter 应返回 self。"""
    pn = ProNavGuidance()
    assert pn.set_target(0.0, 0.0) is pn
    assert pn.set_max_accel(50.0) is pn
    assert pn.set_mode("generalized") is pn
    assert pn.set_nav_constant(4.0) is pn
    assert pn.mode == "generalized"
    assert pn.nav_constant == 4.0


def test_pro_nav_generalized_mode() -> None:
    """广义比例导引应返回垂直于速度平面的加速度分量。"""
    r_rel = np.array([1000.0, 0.0, 0.0])
    v_rel = np.array([-100.0, 50.0, 0.0])
    a = pro_nav_acceleration(r_rel, v_rel, nav_constant=3.0, mode="generalized")
    v_mag = np.linalg.norm(v_rel)
    projection = np.dot(a, v_rel) / (v_mag * v_mag) * v_rel
    assert np.allclose(projection, 0.0, atol=1e-6)


def test_pro_nav_invalid_mode_raises() -> None:
    """非法模式应抛出 ValueError。"""
    with pytest.raises(ValueError, match="未知的比例导引模式"):
        pro_nav_acceleration(np.ones(3), np.ones(3), mode="unknown")

    pn = ProNavGuidance()
    with pytest.raises(ValueError, match="模式必须是 true 或 generalized"):
        pn.set_mode("invalid")


def test_pro_nav_zero_distance_returns_zero() -> None:
    """相对距离接近零时应返回零加速度。"""
    a = pro_nav_acceleration(np.zeros(3), np.array([100.0, 0.0, 0.0]))
    assert np.allclose(a, 0.0)


def test_pro_nav_nonfinite_returns_zero() -> None:
    """非有限结果应返回零加速度。"""
    r_rel = np.array([np.inf, 0.0, 0.0])
    v_rel = np.array([-100.0, 0.0, 0.0])
    a = pro_nav_acceleration(r_rel, v_rel)
    assert np.allclose(a, 0.0)


def test_pro_nav_direction_without_target_fails() -> None:
    """未设置目标时 direction 应失败并返回零矢量。"""
    pn = ProNavGuidance()
    direction = pn.direction(0.0, np.zeros(3), np.array([100.0, 0.0, 0.0]))
    assert pn.failed
    assert np.allclose(direction, 0.0)


def test_pro_nav_direction_with_target() -> None:
    """设置目标后 direction 应返回单位推力方向。"""
    pn = ProNavGuidance()
    pn.set_target(0.0, 0.001, 0.0)
    direction = pn.direction(
        0.0,
        np.array([0.0, 0.0, 0.0]),
        np.array([100.0, 0.0, 0.0]),
    )
    assert np.isclose(np.linalg.norm(direction), 1.0)


def test_pro_nav_acceleration_without_target_fails() -> None:
    """未设置目标时 acceleration 应设置 failed 标志。"""
    pn = ProNavGuidance()
    a = pn.acceleration(np.zeros(3), np.array([100.0, 0.0, 0.0]))
    assert pn.failed
    assert np.allclose(a, 0.0)


def test_pro_nav_max_accel_zero() -> None:
    """最大加速度为 0 时指令应为零。"""
    pn = ProNavGuidance(nav_constant=5.0, max_accel_m_s2=0.0)
    pn.set_target(0.0, 0.001, 0.0)
    a = pn.acceleration(np.array([0.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0]))
    assert np.allclose(a, 0.0)


def test_make_static_target_provider_returns_eci_state() -> None:
    """静态目标提供者应返回随地球自转的 ECI 位置及零速度。"""
    provider = make_static_target_provider(0.0, 0.0, 0.0)
    r0, v0 = provider(0.0)
    r1, v1 = provider(100.0)
    assert r0.shape == (3,)
    assert v0.shape == (3,)
    assert np.allclose(v0, 0.0)
    assert np.allclose(v1, 0.0)
    # 不同历元同一地面点的 ECI 位置不同
    assert not np.allclose(r0, r1)


def test_pro_nav_generalized_zero_velocity_falls_back() -> None:
    """广义模式下弹目相对速度为零时退化为 TPN 形式。"""
    r_rel = np.array([1000.0, 0.0, 0.0])
    v_rel = np.zeros(3)
    a_gpn = pro_nav_acceleration(r_rel, v_rel, nav_constant=3.0, mode="generalized")
    a_tpn = pro_nav_acceleration(r_rel, v_rel, nav_constant=3.0, mode="true")
    assert np.allclose(a_gpn, a_tpn)


def test_pro_nav_zero_los_rate_returns_zero() -> None:
    """视线角速度为零时（相对速度平行于视线）指令应为零。"""
    r_rel = np.array([1000.0, 0.0, 0.0])
    v_rel = np.array([-100.0, 0.0, 0.0])
    a = pro_nav_acceleration(r_rel, v_rel, nav_constant=3.0, mode="true")
    assert np.allclose(a, 0.0, atol=1e-9)


def test_pro_nav_acceleration_bad_input_raises() -> None:
    """异常形状输入应被 reshape 报错或产生可预期结果。"""
    # 长度不足 3 会触发 reshape 错误
    with pytest.raises(ValueError):
        pro_nav_acceleration(np.array([1.0, 0.0]), np.array([-1.0, 0.0, 0.0]))


def test_pro_nav_guidance_uses_target_provider() -> None:
    """ProNavGuidance 通过 target_provider 计算指令。"""
    provider = make_static_target_provider(0.0, 0.001, 0.0)
    pn = ProNavGuidance(nav_constant=4.0, max_accel_m_s2=50.0)
    pn.target_provider = provider
    r_own = np.array([0.0, 0.0, 0.0])
    v_own = np.array([100.0, 0.0, 0.0])
    a = pn.acceleration(r_own, v_own)
    assert np.isfinite(a).all()
    assert np.linalg.norm(a) <= 50.0 + 1e-6


def test_pro_nav_direction_zero_accel_sets_failed() -> None:
    """direction 中加速度被裁剪为零时应设置 failed。"""
    pn = ProNavGuidance(nav_constant=0.0, max_accel_m_s2=0.0)
    pn.set_target(0.0, 0.001, 0.0)
    direction = pn.direction(0.0, np.zeros(3), np.array([100.0, 0.0, 0.0]))
    assert pn.failed
    assert np.allclose(direction, 0.0)
