"""再入点状态切换专项测试。

验证 3-DOF ECI -> ENU -> 6-DOF 的投影链路，以及再入点非零攻角初始化。
"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.frames import (
    ecef_to_eci,
    enu_to_ecef_vec,
    geodetic_to_ecef,
    vel_ecef_to_eci,
)
from ballistic_sim.state_switch import StateSwitchError, project_state
from ballistic_sim.dynamics.six_dof import _quat_from_vectors, _axis_from_quat, _normalize_quat

RTOL = 1e-9


def _make_eci_state(
    lat_deg: float = 28.5,
    lon_deg: float = -80.0,
    alt_m: float = 100e3,
    v_enu: np.ndarray | None = None,
    mass_kg: float = 1000.0,
) -> np.ndarray:
    """构造 3-DOF ECI 样本状态。"""
    if v_enu is None:
        v_enu = np.array([100.0, 200.0, -300.0])
    r_ecef0 = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    v_ecef = enu_to_ecef_vec(v_enu, lat_deg, lon_deg)
    r_eci = ecef_to_eci(r_ecef0, t=0.0)
    v_eci = vel_ecef_to_eci(r_ecef0, v_ecef, t=0.0)
    return np.array([*r_eci, *v_eci, mass_kg], dtype=float)


def _quat_with_aoa(v_enu: np.ndarray, aoa_deg: float) -> np.ndarray:
    """构造使弹轴相对速度方向偏转 ``aoa_deg`` 的四元数（绕 ENU 天向旋转）。"""
    v = np.asarray(v_enu, dtype=float)
    v_hat = v / (np.linalg.norm(v) + 1e-12)
    # 绕 U 轴旋转 aoa
    axis = np.array([0.0, 0.0, 1.0], dtype=float)
    angle = np.deg2rad(aoa_deg)
    s = np.sin(angle)
    c = np.cos(angle)
    body_hat = (
        c * v_hat
        + s * np.cross(axis, v_hat)
        + (1.0 - c) * np.dot(axis, v_hat) * axis
    )
    q = _quat_from_vectors(np.array([1.0, 0.0, 0.0]), body_hat)
    return _normalize_quat(q)


def test_eci_to_enu_to_6dof_preserves_kinematics() -> None:
    """ECI 3-DOF -> ENU 6-DOF 后，位置/速度往返误差可忽略。"""
    y7 = _make_eci_state()
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    omega = np.array([0.0, 0.05, 0.0])

    y13 = project_state(
        y_src=y7,
        src_dim=7,
        dst_dim=13,
        src_frame="ECI",
        dst_frame="ENU",
        lat_deg=28.5,
        lon_deg=-80.0,
        h0=100e3,
        t=0.0,
        quat=quat,
        omega=omega,
    )

    assert y13.size == 13
    # ENU -> ECEF -> ECI 往返
    r_enu = y13[0:3]
    v_enu = y13[3:6]
    r_ecef = geodetic_to_ecef(28.5, -80.0, 100e3) + enu_to_ecef_vec(r_enu, 28.5, -80.0)
    v_ecef = enu_to_ecef_vec(v_enu, 28.5, -80.0)
    r_eci_back = ecef_to_eci(r_ecef, t=0.0)
    v_eci_back = vel_ecef_to_eci(r_ecef, v_ecef, t=0.0)

    assert np.allclose(r_eci_back, y7[0:3], rtol=RTOL)
    assert np.allclose(v_eci_back, y7[3:6], rtol=RTOL)


def test_eci_to_6dof_with_nonzero_aoa() -> None:
    """再入点非零攻角初始化：弹轴与速度方向偏差等于设定的攻角。"""
    v_enu = np.array([1000.0, 200.0, -300.0])
    y7 = _make_eci_state(v_enu=v_enu)
    aoa_deg = 5.0
    quat = _quat_with_aoa(v_enu, aoa_deg)
    omega = np.array([0.0, 0.0, 0.0])

    y13 = project_state(
        y_src=y7,
        src_dim=7,
        dst_dim=13,
        src_frame="ECI",
        dst_frame="ENU",
        lat_deg=28.5,
        lon_deg=-80.0,
        h0=100e3,
        t=0.0,
        quat=quat,
        omega=omega,
    )

    body_axis = _axis_from_quat(_normalize_quat(y13[6:10]))
    vel_dir = y13[3:6] / np.linalg.norm(y13[3:6])
    cos_angle = np.clip(np.dot(body_axis, vel_dir), -1.0, 1.0)
    actual_aoa = np.degrees(np.arccos(cos_angle))
    assert actual_aoa == pytest.approx(aoa_deg, abs=0.5)
    assert np.allclose(y13[10:13], omega, rtol=RTOL)


def test_eci_to_6dof_requires_explicit_quat() -> None:
    """未提供 quat 时禁止 ECI -> 6-DOF 自动升维。"""
    y7 = _make_eci_state()
    with pytest.raises(StateSwitchError, match="禁止"):
        project_state(
            y_src=y7,
            src_dim=7,
            dst_dim=13,
            src_frame="ECI",
            dst_frame="ENU",
            lat_deg=28.5,
            lon_deg=-80.0,
            h0=100e3,
            t=0.0,
        )


def test_eci_to_6dof_auto_upgrade_with_warning() -> None:
    """allow_auto=True 时沿速度方向构造姿态并发出警告。"""
    y7 = _make_eci_state()
    with pytest.warns(UserWarning, match="姿态"):
        y13 = project_state(
            y_src=y7,
            src_dim=7,
            dst_dim=13,
            src_frame="ECI",
            dst_frame="ENU",
            lat_deg=28.5,
            lon_deg=-80.0,
            h0=100e3,
            t=0.0,
            allow_auto=True,
        )
    assert y13.size == 13
    body_axis = _axis_from_quat(_normalize_quat(y13[6:10]))
    vel_dir = y13[3:6] / np.linalg.norm(y13[3:6])
    assert np.allclose(body_axis, vel_dir, atol=1e-12)


def test_eci_to_enu_same_dim() -> None:
    """同维 ECI -> ENU 切换应恢复原始 ENU 速度。"""
    v_enu = np.array([100.0, 200.0, -300.0])
    y7 = _make_eci_state(v_enu=v_enu)
    y7_enu = project_state(
        y_src=y7,
        src_dim=7,
        dst_dim=7,
        src_frame="ECI",
        dst_frame="ENU",
        lat_deg=28.5,
        lon_deg=-80.0,
        h0=100e3,
        t=0.0,
    )
    assert y7_enu.size == 7
    assert np.allclose(y7_enu[3:6], v_enu, rtol=1e-6)


def test_6dof_to_3dof_requires_mass() -> None:
    """6-DOF -> 3-DOF 降维必须显式提供质量。"""
    y13 = np.zeros(13, dtype=float)
    with pytest.raises(StateSwitchError, match="mass_kg"):
        project_state(
            y_src=y13,
            src_dim=13,
            dst_dim=7,
            src_frame="ENU",
            dst_frame="ENU",
        )


def test_6dof_to_3dof_with_mass() -> None:
    """6-DOF -> 3-DOF 降维后质量等于给定值。"""
    y13 = np.zeros(13, dtype=float)
    y7 = project_state(
        y_src=y13,
        src_dim=13,
        dst_dim=7,
        src_frame="ENU",
        dst_frame="ENU",
        mass_kg=123.0,
    )
    assert y7[6] == pytest.approx(123.0)


def test_enu_to_6dof_explicit_quat() -> None:
    """ENU 3-DOF -> 6-DOF 显式传入 quat/omega。"""
    y7 = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 800.0])
    quat = np.array([0.0, 1.0, 0.0, 0.0])
    omega = np.array([0.1, 0.2, 0.3])
    y13 = project_state(
        y_src=y7,
        src_dim=7,
        dst_dim=13,
        src_frame="ENU",
        dst_frame="ENU",
        quat=quat,
        omega=omega,
    )
    assert np.allclose(y13[0:6], y7[0:6], rtol=RTOL)
    assert np.allclose(y13[10:13], omega, rtol=RTOL)


def test_unsupported_switch_raises() -> None:
    """未实现的维度/坐标系切换应抛出 StateSwitchError。"""
    y = np.zeros(8, dtype=float)
    with pytest.raises(StateSwitchError):
        project_state(
            y_src=y,
            src_dim=8,
            dst_dim=13,
            src_frame="ENU",
            dst_frame="ECI",
        )
