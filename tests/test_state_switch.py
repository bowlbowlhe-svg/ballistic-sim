"""状态切换与保真度升降维单元测试。

本模块验证 ``ballistic_sim.state_switch.project_state`` 的契约：
- 允许/禁止的切换方向；
- 同维换系往返一致性；
- 升降维后 r, v, m 不变、速度模长不变；
- 新增自由度（MPM 的 spin/alpha，6-DOF 降维后的 spin）。
"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.constants import WGS84_A
from ballistic_sim.frames import ecef_to_enu_vec, enu_to_ecef_vec, geodetic_to_ecef
from ballistic_sim.state_switch import StateSwitchError, project_state

RTOL = 1e-9


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """相对误差，避免零除。"""
    norm = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-30)
    return float(np.linalg.norm(a - b) / norm)


def _make_3dof_eci(lat_deg: float = 39.9, lon_deg: float = 90.0, h0: float = 100.0) -> np.ndarray:
    """构造一个 3-DOF ECI 样本状态 [r(3), v(3), m]。"""
    r_ecef = geodetic_to_ecef(lat_deg, lon_deg, h0)
    # 局部 ENU 速度 100 m/s 向北、300 m/s 向上
    v_enu = np.array([0.0, 100.0, 300.0])
    v_ecef = enu_to_ecef_vec(v_enu, lat_deg, lon_deg)
    from ballistic_sim.frames import ecef_to_eci, vel_ecef_to_eci

    r_eci = ecef_to_eci(r_ecef, t=0.0)
    v_eci = vel_ecef_to_eci(r_ecef, v_ecef, t=0.0)
    return np.array([*r_eci, *v_eci, 43.0], dtype=float)


def _make_3dof_enu(lat_deg: float = 39.9) -> np.ndarray:
    """构造一个 3-DOF ENU 样本状态 [E, N, U, vE, vN, vU, m]。"""
    return np.array([100.0, 200.0, 150.0, 400.0, 100.0, 200.0, 15.0], dtype=float)


def _make_6dof() -> np.ndarray:
    """构造一个 6-DOF 样本状态 [r(3), v(3), q(4), p, q?, r?]。

    ``project_state`` 仅关心前 6 个分量与第 13 个角速度分量（spin），
    中间姿态四元数与横向角速度在本次升降维测试中被裁剪，因此占位即可。
    """
    y = np.zeros(13, dtype=float)
    y[0:3] = geodetic_to_ecef(39.9, 0.0, 100.0)
    y[3:6] = np.array([300.0, 400.0, 500.0])
    y[6:10] = np.array([1.0, 0.0, 0.0, 0.0])  # 单位四元数占位
    y[12] = 7.5  # spin (p)
    return y


# ==============================================================================
# 允许方向
# ==============================================================================


def test_allowed_6dof_to_3dof_eci() -> None:
    """6-DOF -> 3-DOF ECI 允许，且 r, v, m 不变。"""
    y6 = _make_6dof()
    y3 = project_state(
        y_src=y6,
        src_dim=13,
        dst_dim=7,
        src_frame="ECI",
        dst_frame="ECI",
    )
    assert y3.size == 7
    assert np.allclose(y3[0:3], y6[0:3], rtol=RTOL)
    assert np.allclose(y3[3:6], y6[3:6], rtol=RTOL)
    assert y3[6] == y6[6]


def test_allowed_3dof_eci_to_3dof_enu() -> None:
    """3-DOF ECI -> 3-DOF ENU 允许。"""
    y_eci = _make_3dof_eci()
    y_enu = project_state(
        y_src=y_eci,
        src_dim=7,
        dst_dim=7,
        src_frame="ECI",
        dst_frame="ENU",
        lat_deg=39.9,
        lon_deg=90.0,
        h0=100.0,
    )
    assert y_enu.size == 7
    assert y_enu[6] == y_eci[6]


def test_allowed_3dof_enu_to_mpm() -> None:
    """3-DOF ENU -> MPM 允许，并补充 spin/alpha。"""
    y3 = _make_3dof_enu()
    y8 = project_state(
        y_src=y3,
        src_dim=7,
        dst_dim=8,
        src_frame="ENU",
        dst_frame="ENU",
        spin_rate=2500.0,
    )
    assert y8.size == 8
    assert np.allclose(y8[0:6], y3[0:6], rtol=RTOL)
    assert y8[6] == 2500.0
    assert y8[7] == 0.0


def test_allowed_3dof_eci_to_mpm() -> None:
    """3-DOF ECI -> MPM (跨域升维) 允许。"""
    y_eci = _make_3dof_eci()
    y_mpm = project_state(
        y_src=y_eci,
        src_dim=7,
        dst_dim=8,
        src_frame="ECI",
        dst_frame="ENU",
        lat_deg=39.9,
        lon_deg=90.0,
        h0=100.0,
        spin_rate=1800.0,
    )
    assert y_mpm.size == 8
    assert y_mpm[6] == 1800.0
    assert y_mpm[7] == 0.0


def test_allowed_6dof_to_mpm() -> None:
    """6-DOF -> MPM 允许，spin 来自 6-DOF 角速度。"""
    y6 = _make_6dof()
    y6[12] = 9.5
    y_mpm = project_state(
        y_src=y6,
        src_dim=13,
        dst_dim=8,
        src_frame="ENU",
        dst_frame="ENU",
    )
    assert y_mpm.size == 8
    assert np.allclose(y_mpm[0:6], y6[0:6], rtol=RTOL)
    assert y_mpm[6] == 9.5
    assert y_mpm[7] == 0.0


# ==============================================================================
# 禁止方向
# ==============================================================================


def test_forbidden_mpm_to_3dof_eci() -> None:
    """MPM/6-DOF ENU -> 3-DOF ECI 禁止。"""
    y_mpm = np.array([100.0, 200.0, 50.0, 400.0, 0.0, 0.0, 2500.0, 0.0])
    with pytest.raises(StateSwitchError, match="禁止"):
        project_state(
            y_src=y_mpm,
            src_dim=8,
            dst_dim=7,
            src_frame="ENU",
            dst_frame="ECI",
        )


def test_forbidden_6dof_enu_to_3dof_eci() -> None:
    """6-DOF ENU -> 3-DOF ECI 禁止。"""
    y6 = _make_6dof()
    with pytest.raises(StateSwitchError, match="禁止"):
        project_state(
            y_src=y6,
            src_dim=13,
            dst_dim=7,
            src_frame="ENU",
            dst_frame="ECI",
        )


def test_forbidden_3dof_to_6dof_by_default() -> None:
    """3-DOF -> 6-DOF 自动升维禁止。"""
    y3 = _make_3dof_eci()
    with pytest.raises(StateSwitchError, match="禁止"):
        project_state(
            y_src=y3,
            src_dim=7,
            dst_dim=13,
            src_frame="ECI",
            dst_frame="ECI",
        )


# ==============================================================================
# 守恒量检查
# ==============================================================================


def test_eci_enu_eci_round_trip() -> None:
    """ECI -> ENU -> ECI 往返一致性：位置/速度相对误差 < 1e-9。"""
    y_eci = _make_3dof_eci()
    y_enu = project_state(
        y_src=y_eci,
        src_dim=7,
        dst_dim=7,
        src_frame="ECI",
        dst_frame="ENU",
        lat_deg=39.9,
        lon_deg=90.0,
        h0=100.0,
    )
    y_back = project_state(
        y_src=y_enu,
        src_dim=7,
        dst_dim=7,
        src_frame="ENU",
        dst_frame="ECI",
        lat_deg=39.9,
        lon_deg=90.0,
        h0=100.0,
    )
    assert _relative_error(y_back[0:3], y_eci[0:3]) < RTOL
    assert _relative_error(y_back[3:6], y_eci[3:6]) < RTOL
    assert y_back[6] == y_eci[6]


def test_dimension_change_preserves_rvm() -> None:
    """6-DOF -> 3-DOF 后 r, v, m 不变、速度模长不变。"""
    y6 = _make_6dof()
    y3 = project_state(
        y_src=y6,
        src_dim=13,
        dst_dim=7,
        src_frame="ECI",
        dst_frame="ECI",
    )
    assert np.allclose(y3[0:3], y6[0:3], rtol=RTOL)
    assert np.allclose(y3[3:6], y6[3:6], rtol=RTOL)
    assert y3[6] == y6[6]
    assert np.isclose(np.linalg.norm(y3[3:6]), np.linalg.norm(y6[3:6]), rtol=RTOL)


def test_3dof_enu_to_mpm_preserves_speed() -> None:
    """3-DOF ENU -> MPM 后速度模长不变。"""
    y3 = _make_3dof_enu()
    y_mpm = project_state(
        y_src=y3,
        src_dim=7,
        dst_dim=8,
        src_frame="ENU",
        dst_frame="ENU",
        spin_rate=0.0,
    )
    assert np.isclose(np.linalg.norm(y_mpm[3:6]), np.linalg.norm(y3[3:6]), rtol=RTOL)


# ==============================================================================
# 新增自由度检查
# ==============================================================================


def test_mpm_adds_spin_and_alpha() -> None:
    """3-DOF -> MPM 后新增 spin 与 alpha。"""
    y3 = _make_3dof_enu()
    y_mpm = project_state(
        y_src=y3,
        src_dim=7,
        dst_dim=8,
        src_frame="ENU",
        dst_frame="ENU",
        spin_rate=3000.0,
    )
    assert y_mpm.size == 8
    assert y_mpm[6] == 3000.0
    assert y_mpm[7] == 0.0


def test_6dof_spin_carried_to_mpm() -> None:
    """6-DOF 降维到 MPM 时 spin 自由度被保留。"""
    y6 = _make_6dof()
    y6[12] = 12.5
    y_mpm = project_state(
        y_src=y6,
        src_dim=13,
        dst_dim=8,
        src_frame="ENU",
        dst_frame="ENU",
    )
    assert y_mpm[6] == 12.5


# ==============================================================================
# 真实动力学模型字段一致性
# ==============================================================================


def test_real_mpm_state_has_spin_alpha() -> None:
    """MPMDynamics 真实状态向量包含 spin 与 alpha 自由度。"""
    from ballistic_sim.dynamics import MPMDynamics

    dyn = MPMDynamics(mass_kg=43.0, diameter_m=0.155)
    assert dyn.state_dim() == 8
    y0 = dyn.initial_state(v0=684.0, theta_deg=45.0, az_deg=90.0, h0=0.0)
    assert y0.size == 8
    assert y0[6] != 0.0  # spin
    assert y0[7] == 0.0  # alpha 默认 0


def test_real_point_mass_frames() -> None:
    """PointMassDynamics 支持 ECI 与 ENU 两种原生坐标系。"""
    from ballistic_sim.dynamics import PointMassDynamics

    eci = PointMassDynamics(frame="ECI")
    enu = PointMassDynamics(frame="ENU")
    assert eci.frame == "ECI"
    assert enu.frame == "ENU"
