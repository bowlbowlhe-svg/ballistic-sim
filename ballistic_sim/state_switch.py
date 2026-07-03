"""阶段间状态投影 (fidelity 升降维与坐标系转换)。"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from ballistic_sim.constants import OMEGA_EARTH
from ballistic_sim.frames import (
    ecef_to_eci,
    ecef_to_enu_vec,
    eci_to_ecef,
    enu_to_ecef_vec,
    vel_ecef_to_eci,
    vel_eci_to_ecef,
)

logger = logging.getLogger(__name__)


class StateSwitchError(NotImplementedError):
    """禁止的状态切换方向。"""

    pass


def _roundtrip_check(
    r_src: np.ndarray,
    v_src: np.ndarray,
    r_back: np.ndarray,
    v_back: np.ndarray,
    speed_tol: float = 1e-6,
) -> Tuple[float, float]:
    pos_err = float(np.linalg.norm(r_back - r_src) / max(np.linalg.norm(r_src), 1e-6))
    vel_err = float(np.linalg.norm(v_back - v_src) / max(np.linalg.norm(v_src), 1e-6))
    if pos_err > 1e-9 or vel_err > 1e-9:
        logger.warning(
            "状态切换往返一致性误差: pos_err=%.3e, vel_err=%.3e",
            pos_err,
            vel_err,
        )
    speed_src = float(np.linalg.norm(v_src))
    speed_back = float(np.linalg.norm(v_back))
    if abs(speed_back - speed_src) > speed_tol:
        logger.warning(
            "升维后速度模长跳变: src=%.3f m/s, back=%.3f m/s",
            speed_src,
            speed_back,
        )
    return pos_err, vel_err


def _enu_to_eci(
    r_enu: np.ndarray,
    v_enu: np.ndarray,
    lat_deg: float,
    lon_deg: float,
    h0: float,
    t: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    from ballistic_sim.frames import geodetic_to_ecef

    r_ecef0 = geodetic_to_ecef(lat_deg, lon_deg, h0)
    r_ecef = r_ecef0 + enu_to_ecef_vec(r_enu, lat_deg, lon_deg)
    v_ecef = enu_to_ecef_vec(v_enu, lat_deg, lon_deg)
    r_eci = ecef_to_eci(r_ecef, t)
    v_eci = vel_ecef_to_eci(r_ecef, v_ecef, t)
    return r_eci, v_eci


def _eci_to_enu(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
    lat_deg: float,
    lon_deg: float,
    h0: float,
    t: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    from ballistic_sim.frames import geodetic_to_ecef

    r_ecef = eci_to_ecef(r_eci, t)
    v_ecef = vel_eci_to_ecef(r_eci, v_eci, t)
    r_ecef0 = geodetic_to_ecef(lat_deg, lon_deg, h0)
    r_enu = ecef_to_enu_vec(r_ecef - r_ecef0, lat_deg, lon_deg)
    v_enu = ecef_to_enu_vec(v_ecef, lat_deg, lon_deg)
    return r_enu, v_enu


def project_state(
    y_src: np.ndarray,
    src_dim: int,
    dst_dim: int,
    src_frame: str,
    dst_frame: str,
    lat_deg: float = 0.0,
    lon_deg: float = 0.0,
    h0: float = 0.0,
    spin_rate: float = 0.0,
    t: float = 0.0,
) -> np.ndarray:
    """将源状态投影到目标状态。

    支持 §6.4 允许的切换方向；禁止方向抛出 ``NotImplementedError``。
    """
    y_src = np.asarray(y_src, dtype=float)

    # 同维换系
    if src_dim == dst_dim:
        if src_frame == dst_frame:
            return y_src.copy()
        if src_frame == "ECI" and dst_frame == "ENU":
            r_enu, v_enu = _eci_to_enu(y_src[0:3], y_src[3:6], lat_deg, lon_deg, h0, t)
            y_dst = y_src.copy()
            y_dst[0:3] = r_enu
            y_dst[3:6] = v_enu
            _roundtrip_check(
                y_src[0:3],
                y_src[3:6],
                *_enu_to_eci(r_enu, v_enu, lat_deg, lon_deg, h0, t),
            )
            return y_dst
        if src_frame == "ENU" and dst_frame == "ECI":
            r_eci, v_eci = _enu_to_eci(y_src[0:3], y_src[3:6], lat_deg, lon_deg, h0, t)
            y_dst = y_src.copy()
            y_dst[0:3] = r_eci
            y_dst[3:6] = v_eci
            _roundtrip_check(
                y_src[0:3],
                y_src[3:6],
                *_eci_to_enu(r_eci, v_eci, lat_deg, lon_deg, h0, t),
            )
            return y_dst
        raise StateSwitchError(f"未支持的同维换系: {src_frame} -> {dst_frame}")

    # 3-DOF ENU -> MPM (升维)
    if src_dim == 7 and dst_dim == 8 and src_frame == "ENU" and dst_frame == "ENU":
        y_dst = np.zeros(8, dtype=float)
        y_dst[0:6] = y_src[0:6]
        y_dst[6] = spin_rate
        y_dst[7] = 0.0
        _roundtrip_check(y_src[0:3], y_src[3:6], y_dst[0:3], y_dst[3:6])
        return y_dst

    # 3-DOF ECI -> MPM (跨域升维)
    if src_dim == 7 and dst_dim == 8 and src_frame == "ECI" and dst_frame == "ENU":
        r_enu, v_enu = _eci_to_enu(y_src[0:3], y_src[3:6], lat_deg, lon_deg, h0, t)
        y_dst = np.zeros(8, dtype=float)
        y_dst[0:6] = np.concatenate([r_enu, v_enu])
        y_dst[6] = spin_rate
        y_dst[7] = 0.0
        return y_dst

    # 6-DOF -> 3-DOF (降维)
    if src_dim == 13 and dst_dim == 7:
        if src_frame == "ENU" and dst_frame == "ECI":
            raise StateSwitchError(
                "禁止 6-DOF ENU -> 3-DOF ECI：缺乏全局位置信息定义 ECI 坐标"
            )
        y_dst = y_src[:7].copy()
        return y_dst

    # 6-DOF -> MPM (降维)
    if src_dim == 13 and dst_dim == 8 and dst_frame == "ENU":
        y_dst = np.zeros(8, dtype=float)
        y_dst[0:6] = y_src[0:6]
        y_dst[6] = float(y_src[12])  # p
        y_dst[7] = 0.0
        return y_dst

    # 3-DOF ECI -> 3-DOF ENU (同维换系已在上面处理)
    # MPM/6-DOF ENU -> 3-DOF ECI 禁止
    if src_frame == "ENU" and dst_frame == "ECI":
        raise StateSwitchError("禁止 MPM/6-DOF ENU -> 3-DOF ECI：缺乏全局位置信息定义 ECI 坐标")

    # 3-DOF -> 6-DOF 自动升维禁止
    if src_dim == 7 and dst_dim == 13:
        raise StateSwitchError("禁止 3-DOF -> 6-DOF 自动升维：需用户显式提供初始姿态")

    raise StateSwitchError(
        f"未实现的状态切换: dim {src_dim}({src_frame}) -> {dst_dim}({dst_frame})"
    )
