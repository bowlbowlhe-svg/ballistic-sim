"""可视化工具包。

所有绘图函数约定为纯函数: 输入 ``SimResult``, 返回 ``matplotlib.figure.Figure``。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from ballistic_sim.frames import ecef_to_geodetic, eci_to_ecef
from ballistic_sim.simulator import SimResult


def detect_frame(result: SimResult) -> str:
    """根据状态向量维度推断坐标系。"""
    if result.y.ndim < 2:
        return "UNKNOWN"
    ncol = result.y.shape[1]
    if ncol == 8:
        return "ENU"
    if ncol == 7:
        return "ECI"
    return "UNKNOWN"


def enu_coords(result: SimResult) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 ENU 坐标 (E, N, U); 支持原生 ENU 或由 ECI 转换。"""
    frame = detect_frame(result)
    if frame == "ENU":
        return result.y[:, 0].copy(), result.y[:, 1].copy(), result.y[:, 2].copy()
    if frame == "ECI":
        e, n, u = _eci_to_enu_series(result)
        return e, n, u
    raise ValueError(f"不支持的状态维度: {result.y.shape}")


def _eci_to_enu_series(result: SimResult) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from ballistic_sim.frames import ecef_to_enu_vec

    r_ecef = np.array([eci_to_ecef(result.y[i, 0:3], result.t[i]) for i in range(len(result.t))])
    # 使用发射点作为 ENU 原点
    lat0 = getattr(result, "_lat0_deg", 0.0)
    lon0 = getattr(result, "_lon0_deg", 0.0)
    # 若 result 未携带, 由首点 ECEF 反推
    if lat0 == 0.0 and lon0 == 0.0:
        lat0, lon0, _ = ecef_to_geodetic(r_ecef[0])
    enu = np.array(
        [ecef_to_enu_vec(r_ecef[i] - r_ecef[0], lat0, lon0) for i in range(len(result.t))]
    )
    return enu[:, 0], enu[:, 1], enu[:, 2]


def geodetic_coords(result: SimResult) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回大地坐标序列 (lat_deg, lon_deg, alt_m)。"""
    frame = detect_frame(result)
    if frame == "ENU":
        lat0 = getattr(result, "_lat0_deg", 0.0)
        lon0 = getattr(result, "_lon0_deg", 0.0)
        e, n, u = result.y[:, 0], result.y[:, 1], result.y[:, 2]
        # 小范围近似: 1 deg lat ≈ 111 km
        lat = lat0 + np.degrees(n / 6371000.0)
        lon = lon0 + np.degrees(e / (6371000.0 * np.cos(np.radians(lat0))))
        return lat, lon, u
    if frame == "ECI":
        lla = np.array(
            [
                ecef_to_geodetic(eci_to_ecef(result.y[i, 0:3], result.t[i]))
                for i in range(len(result.t))
            ]
        )
        return lla[:, 0], lla[:, 1], lla[:, 2]
    raise ValueError(f"不支持的状态维度: {result.y.shape}")


def downrange(result: SimResult) -> np.ndarray:
    """返回地面射程 (m)。"""
    e, n, _ = enu_coords(result)
    return np.hypot(e, n)


def altitude(result: SimResult) -> np.ndarray:
    """返回几何高度 (m)。"""
    _, _, u = enu_coords(result)
    return u


def speed(result: SimResult) -> np.ndarray:
    """返回速度大小 (m/s)。"""
    frame = detect_frame(result)
    if frame == "ENU":
        return np.linalg.norm(result.y[:, 3:6], axis=1)
    if frame == "ECI":
        return np.linalg.norm(result.y[:, 3:6], axis=1)
    raise ValueError(f"不支持的状态维度: {result.y.shape}")


def mass(result: SimResult) -> np.ndarray:
    """返回质量序列 (kg); 无质量维则返回常数。"""
    frame = detect_frame(result)
    if frame == "ECI":
        return result.y[:, 6].copy()
    return np.full(len(result.t), np.nan)


def attach_launch_lla(
    result: SimResult, lat_deg: float, lon_deg: float, alt_m: float = 0.0
) -> SimResult:
    """在 result 上附加发射点信息 (供 ENU 转大地坐标使用)。"""
    result._lat0_deg = float(lat_deg)  # type: ignore[attr-defined]
    result._lon0_deg = float(lon_deg)  # type: ignore[attr-defined]
    result._alt0_m = float(alt_m)  # type: ignore[attr-defined]
    return result


def wind_contribution(result: SimResult, wind_enu: np.ndarray) -> Dict[str, Any]:
    """估算恒定风偏贡献 (简化)。"""
    e, n, u = enu_coords(result)
    range_no_wind = np.hypot(e, n)
    drift_e = wind_enu[0] * result.t
    drift_n = wind_enu[1] * result.t
    return {
        "wind_e_m_s": float(wind_enu[0]),
        "wind_n_m_s": float(wind_enu[1]),
        "drift_e_m": float(drift_e[-1]) if drift_e.size else 0.0,
        "drift_n_m": float(drift_n[-1]) if drift_n.size else 0.0,
        "range_m": float(range_no_wind[-1]) if range_no_wind.size else 0.0,
    }


__all__ = [
    "detect_frame",
    "enu_coords",
    "geodetic_coords",
    "downrange",
    "altitude",
    "speed",
    "mass",
    "attach_launch_lla",
    "wind_contribution",
]
