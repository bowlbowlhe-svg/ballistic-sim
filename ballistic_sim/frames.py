r"""统一坐标系转换层。

来源与差异
----------
- 主体继承自 ``D:\Pycharm\游戏与兴趣\火箭仿真程序\lvsim\frames.py``。
- 统一常数来源为 ``ballistic_sim.constants``。

接口约定
--------
- 大地坐标 (lat_deg, lon_deg, alt_m) <-> ECEF。
- ECEF <-> ECI（含速度牵连项）。
- ENU 基矢量及其与 ECEF 的矢量旋转。
"""

from __future__ import annotations

import numpy as np

from ballistic_sim.constants import (
    DEG2RAD,
    OMEGA_EARTH,
    RAD2DEG,
    WGS84_A,
    WGS84_E2,
)

__all__ = [
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "ecef_to_eci",
    "eci_to_ecef",
    "vel_ecef_to_eci",
    "vel_eci_to_ecef",
    "enu_basis",
    "enu_to_ecef_vec",
    "ecef_to_enu_vec",
    "launch_state_eci",
    "haversine_distance",
    "initial_bearing",
]

_B_POLAR = WGS84_A * np.sqrt(1.0 - WGS84_E2)
_ECC2_PRIME = WGS84_E2 / (1.0 - WGS84_E2)


def _as_vec3(v):
    """把输入规整为长度 3 的 float64 一维数组。"""
    return np.asarray(v, dtype=float).reshape(3)


def _rot_z(theta_rad: float) -> np.ndarray:
    """绕 z 轴的旋转矩阵 R(theta)。"""
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """大地坐标 -> ECEF 直角坐标。"""
    lat = lat_deg * DEG2RAD
    lon = lon_deg * DEG2RAD
    h = float(alt_m)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    denom = np.sqrt(max(1.0 - WGS84_E2 * sin_lat * sin_lat, 1e-30))
    N = WGS84_A / denom
    x = (N + h) * cos_lat * np.cos(lon)
    y = (N + h) * cos_lat * np.sin(lon)
    z = (N * (1.0 - WGS84_E2) + h) * sin_lat
    return np.array([x, y, z], dtype=float)


def ecef_to_geodetic(r_ecef) -> tuple[float, float, float]:
    """ECEF 直角坐标 -> 大地坐标（Bowring 闭式法）。"""
    r = _as_vec3(r_ecef)
    x, y, z = r[0], r[1], r[2]
    p = np.hypot(x, y)

    if p < 1e-9:
        lat_deg = 90.0 if z >= 0.0 else -90.0
        lon_deg = 0.0
        alt_m = abs(z) - _B_POLAR
        return (lat_deg, lon_deg, float(alt_m))

    lon = np.arctan2(y, x)
    theta = np.arctan2(z * WGS84_A, p * _B_POLAR)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    lat = np.arctan2(
        z + _ECC2_PRIME * _B_POLAR * sin_t**3,
        p - WGS84_E2 * WGS84_A * cos_t**3,
    )
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    denom = np.sqrt(max(1.0 - WGS84_E2 * sin_lat * sin_lat, 1e-30))
    N = WGS84_A / denom
    if abs(cos_lat) > 1e-6:
        h = p / cos_lat - N
    else:
        h = z / sin_lat - N * (1.0 - WGS84_E2)
    return (float(lat * RAD2DEG), float(lon * RAD2DEG), float(h))


def _earth_rotation_angle(t: float, theta0: float = 0.0) -> float:
    """t 时刻 ECEF 相对 ECI 的旋转角 theta (rad)。"""
    return theta0 + OMEGA_EARTH * float(t)


def ecef_to_eci(r_ecef, t: float, theta0: float = 0.0) -> np.ndarray:
    """ECEF 位置 -> ECI 位置。"""
    theta = _earth_rotation_angle(t, theta0)
    return _rot_z(theta) @ _as_vec3(r_ecef)


def eci_to_ecef(r_eci, t: float, theta0: float = 0.0) -> np.ndarray:
    """ECI 位置 -> ECEF 位置。"""
    theta = _earth_rotation_angle(t, theta0)
    return _rot_z(-theta) @ _as_vec3(r_eci)


def vel_ecef_to_eci(r_ecef, v_ecef, t: float, theta0: float = 0.0) -> np.ndarray:
    """ECEF 速度 -> ECI 速度（含地球自转牵连项）。"""
    r = _as_vec3(r_ecef)
    v = _as_vec3(v_ecef)
    omega = np.array([0.0, 0.0, OMEGA_EARTH], dtype=float)
    theta = _earth_rotation_angle(t, theta0)
    return _rot_z(theta) @ (v + np.cross(omega, r))


def vel_eci_to_ecef(r_eci, v_eci, t: float, theta0: float = 0.0) -> np.ndarray:
    """ECI 速度 -> ECEF 速度。"""
    theta = _earth_rotation_angle(t, theta0)
    R_inv = _rot_z(-theta)
    r_ecef = R_inv @ _as_vec3(r_eci)
    omega = np.array([0.0, 0.0, OMEGA_EARTH], dtype=float)
    return R_inv @ _as_vec3(v_eci) - np.cross(omega, r_ecef)


def enu_basis(lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 ENU 三基矢量在 ECEF 下的单位矢量。"""
    lat = lat_deg * DEG2RAD
    lon = lon_deg * DEG2RAD
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    e_hat = np.array([-sin_lon, cos_lon, 0.0], dtype=float)
    n_hat = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], dtype=float)
    u_hat = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], dtype=float)
    return (e_hat, n_hat, u_hat)


def enu_to_ecef_vec(v_enu, lat_deg: float, lon_deg: float) -> np.ndarray:
    """ENU 矢量 -> ECEF 矢量。"""
    e_hat, n_hat, u_hat = enu_basis(lat_deg, lon_deg)
    v = _as_vec3(v_enu)
    R = np.column_stack((e_hat, n_hat, u_hat))
    return R @ v


def ecef_to_enu_vec(v_ecef, lat_deg: float, lon_deg: float) -> np.ndarray:
    """ECEF 矢量 -> ENU 矢量。"""
    e_hat, n_hat, u_hat = enu_basis(lat_deg, lon_deg)
    v = _as_vec3(v_ecef)
    R = np.column_stack((e_hat, n_hat, u_hat))
    return R.T @ v


def launch_state_eci(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    t: float = 0.0,
    theta0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """由发射点经纬高给出 ECI 下的初始位置与初速度（含地球自转牵连）。"""
    r_ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    r_eci = ecef_to_eci(r_ecef, t, theta0)
    v_eci = vel_ecef_to_eci(r_ecef, np.zeros(3), t, theta0)
    return (r_eci, v_eci)


def haversine_distance(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    """大地线距离 (m)，使用地球平均半径近似。"""
    R = 6371000.0
    phi1, phi2 = np.radians(lat1_deg), np.radians(lat2_deg)
    dphi = np.radians(lat2_deg - lat1_deg)
    dlambda = np.radians(lon2_deg - lon1_deg)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return float(2.0 * R * np.arcsin(np.sqrt(a)))


def initial_bearing(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    """大圆初始方位角 (deg, 自北顺时针)。"""
    phi1, phi2 = np.radians(lat1_deg), np.radians(lat2_deg)
    dlambda = np.radians(lon2_deg - lon1_deg)
    x = np.sin(dlambda) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    brg = np.degrees(np.arctan2(x, y))
    return float((brg + 360.0) % 360.0)
