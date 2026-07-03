r"""统一引力模型。

来源与差异
----------
- 默认 ECI 引力模型继承自 ``D:\Pycharm\游戏与兴趣\火箭仿真程序\lvsim\gravity.py``，
  提供球对称点质量与点质量+J2 扁率摄动两种保真度。
- 近地 ENU 正常重力模型吸收自 ``D:\Pycharm\游戏与兴趣\外弹道方程\ballistics\gravity.py``
  的 WGS84 Somigliana 公式，用于地面/近地快速校核。
- 对外统一为 ``gravity_eci(r, model)`` 与 ``gravity_enu(lat, alt, model)`` 两个接口。

接口约定
--------
- ``gravity_eci`` 输入为惯性系位置矢量（m），输出为引力加速度矢量（m/s²），方向指向地心。
- ``gravity_enu`` 输入为大地纬度（deg）与几何高度（m），输出为 ENU 坐标系下的
  重力加速度向量，通常仅 U 分量为负。
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np

from ballistic_sim.constants import (
    DEG2RAD,
    GM_EARTH,
    J2_EARTH,
    WGS84_A,
    WGS84_E2,
)

_MIN_RADIUS = 1.0e-9  # 极小半径保护阈值 (m)


def _as_vec3(r: Union[float, np.ndarray]) -> np.ndarray:
    """把入参转换为 float64 类型的 (3,) numpy 数组并校验维度。"""
    vec = np.asarray(r, dtype=np.float64).reshape(-1)
    if vec.size != 3:
        raise ValueError(f"位置矢量 r 必须为长度 3 的 3 维矢量，当前长度为 {vec.size}。")
    return vec.copy()


def gravity_magnitude(r: Union[float, np.ndarray]) -> float:
    """点质量重力加速度的标量大小 |a| = GM / |r|² (m/s²)。"""
    vec = _as_vec3(r)
    rmag = float(np.linalg.norm(vec))
    if rmag < _MIN_RADIUS:
        return 0.0
    return float(GM_EARTH / (rmag * rmag))


def gravity_pointmass(r_eci: Union[float, np.ndarray]) -> np.ndarray:
    """球对称点质量引力加速度 a = -GM · r / |r|³ (m/s²)。"""
    vec = _as_vec3(r_eci)
    rmag = float(np.linalg.norm(vec))
    if rmag < _MIN_RADIUS:
        return np.zeros(3, dtype=np.float64)
    return (-GM_EARTH / (rmag**3)) * vec


def gravity_j2(r_eci: Union[float, np.ndarray]) -> np.ndarray:
    """点质量 + J2 扁率摄动的引力加速度 (m/s²)。

    采用标准 ECI/ECEF 近似公式（z 轴沿地球自转轴）::

        k = 1.5 * J2 * GM * RE² / r⁵
        ax = -GM·x/r³ + k·x·(5z²/r² - 1)
        ay = -GM·y/r³ + k·y·(5z²/r² - 1)
        az = -GM·z/r³ + k·z·(5z²/r² - 3)
    """
    vec = _as_vec3(r_eci)
    x, y, z = vec[0], vec[1], vec[2]
    rmag = float(np.linalg.norm(vec))
    if rmag < _MIN_RADIUS:
        return np.zeros(3, dtype=np.float64)

    r2 = rmag * rmag
    r3 = r2 * rmag
    r5 = r2 * r3

    a_point = (-GM_EARTH / r3) * vec
    k = 1.5 * J2_EARTH * GM_EARTH * (WGS84_A * WGS84_A) / r5
    zz5 = 5.0 * (z * z) / r2

    a_j2 = np.array(
        [k * x * (zz5 - 1.0), k * y * (zz5 - 1.0), k * z * (zz5 - 3.0)],
        dtype=np.float64,
    )
    return a_point + a_j2


def gravity_eci(
    r_eci: Union[float, np.ndarray],
    model: Literal["j2", "point", "pointmass"] = "j2",
) -> np.ndarray:
    """ECI 引力加速度矢量 (m/s²)。

    Parameters
    ----------
    r_eci:
        ECI 位置矢量 (m)，长度 3。
    model:
        ``"j2"`` 为点质量+J2（默认），``"point"`` / ``"pointmass"`` 为球对称点质量。
    """
    if model in ("point", "pointmass"):
        return gravity_pointmass(r_eci)
    if model == "j2":
        return gravity_j2(r_eci)
    raise ValueError(f"未知 ECI 引力模型: {model}")


def gravity_wgs84(h: Union[float, np.ndarray], lat: float) -> Union[float, np.ndarray]:
    """WGS84 正常重力 (Somigliana 公式) (m/s²)。

    Parameters
    ----------
    h:
        几何高度 (m)，可为标量或数组。
    lat:
        地理纬度 (deg)。
    """
    lat_rad = lat * DEG2RAD
    sin2 = np.sin(lat_rad) ** 2
    g0 = 9.7803253359 * (1.0 + 0.00193185265241 * sin2) / np.sqrt(1.0 - WGS84_E2 * sin2)
    return g0 * (WGS84_A / (WGS84_A + h)) ** 2


def gravity_simple(h: Union[float, np.ndarray], lat: float) -> Union[float, np.ndarray]:
    """简化重力公式（含纬度与高度修正）。"""
    lat_rad = lat * DEG2RAD
    g0 = 9.780327 * (
        1.0 + 0.0053024 * np.sin(lat_rad) ** 2 - 0.0000058 * np.sin(2.0 * lat_rad) ** 2
    )
    Re = 6378137.0
    return g0 * (Re / (Re + np.maximum(h, 0.0))) ** 2


def gravity_enu(
    lat: float,
    alt: float = 0.0,
    model: Literal["wgs84", "simple"] = "wgs84",
) -> np.ndarray:
    """返回 ENU 坐标系下的重力加速度向量 (m/s²)。

    在局部 ENU 中，重力指向 -U 方向。

    Parameters
    ----------
    lat:
        地理纬度 (deg)。
    alt:
        几何高度 (m)。
    model:
        ``"wgs84"`` 为 Somigliana 正常重力（默认），``"simple"`` 为简化公式。
    """
    if model == "wgs84":
        g = gravity_wgs84(alt, lat)
    elif model == "simple":
        g = gravity_simple(alt, lat)
    else:
        raise ValueError(f"未知 ENU 引力模型: {model}")
    return np.array([0.0, 0.0, -float(g)], dtype=np.float64)
