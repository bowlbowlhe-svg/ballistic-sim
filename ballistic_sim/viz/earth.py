"""地球几何与简化海岸线/大陆轮廓构造（供 3D 可视化使用）。

所有坐标默认使用地球平均半径 6371 km；函数返回 ECEF 直角坐标（m）。
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ballistic_sim.constants import RE_EARTH_MEAN
from ballistic_sim.frames import (
    eci_to_ecef,
    enu_to_ecef_vec,
    geodetic_to_ecef,
)
from ballistic_sim.simulator import SimResult
from ballistic_sim.viz import detect_frame


def sphere_mesh(
    radius: float = RE_EARTH_MEAN,
    n_lat: int = 50,
    n_lon: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造地球球面经纬网格。

    Parameters
    ----------
    radius:
        球体半径（m），默认地球平均半径。
    n_lat, n_lon:
        纬度/经度网格采样数。

    Returns
    -------
    x, y, z:
        形状为 ``(n_lat, n_lon)`` 的 ECEF 坐标数组（m）。
    """
    lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, n_lat)
    lon = np.linspace(-np.pi, np.pi, n_lon)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x = radius * np.cos(lat_grid) * np.cos(lon_grid)
    y = radius * np.cos(lat_grid) * np.sin(lon_grid)
    z = radius * np.sin(lat_grid)
    return x, y, z


def ecef_from_latlon(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    radius: float = RE_EARTH_MEAN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """由经纬度构造球面 ECEF 坐标（m）。"""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z


def coastline_polylines(n_points: int = 360) -> List[np.ndarray]:
    """返回简化的装饰性海岸线/大陆轮廓 ECEF 折线。

    当前实现使用参数化近似曲线，非真实海岸线；可在无外部 shapefile 时提供视觉参考。
    每条折线为 ``(N, 3)`` 的 ECEF 点列（m）。
    """
    lons = np.linspace(-180.0, 180.0, n_points)
    # 若干谐波叠加，形成环绕全球的波浪状轮廓，模拟大陆分布。
    lats = (
        22.0 * np.sin(np.radians(lons))
        + 12.0 * np.sin(np.radians(3.0 * lons))
        + 6.0 * np.sin(np.radians(7.0 * lons))
    )
    lats = np.clip(lats, -60.0, 70.0)
    x, y, z = ecef_from_latlon(lats, lons, RE_EARTH_MEAN)
    return [np.column_stack([x, y, z])]


def result_to_ecef(result: SimResult) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """将 ``SimResult`` 的轨迹转换为 ECEF 坐标与几何高度。

    Returns
    -------
    x_m, y_m, z_m, alt_m:
        长度与 ``result.t`` 相同的数组。
    """
    if result.y.size == 0 or result.t.size == 0:
        raise ValueError("SimResult 为空，无法提取 ECEF 坐标")

    frame = detect_frame(result)
    if frame == "ECI":
        r_ecef = np.array(
            [eci_to_ecef(result.y[i, 0:3], result.t[i]) for i in range(len(result.t))]
        )
        alt = np.linalg.norm(r_ecef, axis=1) - RE_EARTH_MEAN
    elif frame == "ENU":
        e = result.y[:, 0]
        n = result.y[:, 1]
        u = result.y[:, 2]
        lat0 = float(getattr(result, "_lat0_deg", 0.0))
        lon0 = float(getattr(result, "_lon0_deg", 0.0))
        alt0 = float(getattr(result, "_alt0_m", 0.0))
        r0 = geodetic_to_ecef(lat0, lon0, alt0)
        r_ecef = np.array(
            [r0 + enu_to_ecef_vec([e[i], n[i], u[i]], lat0, lon0) for i in range(len(e))]
        )
        alt = u + alt0
    else:
        raise ValueError(f"不支持的状态维度: {result.y.shape}")

    return r_ecef[:, 0], r_ecef[:, 1], r_ecef[:, 2], alt


__all__ = [
    "sphere_mesh",
    "ecef_from_latlon",
    "coastline_polylines",
    "result_to_ecef",
]
