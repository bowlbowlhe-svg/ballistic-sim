"""坐标转换测试。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.frames import (
    ecef_to_eci,
    ecef_to_enu_vec,
    ecef_to_geodetic,
    eci_to_ecef,
    enu_to_ecef_vec,
    geodetic_to_ecef,
    vel_ecef_to_eci,
    vel_eci_to_ecef,
)
from ballistic_sim.constants import WGS84_A

RTOL = 1e-9


def _relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """计算相对误差。"""
    norm = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1e-30)
    return float(np.linalg.norm(a - b) / norm)


def test_ecef_to_eci_and_back() -> None:
    """ECI→ECEF→ECI 往返一致性（位置/速度相对误差 < 1e-9）。"""
    t = 123.45
    r_ecef = np.array([WGS84_A + 1000.0, 2000.0, 3000.0])
    v_ecef = np.array([10.0, 20.0, 30.0])

    r_eci = ecef_to_eci(r_ecef, t)
    v_eci = vel_ecef_to_eci(r_ecef, v_ecef, t)

    r_back = eci_to_ecef(r_eci, t)
    v_back = vel_eci_to_ecef(r_eci, v_eci, t)

    assert _relative_error(r_back, r_ecef) < RTOL
    assert _relative_error(v_back, v_ecef) < RTOL


def test_ecef_to_enu_and_back() -> None:
    """ECEF→ENU→ECEF 往返一致性。"""
    lat, lon = 30.0, 120.0
    v_ecef = np.array([100.0, 200.0, 50.0])
    v_enu = ecef_to_enu_vec(v_ecef, lat, lon)
    v_back = enu_to_ecef_vec(v_enu, lat, lon)
    assert _relative_error(v_back, v_ecef) < RTOL


def test_eci_to_enu_and_back() -> None:
    """ECI→ENU→ECI 往返一致性。"""
    lat, lon = 45.0, 90.0
    t = 60.0
    # 构造一个 ECI 矢量
    r_eci = np.array([WGS84_A, 0.0, 0.0])
    # 先转到 ECEF，再用 ECEF->ENU（ENU 基由 lat/lon 定义，与 t 无关）
    r_ecef = eci_to_ecef(r_eci, t)
    v_enu = ecef_to_enu_vec(r_ecef, lat, lon)
    r_ecef_back = enu_to_ecef_vec(v_enu, lat, lon)
    r_eci_back = ecef_to_eci(r_ecef_back, t)
    assert _relative_error(r_eci_back, r_eci) < RTOL


@pytest.mark.parametrize(
    "lat,lon,alt",
    [
        (0.0, 0.0, 0.0),  # 赤道
        (89.999, 45.0, 100.0),  # 近北极
        (-89.999, 123.0, 100.0),  # 近南极
        (45.0, 116.0, 500.0),  # 中纬度
    ],
)
def test_geodetic_ecef_round_trip(lat: float, lon: float, alt: float) -> None:
    """发射点在赤道/极点/中纬度的测试。"""
    r_ecef = geodetic_to_ecef(lat, lon, alt)
    lat_back, lon_back, alt_back = ecef_to_geodetic(r_ecef)
    assert lat_back == pytest.approx(lat, abs=1e-6)
    assert lon_back == pytest.approx(lon, abs=1e-6)
    assert alt_back == pytest.approx(alt, rel=1e-6)
