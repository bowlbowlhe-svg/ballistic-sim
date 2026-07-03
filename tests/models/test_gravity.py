"""引力模型测试。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.constants import G0_STANDARD, GM_EARTH, WGS84_A
from ballistic_sim.models.gravity import (
    gravity_eci,
    gravity_enu,
    gravity_j2,
    gravity_magnitude,
    gravity_pointmass,
    gravity_wgs84,
)


def test_sea_level_gravity_magnitude() -> None:
    """海平面 WGS84 正常重力 ≈ 9.80665 m/s²（45° 纬度）。"""
    g_vec = gravity_enu(lat=45.0, alt=0.0, model="wgs84")
    assert np.linalg.norm(g_vec) == pytest.approx(G0_STANDARD, rel=1e-4)
    assert gravity_wgs84(0.0, 45.0) == pytest.approx(G0_STANDARD, rel=1e-4)


def test_gravity_pointmass_at_equator() -> None:
    """赤道处点质量重力与 GM / WGS84_A² 一致。"""
    r = np.array([WGS84_A, 0.0, 0.0])
    expected = GM_EARTH / (WGS84_A**2)
    assert gravity_magnitude(r) == pytest.approx(expected, rel=1e-6)


def test_j2_vs_pointmass_low_orbit() -> None:
    """J2 模型与简化点质量模型在低轨道高度差异在合理范围。"""
    h = 400000.0
    r_eq = np.array([WGS84_A + h, 0.0, 0.0])
    r_pol = np.array([0.0, 0.0, WGS84_A + h])
    for r in (r_eq, r_pol):
        g_pm = gravity_pointmass(r)
        g_j2 = gravity_j2(r)
        rel_diff = np.linalg.norm(g_j2 - g_pm) / np.linalg.norm(g_pm)
        assert 1e-5 < rel_diff < 1e-2


def test_gravity_points_to_earth_center() -> None:
    """gravity_eci(r) 方向指向地心。"""
    r = np.array([WGS84_A, 0.0, 1000.0])
    g = gravity_eci(r, model="j2")
    cosine = float(np.dot(g, -r) / (np.linalg.norm(g) * np.linalg.norm(r)))
    assert cosine == pytest.approx(1.0, abs=1e-6)


def test_gravity_zero_near_center() -> None:
    """极近地心时引力模型返回零矢量，避免除零。"""
    r = np.array([1e-10, 0.0, 0.0])
    g = gravity_eci(r, model="j2")
    assert np.allclose(g, 0.0)
