"""补充引力模型测试以提升覆盖率。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.constants import WGS84_A
from ballistic_sim.models.gravity import (
    gravity_eci,
    gravity_enu,
    gravity_j2,
    gravity_magnitude,
    gravity_pointmass,
    gravity_simple,
    gravity_wgs84,
)


def test_gravity_simple_at_equator() -> None:
    """简化重力模型赤道海平面约 9.78 m/s²。"""
    g = gravity_simple(0.0, 0.0)
    assert g == pytest.approx(9.78, abs=1e-2)


def test_gravity_simple_decreases_with_altitude() -> None:
    """简化重力随高度降低。"""
    g0 = gravity_simple(0.0, 0.0)
    g1 = gravity_simple(10e3, 0.0)
    assert g1 < g0


def test_gravity_wgs84_pole_vs_equator() -> None:
    """WGS84 极区重力大于赤道。"""
    g_eq = gravity_wgs84(0.0, 0.0)
    g_po = gravity_wgs84(0.0, 90.0)
    assert g_po > g_eq


def test_gravity_enu_simple() -> None:
    """simple 模型返回 ENU 重力向量。"""
    g_vec = gravity_enu(lat=0.0, alt=0.0, model="simple")
    assert g_vec[0] == 0.0
    assert g_vec[1] == 0.0
    assert g_vec[2] < 0.0


def test_gravity_eci_pointmass_alias() -> None:
    """pointmass 别名等价于 point。"""
    r = np.array([WGS84_A, 0.0, 0.0])
    g1 = gravity_eci(r, model="point")
    g2 = gravity_eci(r, model="pointmass")
    assert np.allclose(g1, g2)


def test_gravity_eci_invalid_model() -> None:
    """未知 ECI 模型抛出 ValueError。"""
    with pytest.raises(ValueError):
        gravity_eci(np.array([WGS84_A, 0.0, 0.0]), model="unknown")


def test_gravity_enu_invalid_model() -> None:
    """未知 ENU 模型抛出 ValueError。"""
    with pytest.raises(ValueError):
        gravity_enu(0.0, alt=0.0, model="unknown")


def test_gravity_magnitude_near_center() -> None:
    """极近地心时重力大小返回 0。"""
    assert gravity_magnitude(np.array([1e-10, 0.0, 0.0])) == 0.0


def test_gravity_pointmass_near_center() -> None:
    """极近地心时点质量引力返回零矢量。"""
    assert np.allclose(gravity_pointmass(np.array([1e-10, 0.0, 0.0])), 0.0)


def test_gravity_j2_near_center() -> None:
    """极近地心时 J2 引力返回零矢量。"""
    assert np.allclose(gravity_j2(np.array([1e-10, 0.0, 0.0])), 0.0)


def test_gravity_j2_vs_pointmass_magnitude() -> None:
    """J2 模型在低地球轨道与点质量差异很小。"""
    r = np.array([WGS84_A + 400e3, 0.0, 0.0])
    g_j2 = gravity_j2(r)
    g_pm = gravity_pointmass(r)
    rel = np.linalg.norm(g_j2 - g_pm) / np.linalg.norm(g_pm)
    assert 1e-6 < rel < 1e-2
