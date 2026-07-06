"""地球几何与坐标转换测试。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.constants import RE_EARTH_MEAN
from ballistic_sim.presets import m107_config, rocket_full_config
from ballistic_sim.simulator import SimResult, simulate
from ballistic_sim.viz import attach_launch_lla
from ballistic_sim.viz.earth import (
    coastline_polylines,
    ecef_from_latlon,
    result_to_ecef,
    sphere_mesh,
)


def test_sphere_mesh_shape() -> None:
    """sphere_mesh 应返回同形三维网格。"""
    x, y, z = sphere_mesh(n_lat=20, n_lon=40)
    assert x.shape == y.shape == z.shape == (20, 40)
    assert np.allclose(np.sqrt(x**2 + y**2 + z**2), RE_EARTH_MEAN)


def test_ecef_from_latlon_origin() -> None:
    """(0,0) 经纬度应对应 X 轴正方向、半径为地球平均半径。"""
    x, y, z = ecef_from_latlon(np.array([0.0]), np.array([0.0]))
    assert x[0] == pytest.approx(RE_EARTH_MEAN)
    assert y[0] == pytest.approx(0.0)
    assert z[0] == pytest.approx(0.0)


def test_coastlines_non_empty() -> None:
    """coastline_polylines 应返回非空 ECEF 折线列表。"""
    polylines = coastline_polylines(n_points=120)
    assert polylines
    for poly in polylines:
        assert poly.ndim == 2
        assert poly.shape[1] == 3
        assert poly.shape[0] > 0


def test_result_to_ecef_enu() -> None:
    """ENU 结果可正确转换为 ECEF 坐标与高度。"""
    result = simulate(m107_config())
    attach_launch_lla(result, 30.0, 120.0, 0.0)
    x, y, z, alt = result_to_ecef(result)
    assert len(x) == len(result.t)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(alt))


def test_result_to_ecef_eci() -> None:
    """ECI 结果可正确转换为 ECEF 坐标与高度。"""
    cfg = rocket_full_config("CZ2F")
    result = simulate(cfg)
    x, y, z, alt = result_to_ecef(result)
    assert len(x) == len(result.t)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(alt))


def test_result_to_ecef_empty_raises() -> None:
    """空结果应抛出 ValueError。"""
    with pytest.raises(ValueError, match="SimResult 为空"):
        result_to_ecef(SimResult())
