"""地形感知落地事件测试。"""

from __future__ import annotations

import numpy as np

from ballistic_sim.models.terrain import NullTerrainModel, TerrainExtent, TerrainModel
from ballistic_sim.phases.events import make_ground_event


def _make_flat_terrain(height: float = 100.0):
    """构造水平地形。"""
    elev = np.array([[height]], dtype=float)
    extent = TerrainExtent(lat_min=0.0, lat_max=1.0, lon_min=0.0, lon_max=1.0)
    return TerrainModel(elev, extent)


def test_ground_event_enu_with_terrain() -> None:
    """ENU 模式下 terrain 高度参与落地判据。"""
    terrain = _make_flat_terrain(100.0)
    ev = make_ground_event(frame="ENU", terrain=terrain, lat0=0.0, lon0=0.0)
    # 在地形上方 150 m
    assert ev(0.0, np.array([0.0, 0.0, 150.0])) > 0.0
    # 低于地形 50 m
    assert ev(0.0, np.array([0.0, 0.0, 50.0])) < 0.0
    # 恰好等于地形高程
    assert abs(ev(0.0, np.array([0.0, 0.0, 100.0]))) < 1e-9


def test_ground_event_enu_without_terrain() -> None:
    """ENU 模式下无 terrain 时行为与旧版一致。"""
    ev = make_ground_event(frame="ENU")
    assert ev(0.0, np.array([0.0, 0.0, 100.0])) == 100.0


def test_ground_event_eci_with_terrain() -> None:
    """ECI 模式下 terrain 高度参与落地判据。"""
    from ballistic_sim.frames import geodetic_to_ecef, ecef_to_eci

    terrain = _make_flat_terrain(100.0)
    ev = make_ground_event(frame="ECI", terrain=terrain)
    # 取发射点上方 150 m 的 ECI 位置
    r_ecef = geodetic_to_ecef(0.0, 0.0, 150.0)
    r_eci = ecef_to_eci(r_ecef, 0.0)
    y = np.concatenate([r_eci, np.zeros(3)])
    assert ev(0.0, y) > 0.0

    r_ecef2 = geodetic_to_ecef(0.0, 0.0, 50.0)
    r_eci2 = ecef_to_eci(r_ecef2, 0.0)
    y2 = np.concatenate([r_eci2, np.zeros(3)])
    assert ev(0.0, y2) < 0.0


def test_ground_event_with_null_terrain() -> None:
    """NullTerrainModel 等效于无地形。"""
    ev = make_ground_event(frame="ENU", terrain=NullTerrainModel(), lat0=0.0, lon0=0.0)
    assert ev(0.0, np.array([0.0, 0.0, 50.0])) == 50.0
