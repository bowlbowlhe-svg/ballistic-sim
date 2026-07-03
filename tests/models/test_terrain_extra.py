"""补充地形模型单元测试以提升覆盖率。"""

from __future__ import annotations

import numpy as np

from ballistic_sim.models.terrain import NullTerrainModel, TerrainExtent, TerrainModel


def test_null_terrain_always_zero() -> None:
    """NullTerrainModel 高程恒为 0。"""
    terr = NullTerrainModel()
    assert terr.height_at(0.0, 0.0) == 0.0
    assert not terr.impact_check(0.0, 0.0, 100.0)


def test_terrain_model_bilinear_interpolation() -> None:
    """TerrainModel 双线性插值。"""
    elev = np.array(
        [
            [0.0, 100.0],
            [100.0, 200.0],
        ],
        dtype=float,
    )
    extent = TerrainExtent(lat_min=0.0, lat_max=1.0, lon_min=0.0, lon_max=1.0)
    terr = TerrainModel(elev, extent, flip_y=False)
    # 角点
    assert terr.height_at(0.0, 0.0) == 0.0
    assert terr.height_at(1.0, 1.0) == 200.0
    # 中心
    assert terr.height_at(0.5, 0.5) == 100.0


def test_terrain_model_outside_returns_zero() -> None:
    """范围外查询返回 0。"""
    elev = np.array([[50.0]], dtype=float)
    extent = TerrainExtent(lat_min=0.0, lat_max=1.0, lon_min=0.0, lon_max=1.0)
    terr = TerrainModel(elev, extent)
    assert terr.height_at(2.0, 2.0) == 0.0


def test_terrain_model_no_data_replaced() -> None:
    """无效值替换为 0。"""
    elev = np.array([[np.nan, -9999.0], [10.0, 20.0]], dtype=float)
    extent = TerrainExtent(lat_min=0.0, lat_max=1.0, lon_min=0.0, lon_max=1.0)
    terr = TerrainModel(elev, extent, no_data_value=-9999.0, flip_y=False)
    assert terr.height_at(0.0, 0.0) == 0.0
    assert terr.height_at(0.0, 1.0) == 0.0


def test_terrain_model_impact_check() -> None:
    """impact_check 正确比较高度。"""
    elev = np.array([[100.0]], dtype=float)
    extent = TerrainExtent(lat_min=0.0, lat_max=1.0, lon_min=0.0, lon_max=1.0)
    terr = TerrainModel(elev, extent)
    assert terr.impact_check(0.5, 0.5, 50.0)
    assert not terr.impact_check(0.5, 0.5, 150.0)


def test_terrain_model_height_at_enu() -> None:
    """height_at_enu 基于 ENU 偏移转换后查询。"""
    elev = np.array([[0.0, 10.0], [10.0, 20.0]], dtype=float)
    extent = TerrainExtent(lat_min=0.0, lat_max=0.01, lon_min=0.0, lon_max=0.01)
    terr = TerrainModel(elev, extent)
    # 在赤道附近 1° ≈ 111 km，0.01° ≈ 1.11 km
    h = terr.height_at_enu(500.0, 500.0, lat0=0.0, lon0=0.0)
    assert h >= 0.0


def test_terrain_model_flip_y_false() -> None:
    """flip_y=False 时第 0 行对应 lat_min。"""
    elev = np.array(
        [
            [0.0, 0.0],
            [100.0, 100.0],
        ],
        dtype=float,
    )
    extent = TerrainExtent(lat_min=0.0, lat_max=1.0, lon_min=0.0, lon_max=1.0)
    terr = TerrainModel(elev, extent, flip_y=False)
    assert terr.height_at(0.0, 0.0) == 0.0
    assert terr.height_at(1.0, 0.0) == 100.0
