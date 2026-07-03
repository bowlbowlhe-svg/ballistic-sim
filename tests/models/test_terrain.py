"""地形模型测试。"""

from __future__ import annotations

import pytest

from ballistic_sim.models.terrain import (
    NullTerrainModel,
    TerrainExtent,
    generate_hilly_terrain,
)


def test_procedural_terrain_reproducible_with_seed() -> None:
    """程序化地形在固定种子下可复现。"""
    extent = TerrainExtent(lat_min=0.0, lat_max=1.0, lon_min=0.0, lon_max=1.0)
    t1 = generate_hilly_terrain(extent, seed=12345)
    t2 = generate_hilly_terrain(extent, seed=12345)
    lat, lon = 0.5, 0.5
    assert t1.height_at(lat, lon) == pytest.approx(t2.height_at(lat, lon))


def test_terrain_cache_returns_same_value() -> None:
    """同一查询点缓存命中后返回相同值。"""
    extent = TerrainExtent(lat_min=0.0, lat_max=1.0, lon_min=0.0, lon_max=1.0)
    terrain = generate_hilly_terrain(extent, seed=42)
    lat, lon = 0.123, 0.456
    h1 = terrain.height_at(lat, lon)
    h2 = terrain.height_at(lat, lon)
    assert h1 == pytest.approx(h2)
    assert terrain._height_at_cached.cache_info().hits >= 1


def test_two_stage_consistent_with_direct_query() -> None:
    """两层检测：精确查询与 impact_check 结果一致。"""
    extent = TerrainExtent(lat_min=0.0, lat_max=1.0, lon_min=0.0, lon_max=1.0)
    terrain = generate_hilly_terrain(extent, seed=7)
    lat, lon = 0.3, 0.7
    terrain_h = terrain.height_at(lat, lon)
    impact = terrain.impact_check(lat, lon, terrain_h - 0.01, margin=0.0)
    no_impact = terrain.impact_check(lat, lon, terrain_h + 100.0, margin=0.0)
    assert impact is True
    assert no_impact is False


def test_null_terrain_returns_zero() -> None:
    """terrain = null 模式下 height_at(lat, lon) 返回 0。"""
    terrain = NullTerrainModel()
    assert terrain.height_at(45.0, 123.0) == 0.0
    assert terrain.height_at(-80.0, -45.0) == 0.0
