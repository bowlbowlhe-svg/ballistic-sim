"""模型缓存单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.config import (
    EnvironmentConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    VehicleConfig,
)
from ballistic_sim.dynamics.common import ModelCache
from ballistic_sim.models.aerodynamics import make_aero
from ballistic_sim.models.atmosphere import make_atmosphere
from ballistic_sim.models.wind import UniformWind
from ballistic_sim.context import _resolve_dynamics_context
from ballistic_sim.simulator import simulate


def test_cache_stores_atmosphere_queries() -> None:
    """同一高度的大气查询命中缓存。"""
    cache = ModelCache(enabled=True)
    atm = make_atmosphere("isa")
    s1 = cache.atmosphere(1000.0, atm)
    s2 = cache.atmosphere(1000.0, atm)
    assert s1 is s2
    assert cache.stats()["atmosphere"] == 1


def test_cache_stores_wind_queries() -> None:
    """同一高度的风场查询命中缓存。"""
    cache = ModelCache(enabled=True)
    wind = UniformWind(e=1.0, n=2.0, u=0.5)
    w1 = cache.wind(50.0, wind)
    w2 = cache.wind(50.0, wind)
    assert w1 is w2
    assert cache.stats()["wind"] == 1


def test_cache_stores_aero_coefficient() -> None:
    """同一 (Ma, alpha) 的气动系数查询命中缓存。"""
    cache = ModelCache(enabled=True)
    aero = make_aero("rocket")
    c1 = cache.aero_coefficient(1.2, 0.05, aero.drag_coefficient)
    c2 = cache.aero_coefficient(1.2, 0.05, aero.drag_coefficient)
    assert c1 == pytest.approx(c2)
    assert cache.stats()["aero"] == 1


def test_cache_stores_named_coefficient() -> None:
    """命名的一维系数表查询命中缓存。"""
    cache = ModelCache(enabled=True)
    table = np.array([[0.0, 0.25], [1.0, 0.30], [2.0, 0.28]])
    from scipy.interpolate import PchipInterpolator

    interp = PchipInterpolator(table[:, 0], table[:, 1])
    c1 = cache.coefficient("CD0", 0.75, interp)
    c2 = cache.coefficient("CD0", 0.75, interp)
    assert c1 == pytest.approx(c2)
    assert cache.stats()["coeff"] == 1


def test_cache_clear_empties_all_buckets() -> None:
    """clear() 清空所有缓存。"""
    cache = ModelCache(enabled=True)
    cache.atmosphere(100.0, make_atmosphere("isa"))
    cache.set("foo", "bar")
    cache.clear()
    assert cache.get("foo") is None
    assert all(v == 0 for v in cache.stats().values())


def test_disabled_cache_returns_same_value() -> None:
    """关闭缓存后结果与无缓存一致。"""
    cache = ModelCache(enabled=False)
    atm = make_atmosphere("isa")
    s1 = cache.atmosphere(1234.5, atm)
    s2 = atm(1234.5)
    assert s1.rho == pytest.approx(s2.rho)
    assert cache.stats()["atmosphere"] == 0


def test_dynamic_context_wraps_models_when_cache_enabled() -> None:
    """启用缓存时 DynamicContext 的 atmosphere/wind 被包装。"""
    cfg = SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(mass_kg=10.0, diameter_m=0.1, cd=0.3),
        launch=LaunchConfig(),
        environment=EnvironmentConfig(),
        options=OptionsConfig(),
    )
    ctx = _resolve_dynamics_context(cfg)
    assert ctx.use_cache is True
    assert ctx.cache.enabled is True
    # 包装后的对象仍可调用
    s = ctx.atmosphere(100.0)
    assert s.rho > 0.0
    w = ctx.wind(10.0)
    assert len(w.vector) == 3


def test_dynamic_context_can_disable_cache() -> None:
    """可通过 options 关闭 DynamicContext 缓存。"""
    cfg = SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(mass_kg=10.0, diameter_m=0.1, cd=0.3),
        launch=LaunchConfig(),
        environment=EnvironmentConfig(),
        options=OptionsConfig(use_cache=False),
    )
    ctx = _resolve_dynamics_context(cfg)
    assert ctx.use_cache is False
    assert ctx.cache.enabled is False


def test_cached_and_uncached_simulation_results_equal() -> None:
    """同一配置在缓存启用/关闭时结果一致。"""
    base_cfg = SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(mass_kg=10.0, diameter_m=0.1, cd=0.3),
        launch=LaunchConfig(elevation_deg=45.0, azimuth_deg=90.0, v0_m_s=700.0),
        environment=EnvironmentConfig(wind_m_s=[2.0, 0.0, 0.0]),
        options=OptionsConfig(
            integrator="RK45",
            rtol=1e-6,
            atol=1e-9,
            max_step=1.0,
            mpm_use_spin=False,
        ),
    )

    cfg_cached = base_cfg.model_copy(deep=True)
    cfg_cached.options.use_cache = True
    res_cached = simulate(cfg_cached, phases=[])

    cfg_uncached = base_cfg.model_copy(deep=True)
    cfg_uncached.options.use_cache = False
    res_uncached = simulate(cfg_uncached, phases=[])

    assert res_cached.stop_reason == res_uncached.stop_reason
    assert res_cached.t[-1] == pytest.approx(res_uncached.t[-1], rel=1e-9)
    assert np.allclose(res_cached.y[-1], res_uncached.y[-1], rtol=1e-9)


def test_cache_capacity_bound() -> None:
    """连续插入超过容量上限的不同 key 后缓存条目数受控。"""
    cache = ModelCache(enabled=True, maxsize=10000)
    atm = make_atmosphere("isa")
    for i in range(10010):
        cache.atmosphere(float(i) * 0.1, atm)
    assert cache.stats()["atmosphere"] <= 10000


def test_cache_lru_eviction() -> None:
    """LRU 淘汰策略保留最近访问的 key。"""
    cache = ModelCache(enabled=True, maxsize=10)
    atm = make_atmosphere("isa")
    cache.atmosphere(1.0, atm)
    cache.atmosphere(2.0, atm)
    # 访问 1.0，使其成为最近使用
    cache.atmosphere(1.0, atm)
    # 继续插入 9 个新 key，总容量 10，应淘汰 2.0 而非 1.0
    for i in range(3, 12):
        cache.atmosphere(float(i), atm)
    assert cache.stats()["atmosphere"] == 10
    assert cache._atm.get(round(1.0, 6)) is not None
    assert cache._atm.get(round(2.0, 6)) is None


def test_cache_clear_with_bounded_cache() -> None:
    """带容量上限的缓存 clear() 后所有桶归零。"""
    cache = ModelCache(enabled=True, maxsize=5)
    cache.atmosphere(100.0, make_atmosphere("isa"))
    cache.wind(10.0, UniformWind())
    cache.set("foo", "bar")
    cache.clear()
    assert cache.get("foo") is None
    assert all(v == 0 for v in cache.stats().values())


def test_cache_maxsize_from_options_config() -> None:
    """DynamicContext 读取 OptionsConfig.cache_maxsize 初始化缓存容量。"""
    cfg = SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(mass_kg=10.0, diameter_m=0.1, cd=0.3),
        launch=LaunchConfig(),
        environment=EnvironmentConfig(),
        options=OptionsConfig(cache_maxsize=7),
    )
    ctx = _resolve_dynamics_context(cfg)
    assert ctx.cache._maxsize == 7
