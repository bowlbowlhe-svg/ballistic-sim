"""补充风场模型测试以提升覆盖率。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.models.wind import (
    CompositeWind,
    DrydenGust,
    LogarithmicWind,
    PowerLawWind,
    ProfileWind,
    UniformWind,
    WindField3D,
    WindState,
    make_wind,
    parse_wind_profile,
)


def test_wind_state_vector_and_array() -> None:
    """WindState 提供 vector 属性与 __array__。"""
    s = WindState(e=1.0, n=2.0, u=3.0)
    assert np.allclose(s.vector, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(np.asarray(s), np.array([1.0, 2.0, 3.0]))


def test_logarithmic_wind_direction() -> None:
    """对数廓线支持风向旋转。"""
    wind = LogarithmicWind(u_ref=10.0, z_ref=10.0, z0=0.03, direction_deg=90.0)
    s = wind(100.0)
    # 90° 风向表示吹向东，故 e 分量主导
    assert abs(s.e) > abs(s.n)


def test_power_law_wind_direction() -> None:
    """幂律廓线支持风向旋转。"""
    wind = PowerLawWind(u_ref=10.0, z_ref=10.0, alpha=0.14, direction_deg=0.0)
    s = wind(100.0)
    # 0° 风向表示吹向北
    assert s.n > 0.0
    assert abs(s.e) < 1e-12


def test_profile_wind_interpolation() -> None:
    """ProfileWind 按高度插值。"""
    profile = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1000.0, 5.0, 0.0, 0.0],
            [2000.0, 10.0, 2.0, 0.0],
        ]
    )
    wind = ProfileWind(profile)
    s500 = wind(500.0)
    s2500 = wind(2500.0)
    assert 0.0 < s500.e < 5.0
    assert s2500.e == pytest.approx(10.0, abs=1e-6)


def test_profile_wind_too_few_points() -> None:
    """ProfileWind 少于 2 个点抛出 ValueError。"""
    with pytest.raises(ValueError):
        ProfileWind(np.array([[0.0, 0.0, 0.0, 0.0]]))


def test_composite_wind_boundary_layer() -> None:
    """CompositeWind 在过渡高度以下使用边界层模型。"""
    boundary = UniformWind(e=1.0, n=2.0, u=0.0)
    profile = ProfileWind(
        np.array(
            [
                [0.0, 10.0, 0.0, 0.0],
                [3000.0, 10.0, 10.0, 0.0],
            ]
        )
    )
    wind = CompositeWind(boundary, profile, transition_h=1000.0, blend=500.0)
    s500 = wind(500.0)
    assert s500.e == 1.0
    assert s500.n == 2.0


def test_composite_wind_blend() -> None:
    """CompositeWind 在过渡区混合两种模型。"""
    boundary = UniformWind(e=0.0, n=0.0, u=0.0)
    profile = ProfileWind(
        np.array(
            [
                [0.0, 10.0, 0.0, 0.0],
                [3000.0, 10.0, 0.0, 0.0],
            ]
        )
    )
    wind = CompositeWind(boundary, profile, transition_h=1000.0, blend=1000.0)
    s1500 = wind(1500.0)
    assert 0.0 < s1500.e < 10.0


def test_dryden_gust_indexed_by_distance() -> None:
    """DrydenGust 可按沿轨距离索引。"""
    base = UniformWind()
    gust = DrydenGust(base, sigma_u=1.0, seed=42, max_distance=1000.0)
    s0 = gust(0.0, s=0.0)
    s1 = gust(0.0, s=500.0)
    assert isinstance(s0, WindState)
    assert isinstance(s1, WindState)


def test_wind_field_3d_no_correction() -> None:
    """WindField3D 无空间修正时退化为 base。"""
    base = UniformWind(e=3.0, n=4.0, u=0.0)
    field = WindField3D(base)
    s = field(100.0, x=1000.0, y=2000.0)
    assert s.e == 3.0
    assert s.n == 4.0


def test_wind_field_3d_with_correction() -> None:
    """WindField3D 叠加空间修正项。"""
    base = UniformWind(e=3.0, n=4.0, u=0.0)

    def corr(x: float, y: float, h: float) -> WindState:
        return WindState(e=1.0, n=-1.0, u=0.0)

    field = WindField3D(base, spatial_correction=corr)
    s = field(100.0, x=1000.0, y=2000.0)
    assert s.e == 4.0
    assert s.n == 3.0


def test_make_wind_factories() -> None:
    """工厂函数覆盖主要模型。"""
    assert isinstance(make_wind("uniform", e=1.0), UniformWind)
    assert isinstance(make_wind("log", u_ref=10.0), LogarithmicWind)
    assert isinstance(make_wind("power", u_ref=10.0), PowerLawWind)
    assert isinstance(make_wind("none"), UniformWind)


def test_make_wind_invalid() -> None:
    """未知风模型抛出 ValueError。"""
    with pytest.raises(ValueError):
        make_wind("unknown")


def test_parse_wind_profile() -> None:
    """parse_wind_profile 解析文本并跳过注释。"""
    text = """
    # h(m) windE(m/s) windN(m/s) windU(m/s)
    0     0   0   0
    1000  5   0   0
    2000  10  2   0
    """
    profile = parse_wind_profile(text)
    assert profile.shape == (3, 4)
    assert profile[-1, 0] == 2000.0


def test_parse_wind_profile_too_few() -> None:
    """parse_wind_profile 数据不足时抛出 ValueError。"""
    with pytest.raises(ValueError):
        parse_wind_profile("0 0 0 0")
