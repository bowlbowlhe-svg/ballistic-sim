"""风模型测试。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.models.wind import (
    DrydenGust,
    LogarithmicWind,
    PowerLawWind,
    UniformWind,
)


def test_uniform_wind_returns_set_values() -> None:
    """均匀风返回设定值。"""
    wind = UniformWind(e=1.0, n=2.0, u=0.5)
    state = wind(100.0)
    assert state.e == pytest.approx(1.0)
    assert state.n == pytest.approx(2.0)
    assert state.u == pytest.approx(0.5)


def test_uniform_wind_with_gradient() -> None:
    """均匀风含梯度时东分量随风高线性变化。"""
    wind = UniformWind(e=0.0, n=0.0, u=0.0, gradient_e=0.01)
    assert wind(0.0).e == pytest.approx(0.0)
    assert wind(100.0).e == pytest.approx(1.0)


def test_logarithmic_wind_increases_with_height() -> None:
    """对数廓线随风高增加。"""
    wind = LogarithmicWind(u_ref=10.0, z_ref=10.0, z0=0.03)
    s0 = wind(10.0)
    s1 = wind(100.0)
    s2 = wind(1000.0)

    def speed(s):
        return np.hypot(s.e, s.n)

    assert speed(s1) > speed(s0)
    assert speed(s2) > speed(s1)


def test_power_law_wind_increases_with_height() -> None:
    """幂律廓线随风高增加。"""
    wind = PowerLawWind(u_ref=10.0, z_ref=10.0, alpha=0.14)
    s0 = wind(10.0)
    s1 = wind(100.0)
    s2 = wind(1000.0)

    def speed(s):
        return np.hypot(s.e, s.n)

    assert speed(s1) > speed(s0)
    assert speed(s2) > speed(s1)


def test_dryden_gust_statistics() -> None:
    """Dryden 阵风输出统计特性合理（均值接近 0，标准差可配置）。"""
    base = UniformWind(e=0.0, n=0.0, u=0.0)
    sigma_u, sigma_v, sigma_w = 2.0, 1.5, 1.0
    gust = DrydenGust(
        base,
        sigma_u=sigma_u,
        sigma_v=sigma_v,
        sigma_w=sigma_w,
        length_u=50.0,
        length_v=50.0,
        length_w=30.0,
        seed=42,
        max_distance=5000.0,
    )
    samples = np.array([gust(float(h)) for h in np.linspace(0.0, 5000.0, 1000)])
    # 均值接近 0。
    assert abs(np.mean(samples[:, 0])) < 0.2 * sigma_u
    assert abs(np.mean(samples[:, 1])) < 0.2 * sigma_v
    assert abs(np.mean(samples[:, 2])) < 0.2 * sigma_w
    # 标准差在配置量级附近（允许较宽容差，因简化低通滤波）。
    assert 0.3 * sigma_u < np.std(samples[:, 0]) < 2.0 * sigma_u
    assert 0.3 * sigma_v < np.std(samples[:, 1]) < 2.0 * sigma_v
    assert 0.3 * sigma_w < np.std(samples[:, 2]) < 2.0 * sigma_w
