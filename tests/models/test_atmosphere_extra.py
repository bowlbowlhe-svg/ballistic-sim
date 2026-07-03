"""补充大气模型测试以提升覆盖率。"""

from __future__ import annotations

import pytest

from ballistic_sim.constants import P0_ISA, RHO0_ISA, T0_ISA
from ballistic_sim.models.atmosphere import NullAtmosphere, StandardAtmosphere, make_atmosphere


def test_null_atmosphere() -> None:
    """NullAtmosphere 密度/压强为 0，温度/声速为海平面标准值。"""
    atm = NullAtmosphere()
    assert atm.density(0.0) == 0.0
    assert atm.pressure(0.0) == 0.0
    assert atm.temperature(0.0) == T0_ISA
    assert atm.sound_speed(0.0) > 0.0


def test_make_atmosphere_none() -> None:
    """make_atmosphere('none') 返回 NullAtmosphere。"""
    atm = make_atmosphere("none")
    assert isinstance(atm, NullAtmosphere)


def test_make_atmosphere_standard() -> None:
    """make_atmosphere('isa') 返回 StandardAtmosphere。"""
    atm = make_atmosphere("isa")
    assert isinstance(atm, StandardAtmosphere)
    assert atm.density(0.0) == pytest.approx(RHO0_ISA, rel=1e-4)


def test_standard_atmosphere_temperature_offset() -> None:
    """delta_t 叠加到温度结果。"""
    atm = StandardAtmosphere(delta_t=10.0)
    T = atm.temperature(0.0)
    assert T == pytest.approx(T0_ISA + 10.0, rel=1e-6)


def test_standard_atmosphere_density_factor() -> None:
    """density_factor 缩放密度。"""
    atm = StandardAtmosphere(density_factor=0.9)
    assert atm.density(0.0) == pytest.approx(0.9 * RHO0_ISA, rel=1e-6)


def test_standard_atmosphere_pressure_offset() -> None:
    """pressure_offset 偏移海平面气压。"""
    atm = StandardAtmosphere(pressure_offset=1000.0)
    assert atm.pressure(0.0) == pytest.approx(P0_ISA + 1000.0, rel=1e-6)


def test_standard_atmosphere_humidity_correction() -> None:
    """相对湿度 > 0 时密度略低（虚温修正）。"""
    dry = StandardAtmosphere(relative_humidity=0.0)
    humid = StandardAtmosphere(relative_humidity=0.8)
    assert humid.density(0.0) < dry.density(0.0)


def test_standard_atmosphere_call_returns_namedtuple_fields() -> None:
    """__call__ 返回包含 T/p/rho/c/mu/q 的对象。"""
    atm = StandardAtmosphere()
    state = atm(0.0)
    assert hasattr(state, "T")
    assert hasattr(state, "p")
    assert hasattr(state, "rho")
    assert hasattr(state, "c")
    assert hasattr(state, "mu")
    assert hasattr(state, "q")


def test_standard_atmosphere_exceeds_86km() -> None:
    """86 km 以上仍返回单调衰减的密度与压强。"""
    atm = StandardAtmosphere()
    rho86 = atm.density(86000.0)
    rho100 = atm.density(100000.0)
    assert rho100 < rho86
    assert rho100 > 0.0
