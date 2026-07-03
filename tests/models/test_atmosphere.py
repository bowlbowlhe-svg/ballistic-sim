"""大气模型测试。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.constants import P0_ISA, RHO0_ISA, T0_ISA
from ballistic_sim.models.atmosphere import StandardAtmosphere, make_atmosphere

# 几何高度下的 ISA 参考真值（与 lvsim.atmosphere 一致，已考虑位势高度修正）。
_ISA_REFERENCE = {
    0.0: {"rho": RHO0_ISA, "p": P0_ISA, "T": T0_ISA},
    11000.0: {"rho": 3.64801e-01, "p": 22699.9368, "T": 216.774},
    20000.0: {"rho": 8.89098e-02, "p": 5529.3006, "T": 216.650},
    32000.0: {"rho": 1.35551e-02, "p": 889.0615, "T": 228.490},
    47000.0: {"rho": 1.49651e-03, "p": 115.8506, "T": 269.684},
    50000.0: {"rho": 1.02687e-03, "p": 79.7787, "T": 270.650},
    71000.0: {"rho": 7.19647e-05, "p": 4.4795, "T": 216.846},
    86000.0: {"rho": 6.95782e-06, "p": 0.3734, "T": 186.946},
}


@pytest.fixture
def atm() -> StandardAtmosphere:
    return StandardAtmosphere()


def test_sea_level_properties(atm: StandardAtmosphere) -> None:
    """海平面密度、压强、温度、声速与标准值一致。"""
    assert atm.density(0.0) == pytest.approx(1.225, rel=1e-4)
    assert atm.pressure(0.0) == pytest.approx(101325.0, rel=1e-4)
    assert atm.temperature(0.0) == pytest.approx(288.15, rel=1e-4)
    assert atm.sound_speed(0.0) == pytest.approx(340.3, rel=1e-3)


@pytest.mark.parametrize("h", [0.0, 11000.0, 20000.0, 32000.0, 47000.0, 50000.0, 71000.0, 86000.0])
def test_isa_table_accuracy(h: float, atm: StandardAtmosphere) -> None:
    """标准高度值与 ISA 表对比，误差 < 0.5%。"""
    ref = _ISA_REFERENCE[h]
    assert atm.density(h) == pytest.approx(ref["rho"], rel=0.005)
    assert atm.pressure(h) == pytest.approx(ref["p"], rel=0.005)
    assert atm.temperature(h) == pytest.approx(ref["T"], rel=0.005)


def test_upper_extension_continuity(atm: StandardAtmosphere) -> None:
    """86 km 处左右两侧值连续（相对误差 < 1e-6）。"""
    h86 = 86000.0
    left = atm(h86)
    right = atm(h86 + 1e-6)
    assert right.rho == pytest.approx(left.rho, rel=1e-6)
    assert right.p == pytest.approx(left.p, rel=1e-6)
    assert right.T == pytest.approx(left.T, rel=1e-6)


def test_upper_extension_monotonic_decay(atm: StandardAtmosphere) -> None:
    """86 km 以上密度/压强/温度单调衰减，100 km 处密度 < 1e-7 kg/m³。"""
    hs = np.linspace(86000.0, 100000.0, 100)
    rhos = np.array([atm.density(h) for h in hs])
    ps = np.array([atm.pressure(h) for h in hs])
    Ts = np.array([atm.temperature(h) for h in hs])
    assert np.all(np.diff(rhos) < 0.0)
    assert np.all(np.diff(ps) < 0.0)
    assert np.all(np.diff(Ts) < 0.0)
    assert atm.density(100000.0) < 1e-7


def test_sound_speed_sea_level(atm: StandardAtmosphere) -> None:
    """sound_speed(0) ≈ 340.3 m/s。"""
    assert atm.sound_speed(0.0) == pytest.approx(340.3, rel=1e-3)


def test_factory_null() -> None:
    """工厂函数可创建真空模型。"""
    null_atm = make_atmosphere("none")
    assert null_atm.density(0.0) == 0.0
    assert null_atm.pressure(0.0) == 0.0
