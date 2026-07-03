"""阻力/气动模型测试。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.models.aerodynamics import (
    DRAG_G1,
    DRAG_G7,
    RocketAeroModel,
    drag_from_table,
)

_G1_REFERENCE = {
    0.5: 0.2557,
    1.0: 0.3667,
    2.0: 0.3544,
    3.0: 0.2742,
}

_G7_REFERENCE = {
    0.5: 0.1197,
    1.0: 0.2810,
    2.0: 0.1916,
    3.0: 0.1602,
}


@pytest.mark.parametrize("ma,ref", _G1_REFERENCE.items())
def test_g1_cd_magnitude(ma: float, ref: float) -> None:
    """G1 阻力系数在 Mach 0.5/1.0/2.0/3.0 与公开参考值量级一致。"""
    cd = DRAG_G1.cd(ma)
    assert cd == pytest.approx(ref, rel=0.05)


@pytest.mark.parametrize("ma,ref", _G7_REFERENCE.items())
def test_g7_cd_magnitude(ma: float, ref: float) -> None:
    """G7 阻力系数在 Mach 0.5/1.0/2.0/3.0 与公开参考值量级一致。"""
    cd = DRAG_G7.cd(ma)
    assert cd == pytest.approx(ref, rel=0.05)


def test_cd_mach_pchip_monotonic() -> None:
    """Cd-Ma 表 PCHIP 插值单调。"""
    aero = RocketAeroModel()
    ma_samples = np.linspace(0.0, 5.0, 200)
    cds = np.array([aero.drag_coefficient(ma) for ma in ma_samples])
    assert np.all(cds >= 0.0)
    assert np.all(cds <= 0.6)


def test_custom_drag_table() -> None:
    """自定义阻力表能正确读取并插值。"""
    ma = np.array([0.0, 1.0, 2.0, 3.0])
    cd = np.array([0.2, 0.5, 0.4, 0.3])
    law = drag_from_table(ma, cd)
    assert law.cd(0.0) == pytest.approx(0.2)
    assert law.cd(1.0) == pytest.approx(0.5)
    assert law.cd(2.0) == pytest.approx(0.4)
    assert law.cd(3.0) == pytest.approx(0.3)
    assert 0.3 < law.cd(1.5) < 0.5
