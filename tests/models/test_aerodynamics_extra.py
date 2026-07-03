"""补充气动模型测试以提升覆盖率。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.models.aerodynamics import (
    ProjectileAeroModel,
    RocketAeroModel,
    ballistic_coefficient,
    drag_accel,
    drag_from_table,
    make_aero,
)


def test_make_aero_rocket() -> None:
    """工厂函数可创建火箭气动模型。"""
    aero = make_aero("rocket")
    assert isinstance(aero, RocketAeroModel)
    assert aero.drag_coefficient(1.0) > 0.0


def test_make_aero_g1_g7() -> None:
    """工厂函数可创建 G1/G7 弹丸气动模型。"""
    g1 = make_aero("g1")
    g7 = make_aero("g7")
    assert isinstance(g1, ProjectileAeroModel)
    assert isinstance(g7, ProjectileAeroModel)
    assert g1.drag_coefficient(1.0) != g7.drag_coefficient(1.0)


def test_make_aero_constant_with_cl_slope() -> None:
    """常值模型接受升力线斜率。"""
    aero = make_aero("constant", cd=0.5, cl_slope=2.0)
    assert aero.drag_coefficient(1.0) == 0.5
    assert aero.lift_coefficient(1.0, alpha=np.radians(5.0)) == pytest.approx(
        2.0 * np.radians(5.0), abs=1e-12
    )


def test_make_aero_invalid() -> None:
    """未知模型名抛出 ValueError。"""
    with pytest.raises(ValueError):
        make_aero("unknown")


def test_rocket_aero_alpha_induced_drag() -> None:
    """攻角诱导项增加阻力系数。"""
    aero = RocketAeroModel(cd_alpha2=1.0)
    cd0 = aero.drag_coefficient(1.0, alpha=0.0)
    cd_alpha = aero.drag_coefficient(1.0, alpha=np.radians(10.0))
    assert cd_alpha > cd0


def test_rocket_aero_extrapolation_bounds() -> None:
    """超表马赫数取边界值；内部取 abs，负马赫等价于正马赫。"""
    aero = RocketAeroModel()
    cd_low = aero.drag_coefficient(0.0)
    cd_high = aero.drag_coefficient(10.0)
    cd_neg = aero.drag_coefficient(-1.0)
    assert cd_low == pytest.approx(0.30, abs=1e-6)
    assert cd_high == pytest.approx(0.18, abs=1e-6)
    assert cd_neg == aero.drag_coefficient(1.0)


def test_rocket_aero_vectorized() -> None:
    """RocketAeroModel 支持数组输入。"""
    aero = RocketAeroModel()
    ma = np.array([0.0, 1.0, 2.0, 5.0])
    cds = aero.drag_coefficient(ma)
    assert cds.shape == (4,)
    assert np.all(cds >= 0.0)


def test_projectile_aero_with_alpha() -> None:
    """ProjectileAeroModel 攻角诱导阻力。"""
    aero = ProjectileAeroModel(cl_slope=1.5)
    cd = aero.drag_coefficient(1.0, alpha=np.radians(10.0))
    assert cd > 0.0
    cl = aero.lift_coefficient(1.0, alpha=np.radians(10.0))
    assert cl == pytest.approx(1.5 * np.radians(10.0), abs=1e-12)


def test_drag_from_table_with_alpha2() -> None:
    """自定义表格支持攻角诱导项。"""
    ma = np.array([0.0, 1.0, 2.0])
    cd0 = np.array([0.2, 0.4, 0.3])
    cd_alpha2 = np.array([0.0, 0.1, 0.2])
    law = drag_from_table(ma, cd0, cd_alpha2)
    cd = law.cd(1.0, alpha=np.radians(10.0))
    assert cd == pytest.approx(0.4 + 0.1 * np.radians(10.0) ** 2, abs=1e-12)


def test_drag_accel_opposes_velocity() -> None:
    """drag_accel 方向与相对速度相反。"""
    v_rel = np.array([100.0, 0.0, 0.0])
    a = drag_accel(rho=1.225, c_snd=340.0, v_rel_vec=v_rel, Aref=1.0, mass=10.0)
    assert a[0] < 0.0
    assert np.isclose(a[1], 0.0, atol=1e-12)
    assert np.isclose(a[2], 0.0, atol=1e-12)


def test_drag_accel_zero_when_stationary() -> None:
    """相对速度为零时阻力加速度为零。"""
    a = drag_accel(rho=1.225, c_snd=340.0, v_rel_vec=np.zeros(3), Aref=1.0, mass=10.0)
    assert np.allclose(a, 0.0)


def test_ballistic_coefficient() -> None:
    """弹道系数公式。"""
    bc = ballistic_coefficient(mass_kg=43.2, form_factor=0.78, diameter_m=0.155)
    expected = 43.2 / (0.78 * 0.155**2)
    assert bc == pytest.approx(expected, rel=1e-6)
