r"""统一气动模型。

来源与差异
----------
- 火箭模式 Cd-Ma 插值继承自 ``D:\Pycharm\游戏与兴趣\火箭仿真程序\lvsim\aerodynamics.py``，
  使用 PCHIP 保形插值，超表取边界末值。
- 弹丸/导弹模式 G1/G7 标准阻力律与自定义阻力系数表继承自
  ``D:\Pycharm\游戏与兴趣\外弹道方程\ballistics\drag.py``。
- 统一为 ``AeroModel`` Protocol/ABC，提供 ``drag_coefficient`` 与
  ``lift_coefficient`` 接口；lift 默认返回 0，可通过构造参数注入。

接口约定
--------
- ``drag_coefficient(mach, alpha=0)``：返回总阻力系数（零攻角 Cd0 加攻角诱导项）。
- ``lift_coefficient(mach, alpha=0)``：返回升力系数（可选，默认 0）。
- 马赫数 ``mach`` 无量纲；攻角 ``alpha`` 以弧度为单位。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Union, runtime_checkable

import numpy as np
from scipy.interpolate import PchipInterpolator


@runtime_checkable
class AeroModel(Protocol):
    """气动模型统一接口。"""

    def drag_coefficient(
        self, mach: Union[float, np.ndarray], alpha: float = 0.0
    ) -> Union[float, np.ndarray]:
        """返回给定马赫数与攻角下的阻力系数。"""
        ...

    def lift_coefficient(
        self, mach: Union[float, np.ndarray], alpha: float = 0.0
    ) -> Union[float, np.ndarray]:
        """返回给定马赫数与攻角下的升力系数（可选）。"""
        ...


def _make_pchip(xs: np.ndarray, ys: np.ndarray) -> PchipInterpolator:
    """创建单调三次保形插值器。"""
    order = np.argsort(xs)
    return PchipInterpolator(xs[order], ys[order])


# ==============================================================================
# 火箭模式：Cd-Ma PCHIP 插值
# ==============================================================================

# 典型轴对称细长箭体的零攻角阻力特征。
_ROCKET_MACH_NODES = np.array(
    [0.0, 0.40, 0.80, 0.90, 1.00, 1.10, 1.20, 1.50, 2.00, 3.00, 4.00, 5.00]
)
_ROCKET_CD_NODES = np.array(
    [0.30, 0.30, 0.31, 0.36, 0.48, 0.55, 0.55, 0.45, 0.36, 0.26, 0.21, 0.18]
)


class RocketAeroModel:
    """火箭气动模型：Cd 随马赫数 PCHIP 插值。

    Parameters
    ----------
    mach_table:
        马赫数采样点，默认采用典型运载火箭代表性标定曲线。
    cd_table:
        对应阻力系数采样点。
    cd_alpha2:
        攻角诱导阻力系数（弧度²），默认 0。
    """

    def __init__(
        self,
        mach_table: Optional[np.ndarray] = None,
        cd_table: Optional[np.ndarray] = None,
        cd_alpha2: float = 0.0,
    ):
        self._mach = np.asarray(
            mach_table if mach_table is not None else _ROCKET_MACH_NODES, dtype=float
        )
        self._cd = np.asarray(cd_table if cd_table is not None else _ROCKET_CD_NODES, dtype=float)
        if self._mach.size < 2 or self._mach.size != self._cd.size:
            raise ValueError("mach_table 与 cd_table 长度必须相同且至少为 2")
        self._interp = _make_pchip(self._mach, self._cd)
        self._ma_min = float(self._mach[0])
        self._ma_max = float(self._mach[-1])
        self._cd_min = float(self._cd[0])
        self._cd_max = float(self._cd[-1])
        self._cd_alpha2 = float(cd_alpha2)

    def _cd_of_mach(self, mach: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        ma = np.abs(np.asarray(mach, dtype=float))
        result: np.ndarray = np.where(ma <= self._ma_min, self._cd_min, self._interp(ma))
        result = np.where(ma >= self._ma_max, self._cd_max, result)
        if np.ndim(mach) == 0:
            return float(result)
        return result

    def drag_coefficient(
        self, mach: Union[float, np.ndarray], alpha: float = 0.0
    ) -> Union[float, np.ndarray]:
        return self._cd_of_mach(mach) + self._cd_alpha2 * float(alpha) ** 2

    def lift_coefficient(
        self, mach: Union[float, np.ndarray], alpha: float = 0.0
    ) -> Union[float, np.ndarray]:
        return 0.0


# 默认火箭 Cd-Ma 模型，供模块级便捷函数使用。
_DEFAULT_ROCKET_AERO = RocketAeroModel()


# ==============================================================================
# 弹丸/导弹模式：G1/G7/自定义阻力律
# ==============================================================================

_G1_MA = np.array(
    [
        0.0,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
        1.05,
        1.1,
        1.15,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.8,
        2.0,
        2.2,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
    ]
)
_G1_CD = np.array(
    [
        0.2629,
        0.2558,
        0.2557,
        0.2563,
        0.258,
        0.2633,
        0.2727,
        0.2854,
        0.3055,
        0.3667,
        0.4271,
        0.4373,
        0.4361,
        0.4339,
        0.4259,
        0.4159,
        0.4054,
        0.3949,
        0.3744,
        0.3544,
        0.3358,
        0.3097,
        0.2742,
        0.2462,
        0.2244,
        0.2076,
        0.1946,
    ]
)

_G7_MA = np.array(
    [
        0.0,
        0.5,
        0.7,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
        1.05,
        1.1,
        1.15,
        1.2,
        1.3,
        1.4,
        1.5,
        1.6,
        1.8,
        2.0,
        2.2,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
    ]
)
_G7_CD = np.array(
    [
        0.1198,
        0.1197,
        0.1244,
        0.1342,
        0.145,
        0.167,
        0.237,
        0.281,
        0.2832,
        0.2733,
        0.2641,
        0.256,
        0.2426,
        0.2304,
        0.2202,
        0.212,
        0.1995,
        0.1916,
        0.1859,
        0.1751,
        0.1602,
        0.1469,
        0.137,
        0.129,
        0.1224,
    ]
)


def _constant_zero(_: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 0.0


@dataclass(frozen=True)
class DragLaw:
    """阻力律封装。"""

    name: str
    cd0_fn: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]
    cd_alpha2_fn: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]

    def cd(self, ma: Union[float, np.ndarray], alpha: float = 0.0) -> Union[float, np.ndarray]:
        """总阻力系数。"""
        return self.cd0_fn(ma) + self.cd_alpha2_fn(ma) * alpha**2

    def __call__(
        self, ma: Union[float, np.ndarray], alpha: float = 0.0
    ) -> Union[float, np.ndarray]:
        return self.cd(ma, alpha)


DRAG_G1 = DragLaw(
    name="G1",
    cd0_fn=_make_pchip(_G1_MA, _G1_CD),
    cd_alpha2_fn=_constant_zero,
)

DRAG_G7 = DragLaw(
    name="G7",
    cd0_fn=_make_pchip(_G7_MA, _G7_CD),
    cd_alpha2_fn=_constant_zero,
)


def drag_from_table(
    ma: np.ndarray,
    cd0: np.ndarray,
    cd_alpha2: Optional[np.ndarray] = None,
) -> DragLaw:
    """从自定义表格创建阻力律。

    Parameters
    ----------
    ma:
        马赫数数组。
    cd0:
        零攻角阻力系数。
    cd_alpha2:
        攻角诱导阻力系数（可选）。
    """
    cd_alpha2 = cd_alpha2 if cd_alpha2 is not None else np.zeros_like(ma)
    return DragLaw(
        name="custom",
        cd0_fn=_make_pchip(ma, cd0),
        cd_alpha2_fn=_make_pchip(ma, cd_alpha2),
    )


class ProjectileAeroModel:
    """弹丸/导弹气动模型：G1/G7/自定义阻力律。

    Parameters
    ----------
    drag_law:
        ``DragLaw`` 实例，默认 G1。
    cl_slope:
        升力线斜率（每弧度），用于估算 Cl = cl_slope * alpha。
    """

    def __init__(
        self,
        drag_law: Optional[DragLaw] = None,
        cl_slope: float = 0.0,
    ):
        self._drag_law = drag_law if drag_law is not None else DRAG_G1
        self._cl_slope = float(cl_slope)

    def drag_coefficient(
        self, mach: Union[float, np.ndarray], alpha: float = 0.0
    ) -> Union[float, np.ndarray]:
        return self._drag_law.cd(mach, alpha)

    def lift_coefficient(
        self, mach: Union[float, np.ndarray], alpha: float = 0.0
    ) -> Union[float, np.ndarray]:
        return self._cl_slope * float(alpha)


# ==============================================================================
# 简易模型与工厂
# ==============================================================================


class ConstantAeroModel:
    """常值阻力系数模型。"""

    def __init__(self, cd: float = 0.3, cl_slope: float = 0.0):
        self._cd = float(cd)
        self._cl_slope = float(cl_slope)

    def drag_coefficient(
        self, mach: Union[float, np.ndarray], alpha: float = 0.0
    ) -> Union[float, np.ndarray]:
        return self._cd

    def lift_coefficient(
        self, mach: Union[float, np.ndarray], alpha: float = 0.0
    ) -> Union[float, np.ndarray]:
        return self._cl_slope * float(alpha)


# ==============================================================================
# 便捷函数（与测试/旧接口兼容）
# ==============================================================================


def cd_of_mach(mach: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """默认火箭 Cd-Ma 表的便捷查询（PCHIP 插值，超表取边界末值）。"""
    return _DEFAULT_ROCKET_AERO.drag_coefficient(mach)


def dynamic_pressure(rho: float, v_rel: float) -> float:
    """动压 q = 0.5 · rho · v_rel² (Pa)。"""
    return 0.5 * float(rho) * float(v_rel) * float(v_rel)


def drag_accel(
    rho: float,
    c_snd: float,
    v_rel_vec: Union[float, np.ndarray],
    Aref: float,
    mass: float,
) -> np.ndarray:
    """气动阻力加速度矢量 a_drag (m/s²)，沿相对速度反方向。

    公式
    ----
        Ma = |v_rel| / c_snd
        Cd = cd_of_mach(Ma)
        a_drag = -0.5 · rho · Cd · Aref / mass · |v_rel| · v_rel_vec
    """
    v = np.asarray(v_rel_vec, dtype=float).reshape(3)
    speed = float(np.linalg.norm(v))
    if speed <= 0.0:
        return np.zeros(3, dtype=float)
    Ma = speed / float(c_snd)
    Cd = cd_of_mach(Ma)
    coeff = -0.5 * float(rho) * Cd * float(Aref) / float(mass) * speed
    return coeff * v


# ==============================================================================
# 工厂函数
# ==============================================================================


def make_aero(
    model: str = "constant",
    **kwargs,
) -> AeroModel:
    """气动模型工厂函数。

    Parameters
    ----------
    model:
        ``"rocket"``、``"projectile"``、``"constant"`` 或 ``"g1"`` / ``"g7"``。
    **kwargs:
        传递给具体模型构造函数的参数。
    """
    name = model.lower()
    if name == "rocket":
        return RocketAeroModel(**kwargs)
    if name == "projectile":
        return ProjectileAeroModel(**kwargs)
    if name == "constant":
        return ConstantAeroModel(**kwargs)
    if name == "g1":
        return ProjectileAeroModel(drag_law=DRAG_G1, **kwargs)
    if name == "g7":
        return ProjectileAeroModel(drag_law=DRAG_G7, **kwargs)
    raise ValueError(f"未知气动模型: {model}")


def ballistic_coefficient(mass_kg: float, form_factor: float, diameter_m: float) -> float:
    """计算弹道系数 BC = m / (i · d²)，单位 kg/m²。"""
    return mass_kg / (form_factor * diameter_m**2)
