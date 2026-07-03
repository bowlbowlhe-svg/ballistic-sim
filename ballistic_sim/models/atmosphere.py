r"""统一大气模型。

来源与差异
----------
- 主体继承自 ``D:\Pycharm\游戏与兴趣\火箭仿真程序\lvsim\atmosphere.py`` 的扩展
  ISA（0–86 km，7 层位势高度模型），保留其预计算分层基准温压方式。
- 吸收 ``D:\Pycharm\游戏与兴趣\外弹道方程\ballistics\atmosphere.py`` 的湿度虚温
  修正、自定义探空廓线能力，以及 ``delta_t`` / ``density_factor`` / ``pressure_offset``
  等工程修正项。
- **差异**：86 km 以上不再硬截断为常值，而是采用指数延拓到零，确保密度、压强、
  温度连续且单调衰减。

接口约定
--------
- 所有物理量均为 SI 单位（高度 m，温度 K，压强 Pa，密度 kg/m³，声速 m/s）。
- 提供 ``AtmosphereModel`` Protocol，统一 ``density`` / ``pressure`` /
  ``temperature`` / ``sound_speed`` 四个入口。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np
from scipy.interpolate import PchipInterpolator

from ballistic_sim.constants import (
    EPSILON_VAPOR,
    G0_STANDARD,
    GAMMA_AIR,
    P0_ISA,
    R_AIR,
    T0_ISA,
)

# ISA 有效地球半径（几何高 -> 位势高），与 lvsim 保持一致 (m)。
_R0_GEOPOT = 6356766.0

# ISA 位势高度分层底界 (m)。
_HB = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0])

# 对应温度梯度 (K/m)。
_LB = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002])

# 模型上界：86 km 几何高对应位势高 (m)。
_H_GEOPOT_MAX = 84852.0
_H_GEOM_MAX = 86000.0


@dataclass(frozen=True)
class AtmosphereState:
    """某高度的大气状态。"""

    h: float  # 几何高度 (m)
    H: float  # 位势高度 (m)
    T: float  # 温度 (K)
    p: float  # 压强 (Pa)
    rho: float  # 密度 (kg/m^3)
    c: float  # 声速 (m/s)
    mu: float  # 动力粘度 (Pa·s)
    q: float = 0.0  # 比湿 (kg/kg)


@runtime_checkable
class AtmosphereModel(Protocol):
    """大气模型统一接口。"""

    def density(self, h: float) -> float:
        """返回高度 h (m) 处的大气密度 (kg/m^3)。"""
        ...

    def pressure(self, h: float) -> float:
        """返回高度 h (m) 处的大气压强 (Pa)。"""
        ...

    def temperature(self, h: float) -> float:
        """返回高度 h (m) 处的大气温度 (K)。"""
        ...

    def sound_speed(self, h: float) -> float:
        """返回高度 h (m) 处的当地声速 (m/s)。"""
        ...


class NullAtmosphere:
    """真空大气（快速模式），密度/压强恒为 0，温度取海平面 ISA 值。"""

    def density(self, h: float) -> float:
        return 0.0

    def pressure(self, h: float) -> float:
        return 0.0

    def temperature(self, h: float) -> float:
        return float(T0_ISA)

    def sound_speed(self, h: float) -> float:
        return float(np.sqrt(GAMMA_AIR * R_AIR * T0_ISA))


def _geopotential(h: float) -> float:
    """几何高度 -> 位势高度。"""
    return _R0_GEOPOT * h / (_R0_GEOPOT + h)


def _saturated_vapor_pressure(t: float) -> float:
    """饱和水汽压 (Pa)，Magnus 公式，T 为 K。"""
    t_c = t - 273.15
    return 6.112 * 100.0 * np.exp((17.67 * t_c) / (t_c + 243.5))


def _virtual_temperature(t: float, q: float) -> float:
    """虚温 (K)。"""
    return t * (1.0 + 0.61 * q)


def _build_isa() -> tuple[np.ndarray, np.ndarray]:
    """预计算 7 层分界面的基准温度 Tb 与基准压强 Pb。"""
    n = _HB.size
    Tb = np.empty(n)
    Pb = np.empty(n)
    Tb[0] = T0_ISA
    Pb[0] = P0_ISA
    for k in range(n - 1):
        dH = _HB[k + 1] - _HB[k]
        Tn = Tb[k] + _LB[k] * dH
        if _LB[k] == 0.0:
            Pb[k + 1] = Pb[k] * np.exp(-G0_STANDARD * dH / (R_AIR * Tb[k]))
        else:
            Pb[k + 1] = Pb[k] * (Tn / Tb[k]) ** (-G0_STANDARD / (R_AIR * _LB[k]))
        Tb[k + 1] = Tn
    return Tb, Pb


_TB_ISA, _PB_ISA = _build_isa()


def _isa_state(h: float) -> tuple[float, float, float, float]:
    """返回 0–86 km 内的 ISA 状态 (T, p, rho, c)，未加湿度修正。"""
    h_g = min(max(float(h), 0.0), _H_GEOM_MAX)
    H = _geopotential(h_g)
    H = min(H, _H_GEOPOT_MAX)

    i = 0
    for k in range(_HB.size):
        if H >= _HB[k]:
            i = k

    dH = H - _HB[i]
    if _LB[i] == 0.0:
        T = _TB_ISA[i]
        p = _PB_ISA[i] * np.exp(-G0_STANDARD * dH / (R_AIR * _TB_ISA[i]))
    else:
        T = _TB_ISA[i] + _LB[i] * dH
        p = _PB_ISA[i] * (T / _TB_ISA[i]) ** (-G0_STANDARD / (R_AIR * _LB[i]))

    rho = p / (R_AIR * T)
    c = np.sqrt(GAMMA_AIR * R_AIR * T)
    return float(T), float(p), float(rho), float(c)


class StandardAtmosphere:
    """统一标准大气模型。

    主体为扩展 ISA（0–86 km），支持湿度虚温修正、自定义探空廓线、温度/密度/气压
    工程修正。86 km 以上采用指数延拓，密度、压强、温度均连续且单调衰减到零，避免
    硬截断。

    Parameters
    ----------
    delta_t:
        温度偏差 (K)，叠加到所有温度结果上（含自定义廓线）。
    density_factor:
        密度修正系数，最终密度乘以该因子。
    pressure_offset:
        地面气压偏移 (Pa)，与海平面基准压强相加。
    relative_humidity:
        相对湿度 (0–1)，用于湿度虚温修正。
    custom_profile:
        自定义探空廓线，形状 (N, 4)：[h(m), T(K), p(Pa), RH(0-1)]。
    """

    def __init__(
        self,
        delta_t: float = 0.0,
        density_factor: float = 1.0,
        pressure_offset: float = 0.0,
        relative_humidity: float = 0.0,
        custom_profile: Optional[np.ndarray] = None,
    ):
        self.delta_t = float(delta_t)
        self.density_factor = float(density_factor)
        self.pressure_offset = float(pressure_offset)
        self.relative_humidity = float(np.clip(relative_humidity, 0.0, 1.0))
        self.custom_profile = custom_profile

        # 重新预计算含 pressure_offset 的 ISA 分层基准。
        self._Tb = [T0_ISA]
        self._Pb = [P0_ISA + self.pressure_offset]
        for k in range(len(_HB) - 1):
            dH = _HB[k + 1] - _HB[k]
            Tn = self._Tb[k] + _LB[k] * dH
            if _LB[k] == 0.0:
                p_next = self._Pb[k] * np.exp(-G0_STANDARD * dH / (R_AIR * self._Tb[k]))
            else:
                p_next = self._Pb[k] * (Tn / self._Tb[k]) ** (-G0_STANDARD / (R_AIR * _LB[k]))
            self._Tb.append(float(Tn))
            self._Pb.append(float(p_next))

        # 自定义廓线插值器。
        self._custom_interp: Optional[Callable[[float], tuple[float, float, float]]] = None
        if custom_profile is not None and len(custom_profile) >= 2:
            heights = np.asarray(custom_profile[:, 0], dtype=float)
            temps = np.asarray(custom_profile[:, 1], dtype=float)
            pressures = np.asarray(custom_profile[:, 2], dtype=float)
            rhs = np.clip(np.asarray(custom_profile[:, 3], dtype=float), 0.0, 1.0)
            self._custom_t = PchipInterpolator(heights, temps)
            self._custom_p = PchipInterpolator(heights, pressures)
            self._custom_rh = PchipInterpolator(heights, rhs)
            self._custom_max_h = float(heights[-1])
            self._custom_interp = lambda h: (
                float(self._custom_t(h)),
                float(self._custom_p(h)),
                float(self._custom_rh(h)),
            )
        else:
            self._custom_max_h = 0.0

        # 预计算 86 km 处的 ISA 干态值，用于高空指数延拓。
        T86, p86, rho86, c86 = self._isa_state_dry(_H_GEOM_MAX)
        self._T86 = T86
        self._p86 = p86
        self._rho86 = rho86
        self._c86 = c86

        # 高空指数延拓的 e-folding 尺度。
        # 86 km 以上大气已极稀薄，采用比标准标高更短的指数衰减尺度，使密度在
        # 100 km 附近迅速降至 <1e-7 kg/m³，同时保持压强、温度连续单调衰减到零。
        # 三者满足理想气体一致性：1/H_rho = 1/H_p - 1/H_T。
        self._H_rho = 3000.0  # 密度 e-folding 尺度 (m)
        self._H_T = 12000.0  # 温度 e-folding 尺度 (m)
        self._H_p = 1.0 / (1.0 / self._H_rho + 1.0 / self._H_T)

    def _isa_state_dry(self, h: float) -> tuple[float, float, float, float]:
        """使用本实例基准的 ISA 干态计算。"""
        h_g = min(max(float(h), 0.0), _H_GEOM_MAX)
        H = _geopotential(h_g)
        H = min(H, _H_GEOPOT_MAX)

        i = 0
        for k in range(_HB.size):
            if H >= _HB[k]:
                i = k

        dH = H - _HB[i]
        if _LB[i] == 0.0:
            T = self._Tb[i]
            p = self._Pb[i] * np.exp(-G0_STANDARD * dH / (R_AIR * self._Tb[i]))
        else:
            T = self._Tb[i] + _LB[i] * dH
            p = self._Pb[i] * (T / self._Tb[i]) ** (-G0_STANDARD / (R_AIR * _LB[i]))

        T += self.delta_t
        rho = p / (R_AIR * T)
        c = np.sqrt(GAMMA_AIR * R_AIR * T)
        return float(T), float(p), float(rho), float(c)

    def _state(self, h: float) -> AtmosphereState:
        """查询高度 h 处的大气状态（含湿度修正）。"""
        h = float(h)

        use_custom = (
            self._custom_interp is not None
            and self.custom_profile is not None
            and h <= self._custom_max_h
        )

        if use_custom:
            assert self._custom_interp is not None
            T, p, rh = self._custom_interp(h)
            T += self.delta_t
        else:
            T, p, rho_dry, c = self._isa_state_dry(h)
            rh = self.relative_humidity

        # 湿度虚温修正。
        es = _saturated_vapor_pressure(T)
        e = rh * es
        # 防止除零或负分母。
        denom = p - (1.0 - EPSILON_VAPOR) * e
        if denom <= 0.0:
            q = 0.0
        else:
            q = EPSILON_VAPOR * e / denom
        q = float(np.clip(q, 0.0, 1.0))
        Tv = _virtual_temperature(T, q)
        rho = p / (R_AIR * Tv) * self.density_factor
        c = np.sqrt(GAMMA_AIR * R_AIR * Tv)

        # Sutherland 公式计算动力粘度。
        mu = 1.458e-6 * T**1.5 / (T + 110.4)

        return AtmosphereState(
            h=h,
            H=_geopotential(h),
            T=T,
            p=p,
            rho=float(rho),
            c=float(c),
            mu=float(mu),
            q=q,
        )

    def _apply_upper_extension(self, h: float, state: AtmosphereState) -> AtmosphereState:
        """对 86 km 以上做指数延拓，保证连续单调衰减到零。"""
        dh = max(h - _H_GEOM_MAX, 0.0)
        T = self._T86 * np.exp(-dh / self._H_T)
        p = self._p86 * np.exp(-dh / self._H_p)
        rho = self._rho86 * np.exp(-dh / self._H_rho)
        c = np.sqrt(GAMMA_AIR * R_AIR * T)
        mu = 1.458e-6 * T**1.5 / (T + 110.4)
        return AtmosphereState(
            h=h,
            H=_geopotential(min(h, _H_GEOM_MAX)),
            T=float(T),
            p=float(p),
            rho=float(rho),
            c=float(c),
            mu=float(mu),
            q=0.0,
        )

    def __call__(self, h: float) -> AtmosphereState:
        """查询高度 h 处的大气状态。"""
        if h <= _H_GEOM_MAX:
            return self._state(h)
        # 86 km 以上：以干 ISA 结果为基础做指数延拓。
        base = self._isa_state_dry(_H_GEOM_MAX)
        state86 = AtmosphereState(
            h=_H_GEOM_MAX,
            H=_H_GEOPOT_MAX,
            T=base[0],
            p=base[1],
            rho=base[2],
            c=base[3],
            mu=1.458e-6 * base[0] ** 1.5 / (base[0] + 110.4),
            q=0.0,
        )
        return self._apply_upper_extension(h, state86)

    def density(self, h: float) -> float:
        return float(self(h).rho)

    def pressure(self, h: float) -> float:
        return float(self(h).p)

    def temperature(self, h: float) -> float:
        return float(self(h).T)

    def sound_speed(self, h: float) -> float:
        return float(self(h).c)


def make_atmosphere(
    model: str = "isa",
    **kwargs,
) -> AtmosphereModel:
    """工厂函数，按名称创建大气模型。

    Parameters
    ----------
    model:
        ``"isa"`` 为标准大气，``"none"`` / ``"null"`` 为真空模型。
    **kwargs:
        传递给 ``StandardAtmosphere`` 的构造参数。
    """
    name = model.lower()
    if name in ("none", "null", "vacuum"):
        return NullAtmosphere()
    if name in ("isa", "us76", "standard"):
        return StandardAtmosphere(**kwargs)
    raise ValueError(f"未知大气模型: {model}")


def parse_profile(text: str) -> np.ndarray:
    """从文本解析自定义大气廓线。

    格式::

        # h(m) T(K) p(hPa) RH(0-1)
        0     288  1013   0.5
        1000  282  900    0.4
    """
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            h = float(parts[0])
            T = float(parts[1])
            p = float(parts[2])
            if p < 2000.0:  # 假设输入的是 hPa
                p *= 100.0
            rh = float(parts[3])
            rows.append([h, T, p, rh])
    if len(rows) < 2:
        raise ValueError("自定义廓线至少需要 2 个数据点")
    return np.array(rows, dtype=float)
