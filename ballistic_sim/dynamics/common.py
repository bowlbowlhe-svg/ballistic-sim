"""共享 RHS 工具与上下文。"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

from ballistic_sim.constants import GM_EARTH, OMEGA_EARTH, WGS84_A
from ballistic_sim.models.atmosphere import AtmosphereModel, AtmosphereState
from ballistic_sim.models.wind import WindModel, WindState


@dataclass
class AeroEnv:
    """某高度的气动环境。"""

    rho: float
    c: float
    p: float
    T: float
    q: float = 0.0
    Ma: float = 0.0


class LRUDict:
    """基于 ``collections.OrderedDict`` 的轻量 LRU 字典。"""

    def __init__(self, maxsize: int = 10000):
        self._maxsize = max(maxsize, 1)
        self._data: OrderedDict[Any, Any] = OrderedDict()

    def get(self, key: Any) -> Any:
        """读取 key；命中时移至最近使用端。"""
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: Any, value: Any) -> None:
        """写入 key 并移至最近使用端；超容时淘汰最久未用项。"""
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        """清空字典。"""
        self._data.clear()


class ModelCache:
    """大气、风、气动系数插值结果缓存。

    对高度/马赫数等查询做轻量级缓存，避免 RHS 与 telemetry 中重复构造插值器
    或重复查表。默认启用，可通过 ``enabled=False`` 关闭。

    线程安全：本实现仅保证单线程 / 单进程安全；多线程写入需外部加锁。
    """

    def __init__(self, enabled: bool = True, maxsize: int = 10000):
        self._enabled = bool(enabled)
        self._maxsize = max(int(maxsize), 1)
        self._atm: LRUDict = LRUDict(self._maxsize)
        self._wind: LRUDict = LRUDict(self._maxsize)
        self._aero: LRUDict = LRUDict(self._maxsize)
        self._coeff: LRUDict = LRUDict(self._maxsize)
        self._generic: LRUDict = LRUDict(self._maxsize)

    @property
    def enabled(self) -> bool:
        """缓存是否启用。"""
        return self._enabled

    def clear(self) -> None:
        """清空所有缓存。"""
        self._atm.clear()
        self._wind.clear()
        self._aero.clear()
        self._coeff.clear()
        self._generic.clear()

    def get(self, key: Any) -> Any:
        """读取通用缓存项。"""
        return self._generic.get(key)

    def set(self, key: Any, value: Any) -> None:
        """写入通用缓存项。"""
        self._generic.put(key, value)

    def atmosphere(
        self,
        h: float,
        model: Callable[[float], AtmosphereState],
    ) -> AtmosphereState:
        """查询并缓存大气状态（按高度）。"""
        if not self._enabled:
            return model(h)
        hk = round(float(h), 6)
        state = self._atm.get(hk)
        if state is None:
            state = model(h)
            self._atm.put(hk, state)
        return state

    def wind(
        self,
        h: float,
        model: Callable[..., WindState],
        s: Optional[float] = None,
    ) -> WindState:
        """查询并缓存风场状态（按高度，可选按沿轨距离）。"""
        if not self._enabled:
            return model(h) if s is None else model(h, s)
        hk = round(float(h), 6)
        sk: Optional[float] = None if s is None else round(float(s), 3)
        key = (hk, sk)
        state = self._wind.get(key)
        if state is None:
            state = model(h) if s is None else model(h, s)
            self._wind.put(key, state)
        return state

    def aero_coefficient(
        self,
        Ma: float,
        alpha: float,
        model: Callable[[float, float], float],
    ) -> float:
        """查询并缓存气动系数（按马赫数与攻角）。"""
        if not self._enabled:
            return float(model(Ma, alpha))
        key = (round(float(Ma), 6), round(float(alpha), 6))
        value = self._aero.get(key)
        if value is None:
            value = float(model(Ma, alpha))
            self._aero.put(key, value)
        return value

    def coefficient(
        self,
        name: str,
        x: float,
        model: Callable[[float], float],
    ) -> float:
        """查询并缓存一维气动系数表（按名称与自变量）。"""
        if not self._enabled:
            return float(model(x))
        key = (name, round(float(x), 6))
        value = self._coeff.get(key)
        if value is None:
            value = float(model(x))
            self._coeff.put(key, value)
        return value

    def stats(self) -> Dict[str, int]:
        """返回当前各缓存桶条目数。"""
        return {
            "atmosphere": len(self._atm),
            "wind": len(self._wind),
            "aero": len(self._aero),
            "coeff": len(self._coeff),
            "generic": len(self._generic),
        }


@dataclass
class _CachedAtmosphere:
    """包装 ``AtmosphereModel``，使其查询结果进入 ``ModelCache``。"""

    cache: ModelCache
    model: AtmosphereModel

    def __call__(self, h: float) -> AtmosphereState:
        return self.cache.atmosphere(h, self.model)

    def density(self, h: float) -> float:
        return float(self(h).rho)

    def pressure(self, h: float) -> float:
        return float(self(h).p)

    def temperature(self, h: float) -> float:
        return float(self(h).T)

    def sound_speed(self, h: float) -> float:
        return float(self(h).c)


@dataclass
class _CachedWind:
    """包装 ``WindModel``，使其查询结果进入 ``ModelCache``。"""

    cache: ModelCache
    model: WindModel

    def __call__(self, h: float, s: Optional[float] = None) -> WindState:
        return self.cache.wind(h, self.model, s=s)


@dataclass
class DynamicContext:
    """RHS 内部使用的环境/模型上下文。"""

    cfg: Any
    atmosphere: AtmosphereModel
    wind: WindModel
    aero: Any
    terrain: Optional[Any] = None
    gravity_model: str = "j2"
    options: Dict[str, bool] = None  # type: ignore[assignment]
    use_cache: bool = True

    def __post_init__(self):
        if self.options is None:
            self.options = {}
        # ``options`` 中可显式覆盖缓存开关
        if "use_cache" in self.options:
            self.use_cache = bool(self.options["use_cache"])

        maxsize = 10000
        if self.cfg is not None:
            cfg_options = getattr(self.cfg, "options", None)
            if cfg_options is not None:
                maxsize = int(getattr(cfg_options, "cache_maxsize", maxsize))
        if self.options is not None and "cache_maxsize" in self.options:
            maxsize = int(self.options["cache_maxsize"])
        self.cache = ModelCache(enabled=self.use_cache, maxsize=maxsize)
        # 保留原始模型引用，便于外部需要时访问
        self._atmosphere_model = self.atmosphere
        self._wind_model = self.wind
        self._aero_model = self.aero

        if self.use_cache:
            self.atmosphere = _CachedAtmosphere(self.cache, self._atmosphere_model)
            self.wind = _CachedWind(self.cache, self._wind_model)

    def clear_cache(self) -> None:
        """清空本上下文关联的模型缓存。"""
        self.cache.clear()


def dynamic_pressure(rho: float, v_rel: float) -> float:
    """动压 q = 0.5 * rho * v_rel^2 (Pa)。"""
    return 0.5 * float(rho) * float(v_rel) * float(v_rel)


def mach_number(v_rel: float, c: float) -> float:
    """马赫数 Ma = |v_rel| / c。"""
    return float(v_rel) / float(c) if float(c) > 0.0 else 0.0


def compute_alpha(
    v_rel: np.ndarray,
    body_axis: np.ndarray,
    max_deg: float = 90.0,
) -> float:
    """攻角 (rad)：相对速度与弹体纵轴夹角。"""
    v = np.asarray(v_rel, dtype=float).reshape(3)
    b = np.asarray(body_axis, dtype=float).reshape(3)
    vn = float(np.linalg.norm(v))
    bn = float(np.linalg.norm(b))
    if vn < 1e-12 or bn < 1e-12:
        return 0.0
    cos_a = float(np.clip(np.dot(v, b) / (vn * bn), -1.0, 1.0))
    return float(np.clip(np.arccos(cos_a), -np.deg2rad(max_deg), np.deg2rad(max_deg)))


def relative_velocity_eci(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
    wind_enu: Optional[np.ndarray] = None,
    lat_deg: float = 0.0,
    lon_deg: float = 0.0,
) -> np.ndarray:
    """ECI 相对气流速度 = v_eci - omega×r - wind_eci。"""
    from ballistic_sim.frames import enu_to_ecef_vec, ecef_to_eci

    v_rel = np.asarray(v_eci, dtype=float).copy()
    omega = np.array([0.0, 0.0, OMEGA_EARTH], dtype=float)
    v_rel -= np.cross(omega, np.asarray(r_eci, dtype=float))
    if wind_enu is not None and np.linalg.norm(wind_enu) > 0.0:
        w_ecef = enu_to_ecef_vec(wind_enu, lat_deg, lon_deg)
        v_rel -= ecef_to_eci(w_ecef, 0.0)
    return v_rel


def rv_to_oe(r: np.ndarray, v: np.ndarray) -> Dict[str, float]:
    """由 ECI 位置速度计算轨道根数（简化）。"""
    r = np.asarray(r, dtype=float).reshape(3)
    v = np.asarray(v, dtype=float).reshape(3)
    mu = GM_EARTH
    rm = float(np.linalg.norm(r))
    vm = float(np.linalg.norm(v))
    h_vec = np.cross(r, v)
    h = float(np.linalg.norm(h_vec))
    e_vec = ((vm * vm - mu / rm) * r - np.dot(r, v) * v) / mu
    e = float(np.linalg.norm(e_vec))
    a = 0.0
    if abs(e - 1.0) > 1e-12:
        a = -mu / (2.0 * (0.5 * vm * vm - mu / rm))
    else:
        a = h * h / mu
    i = float(np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))) if h > 0.0 else 0.0
    return {
        "a": float(a),
        "e": float(e),
        "i_rad": float(i),
        "i_deg": float(np.rad2deg(i)),
        "h_peri_km": float((a * (1.0 - e) - WGS84_A) / 1e3) if a != 0.0 else 0.0,
        "h_apo_km": float((a * (1.0 + e) - WGS84_A) / 1e3) if a != 0.0 else 0.0,
    }
