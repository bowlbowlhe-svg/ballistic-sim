r"""统一风场模型。

来源与差异
----------
- 主体继承自 ``D:\Pycharm\游戏与兴趣\外弹道方程\ballistics\wind.py``，
  保留 UniformWind、LogarithmicWind、PowerLawWind、ProfileWind、CompositeWind、
  DrydenGust 等实现。
- 增加 ``WindField3D`` 三维风场封装，可基于高度廓线叠加水平空间变化。
- GRIB2 气象数据接入通过 ``ballistic_sim.utils.optional_imports.try_import``
  懒导入 ``cfgrib``，缺失时提示安装 ``ballistic_sim[weather]`` extra；模块顶层
  **不直接 import cfgrib**，避免核心包硬依赖气象库。

接口约定
--------
- 风矢量统一采用 ENU（东-北-上）坐标系，单位 m/s。
- ``WindModel.__call__(h, s=None)`` 返回 ``WindState``；``s`` 为可选的沿轨飞行
  距离（m），供空间相关湍流模型索引。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Union, runtime_checkable

import numpy as np
from scipy.interpolate import PchipInterpolator

from ballistic_sim.utils.optional_imports import try_import


@dataclass(frozen=True)
class WindState:
    """某高度的风状态 (ENU, m/s)。"""

    e: float
    n: float
    u: float

    @property
    def vector(self) -> np.ndarray:
        """返回 (3,) numpy 数组。"""
        return np.array([self.e, self.n, self.u], dtype=float)

    def __array__(self):
        """支持 numpy 转换，使 np.array([state, ...]) 得到 (N, 3) 数组。"""
        return np.array([self.e, self.n, self.u], dtype=float)


@runtime_checkable
class WindModel(Protocol):
    """风场模型统一接口。"""

    def __call__(self, h: float, s: Optional[float] = None) -> WindState:
        """返回高度 h（可选沿轨距离 s）处的风。"""
        ...


class UniformWind:
    """均匀风场。"""

    def __init__(
        self,
        e: float = 0.0,
        n: float = 0.0,
        u: float = 0.0,
        gradient_e: float = 0.0,
    ):
        self.e = float(e)
        self.n = float(n)
        self.u = float(u)
        self.gradient_e = float(gradient_e)

    def __call__(self, h: float, s: Optional[float] = None) -> WindState:
        return WindState(e=self.e + self.gradient_e * h, n=self.n, u=self.u)


class LogarithmicWind:
    """对数风廓线（中性大气）。"""

    KAPPA = 0.41

    def __init__(
        self,
        u_ref: float,
        z_ref: float = 10.0,
        z0: float = 0.03,
        direction_deg: float = 0.0,
    ):
        self.u_ref = float(u_ref)
        self.z_ref = float(z_ref)
        self.z0 = float(z0)
        self.direction = float(np.deg2rad(direction_deg))
        self.u_star = u_ref * self.KAPPA / np.log(z_ref / z0)

    def __call__(self, h: float, s: Optional[float] = None) -> WindState:
        h_eff = max(h, self.z0 + 0.01)
        u = self.u_star / self.KAPPA * np.log(h_eff / self.z0)
        e = u * np.sin(self.direction)
        n = u * np.cos(self.direction)
        return WindState(e=e, n=n, u=0.0)


class PowerLawWind:
    """幂律风廓线。"""

    def __init__(
        self,
        u_ref: float,
        z_ref: float = 10.0,
        alpha: float = 0.14,
        direction_deg: float = 0.0,
    ):
        self.u_ref = float(u_ref)
        self.z_ref = float(z_ref)
        self.alpha = float(alpha)
        self.direction = float(np.deg2rad(direction_deg))

    def __call__(self, h: float, s: Optional[float] = None) -> WindState:
        u = self.u_ref * (max(h, 0.1) / self.z_ref) ** self.alpha
        e = u * np.sin(self.direction)
        n = u * np.cos(self.direction)
        return WindState(e=e, n=n, u=0.0)


class ProfileWind:
    """基于用户探空数据的风场。"""

    def __init__(self, profile: np.ndarray):
        """
        Parameters
        ----------
        profile:
            形状 (N, 4): [h(m), windE(m/s), windN(m/s), windU(m/s)]。
        """
        if len(profile) < 2:
            raise ValueError("风场廓线至少需要 2 个点")
        h = np.asarray(profile[:, 0], dtype=float)
        self._interp_e = PchipInterpolator(h, np.asarray(profile[:, 1], dtype=float))
        self._interp_n = PchipInterpolator(h, np.asarray(profile[:, 2], dtype=float))
        self._interp_u = PchipInterpolator(h, np.asarray(profile[:, 3], dtype=float))
        self._h_max = float(h[-1])

    def __call__(self, h: float, s: Optional[float] = None) -> WindState:
        h = float(np.clip(h, 0.0, self._h_max))
        return WindState(
            e=float(self._interp_e(h)),
            n=float(self._interp_n(h)),
            u=float(self._interp_u(h)),
        )


class CompositeWind:
    """组合风场：底层用边界层模型，高层用探空数据。"""

    def __init__(
        self,
        boundary_layer: WindModel,
        profile: Optional[ProfileWind] = None,
        transition_h: float = 1000.0,
        blend: float = 200.0,
    ):
        self.boundary_layer = boundary_layer
        self.profile = profile
        self.transition_h = float(transition_h)
        self.blend = float(blend)

    def __call__(self, h: float, s: Optional[float] = None) -> WindState:
        if self.profile is None or h <= self.transition_h:
            return self.boundary_layer(h, s)
        if h <= self.transition_h + self.blend:
            w = (h - self.transition_h) / self.blend
            b = self.boundary_layer(h, s)
            p = self.profile(h, s)
            return WindState(
                e=(1.0 - w) * b.e + w * p.e,
                n=(1.0 - w) * b.n + w * p.n,
                u=(1.0 - w) * b.u + w * p.u,
            )
        return self.profile(h, s)


class DrydenGust:
    """Dryden 湍流阵风模型（简化版，按沿轨飞行距离索引）。

    预生成一段沿距离空间相关的湍流序列（一阶低通近似 Dryden 谱），调用时以沿轨
    飞行距离 ``s``（自发射点起的水平射程，m）索引。若未提供 ``s``，退回按高度
    ``h`` 索引（兜底行为）。
    """

    def __init__(
        self,
        base_wind: WindModel,
        sigma_u: float = 1.0,
        sigma_v: float = 1.0,
        sigma_w: float = 0.5,
        length_u: float = 530.0,
        length_v: float = 530.0,
        length_w: float = 340.0,
        seed: Optional[int] = None,
        max_distance: float = 200000.0,
    ):
        self.base_wind = base_wind
        self.sigma = np.array([sigma_u, sigma_v, sigma_w], dtype=float)
        self.length = np.array([length_u, length_v, length_w], dtype=float)
        self.rng = np.random.default_rng(seed)
        self._distances = np.linspace(0.0, max_distance, int(max_distance) + 1)
        self._gust = self._generate()

    def _generate(self) -> np.ndarray:
        """生成空间相关湍流序列。

        采用一阶自回归近似 Dryden 谱：g[k] = a·g[k-1] + b·w[k]，其中 a 由相关
        长度决定，b = sigma·sqrt(1-a²)，保证稳态方差为 sigma²。
        """
        n = len(self._distances)
        gust = np.zeros((n, 3), dtype=float)
        dt = float(self._distances[1] - self._distances[0])
        for i in range(3):
            white = self.rng.normal(0.0, 1.0, n)
            tau = self.length[i]
            if tau <= 0.0:
                gust[:, i] = white * self.sigma[i]
                continue
            alpha = dt / (dt + tau)
            a = max(0.0, min(1.0 - alpha, 1.0))
            b = self.sigma[i] * np.sqrt(max(0.0, 1.0 - a * a))
            filtered = np.zeros(n, dtype=float)
            # 从零初始条件出发，避免初始随机大值在强相关序列中长时间漂移。
            filtered[0] = b * white[0]
            for k in range(1, n):
                filtered[k] = a * filtered[k - 1] + b * white[k]
            gust[:, i] = filtered
        # 对有限样本做去均值处理，保证样本均值接近 0（简化 Dryden 模型的工程处理）。
        gust -= gust.mean(axis=0)
        return gust

    def __call__(self, h: float, s: Optional[float] = None) -> WindState:
        key = h if s is None else s
        idx = int(np.clip(key, 0.0, self._distances[-1]))
        idx = min(idx, len(self._distances) - 1)
        base = self.base_wind(h, s)
        g = self._gust[idx]
        return WindState(e=base.e + g[0], n=base.n + g[1], u=base.u + g[2])


# ==============================================================================
# 三维风场与 GRIB2 接入
# ==============================================================================


class WindField3D:
    """三维风场封装。

    在 ``WindModel`` 高度廓线基础上，叠加水平空间相关的修正项。修正项由用户提供的
    回调 ``spatial_correction(x, y, h) -> WindState`` 给出；未提供时退化为纯高度
    廓线。

    坐标约定：x 为下程（东向为正），y 为侧偏（北向为正），h 为几何高度。
    """

    def __init__(
        self,
        base: WindModel,
        spatial_correction: Optional[Callable[[float, float, float], WindState]] = None,
    ):
        self.base = base
        self.spatial_correction = spatial_correction

    def __call__(
        self,
        h: float,
        s: Optional[float] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
    ) -> WindState:
        base = self.base(h, s)
        if self.spatial_correction is None or x is None or y is None:
            return base
        corr = self.spatial_correction(x, y, h)
        return WindState(e=base.e + corr.e, n=base.n + corr.n, u=base.u + corr.u)


class GRIB2WindModel:
    """GRIB2 风场模型（懒加载 cfgrib）。

    构造时仅保存文件路径与变量名；首次调用 ``__call__`` 时才通过导入守卫加载
    ``cfgrib`` / ``xarray``，避免核心包硬依赖气象库。
    """

    def __init__(
        self,
        path: Union[str, Path],
        u_var: str = "u",
        v_var: str = "v",
        level_dim: str = "isobaricInhPa",
    ):
        self.path = Path(path)
        self.u_var = u_var
        self.v_var = v_var
        self.level_dim = level_dim
        self._ds: Optional[Any] = None

    def _load(self) -> Any:
        if self._ds is None:
            xr = try_import("xarray", "weather")
            try_import("cfgrib", "weather")  # 注册 cfgrib 后端
            self._ds = xr.open_dataset(self.path, engine="cfgrib")
        return self._ds

    def __call__(self, h: float, s: Optional[float] = None) -> WindState:
        ds = self._load()
        # 简化实现：按最近高度层返回水平风，竖直风设为 0。
        # 真实实现可扩展为 lat/lon/level 三线性插值。
        levels = np.asarray(ds[self.level_dim], dtype=float) * 100.0  # hPa -> Pa
        level_m = self._pressure_to_height(levels)
        idx = int(np.argmin(np.abs(level_m - h)))
        u = float(ds[self.u_var].isel({self.level_dim: idx}).values)
        v = float(ds[self.v_var].isel({self.level_dim: idx}).values)
        return WindState(e=u, n=v, u=0.0)

    @staticmethod
    def _pressure_to_height(p: np.ndarray) -> np.ndarray:
        """粗略压高公式估算高度（仅用于 GRIB2 层级索引）。"""
        p0 = 101325.0
        return 44330.0 * (1.0 - (p / p0) ** 0.1903)


# ==============================================================================
# 工厂函数与解析
# ==============================================================================


def make_wind(model: str = "uniform", **kwargs) -> WindModel:
    """风场模型工厂函数。

    Parameters
    ----------
    model:
        ``"uniform"``、``"log"``、``"power"``、``"profile"``、``"composite"``、
        ``"dryden"``、``"grib2"``、``"none"``。
    """
    name = model.lower()
    if name in ("none", "null"):
        return UniformWind(e=0.0, n=0.0, u=0.0)
    if name == "uniform":
        return UniformWind(**kwargs)
    if name == "log":
        return LogarithmicWind(**kwargs)
    if name == "power":
        return PowerLawWind(**kwargs)
    if name == "profile":
        return ProfileWind(**kwargs)
    if name == "composite":
        return CompositeWind(**kwargs)
    if name == "dryden":
        return DrydenGust(**kwargs)
    if name == "grib2":
        return GRIB2WindModel(**kwargs)
    raise ValueError(f"未知风场模型: {model}")


def parse_wind_profile(text: str) -> np.ndarray:
    """解析风场廓线文本。

    格式::

        # h(m) windE(m/s) windN(m/s) windU(m/s)
        0     0   0   0
        1000  5   0   0
    """
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
    if len(rows) < 2:
        raise ValueError("风场廓线至少需要 2 个点")
    return np.array(rows, dtype=float)
