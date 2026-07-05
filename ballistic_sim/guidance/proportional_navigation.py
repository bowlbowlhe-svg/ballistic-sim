"""真比例导引 (TPN) 与广义比例导引 (GPN) 实现。

比例导引律输入相对位置/速度，输出加速度指令矢量（ENU/ECI 通用）。
``ProNavGuidance`` 类同时提供 ``direction`` 接口，便于 ``PoweredECIDynamics`` 装配。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import numpy as np

from ballistic_sim.frames import geodetic_to_ecef

__all__ = [
    "ProNavGuidance",
    "pro_nav_acceleration",
    "make_static_target_provider",
]

_EPS = 1e-12


def pro_nav_acceleration(
    r_rel: Any,
    v_rel: Any,
    nav_constant: float = 3.0,
    mode: str = "true",
) -> np.ndarray:
    """比例导引加速度指令。

    Parameters
    ----------
    r_rel:
        弹目相对位置 (目标 - 弹) 三维矢量。
    v_rel:
        弹目相对速度 (目标速度 - 弹速度) 三维矢量。
    nav_constant:
        有效导航比 N'，通常取 3~5。
    mode:
        ``true`` 为真比例导引，加速度垂直于视线；
        ``generalized`` 为广义比例导引，加速度垂直于导弹速度。

    Returns
    -------
    np.ndarray
        三维加速度指令矢量 (与输入坐标系一致)。
    """
    r = np.asarray(r_rel, dtype=float).reshape(3)
    v = np.asarray(v_rel, dtype=float).reshape(3)
    r_mag = float(np.linalg.norm(r))
    if r_mag < _EPS:
        return np.zeros(3, dtype=float)

    los = r / r_mag  # 视线单位矢量
    v_close = -float(np.dot(v, los))  # 接近速度，正值表示靠近

    # 视线角速度矢量：d(los)/dt = (v - los * dot(v, los)) / r_mag
    # 等价于 cross(los, v) / r_mag 的垂直分量形式
    los_rate = (v - los * np.dot(v, los)) / r_mag

    if mode == "true":
        # TPN：指令垂直于视线
        a_cmd = nav_constant * v_close * los_rate
    elif mode == "generalized":
        # GPN：先按视线角速度计算，再投影到垂直于导弹速度平面
        a_gpn = nav_constant * v_close * los_rate
        v_mag = float(np.linalg.norm(v))
        if v_mag < _EPS:
            a_cmd = a_gpn
        else:
            # 保持垂直于速度分量
            a_cmd = a_gpn - np.dot(a_gpn, v) * v / (v_mag * v_mag)
    else:
        raise ValueError(f"未知的比例导引模式: {mode}")

    if not np.all(np.isfinite(a_cmd)):
        return np.zeros(3, dtype=float)
    return a_cmd


def make_static_target_provider(
    lat_deg: float,
    lon_deg: float,
    alt_m: float = 0.0,
) -> Callable[[float], Tuple[np.ndarray, np.ndarray]]:
    """构造静态地面目标状态提供者 (r_eci, v_eci)。"""
    r_ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_m)

    def provider(t: float) -> Tuple[np.ndarray, np.ndarray]:
        from ballistic_sim.frames import ecef_to_eci

        r_eci = ecef_to_eci(r_ecef, float(t))
        return r_eci, np.zeros(3, dtype=float)

    return provider


@dataclass
class ProNavGuidance:
    """比例导引制导律封装。

    可通过 ``target_provider`` 提供运动目标，也可直接调用
    ``acceleration(r_own, v_own, r_target, v_target)``。
    """

    nav_constant: float = 3.0
    max_accel_m_s2: float = 100.0
    mode: str = "true"
    target_provider: Optional[Callable[[float], Tuple[np.ndarray, np.ndarray]]] = None
    failed: bool = field(default=False, init=False)

    def set_target(
        self,
        lat_deg: Optional[float] = None,
        lon_deg: Optional[float] = None,
        alt_m: float = 0.0,
        target_provider: Optional[Callable[[float], Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> "ProNavGuidance":
        """设置静态目标或外部目标提供者，返回 self 以支持链式调用。"""
        if target_provider is not None:
            self.target_provider = target_provider
        elif lat_deg is not None and lon_deg is not None:
            self.target_provider = make_static_target_provider(lat_deg, lon_deg, alt_m)
        return self

    def acceleration(
        self,
        r_own: Any,
        v_own: Any,
        r_target: Optional[Any] = None,
        v_target: Optional[Any] = None,
    ) -> np.ndarray:
        """由弹目状态计算加速度指令。"""
        r_o = np.asarray(r_own, dtype=float).reshape(3)
        v_o = np.asarray(v_own, dtype=float).reshape(3)

        if r_target is not None and v_target is not None:
            r_t = np.asarray(r_target, dtype=float).reshape(3)
            v_t = np.asarray(v_target, dtype=float).reshape(3)
        elif self.target_provider is not None:
            r_t, v_t = self.target_provider(0.0)
        else:
            self.failed = True
            return np.zeros(3, dtype=float)

        r_rel = r_t - r_o
        v_rel = v_t - v_o
        a_cmd = pro_nav_acceleration(r_rel, v_rel, self.nav_constant, self.mode)
        a_mag = float(np.linalg.norm(a_cmd))
        if a_mag > self.max_accel_m_s2:
            a_cmd = a_cmd * (self.max_accel_m_s2 / a_mag)
        self.failed = not np.all(np.isfinite(a_cmd))
        return a_cmd

    def direction(
        self,
        t: float,
        r: Any,
        v: Any,
        m: Optional[float] = None,
    ) -> np.ndarray:
        """供 ``PoweredECIDynamics`` 调用的推力方向接口。

        将加速度指令归一化为单位矢量；若未设置目标则返回零矢量
        （调用方应回退到开环）。
        """
        if self.target_provider is not None:
            r_t, v_t = self.target_provider(float(t))
        else:
            self.failed = True
            return np.zeros(3, dtype=float)
        a_cmd = self.acceleration(r, v, r_t, v_t)
        a_mag = float(np.linalg.norm(a_cmd))
        if a_mag < _EPS:
            self.failed = True
            return np.zeros(3, dtype=float)
        return a_cmd / a_mag

    def set_max_accel(self, max_accel_m_s2: float) -> "ProNavGuidance":
        """设置最大可用加速度。"""
        self.max_accel_m_s2 = float(max_accel_m_s2)
        return self

    def set_mode(self, mode: str) -> "ProNavGuidance":
        """设置导引模式 (true/generalized)。"""
        if mode not in ("true", "generalized"):
            raise ValueError(f"模式必须是 true 或 generalized，得到 {mode}")
        self.mode = mode
        return self

    def set_nav_constant(self, nav_constant: float) -> "ProNavGuidance":
        """设置导航常数。"""
        self.nav_constant = float(nav_constant)
        return self
