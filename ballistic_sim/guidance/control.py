"""6-DOF 控制律接口。

提供鸭舵 (canard) 与脉冲发动机 (RCS/pulse) 控制力矩接口，
并显式禁止从 3-DOF 自动升维到 6-DOF。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

__all__ = [
    "SixDOFControl",
    "ControlMoment",
    "NoAutoDimensionUpgrade",
]


class NoAutoDimensionUpgrade(RuntimeError):
    """禁止 3-DOF -> 6-DOF 自动升维。"""


@dataclass
class ControlMoment:
    """体轴系控制力矩 (N·m)。"""

    L: float = 0.0  # 滚转
    M: float = 0.0  # 俯仰
    N: float = 0.0  # 偏航

    def vector(self) -> np.ndarray:
        return np.array([self.L, self.M, self.N], dtype=float)


@dataclass
class SixDOFControl:
    """6-DOF 控制律参数与接口。

    默认策略：
    - 鸭舵产生俯仰/偏航力矩，比例于攻角/侧滑角误差；
    - 脉冲发动机产生绕任意轴的脉冲力矩，用于姿态快速修正；
    - 必须显式提供初始姿态 (quaternion) 与角速度，否则禁止启用 6-DOF。
    """

    canard_gain: float = 1.0
    pulse_magnitude: float = 100.0
    pulse_deadband_rad: float = 0.05
    options: Dict[str, Any] = field(default_factory=dict)

    def require_initial_attitude(self, y: np.ndarray) -> None:
        """检查状态向量是否包含显式 6-DOF 姿态；否则禁止启用。"""
        y = np.asarray(y, dtype=float)
        if y.size < 13:
            raise NoAutoDimensionUpgrade(
                "6-DOF 控制禁止从 3-DOF 自动升维：须显式提供 13 维状态 (r, v, q, omega)。"
            )
        q = y[6:10]
        if float(np.linalg.norm(q)) < 1e-6:
            raise NoAutoDimensionUpgrade("6-DOF 控制需要显式初始姿态四元数；零四元数视为未提供。")

    def canard_moment(
        self,
        alpha_rad: float,
        beta_rad: float,
        pqr: Optional[np.ndarray] = None,
    ) -> ControlMoment:
        """鸭舵力矩：比例于攻角/侧滑角误差。"""
        g = self.canard_gain
        return ControlMoment(
            L=0.0,
            M=-g * float(alpha_rad),
            N=g * float(beta_rad),
        )

    def pulse_moment(self, attitude_error: np.ndarray) -> ControlMoment:
        """脉冲发动机力矩：在姿态误差超过死区时输出固定幅值脉冲。"""
        err = np.asarray(attitude_error, dtype=float).reshape(3)
        mag = float(np.linalg.norm(err))
        if mag < self.pulse_deadband_rad:
            return ControlMoment()
        direction = err / mag
        vec = self.pulse_magnitude * direction
        return ControlMoment(L=vec[0], M=vec[1], N=vec[2])

    def control_moment(
        self,
        y: np.ndarray,
        alpha_rad: float = 0.0,
        beta_rad: float = 0.0,
        attitude_error: Optional[np.ndarray] = None,
    ) -> ControlMoment:
        """综合控制力矩入口。"""
        self.require_initial_attitude(y)
        m_canard = self.canard_moment(alpha_rad, beta_rad)
        if attitude_error is not None:
            m_pulse = self.pulse_moment(attitude_error)
        else:
            m_pulse = ControlMoment()
        return ControlMoment(
            L=m_canard.L + m_pulse.L,
            M=m_canard.M + m_pulse.M,
            N=m_canard.N + m_pulse.N,
        )
