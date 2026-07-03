"""Phase 抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


@dataclass
class PhaseContext:
    """RHS 调用时的上下文对象。

    把配置、当前 phase、环境模型、制导律等打包，避免 RHS 签名过度膨胀。
    """

    cfg: Any
    phase: "Phase"
    t0: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Phase(ABC):
    """积分阶段抽象基类。

    每个 Phase 描述一段连续的动力学演化，由 ``simulator`` 按列表顺序调用
    ``solve_ivp`` 积分，并在段间通过 ``project_state`` 做状态映射。

    Attributes
    ----------
    name:
        阶段名称（用于事件日志）。
    t_span:
        本阶段的绝对时间区间 ``(t_start, t_end_max)``；``t_end_max`` 为兜底上限。
    dynamics:
        本阶段使用的动力学模块（或工厂函数），需满足 ``rhs(t, y, ctx)`` 接口。
    events:
        本阶段监听的 scipy 事件函数列表（已设置 ``terminal`` / ``direction``）。
    is_terminal:
        本阶段是否为最终阶段（命中/落地/轨道插入）。
    guidance:
        可选的制导律对象/字典；由具体 dynamics 消费。
    """

    name: str
    t_span: Tuple[float, float]
    dynamics: Any
    events: List[Callable[[float, np.ndarray], float]] = field(default_factory=list)
    is_terminal: bool = False
    guidance: Any = None

    def rhs(self, t: float, y: np.ndarray, cfg: Any, phase: "Phase") -> np.ndarray:
        """返回本阶段右端项 ``dY/dt``。

        默认实现把 ``phase`` 自身作为 ``PhaseContext`` 传给 ``self.dynamics.rhs``。
        子类可覆盖以注入阶段专用逻辑（如质量阶跃、制导装订）。
        """
        ctx = PhaseContext(cfg=cfg, phase=phase, t0=self.t_span[0])
        return self.dynamics.rhs(t, y, ctx)

    def process_events(self, sol: Any) -> List[Dict[str, Any]]:
        """解析 ``solve_ivp`` 返回的事件触发信息。

        返回事件记录列表，每项至少含 ``name`` / ``t``；具体 phase 可补充更多字段。
        """
        records: List[Dict[str, Any]] = []
        if not hasattr(sol, "t_events") or sol.t_events is None:
            return records
        for i, times in enumerate(sol.t_events):
            if times is None:
                continue
            for t_ev in times:
                records.append(
                    {
                        "phase": self.name,
                        "event_index": i,
                        "t": float(t_ev),
                    }
                )
        return records

    def final_state(self, sol: Any) -> np.ndarray:
        """返回本阶段末状态（默认取 ``sol.y[:, -1]`` 的拷贝）。"""
        return np.asarray(sol.y[:, -1], dtype=float).copy()

    @abstractmethod
    def state_dim(self) -> int:
        """返回本阶段状态向量维度。"""
        ...

    @abstractmethod
    def native_frame(self) -> str:
        """返回本阶段原生坐标系名称（``ECI`` / ``ENU``）。"""
        ...
