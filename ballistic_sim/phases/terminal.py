"""终止事件 Phase（落地/命中/轨道插入）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np

from ballistic_sim.phases.base import Phase
from ballistic_sim.phases.events import make_ground_event, make_orbit_insertion_event


@dataclass
class TerminalPhase(Phase):
    """终点事件阶段。

    不单独积分，仅作为 ``is_terminal`` 标记；实际终止由上一阶段的事件触发。
    若给出 ``target``，则生成轨道插入事件。
    """

    target: Any = None

    def __post_init__(self):
        self.is_terminal = True
        if not self.events:
            if self.target is not None:
                self.events = [make_orbit_insertion_event(self.target)]
            else:
                self.events = [make_ground_event(frame="ECI")]

    def state_dim(self) -> int:
        return self.dynamics.state_dim()

    def native_frame(self) -> str:
        return self.dynamics.native_frame()

    def process_events(self, sol) -> List[dict]:
        records = super().process_events(sol)
        for rec in records:
            rec["name"] = "终止事件"
        return records

    def rhs(self, t: float, y: np.ndarray, cfg: Any, phase: "Phase") -> np.ndarray:
        """终止阶段不演化。"""
        return np.zeros_like(y)
