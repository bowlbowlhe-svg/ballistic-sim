"""动力上升段 Phase。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ballistic_sim.phases.base import Phase
from ballistic_sim.phases.events import make_burnout_event, make_ground_event


@dataclass
class PoweredPhase(Phase):
    """动力上升/推进阶段。

    结束事件默认包含：推进剂耗尽、落地保护。支持质量阶跃到 ``m_after_separation``。
    """

    m_dry: float = 0.0
    m_after_separation: Optional[float] = None
    sep_name: str = "分离"

    def __post_init__(self):
        if not self.events:
            self.events = [
                make_burnout_event(self.m_dry),
                make_ground_event(frame=self.native_frame()),
            ]

    def state_dim(self) -> int:
        return self.dynamics.state_dim()

    def native_frame(self) -> str:
        return self.dynamics.native_frame()

    def process_events(self, sol) -> List[Dict[str, Any]]:
        records = super().process_events(sol)
        for rec in records:
            if rec["event_index"] == 0:
                rec["name"] = f"{self.sep_name}(燃尽)"
            elif rec["event_index"] == 1:
                rec["name"] = "落地保护"
        return records

    def final_state(self, sol: OdeResult) -> np.ndarray:
        y = super().final_state(sol)
        if self.m_after_separation is not None:
            y = y.copy()
            y[6] = float(self.m_after_separation)
        return y
