"""无动力滑行段 Phase。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np

from ballistic_sim.phases.base import Phase
from ballistic_sim.phases.events import make_apogee_event, make_ground_event


@dataclass
class CoastingPhase(Phase):
    """无动力滑行/滑翔阶段。

    结束事件默认包含：远地点、落地保护。动力学模块应关闭推力。
    """

    def __post_init__(self):
        if not self.events:
            self.events = [
                make_apogee_event(frame=self.native_frame()),
                make_ground_event(frame=self.native_frame()),
            ]

    def state_dim(self) -> int:
        return self.dynamics.state_dim()

    def native_frame(self) -> str:
        return self.dynamics.native_frame()

    def process_events(self, sol) -> List[dict]:
        records = super().process_events(sol)
        for rec in records:
            if rec["event_index"] == 0:
                rec["name"] = "远地点"
            elif rec["event_index"] == 1:
                rec["name"] = "落地保护"
        return records
