"""再入段 Phase（ICBM）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


from ballistic_sim.phases.base import Phase
from ballistic_sim.phases.events import make_ground_event


@dataclass
class ReentryPhase(Phase):
    """再入段。

    可切换为高保真 MPM/6-DOF；默认仍用 3-DOF 质点。
    结束事件为落地。
    """

    fidelity: str = "point_mass"

    def __post_init__(self):
        if not self.events:
            self.events = [make_ground_event(frame=self.native_frame())]

    def state_dim(self) -> int:
        return self.dynamics.state_dim()

    def native_frame(self) -> str:
        return self.dynamics.native_frame()

    def process_events(self, sol) -> List[dict]:
        records = super().process_events(sol)
        for rec in records:
            rec["name"] = "再入落地"
        return records
