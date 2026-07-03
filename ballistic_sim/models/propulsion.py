"""火箭推进模型（从 lvsim 迁移）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ballistic_sim.constants import G0_STANDARD, P0_ISA
from ballistic_sim.models.atmosphere import AtmosphereModel


@dataclass
class PropulsionModel:
    """单级/等效级推进模型。"""

    thrust_sl_N: float
    thrust_vac_N: float
    isp_vac_s: float
    Aref_m2: float = 1.0

    @classmethod
    def from_stage(cls, stage: Dict[str, Any]) -> "PropulsionModel":
        return cls(
            thrust_sl_N=float(stage["thrust_sl"]),
            thrust_vac_N=float(stage["thrust_vac"]),
            isp_vac_s=float(stage["isp_vac"]),
            Aref_m2=float(stage.get("Aref", 1.0)),
        )

    @property
    def exit_area(self) -> float:
        return (self.thrust_vac_N - self.thrust_sl_N) / P0_ISA

    @property
    def mdot(self) -> float:
        return self.thrust_vac_N / (self.isp_vac_s * G0_STANDARD)

    def thrust_at_altitude(self, h_m: float, atmosphere: AtmosphereModel) -> float:
        p_a = atmosphere(max(h_m, 0.0)).p
        return self.thrust_vac_N - p_a * self.exit_area
