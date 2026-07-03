"""两脉冲圆化入轨骨架（滑行 + 远地点切向脉冲）。

提供构建圆化任务 phase 序列的辅助函数，以及远地点圆化关机事件。
实际 PEG 圆化制导由 :mod:`ballistic_sim.guidance.peg` 实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ballistic_sim.constants import GM_EARTH, WGS84_A
from ballistic_sim.guidance.peg import (
    make_apogee_circularization_event,
    make_peg_state,
)

__all__ = [
    "CircPlan",
    "make_circ_plan",
    "build_two_impulse_phases",
]


@dataclass
class CircPlan:
    """两脉冲圆化计划参数。"""

    r_apo_circ_m: float
    transfer_peri_km: float
    circ_stage: Dict[str, Any]


def make_circ_plan(
    apo_km: float,
    peri_km: float,
    circ_stage: Optional[Dict[str, Any]] = None,
) -> CircPlan:
    """构造圆化计划。"""
    if circ_stage is None:
        circ_stage = {
            "name": "圆化级",
            "thrust_vac": 10.0e3,
            "thrust_sl": 10.0e3,
            "isp_vac": 320.0,
            "m_prop": 500.0,
            "m_dry": 1500.0,
            "Aref": 2.0,
        }
    return CircPlan(
        r_apo_circ_m=WGS84_A + apo_km * 1e3,
        transfer_peri_km=float(peri_km),
        circ_stage=dict(circ_stage),
    )


def build_two_impulse_phases(
    ascent_phases: List[Any],
    target: Dict[str, Any],
    circ_plan: CircPlan,
    base_guid: Dict[str, Any],
) -> List[Any]:
    """在两脉冲模式下扩展 phase 序列： ascent + 滑行 + 圆化。

    本函数为骨架：返回的 phase 列表可直接交 :func:`ballistic_sim.phases.builder`
    或上层 driver 使用。具体 Phase 对象由调用方根据自身动力学构造。
    """
    from ballistic_sim.phases.coasting import CoastingPhase
    from ballistic_sim.phases.powered import PoweredPhase

    phases = list(ascent_phases)

    # 滑行段：从第一脉冲关机点滑行到远地点
    coast_dyn = getattr(ascent_phases[-1], "dynamics", None)
    if coast_dyn is None:
        raise ValueError("ascent_phases 最后一个 phase 须携带 dynamics")

    guid_coast = dict(base_guid)
    guid_coast["phase"] = "coasting"
    coast = CoastingPhase(
        name="转移滑行",
        t_span=(0.0, 7200.0),
        dynamics=coast_dyn,
        guidance=guid_coast,
    )
    phases.append(coast)

    # 圆化段：远地点切向脉冲
    from ballistic_sim.dynamics.powered_eci import PoweredECIDynamics

    guid_circ = dict(base_guid)
    guid_circ["phase"] = "peg"
    guid_circ["peg_mode"] = "apogee_circ"
    guid_circ["t_us_start"] = 0.0
    guid_circ["gamma_end_deg"] = 0.0
    guid_circ["lat_deg"] = base_guid.get("lat_deg", 0.0)
    guid_circ["lon_deg"] = base_guid.get("lon_deg", 0.0)
    guid_circ["azimuth_deg"] = base_guid.get("azimuth_deg", 0.0)
    guid_circ["_peg_state"] = make_peg_state(
        target,
        peg_mode="apogee_circ",
        r_apo_circ_m=circ_plan.r_apo_circ_m,
    )
    guid_circ["_peg_stage"] = dict(circ_plan.circ_stage)

    dyn_circ = PoweredECIDynamics(
        stage=circ_plan.circ_stage,
        guidance=guid_circ,
        use_upperstage=True,
    )
    circ = PoweredPhase(
        name="远地点圆化",
        t_span=(0.0, 3600.0),
        dynamics=dyn_circ,
        guidance=guid_circ,
        m_dry=float(circ_plan.circ_stage["m_dry"]),
        sep_name="圆化关机",
    )
    # 追加 γ 过零关机事件
    circ.events.append(make_apogee_circularization_event(circ_plan.r_apo_circ_m))
    phases.append(circ)
    return phases
