"""阶段驱动 (Phase) 子包。"""

from __future__ import annotations

from ballistic_sim.phases.base import Phase, PhaseContext
from ballistic_sim.phases.events import (
    make_apogee_event,
    make_burnout_event,
    make_fairing_event_h,
    make_fairing_event_q,
    make_ground_event,
    make_orbit_insertion_event,
    make_stage_separation_event,
    make_target_distance_event,
)
from ballistic_sim.phases.powered import PoweredPhase
from ballistic_sim.phases.coasting import CoastingPhase
from ballistic_sim.phases.reentry import ReentryPhase
from ballistic_sim.phases.terminal import TerminalPhase
from ballistic_sim.phases.builder import build_phases

__all__ = [
    "Phase",
    "PhaseContext",
    "PoweredPhase",
    "CoastingPhase",
    "ReentryPhase",
    "TerminalPhase",
    "build_phases",
    "make_apogee_event",
    "make_burnout_event",
    "make_fairing_event_h",
    "make_fairing_event_q",
    "make_ground_event",
    "make_orbit_insertion_event",
    "make_stage_separation_event",
    "make_target_distance_event",
]
