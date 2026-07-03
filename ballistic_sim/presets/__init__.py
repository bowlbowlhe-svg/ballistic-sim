"""参数预设库。"""

from __future__ import annotations

from ballistic_sim.presets.loader import list_missiles, list_projectiles, list_rockets
from ballistic_sim.presets.missiles import missile_config, missile_phases
from ballistic_sim.presets.projectiles import (
    m107_config,
    m107_phases,
    projectile_phases,
)
from ballistic_sim.presets.rockets import cz2f_config, cz2f_phases, rocket_config, rocket_phases

__all__ = [
    "m107_config",
    "m107_phases",
    "projectile_phases",
    "cz2f_config",
    "cz2f_phases",
    "rocket_config",
    "rocket_phases",
    "missile_config",
    "missile_phases",
    "list_projectiles",
    "list_missiles",
    "list_rockets",
]
