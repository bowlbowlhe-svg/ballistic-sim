"""参数预设库。"""

from __future__ import annotations

from ballistic_sim.presets.loader import list_missiles, list_projectiles, list_rockets
from ballistic_sim.presets.missiles import (
    missile_config,
    missile_full_config,
)
from ballistic_sim.presets.projectiles import m107_config
from ballistic_sim.presets.rockets import (
    cz2f_config,
    rocket_config,
    rocket_full_config,
    rocket_stages,
)

__all__ = [
    "m107_config",
    "cz2f_config",
    "rocket_config",
    "rocket_full_config",
    "rocket_stages",
    "missile_config",
    "missile_full_config",
    "list_projectiles",
    "list_missiles",
    "list_rockets",
]
