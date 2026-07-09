"""Ballistic Sim — 6-DOF ballistic flight simulation package."""

from ballistic_sim.config import (
    PerturbationConfig,
    SimConfig,
    apply_overrides,
    load_config,
    save_config,
    validate_config,
)
from ballistic_sim.phases.builder import build_phases
from ballistic_sim.simulator import SimResult, simulate

__version__ = "0.6.2"
__all__ = [
    "__version__",
    "SimConfig",
    "simulate",
    "build_phases",
    "validate_config",
    "load_config",
    "save_config",
    "apply_overrides",
    "PerturbationConfig",
    "SimResult",
]


def __dir__() -> list[str]:
    """Return only the curated public API names."""
    return list(__all__)
