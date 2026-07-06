"""Preset factory tests to boost coverage of ballistic_sim.presets."""

from __future__ import annotations

from ballistic_sim.config import SimConfig
from ballistic_sim.phases.builder import build_phases
from ballistic_sim.presets import (
    missile_config,
    missile_full_config,
    m107_config,
    rocket_config,
    rocket_full_config,
)


def test_m107_config_returns_simconfig() -> None:
    """M107 preset returns a valid SimConfig."""
    cfg = m107_config()
    assert isinstance(cfg, SimConfig)
    assert cfg.mission == "projectile"


def test_m107_build_phases_non_empty() -> None:
    """build_phases(m107_config()) returns the expected phases."""
    phases = build_phases(m107_config())
    assert len(phases) >= 2
    assert phases[0].name == "无动力弹道"


def test_m107_build_phases_non_trivial() -> None:
    """build_phases(m107_config()) returns a non-trivial phase list."""
    phases = build_phases(m107_config())
    assert len(phases) >= 2


def test_missile_config_returns_simconfig() -> None:
    """Missile preset returns a valid SimConfig."""
    cfg = missile_config("SRBM_600")
    assert isinstance(cfg, SimConfig)
    assert cfg.mission == "missile"


def test_missile_full_config_phases_non_empty() -> None:
    """missile_full_config + build_phases returns a non-empty phase list."""
    cfg = missile_full_config("SRBM_600")
    phases = build_phases(cfg)
    assert len(phases) >= 1


def test_rocket_config_returns_simconfig() -> None:
    """Rocket preset returns a valid SimConfig."""
    cfg = rocket_config("CZ2F")
    assert isinstance(cfg, SimConfig)
    assert cfg.mission == "rocket"


def test_rocket_full_config_phases_non_empty() -> None:
    """rocket_full_config + build_phases returns a multistage phase list."""
    cfg = rocket_full_config("CZ2F")
    phases = build_phases(cfg)
    assert len(phases) >= 3
