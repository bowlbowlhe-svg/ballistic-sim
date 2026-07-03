"""Preset factory tests to boost coverage of ballistic_sim.presets."""

from __future__ import annotations

from ballistic_sim.config import SimConfig
from ballistic_sim.presets import (
    missile_config,
    missile_phases,
    m107_config,
    m107_phases,
    projectile_phases,
    rocket_config,
    rocket_phases,
)


def test_m107_config_returns_simconfig() -> None:
    """M107 preset returns a valid SimConfig."""
    cfg = m107_config()
    assert isinstance(cfg, SimConfig)
    assert cfg.mission == "projectile"


def test_m107_phases_non_empty() -> None:
    """M107 phase list contains the expected phases."""
    phases = m107_phases()
    assert len(phases) >= 2
    assert phases[0].name == "无动力弹道"


def test_projectile_phases_for_m107() -> None:
    """projectile_phases('M107') returns same shape as m107_phases."""
    phases = projectile_phases("M107")
    assert len(phases) == len(m107_phases())


def test_missile_config_returns_simconfig() -> None:
    """Missile preset returns a valid SimConfig."""
    cfg = missile_config("SRBM_600")
    assert isinstance(cfg, SimConfig)
    assert cfg.mission == "missile"


def test_missile_phases_non_empty() -> None:
    """Missile phase list is non-empty."""
    phases = missile_phases("SRBM_600")
    assert len(phases) >= 1


def test_rocket_config_returns_simconfig() -> None:
    """Rocket preset returns a valid SimConfig."""
    cfg = rocket_config("CZ2F")
    assert isinstance(cfg, SimConfig)
    assert cfg.mission == "rocket"


def test_rocket_phases_non_empty() -> None:
    """Rocket phase list contains multiple powered/coast phases."""
    cfg = rocket_config("CZ2F")
    phases = rocket_phases(cfg)
    assert len(phases) >= 3
