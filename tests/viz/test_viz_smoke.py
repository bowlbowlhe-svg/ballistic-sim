"""Visualization smoke tests for stage 4.

Each ``ballistic_sim.viz.*`` plotting function must accept a minimal
``SimResult`` and return a non-empty matplotlib ``Figure`` without raising.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import matplotlib
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg", force=True)

from ballistic_sim.phases.builder import build_phases  # noqa: E402
from ballistic_sim.presets import m107_config, rocket_full_config  # noqa: E402
from ballistic_sim.simulator import SimResult, simulate  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_plot_fn(module: Any) -> Callable[..., Figure]:
    """Locate a plausible plotting function inside a viz module.

    The exact naming convention is not fixed yet, so we accept either a
    generic ``plot`` export or ``plot_<module_name>``.
    """
    if hasattr(module, "plot") and callable(module.plot):
        return module.plot

    module_name = module.__name__.split(".")[-1]
    candidates = (
        f"plot_{module_name}",
        "plot",
        module_name,
    )
    for name in candidates:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    pytest.skip(f"No plotting function found in {module.__name__}")


def _atmospheric_result() -> SimResult:
    """Return a minimal atmospheric SimResult (M107 projectile)."""
    return simulate(m107_config(), phases=[])


def _exo_result() -> SimResult:
    """Return a minimal exo-atmospheric SimResult (CZ-2F rocket)."""
    cfg = rocket_full_config("CZ2F")
    return simulate(cfg, phases=build_phases(cfg))


_VIZ_CASES: list[tuple[str, Callable[[], SimResult]]] = [
    ("ballistic_sim.viz.trajectory3d", _atmospheric_result),
    ("ballistic_sim.viz.groundtrack", _atmospheric_result),
    ("ballistic_sim.viz.profile", _atmospheric_result),
    ("ballistic_sim.viz.wind_plot", _atmospheric_result),
    ("ballistic_sim.viz.globe3d", _exo_result),
    ("ballistic_sim.viz.orbit_insertion", _exo_result),
    ("ballistic_sim.viz.dvbudget", _exo_result),
    ("ballistic_sim.viz.reentry", _exo_result),
]


@pytest.mark.parametrize("module_name,make_result", _VIZ_CASES)
def test_viz_smoke(module_name: str, make_result: Callable[[], SimResult]) -> None:
    """Smoke test for ``module_name``: Figure returned, no exception."""
    mod = pytest.importorskip(module_name, reason=f"{module_name} not implemented")
    plot_fn = _resolve_plot_fn(mod)

    result = make_result()
    assert result.t.size > 0
    assert result.y.size > 0

    fig = plot_fn(result)
    assert isinstance(fig, Figure), f"Expected matplotlib Figure, got {type(fig)}"
    assert fig.get_axes(), "Figure has no axes"

    import matplotlib.pyplot as plt

    plt.close(fig)
