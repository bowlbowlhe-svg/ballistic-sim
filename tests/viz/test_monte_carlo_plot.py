"""Monte Carlo 可视化测试."""

from __future__ import annotations

import numpy as np

from ballistic_sim.monte_carlo import DispersionResult
from ballistic_sim.viz import plot_dispersion


def test_plot_dispersion_does_not_raise() -> None:
    """plot_dispersion 对有效结果不抛异常."""
    rng = np.random.default_rng(0)
    ranges = rng.normal(1000.0, 10.0, 50)
    crosses = rng.normal(0.0, 5.0, 50)
    tofs = rng.normal(20.0, 0.5, 50)
    angles = rng.normal(45.0, 1.0, 50)
    result = DispersionResult(
        ranges=ranges,
        crosses=crosses,
        tofs=tofs,
        impact_angles=angles,
        range_mean=float(np.mean(ranges)),
        range_std=float(np.std(ranges)),
        cross_mean=float(np.mean(crosses)),
        cross_std=float(np.std(crosses)),
        cep50=1.0,
        cep90=2.0,
        ellipse_major=3.0,
        ellipse_minor=1.5,
        ellipse_angle=30.0,
        samples=50,
    )
    fig = plot_dispersion(result)
    assert fig is not None
    assert len(fig.axes) == 4
