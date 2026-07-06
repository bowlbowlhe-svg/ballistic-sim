"""Δv budget 可视化单元测试。"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure

matplotlib.use("Agg")

from ballistic_sim.phases.builder import build_phases  # noqa: E402
from ballistic_sim.presets import m107_config, rocket_full_config  # noqa: E402
from ballistic_sim.simulator import simulate  # noqa: E402
from ballistic_sim.viz.dvbudget import plot_dv_budget  # noqa: E402


def test_dv_budget_figure_for_rocket() -> None:
    """plot_dv_budget 对火箭结果返回非空 Figure。"""
    cfg = rocket_full_config("CZ2F")
    result = simulate(cfg, phases=build_phases(cfg))
    fig = plot_dv_budget(result, isp_s=300.0)
    assert isinstance(fig, Figure)
    assert fig.get_axes()
    matplotlib.pyplot.close(fig)


def test_dv_budget_figure_for_projectile() -> None:
    """plot_dv_budget 对弹丸结果返回非空 Figure（质量缺失走退化分支）。"""
    result = simulate(m107_config(), phases=[])
    fig = plot_dv_budget(result, isp_s=300.0)
    assert isinstance(fig, Figure)
    assert fig.get_axes()
    matplotlib.pyplot.close(fig)


def test_dv_budget_ideal_delta_v_formula() -> None:
    """Tsiolkovsky 公式对典型质量比给出正确 Δv。"""
    from ballistic_sim.viz.dvbudget import _delta_v_ideal

    isp = 300.0
    m0 = 1000.0
    mdry = 250.0
    expected = isp * 9.80665 * np.log(m0 / mdry)
    assert _delta_v_ideal(0.0, m0, mdry, isp) == pytest.approx(expected, rel=1e-6)


def test_dv_budget_zero_when_m0_leq_mdry() -> None:
    """初始质量不大于干质量时理想 Δv 为 0。"""
    from ballistic_sim.viz.dvbudget import _delta_v_ideal

    assert _delta_v_ideal(0.0, 100.0, 100.0, 300.0) == 0.0
    assert _delta_v_ideal(0.0, 50.0, 100.0, 300.0) == 0.0
