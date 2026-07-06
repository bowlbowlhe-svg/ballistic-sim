"""监控面板可视化测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from matplotlib.figure import Figure

from ballistic_sim.presets import m107_config
from ballistic_sim.simulator import SimResult, simulate
from ballistic_sim.viz.dashboard import plot_dashboard


def test_dashboard_returns_figure() -> None:
    """plot_dashboard 应返回非空 matplotlib Figure。"""
    result = simulate(m107_config(), phases=[])
    fig = plot_dashboard(result)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4


def test_dashboard_writes_file(tmp_path: Path) -> None:
    """提供 output_path 时应写出非空 PNG。"""
    result = simulate(m107_config(), phases=[])
    path = tmp_path / "dashboard.png"
    fig = plot_dashboard(result, output_path=path)
    assert isinstance(fig, Figure)
    assert path.exists()
    assert path.stat().st_size > 0


def test_dashboard_with_rocket_result() -> None:
    """ECI 火箭结果也应正常生成 2×2 面板。"""
    from ballistic_sim.phases.builder import build_phases
    from ballistic_sim.presets import rocket_full_config

    cfg = rocket_full_config("CZ2F")
    result = simulate(cfg, phases=build_phases(cfg))
    fig = plot_dashboard(result)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4


def test_dashboard_output_path_as_str(tmp_path: Path) -> None:
    """output_path 支持字符串路径。"""
    result = simulate(m107_config(), phases=[])
    path = str(tmp_path / "dashboard_str.png")
    fig = plot_dashboard(result, output_path=path)
    assert isinstance(fig, Figure)
    assert Path(path).exists()


def test_dashboard_empty_result_raises() -> None:
    """空结果应抛出 ValueError。"""
    empty = SimResult()
    with pytest.raises(ValueError, match="SimResult 为空"):
        plot_dashboard(empty)
