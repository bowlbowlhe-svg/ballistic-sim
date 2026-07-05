"""交互式 3D 可视化测试。"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

from ballistic_sim.presets import cz2f_config, cz2f_phases, m107_config
from ballistic_sim.simulator import SimResult, simulate


def _block_plotly_import(monkeypatch) -> None:
    """临时阻止 plotly 相关导入，用于测试缺失依赖提示。"""
    for key in list(sys.modules.keys()):
        if key == "plotly" or key.startswith("plotly."):
            monkeypatch.delitem(sys.modules, key, raising=False)

    real_import = builtins.__import__

    def blocked_import(name: str, *args, **kwargs):
        if name == "plotly" or name.startswith("plotly."):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)


def test_import_guard_friendly_error_when_plotly_missing(monkeypatch) -> None:
    """plotly 缺失时应抛出带 [viz3d] 安装提示的 ImportError。"""
    _block_plotly_import(monkeypatch)
    from ballistic_sim.viz import interactive3d

    with pytest.raises(ImportError, match="pip install ballistic_sim\\[viz3d\\]"):
        interactive3d._require_plotly()


plotly = pytest.importorskip("plotly", reason="plotly not installed")

from ballistic_sim.viz.interactive3d import (  # noqa: E402
    demo,
    plot_trajectory_3d,
)


def _rocket_result() -> SimResult:
    """返回一条 CZ-2F 火箭示例轨迹。"""
    cfg = cz2f_config()
    return simulate(cfg, phases=cz2f_phases(cfg))


def test_plot_trajectory_3d_returns_figure_for_eci() -> None:
    """ECI 结果应返回 plotly Figure。"""
    result = _rocket_result()
    fig = plot_trajectory_3d(result)
    assert isinstance(fig, plotly.graph_objects.Figure)
    assert len(fig.data) >= 3


def test_plot_trajectory_3d_returns_figure_for_enu() -> None:
    """ENU 结果也应返回 plotly Figure。"""
    from ballistic_sim.viz import attach_launch_lla

    result = simulate(m107_config(), phases=[])
    attach_launch_lla(result, 0.0, 0.0, 0.0)
    fig = plot_trajectory_3d(result)
    assert isinstance(fig, plotly.graph_objects.Figure)


def test_plot_trajectory_3d_writes_html(tmp_path: Path) -> None:
    """提供 output_path 时应写出非空 HTML 文件。"""
    result = _rocket_result()
    path = tmp_path / "trajectory3d.html"
    fig = plot_trajectory_3d(result, output_path=path)
    assert isinstance(fig, plotly.graph_objects.Figure)
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_trajectory_3d_empty_result_raises() -> None:
    """空结果应抛出 ValueError。"""
    empty = SimResult()
    with pytest.raises(ValueError, match="SimResult 为空"):
        plot_trajectory_3d(empty)


def test_demo_writes_html(tmp_path: Path) -> None:
    """demo() 应生成示例 HTML。"""
    path = tmp_path / "demo.html"
    fig = demo(output_path=path)
    assert isinstance(fig, plotly.graph_objects.Figure)
    assert path.exists()
    assert path.stat().st_size > 0
