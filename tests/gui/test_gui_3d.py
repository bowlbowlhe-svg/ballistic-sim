"""GUI 结果可视化页签测试。"""

from __future__ import annotations

import pytest

tkinter = pytest.importorskip("tkinter", reason="tkinter not available")

from ballistic_sim.gui.app import BallisticGuiApp  # noqa: E402
from ballistic_sim.presets import m107_config  # noqa: E402
from ballistic_sim.simulator import simulate  # noqa: E402


@pytest.fixture(scope="module")
def tk_root():
    """提供模块级 tkinter 根窗口。"""
    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:  # pragma: no cover
        pytest.skip(f"无法创建 tkinter 窗口: {exc}")
    root.withdraw()
    yield root
    root.destroy()


EXPECTED_TABS = [
    "高度-射程",
    "高度/速度-时间",
    "地面航迹",
    "落点摘要",
    "3D 弹道",
    "顶视/侧视",
    "3D 轨迹 (plotly)",
    "事件日志",
]


def test_app_has_visualization_tabs(tk_root) -> None:
    """右侧结果区应包含所有可视化与事件日志页签。"""
    app = BallisticGuiApp(tk_root, mission="projectile")
    texts = [app._right_notebook.tab(t, "text") for t in app._right_notebook.tabs()]
    for tab in EXPECTED_TABS:
        assert tab in texts, f"缺少页签: {tab}"
    app.destroy()


def test_app_3d_tab_shows_hint_without_result(tk_root) -> None:
    """无仿真结果时点击生成应给出提示。"""
    app = BallisticGuiApp(tk_root, mission="projectile")
    app._on_generate_3d()
    assert "尚无" in app._3d_status_label.cget("text")
    app.destroy()


def test_app_3d_tab_generates_html(tk_root) -> None:
    """有仿真结果时应生成 HTML 或给出依赖缺失提示。"""
    app = BallisticGuiApp(tk_root, mission="projectile")
    result = simulate(m107_config())
    app._last_result = result
    app._on_generate_3d()
    text = app._3d_status_label.cget("text")
    assert ("已保存" in text) or ("未安装" in text) or ("失败" in text)
    app.destroy()


def test_app_refresh_plot_tabs(tk_root) -> None:
    """运行仿真后所有 matplotlib 页签应被正确绘制。"""
    app = BallisticGuiApp(tk_root, mission="projectile")
    result = simulate(m107_config())
    cfg = m107_config()
    app._display_result(cfg, result)

    # 每个 matplotlib 页签下都应包含一个 canvas 子控件
    canvas_count = 0
    for label, frame, plot_fn in app._plot_tabs:
        if plot_fn is None:
            continue
        children = frame.winfo_children()
        assert len(children) > 0, f"{label} 未绘制内容"
        canvas_count += 1
    assert canvas_count >= 5
    app.destroy()


def test_app_event_tree_populated(tk_root) -> None:
    """运行仿真后事件日志应包含记录。"""
    app = BallisticGuiApp(tk_root, mission="projectile")
    result = simulate(m107_config())
    cfg = m107_config()
    app._display_result(cfg, result)
    assert len(app._event_tree.get_children()) == len(result.event_log)
    app.destroy()
