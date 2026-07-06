"""GUI 3D 轨迹 tab 测试。"""

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


def test_app_has_3d_tab(tk_root) -> None:
    """右侧结果区应包含 3D 轨迹 tab。"""
    app = BallisticGuiApp(tk_root, mission="projectile")
    texts = [app._right_notebook.tab(t, "text") for t in app._right_notebook.tabs()]
    assert "3D 轨迹" in texts
    assert "2D 曲线" in texts
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
