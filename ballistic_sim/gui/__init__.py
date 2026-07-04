"""GUI 可视化入口。"""

from __future__ import annotations

from ballistic_sim.gui.app import BallisticGuiApp

__all__ = ["run_gui", "BallisticGuiApp"]


def run_gui() -> None:
    """启动弹道模拟 GUI。"""
    try:
        import tkinter as tk
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "GUI 需要 tkinter。请安装带 tkinter 的 Python 发行版。"
        ) from exc

    root = tk.Tk()
    app = BallisticGuiApp(root)
    app.pack(fill="both", expand=True)
    root.mainloop()
