"""GUI 应用层配置读写测试。

覆盖每种 mission / preset 组合在可视化表单中的加载与回读，
确保点击 Run 前不会触发 ``VehicleConfig.stages`` 等嵌套列表字段的解析错误。
"""

from __future__ import annotations

import pytest

tkinter = pytest.importorskip("tkinter", reason="tkinter not available")

from ballistic_sim.gui.app import BallisticGuiApp  # noqa: E402
from ballistic_sim.gui.builder import MISSIONS, list_presets  # noqa: E402


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


def _all_mission_presets() -> list[tuple[str, str | None]]:
    cases: list[tuple[str, str | None]] = []
    for mission in MISSIONS:
        presets = list_presets(mission)
        if presets:
            cases.extend((mission, p) for p in presets)
        else:
            cases.append((mission, None))
    return cases


@pytest.mark.parametrize("mission,preset", _all_mission_presets())
def test_app_read_config_from_forms(tk_root, mission: str, preset: str | None) -> None:
    """所有 mission / preset 组合都能通过 GUI 表单回读出合法 SimConfig。"""
    app = BallisticGuiApp(tk_root, mission=mission)
    try:
        if preset is not None:
            app._current_preset = preset
            app._preset_var.set(preset)
            app._load_config_to_forms()

        cfg = app._read_config_from_forms()
        assert cfg.mission == mission
    finally:
        app.destroy()
