"""GUI 模块冒烟测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

tkinter = pytest.importorskip("tkinter", reason="tkinter not available")

from ballistic_sim.config import (  # noqa: E402
    EnvironmentConfig,
    LaunchConfig,
    VehicleConfig,
)
from ballistic_sim.gui import run_gui  # noqa: E402
from ballistic_sim.gui.app import BallisticGuiApp as AppClass  # noqa: E402
from ballistic_sim.gui.builder import (  # noqa: E402
    build_config_and_phases,
    default_preset,
    dump_config_to_yaml,
    list_presets,
    load_config_from_yaml,
)
from ballistic_sim.gui.fields import (  # noqa: E402
    build_form,
    read_model_variables,
    update_model_variables,
)
from ballistic_sim.gui.runner import SimulationRunner  # noqa: E402


@pytest.fixture(scope="module")
def tk_root():
    """提供模块级 tkinter 根窗口并在所有测试后销毁。"""
    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:  # pragma: no cover
        pytest.skip(f"无法创建 tkinter 窗口: {exc}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def fresh_frame(tk_root):
    """在模块根窗口中提供一个临时 Frame。"""
    frame = tkinter.Frame(tk_root)
    yield frame
    frame.destroy()


def test_import_gui_exports() -> None:
    """GUI 包应导出 run_gui 与 BallisticGuiApp。"""
    assert callable(run_gui)
    assert AppClass is not None


def test_fields_build_form_vehicle(fresh_frame) -> None:
    """build_form 应能为 VehicleConfig 生成变量字典。"""
    vars_ = build_form(fresh_frame, VehicleConfig)
    assert "mass_kg" in vars_
    assert "cd" in vars_
    assert vars_["mass_kg"].get() == pytest.approx(1.0)


def test_fields_read_and_update_model_variables(fresh_frame) -> None:
    """读取/更新 tkinter 变量应能往返模型数据。"""
    vars_ = build_form(fresh_frame, LaunchConfig)

    cfg = LaunchConfig(lat_deg=30.0, lon_deg=120.0, elevation_deg=60.0)
    update_model_variables(vars_, cfg)
    data = read_model_variables(LaunchConfig, vars_)
    assert data["lat_deg"] == pytest.approx(30.0)
    assert data["lon_deg"] == pytest.approx(120.0)
    assert data["elevation_deg"] == pytest.approx(60.0)


def test_fields_literal_renders_combobox(fresh_frame) -> None:
    """Literal 字段应渲染为 Combobox。"""
    vars_ = build_form(fresh_frame, EnvironmentConfig)
    wind_var = vars_["wind_model"]
    assert wind_var.get() == "uniform"


def test_app_constructs_for_all_missions(tk_root) -> None:
    """BallisticGuiApp 应对所有任务类型成功构造。"""
    for mission in ["projectile", "missile", "rocket", "icbm", "suborbital"]:
        app = AppClass(tk_root, mission=mission)
        assert app.current_mission == mission
        app.destroy()


def test_app_forms_populated_from_preset(tk_root) -> None:
    """构造应用后 vehicle mass 表单应反映 M107 预设。"""
    app = AppClass(tk_root, mission="projectile")
    vehicle_vars = app._form_vars["vehicle"]
    assert vehicle_vars["mass_kg"].get() > 40.0
    app.destroy()


def test_app_read_config_roundtrip(tk_root) -> None:
    """从表单读取的配置应与写入时等价。"""
    app = AppClass(tk_root, mission="projectile")
    cfg = app._read_config_from_forms()
    assert cfg.mission == "projectile"
    assert cfg.vehicle.mass_kg > 0
    app.destroy()


def test_builder_list_presets() -> None:
    """list_presets 应返回非空列表或空列表。"""
    assert "M107" in list_presets("projectile")
    assert default_preset("projectile") == "M107"


def test_builder_config_and_phases() -> None:
    """build_config_and_phases 应返回配置与阶段列表。"""
    cfg, phases = build_config_and_phases("projectile", "M107")
    assert cfg.mission == "projectile"
    assert len(phases) >= 1


def test_builder_yaml_roundtrip(tmp_path: Path) -> None:
    """YAML 保存/加载应能往返 SimConfig。"""
    cfg, _ = build_config_and_phases("projectile", "M107")
    path = tmp_path / "cfg.yaml"
    dump_config_to_yaml(cfg, path)
    loaded = load_config_from_yaml(path)
    assert loaded.mission == cfg.mission
    assert loaded.vehicle.mass_kg == pytest.approx(cfg.vehicle.mass_kg)


def test_runner_executes_simulation() -> None:
    """SimulationRunner 应在后台线程运行仿真并返回结果。"""
    cfg, _ = build_config_and_phases("projectile", "M107")
    runner = SimulationRunner()
    runner.run(cfg)
    runner.join(timeout=30.0)
    assert not runner.is_alive()
    kind, payload = runner.result_queue.get_nowait()
    assert kind == "ok"
    assert payload.y.shape[0] > 0


def test_cli_gui_flag_invokes_run_gui(monkeypatch) -> None:
    """``--gui`` 应调用 run_gui。"""
    calls = []

    def fake_run_gui() -> None:
        calls.append(1)

    monkeypatch.setattr(sys.modules["ballistic_sim.gui"], "run_gui", fake_run_gui)
    monkeypatch.setattr(sys, "argv", ["ballistic-sim", "--gui"])

    from ballistic_sim.cli import main

    main()
    assert len(calls) == 1
