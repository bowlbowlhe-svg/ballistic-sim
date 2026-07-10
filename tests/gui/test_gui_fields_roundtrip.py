"""GUI 表单字段读写一致性测试。

重点覆盖 ``VehicleConfig.stages`` 这种 ``Optional[List[BaseModel]]`` 类型，
确保在可视化界面中加载、编辑、回读不会丢失或错位。
"""

from __future__ import annotations

import tkinter as tk

import pytest

tkinter = pytest.importorskip("tkinter", reason="tkinter not available")

from ballistic_sim.config import StageConfig, VehicleConfig  # noqa: E402
from ballistic_sim.gui.fields import (  # noqa: E402
    build_form,
    read_model_variables,
    update_model_variables,
)


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


def _roundtrip_vehicle(tk_root, vehicle: VehicleConfig) -> VehicleConfig:
    """对 VehicleConfig 走一遍表单构建 -> 写入值 -> 读取值 -> 重建模型。"""
    frame = tk.Frame(tk_root)
    variables = build_form(frame, VehicleConfig)
    update_model_variables(variables, vehicle)
    data = read_model_variables(VehicleConfig, variables)
    return VehicleConfig(**data)


def test_vehicle_stages_roundtrip(tk_root) -> None:
    """多级 stage 列表在表单中应能正确序列化与反序列化。"""
    stages = [
        StageConfig(
            name="S1",
            thrust_sl=1000.0,
            thrust_vac=1200.0,
            isp_vac=300.0,
            m_prop=5000.0,
            m_dry=1000.0,
            diameter_m=1.0,
        ),
        StageConfig(
            name="S2",
            thrust_sl=0.0,
            thrust_vac=500.0,
            isp_vac=350.0,
            m_prop=2000.0,
            m_dry=800.0,
        ),
    ]
    original = VehicleConfig(
        mass_kg=10000.0,
        diameter_m=1.0,
        cd=0.3,
        stages=stages,
    )
    restored = _roundtrip_vehicle(tk_root, original)
    assert restored.stages is not None
    assert len(restored.stages) == len(original.stages)
    for orig, back in zip(original.stages, restored.stages):
        assert back.name == orig.name
        assert back.thrust_sl == pytest.approx(orig.thrust_sl)
        assert back.thrust_vac == pytest.approx(orig.thrust_vac)
        assert back.isp_vac == pytest.approx(orig.isp_vac)
        assert back.m_prop == pytest.approx(orig.m_prop)
        assert back.m_dry == pytest.approx(orig.m_dry)


def test_vehicle_no_stages_roundtrip(tk_root) -> None:
    """无 stage 时回读应为 None。"""
    original = VehicleConfig(mass_kg=10.0, diameter_m=0.1, cd=0.3, stages=None)
    restored = _roundtrip_vehicle(tk_root, original)
    assert restored.stages is None
