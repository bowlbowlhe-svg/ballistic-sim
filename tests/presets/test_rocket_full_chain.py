"""运载火箭完整任务链冒烟测试。"""

from __future__ import annotations

import pytest

from ballistic_sim.dynamics.common import rv_to_oe
from ballistic_sim.presets.rockets import (
    list_rockets,
    rocket_full_chain,
    rocket_full_config,
)
from ballistic_sim.simulator import simulate


def _orbital_elements(res) -> dict:
    """从结果末态提取轨道根数。"""
    y = res.y[-1]
    return rv_to_oe(y[0:3], y[3:6])


def test_rocket_full_chain_lists_available_presets() -> None:
    """list_rockets 应包含常用火箭预设。"""
    names = list_rockets()
    assert "CZ2F" in names
    assert "CZ3B" in names
    assert "GENERIC3_3STAGE" in names


def test_rocket_full_config_returns_simconfig() -> None:
    """rocket_full_config 返回有效的 SimConfig 并含多级 stages。"""
    from ballistic_sim.config import SimConfig

    cfg = rocket_full_config("CZ2F")
    assert isinstance(cfg, SimConfig)
    assert cfg.vehicle.stages is not None
    assert len(cfg.vehicle.stages) >= 3


@pytest.mark.slow
def test_generic3_3stage_full_chain_smoke() -> None:
    """GENERIC3_3STAGE 完整链：4 级分离 + 滑行 + 远地点 + 入轨。"""
    name = "GENERIC3_3STAGE"
    cfg = rocket_full_config(name)
    phases = rocket_full_chain(name)
    names = [ph.name for ph in phases]
    assert any("P1" in n for n in names)
    assert any("P4" in n for n in names)
    assert "滑行" in names
    assert "轨道插入" in names

    res = simulate(cfg, phases)
    assert res.stop_reason == "completed"
    assert res.t.size > 0

    event_names = [ev.get("name", "") for ev in res.event_log]
    assert any("P1" in name and "燃尽" in name for name in event_names)
    assert any("P2" in name and "燃尽" in name for name in event_names)
    assert any("P3" in name and "燃尽" in name for name in event_names)
    assert any("P4" in name and "燃尽" in name for name in event_names)
    assert any("远地点" in name for name in event_names)

    oe = _orbital_elements(res)
    # 教学级近似参数：入轨为椭圆，近地点可略低于地表，远地点应显著大于 100 km
    assert oe["h_apo_km"] > 100.0
    assert oe["h_peri_km"] > -100.0
    # 轨道倾角接近目标（允许约 5° 偏差）
    assert abs(oe["i_deg"] - 42.0) < 5.0


@pytest.mark.slow
def test_cz3b_orbital_insertion_smoke() -> None:
    """CZ3B 完整链：助推/芯一/芯二/三级分离 + 轨道插入 sanity。"""
    name = "CZ3B"
    cfg = rocket_full_config(name)
    phases = rocket_full_chain(name)
    names = [ph.name for ph in phases]
    assert any("P1" in n for n in names)
    assert any("P4" in n for n in names)

    res = simulate(cfg, phases)
    assert res.stop_reason == "completed"

    event_names = [ev.get("name", "") for ev in res.event_log]
    assert any("P1" in name and "燃尽" in name for name in event_names)
    assert any("P2" in name and "燃尽" in name for name in event_names)
    assert any("P3" in name and "燃尽" in name for name in event_names)
    assert any("P4" in name and "燃尽" in name for name in event_names)

    oe = _orbital_elements(res)
    # 教学近似参数下轨道偏椭圆，但应为正高度轨道
    assert oe["h_peri_km"] > 100.0
    assert oe["h_apo_km"] > 100.0
    assert abs(oe["i_deg"] - 42.0) < 5.0
