"""弹道导弹完整任务链冒烟测试。"""

from __future__ import annotations

import pytest

from ballistic_sim.frames import eci_to_ecef, ecef_to_geodetic, haversine_distance
from ballistic_sim.phases.builder import build_phases
from ballistic_sim.presets.missiles import (
    list_missiles,
    missile_full_config,
)
from ballistic_sim.simulator import simulate


@pytest.mark.slow
def test_icbm_8000_full_chain_smoke() -> None:
    """ICBM_8000 完整链可完成 3 级上升 -> 中段 -> 再入 -> 落地。"""
    cfg = missile_full_config("ICBM_8000")
    phases = build_phases(cfg)
    names = [ph.name for ph in phases]
    assert names[:3] == ["S1 动力", "S2 动力", "S3 动力"]
    assert "滑行" in names
    assert "再入段" in names
    assert "终点" in names

    res = simulate(cfg, phases)
    assert res.stop_reason == "completed"
    assert res.t.size > 0

    event_names = [ev.get("name", "") for ev in res.event_log]
    assert any("S1 分离" in name for name in event_names)
    assert any("S2 分离" in name for name in event_names)
    assert any("S3 分离" in name for name in event_names)
    assert any("再入" in name for name in event_names)
    assert any("落地" in name for name in event_names)

    # 飞行时间约 30~40 分钟
    assert 1500.0 < res.t[-1] < 3600.0

    # 落点距离约 7000~9500 km
    y_end = res.y[-1]
    r_ecef = eci_to_ecef(y_end[0:3], res.t[-1])
    lat, lon, _ = ecef_to_geodetic(r_ecef)
    downrange = haversine_distance(cfg.launch.lat_deg, cfg.launch.lon_deg, lat, lon)
    assert 7e6 < downrange < 10e6


def test_missile_full_config_lists_available_presets() -> None:
    """list_missiles 应包含 ICBM_8000。"""
    names = list_missiles()
    assert "ICBM_8000" in names
    assert "SRBM_600" in names
