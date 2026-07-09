"""ICBM 跨大气层端到端测试（阶段 3）。

使用多级 StageConfig 描述单级 ICBM，由 build_phases/simulate 自动生成
主动段 → 滑行段 → 再入段：
- 验证全段事件链（燃尽 / 远地点 / 出大气 / 再入 / 落地）；
- 验证全程能量在主动段单调增、无动力段单调降；
- 验证落点距离、再入速度、最大高度在合理量级。
"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.config import (
    EnvironmentConfig,
    GuidanceConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    StageConfig,
    VehicleConfig,
)
from ballistic_sim.constants import GM_EARTH, WGS84_A
from ballistic_sim.frames import ecef_to_geodetic, eci_to_ecef, haversine_distance
from ballistic_sim.simulator import simulate

H_ENTRY = 100e3  # 大气边界，用于出/再入事件


def _icbm_config() -> SimConfig:
    """返回多级 StageConfig 描述的 ICBM 配置，由 simulate(cfg) 自动构建 phase 链。"""
    return SimConfig(
        mission="icbm",
        vehicle=VehicleConfig(
            mass_kg=22000.0,
            diameter_m=0.5,
            cd=0.3,
            stages=[
                StageConfig(
                    name="ICBM-boost",
                    thrust_vac=1.2e6,
                    thrust_sl=1.0e6,
                    isp_vac=300.0,
                    m_prop=20000.0,
                    m_dry=0.0,
                    Aref=0.5,
                ),
            ],
        ),
        launch=LaunchConfig(
            lat_deg=0.0,
            lon_deg=0.0,
            alt_m=1.0,
            azimuth_deg=45.0,
            elevation_deg=90.0,
            v0_m_s=0.0,
        ),
        environment=EnvironmentConfig(
            atmosphere="isa",
            gravity_model="j2",
        ),
        guidance=GuidanceConfig(
            kick_deg=20.0,
            t_pitchover=10.0,
            t_kick_end=60.0,
            use_drag=True,
        ),
        options=OptionsConfig(
            integrator="DOP853",
            rtol=1e-6,
            atol=1e-9,
            max_step=1.0,
            terminate_impact=True,
        ),
    )


@pytest.mark.slow
def test_icbm_end_to_end() -> None:
    """ICBM：自动构建 phase 链，主动段→滑行段→再入段端到端检查。"""
    cfg = _icbm_config()
    result = simulate(cfg)

    assert result.stop_reason == "completed"
    assert result.y.size > 0
    assert len(result.t) > 2

    # ---- 事件链检查 ----
    # builder 生成的动力段 burnout 事件：phase 含 "动力"，name 含 "燃尽"。
    assert any("燃尽" in ev.get("name", "") and "动力" in ev["phase"] for ev in result.event_log)
    # 滑行段默认事件：0=远地点，2=再入（100 km 下降穿越）。
    # 事件 1 为 "出大气"（builder 中固定为 launch.alt_m+1 km 上升穿越），
    # 由于 burnout 高度已高于该阈值，本次仿真不会触发，故仅校验 0/2。
    assert any("滑行" in ev["phase"] and ev["event_index"] == 0 for ev in result.event_log)
    assert any("滑行" in ev["phase"] and ev["event_index"] == 2 for ev in result.event_log)
    assert any("再入段" in ev["phase"] for ev in result.event_log)

    # ---- 全程能量曲线 ----
    r = result.y[:, 0:3]
    v = result.y[:, 3:6]
    energy = 0.5 * np.sum(v * v, axis=1) - GM_EARTH / np.linalg.norm(r, axis=1)

    # 找到 burnout 时刻索引（动力段燃尽事件）
    burnout_ev = next(
        ev for ev in result.event_log if "燃尽" in ev.get("name", "") and "动力" in ev["phase"]
    )
    idx_boost = int(np.searchsorted(result.t, burnout_ev["t"]))

    # 主动段能量增加
    assert energy[idx_boost] > energy[0]
    assert np.all(np.diff(energy[: idx_boost + 1]) >= -1.0)  # 允许数值噪声

    # 无动力段能量单调不增（J2+气动耗散；允许数值噪声 1 kJ/kg）
    assert np.all(np.diff(energy[idx_boost:]) <= 1e3)
    assert energy[-1] < energy[idx_boost]

    # ---- 出/再入检查 ----
    alt = np.linalg.norm(r, axis=1) - WGS84_A
    exits = np.where((alt[:-1] < H_ENTRY) & (alt[1:] >= H_ENTRY))[0]
    entries = np.where((alt[:-1] > H_ENTRY) & (alt[1:] <= H_ENTRY))[0]
    assert exits.size >= 1, "未检测到出大气事件"
    assert entries.size >= 1, "未检测到再入事件"
    assert entries[0] > exits[0]

    # ---- 落点合理性 ----
    r_eci_end = result.y[-1, 0:3]
    r_ecef_end = eci_to_ecef(r_eci_end, float(result.t[-1]))
    lat_f, lon_f, _ = ecef_to_geodetic(r_ecef_end)
    range_m = haversine_distance(cfg.launch.lat_deg, cfg.launch.lon_deg, lat_f, lon_f)

    assert 1e5 < range_m < 5e6, f"射程 {range_m/1e3:.0f} km 不在合理 ICBM 区间"
    assert alt.max() > H_ENTRY, f"最大高度 {alt.max()/1e3:.0f} km 未出大气"
    assert alt.max() < 2e7, f"最大高度 {alt.max()/1e3:.0f} km 过大"
    v_impact = float(np.linalg.norm(result.y[-1, 3:6]))
    assert 1000.0 < v_impact < 10000.0, f"落地速度 {v_impact:.0f} m/s 不合理"
