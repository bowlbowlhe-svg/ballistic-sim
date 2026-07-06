"""火控反解测试（阶段 3）。

使用 M107 155 mm 榴弹配置，给定目标经纬度后反解射角/方位，
再用反解结果跑一次正向仿真，验证落点与目标距离 < 50 m。
"""

from __future__ import annotations

import pytest

from ballistic_sim.config import apply_overrides
from ballistic_sim.frames import (
    ecef_to_geodetic,
    enu_to_ecef_vec,
    geodetic_to_ecef,
    haversine_distance,
)
from ballistic_sim.guidance.firecontrol import solve_firing_solution_latlon
from ballistic_sim.presets.projectiles import m107_config
from ballistic_sim.simulator import simulate


@pytest.mark.parametrize(
    "target_lat, target_lon",
    [
        (39.9, 0.05),  # 约 4.5 km，低伸弹道
        (39.9, 0.10),  # 约 9 km，中伸弹道
        (39.9, 0.15),  # 约 14 km，弯曲弹道
    ],
)
def test_m107_firecontrol_latlon(target_lat: float, target_lon: float) -> None:
    """M107 火控反解：射角/方位收敛且落点距目标 < 50 m。"""
    cfg = m107_config()
    sol = solve_firing_solution_latlon(cfg, target_lat, target_lon)

    assert sol["converged"], f"火控反解未收敛：{sol}"
    assert abs(sol["residual_range_m"]) < 50.0
    assert abs(sol["residual_cross_m"]) < 50.0

    # 用反解诸元做一次正向仿真，检查大地线脱靶量
    run_cfg = apply_overrides(
        cfg,
        {
            "launch.elevation_deg": float(sol["qe_deg"]),
            "launch.azimuth_deg": float(sol["az_deg"]),
        },
    )
    res = simulate(run_cfg)
    # MPM 结果位于发射点 ENU，先转 ECEF 再求大地坐标
    e, n = float(res.y[-1, 0]), float(res.y[-1, 1])
    r_ecef = geodetic_to_ecef(cfg.launch.lat_deg, cfg.launch.lon_deg, 0.0) + enu_to_ecef_vec(
        [e, n, 0.0], cfg.launch.lat_deg, cfg.launch.lon_deg
    )
    latf, lonf, _ = ecef_to_geodetic(r_ecef)
    miss = haversine_distance(target_lat, target_lon, latf, lonf)
    assert miss < 50.0, f"正向仿真脱靶 {miss:.1f} m >= 50 m"


def test_m107_firecontrol_returns_reasonable_qe_az() -> None:
    """反解返回的射角/方位应在物理合理范围内。"""
    cfg = m107_config()
    sol = solve_firing_solution_latlon(cfg, 39.9, 0.10)
    assert 1.0 <= sol["qe_deg"] <= 80.0
    assert 0.0 <= sol["az_deg"] <= 360.0
    # 目标在正东，发射方位应接近 90°
    assert abs((sol["az_deg"] - 90.0 + 180.0) % 360.0 - 180.0) < 2.0
