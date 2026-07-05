"""弹道整形与地理 targeting 单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.guidance.targeting import (
    max_range_missile,
    solve_missile_range,
    solve_missile_target,
    solve_qe_az,
)

# 导弹 targeting 对积分精度要求为 km 级，使用宽松容差加速回归测试。
_FAST_OVERRIDES = {
    "options.integrator": "RK45",
    "options.rtol": 1e-5,
    "options.atol": 1e-7,
    "options.max_step": 5.0,
}


def test_solve_qe_az_hits_east_target() -> None:
    """对正东目标反解后正向落点脱靶 < 50 m。"""
    out = solve_qe_az(39.9, 0.0, 39.9, 0.10)
    assert out["converged"]
    assert abs(out["residual_range_m"]) < 50.0
    assert abs(out["residual_cross_m"]) < 50.0
    assert 1.0 <= out["qe_deg"] <= 80.0
    # 目标在正东，方位接近 90°
    assert abs((out["az_deg"] - 90.0 + 180.0) % 360.0 - 180.0) < 2.0


def test_solve_qe_az_returns_target_metadata() -> None:
    """反解结果包含目标距离与真方位。"""
    out = solve_qe_az(0.0, 0.0, 0.0, 0.05)
    assert out["target_range_m"] > 0.0
    assert 0.0 <= out["bearing_deg"] < 360.0


def test_max_range_missile_srBM() -> None:
    """SRBM_600 最大射程反解应收敛并返回有效射程。"""
    out = max_range_missile("SRBM_600", overrides=_FAST_OVERRIDES)
    assert out["converged"]
    assert 5.0 <= out["best_kick_deg"] <= 80.0
    assert out["max_range_m"] > 100e3
    assert out["result"] is not None


def test_solve_missile_range_hit_known_preset() -> None:
    """以最大射程的 70% 为目标的反解应收敛。"""
    max_info = max_range_missile("SRBM_600", overrides=_FAST_OVERRIDES)
    target_range_m = max_info["max_range_m"] * 0.7
    out = solve_missile_range(
        "SRBM_600",
        target_range_m,
        branch="loft",
        overrides=_FAST_OVERRIDES,
    )
    assert out["converged"]
    assert 5.0 <= out["kick_deg"] <= 80.0
    assert abs(out["residual_range_m"]) <= max(5000.0, 0.05 * target_range_m)
    assert out["result"] is not None


def test_solve_missile_target_hits_east() -> None:
    """对正东约 800 km 目标反解后落点脱靶 < 5 km。"""
    lat0, lon0 = 39.9, 116.0
    # 近似构造正东 800 km 目标
    target_lon = lon0 + 800e3 / (111320.0 * np.cos(np.radians(lat0)))
    out = solve_missile_target(
        "SRBM_600",
        lat0,
        target_lon,
        branch="minenergy",
        overrides=_FAST_OVERRIDES,
    )
    assert out["converged"]
    assert abs(out["residual_range_m"]) < 5000.0
    assert abs(out["residual_cross_m"]) < 5000.0
    assert 5.0 <= out["kick_deg"] <= 80.0


def test_unknown_missile_raises() -> None:
    """未知导弹预设应抛出 KeyError。"""
    with pytest.raises(KeyError):
        max_range_missile("DF-5")
