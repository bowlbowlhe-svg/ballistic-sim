"""弹道整形与地理 targeting 单元测试。"""

from __future__ import annotations

import pytest

from ballistic_sim.guidance.targeting import (
    max_range_missile,
    solve_missile_range,
    solve_missile_target,
    solve_qe_az,
)


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


def test_max_range_missile_is_placeholder() -> None:
    """max_range_missile 当前为占位实现。"""
    with pytest.raises(NotImplementedError):
        max_range_missile("DF-5")


def test_solve_missile_range_is_placeholder() -> None:
    """solve_missile_range 当前为占位实现。"""
    with pytest.raises(NotImplementedError):
        solve_missile_range("DF-5", 5e6)


def test_solve_missile_target_is_placeholder() -> None:
    """solve_missile_target 当前为占位实现。"""
    with pytest.raises(NotImplementedError):
        solve_missile_target("DF-5", 0.0, 0.0)
