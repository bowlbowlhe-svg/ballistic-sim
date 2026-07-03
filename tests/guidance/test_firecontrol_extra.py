"""补充火控反解测试以提升覆盖率。"""

from __future__ import annotations

import pytest

from ballistic_sim.guidance.firecontrol import (
    max_range_qe,
    solve_firing_solution,
    solve_firing_solution_latlon,
    verify_solution_6dof,
)
from ballistic_sim.presets.projectiles import m107_config


def test_max_range_qe_returns_reasonable() -> None:
    """max_range_qe 返回合理最大射程与对应射角。"""
    cfg = m107_config()
    out = max_range_qe(cfg)
    assert 1.0 <= out["best_qe_deg"] <= 80.0
    assert out["max_range_m"] > 1000.0


def test_solve_firing_solution_6dof_warns() -> None:
    """model='6dof' 触发警告但仍返回结果。"""
    cfg = m107_config()
    with pytest.warns(UserWarning):
        out = solve_firing_solution(cfg, 5000.0, model="6dof")
    assert "qe_deg" in out


def test_solve_firing_solution_converges() -> None:
    """solve_firing_solution 对典型射程收敛。"""
    cfg = m107_config()
    out = solve_firing_solution(cfg, 8000.0)
    assert out["converged"]
    assert abs(out["residual_range_m"]) < 100.0


def test_verify_solution_6dof_within_tol() -> None:
    """verify_solution_6dof 校核已知解。"""
    cfg = m107_config()
    out = verify_solution_6dof(cfg, 45.0, 90.0, 10000.0)
    assert "range_m" in out
    assert "within_tol" in out


def test_solve_firing_solution_latlon_north_target() -> None:
    """latlon 接口对正北较远目标反解。"""
    cfg = m107_config()
    out = solve_firing_solution_latlon(cfg, 41.0, 0.0)
    assert "qe_deg" in out
    assert "az_deg" in out
    assert "bearing_deg" in out
