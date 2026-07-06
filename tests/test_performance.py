"""轻量级性能回归测试。

这些测试标记为 ``slow``，默认不会运行；CI 或本地验收时可显式启用::

    pytest tests/test_performance.py -m slow
"""

from __future__ import annotations

import time

import pytest

from ballistic_sim.monte_carlo import PerturbationConfig, monte_carlo_simulation
from ballistic_sim.presets import m107_config


@pytest.mark.slow
def test_monte_carlo_process_100_under_60s() -> None:
    """100 样本 process 后端 Monte Carlo 应在 60 秒内完成。"""
    cfg = m107_config()
    cfg.options.mpm_use_spin = False
    t0 = time.perf_counter()
    result = monte_carlo_simulation(
        cfg,
        perturb=PerturbationConfig(),
        n_samples=100,
        backend="process",
        n_jobs=-1,
        seed=42,
    )
    elapsed = time.perf_counter() - t0
    assert result.samples == 100
    assert elapsed < 60.0, f"100 样本 MC process 后端耗时 {elapsed:.1f}s，超过 60s 阈值"


@pytest.mark.slow
def test_single_projectile_under_5s() -> None:
    """单发 projectile 仿真应在 5 秒内完成。"""
    from ballistic_sim.presets import m107_config
    from ballistic_sim.simulator import simulate

    cfg = m107_config()
    t0 = time.perf_counter()
    result = simulate(cfg)
    elapsed = time.perf_counter() - t0
    assert result.stop_reason != ""
    assert elapsed < 5.0, f"单发 projectile 耗时 {elapsed:.1f}s，超过 5s 阈值"
