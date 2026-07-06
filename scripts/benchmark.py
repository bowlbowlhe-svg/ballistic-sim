"""阶段 1 性能基准脚本。

运行三组典型负载并输出 JSON 报告：
1. 单发 projectile 仿真（M107）
2. 100 样本 Monte Carlo（process 后端）
3. 10 样本 6-DOF 再入
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

from ballistic_sim import __version__
from ballistic_sim.monte_carlo import PerturbationConfig, monte_carlo_simulation
from ballistic_sim.phases.builder import build_phases
from ballistic_sim.presets import m107_config, rocket_full_config
from ballistic_sim.simulator import simulate


def _timed(label: str, fn) -> tuple[Any, float]:
    """执行函数并返回 (结果, 耗时秒)。"""
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    print(f"  {label}: {elapsed:.3f} s")
    return result, elapsed


def _benchmark_single_shot() -> Dict[str, Any]:
    """单发 M107 projectile 仿真。"""
    cfg = m107_config()
    phases = build_phases(cfg)
    result, elapsed = _timed(
        "单发 projectile", lambda: simulate(cfg, phases=phases)
    )
    return {
        "name": "single_projectile",
        "n_samples": 1,
        "elapsed_s": elapsed,
        "stop_reason": result.stop_reason,
        "n_points": int(result.t.size),
    }


def _benchmark_monte_carlo_process() -> Dict[str, Any]:
    """100 样本 process 后端 Monte Carlo。"""
    cfg = m107_config()
    cfg.options.mpm_use_spin = False
    perturb = PerturbationConfig()
    result, elapsed = _timed(
        "100 样本 MC (process)",
        lambda: monte_carlo_simulation(
            cfg,
            perturb=perturb,
            n_samples=100,
            backend="process",
            n_jobs=2,
            seed=42,
        ),
    )
    return {
        "name": "monte_carlo_process_100",
        "n_samples": 100,
        "elapsed_s": elapsed,
        "valid_samples": int(result.samples),
        "cep50_m": float(result.cep50),
        "cep90_m": float(result.cep90),
    }


def _benchmark_sixdof_reentry() -> Dict[str, Any]:
    """10 样本 CZ-2F 预设链仿真（复刻旧 ``rocket_phases`` 行为，不启用真实 6-DOF 再入）。"""

    def _run_samples() -> list:
        results = []
        for i in range(3):
            cfg = rocket_full_config("CZ2F")
            # 轻微扰动程序转弯角以产生不同入轨条件
            cfg = cfg.model_copy(
                deep=True,
                update={
                    "guidance": cfg.guidance.model_copy(update={"kick_deg": 3.0 + i * 0.2}),
                },
            )
            phases = build_phases(cfg)
            results.append(simulate(cfg, phases=phases))
        return results

    sample_results, elapsed = _timed("10 样本 CZ-2F 预设链", _run_samples)
    completed = sum(1 for r in sample_results if r.stop_reason == "completed")
    return {
        "name": "sixdof_reentry_3",
        "n_samples": 3,
        "elapsed_s": elapsed,
        "completed": completed,
        "avg_points": float(sum(int(r.t.size) for r in sample_results)) / len(sample_results),
    }


def main() -> None:
    """入口。"""
    print(f"Ballistic Sim v{__version__} — Stage 1 Benchmark")
    print("-" * 50)

    benchmarks = [
        _benchmark_single_shot(),
        _benchmark_monte_carlo_process(),
        _benchmark_sixdof_reentry(),
    ]

    total = sum(b["elapsed_s"] for b in benchmarks)
    report: Dict[str, Any] = {
        "version": __version__,
        "benchmarks": benchmarks,
        "total_elapsed_s": total,
    }

    out_path = Path("benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("-" * 50)
    print(f"总耗时: {total:.3f} s")
    print(f"结果已写入: {out_path.resolve()}")


if __name__ == "__main__":
    main()
