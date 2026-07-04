"""阶段 2.2 蒙特卡洛后端对拍验证脚本."""

from __future__ import annotations

import numpy as np

from ballistic_sim.config import DistributionConfig, PerturbationConfig
from ballistic_sim.monte_carlo import monte_carlo_simulation
from ballistic_sim.presets import m107_config


def _rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def _run_case(
    label: str,
    perturb: PerturbationConfig,
    n_samples: int = 200,
    seed: int = 42,
) -> dict:
    cfg = m107_config()
    # batch 后端要求关闭 spin/dynamic_alpha
    cfg.options.mpm_use_spin = False
    cfg.options.mpm_use_dynamic_alpha = False

    res_process = monte_carlo_simulation(
        cfg,
        perturb=perturb,
        n_samples=n_samples,
        backend="process",
        n_jobs=1,
        seed=seed,
    )
    res_batch = monte_carlo_simulation(
        cfg,
        perturb=perturb,
        n_samples=n_samples,
        backend="batch",
        seed=seed,
    )

    stats = {
        "label": label,
        "samples": n_samples,
        "process": {
            "range_mean": res_process.range_mean,
            "range_std": res_process.range_std,
            "cross_mean": res_process.cross_mean,
            "cross_std": res_process.cross_std,
            "tof_mean": float(np.mean(res_process.tofs)),
            "tof_std": float(np.std(res_process.tofs)),
        },
        "batch": {
            "range_mean": res_batch.range_mean,
            "range_std": res_batch.range_std,
            "cross_mean": res_batch.cross_mean,
            "cross_std": res_batch.cross_std,
            "tof_mean": float(np.mean(res_batch.tofs)),
            "tof_std": float(np.std(res_batch.tofs)),
        },
    }
    stats["diff_pct"] = {
        "range_mean": _rel_diff(
            stats["process"]["range_mean"], stats["batch"]["range_mean"]
        )
        * 100,
        "range_std": _rel_diff(
            stats["process"]["range_std"], stats["batch"]["range_std"]
        )
        * 100,
        "cross_mean": _rel_diff(
            stats["process"]["cross_mean"], stats["batch"]["cross_mean"]
        )
        * 100,
        "cross_std": _rel_diff(
            stats["process"]["cross_std"], stats["batch"]["cross_std"]
        )
        * 100,
        "tof_mean": _rel_diff(
            stats["process"]["tof_mean"], stats["batch"]["tof_mean"]
        )
        * 100,
        "tof_std": _rel_diff(
            stats["process"]["tof_std"], stats["batch"]["tof_std"]
        )
        * 100,
    }
    return stats


def _print_case(stats: dict) -> None:
    print(f"\n=== {stats['label']} (n={stats['samples']}) ===")
    print(
        f"process range: {stats['process']['range_mean']:.2f} ± "
        f"{stats['process']['range_std']:.2f} m"
    )
    print(
        f"batch   range: {stats['batch']['range_mean']:.2f} ± "
        f"{stats['batch']['range_std']:.2f} m"
    )
    print(
        f"process cross: {stats['process']['cross_mean']:.2f} ± "
        f"{stats['process']['cross_std']:.2f} m"
    )
    print(
        f"batch   cross: {stats['batch']['cross_mean']:.2f} ± "
        f"{stats['batch']['cross_std']:.2f} m"
    )
    print(
        f"process tof:   {stats['process']['tof_mean']:.2f} ± "
        f"{stats['process']['tof_std']:.2f} s"
    )
    print(
        f"batch   tof:   {stats['batch']['tof_mean']:.2f} ± "
        f"{stats['batch']['tof_std']:.2f} s"
    )
    print("差异 %:")
    for key, val in stats["diff_pct"].items():
        print(f"  {key}: {val:.4f}%")
    ok = all(v < 1.0 for v in stats["diff_pct"].values())
    print(f"对拍结果: {'PASS' if ok else 'FAIL'} (<1%)")


def main() -> None:
    # 零扰动
    zero_perturb = PerturbationConfig(
        v0_m_s=DistributionConfig(std=0.0),
        elevation_deg=DistributionConfig(std=0.0),
        azimuth_deg=DistributionConfig(std=0.0),
        wind_e=DistributionConfig(std=0.0),
        wind_n=DistributionConfig(std=0.0),
    )
    stats_zero = _run_case("M107 默认配置 零扰动", zero_perturb)
    _print_case(stats_zero)

    # 非零扰动
    nonzero_perturb = PerturbationConfig(
        v0_m_s=DistributionConfig(std=2.0),
        elevation_deg=DistributionConfig(std=0.05),
        wind_e=DistributionConfig(std=1.0),
        wind_n=DistributionConfig(std=1.0),
    )
    stats_nonzero = _run_case("M107 非零扰动 (v0=2m/s, wind=1m/s)", nonzero_perturb)
    _print_case(stats_nonzero)


if __name__ == "__main__":
    main()
