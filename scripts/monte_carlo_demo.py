"""蒙特卡洛扰动演示脚本。

- 对 M107 做 100 次蒙特卡洛（风/射角小扰动），输出落点散布与 CEP。
- 对 CZ-2F 做 10 次蒙特卡洛（质量/推力小扰动），输出入轨根数散布。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ballistic_sim.config import apply_overrides
from ballistic_sim.constants import WGS84_A
from ballistic_sim.dynamics.common import rv_to_oe
from ballistic_sim.presets import cz2f_config, cz2f_phases, m107_config, m107_phases
from ballistic_sim.simulator import simulate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "out" / "monte_carlo_demo.json"


@dataclass
class M107Outcome:
    range_m: float
    impact_angle_deg: float
    tof_s: float


@dataclass
class Cz2fOutcome:
    h_seco_m: float
    v_seco_m_s: float
    h_peri_km: float
    h_apo_km: float
    inc_deg: float
    eccentricity: float


def _compute_cep(ranges_m: np.ndarray) -> float:
    """50% 圆概率误差（按样本排序估计）。"""
    n = len(ranges_m)
    sorted_r = np.sort(np.asarray(ranges_m, dtype=float))
    idx = int(np.clip(0.5 * (n - 1), 0, n - 1))
    return float(sorted_r[idx])


def _m107_run(
    *,
    wind_e: float = 0.0,
    wind_n: float = 0.0,
    qe_offset_deg: float = 0.0,
    az_offset_deg: float = 0.0,
) -> M107Outcome:
    cfg = m107_config()
    cfg = apply_overrides(
        cfg,
        {
            "launch.elevation_deg": cfg.launch.elevation_deg + qe_offset_deg,
            "launch.azimuth_deg": cfg.launch.azimuth_deg + az_offset_deg,
            "environment.wind_m_s": [wind_e, wind_n, 0.0],
        },
    )
    result = simulate(cfg, phases=m107_phases())
    y = result.y
    idx = -1
    return M107Outcome(
        range_m=float(np.linalg.norm(y[idx, :2])),
        impact_angle_deg=float(np.rad2deg(np.arctan2(-y[idx, 5], np.linalg.norm(y[idx, 3:5])))),
        tof_s=float(result.t[idx]),
    )


def _cz2f_run(
    *,
    payload_offset_kg: float = 0.0,
    thrust_offset_ratio: float = 0.0,
) -> Cz2fOutcome:
    cfg = cz2f_config()
    phases = cz2f_phases(cfg)
    if payload_offset_kg != 0.0:
        # 通过改变 payload 质量间接扰动各级干/湿质量
        cfg = cz2f_config(payload_mass_kg=8000.0 + payload_offset_kg)
        phases = cz2f_phases(cfg)
    if thrust_offset_ratio != 0.0:
        for ph in phases:
            stage = getattr(ph.dynamics, "stage", None)
            if stage is None:
                continue
            for key in ("thrust_vac", "thrust_sl"):
                if key in stage:
                    stage[key] = float(stage[key]) * (1.0 + thrust_offset_ratio)
    result = simulate(cfg, phases=phases)
    y = result.y
    idx = -1
    r_eci = y[idx, 0:3]
    v_eci = y[idx, 3:6]
    oe = rv_to_oe(r_eci, v_eci)
    return Cz2fOutcome(
        h_seco_m=float(np.linalg.norm(r_eci)) - WGS84_A,
        v_seco_m_s=float(np.linalg.norm(v_eci)),
        h_peri_km=float(oe["h_peri_km"]),
        h_apo_km=float(oe["h_apo_km"]),
        inc_deg=float(oe["i_deg"]),
        eccentricity=float(oe["e"]),
    )


def run_m107_monte_carlo(n: int = 100, seed: int = 42) -> Dict[str, Any]:
    """M107 风/射角小扰动蒙特卡洛。"""
    rng = np.random.default_rng(seed)
    outcomes: List[M107Outcome] = []
    for _ in range(n):
        wind_e = rng.normal(0.0, 2.0)
        wind_n = rng.normal(0.0, 2.0)
        qe_off = rng.normal(0.0, 0.05)
        az_off = rng.normal(0.0, 0.05)
        outcomes.append(
            _m107_run(
                wind_e=wind_e,
                wind_n=wind_n,
                qe_offset_deg=qe_off,
                az_offset_deg=az_off,
            )
        )

    ranges = np.array([o.range_m for o in outcomes])
    impact_angles = np.array([o.impact_angle_deg for o in outcomes])
    tofs = np.array([o.tof_s for o in outcomes])

    return {
        "n_samples": n,
        "range_mean_m": float(np.mean(ranges)),
        "range_std_m": float(np.std(ranges)),
        "cep50_m": _compute_cep(ranges),
        "impact_angle_mean_deg": float(np.mean(impact_angles)),
        "impact_angle_std_deg": float(np.std(impact_angles)),
        "tof_mean_s": float(np.mean(tofs)),
        "tof_std_s": float(np.std(tofs)),
    }


def run_cz2f_monte_carlo(n: int = 10, seed: int = 42) -> Dict[str, Any]:
    """CZ-2F 质量/推力小扰动蒙特卡洛。"""
    rng = np.random.default_rng(seed)
    outcomes: List[Cz2fOutcome] = []
    for _ in range(n):
        payload_off = rng.normal(0.0, 50.0)
        thrust_ratio = rng.normal(0.0, 0.005)
        outcomes.append(
            _cz2f_run(
                payload_offset_kg=payload_off,
                thrust_offset_ratio=thrust_ratio,
            )
        )

    def _stats(arr: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "n_samples": n,
        "h_seco_m": _stats(np.array([o.h_seco_m for o in outcomes])),
        "v_seco_m_s": _stats(np.array([o.v_seco_m_s for o in outcomes])),
        "h_peri_km": _stats(np.array([o.h_peri_km for o in outcomes])),
        "h_apo_km": _stats(np.array([o.h_apo_km for o in outcomes])),
        "inc_deg": _stats(np.array([o.inc_deg for o in outcomes])),
        "eccentricity": _stats(np.array([o.eccentricity for o in outcomes])),
    }


def main() -> int:
    print("Running M107 Monte Carlo (n=100)...")
    m107_stats = run_m107_monte_carlo()
    print(f"  CEP50 = {m107_stats['cep50_m']:.2f} m")
    print(f"  range std = {m107_stats['range_std_m']:.2f} m")

    print("Running CZ-2F Monte Carlo (n=10)...")
    cz2f_stats = run_cz2f_monte_carlo()
    print(f"  inc std = {cz2f_stats['inc_deg']['std']:.4f} deg")
    print(f"  h_apo std = {cz2f_stats['h_apo_km']['std']:.4f} km")

    result = {
        "M107": m107_stats,
        "CZ2F": cz2f_stats,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
