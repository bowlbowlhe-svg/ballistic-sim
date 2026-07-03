"""阶段 2 MVP 演示脚本。

域 I：M107 155mm 榴弹 MPM 落地
域 II：CZ-2F powered_eci 入轨
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ballistic_sim.config import SimConfig
from ballistic_sim.dynamics.common import rv_to_oe
from ballistic_sim.frames import ecef_to_geodetic, eci_to_ecef
from ballistic_sim.models.atmosphere import StandardAtmosphere
from ballistic_sim.models.wind import UniformWind
from ballistic_sim.presets import cz2f_config, cz2f_phases, m107_config
from ballistic_sim.simulator import SimResult, simulate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden"


def _load_golden(name: str) -> dict:
    with open(GOLDEN_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


def _relative_err(a: float, b: float) -> float:
    if abs(b) < 1e-12:
        return float("inf")
    return (a - b) / b


def run_m107() -> SimResult:
    """域 I MVP：M107 MPM 落地。"""
    cfg = m107_config()
    result = simulate(cfg, phases=[])
    return result


def _m107_summary(result: SimResult) -> dict:
    y = result.y
    t = result.t
    # ENU 坐标
    alt = y[:, 2]
    # 落地为最后一个 alt 最接近 0 的点
    idx = int(np.argmin(np.abs(alt)))
    # 但通常 solve_ivp 会略过 0；取最后一点
    idx = -1
    range_ground = np.linalg.norm(y[idx, :2])
    impact_angle = float(np.rad2deg(np.arctan2(-y[idx, 5], np.linalg.norm(y[idx, 3:5]))))
    v_impact = float(np.linalg.norm(y[idx, 3:6]))
    return {
        "tof_s": float(t[idx]),
        "range_m": float(range_ground),
        "impact_angle_deg": impact_angle,
        "v_impact_m_s": v_impact,
        "max_alt_m": float(np.max(alt)),
    }


def run_cz2f() -> SimResult:
    """域 II MVP：CZ-2F powered_eci 入轨。"""
    cfg = cz2f_config()
    phases = cz2f_phases(cfg)
    result = simulate(cfg, phases=phases)
    return result


def _cz2f_summary(result: SimResult) -> dict:
    y = result.y
    t = result.t
    # SECO 取最后一个动力段结束时刻（简化取末态）
    idx = -1
    r_eci = y[idx, 0:3]
    v_eci = y[idx, 3:6]
    oe = rv_to_oe(r_eci, v_eci)
    r_ecef = eci_to_ecef(r_eci, float(t[idx]))
    _, _, h = ecef_to_geodetic(r_ecef)
    return {
        "t_seco_s": float(t[idx]),
        "h_seco_m": float(h),
        "v_seco_m_s": float(np.linalg.norm(v_eci)),
        "h_peri_km": oe["h_peri_km"],
        "h_apo_km": oe["h_apo_km"],
        "inc_deg": oe["i_deg"],
        "eccentricity": oe["e"],
    }


def main():
    print("=" * 64)
    print("域 I MVP: M107 MPM 落地")
    print("=" * 64)
    res_m107 = run_m107()
    summary_m107 = _m107_summary(res_m107)
    golden_m107 = _load_golden("atmospheric_m107.json")["scalars"]
    print(f"仿真 TOF:     {summary_m107['tof_s']:.3f} s")
    print(f"仿真 射程:    {summary_m107['range_m']:.1f} m")
    print(f"仿真 落角:    {summary_m107['impact_angle_deg']:.2f} deg")
    print(f"仿真 落速:    {summary_m107['v_impact_m_s']:.2f} m/s")
    print(f"仿真 最大高:  {summary_m107['max_alt_m']:.1f} m")
    print("-" * 64)
    print("黄金文件对比:")
    for key in golden_m107:
        if key in summary_m107:
            err = _relative_err(summary_m107[key], golden_m107[key])
            print(
                f"  {key:20s} 仿真={summary_m107[key]:14.4f}  "
                f"黄金={golden_m107[key]:14.4f}  相对误差={err:+.4%}"
            )

    print()
    print("=" * 64)
    print("域 II MVP: CZ-2F powered_eci 入轨")
    print("=" * 64)
    res_cz2f = run_cz2f()
    summary_cz2f = _cz2f_summary(res_cz2f)
    golden_cz2f = _load_golden("exo_cz2f.json")["scalars"]
    print(f"仿真 SECO 时间:   {summary_cz2f['t_seco_s']:.3f} s")
    print(f"仿真 SECO 高度:   {summary_cz2f['h_seco_m'] / 1e3:.2f} km")
    print(f"仿真 SECO 速度:   {summary_cz2f['v_seco_m_s']:.2f} m/s")
    print(f"仿真 近地点:      {summary_cz2f['h_peri_km']:.2f} km")
    print(f"仿真 远地点:      {summary_cz2f['h_apo_km']:.2f} km")
    print(f"仿真 倾角:        {summary_cz2f['inc_deg']:.3f} deg")
    print(f"仿真 偏心率:      {summary_cz2f['eccentricity']:.5f}")
    print("-" * 64)
    print("黄金文件对比:")
    for key in golden_cz2f:
        if key in summary_cz2f:
            err = _relative_err(summary_cz2f[key], golden_cz2f[key])
            print(
                f"  {key:20s} 仿真={summary_cz2f[key]:14.4f}  "
                f"黄金={golden_cz2f[key]:14.4f}  相对误差={err:+.4%}"
            )


if __name__ == "__main__":
    main()
