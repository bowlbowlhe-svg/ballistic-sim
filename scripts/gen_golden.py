"""Generate local golden reference files from the two source projects.

Usage (from the project root):

    PYTHONPATH="D:/Pycharm/游戏与兴趣/外弹道方程;D:/Pycharm/游戏与兴趣/火箭仿真程序" \
        python scripts/gen_golden.py

This script does **not** modify the source projects; it only reads their public
APIs and writes JSON files under ``tests/golden/``.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden"

OUTER_BALLISTICS = Path("D:/Pycharm/游戏与兴趣/外弹道方程")
ROCKET_ROOT = Path("D:/Pycharm/游戏与兴趣/火箭仿真程序")


def _ensure_source_roots() -> None:
    for root in (OUTER_BALLISTICS, ROCKET_ROOT):
        r = str(root)
        if r not in sys.path:
            sys.path.insert(0, r)


def _downsample(arr: np.ndarray, target: int) -> np.ndarray:
    """Return approximately *target* evenly spaced samples from *arr*."""
    arr = np.asarray(arr)
    n = arr.shape[0]
    if n <= target:
        return arr
    idx = np.round(np.linspace(0, n - 1, target)).astype(int)
    return arr[idx]


def _to_list(arr: np.ndarray) -> List[Any]:
    return np.asarray(arr).tolist()


def generate_atmospheric_m107() -> Dict[str, Any]:
    """Run the M107 projectile simulation and extract key results."""
    from ballistics import config as ballistics_config

    cfg = ballistics_config.build_config("M107", model="mpm")
    res = ballistics_config.simulate(cfg)

    # Downsample time series to a compact, roughly 200-point representation.
    target_points = 200
    t = _downsample(res.t, target_points)
    pos = _downsample(res.pos, target_points)
    vel = _downsample(res.vel, target_points)
    speed = np.linalg.norm(vel, axis=1)
    range_ground = np.hypot(pos[:, 0], pos[:, 1])

    return {
        "config_snapshot": {
            "source_project": "外弹道方程",
            "preset": "M107",
            "model": "mpm",
            "v0_m_s": float(cfg.fire.v0),
            "qe_deg": float(cfg.fire.theta0),
            "az_deg": float(cfg.fire.az0),
            "h0_m": float(cfg.fire.h0),
            "lat_deg": float(cfg.fire.lat),
            "lon_deg": float(cfg.fire.lon),
        },
        "scalars": {
            "range_m": float(res.range_m),
            "geodetic_range_m": float(res.geodetic_range_m),
            "impact_angle_deg": float(res.impact_angle),
            "tof_s": float(res.tof),
            "max_alt_m": float(res.max_alt),
            "v_impact_m_s": float(res.v_impact),
            "landed": bool(res.landed),
        },
        "timeseries": {
            "t_s": _to_list(t),
            "alt_m": _to_list(pos[:, 2]),
            "speed_m_s": _to_list(speed),
            "range_ground_m": _to_list(range_ground),
        },
        "source_project": "D:/Pycharm/游戏与兴趣/外弹道方程",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def generate_exo_cz2f() -> Dict[str, Any]:
    """Run the CZ-2F default mission and extract key results."""
    from lvsim import config as rocket_config, simulator

    cfg = rocket_config.build_config()
    mission = simulator.simulate_mission(cfg, dt_grid=0.5)

    ins = mission["insertion"]
    summary = mission["summary"]
    events = {evt["name"]: float(evt["t"]) for evt in mission["events"]}

    # Downsample to roughly 500 points.
    target_points = 500
    t = _downsample(mission["t"], target_points)
    h = _downsample(mission["h"], target_points)
    v = _downsample(mission["v_inertial"], target_points)
    m = _downsample(mission["m"], target_points)
    q = _downsample(mission["q"], target_points)
    accel_g = _downsample(mission["accel_g"], target_points)

    return {
        "config_snapshot": {
            "source_project": "火箭仿真程序",
            "vehicle": cfg["vehicle"]["name"],
            "launch_site": "JSLC (default)",
            "launch_lat_deg": float(cfg["launch"]["lat"]),
            "launch_lon_deg": float(cfg["launch"]["lon"]),
            "launch_alt_m": float(cfg["launch"]["alt"]),
            "launch_azimuth_deg": float(cfg["launch"]["azimuth_deg"]),
            "target": cfg["target"]["name"],
            "payload": cfg["payload"]["name"],
            "payload_mass_kg": float(cfg["payload"]["m"]),
            "target_peri_km": float(cfg["target"]["peri_km"]),
            "target_apo_km": float(cfg["target"]["apo_km"]),
            "target_inc_deg": float(cfg["target"]["inc_deg"]),
        },
        "scalars": {
            "h_seco_m": float(summary["h_seco"]),
            "v_seco_m_s": float(summary["v_seco_inertial"]),
            "m_seco_kg": float(mission["m"][-1]),
            "h_peri_km": float(ins["h_peri_km"]),
            "h_apo_km": float(ins["h_apo_km"]),
            "inc_deg": float(ins["inc_deg"]),
            "period_min": float(ins["period_min"]),
            "a_km": float(ins["a_km"]),
            "eccentricity": float(ins["e"]),
            "max_q_pa": float(summary["max_q"]),
            "h_at_maxq_m": float(summary["h_at_maxq"]),
            "t_seco_s": float(summary["t_seco"]),
            "apogee_m": float(summary["apogee"]),
        },
        "event_times": events,
        "timeseries": {
            "t_s": _to_list(t),
            "alt_m": _to_list(h),
            "speed_m_s": _to_list(v),
            "mass_kg": _to_list(m),
            "q_pa": _to_list(q),
            "accel_g": _to_list(accel_g),
        },
        "source_project": "D:/Pycharm/游戏与兴趣/火箭仿真程序",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    _ensure_source_roots()
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    m107 = generate_atmospheric_m107()
    cz2f = generate_exo_cz2f()

    for name, data in (
        ("atmospheric_m107.json", m107),
        ("exo_cz2f.json", cz2f),
    ):
        path = GOLDEN_DIR / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OK] wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
