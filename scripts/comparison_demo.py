"""M107 多工况射程对比演示脚本。

对比三种工况：
- 真空无阻力
- 有大气阻力（ISA + G1 弹）
- 有大气阻力 + 常值侧风
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ballistic_sim.config import apply_overrides
from ballistic_sim.presets import m107_config
from ballistic_sim.simulator import simulate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "out" / "comparison_demo.json"


def _range_of(result) -> float:
    return float(np.linalg.norm(result.y[-1, :2]))


def run_vacuum() -> Dict[str, Any]:
    """真空无阻力工况：关闭大气与风。"""
    cfg = m107_config()
    cfg = apply_overrides(
        cfg,
        {
            "environment.atmosphere": "none",
            "environment.wind_m_s": [0.0, 0.0, 0.0],
        },
    )
    result = simulate(cfg)
    return {
        "case": "vacuum",
        "range_m": _range_of(result),
        "tof_s": float(result.t[-1]),
        "max_alt_m": float(np.max(result.y[:, 2])),
    }


def run_with_drag() -> Dict[str, Any]:
    """有阻力工况：ISA + G1。"""
    cfg = m107_config()
    result = simulate(cfg)
    return {
        "case": "drag",
        "range_m": _range_of(result),
        "tof_s": float(result.t[-1]),
        "max_alt_m": float(np.max(result.y[:, 2])),
    }


def run_with_wind() -> Dict[str, Any]:
    """有阻力 + 常值侧风工况。"""
    cfg = m107_config()
    cfg = apply_overrides(
        cfg,
        {
            "environment.wind_m_s": [0.0, 5.0, 0.0],
        },
    )
    result = simulate(cfg)
    return {
        "case": "drag_and_wind",
        "range_m": _range_of(result),
        "tof_s": float(result.t[-1]),
        "max_alt_m": float(np.max(result.y[:, 2])),
    }


def main() -> int:
    print("Running M107 comparison cases...")
    cases = [run_vacuum(), run_with_drag(), run_with_wind()]
    for c in cases:
        print(
            f"  {c['case']}: range={c['range_m']:.1f} m, "
            f"TOF={c['tof_s']:.2f} s, max_alt={c['max_alt_m']:.1f} m"
        )

    drag_vs_vac = 1.0 - cases[1]["range_m"] / cases[0]["range_m"]
    wind_vs_drag = (cases[2]["range_m"] - cases[1]["range_m"]) / cases[1]["range_m"]
    summary = {
        "cases": cases,
        "drag_reduction": float(drag_vs_vac),
        "wind_relative_change": float(wind_vs_drag),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
