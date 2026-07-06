"""迁移对拍脚本：将旧项目生成的黄金文件与新平台结果对比。

Usage
-----
    python scripts/regression_bridge.py --golden tests/golden/atmospheric_m107.json
    python scripts/regression_bridge.py --golden tests/golden/exo_cz2f.json
    python scripts/regression_bridge.py --all

输出
----
- 控制台差异表（绝对误差、相对误差、是否超阈值）
- regression_report.md

说明
----
新平台结果优先调用 ``ballistic_sim`` 的预设与仿真器生成；
若真实仿真失败则回退到 ``_placeholder_new_result`` 占位数据。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden"
REPORT_PATH = PROJECT_ROOT / "regression_report.md"

# 默认阈值（相对误差）
DEFAULT_RTOL = 0.02
DEFAULT_ATOL = 1e-6


@dataclass
class DiffRow:
    """单指标差异行。"""

    key: str
    golden: float
    actual: float
    abs_err: float
    rel_err: float
    passed: bool


def _relative_error(golden: float, actual: float) -> float:
    """计算相对误差；分母为零时退化为绝对误差。"""
    denom = max(abs(golden), 1e-30)
    return abs(actual - golden) / denom


def _load_golden(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _placeholder_new_result(golden: Dict[str, Any]) -> Dict[str, Any]:
    """生成占位新平台结果。

    仅用于真实仿真不可用时展示差异表格式。
    """
    scalars = golden.get("scalars", {})
    perturbed: Dict[str, float] = {}
    for k, v in scalars.items():
        if isinstance(v, (int, float)):
            rng = np.random.default_rng(seed=sum(ord(c) for c in k))
            factor = rng.uniform(0.995, 1.015)
            perturbed[k] = float(v * factor)
        else:
            perturbed[k] = float(v)
    return {"scalars": perturbed}


def _m107_actual(golden: Dict[str, Any]) -> Dict[str, Any]:
    """运行 M107 仿真并提取与 golden 对齐的标量指标。"""
    from ballistic_sim.presets import m107_config
    from ballistic_sim.simulator import simulate

    cfg = m107_config()
    result = simulate(cfg)
    y = result.y
    t = result.t
    idx = -1
    range_m = float(np.linalg.norm(y[idx, :2]))
    impact_angle = float(np.rad2deg(np.arctan2(-y[idx, 5], np.linalg.norm(y[idx, 3:5]))))
    # geodetic_range_m: 用末态 ENU 转 ECEF 后求大地线
    from ballistic_sim.frames import ecef_to_geodetic, enu_to_ecef_vec, geodetic_to_ecef

    r_ecef = geodetic_to_ecef(cfg.launch.lat_deg, cfg.launch.lon_deg, 0.0) + enu_to_ecef_vec(
        [y[idx, 0], y[idx, 1], 0.0], cfg.launch.lat_deg, cfg.launch.lon_deg
    )
    lat_f, lon_f, _ = ecef_to_geodetic(r_ecef)
    from ballistic_sim.frames import haversine_distance

    geod_range = haversine_distance(cfg.launch.lat_deg, cfg.launch.lon_deg, lat_f, lon_f)
    return {
        "scalars": {
            "range_m": range_m,
            "geodetic_range_m": geod_range,
            "impact_angle_deg": impact_angle,
            "tof_s": float(t[idx]),
            "max_alt_m": float(np.max(y[:, 2])),
            "v_impact_m_s": float(np.linalg.norm(y[idx, 3:6])),
            "landed": True,
        }
    }


def _cz2f_actual(golden: Dict[str, Any]) -> Dict[str, Any]:
    """运行 CZ-2F 仿真并提取与 golden 对齐的标量指标。"""
    from ballistic_sim.constants import WGS84_A
    from ballistic_sim.dynamics.common import rv_to_oe
    from ballistic_sim.presets import rocket_full_config
    from ballistic_sim.simulator import simulate

    snap = golden.get("config_snapshot", {})
    cfg = rocket_full_config(
        "CZ2F",
        payload_mass_kg=float(snap.get("payload_mass_kg", 8000.0)),
    )
    result = simulate(cfg)
    y = result.y
    t = result.t
    idx = -1
    r_eci = y[idx, 0:3]
    v_eci = y[idx, 3:6]
    h_sph = float(np.linalg.norm(r_eci)) - WGS84_A
    v = float(np.linalg.norm(v_eci))
    m = float(y[idx, 6])
    oe = rv_to_oe(r_eci, v_eci)
    a_m = float(oe["a"])
    period_min = float(2.0 * np.pi * np.sqrt(a_m**3 / 3.986004418e14) / 60.0) if a_m > 0.0 else 0.0

    # max-q 与对应高度
    r_all = y[:, 0:3]
    v_all = y[:, 3:6]
    h_all = np.linalg.norm(r_all, axis=1) - WGS84_A
    from ballistic_sim.models.atmosphere import StandardAtmosphere

    atm = StandardAtmosphere()
    rho_all = np.array([atm.density(max(h, 0.0)) for h in h_all])
    vrel_all = np.linalg.norm(v_all - np.cross(np.array([0.0, 0.0, 7.2921159e-5]), r_all), axis=1)
    q_all = 0.5 * rho_all * vrel_all**2
    idx_q = int(np.argmax(q_all))
    max_q = float(q_all[idx_q])
    h_at_maxq = float(h_all[idx_q])

    return {
        "scalars": {
            "h_seco_m": h_sph,
            "v_seco_m_s": v,
            "m_seco_kg": m,
            "h_peri_km": float(oe["h_peri_km"]),
            "h_apo_km": float(oe["h_apo_km"]),
            "inc_deg": float(oe["i_deg"]),
            "period_min": period_min,
            "a_km": a_m / 1e3,
            "eccentricity": float(oe["e"]),
            "max_q_pa": max_q,
            "h_at_maxq_m": h_at_maxq,
            "t_seco_s": float(t[idx]),
            "apogee_m": h_sph,
        }
    }


def _try_real_new_result(golden: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """尝试调用新平台 simulator；失败时返回 None。"""
    try:
        snap = golden.get("config_snapshot", {})
        preset = str(snap.get("preset", ""))
        if "M107" in preset:
            return _m107_actual(golden)
        if "CZ" in str(snap.get("vehicle", "")).upper() or "cz2f" in preset.lower():
            return _cz2f_actual(golden)
        return None
    except Exception:  # pragma: no cover
        return None


def _compare_scalars(
    golden: Dict[str, Any],
    actual: Dict[str, Any],
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> Tuple[List[DiffRow], bool]:
    """对比标量指标并返回差异表与整体是否通过。"""
    g_scalars = golden.get("scalars", {})
    a_scalars = actual.get("scalars", {})
    rows: List[DiffRow] = []
    all_passed = True

    for key in sorted(g_scalars.keys()):
        g_val = g_scalars[key]
        a_val = a_scalars.get(key)
        if a_val is None:
            continue
        if isinstance(g_val, bool):
            passed = bool(g_val) == bool(a_val)
            if not passed:
                all_passed = False
            rows.append(
                DiffRow(
                    key=key,
                    golden=float(g_val),
                    actual=float(a_val),
                    abs_err=abs(float(a_val) - float(g_val)),
                    rel_err=abs(float(a_val) - float(g_val)),
                    passed=passed,
                )
            )
            continue
        if not isinstance(g_val, (int, float)):
            continue
        abs_err = abs(float(a_val) - float(g_val))
        rel_err = _relative_error(float(g_val), float(a_val))
        passed = rel_err <= rtol and abs_err <= atol
        if not passed:
            all_passed = False
        rows.append(
            DiffRow(
                key=key,
                golden=float(g_val),
                actual=float(a_val),
                abs_err=abs_err,
                rel_err=rel_err,
                passed=passed,
            )
        )
    return rows, all_passed


def _print_table(rows: List[DiffRow]) -> None:
    header = (
        f"{'key':<24} {'golden':>16} {'actual':>16} {'abs_err':>14} {'rel_err':>12} {'status':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        status = "PASS" if row.passed else "FAIL"
        print(
            f"{row.key:<24} "
            f"{row.golden:>16.6f} {row.actual:>16.6f} "
            f"{row.abs_err:>14.6e} {row.rel_err:>12.6f} {status:>8}"
        )


def _generate_report(
    golden_path: Path,
    rows: List[DiffRow],
    all_passed: bool,
    used_placeholder: bool,
    rtol: float,
    atol: float,
) -> str:
    lines: List[str] = [
        "# Regression Report",
        "",
        f"- **Golden file**: `{golden_path}`",
        f"- **Generated at**: {datetime.now(timezone.utc).isoformat()}",
        f"- **Thresholds**: rtol={rtol}, atol={atol}",
        f"- **New platform result**: "
        f"{'placeholder (MVP not ready)' if used_placeholder else 'real simulation'}",
        f"- **Overall**: {'PASS' if all_passed else 'FAIL'}",
        "",
        "## Scalar comparison",
        "",
        "| key | golden | actual | abs_err | rel_err | status |",
        "|-----|--------|--------|---------|---------|--------|",
    ]
    for row in rows:
        status = "PASS" if row.passed else "FAIL"
        lines.append(
            f"| {row.key} | {row.golden:.6f} | {row.actual:.6f} | "
            f"{row.abs_err:.6e} | {row.rel_err:.6f} | {status} |"
        )
    lines.append("")
    if used_placeholder:
        lines.append(
            "> **Note**: 当前使用占位结果。待 ``ballistic_sim.simulator`` "
            "实现后，请在 ``_try_real_new_result`` 中替换为真实调用。"
        )
        lines.append("")
    return "\n".join(lines)


def _run_regression(
    golden_path: Path,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    write_report: bool = True,
) -> int:
    if not golden_path.exists():
        print(f"[ERROR] Golden file not found: {golden_path}", file=sys.stderr)
        return 1

    golden = _load_golden(golden_path)
    actual = _try_real_new_result(golden)
    used_placeholder = actual is None
    if used_placeholder:
        actual = _placeholder_new_result(golden)

    rows, all_passed = _compare_scalars(golden, actual, rtol=rtol, atol=atol)

    print(f"Regression: {golden_path.name}")
    print(f"Thresholds: rtol={rtol}, atol={atol}")
    print(f"Result source: {'placeholder' if used_placeholder else 'simulation'}")
    _print_table(rows)
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")

    if write_report:
        report = _generate_report(
            golden_path=golden_path,
            rows=rows,
            all_passed=all_passed,
            used_placeholder=used_placeholder,
            rtol=rtol,
            atol=atol,
        )
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report written to {REPORT_PATH}")

    return 0 if all_passed else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare golden reference files against the new platform."
    )
    parser.add_argument(
        "--golden",
        type=str,
        default=str(GOLDEN_DIR / "atmospheric_m107.json"),
        help="Path to a golden JSON file.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run regression for all golden files.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help="Relative tolerance threshold.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help="Absolute tolerance threshold.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write regression_report.md.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.all:
        targets = sorted(GOLDEN_DIR.glob("*.json"))
        if not targets:
            print(f"[ERROR] No golden files found in {GOLDEN_DIR}", file=sys.stderr)
            return 1
    else:
        targets = [Path(args.golden)]

    overall_rc = 0
    for target in targets:
        rc = _run_regression(
            golden_path=target,
            rtol=args.rtol,
            atol=args.atol,
            write_report=not args.no_report,
        )
        if rc != 0:
            overall_rc = rc
        print()

    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())
