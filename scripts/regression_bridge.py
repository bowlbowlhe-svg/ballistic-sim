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
当前阶段 Developer 的 simulator/phases MVP 尚未就绪，因此新平台结果
使用 ``_placeholder_new_result`` 生成的占位数据。占位数据刻意与黄金
文件保持小幅差异，以展示差异表格式；待 ``ballistic_sim.simulator``
实现后，替换为真实调用即可。
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

    当前 MVP 未实现，因此通过黄金数据乘以小幅扰动来模拟新平台输出，
    仅用于验证对拍脚本格式与阈值判定逻辑。
    """
    scalars = golden.get("scalars", {})
    perturbed: Dict[str, float] = {}
    for k, v in scalars.items():
        if isinstance(v, (int, float)):
            # 0.5% ~ 2% 的随机扰动，固定种子保证可重复
            rng = np.random.default_rng(seed=sum(ord(c) for c in k))
            factor = rng.uniform(0.995, 1.015)
            perturbed[k] = float(v * factor)
        else:
            perturbed[k] = float(v)
    return {"scalars": perturbed}


def _try_real_new_result(golden: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """尝试调用新平台 simulator；未就绪时返回 None。"""
    try:
        # TODO: 替换为真实调用，例如
        # from ballistic_sim.simulator import run_simulation
        # return run_simulation(_golden_to_sim_config(golden))
        return None  # pragma: no cover
    except Exception:  # pragma: no cover
        return None


def _golden_to_sim_config(golden: Dict[str, Any]) -> Dict[str, Any]:
    """把黄金文件配置快照转换为 SimConfig 字典（占位）。"""
    snap = golden.get("config_snapshot", {})
    mission = "projectile" if "M107" in str(snap.get("preset", "")) else "rocket"
    return {
        "mission": mission,
        "launch": {
            "lat_deg": snap.get("lat_deg", snap.get("launch_lat_deg", 0.0)),
            "lon_deg": snap.get("lon_deg", snap.get("launch_lon_deg", 0.0)),
            "alt_m": snap.get("h0_m", snap.get("launch_alt_m", 0.0)),
            "azimuth_deg": snap.get("az_deg", snap.get("launch_azimuth_deg", 0.0)),
            "elevation_deg": snap.get("qe_deg", 45.0),
            "v0_m_s": snap.get("v0_m_s", 0.0),
        },
    }


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
        f"- **New platform result**: {'placeholder (MVP not ready)' if used_placeholder else 'real simulation'}",
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
