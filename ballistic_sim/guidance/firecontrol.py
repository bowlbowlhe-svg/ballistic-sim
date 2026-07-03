r"""火控反解：由目标位置反推射角 (qe) 与方位角 (az)。

来源：``D:\Pycharm\游戏与兴趣\外弹道方程\ballistics\firecontrol.py``。
迁移到统一平台后，直接操作 :class:`ballistic_sim.config.SimConfig` 与
:func:`ballistic_sim.simulator.simulate`。
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from ballistic_sim.config import SimConfig, apply_overrides
from ballistic_sim.frames import haversine_distance, initial_bearing
from ballistic_sim.simulator import simulate

RANGE_ABS_TOL_M = 500.0
RANGE_REL_TOL = 0.05
CROSS_ABS_TOL_M = 50.0


def _simulate_impact_en(
    cfg: SimConfig,
    qe_deg: float,
    az_deg: float,
) -> Tuple[float, float, Any]:
    """跑一次仿真，返回固定 ENU 系下落点 (E, N) (m) 与 result。"""
    overrides = {"launch.elevation_deg": float(qe_deg), "launch.azimuth_deg": float(az_deg)}
    run_cfg = apply_overrides(cfg, overrides)
    try:
        res = simulate(run_cfg, phases=[])
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan"), None
    if res.y.size == 0:
        return float("nan"), float("nan"), None
    pos = res.y[-1, :]
    return float(pos[0]), float(pos[1]), res


def _project(e: float, n: float, base_az_deg: float) -> Tuple[float, float]:
    """把固定 ENU 落点投影到以 base_az 为基准视线的 (射程, 横向)。"""
    az = np.deg2rad(base_az_deg)
    s, c = np.sin(az), np.cos(az)
    downrange = e * s + n * c
    cross = e * c - n * s
    return float(downrange), float(cross)


def _simulate_range_cross(
    cfg: SimConfig,
    qe_deg: float,
    az_deg: float,
    base_az_deg: Optional[float] = None,
) -> Tuple[float, float, Any]:
    """跑一次仿真，返回 (射程 m, 横向 m, result)。"""
    e, n, res = _simulate_impact_en(cfg, qe_deg, az_deg)
    if res is None or not (np.isfinite(e) and np.isfinite(n)):
        return 0.0, 0.0, None
    base = az_deg if base_az_deg is None else base_az_deg
    rng, cross = _project(e, n, base)
    return rng, cross, res


def max_range_qe(
    cfg: SimConfig,
    qe_bounds: Tuple[float, float] = (15.0, 80.0),
) -> Dict[str, Any]:
    """一维有界反解最大射程对应射角。"""
    az0 = float(cfg.launch.azimuth_deg)

    def neg_range(qe: float) -> float:
        rng, _, _ = _simulate_range_cross(cfg, float(qe), az0)
        return -rng

    opt = minimize_scalar(neg_range, bounds=qe_bounds, method="bounded", options={"xatol": 0.2})
    best_qe = float(opt.x)
    best_range = -float(opt.fun)
    _, _, result = _simulate_range_cross(cfg, best_qe, az0)
    return {
        "best_qe_deg": best_qe,
        "max_range_m": best_range,
        "result": result,
    }


def _scan_qe_for_range(
    cfg: SimConfig,
    target_range_m: float,
    az_deg: float,
    qe_bounds: Tuple[float, float],
) -> float:
    """一维扫描估计命中 target_range 的射角初值（取低伸弹道分支）。"""
    qes = np.linspace(qe_bounds[0], min(45.0, qe_bounds[1]), 16)
    best_qe = float(qes[0])
    best_err = float("inf")
    for qe in qes:
        rng, _, _ = _simulate_range_cross(cfg, float(qe), az_deg, base_az_deg=az_deg)
        err = abs(rng - target_range_m)
        if err < best_err:
            best_err = err
            best_qe = float(qe)
    return best_qe


def solve_firing_solution(
    cfg: SimConfig,
    target_range_m: float,
    target_cross_m: float = 0.0,
    model: str = "mpm",
) -> Dict[str, Any]:
    """二维最小二乘反解命中诸元 (qe, az)。

    Args:
        cfg: 基础 SimConfig（会被本函数覆盖 qe/az）。
        target_range_m: 目标平面射程 (m)。
        target_cross_m: 目标横向偏移 (m)。
        model: 仅保留兼容性，当前统一走 MPM/point_mass。

    Returns:
        dict: qe_deg, az_deg, converged, residual_range_m, residual_cross_m,
        iterations, result。
    """
    if model == "6dof":
        warnings.warn(
            "solve_firing_solution(model='6dof') 为实验性；当前统一走 MPM 反解。",
            stacklevel=2,
        )
    base_az = float(cfg.launch.azimuth_deg)
    qe_bounds = (1.0, 80.0)

    qe0 = _scan_qe_for_range(cfg, target_range_m, base_az, qe_bounds)
    if target_range_m > 1.0:
        az0 = base_az + float(np.rad2deg(np.arctan2(target_cross_m, target_range_m)))
    else:
        az0 = base_az

    def residuals(x: np.ndarray) -> np.ndarray:
        qe, az = float(x[0]), float(x[1])
        rng, cross, res = _simulate_range_cross(cfg, qe, az, base_az_deg=base_az)
        if res is None:
            return np.array([target_range_m, target_cross_m + 1.0e4])
        return np.array([rng - target_range_m, cross - target_cross_m])

    sol = least_squares(
        residuals,
        x0=np.array([qe0, az0]),
        bounds=(
            np.array([qe_bounds[0], base_az - 45.0]),
            np.array([qe_bounds[1], base_az + 45.0]),
        ),
        xtol=1e-4,
        ftol=1e-4,
        diff_step=1e-2,
    )

    qe_best, az_best = float(sol.x[0]), float(sol.x[1])
    rng, cross, result = _simulate_range_cross(cfg, qe_best, az_best, base_az_deg=base_az)
    res_range = rng - target_range_m
    res_cross = cross - target_cross_m

    range_tol = max(RANGE_ABS_TOL_M, RANGE_REL_TOL * abs(target_range_m))
    converged = (
        result is not None
        and abs(res_range) <= range_tol
        and abs(res_cross) <= max(CROSS_ABS_TOL_M, RANGE_REL_TOL * abs(target_range_m))
    )

    return {
        "qe_deg": qe_best,
        "az_deg": az_best,
        "converged": bool(converged),
        "residual_range_m": float(res_range),
        "residual_cross_m": float(res_cross),
        "iterations": int(sol.nfev),
        "result": result,
    }


def solve_firing_solution_latlon(
    cfg: SimConfig,
    target_lat: float,
    target_lon: float,
    model: str = "mpm",
) -> Dict[str, Any]:
    """便捷封装：给目标经纬度，自动算大地线射程与方位后反解命中诸元。"""
    shooter_lat = float(cfg.launch.lat_deg)
    shooter_lon = float(cfg.launch.lon_deg)

    target_range_m = haversine_distance(shooter_lat, shooter_lon, target_lat, target_lon)
    bearing_deg = initial_bearing(shooter_lat, shooter_lon, target_lat, target_lon)

    cfg = apply_overrides(cfg, {"launch.azimuth_deg": float(bearing_deg)})
    out = solve_firing_solution(cfg, target_range_m, 0.0, model=model)
    out["target_range_m"] = float(target_range_m)
    out["bearing_deg"] = float(bearing_deg)
    return out


def verify_solution_6dof(
    cfg: SimConfig,
    qe_deg: float,
    az_deg: float,
    target_range_m: float,
    target_cross_m: float = 0.0,
) -> Dict[str, Any]:
    """用单次仿真校核一组命中诸元（当前与 MPM 复用同一接口）。"""
    base_az = float(cfg.launch.azimuth_deg)
    rng, cross, res = _simulate_range_cross(cfg, qe_deg, az_deg, base_az_deg=base_az)
    res_range = rng - target_range_m
    res_cross = cross - target_cross_m
    range_tol = max(RANGE_ABS_TOL_M, RANGE_REL_TOL * abs(target_range_m))
    within = (
        res is not None
        and abs(res_range) <= range_tol
        and abs(res_cross) <= max(CROSS_ABS_TOL_M, RANGE_REL_TOL * abs(target_range_m))
    )
    return {
        "range_m": float(rng),
        "cross_m": float(cross),
        "residual_range_m": float(res_range),
        "residual_cross_m": float(res_cross),
        "within_tol": bool(within),
        "result": res,
    }
