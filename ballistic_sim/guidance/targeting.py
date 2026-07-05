"""弹道整形与地理 targeting（从 ballistics/shaping.py 迁移）。

- 弹道导弹：最小能量 / 高抛 / 压低 三类弹道反解 kick / gamma。
- 炮弹/MPM：``solve_qe_az`` 反解射角/方位命中给定经纬度目标。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq, least_squares, minimize_scalar

from ballistic_sim.config import apply_overrides
from ballistic_sim.frames import haversine_distance, initial_bearing
from ballistic_sim.guidance.firecontrol import solve_firing_solution_latlon
from ballistic_sim.presets.missiles import missile_config, missile_phases
from ballistic_sim.simulator import simulate

__all__ = [
    "max_range_missile",
    "solve_missile_range",
    "solve_missile_target",
    "solve_qe_az",
]

_RANGE_CEILING_M = 1.6e7
_GEOD_OVER_CHORD_MAX = 1.05
_CHORD_MIN_M = 1.0e4
_T_SHAPE_DUR_DEFAULT = 60.0
_R_EARTH_M = 6371000.0


def _sane_range(res: Any) -> Optional[float]:
    """spike 净化：返回有效大地线射程或 None。"""
    if res is None or res.y.size == 0:
        return None
    # 使用末态 ECEF 位置计算 geodetic 落点
    from ballistic_sim.frames import eci_to_ecef, ecef_to_geodetic

    r_eci = res.y[-1, 0:3]
    r_ecef = eci_to_ecef(r_eci, float(res.t[-1]))
    latf, lonf, _ = ecef_to_geodetic(r_ecef)
    rng = haversine_distance(res.post.get("lat0", 0.0), res.post.get("lon0", 0.0), latf, lonf)
    if not np.isfinite(rng) or rng <= 0.0 or rng > _RANGE_CEILING_M:
        return None
    chord = float(np.linalg.norm(r_ecef - res.post.get("r0_ecef", r_ecef)))
    if chord > _CHORD_MIN_M and rng > _GEOD_OVER_CHORD_MAX * chord:
        return None
    return rng


def _impact_latlon(res: Any) -> Tuple[Optional[float], Optional[float]]:
    """返回末态大地经纬度。"""
    if res is None or res.y.size == 0:
        return None, None
    from ballistic_sim.frames import eci_to_ecef, ecef_to_geodetic

    r_eci = res.y[-1, 0:3]
    r_ecef = eci_to_ecef(r_eci, float(res.t[-1]))
    lat, lon, _ = ecef_to_geodetic(r_ecef)
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return None, None
    return float(lat), float(lon)


def _cross_track_distance(
    lat0_deg: float,
    lon0_deg: float,
    lat1_deg: float,
    lon1_deg: float,
    latp_deg: float,
    lonp_deg: float,
) -> float:
    """计算点 P 相对大圆航线 A->B 的带符号 cross-track 距离 (m)。

    正值表示 P 在航线右侧（自 A 看 B 的顺时针方向），负值表示左侧。
    """
    d13 = haversine_distance(lat0_deg, lon0_deg, latp_deg, lonp_deg)
    theta12 = np.radians(initial_bearing(lat0_deg, lon0_deg, lat1_deg, lon1_deg))
    theta13 = np.radians(initial_bearing(lat0_deg, lon0_deg, latp_deg, lonp_deg))
    dang = d13 / _R_EARTH_M
    cross_ang = np.arcsin(np.clip(np.sin(dang) * np.sin(theta13 - theta12), -1.0, 1.0))
    return float(_R_EARTH_M * cross_ang)


def _simulate_shaped(
    name: str,
    kick_deg: float,
    *,
    az_deg: Optional[float] = None,
    gamma_bo_target_deg: Optional[float] = None,
    t_shape_start: Optional[float] = None,
    t_shape_dur: float = _T_SHAPE_DUR_DEFAULT,
    use_drag: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """跑一次整形后的动力上升 + 滑行。

    通过覆盖导弹预设的 ``kick_deg``（与可选发射方位 ``az_deg``）实现弹道整形。
    ``gamma_bo_target_deg`` 保留给后续上面级/能量整形扩展，当前版本暂不生效。
    """
    cfg = missile_config(name)
    ov = {"guidance.kick_deg": float(kick_deg)}
    if az_deg is not None:
        ov["launch.azimuth_deg"] = float(az_deg)
    if overrides:
        ov.update(overrides)
    cfg = apply_overrides(cfg, ov)

    phases = missile_phases(name)
    for ph in phases:
        g = getattr(ph, "guidance", None)
        if isinstance(g, dict):
            g["kick_deg"] = float(kick_deg)
            if az_deg is not None:
                g["azimuth_deg"] = float(az_deg)
            # 保留扩展字段，但不启用（避免破坏现有开环制导时序）
            if gamma_bo_target_deg is not None:
                g.setdefault("_gamma_bo_target_deg", float(gamma_bo_target_deg))
                start = float(t_shape_start) if t_shape_start is not None else 0.0
                g.setdefault("_t_shape_start", start)
                g.setdefault("_t_shape_dur", float(t_shape_dur))

    for ph in phases:
        dyn = getattr(ph, "dynamics", None)
        if dyn is not None and hasattr(dyn, "modes") and isinstance(dyn.modes, dict):
            dyn.modes["drag"] = bool(use_drag)

    return simulate(cfg, phases=phases)


def _range_for_kick(
    name: str,
    kick_deg: float,
    use_drag: bool,
    overrides: Optional[Dict[str, Any]],
) -> Optional[float]:
    """单次 kick 对应的大地线射程（失败时返回 None）。"""
    try:
        res = _simulate_shaped(
            name,
            kick_deg,
            use_drag=use_drag,
            overrides=overrides,
        )
        return _sane_range(res)
    except Exception:
        return None


def max_range_missile(
    name: str,
    *,
    kick_bounds: Tuple[float, float] = (5.0, 80.0),
    use_drag: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
    gamma_bo_target_deg: Optional[float] = None,
) -> Dict[str, Any]:
    """一维反解某导弹最大射程对应 kick。

    Returns:
        dict: best_kick_deg, max_range_m, result, converged。
    """

    def neg_range(kick: float) -> float:
        rng = _range_for_kick(name, float(kick), use_drag, overrides)
        if rng is None:
            return 1.0e12
        return -float(rng)

    opt = minimize_scalar(
        neg_range,
        bounds=kick_bounds,
        method="bounded",
        options={"xatol": 0.25},
    )
    best_kick = float(opt.x)
    best_range = -float(opt.fun)
    if best_range >= 1.0e11:
        return {
            "best_kick_deg": best_kick,
            "max_range_m": float("nan"),
            "result": None,
            "converged": False,
        }

    # gamma_bo_target_deg 保留扩展：如未来需要可注入 upperstage 制导
    _ = gamma_bo_target_deg

    result = _simulate_shaped(name, best_kick, use_drag=use_drag, overrides=overrides)
    return {
        "best_kick_deg": best_kick,
        "max_range_m": best_range,
        "result": result,
        "converged": _sane_range(result) is not None,
    }


def _scan_kick_range(
    name: str,
    target_range_m: float,
    kick_bounds: Tuple[float, float],
    use_drag: bool,
    overrides: Optional[Dict[str, Any]],
    n_grid: int = 14,
) -> List[Tuple[float, float, float]]:
    """扫描 kick，返回每个采样点的 (kick, range, error)。"""
    kicks = np.linspace(kick_bounds[0], kick_bounds[1], n_grid)
    records: List[Tuple[float, float, float]] = []
    for k in kicks:
        rng = _range_for_kick(name, float(k), use_drag, overrides)
        if rng is None or not np.isfinite(rng):
            continue
        records.append((float(k), rng, rng - target_range_m))
    return records


def _refine_root(
    name: str,
    target_range_m: float,
    k_low: float,
    k_high: float,
    use_drag: bool,
    overrides: Optional[Dict[str, Any]],
) -> Optional[Tuple[float, float, Any]]:
    """在 [k_low, k_high] 内用 brentq 找 range=target 的 kick。"""
    e_low = (_range_for_kick(name, k_low, use_drag, overrides) or 0.0) - target_range_m
    e_high = (_range_for_kick(name, k_high, use_drag, overrides) or 0.0) - target_range_m
    if not np.isfinite(e_low) or not np.isfinite(e_high):
        return None
    if e_low * e_high > 0:
        return None

    def err(k: float) -> float:
        return (_range_for_kick(name, float(k), use_drag, overrides) or 0.0) - target_range_m

    try:
        k_sol = brentq(err, k_low, k_high, xtol=0.25, maxiter=20)
    except ValueError:
        return None
    res = _simulate_shaped(name, float(k_sol), use_drag=use_drag, overrides=overrides)
    rng = _sane_range(res)
    if rng is None:
        return None
    return float(k_sol), rng, res


def solve_missile_range(
    name: str,
    target_range_m: float,
    *,
    branch: str = "minenergy",
    kick_bounds: Tuple[float, float] = (5.0, 80.0),
    use_drag: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
    gamma_bo_target_deg: Optional[float] = None,
    range_abs_tol_m: float = 2000.0,
    range_rel_tol: float = 0.03,
) -> Dict[str, Any]:
    """给定下程射程反解整形参数，区分 minenergy / loft / depressed。

    Returns:
        dict: kick_deg, range_m, residual_range_m, converged, result。
    """
    target_range_m = float(target_range_m)
    tol = max(range_abs_tol_m, range_rel_tol * abs(target_range_m))

    # 1) 粗扫描：同时获得最大射程 kick 与所有过靶 bracket
    records = _scan_kick_range(
        name, target_range_m, kick_bounds, use_drag, overrides, n_grid=10
    )
    if not records:
        return {
            "kick_deg": float("nan"),
            "range_m": float("nan"),
            "residual_range_m": float("nan"),
            "converged": False,
            "result": None,
        }

    roots: List[Tuple[float, float, Any]] = []
    for i in range(len(records) - 1):
        e1 = records[i][2]
        e2 = records[i + 1][2]
        if not np.isfinite(e1) or not np.isfinite(e2):
            continue
        if e1 * e2 <= 0.0 and records[i + 1][0] - records[i][0] > 0.1:
            refined = _refine_root(
                name,
                target_range_m,
                records[i][0],
                records[i + 1][0],
                use_drag,
                overrides,
            )
            if refined is not None:
                roots.append(refined)

    # 2) 无符号穿越：退化为最小绝对误差
    if not roots:
        best = min(records, key=lambda r: abs(r[2]))
        kick_best, rng_best, _ = best
        res = _simulate_shaped(name, kick_best, use_drag=use_drag, overrides=overrides)
        residual = rng_best - target_range_m
        return {
            "kick_deg": kick_best,
            "range_m": rng_best,
            "residual_range_m": residual,
            "converged": abs(residual) <= tol,
            "result": res,
        }

    # 3) 多根分支选择
    roots.sort(key=lambda x: x[0])
    if branch == "depressed":
        kick, rng, res = roots[0]
    elif branch == "loft":
        kick, rng, res = roots[-1]
    else:  # minenergy：取最低 kick 解（压低弹道能量更小）
        kick, rng, res = roots[0]

    # gamma_bo_target_deg 保留扩展
    _ = gamma_bo_target_deg

    residual = rng - target_range_m
    return {
        "kick_deg": kick,
        "range_m": rng,
        "residual_range_m": residual,
        "converged": abs(residual) <= tol,
        "result": res,
    }


def solve_missile_target(
    name: str,
    target_lat: float,
    target_lon: float,
    *,
    branch: str = "loft",
    use_drag: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
    az_window_deg: float = 25.0,
    kick_bounds: Tuple[float, float] = (5.0, 80.0),
    range_abs_tol_m: float = 2000.0,
    range_rel_tol: float = 0.03,
    cross_abs_tol_m: float = 2000.0,
    max_range: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """反解 (发射方位 az, 整形 kick) 命中目标经纬度。

    先按大圆射程调用 ``solve_missile_range`` 得到 kick，再微调 azimuth 使
    cross-track 误差收敛。
    """
    m = missile_config(name)
    lat0 = float(m.launch.lat_deg)
    lon0 = float(m.launch.lon_deg)

    target_range_m = haversine_distance(lat0, lon0, target_lat, target_lon)
    bearing_deg = initial_bearing(lat0, lon0, target_lat, target_lon)

    # 先用射程反解得到 kick 初值
    rng_sol = solve_missile_range(
        name,
        target_range_m,
        branch=branch,
        kick_bounds=kick_bounds,
        use_drag=use_drag,
        overrides=overrides,
        range_abs_tol_m=range_abs_tol_m,
        range_rel_tol=range_rel_tol,
    )
    if rng_sol["result"] is None:
        return {
            "az_deg": bearing_deg,
            "kick_deg": rng_sol["kick_deg"],
            "target_range_m": target_range_m,
            "bearing_deg": bearing_deg,
            "converged": False,
            "residual_range_m": rng_sol["residual_range_m"],
            "residual_cross_m": float("nan"),
            "result": None,
        }

    kick0 = float(rng_sol["kick_deg"])
    az0 = float(bearing_deg)

    def residuals(x: np.ndarray) -> np.ndarray:
        kick = float(np.clip(x[0], kick_bounds[0], kick_bounds[1]))
        az = az0 + float(np.clip(x[1], -az_window_deg, az_window_deg))
        try:
            res = _simulate_shaped(
                name,
                kick,
                az_deg=az,
                use_drag=use_drag,
                overrides=overrides,
            )
        except Exception:
            return np.array([target_range_m, cross_abs_tol_m + 1.0e4])
        latp, lonp = _impact_latlon(res)
        if latp is None or lonp is None:
            return np.array([target_range_m, cross_abs_tol_m + 1.0e4])
        d13 = haversine_distance(lat0, lon0, latp, lonp)
        cross = _cross_track_distance(lat0, lon0, target_lat, target_lon, latp, lonp)
        # 沿大圆航线的下程残差
        ratio = np.clip(
            np.cos(d13 / _R_EARTH_M) / max(np.cos(cross / _R_EARTH_M), 1e-12),
            -1.0,
            1.0,
        )
        along = _R_EARTH_M * np.arccos(ratio)
        return np.array([along - target_range_m, cross])

    sol = least_squares(
        residuals,
        x0=np.array([kick0, 0.0]),
        bounds=(
            np.array([kick_bounds[0], -az_window_deg]),
            np.array([kick_bounds[1], az_window_deg]),
        ),
        xtol=1e-3,
        ftol=1e-3,
        diff_step=np.array([0.5, 0.5]),
        max_nfev=40,
    )

    kick_best = float(sol.x[0])
    az_best = az0 + float(sol.x[1])
    res_best = _simulate_shaped(
        name,
        kick_best,
        az_deg=az_best,
        use_drag=use_drag,
        overrides=overrides,
    )
    latp, lonp = _impact_latlon(res_best)
    if latp is None or lonp is None:
        return {
            "az_deg": az_best,
            "kick_deg": kick_best,
            "target_range_m": target_range_m,
            "bearing_deg": bearing_deg,
            "converged": False,
            "residual_range_m": float("nan"),
            "residual_cross_m": float("nan"),
            "result": res_best,
        }

    d13 = haversine_distance(lat0, lon0, latp, lonp)
    cross = _cross_track_distance(lat0, lon0, target_lat, target_lon, latp, lonp)
    ratio = np.clip(np.cos(d13 / _R_EARTH_M) / max(np.cos(cross / _R_EARTH_M), 1e-12), -1.0, 1.0)
    along = _R_EARTH_M * np.arccos(ratio)
    res_range = along - target_range_m
    res_cross = cross

    range_tol = max(range_abs_tol_m, range_rel_tol * abs(target_range_m))
    converged = (
        abs(res_range) <= range_tol
        and abs(res_cross) <= cross_abs_tol_m
    )

    # max_range 参数保留兼容，供调用方复用最大射程信息
    _ = max_range

    return {
        "az_deg": float(az_best),
        "kick_deg": float(kick_best),
        "target_range_m": float(target_range_m),
        "bearing_deg": float(bearing_deg),
        "converged": bool(converged),
        "residual_range_m": float(res_range),
        "residual_cross_m": float(res_cross),
        "result": res_best,
    }


def solve_qe_az(
    lat: float,
    lon: float,
    target_lat: float,
    target_lon: float,
    *,
    cfg: Optional[Any] = None,
) -> Dict[str, Any]:
    """为 MPM/point_mass 模型反解射角/方位，命中给定地面目标。

    Args:
        lat/lon: 发射点经纬度 (deg)。
        target_lat/target_lon: 目标点经纬度 (deg)。
        cfg: 可选的基础 SimConfig；缺省时构造一个典型 M107 类似的配置。

    Returns:
        dict: qe_deg, az_deg, target_range_m, bearing_deg, converged, result 等。
    """
    from ballistic_sim.config import (
        EnvironmentConfig,
        GuidanceConfig,
        LaunchConfig,
        OptionsConfig,
        SimConfig,
        VehicleConfig,
    )

    if cfg is None:
        cfg = SimConfig(
            mission="projectile",
            vehicle=VehicleConfig(mass_kg=43.2, diameter_m=0.155, cd=0.78),
            launch=LaunchConfig(
                lat_deg=float(lat),
                lon_deg=float(lon),
                alt_m=0.0,
                azimuth_deg=0.0,
                elevation_deg=45.0,
                v0_m_s=684.0,
            ),
            environment=EnvironmentConfig(atmosphere="isa", wind_m_s=[0.0, 0.0, 0.0]),
            guidance=GuidanceConfig(),
            options=OptionsConfig(integrator="DOP853", terminate_impact=True),
        )
    else:
        cfg = cfg.model_copy(deep=True)
        cfg.launch.lat_deg = float(lat)
        cfg.launch.lon_deg = float(lon)

    target_range_m = haversine_distance(lat, lon, target_lat, target_lon)
    bearing_deg = initial_bearing(lat, lon, target_lat, target_lon)

    out = solve_firing_solution_latlon(cfg, target_lat, target_lon, model="mpm")
    return {
        "qe_deg": float(out["qe_deg"]),
        "az_deg": float(out["az_deg"]),
        "target_range_m": float(target_range_m),
        "bearing_deg": float(bearing_deg),
        "converged": bool(out["converged"]),
        "residual_range_m": float(out.get("residual_range_m", 0.0)),
        "residual_cross_m": float(out.get("residual_cross_m", 0.0)),
        "result": out.get("result"),
    }
