"""弹道整形与地理 targeting（从 ballistics/shaping.py 迁移）。

- 弹道导弹：最小能量 / 高抛 / 压低 三类弹道反解 kick / gamma。
- 炮弹/MPM：``solve_qe_az`` 反解射角/方位命中给定经纬度目标。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ballistic_sim.frames import haversine_distance, initial_bearing
from ballistic_sim.guidance.firecontrol import solve_firing_solution_latlon

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


def _simulate_shaped(
    name: str,
    kick_deg: float,
    *,
    gamma_bo_target_deg: Optional[float] = None,
    t_shape_start: Optional[float] = None,
    t_shape_dur: float = _T_SHAPE_DUR_DEFAULT,
    use_drag: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
):
    """跑一次整形后的动力上升 + 滑行（占位：需调用方提供导弹预设）。"""
    raise NotImplementedError(
        "_simulate_shaped 需要项目阶段 4 的导弹预设与 PoweredECI 装配链；"
        "当前保留函数签名供后续填充。"
    )


def max_range_missile(
    name: str,
    *,
    kick_bounds: Tuple[float, float] = (5.0, 80.0),
    use_drag: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
    gamma_bo_target_deg: Optional[float] = None,
) -> Dict[str, Any]:
    """一维反解某导弹最大射程对应 kick（占位）。"""
    raise NotImplementedError("max_range_missile 待阶段 4 导弹预设就绪后实现")


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
    """给定下程射程反解整形参数，区分 minenergy / loft / depressed（占位）。"""
    raise NotImplementedError("solve_missile_range 待阶段 4 导弹预设就绪后实现")


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
    """反解 (发射方位 az, 整形 kick) 命中目标经纬度（占位）。"""
    raise NotImplementedError("solve_missile_target 待阶段 4 导弹预设就绪后实现")


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
