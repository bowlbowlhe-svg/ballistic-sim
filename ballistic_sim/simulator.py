"""统一多阶段仿真器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.integrate import solve_ivp

from ballistic_sim.config import SimConfig, apply_overrides
from ballistic_sim.dynamics.common import DynamicContext
from ballistic_sim.models.terrain import NullTerrainModel, TerrainExtent, make_terrain
from ballistic_sim.models.wind import UniformWind, WindModel, make_wind, parse_wind_profile
from ballistic_sim.phases.base import Phase
from ballistic_sim.state_switch import project_state


@dataclass
class SimResult:
    """统一仿真结果。"""

    t: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    y: np.ndarray = field(default_factory=lambda: np.array([], dtype=float).reshape(0, 0))
    event_log: List[Dict[str, Any]] = field(default_factory=list)
    post: Dict[str, Any] = field(default_factory=dict)
    stop_reason: str = ""
    phase_bounds: List[float] = field(default_factory=list)


def _resolve_wind(cfg: SimConfig) -> WindModel:
    """由 ``EnvironmentConfig`` 构建风场模型。"""
    env = cfg.environment
    name = env.wind_model.lower()

    if name in ("none", "null"):
        return make_wind("none")

    if name == "uniform":
        w = env.wind_m_s if env.wind_m_s else [0.0, 0.0, 0.0]
        return UniformWind(e=w[0], n=w[1], u=w[2])

    if name == "log":
        return make_wind(
            "log",
            u_ref=env.wind_u_ref,
            z_ref=env.wind_z_ref,
            z0=env.wind_z0,
            direction_deg=env.wind_direction_deg,
        )

    if name == "power":
        return make_wind(
            "power",
            u_ref=env.wind_u_ref,
            z_ref=env.wind_z_ref,
            alpha=env.wind_alpha,
            direction_deg=env.wind_direction_deg,
        )

    if name == "profile":
        text = env.wind_profile_text
        if text is None and env.wind_profile_path:
            with open(env.wind_profile_path, "r", encoding="utf-8") as f:
                text = f.read()
        if text is None:
            raise ValueError("profile 风模型需要提供 wind_profile_text 或 wind_profile_path")
        profile = parse_wind_profile(text)
        return make_wind("profile", profile=profile)

    if name == "composite":
        boundary = make_wind(
            "log",
            u_ref=env.wind_u_ref,
            z_ref=env.wind_z_ref,
            z0=env.wind_z0,
            direction_deg=env.wind_direction_deg,
        )
        profile_model: Optional[Any] = None
        if env.wind_profile_text or env.wind_profile_path:
            text = env.wind_profile_text
            if text is None and env.wind_profile_path:
                with open(env.wind_profile_path, "r", encoding="utf-8") as f:
                    text = f.read()
            if text is not None:
                profile_model = make_wind("profile", profile=parse_wind_profile(text))
        return make_wind("composite", boundary_layer=boundary, profile=profile_model)

    if name == "dryden":
        base = _resolve_wind(
            apply_overrides(
                cfg,
                {"environment.wind_model": "uniform"},
            )
        )
        return make_wind("dryden", base_wind=base, seed=env.wind_dryden_seed)

    raise ValueError(f"未知风场模型: {env.wind_model}")


def _resolve_terrain(cfg: SimConfig):
    """由 ``EnvironmentConfig`` 构建地形模型。"""
    env = cfg.environment
    if not env.terrain_enabled:
        return NullTerrainModel()

    name = env.terrain_model.lower()
    # 兼容旧版 use_terrain=True 但未指定 terrain_model 的配置，默认使用 hilly
    if name in ("null", "none") and env.use_terrain:
        name = "hilly"
    extent = None
    if env.terrain_extent is not None:
        lat_min, lat_max, lon_min, lon_max = env.terrain_extent
        extent = TerrainExtent(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)

    kwargs: Dict[str, Any] = {}
    if env.terrain_max_height is not None:
        kwargs["max_height"] = env.terrain_max_height

    path = env.terrain_path
    if name == "hilly":
        return make_terrain("hilly", extent=extent or TerrainExtent(0.0, 1.0, 0.0, 1.0), **kwargs)
    if name in ("numpy", "image", "geotiff", "srtm_dir", "srtm_files"):
        if path is None:
            raise ValueError(f"{env.terrain_model} 地形模型需要提供 terrain_path")
        return make_terrain(name, path=path, extent=extent, **kwargs)

    raise ValueError(f"未知地形模型: {env.terrain_model}")


def _resolve_dynamics_context(cfg: SimConfig) -> DynamicContext:
    """由 ``SimConfig`` 构建动力学上下文。"""
    from ballistic_sim.models.atmosphere import make_atmosphere
    from ballistic_sim.models.aerodynamics import make_aero

    atm = make_atmosphere(
        cfg.environment.atmosphere,
        delta_t=cfg.environment.delta_t,
        density_factor=cfg.environment.density_factor,
    )
    wind = _resolve_wind(cfg)
    terrain = _resolve_terrain(cfg)
    if cfg.mission == "projectile":
        aero = make_aero("g1", cl_slope=0.0)
    elif cfg.mission in ("rocket", "icbm", "missile", "suborbital"):
        aero = make_aero("rocket")
    else:
        aero = make_aero("constant", cd=cfg.vehicle.cd)
    return DynamicContext(
        cfg=cfg,
        atmosphere=atm,
        wind=wind,
        aero=aero,
        terrain=terrain,
        gravity_model=cfg.environment.gravity_model,
        options={},
    )


def simulate(cfg: SimConfig, phases: List[Phase]) -> SimResult:
    """统一仿真主循环。

    按 ``phases`` 顺序逐段调用 ``solve_ivp``，段间通过 ``project_state`` 映射状态，
    拼接全程轨迹并记录事件日志。
    """
    from ballistic_sim.phases.builder import build_phases

    if not phases:
        phases = build_phases(cfg)

    dyn_ctx = _resolve_dynamics_context(cfg)
    cfg._dynamics_context = dyn_ctx  # type: ignore[attr-defined]

    method = cfg.options.integrator
    rtol = cfg.options.rtol
    atol = cfg.options.atol
    max_step = cfg.options.max_step

    # 初始状态由第一个 phase 的动力学模块提供
    first = phases[0]
    y0 = first.dynamics.initial_state(
        v0=cfg.launch.v0_m_s,
        theta_deg=cfg.launch.elevation_deg,
        az_deg=cfg.launch.azimuth_deg,
        h0=cfg.launch.alt_m,
    )
    t_abs = cfg.launch.t0_s

    t_all = [t_abs]
    y_all: List[np.ndarray] = [np.asarray(y0, dtype=float).reshape(1, -1)]
    event_log: List[Dict[str, Any]] = []
    phase_bounds: List[float] = [t_abs]
    stop_reason = "completed"

    for ip, ph in enumerate(phases):
        if ph.is_terminal:
            break

        t_start = t_abs
        t_end_max = t_start + (ph.t_span[1] - ph.t_span[0])
        # 上面级制导装订：进入时刻与实测飞行路径角
        if ph.guidance is not None:
            if "t_us_start" in ph.guidance:
                ph.guidance["t_us_start"] = float(t_start)
            if ph.guidance.get("gamma0_deg") is None:
                r0 = y0[0:3]
                v0 = y0[3:6]
                vn = float(np.linalg.norm(v0))
                up = r0 / max(np.linalg.norm(r0), 1e-12)
                gamma0 = (
                    float(np.arcsin(np.clip(np.dot(v0, up) / vn, -1.0, 1.0))) if vn > 1.0 else 0.0
                )
                ph.guidance["gamma0_deg"] = float(np.degrees(gamma0))

        def _rhs(t: float, y: np.ndarray) -> np.ndarray:
            return ph.rhs(t, y, cfg, ph)

        events = ph.events if ph.events else None
        sol = solve_ivp(
            _rhs,
            (t_start, t_end_max),
            y0,
            method=method,
            events=events,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            dense_output=True,
        )

        if not sol.success:
            stop_reason = f"integration_failed@{ph.name}"
            break

        # 记录事件（process_events 已返回绝对时刻）
        for ev in ph.process_events(sol):
            ev["t"] = ev.get("t", float(sol.t[-1]))
            event_log.append(ev)

        y_end = ph.final_state(sol)
        t_abs = float(sol.t[-1])

        # 轨迹拼接（按约 0.5 s 重采样，含端点）
        t_loc = sol.t
        if len(t_loc) < 2:
            t_loc = np.array([0.0, sol.t[-1]])
            y_loc = np.column_stack([y0, y_end])
        else:
            y_loc = sol.y
        # 避免首点重复
        if t_all and abs(t_loc[0] - t_all[-1]) < 1e-9:
            t_loc = t_loc[1:]
            y_loc = y_loc[:, 1:]
        t_all.extend(t_loc.tolist())
        y_all.append(np.asarray(y_loc.T, dtype=float))

        phase_bounds.append(t_abs)

        # 阶段间状态投影
        if ip + 1 < len(phases):
            next_ph = phases[ip + 1]
            src_dim = ph.state_dim()
            dst_dim = next_ph.state_dim()
            src_frame = ph.native_frame()
            dst_frame = next_ph.native_frame()
            if src_dim != dst_dim or src_frame != dst_frame:
                extras = getattr(next_ph, "projection_extras", {})
                y0 = project_state(
                    y_end,
                    src_dim=src_dim,
                    dst_dim=dst_dim,
                    src_frame=src_frame,
                    dst_frame=dst_frame,
                    lat_deg=cfg.launch.lat_deg,
                    lon_deg=cfg.launch.lon_deg,
                    h0=cfg.launch.alt_m,
                    t=t_abs,
                    **extras,
                )
            else:
                y0 = y_end.copy()
        else:
            y0 = y_end.copy()

    # 合并轨迹
    y_array = np.concatenate(y_all, axis=0)
    t_array = np.array(t_all, dtype=float)

    result = SimResult(
        t=t_array,
        y=y_array,
        event_log=event_log,
        stop_reason=stop_reason,
        phase_bounds=phase_bounds,
    )
    result.post = _postprocess(cfg, result)
    return result


def _postprocess(cfg: SimConfig, result: SimResult) -> Dict[str, Any]:
    """简单后处理。"""
    if result.y.size == 0:
        return {}
    from ballistic_sim.frames import geodetic_to_ecef

    lat0 = float(cfg.launch.lat_deg)
    lon0 = float(cfg.launch.lon_deg)
    r0_ecef = geodetic_to_ecef(lat0, lon0, float(cfg.launch.alt_m))
    y_end = result.y[-1]
    out: Dict[str, Any] = {
        "lat0": lat0,
        "lon0": lon0,
        "r0_ecef": r0_ecef,
        "t_end_s": float(result.t[-1]),
    }
    if y_end.size >= 6:
        out["r_end_m"] = y_end[0:3].tolist()
        out["v_end_m_s"] = y_end[3:6].tolist()
    return out
