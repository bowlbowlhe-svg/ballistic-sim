"""统一多阶段仿真器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
from scipy.integrate import solve_ivp

from ballistic_sim.config import SimConfig
from ballistic_sim.dynamics.common import DynamicContext
from ballistic_sim.models.atmosphere import StandardAtmosphere
from ballistic_sim.models.wind import UniformWind
from ballistic_sim.models.wind import WindModel
from ballistic_sim.phases.base import Phase, PhaseContext
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


def _resolve_dynamics_context(cfg: SimConfig) -> DynamicContext:
    """由 ``SimConfig`` 构建动力学上下文。"""
    from ballistic_sim.models.atmosphere import make_atmosphere
    from ballistic_sim.models.wind import make_wind
    from ballistic_sim.models.aerodynamics import make_aero

    atm = make_atmosphere(
        cfg.environment.atmosphere,
    )
    wind = make_wind("uniform")
    if cfg.environment.wind_m_s:
        w = cfg.environment.wind_m_s
        wind = UniformWind(e=w[0], n=w[1], u=w[2])
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
    y_all = [np.asarray(y0, dtype=float).reshape(1, -1)]
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
        ctx = PhaseContext(cfg=cfg, phase=ph, t0=t_start)

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

        # 记录事件
        for ev in ph.process_events(sol):
            ev["t"] = ev.get("t", float(sol.t[-1])) + t_start
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
        y_all.append(y_loc.T)

        phase_bounds.append(t_abs)

        # 阶段间状态投影
        if ip + 1 < len(phases):
            next_ph = phases[ip + 1]
            src_dim = ph.state_dim()
            dst_dim = next_ph.state_dim()
            src_frame = ph.native_frame()
            dst_frame = next_ph.native_frame()
            if src_dim != dst_dim or src_frame != dst_frame:
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
    y_end = result.y[-1]
    if y_end.size >= 7:
        r_end = y_end[0:3]
        v_end = y_end[3:6]
        return {
            "r_end_m": r_end.tolist(),
            "v_end_m_s": v_end.tolist(),
            "t_end_s": float(result.t[-1]),
        }
    return {}
