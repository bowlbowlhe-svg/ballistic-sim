"""FastAPI Web API for ballistic_sim."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ballistic_sim.api.dependencies import require_fastapi
from ballistic_sim.api.models import (
    FireControlRequest,
    FireControlResponse,
    HealthResponse,
    MonteCarloRequest,
    MonteCarloResponse,
    SimulateRequest,
    SimulateResponse,
    Trajectory3DRequest,
    Trajectory3DResponse,
)
from ballistic_sim.config import SimConfig, ValidationIssue, apply_overrides, validate_config
from ballistic_sim.monte_carlo import PerturbationConfig, monte_carlo_simulation

from ballistic_sim.presets import (
    list_missiles,
    m107_config,
)
from ballistic_sim.simulator import SimResult, simulate
from ballistic_sim.viz import (
    attach_launch_lla,
    detect_frame,
    downrange,
    geodetic_coords,
    speed,
)
from ballistic_sim.viz.earth import result_to_ecef

require_fastapi()
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402


def _format_validation_issues(issues: List[ValidationIssue]) -> str:
    """将 ValidationIssue 列表格式化为可读消息。"""
    lines = ["SimConfig validation failed:"]
    for issue in issues:
        lines.append(f"  [{issue.severity}] {issue.path}: {issue.message}")
    return "\n".join(lines)


def _check_config_or_raise(cfg: SimConfig) -> None:
    """校验配置；若存在 ERROR 则抛出 HTTPException(400)。"""
    issues = validate_config(cfg)
    errors = [i for i in issues if i.severity == "ERROR"]
    if errors:
        raise HTTPException(status_code=400, detail=_format_validation_issues(issues))


def _web_root() -> Path:
    """Return the project ``web/`` directory regardless of CWD."""
    return Path(__file__).resolve().parent.parent.parent / "web"


def _sim_request_from_trajectory_request(request: Trajectory3DRequest) -> SimulateRequest:
    """将 3D 可视化请求体转换为普通仿真请求体。"""
    return SimulateRequest(
        vehicle=request.vehicle,
        launch=request.launch,
        environment=request.environment,
        guidance=request.guidance,
        options=request.options,
        preset=request.preset,
        include_trajectory=False,
    )


def _merge_request(cfg: SimConfig, request: SimulateRequest) -> SimConfig:
    """Merge explicitly-set request fields into a base SimConfig."""
    overrides: Dict[str, Any] = {}
    if request.vehicle is not None:
        for key, value in request.vehicle.model_dump(mode="json", exclude_unset=True).items():
            overrides[f"vehicle.{key}"] = value
    if request.launch is not None:
        for key, value in request.launch.model_dump(mode="json", exclude_unset=True).items():
            overrides[f"launch.{key}"] = value
    if request.environment is not None:
        for key, value in request.environment.model_dump(mode="json", exclude_unset=True).items():
            overrides[f"environment.{key}"] = value
    if request.guidance is not None:
        for key, value in request.guidance.model_dump(mode="json", exclude_unset=True).items():
            overrides[f"guidance.{key}"] = value
    if request.options is not None:
        for key, value in request.options.model_dump(mode="json", exclude_unset=True).items():
            overrides[f"options.{key}"] = value
    if overrides:
        return apply_overrides(cfg, overrides)
    return cfg


def _build_config(
    mission: str,
    request: SimulateRequest,
) -> SimConfig:
    """Build a SimConfig for the requested mission."""
    if mission == "projectile":
        preset = request.preset or "M107"
        if preset == "M107":
            cfg = m107_config()
        else:
            from ballistic_sim.presets.projectiles import _projectile_config_from_preset

            cfg = _projectile_config_from_preset(preset)
    elif mission == "missile":
        name = request.preset or list_missiles()[0]
        from ballistic_sim.presets.missiles import missile_full_config

        cfg = missile_full_config(name)
    elif mission == "rocket":
        name = request.preset or "CZ2F"
        from ballistic_sim.presets.rockets import rocket_full_config

        cfg = rocket_full_config(name)
    elif mission == "icbm":
        if request.preset:
            from ballistic_sim.presets.missiles import missile_full_config

            cfg = missile_full_config(request.preset)
        else:
            cfg = SimConfig(mission=mission)
    elif mission == "suborbital":
        cfg = SimConfig(mission=mission)
    else:
        raise ValueError(f"未支持的任务类型: {mission}")
    cfg = _merge_request(cfg, request)
    return cfg


def _compute_summary(cfg: SimConfig, result: SimResult) -> Dict[str, Any]:
    """Compute a JSON-friendly summary from a SimResult."""
    attach_launch_lla(result, cfg.launch.lat_deg, cfg.launch.lon_deg, cfg.launch.alt_m)
    summary: Dict[str, Any] = {
        "mission": cfg.mission,
        "stop_reason": result.stop_reason,
        "frame": detect_frame(result),
        "n_points": int(result.t.size),
        "phase_bounds": [float(x) for x in result.phase_bounds],
        "events": [
            {"phase": ev.get("phase"), "name": ev.get("name"), "t": ev.get("t")}
            for ev in result.event_log
        ],
    }
    if result.y.size == 0:
        return summary

    t_end = float(result.t[-1]) if result.t.size else None
    lat, lon, alt = geodetic_coords(result)
    rng = downrange(result)
    summary.update(
        {
            "t_end_s": t_end,
            "range_m": float(rng[-1]) if rng.size else None,
            "apogee_m": float(np.max(alt)) if alt.size else None,
            "impact_lat": float(lat[-1]) if lat.size else None,
            "impact_lon": float(lon[-1]) if lon.size else None,
            "impact_speed_m_s": float(speed(result)[-1]) if result.t.size else None,
        }
    )
    return summary


def _build_trajectory(result: SimResult) -> Optional[Dict[str, List[float]]]:
    """Build a JSON-friendly trajectory series if data is available."""
    if result.y.size == 0 or result.t.size == 0:
        return None
    lat, lon, alt = geodetic_coords(result)
    rng = downrange(result)
    spd = speed(result)
    return {
        "t": [float(x) for x in result.t],
        "lat": [float(x) for x in lat],
        "lon": [float(x) for x in lon],
        "alt_m": [float(x) for x in alt],
        "range_m": [float(x) for x in rng],
        "speed_m_s": [float(x) for x in spd],
    }


def _run_simulation(
    mission: str,
    request: SimulateRequest,
) -> SimulateResponse:
    """Run a single simulation and package the response."""
    cfg = _build_config(mission, request)
    _check_config_or_raise(cfg)
    result = simulate(cfg)
    summary = _compute_summary(cfg, result)
    trajectory: Optional[Dict[str, List[float]]] = None
    if request.include_trajectory:
        trajectory = _build_trajectory(result)
    return SimulateResponse(
        mission=cfg.mission,
        stop_reason=result.stop_reason,
        t_end_s=summary.get("t_end_s"),
        range_m=summary.get("range_m"),
        apogee_m=summary.get("apogee_m"),
        impact_lat=summary.get("impact_lat"),
        impact_lon=summary.get("impact_lon"),
        summary=summary,
        trajectory=trajectory,
    )


def _run_monte_carlo(request: MonteCarloRequest) -> MonteCarloResponse:
    """Run a Monte Carlo dispersion study and package the response."""
    sim_request = SimulateRequest(
        vehicle=request.vehicle,
        launch=request.launch,
        environment=request.environment,
        guidance=request.guidance,
        options=request.options,
        preset=request.preset,
        include_trajectory=False,
    )
    cfg = _build_config(request.mission, sim_request)
    _check_config_or_raise(cfg)
    if request.backend in ("batch", "gpu"):
        cfg = apply_overrides(
            cfg,
            {"options.mpm_use_spin": False, "options.mpm_use_dynamic_alpha": False},
        )
    perturb = request.perturbations if request.perturbations is not None else PerturbationConfig()
    mc = monte_carlo_simulation(
        cfg,
        perturb=perturb,
        n_samples=request.n_samples,
        backend=request.backend,
        n_jobs=request.n_jobs,
        seed=request.seed,
    )
    return MonteCarloResponse(
        samples=int(mc.samples),
        range_mean_m=float(mc.range_mean),
        range_std_m=float(mc.range_std),
        cross_mean_m=float(mc.cross_mean),
        cross_std_m=float(mc.cross_std),
        cep50_m=float(mc.cep50),
        cep90_m=float(mc.cep90),
        ellipse_major_m=float(mc.ellipse_major),
        ellipse_minor_m=float(mc.ellipse_minor),
        ellipse_angle_deg=float(mc.ellipse_angle),
    )


def _solve_fire_control(request: FireControlRequest) -> FireControlResponse:
    """Solve firing solution for projectile/missile."""
    from ballistic_sim.guidance.firecontrol import solve_firing_solution_latlon

    sim_request = SimulateRequest(
        vehicle=request.vehicle,
        launch=request.launch,
        environment=request.environment,
        guidance=request.guidance,
        options=request.options,
        preset=request.preset,
        include_trajectory=False,
    )
    cfg = _build_config(request.mission, sim_request)
    _check_config_or_raise(cfg)
    try:
        sol = solve_firing_solution_latlon(
            cfg,
            float(request.target_lat_deg),
            float(request.target_lon_deg),
            model="mpm",
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"火控求解失败: {exc}") from exc

    return FireControlResponse(
        mission=request.mission,
        qe_deg=sol.get("qe_deg"),
        az_deg=sol.get("az_deg"),
        converged=bool(sol.get("converged", False)),
        target_lat_deg=request.target_lat_deg,
        target_lon_deg=request.target_lon_deg,
        target_range_m=sol.get("target_range_m"),
        bearing_deg=sol.get("bearing_deg"),
        residual_range_m=sol.get("residual_range_m"),
        residual_cross_m=sol.get("residual_cross_m"),
        iterations=sol.get("iterations"),
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Ballistic Sim API",
        description="Web API for ballistic flight simulation.",
        version="0.7.2",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        from ballistic_sim import __version__

        return HealthResponse(status="ok", version=__version__)

    @app.post("/simulate/monte-carlo", response_model=MonteCarloResponse)
    def monte_carlo_endpoint(request: MonteCarloRequest) -> MonteCarloResponse:
        try:
            return _run_monte_carlo(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/simulate/{mission}", response_model=SimulateResponse)
    def simulate_endpoint(mission: str, request: SimulateRequest) -> SimulateResponse:
        try:
            return _run_simulation(mission, request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/config/validate", response_model=List[ValidationIssue])
    def validate_endpoint(cfg: SimConfig) -> List[ValidationIssue]:
        """接收完整 SimConfig JSON，返回结构化业务校验结果。"""
        return validate_config(cfg)

    @app.post("/firecontrol/solve", response_model=FireControlResponse)
    def firecontrol_endpoint(request: FireControlRequest) -> FireControlResponse:
        if request.mission not in ("projectile", "missile"):
            raise HTTPException(
                status_code=501,
                detail="当前仅 projectile/missile 支持火控求解",
            )
        return _solve_fire_control(request)

    @app.post("/viz/trajectory3d", response_model=Trajectory3DResponse)
    def trajectory3d_endpoint(request: Trajectory3DRequest) -> Trajectory3DResponse:
        """接收仿真参数并返回 3D 轨迹 JSON 点序列或 HTML 字符串。"""
        sim_request = _sim_request_from_trajectory_request(request)
        try:
            cfg = _build_config(request.mission, sim_request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        _check_config_or_raise(cfg)
        result = simulate(cfg)

        if request.format == "html":
            try:
                from ballistic_sim.viz.interactive3d import plot_trajectory_3d
            except ImportError as exc:
                raise HTTPException(
                    status_code=501,
                    detail=f"3D 可视化依赖缺失: {exc}",
                ) from exc
            fig = plot_trajectory_3d(
                result,
                show_earth=request.include_earth,
                show_coastlines=request.include_earth,
            )
            html = fig.to_html(full_html=True, include_plotlyjs="cdn")
            return Trajectory3DResponse(
                mission=cfg.mission,
                stop_reason=result.stop_reason,
                format="html",
                html=html,
            )

        x_m, y_m, z_m, alt_m = result_to_ecef(result)
        return Trajectory3DResponse(
            mission=cfg.mission,
            stop_reason=result.stop_reason,
            format="json",
            x_km=[float(v) for v in x_m / 1e3],
            y_km=[float(v) for v in y_m / 1e3],
            z_km=[float(v) for v in z_m / 1e3],
            alt_m=[float(v) for v in alt_m],
        )

    web_root = _web_root()
    if web_root.is_dir():
        app.mount("/", StaticFiles(directory=str(web_root), html=True), name="static")

    return app
