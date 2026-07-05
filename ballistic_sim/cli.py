"""统一命令行入口。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ballistic_sim.config import (
    GuidanceConfig,
    LaunchConfig,
    SimConfig,
    apply_overrides,
)
from ballistic_sim.dynamics.common import rv_to_oe
from ballistic_sim.frames import ecef_to_geodetic, eci_to_ecef, haversine_distance
from ballistic_sim.presets import (
    list_missiles,
    missile_config,
    missile_phases,
    m107_config,
    m107_phases,
    projectile_phases,
    rocket_config,
    rocket_phases,
)
from ballistic_sim.simulator import SimResult, simulate
from ballistic_sim.monte_carlo import PerturbationConfig, monte_carlo_simulation
from ballistic_sim.viz import (
    altitude,
    attach_launch_lla,
    detect_frame,
    plot_dispersion,
)


def _default_out_dir(mission: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("out") / f"{mission}_{ts}"


def _normalize_rocket_name(name: str) -> str:
    """火箭名称别名映射。"""
    aliases = {
        "CZ-2F": "CZ2F",
        "CZ-2C": "CZ2C",
        "LONG-MARCH-2F": "CZ2F",
        "LONG-MARCH-2C": "CZ2C",
    }
    return aliases.get(name.upper(), name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ballistic Sim — 统一弹道仿真 CLI")
    parser.add_argument(
        "--mission",
        required=False,
        choices=["projectile", "missile", "rocket", "icbm", "suborbital"],
        help="任务类型",
    )
    parser.add_argument("--config", type=str, help="YAML/JSON 配置文件路径")
    parser.add_argument("--preset", type=str, help="弹丸/火炮预设名")
    parser.add_argument("--rocket", type=str, help="火箭预设名")
    parser.add_argument("--missile", type=str, help="导弹预设名")
    parser.add_argument("--qe", type=float, help="发射仰角 (deg)")
    parser.add_argument("--az", type=float, help="发射方位角 (deg)")
    parser.add_argument("--target-lat", type=float, help="目标纬度 (deg)")
    parser.add_argument("--target-lon", type=float, help="目标经度 (deg)")
    parser.add_argument("--viz", action="store_true", default=True, help="启用可视化 (默认开启)")
    parser.add_argument("--no-viz", action="store_true", help="关闭可视化")
    parser.add_argument("--out-dir", type=str, help="输出目录 (默认 out/<mission>_<timestamp>/)")
    parser.add_argument("--no-summary", action="store_true", help="不保存 result_summary.json")
    parser.add_argument("--monte-carlo", action="store_true", help="启用 Monte Carlo 散布分析")
    parser.add_argument(
        "--mc-backend", type=str, default="auto", help="MC 后端 auto/process/batch/gpu"
    )
    parser.add_argument("--mc-n-jobs", type=int, default=-1, help="MC process 后端并行数")
    parser.add_argument("--mc-seed", type=int, default=42, help="MC 随机种子")
    parser.add_argument("--mc-samples", type=int, default=100, help="MC 样本数")
    parser.add_argument(
        "--wind-model", type=str, help="风场模型 (uniform/log/power/profile/composite/dryden/none)"
    )
    parser.add_argument("--wind-profile", type=str, help="风场廓线文件路径")
    parser.add_argument(
        "--terrain-model",
        type=str,
        help="地形模型 (null/hilly/numpy/image/geotiff/srtm_dir/srtm_files)",
    )
    parser.add_argument("--terrain-path", type=str, help="地形文件或目录路径")
    parser.add_argument(
        "--terrain-extent",
        type=str,
        help="地形范围 lat_min,lat_max,lon_min,lon_max",
    )
    parser.add_argument("--gui", action="store_true", help="启动图形界面")
    parser.add_argument("--serve", action="store_true", help="启动 Web API 服务")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务监听地址")
    parser.add_argument("--port", type=int, default=8000, help="服务监听端口")
    parser.add_argument("--reload", action="store_true", help="启用热重载")
    return parser.parse_args()


def _build_projectile_config(args: argparse.Namespace) -> tuple[SimConfig, list]:
    preset = args.preset or "M107"
    cfg = m107_config() if preset == "M107" else _projectile_config_from_preset(preset)
    if args.qe is not None:
        # 仰角物理上限为 90°, 对非法输入做钳位保护
        qe = float(np.clip(args.qe, 0.0, 90.0))
        cfg = apply_overrides(cfg, {"launch.elevation_deg": qe})
    if args.az is not None:
        cfg = apply_overrides(cfg, {"launch.azimuth_deg": args.az})
    phases = m107_phases() if preset == "M107" else projectile_phases(preset)
    return cfg, phases


def _projectile_config_from_preset(name: str) -> SimConfig:
    from ballistic_sim.presets.projectiles import _projectile_config_from_preset

    return _projectile_config_from_preset(name)


def _build_missile_config(args: argparse.Namespace) -> tuple[SimConfig, list]:
    name = args.missile or list_missiles()[0]
    cfg = missile_config(name)
    if args.az is not None:
        cfg = apply_overrides(cfg, {"launch.azimuth_deg": args.az})
    if args.target_lat is not None:
        cfg = apply_overrides(cfg, {"guidance.target_lat_deg": args.target_lat})
    if args.target_lon is not None:
        cfg = apply_overrides(cfg, {"guidance.target_lon_deg": args.target_lon})
    phases = missile_phases(name)
    return cfg, phases


def _build_rocket_config(args: argparse.Namespace) -> tuple[SimConfig, list]:
    name = _normalize_rocket_name(args.rocket or "CZ2F")
    cfg = rocket_config(name)
    if args.az is not None:
        cfg = apply_overrides(cfg, {"launch.azimuth_deg": args.az})
    if args.qe is not None:
        cfg = apply_overrides(cfg, {"launch.elevation_deg": args.qe})
    phases = rocket_phases(cfg, name=name)
    return cfg, phases


def _build_icbm_config(args: argparse.Namespace) -> tuple[SimConfig, list]:
    """占位 ICBM 配置 (单级大推力 + 滑行 + 再入)。"""
    from ballistic_sim.config import EnvironmentConfig, OptionsConfig, VehicleConfig
    from ballistic_sim.dynamics.powered_eci import PoweredECIDynamics
    from ballistic_sim.phases.coasting import CoastingPhase
    from ballistic_sim.phases.powered import PoweredPhase
    from ballistic_sim.phases.reentry import ReentryPhase
    from ballistic_sim.phases.terminal import TerminalPhase

    lat = 0.0
    lon = 0.0
    azimuth = args.az if args.az is not None else 45.0
    warhead_mass = 1000.0
    stage: Dict[str, Any] = dict(
        name="ICBM-boost",
        thrust_vac=1.2e6,
        thrust_sl=1.0e6,
        isp_vac=300.0,
        m_prop=20000.0,
        m_dry=warhead_mass,
        Aref=1.0,
    )
    guid = dict(
        lat_deg=lat,
        lon_deg=lon,
        azimuth_deg=azimuth,
        t_pitchover=10.0,
        kick_deg=20.0,
        t_kick_end=60.0,
    )
    dyn_boost = PoweredECIDynamics(stage=stage, guidance=guid)
    t_burn = float(stage["m_prop"]) / dyn_boost.prop.mdot
    ph_boost = PoweredPhase(
        name="主动段",
        t_span=(0.0, t_burn * 1.2),
        dynamics=dyn_boost,
        guidance=guid,
        m_dry=warhead_mass,
        sep_name="关机",
    )
    coast_stage: Dict[str, Any] = dict(
        name="ICBM-coast",
        thrust_vac=0.0,
        thrust_sl=0.0,
        isp_vac=1.0,
        m_prop=0.0,
        m_dry=warhead_mass,
        Aref=0.5,
    )
    dyn_coast = PoweredECIDynamics(
        stage=coast_stage,
        guidance=guid,
        modes={"thrust": False, "drag": True, "j2": True},
    )

    def _alt_event(h_m: float, direction: int, terminal: bool = False):
        def ev(t: float, y: np.ndarray) -> float:
            return float(np.linalg.norm(y[0:3]) - 6378137.0 - h_m)

        ev.terminal = terminal  # type: ignore[attr-defined]
        ev.direction = direction  # type: ignore[attr-defined]
        return ev

    def _apo_event():
        def ev(t: float, y: np.ndarray) -> float:
            return float(np.dot(y[0:3], y[3:6]))

        ev.terminal = False  # type: ignore[attr-defined]
        ev.direction = -1  # type: ignore[attr-defined]
        return ev

    ph_coast = CoastingPhase(
        name="中段",
        t_span=(0.0, 7200.0),
        dynamics=dyn_coast,
        guidance=guid,
        events=[_apo_event(), _alt_event(100e3, +1), _alt_event(100e3, -1, terminal=True)],
    )
    ph_reentry = ReentryPhase(
        name="再入段",
        t_span=(0.0, 7200.0),
        dynamics=dyn_coast,
        events=[_alt_event(0.0, -1, terminal=True)],
    )
    ph_terminal = TerminalPhase(
        name="终点",
        t_span=(0.0, 7200.0),
        dynamics=dyn_coast,
    )
    cfg = SimConfig(
        mission="icbm",
        vehicle=VehicleConfig(
            mass_kg=float(stage["m_prop"]) + float(stage["m_dry"]),
            diameter_m=0.5,
            cd=0.3,
        ),
        launch=LaunchConfig(
            lat_deg=lat,
            lon_deg=lon,
            alt_m=1.0,
            azimuth_deg=azimuth,
            elevation_deg=90.0,
            v0_m_s=0.0,
        ),
        environment=EnvironmentConfig(
            atmosphere="isa",
            gravity_model="j2",
        ),
        guidance=GuidanceConfig(),
        options=OptionsConfig(
            integrator="DOP853",
            rtol=1e-6,
            atol=1e-9,
            max_step=1.0,
            terminate_impact=True,
        ),
    )
    return cfg, [ph_boost, ph_coast, ph_reentry, ph_terminal]


def _build_suborbital_config(args: argparse.Namespace) -> tuple[SimConfig, list]:
    """亚轨道任务: 简化单级火箭, 主动段后关闭推力落回。"""
    from ballistic_sim.config import EnvironmentConfig, OptionsConfig, VehicleConfig
    from ballistic_sim.dynamics.powered_eci import PoweredECIDynamics
    from ballistic_sim.phases.coasting import CoastingPhase
    from ballistic_sim.phases.powered import PoweredPhase
    from ballistic_sim.phases.terminal import TerminalPhase

    mass = 1000.0
    thrust = 25.0e3
    isp = 290.0
    thrust / (isp * 9.80665)
    m_prop = 0.8 * mass
    m_dry = mass - m_prop
    Aref = 0.196

    stage: Dict[str, Any] = dict(
        name="Suborbital-boost",
        thrust_sl=thrust,
        thrust_vac=thrust * 1.1,
        isp_vac=isp,
        m_prop=m_prop,
        m_dry=m_dry,
        Aref=Aref,
    )
    guid = dict(
        lat_deg=0.0,
        lon_deg=0.0,
        azimuth_deg=args.az if args.az is not None else 90.0,
        t_pitchover=5.0,
        kick_deg=args.qe if args.qe is not None else 85.0,
        t_kick_end=15.0,
    )
    dyn_boost = PoweredECIDynamics(stage=stage, guidance=guid)
    t_burn = float(stage["m_prop"]) / dyn_boost.prop.mdot
    ph_boost = PoweredPhase(
        name="动力上升",
        t_span=(0.0, t_burn * 1.5),
        dynamics=dyn_boost,
        guidance=guid,
        m_dry=m_dry,
        sep_name="燃尽",
    )

    coast_stage: Dict[str, Any] = dict(
        name="Suborbital-coast",
        thrust_vac=0.0,
        thrust_sl=0.0,
        isp_vac=1.0,
        m_prop=0.0,
        m_dry=m_dry,
        Aref=Aref,
    )
    dyn_coast = PoweredECIDynamics(
        stage=coast_stage,
        guidance=guid,
        modes={"thrust": False, "drag": True, "j2": True},
    )
    ph_coast = CoastingPhase(
        name="无动力回落",
        t_span=(0.0, 3600.0),
        dynamics=dyn_coast,
        guidance=guid,
    )
    ph_terminal = TerminalPhase(
        name="终点",
        t_span=(0.0, 3600.0),
        dynamics=dyn_coast,
    )

    cfg = SimConfig(
        mission="suborbital",
        vehicle=VehicleConfig(
            mass_kg=mass,
            diameter_m=0.5,
            cd=0.3,
            area_ref_m2=Aref,
        ),
        launch=LaunchConfig(
            lat_deg=0.0,
            lon_deg=0.0,
            alt_m=1.0,
            azimuth_deg=guid["azimuth_deg"],
            elevation_deg=90.0,
            v0_m_s=0.0,
        ),
        environment=EnvironmentConfig(
            atmosphere="isa",
            gravity_model="j2",
        ),
        guidance=GuidanceConfig(),
        options=OptionsConfig(
            integrator="DOP853",
            rtol=1e-6,
            atol=1e-9,
            max_step=1.0,
            terminate_impact=True,
        ),
    )
    return cfg, [ph_boost, ph_coast, ph_terminal]


def _parse_terrain_extent(value: Optional[str]) -> Optional[list[float]]:
    """解析 --terrain-extent 为 [lat_min, lat_max, lon_min, lon_max]。"""
    if value is None:
        return None
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--terrain-extent 需为 lat_min,lat_max,lon_min,lon_max")
    return parts


def _apply_environment_overrides(cfg: SimConfig, args: argparse.Namespace) -> SimConfig:
    """将 CLI 风场/地形参数通过 apply_overrides 注入配置。"""
    overrides: Dict[str, Any] = {}
    if args.wind_model is not None:
        overrides["environment.wind_model"] = args.wind_model
    if args.wind_profile is not None:
        overrides["environment.wind_profile_path"] = args.wind_profile
    if args.terrain_model is not None:
        overrides["environment.terrain_model"] = args.terrain_model
    if args.terrain_path is not None:
        overrides["environment.terrain_path"] = args.terrain_path
    extent = _parse_terrain_extent(args.terrain_extent)
    if extent is not None:
        overrides["environment.terrain_extent"] = extent
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def _load_config_file(path: str) -> SimConfig:
    """从 YAML/JSON 配置文件加载 SimConfig。"""
    from ballistic_sim.config import load_config

    return load_config(path)


def _apply_common_overrides(
    cfg: SimConfig, args: argparse.Namespace
) -> SimConfig:
    """将 CLI 通用参数通过 apply_overrides 注入配置并返回。"""
    if args.qe is not None:
        # 仰角物理上限为 90°, 对非法输入做钳位保护
        qe = float(np.clip(args.qe, 0.0, 90.0))
        cfg = apply_overrides(cfg, {"launch.elevation_deg": qe})
    if args.az is not None:
        cfg = apply_overrides(cfg, {"launch.azimuth_deg": args.az})
    if args.target_lat is not None:
        cfg = apply_overrides(cfg, {"guidance.target_lat_deg": args.target_lat})
    if args.target_lon is not None:
        cfg = apply_overrides(cfg, {"guidance.target_lon_deg": args.target_lon})
    cfg = _apply_environment_overrides(cfg, args)
    return cfg


def _build_config_and_phases(args: argparse.Namespace) -> tuple[SimConfig, list]:
    from ballistic_sim.phases.builder import build_phases

    if args.config is not None:
        cfg = _load_config_file(args.config)
        # CLI 显式 --mission 优先级高于配置文件
        if args.mission is not None:
            cfg = apply_overrides(cfg, {"mission": args.mission})
        phases = build_phases(cfg)
    elif args.mission == "projectile":
        cfg, phases = _build_projectile_config(args)
    elif args.mission == "missile":
        cfg, phases = _build_missile_config(args)
    elif args.mission == "rocket":
        cfg, phases = _build_rocket_config(args)
    elif args.mission == "icbm":
        cfg, phases = _build_icbm_config(args)
    elif args.mission == "suborbital":
        cfg, phases = _build_suborbital_config(args)
    else:
        raise ValueError("未提供 --mission 或 --config")
    cfg = _apply_common_overrides(cfg, args)
    return cfg, phases


def _compute_summary(cfg: SimConfig, result: SimResult) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "mission": cfg.mission,
        "stop_reason": result.stop_reason,
        "t_end_s": float(result.t[-1]) if result.t.size else None,
        "n_points": int(result.t.size),
        "phase_bounds": [float(x) for x in result.phase_bounds],
        "events": [
            {"phase": ev.get("phase"), "name": ev.get("name"), "t": ev.get("t")}
            for ev in result.event_log
        ],
    }
    if result.y.size == 0:
        return summary

    frame = detect_frame(result)
    summary["frame"] = frame
    if frame == "ENU":
        e, n, u = result.y[:, 0], result.y[:, 1], result.y[:, 2]
        summary["range_m"] = float(np.hypot(e[-1], n[-1]))
        summary["max_alt_m"] = float(np.max(u))
        summary["impact_speed_m_s"] = float(np.linalg.norm(result.y[-1, 3:6]))
    elif frame == "ECI":
        r = result.y[-1, 0:3]
        v = result.y[-1, 3:6]
        r_ecef = eci_to_ecef(r, float(result.t[-1]))
        lat_f, lon_f, alt_f = ecef_to_geodetic(r_ecef)
        summary["lat_end_deg"] = float(lat_f)
        summary["lon_end_deg"] = float(lon_f)
        summary["alt_end_m"] = float(alt_f)
        summary["v_end_m_s"] = float(np.linalg.norm(v))
        summary["range_m"] = float(
            haversine_distance(cfg.launch.lat_deg, cfg.launch.lon_deg, lat_f, lon_f)
        )
        oe = rv_to_oe(r, v)
        summary["orbit"] = {
            "a_m": float(oe["a"]),
            "e": float(oe["e"]),
            "i_deg": float(oe["i_deg"]),
            "h_peri_km": float(oe["h_peri_km"]),
            "h_apo_km": float(oe["h_apo_km"]),
        }
    return summary


def _save_visualizations(result: SimResult, cfg: SimConfig, out_dir: Path) -> List[str]:
    from matplotlib import use

    use("Agg")  # 非交互后端
    saved: List[str] = []

    frame = detect_frame(result)
    if frame == "ENU":
        from ballistic_sim.viz.groundtrack import plot_groundtrack, plot_impact_summary
        from ballistic_sim.viz.profile import plot_altitude_range, plot_velocity_profile
        from ballistic_sim.viz.trajectory3d import plot_trajectory3d, plot_trajectory3d_topdown
        from ballistic_sim.viz.wind_plot import plot_wind_contribution, plot_wind_profile

        fig = plot_trajectory3d(result)
        path = out_dir / "trajectory3d.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

        fig = plot_trajectory3d_topdown(result)
        path = out_dir / "trajectory3d_topdown.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

        fig = plot_groundtrack(result)
        path = out_dir / "groundtrack.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

        fig = plot_impact_summary(result)
        path = out_dir / "impact_summary.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

        fig = plot_altitude_range(result)
        path = out_dir / "altitude_range.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

        fig = plot_velocity_profile(result)
        path = out_dir / "velocity_profile.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

        # 风廓线使用占位
        heights = np.linspace(0, float(np.max(altitude(result))), 50)
        wind_enu = np.zeros((50, 3))
        fig = plot_wind_profile(heights, wind_enu)
        path = out_dir / "wind_profile.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

        fig = plot_wind_contribution(result, wind_enu, heights)
        path = out_dir / "wind_contribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

    elif frame == "ECI":
        from ballistic_sim.viz.dvbudget import plot_dv_budget
        from ballistic_sim.viz.globe3d import plot_globe3d
        from ballistic_sim.viz.groundtrack import plot_groundtrack
        from ballistic_sim.viz.orbit_insertion import plot_orbit_insertion
        from ballistic_sim.viz.profile import plot_altitude_range, plot_velocity_profile
        from ballistic_sim.viz.reentry import plot_reentry_heat
        from ballistic_sim.viz.trajectory3d import plot_trajectory3d

        fig = plot_globe3d(result)
        path = out_dir / "globe3d.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

        fig = plot_groundtrack(result)
        path = out_dir / "groundtrack.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

        fig = plot_altitude_range(result)
        path = out_dir / "altitude_range.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

        fig = plot_velocity_profile(result)
        path = out_dir / "velocity_profile.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

        fig = plot_orbit_insertion(result)
        path = out_dir / "orbit_insertion.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

        fig = plot_dv_budget(result)
        path = out_dir / "dv_budget.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

        fig = plot_reentry_heat(result)
        path = out_dir / "reentry_heat.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

        fig = plot_trajectory3d(result)
        path = out_dir / "trajectory3d.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

    return saved


def _mc_summary(mc_result: Any) -> Dict[str, Any]:
    return {
        "samples": int(mc_result.samples),
        "range_mean_m": float(mc_result.range_mean),
        "range_std_m": float(mc_result.range_std),
        "cross_mean_m": float(mc_result.cross_mean),
        "cross_std_m": float(mc_result.cross_std),
        "cep50_m": float(mc_result.cep50),
        "cep90_m": float(mc_result.cep90),
        "ellipse_major_m": float(mc_result.ellipse_major),
        "ellipse_minor_m": float(mc_result.ellipse_minor),
        "ellipse_angle_deg": float(mc_result.ellipse_angle),
    }


def main() -> None:
    args = _parse_args()

    if args.serve:
        from ballistic_sim.api import create_app
        from ballistic_sim.api.dependencies import require_uvicorn

        require_uvicorn()
        import uvicorn

        uvicorn.run(create_app(), host=args.host, port=args.port, reload=args.reload)
        return

    if args.gui:
        from ballistic_sim.gui import run_gui

        run_gui()
        return

    if args.mission is None and args.config is None:
        raise SystemExit("错误: --mission 是必需参数（--gui/--serve/--config 除外）")

    cfg, phases = _build_config_and_phases(args)

    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(cfg.mission)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_viz: List[str] = []

    if args.monte_carlo:
        if args.mc_backend in ("batch", "gpu"):
            cfg = apply_overrides(cfg, {"options.mpm_use_spin": False})
        mc_result = monte_carlo_simulation(
            cfg,
            perturb=PerturbationConfig(),
            n_samples=args.mc_samples,
            backend=args.mc_backend,
            n_jobs=args.mc_n_jobs,
            seed=args.mc_seed,
        )
        mc_summary = _mc_summary(mc_result)

        if args.viz and not args.no_viz:
            from matplotlib import use

            use("Agg")
            fig = plot_dispersion(mc_result)
            path = out_dir / "montecarlo_plot.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            saved_viz.append(str(path))

        if not args.no_summary:
            summary_path = out_dir / "mc_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(mc_summary, f, ensure_ascii=False, indent=2)
            saved_viz.append(str(summary_path))

        print(f"Mission : {cfg.mission}")
        print(f"Mode    : Monte Carlo ({args.mc_backend})")
        print(f"Samples : {mc_result.samples}")
        print(f"Range   : {mc_result.range_mean / 1e3:.2f} ± {mc_result.range_std / 1e3:.2f} km")
        print(f"CEP50   : {mc_result.cep50:.1f} m")
        print(f"CEP90   : {mc_result.cep90:.1f} m")
        print(f"Output  : {out_dir}")
        return

    result = simulate(cfg, phases=phases)
    attach_launch_lla(result, cfg.launch.lat_deg, cfg.launch.lon_deg, cfg.launch.alt_m)

    summary = _compute_summary(cfg, result)

    if args.viz and not args.no_viz:
        saved_viz = _save_visualizations(result, cfg, out_dir)

    if not args.no_summary:
        summary_path = out_dir / "result_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        saved_viz.append(str(summary_path))

    print(f"Mission : {cfg.mission}")
    print(f"Stop    : {summary['stop_reason']}")
    print(f"TOF     : {summary.get('t_end_s')} s")
    if "range_m" in summary:
        print(f"Range   : {summary['range_m'] / 1e3:.2f} km")
    if "v_end_m_s" in summary:
        print(f"V_end   : {summary['v_end_m_s']:.1f} m/s")
    if "orbit" in summary:
        orb = summary["orbit"]
        print(f"Orbit   : a={orb['a_m'] / 1e3:.1f} km, e={orb['e']:.4f}, i={orb['i_deg']:.2f}°")
    print(f"Output  : {out_dir}")


if __name__ == "__main__":
    main()
