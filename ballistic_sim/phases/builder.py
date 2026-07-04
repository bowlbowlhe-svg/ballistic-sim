"""由 SimConfig 构建 Phase 列表。"""

from __future__ import annotations

from typing import List

from ballistic_sim.config import SimConfig
from ballistic_sim.dynamics.mpm import MPMOptions
from ballistic_sim.phases.base import Phase
from ballistic_sim.phases.coasting import CoastingPhase
from ballistic_sim.phases.powered import PoweredPhase
from ballistic_sim.phases.reentry import ReentryPhase
from ballistic_sim.phases.terminal import TerminalPhase


def build_phases(cfg: SimConfig) -> List[Phase]:
    """按任务类型构造 Phase 序列。"""
    if cfg.mission == "projectile":
        return _build_projectile_phases(cfg)
    if cfg.mission in ("rocket", "icbm", "missile", "suborbital"):
        return _build_rocket_phases(cfg)
    raise ValueError(f"未支持的任务类型: {cfg.mission}")


def _build_projectile_phases(cfg: SimConfig) -> List[Phase]:
    from ballistic_sim.dynamics.mpm import MPMDynamics

    opt = MPMOptions(
        use_drag=True,
        use_wind=bool(cfg.environment.wind_m_s),
        use_coriolis=True,
        use_spin=True,
        use_dynamic_alpha=False,
        method=cfg.options.integrator,
        rtol=cfg.options.rtol,
        atol=cfg.options.atol,
    )
    dyn = MPMDynamics(
        mass_kg=cfg.vehicle.mass_kg,
        diameter_m=cfg.vehicle.diameter_m,
        form_factor=cfg.vehicle.cd or 1.0,
        options=opt,
        lat_deg=cfg.launch.lat_deg,
    )
    return [
        PoweredPhase(
            name="无动力弹道",
            t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 3000.0),
            dynamics=dyn,
            guidance=None,
            m_dry=cfg.vehicle.mass_kg,
            sep_name="落地",
        ),
        TerminalPhase(
            name="终点",
            t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 3000.0),
            dynamics=dyn,
        ),
    ]


def _build_rocket_phases(cfg: SimConfig) -> List[Phase]:
    from ballistic_sim.dynamics.powered_eci import PoweredECIDynamics

    # 简化：单级动力 + 滑行 + 终端；真实 CZ-2F 在 MVP 脚本中手动构建。
    stage = {
        "thrust_sl": cfg.vehicle.thrust_N or 6.0e6,
        "thrust_vac": (cfg.vehicle.thrust_N or 6.0e6) * 1.1,
        "isp_vac": 290.0,
        "m_prop": cfg.vehicle.mass_kg * 0.8,
        "m_dry": cfg.vehicle.mass_kg * 0.2,
        "Aref": cfg.vehicle.area_ref_m2 or 10.0,
    }
    guidance = {
        "lat_deg": cfg.launch.lat_deg,
        "lon_deg": cfg.launch.lon_deg,
        "azimuth_deg": cfg.launch.azimuth_deg,
        "t_pitchover": 10.0,
        "kick_deg": cfg.guidance.kick_deg or 3.0,
        "t_kick_end": 25.0,
    }
    dyn = PoweredECIDynamics(stage=stage, guidance=guidance)
    t_burn_est = float(stage["m_prop"]) / dyn.prop.mdot
    phases: List[Phase] = [
        PoweredPhase(
            name="动力上升",
            t_span=(cfg.launch.t0_s, cfg.launch.t0_s + t_burn_est * 2.0),
            dynamics=dyn,
            guidance=guidance,
            m_dry=float(stage["m_dry"]),
            sep_name="燃尽",
        ),
        CoastingPhase(
            name="滑行",
            t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 3600.0),
            dynamics=dyn,
        ),
    ]
    if cfg.options.sixdof_reentry:
        from ballistic_sim.dynamics.six_dof import SixDOFDynamics

        six_dof_dyn = SixDOFDynamics(
            mass_kg=float(stage["m_dry"]),
            diameter_m=cfg.vehicle.diameter_m,
            form_factor=cfg.vehicle.cd or 1.0,
            Ix=cfg.vehicle.Ix or 0.1,
            It=cfg.vehicle.It or 1.0,
            x_cp_cg=cfg.vehicle.x_cp_cg or 0.05,
            lat_deg=cfg.launch.lat_deg,
            twist_cal=cfg.vehicle.twist_cal or 20.0,
            options={"drag": True, "gravity": True, "coriolis": True, "thrust": False},
        )
        phases.append(
            ReentryPhase(
                name="再入段",
                t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 7200.0),
                dynamics=six_dof_dyn,
                fidelity="sixdof",
            )
        )
    phases.append(
        TerminalPhase(
            name="轨道插入",
            t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 3600.0),
            dynamics=dyn,
        ),
    )
    return phases
