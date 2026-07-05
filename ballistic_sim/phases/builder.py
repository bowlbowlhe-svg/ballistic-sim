"""由 SimConfig 构建 Phase 列表。"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ballistic_sim.config import SimConfig, StageConfig
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


def build_phases_legacy(cfg: SimConfig) -> List[Phase]:
    """保留阶段 1 的旧版 builder，用于回归兼容。"""
    if cfg.mission == "projectile":
        return _build_projectile_phases(cfg)
    if cfg.mission in ("rocket", "icbm", "missile", "suborbital"):
        return _build_rocket_phases_legacy(cfg)
    raise ValueError(f"未支持的任务类型: {cfg.mission}")


def _build_projectile_phases(cfg: SimConfig) -> List[Phase]:
    from ballistic_sim.dynamics.mpm import MPMDynamics
    from ballistic_sim.simulator import _resolve_terrain

    terrain = _resolve_terrain(cfg)
    opt = MPMOptions(
        use_drag=True,
        use_wind=bool(cfg.environment.wind_m_s),
        use_coriolis=True,
        use_spin=cfg.options.mpm_use_spin,
        use_dynamic_alpha=cfg.options.mpm_use_dynamic_alpha,
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
            terrain=terrain,
            lat0=cfg.launch.lat_deg,
            lon0=cfg.launch.lon_deg,
        ),
        TerminalPhase(
            name="终点",
            t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 3000.0),
            dynamics=dyn,
            terrain=terrain,
            lat0=cfg.launch.lat_deg,
            lon0=cfg.launch.lon_deg,
        ),
    ]


def _build_rocket_phases(cfg: SimConfig) -> List[Phase]:
    """火箭/导弹/ICBM/suborbital phase 构造入口。

    若 ``VehicleConfig.stages`` 已提供，按多级质量记账生成
    ``PoweredPhase → CoastingPhase → (ReentryPhase) → TerminalPhase``；
    否则回退到阶段 1 的单级近似逻辑。
    """
    if cfg.vehicle.stages is not None:
        return _build_multistage_phases(cfg)
    return _build_rocket_phases_legacy(cfg)


def _build_rocket_phases_legacy(cfg: SimConfig) -> List[Phase]:
    from ballistic_sim.dynamics.powered_eci import PoweredECIDynamics
    from ballistic_sim.simulator import _resolve_terrain

    terrain = _resolve_terrain(cfg)

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
            terrain=terrain,
            lat0=cfg.launch.lat_deg,
            lon0=cfg.launch.lon_deg,
        ),
        CoastingPhase(
            name="滑行",
            t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 3600.0),
            dynamics=dyn,
            terrain=terrain,
            lat0=cfg.launch.lat_deg,
            lon0=cfg.launch.lon_deg,
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
                terrain=terrain,
                lat0=cfg.launch.lat_deg,
                lon0=cfg.launch.lon_deg,
            )
        )
    phases.append(
        TerminalPhase(
            name="轨道插入",
            t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 3600.0),
            dynamics=dyn,
            terrain=terrain,
            lat0=cfg.launch.lat_deg,
            lon0=cfg.launch.lon_deg,
        ),
    )
    return phases


def _stage_dicts_with_accounting(
    stages: List[StageConfig], payload_mass: float
) -> List[Dict[str, Any]]:
    """将 ``StageConfig`` 列表转换为带质量记账的 stage dict 列表。"""
    n = len(stages)
    stage_dicts: List[Dict[str, Any]] = []
    for i, s in enumerate(stages):
        m_above = (
            sum(stages[j].m_dry + stages[j].m_prop for j in range(i + 1, n))
            + payload_mass
        )
        m_dry_phase = s.m_dry + m_above
        m_after_sep = m_dry_phase - s.m_dry
        Aref = s.Aref
        if Aref is None and s.diameter_m is not None:
            Aref = float(np.pi * (s.diameter_m / 2.0) ** 2)
        if Aref is None:
            Aref = 1.0
        stage_dicts.append(
            {
                "name": s.name or f"S{i+1}",
                "thrust_sl": float(s.thrust_sl),
                "thrust_vac": float(s.thrust_vac),
                "isp_vac": float(s.isp_vac),
                "m_prop": float(s.m_prop),
                "m_dry": float(m_dry_phase),
                "Aref": float(Aref),
                "m_after_sep": float(m_after_sep),
                "stage_dry": float(s.m_dry),
                "burn_time": float(s.burn_time) if s.burn_time is not None else None,
            }
        )
    return stage_dicts


def _build_multistage_phases(cfg: SimConfig) -> List[Phase]:
    """多级火箭/导弹/ICBM phase 链生成。

    流程：
    1. 为每一级生成 ``PoweredPhase``， burnout 事件触发级间分离与质量阶跃。
    2. 末级燃尽后进入 ``CoastingPhase``。
    3. icbm/missile 或 ``sixdof_reentry=True`` 时追加 ``ReentryPhase``。
    4. 最后追加 ``TerminalPhase`` 作为终止标记。
    """
    from ballistic_sim.constants import G0_STANDARD
    from ballistic_sim.dynamics.powered_eci import PoweredECIDynamics
    from ballistic_sim.simulator import _resolve_terrain

    terrain = _resolve_terrain(cfg)
    stages = cfg.vehicle.stages
    assert stages is not None and len(stages) > 0

    base_guid: Dict[str, Any] = {
        "lat_deg": cfg.launch.lat_deg,
        "lon_deg": cfg.launch.lon_deg,
        "azimuth_deg": cfg.launch.azimuth_deg,
        "t_pitchover": 10.0,
        "kick_deg": cfg.guidance.kick_deg if cfg.guidance.kick_deg != 0.0 else 3.0,
        "t_kick_end": 25.0,
    }

    total_stage_mass = sum(s.m_dry + s.m_prop for s in stages)
    payload_mass = max(float(cfg.vehicle.mass_kg) - float(total_stage_mass), 0.0)
    stage_dicts = _stage_dicts_with_accounting(stages, payload_mass)

    phases: List[Phase] = []
    cumulative_burn = 0.0
    n = len(stage_dicts)
    for i, sd in enumerate(stage_dicts):
        guid = dict(base_guid)
        use_upperstage = False
        # 末级火箭采用线性俯仰上面级制导
        if i == n - 1 and cfg.mission == "rocket":
            use_upperstage = True
            mdot = sd["thrust_vac"] / (sd["isp_vac"] * G0_STANDARD)
            t_burn = sd["m_prop"] / mdot
            guid.update(
                {
                    "t_us_start": float(cumulative_burn),
                    "gamma_end_deg": 0.0,
                    "gamma0_deg": None,
                    "pitch_rate_dps": None,
                    "t_us_dur": float(t_burn),
                }
            )
        dyn = PoweredECIDynamics(stage=sd, guidance=guid, use_upperstage=use_upperstage)
        t_burn = sd["m_prop"] / dyn.prop.mdot
        phases.append(
            PoweredPhase(
                name=f"{sd['name']} 动力",
                t_span=(cfg.launch.t0_s, cfg.launch.t0_s + t_burn * 1.5),
                dynamics=dyn,
                guidance=guid,
                m_dry=float(sd["m_dry"]),
                m_after_separation=float(sd["m_after_sep"]),
                sep_name=f"{sd['name']} 分离",
                terrain=terrain,
                lat0=cfg.launch.lat_deg,
                lon0=cfg.launch.lon_deg,
            )
        )
        cumulative_burn += t_burn

    # 滑行段：关闭推力，质量为末级分离后质量
    last_sd = dict(stage_dicts[-1])
    last_sd.update(
        thrust_vac=0.0,
        thrust_sl=0.0,
        isp_vac=1.0,
        m_prop=0.0,
        m_dry=float(last_sd["m_after_sep"]),
    )
    dyn_coast = PoweredECIDynamics(
        stage=last_sd,
        guidance=base_guid,
        modes={"thrust": False, "drag": True, "j2": True},
    )
    phases.append(
        CoastingPhase(
            name="滑行",
            t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 7200.0),
            dynamics=dyn_coast,
            terrain=terrain,
            lat0=cfg.launch.lat_deg,
            lon0=cfg.launch.lon_deg,
        )
    )

    # 再入段：icbm/missile 默认加入；rocket 仅在 sixdof_reentry 时加入
    if cfg.mission in ("icbm", "missile") or cfg.options.sixdof_reentry:
        if cfg.options.sixdof_reentry:
            from ballistic_sim.dynamics.six_dof import SixDOFDynamics

            six_dof_dyn = SixDOFDynamics(
                mass_kg=float(last_sd["m_dry"]),
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
                    terrain=terrain,
                    lat0=cfg.launch.lat_deg,
                    lon0=cfg.launch.lon_deg,
                )
            )
        else:
            phases.append(
                ReentryPhase(
                    name="再入段",
                    t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 7200.0),
                    dynamics=dyn_coast,
                    terrain=terrain,
                    lat0=cfg.launch.lat_deg,
                    lon0=cfg.launch.lon_deg,
                )
            )

    terminal_name = "终点" if cfg.mission in ("icbm", "missile", "suborbital") else "轨道插入"
    phases.append(
        TerminalPhase(
            name=terminal_name,
            t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 7200.0),
            dynamics=dyn_coast,
            terrain=terrain,
            lat0=cfg.launch.lat_deg,
            lon0=cfg.launch.lon_deg,
        ),
    )
    return phases
