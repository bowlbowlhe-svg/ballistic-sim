"""由 SimConfig 构建 Phase 列表。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ballistic_sim.config import SimConfig, StageConfig
from ballistic_sim.constants import WGS84_A
from ballistic_sim.dynamics.mpm import MPMOptions
from ballistic_sim.guidance.aag import make_aag_state
from ballistic_sim.guidance.energy_management import EnergyManagementGuidance
from ballistic_sim.guidance.proportional_navigation import ProNavGuidance
from ballistic_sim.guidance.reentry_guidance import ReentryGuidance
from ballistic_sim.phases.base import Phase
from ballistic_sim.phases.coasting import CoastingPhase
from ballistic_sim.phases.powered import PoweredPhase
from ballistic_sim.phases.events import make_apogee_event
from ballistic_sim.phases.reentry import ReentryPhase
from ballistic_sim.phases.terminal import TerminalPhase


def _make_aag_target(cfg: SimConfig) -> dict:
    """由配置构造 AAG 终端目标 (r_T_m, v_T_ms, gamma_T)。"""
    from ballistic_sim.constants import GM_EARTH, WGS84_A

    r_T = WGS84_A + float(cfg.guidance.target_alt_m)
    if cfg.guidance.terminal_velocity_m_s is not None:
        v_T = float(cfg.guidance.terminal_velocity_m_s)
    else:
        v_T = float(np.sqrt(GM_EARTH / max(r_T, 1.0)))
    gamma_T = float(cfg.guidance.terminal_fpa_deg) * np.pi / 180.0
    return {"r_T_m": r_T, "v_T_ms": v_T, "gamma_T": gamma_T}


def _inject_powered_guidance(
    guid: Dict[str, Any],
    cfg: SimConfig,
    stage: Dict[str, Any],
    is_terminal_stage: bool,
) -> Dict[str, Any]:
    """根据 ``guidance_law`` 为动力段制导字典注入状态对象。"""
    law = cfg.guidance.guidance_law
    if law == "none":
        return guid

    if law == "peg":
        # PEG 已在 open_loop 中通过 phase == "peg" + _peg_state/_peg_stage 支持
        guid["phase"] = "peg"
        return guid

    if law == "aag":
        target = _make_aag_target(cfg)
        state = make_aag_state(
            target,
            replan_period=cfg.guidance.guidance_replan_period,
            max_iter=cfg.guidance.aag_max_iter,
            position_gain=cfg.guidance.aag_position_gain,
        )
        guid["phase"] = "aag"
        guid["_aag_state"] = state
        guid["_aag_stage"] = stage
        return guid

    if law == "proportional" and is_terminal_stage and cfg.mission == "missile":
        if cfg.guidance.target_lat_deg is not None and cfg.guidance.target_lon_deg is not None:
            pn = ProNavGuidance(nav_constant=cfg.guidance.nav_constant).set_target(
                cfg.guidance.target_lat_deg,
                cfg.guidance.target_lon_deg,
                cfg.guidance.target_alt_m,
            )
            guid["phase"] = "proportional"
            guid["_pronav_guidance"] = pn
        return guid

    return guid


def _make_reentry_guidance(cfg: SimConfig) -> Optional[Any]:
    """构造再入/滑行段制导对象（若配置启用）。"""
    law = cfg.guidance.guidance_law
    if law not in ("reentry", "energy"):
        return None
    if cfg.guidance.target_lat_deg is None or cfg.guidance.target_lon_deg is None:
        return None

    if law == "reentry":
        energy = cfg.guidance.energy_target_j_kg
        return ReentryGuidance(
            target_lat_deg=cfg.guidance.target_lat_deg,
            target_lon_deg=cfg.guidance.target_lon_deg,
            target_energy_j_kg=energy,
            max_bank_deg=cfg.guidance.max_bank_deg,
            nominal_aoa_deg=cfg.guidance.nominal_aoa_deg,
            bank_gain=cfg.guidance.reentry_bank_gain,
        )

    # energy
    energy = cfg.guidance.energy_target_j_kg
    return EnergyManagementGuidance(
        energy_target_j_kg=energy if energy is not None else -0.5 * 9.80665 * 6371000.0,
        energy_slope_j_kg_m=cfg.guidance.energy_slope_j_kg_m,
        kp=cfg.guidance.energy_kp,
        max_aoa_deg=cfg.guidance.max_bank_deg,
    )


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
    guidance = _inject_powered_guidance(guidance, cfg, stage, is_terminal_stage=True)
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
    reentry_guid = _make_reentry_guidance(cfg)
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
            guidance=reentry_guid,
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
                guidance=reentry_guid,
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
        "t_pitchover": cfg.guidance.t_pitchover,
        "kick_deg": cfg.guidance.kick_deg if cfg.guidance.kick_deg != 0.0 else 3.0,
        "t_kick_end": cfg.guidance.t_kick_end,
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
        is_last = i == n - 1
        # 末级火箭默认采用线性俯仰上面级制导；若配置显式制导律则优先使用制导律
        if is_last and cfg.mission == "rocket" and cfg.guidance.guidance_law == "none":
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
        guid = _inject_powered_guidance(
            guid, cfg, sd, is_terminal_stage=(is_last and cfg.mission == "missile")
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

    # 滑行/再入/终点：rocket/icbm/missile/suborbital 均插入滑行段；
    # rocket 任务默认以轨道插入为终点，滑行段默认只带远地点/落地事件。
    insert_reentry = cfg.mission in ("icbm", "missile") or cfg.options.sixdof_reentry
    if cfg.mission in ("rocket", "icbm", "missile", "suborbital"):
        if insert_reentry:
            # 对 icbm/missile，滑行段以 100 km 再入高度为 terminal 事件，
            # 使后续 ReentryPhase 能接管积分；避免地面事件在滑行段提前终止。
            _h_entry = 100e3
            _r_entry = cfg.launch.alt_m + 1e3  # 上升段出大气事件 (非 terminal)

            def _altitude_event(
                h_m: float, direction: int, terminal: bool = False, name: str = "高度事件"
            ):
                def ev(t: float, y: np.ndarray) -> float:
                    r = np.asarray(y[0:3], dtype=float)
                    return float(np.linalg.norm(r) - WGS84_A - h_m)

                ev.terminal = terminal  # type: ignore[attr-defined]
                ev.direction = direction  # type: ignore[attr-defined]
                ev.name = name  # type: ignore[attr-defined]
                return ev

            apogee_ev = make_apogee_event(frame="ECI")
            apogee_ev.name = "远地点"  # type: ignore[attr-defined]
            coast_events = [
                apogee_ev,
                _altitude_event(_r_entry, +1, terminal=False, name="出大气"),
                _altitude_event(_h_entry, -1, terminal=True, name="再入"),
            ]
        else:
            coast_events = []

        phases.append(
            CoastingPhase(
                name="滑行",
                t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 7200.0),
                dynamics=dyn_coast,
                events=coast_events,
                terrain=terrain,
                lat0=cfg.launch.lat_deg,
                lon0=cfg.launch.lon_deg,
            )
        )

    reentry_guid = _make_reentry_guidance(cfg)
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
                guidance=reentry_guid,
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
                    guidance=reentry_guid,
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
                    guidance=reentry_guid,
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
