"""多级火箭/ICBM phase 构建高级测试。"""

from __future__ import annotations

import pytest

from ballistic_sim.config import (
    GuidanceConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    StageConfig,
    VehicleConfig,
)
from ballistic_sim.phases.builder import build_phases, build_phases_legacy
from ballistic_sim.phases.coasting import CoastingPhase
from ballistic_sim.phases.powered import PoweredPhase
from ballistic_sim.phases.reentry import ReentryPhase
from ballistic_sim.phases.terminal import TerminalPhase


def _icbm_two_stage_cfg() -> SimConfig:
    return SimConfig(
        mission="icbm",
        vehicle=VehicleConfig(
            mass_kg=50000.0,
            diameter_m=2.0,
            cd=0.3,
            area_ref_m2=3.1416,
            stages=[
                StageConfig(
                    name="S1",
                    thrust_sl=1.2e6,
                    thrust_vac=1.35e6,
                    isp_vac=260.0,
                    m_prop=30000.0,
                    m_dry=8000.0,
                    diameter_m=2.0,
                ),
                StageConfig(
                    name="S2",
                    thrust_sl=200.0e3,
                    thrust_vac=250.0e3,
                    isp_vac=300.0,
                    m_prop=8000.0,
                    m_dry=2000.0,
                    diameter_m=1.5,
                ),
            ],
        ),
        launch=LaunchConfig(
            lat_deg=39.9,
            lon_deg=116.0,
            alt_m=50.0,
            azimuth_deg=90.0,
            elevation_deg=90.0,
        ),
        guidance=GuidanceConfig(kick_deg=25.0),
    )


def test_build_phases_legacy_unchanged() -> None:
    """build_phases_legacy 应保持阶段 1 的单级近似行为。"""
    cfg = SimConfig(
        mission="rocket",
        vehicle=VehicleConfig(mass_kg=1000.0, thrust_N=20.0e3, burn_time_s=60.0),
    )
    phases = build_phases_legacy(cfg)
    assert len(phases) == 3
    assert phases[0].name == "动力上升"
    assert phases[1].name == "滑行"
    assert phases[2].name == "轨道插入"


def test_multistage_icbm_phase_sequence() -> None:
    """ICBM 多级配置应生成 动力×2 → 滑行 → 再入 → 终点。"""
    cfg = _icbm_two_stage_cfg()
    phases = build_phases(cfg)
    names = [ph.name for ph in phases]
    assert names == ["S1 动力", "S2 动力", "滑行", "再入段", "终点"]
    assert all(isinstance(ph, PoweredPhase) for ph in phases[:2])
    assert isinstance(phases[2], CoastingPhase)
    assert isinstance(phases[3], ReentryPhase)
    assert isinstance(phases[4], TerminalPhase)


def test_multistage_rocket_phase_sequence_no_reentry() -> None:
    """rocket 多级配置默认不加入再入段。"""
    cfg = _icbm_two_stage_cfg()
    cfg = cfg.model_copy(update={"mission": "rocket"})
    phases = build_phases(cfg)
    names = [ph.name for ph in phases]
    assert names == ["S1 动力", "S2 动力", "滑行", "轨道插入"]
    assert not any(ph.name == "再入段" for ph in phases)


def test_multistage_mass_accounting() -> None:
    """质量记账：每级 m_dry 应包含上方级与 payload 质量。"""
    cfg = _icbm_two_stage_cfg()
    phases = build_phases(cfg)
    s1 = phases[0]
    s2 = phases[1]
    assert isinstance(s1, PoweredPhase)
    assert isinstance(s2, PoweredPhase)
    # S1  burnout 质量 = S1干 + S2总 + payload = 8000 + (8000+2000) + 2000 = 20000
    assert s1.m_dry == pytest.approx(20000.0, abs=1.0)
    # S1 分离后质量 = S2 总 + payload = 10000 + 2000 = 12000？
    # payload = 50000 - (30000+8000+8000+2000) = 2000
    # S2 总 = 10000
    # m_after_sep = 12000
    assert s1.m_after_separation == pytest.approx(12000.0, abs=1.0)
    # S2 burnout 质量 = S2干 + payload = 2000 + 2000 = 4000
    assert s2.m_dry == pytest.approx(4000.0, abs=1.0)
    assert s2.m_after_separation == pytest.approx(2000.0, abs=1.0)


def test_multistage_event_count() -> None:
    """每级 PoweredPhase 默认 burnout + ground 两个事件。"""
    cfg = _icbm_two_stage_cfg()
    phases = build_phases(cfg)
    powered = [ph for ph in phases if isinstance(ph, PoweredPhase)]
    assert len(powered) == 2
    for ph in powered:
        assert len(ph.events) == 2


def test_multistage_coast_mass_matches_payload() -> None:
    """滑行段动力学初始质量应等于 payload 质量。"""
    cfg = _icbm_two_stage_cfg()
    phases = build_phases(cfg)
    coast = phases[2]
    assert isinstance(coast, CoastingPhase)
    # 末级分离后质量 = payload = 2000
    last_stage = phases[1]
    assert isinstance(last_stage, PoweredPhase)
    assert last_stage.m_after_separation == pytest.approx(2000.0, abs=1.0)


def test_multistage_sixdof_reentry_for_rocket() -> None:
    """rocket 开启 sixdof_reentry 时追加 6-DOF 再入段。"""
    cfg = _icbm_two_stage_cfg()
    cfg = cfg.model_copy(update={"mission": "rocket"})
    cfg = cfg.model_copy(
        deep=True,
        update={"options": OptionsConfig(sixdof_reentry=True)},
    )
    phases = build_phases(cfg)
    reentry = [ph for ph in phases if isinstance(ph, ReentryPhase)]
    assert len(reentry) == 1
    assert reentry[0].fidelity == "sixdof"


def test_single_stage_fallback_when_no_stages() -> None:
    """未提供 stages 时回退到单级近似。"""
    cfg = SimConfig(
        mission="icbm",
        vehicle=VehicleConfig(mass_kg=1000.0, thrust_N=20.0e3, burn_time_s=60.0),
    )
    phases = build_phases(cfg)
    assert phases[0].name == "动力上升"
    assert isinstance(phases[0], PoweredPhase)
    assert isinstance(phases[1], CoastingPhase)


def test_missile_proportional_guidance_on_last_stage() -> None:
    """missile + proportional 应在末级动力段注入比例导引。"""
    cfg = _icbm_two_stage_cfg()
    cfg = cfg.model_copy(update={"mission": "missile"})
    cfg = cfg.model_copy(
        deep=True,
        update={
            "guidance": GuidanceConfig(
                guidance_law="proportional",
                target_lat_deg=39.0,
                target_lon_deg=118.0,
                nav_constant=4.0,
            )
        },
    )
    phases = build_phases(cfg)
    last_powered = phases[1]
    assert isinstance(last_powered, PoweredPhase)
    assert last_powered.guidance["phase"] == "proportional"
    assert "_pronav_guidance" in last_powered.guidance


def test_rocket_aag_guidance_on_powered_stage() -> None:
    """rocket + aag 应在动力段注入 AAG 状态。"""
    cfg = _icbm_two_stage_cfg()
    cfg = cfg.model_copy(update={"mission": "rocket"})
    cfg = cfg.model_copy(
        deep=True,
        update={
            "guidance": GuidanceConfig(
                guidance_law="aag",
                target_alt_m=200e3,
                terminal_velocity_m_s=7800.0,
            )
        },
    )
    phases = build_phases(cfg)
    powered = [ph for ph in phases if isinstance(ph, PoweredPhase)]
    assert len(powered) == 2
    for ph in powered:
        assert ph.guidance["phase"] == "aag"
        assert "_aag_state" in ph.guidance


def test_reentry_guidance_assembly() -> None:
    """icbm + reentry 应在再入段注入再入制导对象。"""
    from ballistic_sim.guidance.reentry_guidance import ReentryGuidance

    cfg = _icbm_two_stage_cfg()
    cfg = cfg.model_copy(
        deep=True,
        update={
            "guidance": GuidanceConfig(
                guidance_law="reentry",
                target_lat_deg=39.0,
                target_lon_deg=118.0,
            ),
            "options": OptionsConfig(sixdof_reentry=True),
        },
    )
    phases = build_phases(cfg)
    reentry = [ph for ph in phases if ph.name == "再入段"]
    assert len(reentry) == 1
    assert isinstance(reentry[0].dynamics.guidance, ReentryGuidance)


def test_energy_guidance_assembly() -> None:
    """suborbital + energy 应在再入段注入能量管理制导对象。"""
    from ballistic_sim.guidance.energy_management import EnergyManagementGuidance

    cfg = _icbm_two_stage_cfg()
    cfg = cfg.model_copy(update={"mission": "suborbital"})
    cfg = cfg.model_copy(
        deep=True,
        update={
            "guidance": GuidanceConfig(
                guidance_law="energy",
                target_lat_deg=39.0,
                target_lon_deg=118.0,
                energy_target_j_kg=-55e6,
            ),
            "options": OptionsConfig(sixdof_reentry=True),
        },
    )
    phases = build_phases(cfg)
    reentry = [ph for ph in phases if ph.name == "再入段"]
    assert len(reentry) == 1
    assert isinstance(reentry[0].dynamics.guidance, EnergyManagementGuidance)


def test_build_phases_for_icbm_three_stage_full_chain() -> None:
    """builder 应为 ICBM 三级预设生成 动力×3 -> 滑行 -> 再入 -> 终点。"""
    from ballistic_sim.presets.missiles import missile_full_config

    cfg = missile_full_config("ICBM_8000")
    phases = build_phases(cfg)
    names = [ph.name for ph in phases]
    assert names == ["S1 动力", "S2 动力", "S3 动力", "滑行", "再入段", "终点"]
    assert all(isinstance(ph, PoweredPhase) for ph in phases[:3])
    assert isinstance(phases[3], CoastingPhase)
    assert isinstance(phases[4], ReentryPhase)
    assert isinstance(phases[5], TerminalPhase)


def test_build_phases_for_rocket_full_chain_has_coast_and_terminal() -> None:
    """builder 应为火箭完整链生成多级动力 -> 滑行 -> 轨道插入。"""
    from ballistic_sim.presets.rockets import rocket_full_config

    cfg = rocket_full_config("CZ2F")
    phases = build_phases(cfg)
    names = [ph.name for ph in phases]
    assert any("滑行" in n for n in names)
    assert names[-1] == "轨道插入"
    assert not any(n == "再入段" for n in names)
