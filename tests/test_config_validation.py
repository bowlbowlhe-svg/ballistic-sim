"""业务级跨字段配置校验测试。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ballistic_sim.config import (
    GuidanceConfig,
    LaunchConfig,
    MonteCarloConfig,
    OptionsConfig,
    SimConfig,
    StageConfig,
    VehicleConfig,
    load_config,
    validate_config,
)


def _find_issue(issues, path: str, severity: str | None = None):
    """Helper：按 path（及可选 severity）查找校验问题。"""
    for issue in issues:
        if issue.path == path:
            if severity is None or issue.severity == severity:
                return issue
    return None


def test_tw_ratio_error_for_low_thrust():
    """动力任务 T/W <= 1 应报 ERROR。"""
    cfg = SimConfig(
        mission="rocket",
        vehicle=VehicleConfig(mass_kg=1000.0, thrust_N=5000.0, burn_time_s=60.0),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "vehicle.thrust_N", "ERROR")
    assert issue is not None
    assert "T/W" in issue.message


def test_tw_ok_for_sufficient_thrust():
    """T/W > 1 时不应报 ERROR。"""
    cfg = SimConfig(
        mission="rocket",
        vehicle=VehicleConfig(mass_kg=1000.0, thrust_N=20000.0, burn_time_s=60.0),
    )
    issues = validate_config(cfg)
    assert _find_issue(issues, "vehicle.thrust_N", "ERROR") is None


def test_thrust_without_burn_time_warning():
    """设置了 thrust_N 但未设置 burn_time_s 时应报 WARNING。"""
    cfg = SimConfig(
        mission="rocket",
        vehicle=VehicleConfig(mass_kg=1000.0, thrust_N=20000.0),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "vehicle", "WARNING")
    assert issue is not None
    assert "burn_time_s" in issue.message


def test_zero_thrust_with_positive_burn_time_error():
    """推力为零但 burn_time_s > 0 时报 ERROR。"""
    cfg = SimConfig(
        mission="rocket",
        vehicle=VehicleConfig(mass_kg=1000.0, thrust_N=0.0, burn_time_s=60.0),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "vehicle.thrust_N", "ERROR")
    assert issue is not None


def test_negative_burn_time_error():
    """burn_time_s <= 0 时报 ERROR。"""
    cfg = SimConfig(
        mission="rocket",
        vehicle=VehicleConfig.model_construct(mass_kg=1000.0, thrust_N=20000.0, burn_time_s=0.0),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "vehicle.burn_time_s", "ERROR")
    assert issue is not None


def test_negative_elevation_warning_for_projectile():
    """projectile 负仰角应报 WARNING。"""
    cfg = SimConfig(
        mission="projectile",
        launch=LaunchConfig(elevation_deg=-5.0),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "launch.elevation_deg", "WARNING")
    assert issue is not None


def test_high_elevation_warning():
    """仰角 > 85° 应报 WARNING。"""
    cfg = SimConfig(
        mission="rocket",
        launch=LaunchConfig(elevation_deg=88.0),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "launch.elevation_deg", "WARNING")
    assert issue is not None
    assert "过大" in issue.message


def test_missile_without_target_warning():
    """missile 未设置目标经纬度应报 WARNING。"""
    cfg = SimConfig(
        mission="missile",
        guidance=GuidanceConfig(),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "guidance", "WARNING")
    assert issue is not None
    assert "目标经纬度" in issue.message


def test_projectile_with_target_warning():
    """projectile 设置目标经纬度应报 WARNING。"""
    cfg = SimConfig(
        mission="projectile",
        guidance=GuidanceConfig(target_lat_deg=0.0, target_lon_deg=0.0),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "guidance", "WARNING")
    assert issue is not None


def test_mc_negative_samples_error():
    """Monte Carlo 样本数 <= 0 报 ERROR。"""
    cfg = SimConfig(
        mission="projectile",
        options=OptionsConfig(
            monte_carlo=MonteCarloConfig.model_construct(num_samples=0, backend="batch")
        ),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "options.monte_carlo.num_samples", "ERROR")
    assert issue is not None


def test_mc_invalid_backend_error():
    """Monte Carlo backend 非法时报 ERROR。"""
    cfg = SimConfig(
        mission="projectile",
        options=OptionsConfig(
            monte_carlo=MonteCarloConfig.model_construct(num_samples=10, backend="mpi")
        ),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "options.monte_carlo.backend", "ERROR")
    assert issue is not None


def test_mc_small_samples_warning():
    """Monte Carlo 样本数 < 10 报 WARNING。"""
    cfg = SimConfig(
        mission="projectile",
        options=OptionsConfig(monte_carlo=MonteCarloConfig(num_samples=5, backend="batch")),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "options.monte_carlo.num_samples", "WARNING")
    assert issue is not None


def test_stages_empty_error():
    """动力任务 stages 为空列表时报 ERROR。"""
    cfg = SimConfig(
        mission="icbm",
        vehicle=VehicleConfig(stages=[]),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "vehicle.stages", "ERROR")
    assert issue is not None


def test_stages_total_mass_exceeds_error():
    """各级质量之和超过 vehicle.mass_kg 时报 ERROR。"""
    cfg = SimConfig(
        mission="icbm",
        vehicle=VehicleConfig(
            mass_kg=1000.0,
            stages=[
                StageConfig(
                    thrust_sl=20000.0,
                    thrust_vac=22000.0,
                    isp_vac=250.0,
                    m_prop=600.0,
                    m_dry=500.0,
                )
            ],
        ),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "vehicle.mass_kg", "ERROR")
    assert issue is not None


def test_stages_tw_error():
    """多级起飞 T/W <= 1 时报 ERROR。"""
    cfg = SimConfig(
        mission="icbm",
        vehicle=VehicleConfig(
            mass_kg=10000.0,
            stages=[
                StageConfig(
                    thrust_sl=50000.0,
                    thrust_vac=60000.0,
                    isp_vac=250.0,
                    m_prop=5000.0,
                    m_dry=2000.0,
                )
            ],
        ),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "vehicle.stages[0].thrust_sl", "ERROR")
    assert issue is not None
    assert "T/W" in issue.message


def test_stages_valid_no_error():
    """合法多级配置不产生 ERROR。"""
    cfg = SimConfig(
        mission="icbm",
        vehicle=VehicleConfig(
            mass_kg=50000.0,
            stages=[
                StageConfig(
                    name="S1",
                    thrust_sl=1.2e6,
                    thrust_vac=1.35e6,
                    isp_vac=260.0,
                    m_prop=30000.0,
                    m_dry=8000.0,
                ),
                StageConfig(
                    name="S2",
                    thrust_sl=200.0e3,
                    thrust_vac=250.0e3,
                    isp_vac=300.0,
                    m_prop=8000.0,
                    m_dry=2000.0,
                ),
            ],
        ),
    )
    issues = validate_config(cfg)
    errors = [i for i in issues if i.severity == "ERROR"]
    assert not errors


def test_proportional_guidance_for_non_missile_warning():
    """proportional 制导律用于非 missile 任务时报 WARNING。"""
    cfg = SimConfig(
        mission="rocket",
        guidance=GuidanceConfig(guidance_law="proportional"),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "guidance.guidance_law", "WARNING")
    assert issue is not None


def test_stage_negative_m_prop_error():
    """stage m_prop < 0 时报 ERROR。"""
    stage = StageConfig.model_construct(
        thrust_sl=1.2e6,
        thrust_vac=1.35e6,
        isp_vac=260.0,
        m_prop=-100.0,
        m_dry=8000.0,
    )
    cfg = SimConfig(
        mission="icbm",
        vehicle=VehicleConfig.model_construct(mass_kg=50000.0, stages=[stage]),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "vehicle.stages[0].m_prop", "ERROR")
    assert issue is not None


def test_stage_negative_m_dry_error():
    """stage m_dry < 0 时报 ERROR。"""
    stage = StageConfig.model_construct(
        thrust_sl=1.2e6,
        thrust_vac=1.35e6,
        isp_vac=260.0,
        m_prop=30000.0,
        m_dry=-100.0,
    )
    cfg = SimConfig(
        mission="icbm",
        vehicle=VehicleConfig.model_construct(mass_kg=50000.0, stages=[stage]),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "vehicle.stages[0].m_dry", "ERROR")
    assert issue is not None


def test_stage_non_positive_thrust_vac_error():
    """stage thrust_vac <= 0 时报 ERROR。"""
    cfg = SimConfig(
        mission="icbm",
        vehicle=VehicleConfig(
            mass_kg=50000.0,
            stages=[
                StageConfig(
                    thrust_sl=1.2e6,
                    thrust_vac=0.0,
                    isp_vac=260.0,
                    m_prop=30000.0,
                    m_dry=8000.0,
                )
            ],
        ),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "vehicle.stages[0].thrust_vac", "ERROR")
    assert issue is not None


def test_stage_non_positive_isp_vac_error():
    """stage isp_vac <= 0 时报 ERROR。"""
    stage = StageConfig.model_construct(
        thrust_sl=1.2e6,
        thrust_vac=1.35e6,
        isp_vac=0.0,
        m_prop=30000.0,
        m_dry=8000.0,
    )
    cfg = SimConfig(
        mission="icbm",
        vehicle=VehicleConfig.model_construct(mass_kg=50000.0, stages=[stage]),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "vehicle.stages[0].isp_vac", "ERROR")
    assert issue is not None


def test_stage_thrust_sl_greater_than_vac_warning():
    """stage thrust_sl > thrust_vac 时报 WARNING。"""
    cfg = SimConfig(
        mission="icbm",
        vehicle=VehicleConfig(
            mass_kg=50000.0,
            stages=[
                StageConfig(
                    thrust_sl=1.4e6,
                    thrust_vac=1.35e6,
                    isp_vac=260.0,
                    m_prop=30000.0,
                    m_dry=8000.0,
                )
            ],
        ),
    )
    issues = validate_config(cfg)
    issue = _find_issue(issues, "vehicle.stages[0].thrust_sl", "WARNING")
    assert issue is not None


def test_load_config_rejects_non_dict_json():
    """load_config 对顶层非 dict 的 JSON 应抛 ValueError。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cfg.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(ValueError, match="顶层必须是字典"):
            load_config(str(path))
