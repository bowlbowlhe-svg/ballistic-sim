"""GUI 配置与 phase 构造辅助函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from ballistic_sim.config import (
    EnvironmentConfig,
    GuidanceConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    VehicleConfig,
)
from ballistic_sim.phases.base import Phase
from ballistic_sim.phases.builder import build_phases
from ballistic_sim.presets import (
    list_missiles,
    list_projectiles,
    list_rockets,
    m107_config,
    missile_full_config,
    rocket_full_config,
)

MISSIONS = ["projectile", "missile", "rocket", "icbm", "suborbital"]


def list_presets(mission: str) -> List[str]:
    """返回指定任务类型的可用预设名。"""
    if mission == "projectile":
        return list_projectiles()
    if mission == "missile":
        return list_missiles()
    if mission == "rocket":
        return list_rockets()
    return []


def default_preset(mission: str) -> Optional[str]:
    """返回指定任务类型的默认预设名。"""
    presets = list_presets(mission)
    if mission == "projectile":
        return "M107" if "M107" in presets else (presets[0] if presets else None)
    if mission == "missile":
        return presets[0] if presets else None
    if mission == "rocket":
        return "CZ2F" if "CZ2F" in presets else (presets[0] if presets else None)
    return None


def _default_vehicle_for_mission(mission: str) -> VehicleConfig:
    """为无预设任务构造合理的默认飞行器参数。"""
    if mission == "icbm":
        return VehicleConfig(mass_kg=21000.0, diameter_m=0.5, cd=0.3)
    if mission == "suborbital":
        return VehicleConfig(
            mass_kg=1000.0,
            diameter_m=0.5,
            cd=0.3,
            area_ref_m2=0.196,
            thrust_N=25.0e3,
            burn_time_s=40.0,
        )
    return VehicleConfig()


def build_config_and_phases(
    mission: str, preset: Optional[str] = None
) -> Tuple[SimConfig, List[Phase]]:
    """根据任务类型与预设构造 ``SimConfig`` 与 ``Phase`` 列表。"""
    if mission not in MISSIONS:
        raise ValueError(f"未支持的任务类型: {mission}")

    if mission == "projectile":
        name = preset or "M107"
        cfg = m107_config() if name == "M107" else _projectile_config(name)
        phases = build_phases(cfg)
        return cfg, phases

    if mission == "missile":
        name = preset or default_preset(mission) or list_missiles()[0]
        cfg = missile_full_config(name)
        phases = build_phases(cfg)
        return cfg, phases

    if mission == "rocket":
        name = preset or "CZ2F"
        cfg = rocket_full_config(name)
        phases = build_phases(cfg)
        return cfg, phases

    # icbm / suborbital 使用通用配置 + builder
    vehicle = _default_vehicle_for_mission(mission)
    cfg = SimConfig(
        mission=mission,
        vehicle=vehicle,
        launch=LaunchConfig(),
        environment=EnvironmentConfig(),
        guidance=GuidanceConfig(),
        options=OptionsConfig(
            integrator="DOP853",
            rtol=1e-6,
            atol=1e-9,
            max_step=1.0,
            terminate_impact=True,
        ),
    )
    phases = build_phases(cfg)
    return cfg, phases


def _projectile_config(name: str) -> SimConfig:
    """加载非 M107 弹丸预设配置。"""
    from ballistic_sim.presets.projectiles import _projectile_config_from_preset

    return _projectile_config_from_preset(name)


def submodel_classes() -> Dict[str, type]:
    """返回 SimConfig 子模型类字典。"""
    return {
        "vehicle": VehicleConfig,
        "launch": LaunchConfig,
        "environment": EnvironmentConfig,
        "guidance": GuidanceConfig,
        "options": OptionsConfig,
    }


def dump_config_to_yaml(cfg: SimConfig, path: Path) -> None:
    """将 ``SimConfig`` 保存为 YAML 文件。"""
    data = cfg.model_dump(mode="json")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_config_from_yaml(path: Path) -> SimConfig:
    """从 YAML 文件加载 ``SimConfig``。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("YAML 内容不是字典")
    return SimConfig(**data)
