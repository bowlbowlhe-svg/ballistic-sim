"""配置文件 YAML/JSON 加载与保存测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ballistic_sim.config import (
    GuidanceConfig,
    LaunchConfig,
    SimConfig,
    StageConfig,
    VehicleConfig,
    apply_overrides,
    load_config,
    save_config,
)


def _minimal_rocket_dict() -> dict:
    return {
        "mission": "icbm",
        "vehicle": {
            "mass_kg": 3000.0,
            "diameter_m": 1.0,
            "cd": 0.3,
            "stages": [
                {
                    "name": "S1",
                    "thrust_sl": 50.0e3,
                    "thrust_vac": 60.0e3,
                    "isp_vac": 250.0,
                    "m_prop": 1500.0,
                    "m_dry": 500.0,
                    "diameter_m": 1.0,
                }
            ],
        },
        "launch": {
            "lat_deg": 0.0,
            "lon_deg": 0.0,
            "alt_m": 0.0,
            "azimuth_deg": 90.0,
            "elevation_deg": 90.0,
        },
        "guidance": {"guidance_law": "none", "kick_deg": 10.0},
    }


def test_load_yaml_config(tmp_path: Path) -> None:
    """load_config 可正确读取 YAML 配置。"""
    path = tmp_path / "cfg.yaml"
    data = _minimal_rocket_dict()
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    cfg = load_config(path)
    assert cfg.mission == "icbm"
    assert cfg.vehicle.mass_kg == 3000.0
    assert len(cfg.vehicle.stages) == 1
    assert cfg.vehicle.stages[0].name == "S1"


def test_load_json_config(tmp_path: Path) -> None:
    """load_config 可正确读取 JSON 配置。"""
    path = tmp_path / "cfg.json"
    data = _minimal_rocket_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    cfg = load_config(path)
    assert cfg.mission == "icbm"
    assert cfg.vehicle.stages[0].thrust_vac == 60.0e3


def test_load_unsupported_extension_raises(tmp_path: Path) -> None:
    """不支持的扩展名应抛出 ValueError。"""
    path = tmp_path / "cfg.toml"
    path.write_text("mission = 'icbm'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持的配置格式"):
        load_config(path)


def test_save_yaml_roundtrip(tmp_path: Path) -> None:
    """save_config YAML 往返后配置等价。"""
    original = SimConfig(
        mission="rocket",
        vehicle=VehicleConfig(mass_kg=1000.0, thrust_N=20.0e3, burn_time_s=60.0),
        launch=LaunchConfig(lat_deg=30.0, lon_deg=120.0, azimuth_deg=90.0),
        guidance=GuidanceConfig(kick_deg=5.0),
    )
    path = tmp_path / "saved.yaml"
    save_config(original, path)
    loaded = load_config(path)
    assert loaded.mission == original.mission
    assert loaded.vehicle.mass_kg == original.vehicle.mass_kg
    assert loaded.vehicle.thrust_N == original.vehicle.thrust_N
    assert loaded.launch.lat_deg == original.launch.lat_deg
    assert loaded.guidance.kick_deg == original.guidance.kick_deg


def test_save_json_roundtrip(tmp_path: Path) -> None:
    """save_config JSON 往返后配置等价。"""
    original = SimConfig(
        mission="icbm",
        vehicle=VehicleConfig(
            mass_kg=5000.0,
            stages=[
                StageConfig(
                    name="S1",
                    thrust_sl=100.0e3,
                    thrust_vac=120.0e3,
                    isp_vac=260.0,
                    m_prop=3000.0,
                    m_dry=1000.0,
                )
            ],
        ),
        launch=LaunchConfig(azimuth_deg=45.0),
    )
    path = tmp_path / "saved.json"
    save_config(original, path)
    loaded = load_config(path)
    assert loaded.mission == original.mission
    assert loaded.vehicle.stages[0].name == "S1"
    assert loaded.vehicle.stages[0].m_prop == 3000.0


def test_save_unsupported_extension_raises(tmp_path: Path) -> None:
    """保存到不支持的扩展名应抛出 ValueError。"""
    cfg = SimConfig(mission="projectile")
    path = tmp_path / "cfg.toml"
    with pytest.raises(ValueError, match="不支持的配置格式"):
        save_config(cfg, path)


def test_cli_override_priority_over_config(tmp_path: Path) -> None:
    """CLI 点路径覆盖优先级应高于配置文件中的值。"""
    base = SimConfig(
        mission="icbm",
        vehicle=VehicleConfig(mass_kg=1000.0),
        launch=LaunchConfig(elevation_deg=45.0, azimuth_deg=90.0),
    )
    path = tmp_path / "base.yaml"
    save_config(base, path)
    loaded = load_config(path)
    overridden = apply_overrides(
        loaded, {"launch.elevation_deg": 60.0, "launch.azimuth_deg": 180.0}
    )
    assert overridden.launch.elevation_deg == 60.0
    assert overridden.launch.azimuth_deg == 180.0


def test_apply_overrides_nested_create() -> None:
    """apply_overrides 可在嵌套路径不存在时自动创建中间 dict。"""
    cfg = SimConfig(mission="projectile")
    overridden = apply_overrides(cfg, {"options.monte_carlo.num_samples": 50})
    assert overridden.options.monte_carlo is not None
    assert overridden.options.monte_carlo.num_samples == 50


def test_load_config_preserves_extra_stage_fields(tmp_path: Path) -> None:
    """StageConfig 的 extra 字段在加载后仍可访问。"""
    data = _minimal_rocket_dict()
    data["vehicle"]["stages"][0]["drag"] = "G1"
    path = tmp_path / "cfg.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    cfg = load_config(path)
    assert cfg.vehicle.stages[0].model_extra.get("drag") == "G1"
