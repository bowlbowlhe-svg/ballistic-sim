"""VehicleConfig 默认值与覆盖测试。"""

from __future__ import annotations

import pytest

from ballistic_sim.config import VehicleConfig


def test_vehicle_config_default_values() -> None:
    """VehicleConfig() 的默认值应符合 MPM / 6-DOF 当前默认。"""
    cfg = VehicleConfig()
    assert cfg.mass_kg == pytest.approx(1.0)
    assert cfg.diameter_m == pytest.approx(0.1)
    assert cfg.cd == pytest.approx(0.3)
    assert cfg.twist_cal == pytest.approx(20.0)
    assert cfg.Ix == pytest.approx(0.1)
    assert cfg.It == pytest.approx(1.0)
    assert cfg.x_cp_cg == pytest.approx(0.05)
    assert cfg.area_ref_m2 is None
    assert cfg.thrust_N is None
    assert cfg.burn_time_s is None
    assert cfg.stages is None
    assert cfg.drag_law is None


def test_vehicle_config_override_values() -> None:
    """显式覆盖 VehicleConfig 字段仍应生效。"""
    cfg = VehicleConfig(
        mass_kg=43.2,
        diameter_m=0.155,
        cd=0.78,
        twist_cal=25.0,
        Ix=0.144,
        It=1.22,
        x_cp_cg=0.03,
        drag_law="G7",
    )
    assert cfg.mass_kg == pytest.approx(43.2)
    assert cfg.diameter_m == pytest.approx(0.155)
    assert cfg.cd == pytest.approx(0.78)
    assert cfg.twist_cal == pytest.approx(25.0)
    assert cfg.Ix == pytest.approx(0.144)
    assert cfg.It == pytest.approx(1.22)
    assert cfg.x_cp_cg == pytest.approx(0.03)
    assert cfg.drag_law == "G7"


def test_vehicle_config_optional_fields_must_be_positive() -> None:
    """Ix / It / twist_cal 必须为正（pydantic 校验）。"""
    with pytest.raises(ValueError):
        VehicleConfig(Ix=0.0)
    with pytest.raises(ValueError):
        VehicleConfig(It=-1.0)
    with pytest.raises(ValueError):
        VehicleConfig(twist_cal=-20.0)
