"""补充事件工厂单元测试以提升覆盖率。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.constants import WGS84_A, GM_EARTH
from ballistic_sim.frames import ecef_to_eci, geodetic_to_ecef
from ballistic_sim.phases.events import (
    make_apogee_event,
    make_burnout_event,
    make_fairing_event_h,
    make_fairing_event_q,
    make_fairing_jettison_event,
    make_ground_event,
    make_orbit_insertion_event,
    make_stage_separation_event,
    make_target_distance_event,
)


def test_ground_event_enu() -> None:
    """ENU 落地事件直接比较高度。"""
    ev = make_ground_event(frame="ENU", h_target=0.0)
    assert ev(0.0, np.array([0.0, 0.0, 10.0])) == 10.0
    assert ev(0.0, np.array([0.0, 0.0, -1.0])) == -1.0
    assert ev.terminal is True


def test_apogee_event_enu() -> None:
    """ENU 远地点事件比较竖直速度。"""
    ev = make_apogee_event(frame="ENU")
    assert ev(0.0, np.array([0.0, 0.0, 100.0, 0.0, 0.0, 10.0])) == 10.0
    assert ev.direction == -1
    assert ev.terminal is False


def test_burnout_event() -> None:
    """燃尽事件比较质量。"""
    ev = make_burnout_event(m_dry=1000.0)
    assert ev(0.0, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1500.0])) == 500.0
    assert ev(0.0, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 500.0])) == -500.0


def test_stage_separation_event() -> None:
    """固定时刻分离事件。"""
    ev = make_stage_separation_event(t_sep=100.0)
    assert ev(50.0, np.zeros(7)) == 50.0
    assert ev(150.0, np.zeros(7)) == -50.0


def test_stage_separation_event_none() -> None:
    """t_sep 为 None 时事件恒为正（不触发）。"""
    ev = make_stage_separation_event(t_sep=None)
    assert ev(0.0, np.zeros(7)) == 1.0


def test_target_distance_event() -> None:
    """与目标距离事件。"""
    target = np.array([100.0, 0.0, 0.0])
    ev = make_target_distance_event(target, distance_m=10.0)
    r = np.array([95.0, 0.0, 0.0])
    val = ev(0.0, np.concatenate([r, np.zeros(3)]))
    assert val < 0.0


def test_orbit_insertion_event() -> None:
    """轨道插入半长轴事件。"""
    ev = make_orbit_insertion_event({"peri_km": 200.0, "apo_km": 350.0})
    a_target = 0.5 * (WGS84_A + 200e3 + WGS84_A + 350e3)
    r = np.array([a_target, 0.0, 0.0])
    v = np.array([0.0, np.sqrt(GM_EARTH / a_target), 0.0])
    y = np.concatenate([r, v])
    assert abs(ev(0.0, y)) < 1.0
    assert ev.terminal is True


def test_orbit_insertion_event_h_km() -> None:
    """轨道插入事件支持 h_km 目标。"""
    ev = make_orbit_insertion_event({"h_km": 300.0})
    assert ev.terminal is True


def test_orbit_insertion_event_invalid() -> None:
    """target 缺少必要字段时抛出 ValueError。"""
    with pytest.raises(ValueError):
        make_orbit_insertion_event({})


def test_fairing_event_h_enu() -> None:
    """ENU 抛整流罩高度事件。"""
    ev = make_fairing_event_h(100e3, frame="ENU")
    assert ev(0.0, np.array([0.0, 0.0, 50e3])) < 0.0
    assert ev(0.0, np.array([0.0, 0.0, 150e3])) > 0.0
    assert ev.terminal is False


def test_fairing_event_q() -> None:
    """动压下穿抛整流罩事件。"""

    def q_fn(t: float, y: np.ndarray) -> float:
        return 100.0

    ev = make_fairing_event_q(q_fn, q_thresh_pa=200.0)
    assert ev(0.0, np.zeros(7)) < 0.0
    assert ev.terminal is False


def test_fairing_event_h_eci() -> None:
    """ECI 模式下几何高度抛整流罩事件。"""
    ev = make_fairing_event_h(100e3, frame="ECI", theta0=0.0)
    r_ecef = geodetic_to_ecef(0.0, 0.0, 150e3)
    r_eci = ecef_to_eci(r_ecef, 0.0)
    y = np.concatenate([r_eci, np.zeros(3)])
    assert ev(0.0, y) > 0.0

    r_ecef2 = geodetic_to_ecef(0.0, 0.0, 50e3)
    r_eci2 = ecef_to_eci(r_ecef2, 0.0)
    y2 = np.concatenate([r_eci2, np.zeros(3)])
    assert ev(0.0, y2) < 0.0
    assert ev.terminal is False


def test_fairing_jettison_q_mode() -> None:
    """mode='q' 时按动压下穿阈值构造事件。"""

    def q_fn(t: float, y: np.ndarray) -> float:
        return 100.0

    ev = make_fairing_jettison_event(mode="q", q_fn=q_fn, q_thresh_pa=200.0)
    assert ev(0.0, np.zeros(7)) < 0.0
    assert ev.terminal is False


def test_fairing_jettison_invalid_mode() -> None:
    """无效抛罩模式应抛出 ValueError。"""
    with pytest.raises(ValueError) as exc_info:
        make_fairing_jettison_event(mode="invalid", h_m=100e3)
    assert "未知的抛罩模式" in str(exc_info.value)
