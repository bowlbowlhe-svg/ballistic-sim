"""三段式开环制导单元测试。

验证：
- 当地上/水平方向、飞行路径角计算；
- 垂直起飞、程序俯仰、重力转弯三段的角度切换；
- 上面级线性俯仰律从 gamma0 压到 gamma_end。
"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.constants import WGS84_A
from ballistic_sim.frames import ecef_to_eci, enu_basis
from ballistic_sim.guidance.open_loop import (
    flight_path_angle,
    local_horizontal,
    local_up,
    thrust_dir_eci,
    thrust_dir_upperstage,
)

_LAT = 30.0
_LON = 0.0
_AZ = 90.0  # 向东发射，使下程与当地水平面正交


def _sample_state():
    """构造一个非零纬度、非零下程方向的状态。"""
    # 在 ECI/ECEF 历元对齐时，位置近似为 ECEF
    lat = np.radians(_LAT)
    r = np.array(
        [WGS84_A * np.cos(lat), 0.0, WGS84_A * np.sin(lat)],
        dtype=float,
    )
    up = local_up(r)
    # 下程方向：东向水平
    e_hat, _, _ = enu_basis(_LAT, _LON)
    dr_ecef = np.sin(np.radians(_AZ)) * e_hat + np.cos(np.radians(_AZ)) * np.array(
        [-np.sin(lat), 0.0, np.cos(lat)], dtype=float
    )
    dr_eci = ecef_to_eci(dr_ecef, 0.0)
    return r, up, dr_eci


def test_local_up_normalizes() -> None:
    """local_up 应返回位置矢量单位化。"""
    r = np.array([1.0, 2.0, 3.0])
    up = local_up(r)
    assert np.isclose(np.linalg.norm(up), 1.0)
    assert np.allclose(up, r / np.linalg.norm(r))


def test_flight_path_angle_horizontal() -> None:
    """水平速度时飞行路径角为 0。"""
    r = np.array([0.0, 0.0, WGS84_A])
    v = np.array([100.0, 0.0, 0.0])
    assert flight_path_angle(r, v) == pytest.approx(0.0, abs=1e-12)


def test_flight_path_angle_vertical_up() -> None:
    """竖直向上速度时飞行路径角为 +90°。"""
    r = np.array([0.0, 0.0, WGS84_A])
    v = np.array([0.0, 0.0, 100.0])
    assert flight_path_angle(r, v) == pytest.approx(np.pi / 2.0, abs=1e-12)


def test_local_horizontal_in_vertical_plane() -> None:
    """local_horizontal 返回的矢量应在 r 与 ref 构成的竖直平面内且与当地水平面正交。"""
    r, up, dr = _sample_state()
    h = local_horizontal(r, dr)
    assert np.isclose(np.linalg.norm(h), 1.0)
    assert np.isclose(np.dot(h, up), 0.0, atol=1e-12)
    # h 在 up 与 dr 张成的平面内：三重积为 0
    assert np.isclose(np.dot(np.cross(up, dr), h), 0.0, atol=1e-12)


def test_thrust_dir_vertical_phase() -> None:
    """t < t_pitchover 时推力沿当地上方向。"""
    r, up, _ = _sample_state()
    guid = {
        "phase": "ascent",
        "lat_deg": _LAT,
        "lon_deg": _LON,
        "azimuth_deg": _AZ,
        "t_pitchover": 10.0,
        "t_kick_end": 30.0,
        "kick_deg": 30.0,
    }
    d = thrust_dir_eci(0.0, r, np.zeros(3), guid)
    assert np.allclose(d, up, atol=1e-12)


def test_thrust_dir_pitchover_interpolates() -> None:
    """程序俯仰段内推力方向从竖直线性倾斜到 kick_deg。"""
    r, up, dr = _sample_state()
    t_pitchover = 10.0
    t_kick_end = 30.0
    kick_deg = 30.0
    guid = {
        "phase": "ascent",
        "lat_deg": _LAT,
        "lon_deg": _LON,
        "azimuth_deg": _AZ,
        "t_pitchover": t_pitchover,
        "t_kick_end": t_kick_end,
        "kick_deg": kick_deg,
    }
    # 起点仍为上
    d_start = thrust_dir_eci(t_pitchover, r, np.zeros(3), guid)
    assert np.allclose(d_start, up, atol=1e-6)

    # 终点倾斜 kick_deg（在 t_kick_end 之前瞬间仍处于程序俯仰段）
    d_end = thrust_dir_eci(t_kick_end - 1e-9, r, np.zeros(3), guid)
    assert np.isclose(np.dot(d_end, up), np.cos(np.radians(kick_deg)), atol=1e-6)
    assert np.isclose(np.dot(d_end, dr), np.sin(np.radians(kick_deg)), atol=1e-6)

    # 中点倾斜一半
    t_mid = 0.5 * (t_pitchover + t_kick_end)
    d_mid = thrust_dir_eci(t_mid, r, np.zeros(3), guid)
    half_kick = np.radians(kick_deg / 2.0)
    assert np.isclose(np.dot(d_mid, up), np.cos(half_kick), atol=1e-6)
    assert np.isclose(np.dot(d_mid, dr), np.sin(half_kick), atol=1e-6)


def test_thrust_dir_gravity_turn_follows_velocity() -> None:
    """t >= t_kick_end 且存在相对速度时推力沿相对速度方向。"""
    r, up, dr = _sample_state()
    guid = {
        "phase": "ascent",
        "lat_deg": _LAT,
        "lon_deg": _LON,
        "azimuth_deg": _AZ,
        "t_pitchover": 10.0,
        "t_kick_end": 30.0,
        "kick_deg": 30.0,
    }
    # 给定一个沿下程的水平速度
    v = 500.0 * dr
    d = thrust_dir_eci(40.0, r, v, guid)
    v_rel = v - np.cross(np.array([0.0, 0.0, 7.292115e-5]), r)
    assert np.allclose(d, v_rel / np.linalg.norm(v_rel), atol=1e-6)


def test_thrust_dir_upperstage_linear_pitch() -> None:
    """上面级线性俯仰律在 t_us_dur 内把飞行路径角从 gamma0 压到 gamma_end。"""
    r, up, dr = _sample_state()
    # 构造一个与水平面成 30° 向上的速度
    gamma0 = np.radians(30.0)
    v = 1000.0 * (np.cos(gamma0) * dr + np.sin(gamma0) * up)

    guid = {
        "phase": "upperstage",
        "t_us_start": 100.0,
        "t_us_dur": 50.0,
        "gamma_end_deg": 0.0,
        "lat_deg": _LAT,
        "lon_deg": _LON,
        "azimuth_deg": _AZ,
    }

    # 起始时刻对应 gamma0
    d0 = thrust_dir_upperstage(100.0, r, v, guid)
    assert np.isclose(np.dot(d0, up), np.sin(gamma0), atol=1e-6)
    assert np.isclose(np.dot(d0, dr), np.cos(gamma0), atol=1e-6)

    # 结束时刻压到 gamma_end=0
    d1 = thrust_dir_upperstage(150.0, r, v, guid)
    assert np.isclose(np.dot(d1, up), 0.0, atol=1e-6)
    assert np.isclose(np.dot(d1, dr), 1.0, atol=1e-6)

    # 中间时刻线性插值
    t_mid = 125.0
    d_mid = thrust_dir_upperstage(t_mid, r, v, guid)
    gamma_mid = np.arcsin(np.clip(np.dot(d_mid, up), -1.0, 1.0))
    assert gamma_mid == pytest.approx(gamma0 / 2.0, abs=1e-6)
