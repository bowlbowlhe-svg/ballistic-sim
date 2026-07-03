"""补充开环制导单元测试以提升覆盖率。"""

from __future__ import annotations

import numpy as np

from ballistic_sim.constants import WGS84_A
from ballistic_sim.frames import ecef_to_eci, enu_basis
from ballistic_sim.guidance.open_loop import (
    _pitch_dir_eci,
    _safe_normalize,
    thrust_dir_eci,
    thrust_dir_upperstage,
)

_LAT = 30.0
_LON = 0.0
_AZ = 90.0


def _sample_state():
    lat = np.radians(_LAT)
    r = np.array(
        [WGS84_A * np.cos(lat), 0.0, WGS84_A * np.sin(lat)],
        dtype=float,
    )
    e_hat, _n, _u = enu_basis(_LAT, _LON)
    dr_ecef = e_hat
    dr_eci = ecef_to_eci(dr_ecef, 0.0)
    return r, dr_eci


def test_safe_normalize_zero_fallback() -> None:
    """_safe_normalize 对零矢量返回 fallback。"""
    fallback = np.array([1.0, 0.0, 0.0])
    out = _safe_normalize(np.zeros(3), fallback)
    assert np.allclose(out, fallback)


def test_pitch_dir_eci_zero_horizontal() -> None:
    """下程与当地上平行时，_pitch_dir_eci 仍返回归一化方向。"""
    up = np.array([0.0, 0.0, 1.0])
    d = _pitch_dir_eci(up, up, 30.0)
    assert np.isclose(np.linalg.norm(d), 1.0)


def test_thrust_dir_upperstage_pitch_rate() -> None:
    """上面级制导使用 pitch_rate_dps 分支。"""
    r = np.array([0.0, 0.0, WGS84_A])
    v = np.array([100.0, 0.0, 100.0])
    guid = {
        "phase": "upperstage",
        "t_us_start": 0.0,
        "gamma_end_deg": 0.0,
        "pitch_rate_dps": 1.0,
        "lat_deg": 0.0,
        "lon_deg": 0.0,
        "azimuth_deg": 90.0,
    }
    d0 = thrust_dir_upperstage(0.0, r, v, guid)
    d10 = thrust_dir_upperstage(10.0, r, v, guid)
    # 10°/s 转动 10 s 后应接近水平
    assert d10[2] < d0[2]


def test_thrust_dir_upperstage_no_rate_or_dur() -> None:
    """上面级制导无 rate 也无 dur 时直接取 gamma_end。"""
    r = np.array([0.0, 0.0, WGS84_A])
    v = np.array([100.0, 0.0, 100.0])
    guid = {
        "phase": "upperstage",
        "t_us_start": 0.0,
        "gamma_end_deg": 0.0,
        "lat_deg": 0.0,
        "lon_deg": 0.0,
        "azimuth_deg": 90.0,
    }
    d = thrust_dir_upperstage(5.0, r, v, guid)
    # gamma_end=0 表示水平
    assert abs(d[2]) < 1e-6


def test_thrust_dir_upperstage_no_azimuth_fallback() -> None:
    """上面级制导无方位信息时回退到默认水平方向。"""
    r = np.array([0.0, 0.0, WGS84_A])
    v = np.zeros(3)
    guid = {
        "phase": "upperstage",
        "t_us_start": 0.0,
        "gamma_end_deg": 0.0,
    }
    d = thrust_dir_upperstage(0.0, r, v, guid)
    assert np.isclose(np.linalg.norm(d), 1.0)


def test_thrust_dir_eci_zero_span() -> None:
    """t_pitchover == t_kick_end 时重力转弯分支回退到 kick 方向。"""
    r, dr = _sample_state()
    guid = {
        "phase": "ascent",
        "lat_deg": _LAT,
        "lon_deg": _LON,
        "azimuth_deg": _AZ,
        "t_pitchover": 10.0,
        "t_kick_end": 10.0,
        "kick_deg": 30.0,
    }
    d = thrust_dir_eci(10.0, r, np.zeros(3), guid)
    # 零相对速度时回退到下程 kick 方向（只需验证为单位矢量且不在竖直方向）
    assert np.isclose(np.linalg.norm(d), 1.0)
    assert abs(d[2]) < np.cos(np.radians(30.0))


def test_thrust_dir_eci_peg_without_state() -> None:
    """phase='peg' 但缺少 _peg_state 时回退上面级律。"""
    r, dr = _sample_state()
    guid = {
        "phase": "peg",
        "t_us_start": 0.0,
        "gamma_end_deg": 0.0,
        "lat_deg": _LAT,
        "lon_deg": _LON,
        "azimuth_deg": _AZ,
    }
    d = thrust_dir_eci(0.0, r, np.zeros(3), guid)
    assert np.isclose(np.linalg.norm(d), 1.0)
