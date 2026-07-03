"""三段式开环制导与上面级线性俯仰律 (从 lvsim 迁移)。

物理模型
--------
1. 垂直起飞段 ``t < t_pitchover``：推力沿当地上方向。
2. 程序俯仰段 ``t_pitchover <= t < t_kick_end``：向下程方向线性倾斜到 ``kick_deg``。
3. 重力转弯段 ``t >= t_kick_end``：推力沿相对速度方向。

上面级 (``phase='upperstage'``)：指令飞行路径角从 ``gamma0`` 线性压到
``gamma_end`` (≈0)，使速度逐渐水平化。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ballistic_sim.constants import DEG2RAD, OMEGA_EARTH
from ballistic_sim.frames import ecef_to_eci, enu_basis

__all__ = [
    "thrust_dir_eci",
    "thrust_dir_upperstage",
    "local_up",
    "local_horizontal",
    "flight_path_angle",
]

_VREL_MIN = 1.0
_EPS = 1e-12
_OMEGA_VEC = np.array([0.0, 0.0, OMEGA_EARTH], dtype=float)


def _safe_normalize(v, fallback):
    """归一化矢量，零模长时返回 fallback。"""
    arr = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(arr))
    if n < _EPS:
        return np.asarray(fallback, dtype=float).reshape(3).copy()
    return arr / n


def local_up(r_eci):
    """ECI 下当地上方向 (位置矢量单位化)。"""
    return _safe_normalize(r_eci, np.array([0.0, 0.0, 1.0], dtype=float))


def _downrange_hat_eci(guid: Dict[str, Any], t_now_for_frame: float) -> np.ndarray:
    """发射方位角确定的下程水平方向 (ECI)。"""
    az = float(guid["azimuth_deg"]) * DEG2RAD
    lat = float(guid["lat_deg"])
    lon = float(guid["lon_deg"])
    e_hat, n_hat, _u_hat = enu_basis(lat, lon)
    d_ecef = np.sin(az) * e_hat + np.cos(az) * n_hat
    d_eci = ecef_to_eci(d_ecef, t_now_for_frame)
    return _safe_normalize(d_eci, np.array([1.0, 0.0, 0.0], dtype=float))


def _pitch_dir_eci(up_hat: np.ndarray, dr_hat: np.ndarray, tilt_deg: float) -> np.ndarray:
    """由 up 与下程方向合成倾斜 tilt_deg 后的推力方向。"""
    horiz = dr_hat - np.dot(dr_hat, up_hat) * up_hat
    horiz = _safe_normalize(horiz, dr_hat)
    tilt = float(tilt_deg) * DEG2RAD
    d = np.cos(tilt) * up_hat + np.sin(tilt) * horiz
    return _safe_normalize(d, up_hat)


def local_horizontal(r_eci, ref_dir):
    """返回包含 ref_dir 的竖直平面内的当地水平单位矢量。"""
    r = np.asarray(r_eci, dtype=float).reshape(3)
    ref = np.asarray(ref_dir, dtype=float).reshape(3)
    up_hat = local_up(r)
    horiz = ref - np.dot(ref, up_hat) * up_hat
    return _safe_normalize(horiz, np.array([1.0, 0.0, 0.0], dtype=float))


def flight_path_angle(r_eci, v_eci) -> float:
    """惯性飞行路径角 γ：速度与当地水平面夹角 (rad)，向上为正。"""
    r = np.asarray(r_eci, dtype=float).reshape(3)
    v = np.asarray(v_eci, dtype=float).reshape(3)
    vn = float(np.linalg.norm(v))
    if vn < _VREL_MIN:
        return 0.0
    up_hat = local_up(r)
    s = float(np.dot(v, up_hat)) / vn
    return float(np.arcsin(np.clip(s, -1.0, 1.0)))


def thrust_dir_upperstage(
    t: float,
    r_eci,
    v_eci,
    guid_upper: Dict[str, Any],
    t_now_for_frame: Optional[float] = None,
) -> np.ndarray:
    """上面级线性俯仰律：把推力方向从当前 γ 线性压到 gamma_end。"""
    if t_now_for_frame is None:
        t_now_for_frame = t

    r = np.asarray(r_eci, dtype=float).reshape(3)
    v = np.asarray(v_eci, dtype=float).reshape(3)
    up_hat = local_up(r)

    t_start = float(guid_upper["t_us_start"])
    gamma_end = float(guid_upper["gamma_end_deg"]) * DEG2RAD

    g0_param = guid_upper.get("gamma0_deg", None)
    if g0_param is not None:
        gamma0 = float(g0_param) * DEG2RAD
    else:
        gamma0 = flight_path_angle(r, v)

    dt = max(t - t_start, 0.0)
    rate = guid_upper.get("pitch_rate_dps", None)
    dur = guid_upper.get("t_us_dur", None)

    if rate is not None and float(rate) > 0.0:
        r_rad = float(rate) * DEG2RAD
        if gamma_end <= gamma0:
            gamma_cmd = max(gamma0 - r_rad * dt, gamma_end)
        else:
            gamma_cmd = min(gamma0 + r_rad * dt, gamma_end)
    elif dur is not None and float(dur) > _EPS:
        frac = min(max(dt / float(dur), 0.0), 1.0)
        gamma_cmd = gamma0 + (gamma_end - gamma0) * frac
    else:
        gamma_cmd = gamma_end

    vn = float(np.linalg.norm(v))
    h_hat = None
    if vn >= _VREL_MIN:
        horiz = v - np.dot(v, up_hat) * up_hat
        if float(np.linalg.norm(horiz)) >= _EPS:
            h_hat = horiz / float(np.linalg.norm(horiz))
    if h_hat is None:
        if all(k in guid_upper for k in ("lat_deg", "lon_deg", "azimuth_deg")):
            dr_hat = _downrange_hat_eci(guid_upper, t_now_for_frame)
            h_hat = local_horizontal(r, dr_hat)
        else:
            h_hat = local_horizontal(r, np.array([1.0, 0.0, 0.0]))

    d = np.cos(gamma_cmd) * h_hat + np.sin(gamma_cmd) * up_hat
    return _safe_normalize(d, h_hat)


def _thrust_dir_peg(
    t: float,
    r_eci,
    v_eci,
    guid: Dict[str, Any],
    t_now_for_frame: float,
    m: Optional[float],
):
    """PEG 闭环制导转发；数值失败时回退上面级开环律。"""
    from ballistic_sim.guidance import peg as _peg

    state = guid.get("_peg_state", None)
    stage = guid.get("_peg_stage", None)
    if state is None or stage is None:
        return thrust_dir_upperstage(t, r_eci, v_eci, guid, t_now_for_frame)

    m_now = m if m is not None else float(stage.get("m_start", stage.get("m_dry", 1.0)))
    d = _peg.peg_thrust_dir(t, r_eci, v_eci, m_now, stage, state)
    if d is None:
        return thrust_dir_upperstage(t, r_eci, v_eci, guid, t_now_for_frame)
    return _safe_normalize(d, local_up(r_eci))


def thrust_dir_eci(
    t: float,
    r_eci,
    v_eci,
    guid: Dict[str, Any],
    t_now_for_frame: Optional[float] = None,
    m: Optional[float] = None,
) -> np.ndarray:
    """统一开环制导入口。"""
    if t_now_for_frame is None:
        t_now_for_frame = t

    phase = guid.get("phase", "ascent")
    if phase == "peg":
        return _thrust_dir_peg(t, r_eci, v_eci, guid, t_now_for_frame, m)
    if phase == "upperstage":
        return thrust_dir_upperstage(t, r_eci, v_eci, guid, t_now_for_frame)

    r = np.asarray(r_eci, dtype=float).reshape(3)
    v = np.asarray(v_eci, dtype=float).reshape(3)

    t_po = float(guid["t_pitchover"])
    t_ke = float(guid["t_kick_end"])
    kick = float(guid["kick_deg"])

    up_hat = local_up(r)

    if t < t_po:
        return up_hat

    if t >= t_ke:
        v_rel = v - np.cross(_OMEGA_VEC, r)
        if float(np.linalg.norm(v_rel)) >= _VREL_MIN:
            return _safe_normalize(v_rel, up_hat)
        dr_hat = _downrange_hat_eci(guid, t_now_for_frame)
        return _pitch_dir_eci(up_hat, dr_hat, kick)

    dr_hat = _downrange_hat_eci(guid, t_now_for_frame)
    span = t_ke - t_po
    if span <= _EPS:
        frac = 1.0
    else:
        frac = (t - t_po) / span
        frac = min(max(frac, 0.0), 1.0)
    return _pitch_dir_eci(up_hat, dr_hat, kick * frac)
