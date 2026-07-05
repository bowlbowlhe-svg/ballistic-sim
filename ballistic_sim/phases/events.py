"""统一事件函数工厂。"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from ballistic_sim.constants import WGS84_A
from ballistic_sim.frames import ecef_to_geodetic, eci_to_ecef


def _set_event_attrs(
    fn: Callable[[float, np.ndarray], float],
    terminal: bool = True,
    direction: int = 0,
) -> Callable[[float, np.ndarray], float]:
    fn.terminal = terminal  # type: ignore[attr-defined]
    fn.direction = direction  # type: ignore[attr-defined]
    return fn


def make_ground_event(
    frame: str = "ECI",
    h_target: float = 0.0,
    theta0: float = 0.0,
    terrain: Optional[Any] = None,
    lat0: float = 0.0,
    lon0: float = 0.0,
) -> Callable[[float, np.ndarray], float]:
    """落地/触地事件。

    Parameters
    ----------
    frame:
        状态所在坐标系；``ECI`` 时通过 ``eci_to_ecef + ecef_to_geodetic`` 求椭球高，
        ``ENU`` 时直接用 ``y[2] - h_target``。
    h_target:
        触发高度 (m)。
    theta0:
        ECI/ECEF 历元夹角 (rad)。
    terrain:
        可选地形模型；提供时落地判据改为 ``alt - h_terrain``。
    lat0, lon0:
        发射点经纬度 (deg)，用于 ENU 地形偏移或 ECI 高程查询辅助。
    """

    def ev(t: float, y: np.ndarray) -> float:
        if frame.upper() == "ENU":
            e = float(y[0])
            n = float(y[1])
            alt = float(y[2])
            if terrain is not None:
                h_terrain = terrain.height_at_enu(e, n, lat0, lon0)
                return alt - h_terrain - h_target
            return alt - h_target
        r_eci = np.asarray(y[0:3], dtype=float)
        r_ecef = eci_to_ecef(r_eci, float(t), theta0)
        lat, lon, alt = ecef_to_geodetic(r_ecef)
        if terrain is not None:
            h_terrain = terrain.height_at(lat, lon)
            return alt - h_terrain - h_target
        return alt - h_target

    return _set_event_attrs(ev, terminal=True, direction=-1)


def make_apogee_event(frame: str = "ECI") -> Callable[[float, np.ndarray], float]:
    """远地点事件：径向速度/竖直速度由正变负。"""

    def ev(t: float, y: np.ndarray) -> float:
        if frame.upper() == "ENU":
            return float(y[5])
        r = np.asarray(y[0:3], dtype=float)
        v = np.asarray(y[3:6], dtype=float)
        return float(np.dot(r, v))

    return _set_event_attrs(ev, terminal=False, direction=-1)


def make_burnout_event(m_dry: float) -> Callable[[float, np.ndarray], float]:
    """推进剂耗尽事件：当前质量降到 ``m_dry`` 时触发。"""

    def ev(t: float, y: np.ndarray) -> float:
        return float(y[6]) - float(m_dry)

    return _set_event_attrs(ev, terminal=True, direction=-1)


def make_stage_separation_event(
    t_sep: Optional[float] = None,
    m_dry: Optional[float] = None,
) -> Callable[[float, np.ndarray], float]:
    """级间分离事件工厂。

    参数
    ----
    t_sep:
        固定分离时刻 (s)。提供时按 ``t_sep - t`` 触发。
    m_dry:
        按质量触发：当状态第 7 维 (质量) 降到 ``m_dry`` 时触发。
        ``t_sep`` 与 ``m_dry`` 同时提供时优先使用 ``m_dry``;
        两者均为 None 时事件恒为正 (不触发)。
    """

    def ev(t: float, y: np.ndarray) -> float:
        if m_dry is not None:
            return float(y[6]) - float(m_dry)
        if t_sep is not None:
            return float(t_sep) - float(t)
        return 1.0

    direction = -1 if m_dry is not None else -1
    return _set_event_attrs(ev, terminal=True, direction=direction)


def make_target_distance_event(
    target_eci: np.ndarray,
    distance_m: float,
) -> Callable[[float, np.ndarray], float]:
    """与目标点距离小于阈值事件（用于导弹命中）。"""
    target = np.asarray(target_eci, dtype=float).reshape(3)
    d2 = float(distance_m) ** 2

    def ev(t: float, y: np.ndarray) -> float:
        r = np.asarray(y[0:3], dtype=float)
        return float(np.dot(r - target, r - target)) - d2

    return _set_event_attrs(ev, terminal=True, direction=-1)


def make_orbit_insertion_event(
    target: dict,
    frame: str = "ECI",
) -> Callable[[float, np.ndarray], float]:
    """轨道插入能量/半长轴达标事件。

    ``target`` 需含 ``peri_km``、``apo_km``，或 ``h_km``、``inc_deg``。
    采用半长轴达标判据，对状态连续、利于 brentq 求根。
    """
    from ballistic_sim.dynamics.common import rv_to_oe

    if "peri_km" in target and "apo_km" in target:
        peri_m = float(target["peri_km"]) * 1e3
        apo_m = float(target["apo_km"]) * 1e3
        a_target = 0.5 * (WGS84_A + peri_m + WGS84_A + apo_m)
    elif "h_km" in target:
        a_target = WGS84_A + float(target["h_km"]) * 1e3
    else:
        raise ValueError("target 需含 peri_km/apo_km 或 h_km")

    def ev(t: float, y: np.ndarray) -> float:
        r = np.asarray(y[0:3], dtype=float)
        v = np.asarray(y[3:6], dtype=float)
        oe = rv_to_oe(r, v)
        return float(oe["a"] - a_target)

    return _set_event_attrs(ev, terminal=True, direction=1)


def make_fairing_event_h(
    h_thresh_m: float,
    frame: str = "ECI",
    theta0: float = 0.0,
) -> Callable[[float, np.ndarray], float]:
    """几何高度上穿阈值事件（抛整流罩，非 terminal）。"""

    def ev(t: float, y: np.ndarray) -> float:
        if frame.upper() == "ENU":
            h = float(y[2])
        else:
            r_ecef = eci_to_ecef(np.asarray(y[0:3], dtype=float), float(t), theta0)
            _, _, h = ecef_to_geodetic(r_ecef)
        return h - float(h_thresh_m)

    return _set_event_attrs(ev, terminal=False, direction=1)


def make_fairing_event_q(
    q_fn: Callable[[float, np.ndarray], float],
    q_thresh_pa: float,
) -> Callable[[float, np.ndarray], float]:
    """动压下穿阈值事件（抛整流罩，非 terminal）。"""

    def ev(t: float, y: np.ndarray) -> float:
        return q_fn(t, y) - float(q_thresh_pa)

    return _set_event_attrs(ev, terminal=False, direction=-1)


def make_fairing_jettison_event(
    mode: str = "altitude",
    h_m: Optional[float] = None,
    q_fn: Optional[Callable[[float, np.ndarray], float]] = None,
    q_thresh_pa: Optional[float] = None,
    frame: str = "ECI",
    theta0: float = 0.0,
) -> Callable[[float, np.ndarray], float]:
    """整流罩抛罩事件工厂。

    Parameters
    ----------
    mode:
        ``"altitude"`` 按几何高度上穿阈值触发;
        ``"q"`` 按动压下穿阈值触发。
    h_m:
        ``mode="altitude"`` 时的高度阈值 (m)。
    q_fn:
        ``mode="q"`` 时的动压函数 ``q(t, y)``。
    q_thresh_pa:
        ``mode="q"`` 时的动压阈值 (Pa)。
    frame, theta0:
        高度模式下的坐标系参数, 同 ``make_fairing_event_h``。
    """
    if mode == "altitude":
        if h_m is None:
            raise ValueError("altitude 模式需要提供 h_m")
        return make_fairing_event_h(float(h_m), frame=frame, theta0=theta0)
    if mode == "q":
        if q_fn is None or q_thresh_pa is None:
            raise ValueError("q 模式需要提供 q_fn 与 q_thresh_pa")
        return make_fairing_event_q(q_fn, float(q_thresh_pa))
    raise ValueError(f"未知的抛罩模式: {mode}")
