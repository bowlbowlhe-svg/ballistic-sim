"""PEG (Powered Explicit Guidance) 真空段闭环制导简化实现。

核心逻辑：velocity-to-be-gained 迭代 + 周期重规划；数值失败时返回 None，
由调用方回退到开环上面级律。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

from ballistic_sim.constants import G0_STANDARD, GM_EARTH, WGS84_A

__all__ = [
    "PegState",
    "make_peg_state",
    "peg_thrust_dir",
    "make_target_terminal",
    "make_orbit_reached_event",
    "make_apogee_circularization_event",
]

_EPS = 1e-12
_VGO_CUTOFF = 30.0
_TGO_MAX = 5000.0
_GAMMA_GUARD = np.radians(0.5)
_CIRC_VGO_TOL = 1.0
_ENTRY_REACHED_DT = 1e-3


def _safe_unit(v, fallback):
    arr = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(arr))
    if n < _EPS:
        return np.asarray(fallback, dtype=float).reshape(3).copy()
    return arr / n


def make_target_terminal(target: Dict[str, Any], mu: float = GM_EARTH) -> Dict[str, Any]:
    """由目标轨道构造终端约束 (r_T, v_T, gamma_T=0)。"""
    peri_km = float(target["peri_km"])
    apo_km = float(target.get("apo_km", peri_km))
    inc = target.get("inc_deg", None)
    r_T = WGS84_A + peri_km * 1e3
    a_T = WGS84_A + 0.5 * (peri_km + apo_km) * 1e3
    v2 = mu * (2.0 / r_T - 1.0 / a_T)
    v_T = float(np.sqrt(max(v2, 0.0)))
    return {
        "r_T_m": float(r_T),
        "v_T_ms": v_T,
        "gamma_T": 0.0,
        "inc_deg": None if inc is None else float(inc),
        "a_T_m": float(a_T),
    }


@dataclass
class PegState:
    """PEG 重规划状态机。"""

    target_terminal: Dict[str, Any]
    replan_period: float = 2.0
    max_iter: int = 20
    ve_eff: Optional[float] = None
    peg_mode: str = "periapsis"
    r_apo_circ_m: Optional[float] = None
    t_last_plan: Optional[float] = None
    dir_eci: Optional[np.ndarray] = None
    vgo_dir_eci: Optional[np.ndarray] = None
    vgo_mag: float = float("inf")
    vgo_pure: float = float("inf")
    t_go: float = float("nan")
    failed: bool = False
    n_replans: int = 0


def make_peg_state(
    target: Dict[str, Any],
    replan_period: float = 2.0,
    max_iter: int = 20,
    ve_eff: Optional[float] = None,
    mu: float = GM_EARTH,
    peg_mode: str = "periapsis",
    r_apo_circ_m: Optional[float] = None,
) -> PegState:
    """工厂：由目标轨道构造 PegState。"""
    if peg_mode == "apogee_circ":
        r_apo = r_apo_circ_m or WGS84_A + float(target.get("apo_km", target["peri_km"])) * 1e3
        tt = {
            "r_T_m": float(r_apo),
            "v_T_ms": float(np.sqrt(mu / r_apo)),
            "gamma_T": 0.0,
            "inc_deg": target.get("inc_deg", None),
            "a_T_m": float(r_apo),
        }
        return PegState(
            tt,
            replan_period=replan_period,
            max_iter=max_iter,
            ve_eff=ve_eff,
            peg_mode="apogee_circ",
            r_apo_circ_m=float(r_apo),
        )
    return PegState(
        make_target_terminal(target, mu=mu),
        replan_period=replan_period,
        max_iter=max_iter,
        ve_eff=ve_eff,
        peg_mode="periapsis",
    )


def _effective_exhaust_velocity(stage: Dict[str, Any]) -> float:
    isp = float(stage.get("isp_vac", 0.0))
    return isp * G0_STANDARD if isp > 0.0 else 0.0


def _estimate_tgo(vgo_mag: float, m_now: float, thrust: float, ve_eff: float) -> float:
    if not (
        np.isfinite(vgo_mag) and np.isfinite(m_now) and np.isfinite(thrust) and np.isfinite(ve_eff)
    ):
        return float("nan")
    if thrust <= 0.0 or ve_eff <= 0.0 or m_now <= 0.0:
        return float("nan")
    mdot = thrust / ve_eff
    if mdot <= 0.0:
        return float("nan")
    m_f = m_now * np.exp(-vgo_mag / ve_eff)
    return float((m_now - m_f) / mdot)


def _terminal_velocity_eci(r_eci, v_eci, tt):
    r = np.asarray(r_eci, dtype=float).reshape(3)
    v = np.asarray(v_eci, dtype=float).reshape(3)
    up_hat = _safe_unit(r, np.array([0.0, 0.0, 1.0]))
    h_vec = np.cross(r, v)
    n_hat = _safe_unit(h_vec, np.array([0.0, 0.0, 1.0]))
    h_hat = _safe_unit(
        np.cross(n_hat, up_hat),
        _safe_unit(v - np.dot(v, up_hat) * up_hat, np.array([1.0, 0.0, 0.0])),
    )
    v_T = float(tt["v_T_ms"])
    gamma_T = float(tt["gamma_T"])
    v_target = v_T * (np.cos(gamma_T) * h_hat + np.sin(gamma_T) * up_hat)
    return v_target, up_hat, h_hat


def replan(
    state: PegState,
    t_abs: float,
    r_eci,
    v_eci,
    m_now: float,
    stage: Dict[str, Any],
) -> bool:
    """重规划：解 v_go 与待飞时间，更新 state。"""
    r = np.asarray(r_eci, dtype=float).reshape(3)
    v = np.asarray(v_eci, dtype=float).reshape(3)
    mu = GM_EARTH

    thrust = float(stage.get("thrust_vac", 0.0))
    ve_eff = state.ve_eff if state.ve_eff is not None else _effective_exhaust_velocity(stage)
    tt = state.target_terminal

    if not (
        np.all(np.isfinite(r))
        and np.all(np.isfinite(v))
        and np.isfinite(m_now)
        and thrust > 0.0
        and ve_eff > 0.0
    ):
        state.failed = True
        return False

    v_target, up_hat, h_hat = _terminal_velocity_eci(r, v, tt)
    n_orb = np.cross(r, v)
    n_hat = _safe_unit(n_orb, np.array([0.0, 0.0, 1.0]))
    r_mag0 = float(np.linalg.norm(r))
    g0 = mu / max(r_mag0 * r_mag0, 1.0)
    v_T = float(tt["v_T_ms"])

    vgo = v_target - v
    ok = False
    t_go = float("nan")
    for _ in range(max(state.max_iter, 1)):
        vgo_mag = float(np.linalg.norm(vgo))
        t_go = _estimate_tgo(vgo_mag, m_now, thrust, ve_eff)
        if not np.isfinite(t_go) or t_go <= 0.0 or t_go > _TGO_MAX:
            if np.isfinite(vgo_mag) and vgo_mag < _VGO_CUTOFF:
                ok = True
            break
        vh = float(np.dot(v, h_hat))
        omega = (0.5 * (vh + v_T)) / max(r_mag0, 1.0)
        dtheta = float(np.clip(omega * t_go, -np.pi, np.pi))
        side = _safe_unit(np.cross(n_hat, up_hat), h_hat)
        up_T = _safe_unit(np.cos(dtheta) * up_hat + np.sin(dtheta) * side, up_hat)
        h_T = _safe_unit(np.cross(n_hat, up_T), h_hat)
        gamma_T = float(tt["gamma_T"])
        v_target = v_T * (np.cos(gamma_T) * h_T + np.sin(gamma_T) * up_T)
        up_avg = _safe_unit(up_hat + up_T, up_hat)
        v_grav_mag = g0 * t_go
        vgo_new = (v_target - v) + v_grav_mag * up_avg
        if not np.all(np.isfinite(vgo_new)):
            break
        vgo_new = 0.5 * vgo + 0.5 * vgo_new
        if float(np.linalg.norm(vgo_new - vgo)) < 1.0:
            vgo = vgo_new
            ok = True
            break
        vgo = vgo_new

    vgo_mag = float(np.linalg.norm(vgo))
    t_go = _estimate_tgo(vgo_mag, m_now, thrust, ve_eff)

    if not ok and not (
        np.isfinite(vgo_mag)
        and (vgo_mag < _VGO_CUTOFF or (np.isfinite(t_go) and 0.0 < t_go <= _TGO_MAX))
    ):
        state.failed = True
        return False

    state.failed = False
    state.vgo_dir_eci = _safe_unit(vgo, h_hat)
    state.vgo_mag = vgo_mag
    state.vgo_pure = float(np.linalg.norm(v_T * h_hat - v))
    state.t_go = t_go if np.isfinite(t_go) else 0.0
    state.dir_eci = state.vgo_dir_eci.copy()
    state.t_last_plan = float(t_abs)
    state.n_replans += 1
    return True


def replan_apogee_circularization(
    state: PegState,
    t_abs: float,
    r_eci,
    v_eci,
    m_now: float,
    stage: Dict[str, Any],
) -> bool:
    """远地点圆化重规划：切向待增速度。"""
    r = np.asarray(r_eci, dtype=float).reshape(3)
    v = np.asarray(v_eci, dtype=float).reshape(3)
    mu = GM_EARTH

    thrust = float(stage.get("thrust_vac", 0.0))
    ve_eff = state.ve_eff if state.ve_eff is not None else _effective_exhaust_velocity(stage)

    r_mag = float(np.linalg.norm(r))
    if not (
        np.all(np.isfinite(r))
        and np.all(np.isfinite(v))
        and np.isfinite(m_now)
        and thrust > 0.0
        and ve_eff > 0.0
        and r_mag > _EPS
    ):
        state.failed = True
        return False

    up = r / r_mag
    h_vec = v - np.dot(v, up) * up
    h_mag = float(np.linalg.norm(h_vec))
    if h_mag < _EPS:
        state.failed = True
        return False
    h_hat = h_vec / h_mag

    v_circ = float(np.sqrt(mu / r_mag))
    v_target = v_circ * h_hat
    vgo = v_target - v
    vgo_mag = float(np.linalg.norm(vgo))
    t_go = _estimate_tgo(vgo_mag, m_now, thrust, ve_eff)
    if not (np.isfinite(vgo_mag) and np.isfinite(t_go) and 0.0 < t_go <= _TGO_MAX):
        if np.isfinite(vgo_mag) and vgo_mag < _VGO_CUTOFF:
            state.failed = False
            state.vgo_dir_eci = _safe_unit(vgo, h_hat)
            state.vgo_mag = vgo_mag
            state.vgo_pure = vgo_mag
            state.t_go = 0.0
            state.dir_eci = state.vgo_dir_eci.copy()
            state.t_last_plan = float(t_abs)
            state.n_replans += 1
            return True
        state.failed = True
        return False

    state.failed = False
    state.vgo_dir_eci = _safe_unit(vgo, h_hat)
    state.vgo_mag = vgo_mag
    state.vgo_pure = vgo_mag
    state.t_go = t_go
    state.dir_eci = state.vgo_dir_eci.copy()
    state.t_last_plan = float(t_abs)
    state.n_replans += 1
    return True


def peg_thrust_dir(
    t_abs: float,
    r_eci,
    v_eci,
    m_now: float,
    stage: Dict[str, Any],
    state: PegState,
):
    """PEG 逐步推力方向。"""
    need = state.t_last_plan is None or (t_abs - state.t_last_plan) >= state.replan_period
    if need:
        if getattr(state, "peg_mode", "periapsis") == "apogee_circ":
            replan_apogee_circularization(state, t_abs, r_eci, v_eci, m_now, stage)
        else:
            replan(state, t_abs, r_eci, v_eci, m_now, stage)
    if state.failed or state.dir_eci is None:
        return None
    return state.dir_eci.copy()


def make_orbit_reached_event(target: Dict[str, Any], mu: float = GM_EARTH) -> Callable:
    """目标轨道能量达标关机事件。"""
    tt = make_target_terminal(target, mu=mu)
    a_T = tt["a_T_m"]
    energy_target = -mu / (2.0 * a_T)
    ev_state: Dict[str, Any] = {"t0": None, "entry_reached": False}

    def ev_orbit_reached(t: float, Y: np.ndarray) -> float:
        r = np.asarray(Y[0:3], dtype=float)
        v = np.asarray(Y[3:6], dtype=float)
        r_mag = float(np.linalg.norm(r))
        v_mag = float(np.linalg.norm(v))
        if r_mag < _EPS:
            return 1.0
        energy_now = 0.5 * v_mag * v_mag - mu / r_mag
        g = float(energy_target - energy_now)
        if ev_state["t0"] is None:
            ev_state["t0"] = float(t)
            ev_state["entry_reached"] = g <= 0.0
        if ev_state["entry_reached"]:
            return (ev_state["t0"] + _ENTRY_REACHED_DT) - float(t)
        return g

    ev_orbit_reached.terminal = True  # type: ignore[attr-defined]
    ev_orbit_reached.direction = -1  # type: ignore[attr-defined]
    return ev_orbit_reached


def make_apogee_circularization_event(r_apo_m: float, mu: float = GM_EARTH) -> Callable:
    """远地点圆化 γ 过零关机事件（含水平速度门控）。"""
    float(r_apo_m)
    ev_state: Dict[str, Any] = {"seen": False, "t_cut": None}

    def ev_apo_circ(t: float, Y: np.ndarray) -> float:
        if ev_state["t_cut"] is not None:
            return (ev_state["t_cut"] + _ENTRY_REACHED_DT) - float(t)
        r = np.asarray(Y[0:3], dtype=float)
        v = np.asarray(Y[3:6], dtype=float)
        r_mag = float(np.linalg.norm(r))
        v_mag = float(np.linalg.norm(v))
        if r_mag < _EPS or v_mag < _EPS:
            return 1.0
        up_hat = r / r_mag
        gamma = float(np.arcsin(np.clip(np.dot(v, up_hat) / v_mag, -1.0, 1.0)))
        v_circ_now = float(np.sqrt(mu / r_mag))
        h_vec = v - np.dot(v, up_hat) * up_hat
        v_horiz = float(np.linalg.norm(h_vec))
        near_gamma = abs(gamma) < _GAMMA_GUARD
        deficit = v_circ_now - v_horiz
        if near_gamma and deficit > _CIRC_VGO_TOL:
            ev_state["seen"] = True
            return 1.0
        first = not ev_state["seen"]
        ev_state["seen"] = True
        if near_gamma and deficit <= _CIRC_VGO_TOL:
            ev_state["t_cut"] = float(t)
            return _ENTRY_REACHED_DT
        if first and gamma <= 0.0:
            ev_state["t_cut"] = float(t)
            return _ENTRY_REACHED_DT
        return gamma

    ev_apo_circ.terminal = True  # type: ignore[attr-defined]
    ev_apo_circ.direction = -1  # type: ignore[attr-defined]
    return ev_apo_circ
