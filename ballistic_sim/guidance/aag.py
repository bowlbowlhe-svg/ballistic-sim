"""增广显式制导 (AAG, Augmented Explicit Guidance) 简化实现。

AAG 在 PEG 基础上增加终端位置约束，通过迭代求解待飞时间 t_go 与速度增量，
输出推力方向单位矢量或所需加速度矢量。数值失败时设置 ``failed`` 标志，
由调用方回退到开环/PEG。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ballistic_sim.constants import G0_STANDARD, GM_EARTH, WGS84_A
from ballistic_sim.guidance.open_loop import local_up

__all__ = [
    "AAGState",
    "make_aag_state",
    "aag_acceleration",
    "aag_thrust_dir",
]

_EPS = 1e-12
_TGO_MAX = 5000.0
_ITER_MAX = 30


def _safe_unit(v: Any, fallback: Any) -> np.ndarray:
    """归一化矢量，零模长时回退。"""
    arr = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(arr))
    if n < _EPS:
        return np.asarray(fallback, dtype=float).reshape(3).copy()
    return arr / n


def _terminal_velocity_eci(
    r_eci: Any,
    v_eci: Any,
    terminal: Dict[str, Any],
) -> np.ndarray:
    """由终端约束构造目标速度矢量 (ECI)。"""
    r = np.asarray(r_eci, dtype=float).reshape(3)
    v = np.asarray(v_eci, dtype=float).reshape(3)
    up_hat = local_up(r)

    # 轨道角动量方向作为水平面内横向参考
    h_vec = np.cross(r, v)
    n_hat = _safe_unit(h_vec, np.array([0.0, 0.0, 1.0]))
    h_hat = _safe_unit(
        np.cross(n_hat, up_hat),
        _safe_unit(v - np.dot(v, up_hat) * up_hat, np.array([1.0, 0.0, 0.0])),
    )

    v_T = float(terminal["v_T_ms"])
    gamma_T = float(terminal.get("gamma_T", 0.0))
    return v_T * (np.cos(gamma_T) * h_hat + np.sin(gamma_T) * up_hat)


@dataclass
class AAGState:
    """AAG 重规划状态。"""

    terminal: Dict[str, Any]
    replan_period: float = 2.0
    max_iter: int = _ITER_MAX
    ve_eff: Optional[float] = None
    t_last_plan: Optional[float] = None
    dir_eci: Optional[np.ndarray] = None
    acc_eci: Optional[np.ndarray] = None
    t_go: float = float("nan")
    failed: bool = False
    n_replans: int = 0


def make_aag_state(
    target: Dict[str, Any],
    replan_period: float = 2.0,
    max_iter: int = _ITER_MAX,
    ve_eff: Optional[float] = None,
    mu: float = GM_EARTH,
) -> AAGState:
    """由目标轨道/终端状态构造 AAGState。

    ``target`` 支持两种形式：
    - 轨道形式：``{"peri_km": ..., "apo_km": ..., "inc_deg": ...}``
    - 终端状态形式：``{"r_T_m": ..., "v_T_ms": ..., "gamma_T": ...}``
    """
    if "r_T_m" in target and "v_T_ms" in target:
        terminal = {
            "r_T_m": float(target["r_T_m"]),
            "v_T_ms": float(target["v_T_ms"]),
            "gamma_T": float(target.get("gamma_T", 0.0)),
        }
    else:
        peri_km = float(target["peri_km"])
        apo_km = float(target.get("apo_km", peri_km))
        r_T = WGS84_A + peri_km * 1e3
        a_T = WGS84_A + 0.5 * (peri_km + apo_km) * 1e3
        v2 = mu * (2.0 / r_T - 1.0 / a_T)
        terminal = {
            "r_T_m": float(r_T),
            "v_T_ms": float(np.sqrt(max(v2, 0.0))),
            "gamma_T": 0.0,
        }
    return AAGState(terminal, replan_period=replan_period, max_iter=max_iter, ve_eff=ve_eff)


def _effective_exhaust_velocity(stage: Dict[str, Any]) -> float:
    isp = float(stage.get("isp_vac", 0.0))
    return isp * G0_STANDARD if isp > 0.0 else 0.0


def _estimate_tgo(vgo_mag: float, m_now: float, thrust: float, ve_eff: float) -> float:
    """由火箭方程估算待飞时间。"""
    if not (np.isfinite(vgo_mag) and np.isfinite(m_now) and np.isfinite(thrust)
            and np.isfinite(ve_eff)):
        return float("nan")
    if thrust <= 0.0 or ve_eff <= 0.0 or m_now <= 0.0:
        return float("nan")
    mdot = thrust / ve_eff
    if mdot <= 0.0:
        return float("nan")
    m_f = m_now * np.exp(-vgo_mag / ve_eff)
    return float((m_now - m_f) / mdot)


def _target_position_eci(r_now: np.ndarray, terminal: Dict[str, Any]) -> np.ndarray:
    """由终端半径与当前位置方向估算终端位置矢量。"""
    r_T_m = float(terminal["r_T_m"])
    up_now = _safe_unit(r_now, np.array([0.0, 0.0, 1.0]))
    return r_T_m * up_now


def aag_replan(
    state: AAGState,
    t_abs: float,
    r_eci: Any,
    v_eci: Any,
    m_now: float,
    stage: Dict[str, Any],
) -> bool:
    """AAG 重规划：迭代求解待增速度方向与待飞时间。

    算法思路
    --------
    1. 用终端半径/速度/弹道倾角构造目标速度矢量 ``v_T``。
    2. 估算终端位置 ``r_T``（保持当前地心方向）。
    3. 通过平均重力积分修正速度增量，迭代收敛 ``t_go``。
    4. 推力方向取所需速度增量方向。
    """
    r = np.asarray(r_eci, dtype=float).reshape(3)
    v = np.asarray(v_eci, dtype=float).reshape(3)
    mu = GM_EARTH

    thrust = float(stage.get("thrust_vac", 0.0))
    ve_eff = state.ve_eff if state.ve_eff is not None else _effective_exhaust_velocity(stage)
    terminal = state.terminal

    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v)) and np.isfinite(m_now)
            and thrust > 0.0 and ve_eff > 0.0):
        state.failed = True
        return False

    r_mag = float(np.linalg.norm(r))
    if r_mag < _EPS:
        state.failed = True
        return False

    v_target = _terminal_velocity_eci(r, v, terminal)
    r_target = _target_position_eci(r, terminal)
    g0 = mu / max(r_mag * r_mag, 1.0)

    # 初始速度增量
    vgo = v_target - v
    t_go = float("nan")
    ok = False

    for _ in range(max(state.max_iter, 1)):
        vgo_mag = float(np.linalg.norm(vgo))
        t_go = _estimate_tgo(vgo_mag, m_now, thrust, ve_eff)
        if not np.isfinite(t_go) or t_go <= 0.0 or t_go > _TGO_MAX:
            break

        # 平均重力损失：g0 沿平均上方向积分 t_go
        up_avg = _safe_unit(r + r_target, local_up(r))
        v_grav = g0 * t_go * up_avg

        # 平均速度方向作为地心转动近似
        v_avg = 0.5 * (v + v_target)
        omega = v_avg / max(r_mag, 1.0)
        dtheta = float(np.clip(np.linalg.norm(omega) * t_go, -np.pi, np.pi))
        side = _safe_unit(np.cross(local_up(r), v_avg), v_avg)
        r_T_rot = (
            np.cos(dtheta) * r_target
            + np.sin(dtheta) * side * np.linalg.norm(r_target)
        )
        up_T = local_up(r_T_rot)

        # 重新构造终端速度方向
        h_vec = np.cross(r_T_rot, v)
        n_hat = _safe_unit(h_vec, np.array([0.0, 0.0, 1.0]))
        h_T = _safe_unit(
            np.cross(n_hat, up_T),
            _safe_unit(v - np.dot(v, up_T) * up_T, np.array([1.0, 0.0, 0.0])),
        )
        gamma_T = float(terminal.get("gamma_T", 0.0))
        v_T_rot = float(terminal["v_T_ms"]) * (np.cos(gamma_T) * h_T + np.sin(gamma_T) * up_T)

        vgo_new = (v_T_rot - v) + v_grav
        if not np.all(np.isfinite(vgo_new)):
            break
        # 阻尼迭代
        vgo_new = 0.5 * vgo + 0.5 * vgo_new
        if float(np.linalg.norm(vgo_new - vgo)) < 1.0:
            vgo = vgo_new
            ok = True
            break
        vgo = vgo_new

    vgo_mag = float(np.linalg.norm(vgo))
    t_go = _estimate_tgo(vgo_mag, m_now, thrust, ve_eff)

    if not ok and not (np.isfinite(vgo_mag) and np.isfinite(t_go) and 0.0 < t_go <= _TGO_MAX):
        state.failed = True
        return False

    state.failed = False
    state.dir_eci = _safe_unit(vgo, local_up(r))
    state.acc_eci = vgo  # 原始速度增量方向，由调用方决定是否作为加速度
    state.t_go = t_go if np.isfinite(t_go) else 0.0
    state.t_last_plan = float(t_abs)
    state.n_replans += 1
    return True


def aag_thrust_dir(
    t_abs: float,
    r_eci: Any,
    v_eci: Any,
    m_now: float,
    stage: Dict[str, Any],
    state: AAGState,
) -> Optional[np.ndarray]:
    """AAG 逐步推力方向入口。"""
    need = state.t_last_plan is None or (t_abs - state.t_last_plan) >= state.replan_period
    if need:
        aag_replan(state, t_abs, r_eci, v_eci, m_now, stage)
    if state.failed or state.dir_eci is None:
        return None
    return state.dir_eci.copy()


def aag_acceleration(
    t_abs: float,
    r_eci: Any,
    v_eci: Any,
    m_now: float,
    stage: Dict[str, Any],
    state: AAGState,
) -> Optional[np.ndarray]:
    """AAG 所需加速度矢量入口（含推力大小归一化）。"""
    d = aag_thrust_dir(t_abs, r_eci, v_eci, m_now, stage, state)
    if d is None:
        return None
    thrust = float(stage.get("thrust_vac", 0.0))
    if m_now > 0.0:
        return (thrust / m_now) * d
    return d * 0.0
