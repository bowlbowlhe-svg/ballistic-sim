"""ECI 多级动力上升段运动方程（从 lvsim 迁移）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import numpy as np

from ballistic_sim.constants import DEG2RAD, OMEGA_EARTH, WGS84_A
from ballistic_sim.dynamics.common import (
    AeroEnv,
    DynamicContext,
    compute_alpha,
    dynamic_pressure,
    mach_number,
)
from ballistic_sim.frames import ecef_to_geodetic, eci_to_ecef, enu_basis
from ballistic_sim.models.gravity import gravity_eci
from ballistic_sim.models.propulsion import PropulsionModel


@dataclass
class PoweredECIDynamics:
    """ECI 动力上升段 3-DOF 质点 RHS。

    状态向量 ``[r_eci(3), v_eci(3), m]``，推进期间 ``dm/dt = -mdot``。
    制导律支持三段式开环（垂直起飞 -> 程序俯仰 -> 重力转弯）以及上面级线性压平。
    """

    stage: Dict[str, Any] = field(default_factory=dict)
    guidance: Dict[str, Any] = field(default_factory=dict)
    modes: Dict[str, bool] = field(
        default_factory=lambda: {"drag": True, "j2": True, "thrust": True}
    )
    use_upperstage: bool = False

    def __post_init__(self):
        self.prop = PropulsionModel.from_stage(self.stage)
        self._omega_vec = np.array([0.0, 0.0, OMEGA_EARTH], dtype=float)

    def state_dim(self) -> int:
        return 7

    def native_frame(self) -> str:
        return "ECI"

    def initial_state(
        self,
        v0: float,
        theta_deg: float,
        az_deg: float,
        h0: float = 0.0,
    ) -> np.ndarray:
        """由发射点构建 ECI 初态。"""
        from ballistic_sim.frames import launch_state_eci

        lat = float(self.guidance.get("lat_deg", 0.0))
        lon = float(self.guidance.get("lon_deg", 0.0))
        r0, v0_eci = launch_state_eci(lat, lon, h0)
        m0 = float(self.stage.get("m_dry", 0.0)) + float(self.stage.get("m_prop", 0.0))
        return np.concatenate([r0, v0_eci, [m0]])

    def _altitude(self, r: np.ndarray) -> float:
        r_ecef = eci_to_ecef(r, 0.0)
        _, _, alt = ecef_to_geodetic(r_ecef)
        return float(alt)

    def _rel_velocity(self, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        return v - np.cross(self._omega_vec, r)

    def _safe_normalize(self, vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(vec))
        if n < 1e-12:
            return np.asarray(fallback, dtype=float).copy()
        return vec / n

    def _local_up(self, r: np.ndarray) -> np.ndarray:
        return self._safe_normalize(r, np.array([0.0, 0.0, 1.0]))

    def _downrange_hat(self, t: float) -> np.ndarray:
        az = float(self.guidance.get("azimuth_deg", 0.0)) * DEG2RAD
        lat = float(self.guidance.get("lat_deg", 0.0))
        lon = float(self.guidance.get("lon_deg", 0.0))
        e_hat, n_hat, _ = enu_basis(lat, lon)
        d_ecef = np.sin(az) * e_hat + np.cos(az) * n_hat
        from ballistic_sim.frames import ecef_to_eci

        d_eci = ecef_to_eci(d_ecef, t)
        return self._safe_normalize(d_eci, np.array([1.0, 0.0, 0.0]))

    def _pitch_dir(self, up: np.ndarray, dr: np.ndarray, tilt_deg: float) -> np.ndarray:
        horiz = dr - np.dot(dr, up) * up
        horiz = self._safe_normalize(horiz, dr)
        tilt = float(tilt_deg) * DEG2RAD
        d = np.cos(tilt) * up + np.sin(tilt) * horiz
        return self._safe_normalize(d, up)

    def _flight_path_angle(self, r: np.ndarray, v: np.ndarray) -> float:
        vn = float(np.linalg.norm(v))
        if vn < 1.0:
            return 0.0
        up = self._local_up(r)
        return float(np.arcsin(np.clip(np.dot(v, up) / vn, -1.0, 1.0)))

    def _thrust_dir(self, t: float, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        g = self.guidance
        if self.use_upperstage:
            return self._thrust_dir_upperstage(t, r, v, g)
        t_pitch = float(g.get("t_pitchover", 0.0))
        t_kick_end = float(g.get("t_kick_end", 0.0))
        kick_deg = float(g.get("kick_deg", 0.0))
        up = self._local_up(r)
        dr = self._downrange_hat(t)
        if t < t_pitch:
            return up
        if t_kick_end <= t_pitch or t < t_kick_end:
            frac = (
                0.0
                if t_kick_end <= t_pitch
                else min(max((t - t_pitch) / (t_kick_end - t_pitch), 0.0), 1.0)
            )
            return self._pitch_dir(up, dr, kick_deg * frac)
        v_rel = self._rel_velocity(r, v)
        vn = float(np.linalg.norm(v_rel))
        if vn >= 1.0:
            return self._safe_normalize(v_rel, dr)
        return self._pitch_dir(up, dr, kick_deg)

    def _thrust_dir_upperstage(
        self, t: float, r: np.ndarray, v: np.ndarray, g: Dict[str, Any]
    ) -> np.ndarray:
        up = self._local_up(r)
        t_start = float(g.get("t_us_start", 0.0))
        gamma_end = float(g.get("gamma_end_deg", 0.0)) * DEG2RAD
        gamma0_deg = g.get("gamma0_deg", None)
        if gamma0_deg is None:
            gamma0 = self._flight_path_angle(r, v)
        else:
            gamma0 = float(gamma0_deg) * DEG2RAD
        rate = g.get("pitch_rate_dps", None)
        dur = g.get("t_us_dur", None)
        dt = max(t - t_start, 0.0)
        if rate is not None and float(rate) > 0.0:
            r_rad = float(rate) * DEG2RAD
            gamma_cmd = max(gamma0 - r_rad * dt, gamma_end)
        elif dur is not None and float(dur) > 1e-12:
            frac = min(max(dt / float(dur), 0.0), 1.0)
            gamma_cmd = gamma0 + (gamma_end - gamma0) * frac
        else:
            gamma_cmd = gamma_end
        vn = float(np.linalg.norm(v))
        h_hat = None
        if vn >= 1.0:
            horiz = v - np.dot(v, up) * up
            if float(np.linalg.norm(horiz)) >= 1e-12:
                h_hat = horiz / float(np.linalg.norm(horiz))
        if h_hat is None:
            dr = self._downrange_hat(t)
            horiz = dr - np.dot(dr, up) * up
            h_hat = self._safe_normalize(horiz, np.array([1.0, 0.0, 0.0]))
        d = np.cos(gamma_cmd) * h_hat + np.sin(gamma_cmd) * up
        return self._safe_normalize(d, h_hat)

    def _drag_accel(
        self,
        ctx: DynamicContext,
        v_rel: np.ndarray,
        env: AeroEnv,
        mass: float,
    ) -> np.ndarray:
        if not self.modes.get("drag", True) or env.rho <= 0.0:
            return np.zeros(3, dtype=float)
        vm = float(np.linalg.norm(v_rel))
        if vm <= 0.0:
            return np.zeros(3, dtype=float)
        alpha = 0.0
        cd = 0.3
        if ctx.aero is not None:
            alpha = compute_alpha(v_rel, v_rel)
            cd = float(ctx.aero.drag_coefficient(env.Ma, alpha))
        coeff = -0.5 * env.rho * cd * self.prop.Aref_m2 / mass * vm
        return coeff * v_rel

    def rhs(self, t: float, y: np.ndarray, ctx: Any) -> np.ndarray:
        from ballistic_sim.phases.base import PhaseContext

        if isinstance(ctx, PhaseContext):
            dyn_ctx = getattr(ctx.cfg, "_dynamics_context", None)
        else:
            dyn_ctx = ctx

        y = np.asarray(y, dtype=float)
        r = y[0:3].copy()
        v = y[3:6].copy()
        m = float(y[6])

        h = self._altitude(r)
        atm = dyn_ctx.atmosphere(max(h, 0.0))
        v_rel = self._rel_velocity(r, v)
        vm = float(np.linalg.norm(v_rel))
        Ma = mach_number(vm, atm.c)
        env = AeroEnv(
            rho=atm.rho,
            c=atm.c,
            p=atm.p,
            T=atm.T,
            q=dynamic_pressure(atm.rho, vm),
            Ma=Ma,
        )

        a = np.zeros(3, dtype=float)
        if self.modes.get("j2", True):
            a += gravity_eci(r, model="j2")
        else:
            a += gravity_eci(r, model="point")

        if self.modes.get("thrust", True):
            thrust = self.prop.thrust_at_altitude(h, dyn_ctx.atmosphere)
            dir_hat = self._thrust_dir(t, r, v)
            a += (thrust / m) * dir_hat

        a += self._drag_accel(dyn_ctx, v_rel, env, m)

        return np.concatenate([v, a, [-self.prop.mdot]])

    def telemetry(self, t: float, y: np.ndarray, ctx: Any) -> Dict[str, Any]:
        from ballistic_sim.phases.base import PhaseContext

        if isinstance(ctx, PhaseContext):
            dyn_ctx = getattr(ctx.cfg, "_dynamics_context", None)
        else:
            dyn_ctx = ctx

        y = np.asarray(y, dtype=float)
        r = y[0:3]
        v = y[3:6]
        m = float(y[6])
        h = self._altitude(r)
        atm = dyn_ctx.atmosphere(max(h, 0.0))
        v_rel = self._rel_velocity(r, v)
        vm = float(np.linalg.norm(v_rel))
        Ma = mach_number(vm, atm.c)
        thrust = self.prop.thrust_at_altitude(h, dyn_ctx.atmosphere)
        return {
            "h": h,
            "v_inertial": float(np.linalg.norm(v)),
            "v_rel": vm,
            "q": dynamic_pressure(atm.rho, vm),
            "Ma": Ma,
            "thrust": thrust,
            "accel_g": float(np.linalg.norm(self.rhs(t, y, ctx)[3:6])) / 9.80665,
            "dir_hat": self._thrust_dir(t, r, v),
        }
