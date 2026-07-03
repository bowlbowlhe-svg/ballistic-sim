"""3-DOF 质点模型（ECI/ENU）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal

import numpy as np

from ballistic_sim.constants import OMEGA_EARTH
from ballistic_sim.dynamics.common import (
    AeroEnv,
    DynamicContext,
    compute_alpha,
    dynamic_pressure,
    mach_number,
    relative_velocity_eci,
)
from ballistic_sim.frames import (
    ecef_to_geodetic,
    eci_to_ecef,
)
from ballistic_sim.models.gravity import gravity_eci, gravity_enu


@dataclass
class PointMassDynamics:
    """3-DOF 质点运动方程。

    支持 ``ECI``（域 II）与 ``ENU``（域 I）两种原生坐标系，由 ``frame`` 指定。
    状态向量 ``[r(3), v(3), m]``（质量对无动力段恒定）。
    """

    frame: Literal["ECI", "ENU"] = "ECI"
    mass: float = 1.0
    Aref: float = 0.1
    options: Dict[str, bool] = field(default_factory=lambda: {"drag": True, "j2": True})

    def _aero_env(
        self,
        ctx: DynamicContext,
        r: np.ndarray,
        v: np.ndarray,
        lat0: float = 0.0,
        lon0: float = 0.0,
    ) -> AeroEnv:
        """计算当前气动环境。"""
        if self.frame == "ENU":
            h = float(r[2])
            v_rel = v.copy()
        else:
            r_ecef = eci_to_ecef(r, 0.0)
            _, _, h = ecef_to_geodetic(r_ecef)
            w_enu = ctx.wind(h) if ctx.wind is not None else np.zeros(3)
            v_rel = relative_velocity_eci(r, v, w_enu, lat0, lon0)

        atm = ctx.atmosphere(max(h, 0.0))
        vm = float(np.linalg.norm(v_rel))
        Ma = mach_number(vm, atm.c)
        return AeroEnv(
            rho=atm.rho,
            c=atm.c,
            p=atm.p,
            T=atm.T,
            q=dynamic_pressure(atm.rho, vm),
            Ma=Ma,
        )

    def _drag_accel(
        self,
        ctx: DynamicContext,
        v_rel: np.ndarray,
        env: AeroEnv,
        mass: float,
    ) -> np.ndarray:
        """气动阻力加速度。"""
        if not self.options.get("drag", True) or env.rho <= 0.0:
            return np.zeros(3, dtype=float)
        vm = float(np.linalg.norm(v_rel))
        if vm <= 0.0:
            return np.zeros(3, dtype=float)
        alpha = 0.0
        if ctx.aero is not None:
            alpha = compute_alpha(v_rel, v_rel)
            cd = float(ctx.aero.drag_coefficient(env.Ma, alpha))
        else:
            cd = 0.3
        coeff = -0.5 * env.rho * cd * self.Aref / mass * vm
        return coeff * v_rel

    def rhs(self, t: float, y: np.ndarray, ctx: Any) -> np.ndarray:
        """RHS 入口，兼容 ``PhaseContext`` / ``DynamicContext``。"""
        from ballistic_sim.phases.base import PhaseContext

        if isinstance(ctx, PhaseContext):
            dyn_ctx = getattr(ctx.cfg, "_dynamics_context", None)
            cfg = ctx.cfg
        else:
            dyn_ctx = ctx
            cfg = None
        if dyn_ctx is None:
            raise RuntimeError("PointMassDynamics requires a DynamicContext")

        y = np.asarray(y, dtype=float)
        r = y[0:3].copy()
        v = y[3:6].copy()
        m = float(y[6]) if y.size > 6 else self.mass

        lat0 = lon0 = 0.0
        if cfg is not None:
            lat0 = float(cfg.launch.lat_deg)
            lon0 = float(cfg.launch.lon_deg)

        env = self._aero_env(dyn_ctx, r, v, lat0, lon0)

        if self.frame == "ENU":
            a = gravity_enu(lat0, r[2], model="wgs84")
            v_rel = v.copy()
            if dyn_ctx.wind is not None:
                w = dyn_ctx.wind(r[2])
                v_rel -= w.vector
            a += self._drag_accel(dyn_ctx, v_rel, env, m)
            if self.options.get("coriolis", False):
                lat_rad = np.deg2rad(lat0)
                Omega_n = OMEGA_EARTH * np.cos(lat_rad)
                Omega_u = OMEGA_EARTH * np.sin(lat_rad)
                a[0] += -2.0 * (Omega_n * v[2] - Omega_u * v[1])
                a[1] += -2.0 * (Omega_u * v[0])
                a[2] += 2.0 * (Omega_n * v[0])
        else:
            gm = gravity_eci(r, model="j2" if self.options.get("j2", True) else "point")
            a = gm.copy()
            w_enu = dyn_ctx.wind(0.0) if dyn_ctx.wind is not None else None
            wind_vec = None
            if w_enu is not None and np.linalg.norm(w_enu.vector) > 0.0:
                wind_vec = w_enu.vector
            v_rel = relative_velocity_eci(r, v, wind_vec, lat0, lon0)
            a += self._drag_accel(dyn_ctx, v_rel, env, m)

        return np.concatenate([v, a, [0.0]])

    def state_dim(self) -> int:
        """状态向量维度：``[r(3), v(3), m]``。"""
        return 7

    def native_frame(self) -> str:
        """原生坐标系。"""
        return self.frame

    def telemetry(self, t: float, y: np.ndarray, ctx: Any) -> Dict[str, Any]:
        """返回当前时刻派生量字典。"""
        from ballistic_sim.phases.base import PhaseContext

        if isinstance(ctx, PhaseContext):
            dyn_ctx = getattr(ctx.cfg, "_dynamics_context", None)
            cfg = ctx.cfg
        else:
            dyn_ctx = ctx
            cfg = None
        if dyn_ctx is None:
            raise RuntimeError("PointMassDynamics telemetry requires a DynamicContext")

        y = np.asarray(y, dtype=float)
        r = y[0:3]
        v = y[3:6]
        lat0 = lon0 = 0.0
        if cfg is not None:
            lat0 = float(cfg.launch.lat_deg)
            lon0 = float(cfg.launch.lon_deg)
        env = self._aero_env(dyn_ctx, r, v, lat0, lon0)
        return {
            "h": float(r[2]) if self.frame == "ENU" else float(np.linalg.norm(r) - 6371e3),
            "v_inertial": float(np.linalg.norm(v)),
            "q": env.q,
            "Ma": env.Ma,
            "rho": env.rho,
        }
