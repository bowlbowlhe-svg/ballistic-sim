"""ECI 多级动力上升段运动方程（从 lvsim 迁移）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from ballistic_sim.constants import OMEGA_EARTH, WGS84_A
from ballistic_sim.dynamics.common import (
    AeroEnv,
    DynamicContext,
    compute_alpha,
    dynamic_pressure,
    mach_number,
)
from ballistic_sim.guidance.open_loop import (
    thrust_dir_eci,
    thrust_dir_upperstage,
)
from ballistic_sim.models.gravity import gravity_eci
from ballistic_sim.models.propulsion import PropulsionModel


@dataclass
class PoweredECIDynamics:
    """ECI 动力上升段 3-DOF 质点 RHS。

    状态向量 ``[r_eci(3), v_eci(3), m]``，推进期间 ``dm/dt = -mdot``。
    制导律复用 :mod:`ballistic_sim.guidance.open_loop` / :mod:`ballistic_sim.guidance.peg`。
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
        """几何高度：球近似高（与 lvsim 一致）。"""
        return float(np.linalg.norm(r)) - WGS84_A

    def _rel_velocity(self, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        return v - np.cross(self._omega_vec, r)

    def _thrust_dir(self, t: float, r: np.ndarray, v: np.ndarray, m: float) -> np.ndarray:
        """调用统一制导入口。"""
        g = self.guidance
        if self.use_upperstage:
            return thrust_dir_upperstage(t, r, v, g)
        return thrust_dir_eci(t, r, v, g, m=m)

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
            cd = ctx.cache.aero_coefficient(env.Ma, alpha, ctx.aero.drag_coefficient)
        coeff = -0.5 * env.rho * cd * self.prop.Aref_m2 / mass * vm
        return coeff * v_rel

    def rhs(self, t: float, y: np.ndarray, ctx: Any) -> np.ndarray:
        from ballistic_sim.phases.base import PhaseContext

        if isinstance(ctx, PhaseContext):
            dyn_ctx = getattr(ctx.cfg, "_dynamics_context", None)
        else:
            dyn_ctx = ctx
        if dyn_ctx is None:
            raise RuntimeError("PoweredECIDynamics requires a DynamicContext")

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
            dir_hat = self._thrust_dir(t, r, v, m)
            a += (thrust / m) * dir_hat

        a += self._drag_accel(dyn_ctx, v_rel, env, m)

        return np.concatenate([v, a, [-self.prop.mdot]])

    def telemetry(self, t: float, y: np.ndarray, ctx: Any) -> Dict[str, Any]:
        from ballistic_sim.phases.base import PhaseContext

        if isinstance(ctx, PhaseContext):
            dyn_ctx = getattr(ctx.cfg, "_dynamics_context", None)
        else:
            dyn_ctx = ctx
        if dyn_ctx is None:
            raise RuntimeError("PoweredECIDynamics telemetry requires a DynamicContext")

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
        # 复用 RHS 中的加速度计算，避免再次查询大气/风/气动
        accel = self.rhs(t, y, ctx)[3:6]
        return {
            "h": h,
            "v_inertial": float(np.linalg.norm(v)),
            "v_rel": vm,
            "q": dynamic_pressure(atm.rho, vm),
            "Ma": Ma,
            "thrust": thrust,
            "accel_g": float(np.linalg.norm(accel)) / 9.80665,
            "dir_hat": self._thrust_dir(t, r, v, m),
        }
