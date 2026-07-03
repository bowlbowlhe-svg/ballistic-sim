r"""修正质点模型 (Modified Point Mass, MPM)。

来源：``D:\Pycharm\游戏与兴趣\外弹道方程\ballistics\mpm.py``。
迁移到统一平台后，注入 ``ballistic_sim.models.atmosphere.AtmosphereModel`` 与
``ballistic_sim.models.wind.WindModel``，并复用 ``ballistic_sim.frames`` 坐标转换。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from scipy.interpolate import PchipInterpolator

from ballistic_sim.constants import DEG2RAD, OMEGA_EARTH
from ballistic_sim.dynamics.common import (
    AeroEnv,
    DynamicContext,
    dynamic_pressure,
    mach_number,
)
from ballistic_sim.models.gravity import gravity_wgs84


@dataclass
class MPMOptions:
    """MPM 仿真选项。"""

    use_drag: bool = True
    use_wind: bool = True
    use_coriolis: bool = True
    use_spin: bool = True
    use_dynamic_alpha: bool = False
    alpha0_deg: float = 0.1
    method: str = "DOP853"
    rtol: float = 1e-6
    atol: float = 1e-9


@dataclass
class MPMDynamics:
    """修正质点模型 RHS。

    状态向量（标准 8 维）::

        [E, N, U, vE, vN, vU, spin, alpha]

    坐标系为发射点 ENU；风、科氏力、自转偏流可选。
    """

    mass_kg: float = 1.0
    diameter_m: float = 0.1
    form_factor: float = 1.0
    drag_law: Any = None
    twist_cal: float = 20.0
    Ix: float = 0.1
    It: float = 1.0
    CMa_table: Optional[np.ndarray] = None
    CLa_table: Optional[np.ndarray] = None
    Clp_table: Optional[np.ndarray] = None
    CMq_table: Optional[np.ndarray] = None
    options: MPMOptions = field(default_factory=MPMOptions)
    lat_deg: float = 39.9

    def __post_init__(self):
        self.area = float(np.pi * self.diameter_m**2 / 4.0)
        if self.drag_law is None:
            from ballistic_sim.models.aerodynamics import DRAG_G1

            self.drag_law = DRAG_G1
        self._CMa = self._make_aero_interp(self.CMa_table, 3.5)
        self._CLa = self._make_aero_interp(self.CLa_table, 1.6)
        self._Clp = self._make_aero_interp(self.Clp_table, -0.025)
        self._CMq = self._make_aero_interp(self.CMq_table, -1.5)
        self._p0 = 0.0

    @staticmethod
    def _make_aero_interp(table: Optional[np.ndarray], default_value: float = 1.0):
        """从气动系数表创建插值函数。"""
        if table is None or len(table) == 0:
            return lambda ma: default_value
        return PchipInterpolator(table[:, 0], table[:, 1])

    def _spin_rate(self, v0: float) -> float:
        """由初速/缠距计算炮口自转率 (rad/s)。"""
        return 2.0 * np.pi * v0 / (self.twist_cal * self.diameter_m)

    def state_dim(self) -> int:
        return 8

    def native_frame(self) -> str:
        return "ENU"

    def initial_state(
        self,
        v0: float,
        theta_deg: float,
        az_deg: float,
        h0: float = 0.0,
    ) -> np.ndarray:
        """构造 MPM 初始状态。"""
        theta = np.deg2rad(theta_deg)
        az = np.deg2rad(az_deg)
        vh = v0 * np.cos(theta)
        alpha0 = self.options.alpha0_deg * DEG2RAD if self.options.use_dynamic_alpha else 0.0
        return np.array(
            [
                0.0,
                0.0,
                h0,
                vh * np.sin(az),
                vh * np.cos(az),
                v0 * np.sin(theta),
                self._spin_rate(v0),
                alpha0,
            ],
            dtype=float,
        )

    def _aero(self, ctx: DynamicContext, U: float, v: np.ndarray) -> AeroEnv:
        atm = ctx.atmosphere(max(U, 0.0))
        wind_vec = np.zeros(3, dtype=float)
        if self.options.use_wind and ctx.wind is not None:
            w = ctx.wind(U)
            wind_vec = w.vector
        vr = v - wind_vec
        Vr = float(np.linalg.norm(vr)) + 1e-12
        Ma = mach_number(Vr, atm.c)
        return AeroEnv(
            rho=atm.rho,
            c=atm.c,
            p=atm.p,
            T=atm.T,
            q=dynamic_pressure(atm.rho, Vr),
            Ma=Ma,
        )

    def rhs(self, t: float, y: np.ndarray, ctx: Any) -> np.ndarray:
        """MPM 右端项。"""
        from ballistic_sim.phases.base import PhaseContext

        if isinstance(ctx, PhaseContext):
            dyn_ctx = getattr(ctx.cfg, "_dynamics_context", None)
        else:
            dyn_ctx = ctx
        if dyn_ctx is None:
            raise RuntimeError("MPMDynamics requires a DynamicContext")

        y = np.asarray(y, dtype=float)
        U = float(y[2])
        v = y[3:6].copy()
        p = float(y[6])
        alpha = float(y[7])

        env = self._aero(dyn_ctx, U, v)
        rho = env.rho
        c = env.c
        if self.options.use_wind and dyn_ctx.wind is not None:
            Vr = float(np.linalg.norm(v - dyn_ctx.wind(U).vector))
        else:
            Vr = float(np.linalg.norm(v))
        Vr = max(Vr, 1e-12)
        Ma = Vr / c if c > 0.0 else 0.0
        Cd = float(self.drag_law(Ma, alpha)) * self.form_factor
        q = 0.5 * rho * Vr**2

        a = np.zeros(3, dtype=float)
        if self.options.use_drag:
            k = -q * Cd * self.area / self.mass_kg / Vr
            a += k * v

        g = gravity_wgs84(U, self.lat_deg)
        a[2] -= g

        if self.options.use_coriolis:
            lat_rad = np.deg2rad(self.lat_deg)
            Omega_n = OMEGA_EARTH * np.cos(lat_rad)
            Omega_u = OMEGA_EARTH * np.sin(lat_rad)
            a[0] += -2.0 * (Omega_n * v[2] - Omega_u * v[1])
            a[1] += -2.0 * (Omega_u * v[0])
            a[2] += 2.0 * (Omega_n * v[0])

        dp = 0.0
        alpha_deg = abs(alpha) / DEG2RAD
        if (
            self.options.use_spin
            and self.options.use_drag
            and Vr > 1.0
            and float(self._CMa(Ma)) > 1e-6
        ):
            CMa = float(self._CMa(Ma))
            CLa = float(self._CLa(Ma))
            Clp = float(self._Clp(Ma))
            cross = np.cross(v, a)
            f = -(2.0 * self.Ix * p) / (rho * self.area * self.diameter_m * Vr**4 * CMa)
            ae = f * cross
            ae_mag = float(np.linalg.norm(ae))
            kl = q * self.area * CLa / self.mass_kg
            a += kl * ae
            dp = (rho * self.area * self.diameter_m**2 * Vr * Clp / (2.0 * self.Ix)) * p
            alpha_deg = ae_mag / DEG2RAD

        dalpha = 0.0
        if self.options.use_dynamic_alpha:
            CMa = float(self._CMa(Ma))
            omega_n = rho * self.area * self.diameter_m * CMa * Vr / (2.0 * self.It)
            CMq = float(self._CMq(Ma))
            if omega_n > 1e-9:
                zeta = (
                    -0.25
                    * rho
                    * Vr
                    * self.area
                    * self.diameter_m**2
                    * CMq
                    / self.It
                    / (2.0 * omega_n)
                )
                zeta = float(np.clip(zeta, 0.0, 1.0))
            else:
                zeta = 0.0
            alpha_target = alpha_deg * DEG2RAD if alpha_deg > 0 else 0.0
            dalpha = -2.0 * zeta * omega_n * alpha + omega_n**2 * (alpha_target - alpha)

        return np.concatenate([v, a, [dp, dalpha]])
