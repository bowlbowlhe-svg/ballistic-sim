"""6-DOF 刚体动力学 (轴对称弹，四元数弹轴 + 横向角速度).

状态向量布局 (与源项目 ``ballistics.sixdof`` 保持一致) ::

    [E, N, U, vE, vN, vU, qx, qy, qz, qw, omega_y, omega_z, p]

其中 ``q = [x, y, z, w]`` 为 scalar-last 四元数，表示 **体轴 -> ENU** 的主动旋转。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import numpy as np
from scipy.interpolate import PchipInterpolator

from ballistic_sim.constants import OMEGA_EARTH, RAD2DEG
from ballistic_sim.dynamics.common import (
    AeroEnv,
    DynamicContext,
    dynamic_pressure,
    mach_number,
)
from ballistic_sim.models.gravity import gravity_wgs84


def _make_aero_interp(table: Optional[np.ndarray], default_value: float = 0.0):
    """从气动系数表创建马赫数插值函数。"""
    if table is None or len(table) == 0:
        return lambda ma: float(default_value)
    return PchipInterpolator(
        np.asarray(table[:, 0], dtype=float),
        np.asarray(table[:, 1], dtype=float),
    )


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法 (scalar last ``[x, y, z, w]``)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """用四元数 ``q`` 主动旋转向量 ``v``。"""
    qv = np.array([v[0], v[1], v[2], 0.0])
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
    return _quat_multiply(_quat_multiply(q, qv), q_conj)[:3]


def _axis_from_quat(q: np.ndarray) -> np.ndarray:
    """从四元数提取弹体 x 轴（弹轴）方向。"""
    return _quat_rotate(q, np.array([1.0, 0.0, 0.0]))


def _quat_from_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """构造将 ``v_from`` 旋转到 ``v_to`` 的最短路径四元数 (scalar last)."""
    v_from = v_from / (np.linalg.norm(v_from) + 1e-12)
    v_to = v_to / (np.linalg.norm(v_to) + 1e-12)
    dot = np.clip(np.dot(v_from, v_to), -1.0, 1.0)
    if dot > 0.999999:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = np.cross(v_from, v_to)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    angle = np.arccos(dot)
    s = np.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)])


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    """归一化四元数，零四元数返回单位四元数。"""
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / norm


@dataclass
class SixDOFDynamics:
    """6-DOF 刚体弹道 RHS。

    状态向量 ``[r(3), v(3), q(4), omega_y, omega_z, p]`` 共 13 维，
    原生坐标系为发射点 ENU。
    """

    mass_kg: float = 1.0
    diameter_m: float = 0.1
    form_factor: float = 1.0
    Ix: float = 0.1
    It: float = 1.0
    x_cp_cg: float = 0.05  # 压心在质心前方为正
    lat_deg: float = 0.0
    twist_cal: float = 20.0
    CD0_table: Optional[np.ndarray] = None
    CDa2_table: Optional[np.ndarray] = None
    CNa_table: Optional[np.ndarray] = None
    CMa_table: Optional[np.ndarray] = None
    CMq_table: Optional[np.ndarray] = None
    Clp_table: Optional[np.ndarray] = None
    CYpa_table: Optional[np.ndarray] = None
    control: Optional[Any] = None
    options: Dict[str, bool] = field(
        default_factory=lambda: {
            "drag": True,
            "gravity": True,
            "coriolis": True,
            "thrust": False,
        }
    )
    thrust_N: float = 0.0
    burn_time_s: float = 0.0

    def __post_init__(self):
        self.area = float(np.pi * self.diameter_m**2 / 4.0)
        if abs(self.x_cp_cg) < 1e-6:
            warnings.warn(
                "6-DOF: Xcp≈Xcg（静稳定距≈0），退化为默认 0.05 m; "
                "请显式设置 x_cp_cg 以反映真实布局。",
                stacklevel=2,
            )
            object.__setattr__(self, "x_cp_cg", 0.05)
        self._CD0 = _make_aero_interp(self.CD0_table, 0.25)
        self._CDa2 = _make_aero_interp(self.CDa2_table, 0.05)
        self._CNa = _make_aero_interp(self.CNa_table, 1.6)
        self._CMa = _make_aero_interp(self.CMa_table, -3.5)
        self._CMq = _make_aero_interp(self.CMq_table, -1.5)
        self._Clp = _make_aero_interp(self.Clp_table, -0.025)
        self._CYpa = _make_aero_interp(self.CYpa_table, 0.0)

    def _spin_rate(self, v0: float) -> float:
        """由初速与缠距计算炮口自转角速度 (rad/s)。"""
        return 2.0 * np.pi * v0 / (self.twist_cal * self.diameter_m)

    def state_dim(self) -> int:
        """状态向量维度：13。"""
        return 13

    def native_frame(self) -> Literal["ENU"]:
        """原生坐标系：ENU。"""
        return "ENU"

    def initial_state(
        self,
        v0: float,
        theta_deg: float,
        az_deg: float,
        h0: float = 0.0,
    ) -> np.ndarray:
        """构造 6-DOF 初始状态；弹轴沿初速方向，横向角速度为 0。"""
        theta = np.deg2rad(theta_deg)
        az = np.deg2rad(az_deg)
        vh = v0 * np.cos(theta)
        v_enu = np.array([vh * np.sin(az), vh * np.cos(az), v0 * np.sin(theta)])
        r_enu = np.array([0.0, 0.0, h0])
        q = _quat_from_vectors(np.array([1.0, 0.0, 0.0]), v_enu)
        p0 = self._spin_rate(v0)
        return np.concatenate([r_enu, v_enu, q, np.array([0.0, 0.0, p0])])

    @staticmethod
    def _resolve_context(ctx: Any) -> DynamicContext:
        """兼容 ``DynamicContext`` 与 ``PhaseContext``。"""
        from ballistic_sim.phases.base import PhaseContext

        if isinstance(ctx, PhaseContext):
            dyn_ctx = getattr(ctx.cfg, "_dynamics_context", None)
        else:
            dyn_ctx = ctx
        if dyn_ctx is None:
            raise RuntimeError("SixDOFDynamics requires a DynamicContext")
        return dyn_ctx

    def _aero_env(
        self,
        ctx: DynamicContext,
        h: float,
        v: np.ndarray,
    ) -> tuple[AeroEnv, np.ndarray, float]:
        """计算气动环境、气流相对速度与总速。"""
        atm = ctx.atmosphere(max(h, 0.0))
        wind_vec = np.zeros(3, dtype=float)
        if ctx.wind is not None:
            wind_vec = ctx.wind(h).vector
        v_air = v - wind_vec
        V = float(np.linalg.norm(v_air)) + 1e-12
        Ma = mach_number(V, atm.c)
        env = AeroEnv(
            rho=atm.rho,
            c=atm.c,
            p=atm.p,
            T=atm.T,
            q=dynamic_pressure(atm.rho, V),
            Ma=Ma,
        )
        return env, v_air, V

    @staticmethod
    def _body_axes(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """由弹轴构造垂直于弹轴的体轴 y/z 方向 (ENU)。"""
        ref = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(s, ref)) > 0.99:
            ref = np.array([1.0, 0.0, 0.0])
        z_b = np.cross(s, ref)
        z_b = z_b / (np.linalg.norm(z_b) + 1e-12)
        y_b = np.cross(z_b, s)
        y_b = y_b / (np.linalg.norm(y_b) + 1e-12)
        return y_b, z_b

    def rhs(self, t: float, y: np.ndarray, ctx: Any) -> np.ndarray:
        """6-DOF 右端项。"""
        dyn_ctx = self._resolve_context(ctx)
        y = np.asarray(y, dtype=float)
        r = y[0:3].copy()
        v = y[3:6].copy()
        q = _normalize_quat(y[6:10])
        omega_y = float(y[10])
        omega_z = float(y[11])
        p = float(y[12])

        s = _axis_from_quat(q)
        y_b, z_b = self._body_axes(s)

        h = float(r[2])
        env, v_air, V = self._aero_env(dyn_ctx, h, v)
        rho = env.rho
        Ma = env.Ma

        u_b = float(np.dot(v_air, s))
        v_b = float(np.dot(v_air, y_b))
        w_b = float(np.dot(v_air, z_b))

        alpha = np.arctan2(w_b, u_b)
        beta = np.arctan2(v_b, u_b)
        phi = float(np.sqrt(alpha**2 + beta**2))

        qdyn = env.q
        S = self.area
        d = self.diameter_m

        force_enu = np.zeros(3, dtype=float)
        moment_y = 0.0
        moment_z = 0.0
        dp = 0.0

        if self.options.get("drag", True):
            CD = self.form_factor * (self._CD0(Ma) + self._CDa2(Ma) * phi**2)
            CN = self._CNa(Ma) * phi if phi > 1e-6 else 0.0

            drag_b_mag = CD * qdyn * S
            normal_b_mag = CN * qdyn * S

            drag_enu = -drag_b_mag * s

            v_perp = v_air - u_b * s
            v_perp_norm = float(np.linalg.norm(v_perp))
            normal_enu = np.zeros(3, dtype=float)
            if v_perp_norm > 1e-12:
                normal_enu = normal_b_mag * v_perp / v_perp_norm

            magnus_enu = np.zeros(3, dtype=float)
            if phi > 1e-6 and abs(p) > 1e-6 and v_perp_norm > 1e-12:
                magnus_dir = np.cross(s, v_perp / v_perp_norm)
                magnus_enu = self._CYpa(Ma) * qdyn * S * (p * d / V) * magnus_dir

            force_enu = drag_enu + normal_enu + magnus_enu

            normal_b = np.array(
                [0.0, float(np.dot(normal_enu, y_b)), float(np.dot(normal_enu, z_b))]
            )
            moment_y = -self.x_cp_cg * normal_b[2]
            moment_z = self.x_cp_cg * normal_b[1]

            # 俯仰/偏航阻尼力矩
            moment_y += self._CMq(Ma) * qdyn * S * d**2 * omega_y / (2.0 * V)
            moment_z += self._CMq(Ma) * qdyn * S * d**2 * omega_z / (2.0 * V)

            # 自转衰减
            dp = (rho * S * d**2 * V * self._Clp(Ma) / (2.0 * self.Ix)) * p

        if self.control is not None:
            ctrl = self.control.control_moment(y, alpha, beta)
            moment_y += float(ctrl.M)
            moment_z += float(ctrl.N)
            dp += float(ctrl.L) / self.Ix

        accel = force_enu / self.mass_kg

        if self.options.get("gravity", True):
            accel[2] -= gravity_wgs84(h, self.lat_deg)

        if self.options.get("coriolis", True):
            lat_rad = np.deg2rad(self.lat_deg)
            Omega_n = OMEGA_EARTH * np.cos(lat_rad)
            Omega_u = OMEGA_EARTH * np.sin(lat_rad)
            coriolis = np.zeros(3, dtype=float)
            coriolis[0] = -2.0 * (Omega_n * v[2] - Omega_u * v[1])
            coriolis[1] = -2.0 * (Omega_u * v[0])
            coriolis[2] = 2.0 * (Omega_n * v[0])
            accel += coriolis

        if self.options.get("thrust", False) and self.thrust_N > 0.0 and t < self.burn_time_s:
            accel += (self.thrust_N / self.mass_kg) * s

        domega_y = moment_y / self.It
        domega_z = moment_z / self.It

        # 四元数运动学：世界系角速度左乘
        omega_enu = omega_y * y_b + omega_z * z_b
        dq = 0.5 * _quat_multiply(np.array([omega_enu[0], omega_enu[1], omega_enu[2], 0.0]), q)

        return np.concatenate([v, accel, dq, np.array([domega_y, domega_z, dp])])

    def telemetry(self, t: float, y: np.ndarray, ctx: Any) -> Dict[str, Any]:
        """返回当前时刻派生量字典。"""
        dyn_ctx = self._resolve_context(ctx)
        y = np.asarray(y, dtype=float)
        r = y[0:3]
        v = y[3:6]
        q = _normalize_quat(y[6:10])
        p = float(y[12])

        h = float(r[2])
        env, v_air, _V = self._aero_env(dyn_ctx, h, v)

        s = _axis_from_quat(q)
        y_b, z_b = self._body_axes(s)
        u_b = float(np.dot(v_air, s))
        v_b = float(np.dot(v_air, y_b))
        w_b = float(np.dot(v_air, z_b))
        alpha = np.arctan2(w_b, u_b)
        beta = np.arctan2(v_b, u_b)

        thrust_active = float(
            self.thrust_N if (self.options.get("thrust", False) and t < self.burn_time_s) else 0.0
        )
        return {
            "h": h,
            "v_inertial": float(np.linalg.norm(v)),
            "alpha_deg": float(RAD2DEG * alpha),
            "beta_deg": float(RAD2DEG * beta),
            "Ma": env.Ma,
            "q": env.q,
            "rho": env.rho,
            "spin_rpm": float(p * 60.0 / (2.0 * np.pi)),
            "thrust_N": thrust_active,
        }
