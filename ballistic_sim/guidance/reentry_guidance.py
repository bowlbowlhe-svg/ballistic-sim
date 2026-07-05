"""再入制导简化实现。

基于阻力加速度/能量剖面，按剩余射程调整倾侧角符号与大小，
输出倾侧角、攻角指令；同时提供 ``control_moment`` 接口供 6-DOF 消费。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ballistic_sim.constants import DEG2RAD, GM_EARTH, RAD2DEG, WGS84_A
from ballistic_sim.frames import ecef_to_geodetic, eci_to_ecef
from ballistic_sim.guidance.control import ControlMoment

__all__ = [
    "ReentryGuidance",
    "drag_acceleration",
    "specific_energy",
]

_EPS = 1e-12
_G0 = 9.80665


def specific_energy(r_eci: Any, v_eci: Any, mu: float = GM_EARTH) -> float:
    """比机械能 E = v^2/2 - mu/r (J/kg)。"""
    v = np.asarray(v_eci, dtype=float).reshape(3)
    r = np.asarray(r_eci, dtype=float).reshape(3)
    r_mag = float(np.linalg.norm(r))
    if r_mag < _EPS:
        return 0.0
    return 0.5 * float(np.dot(v, v)) - mu / r_mag


def drag_acceleration(
    r_eci: Any,
    v_eci: Any,
    rho: float,
    cd: float,
    area_m2: float,
    mass_kg: float,
) -> float:
    """当前阻力加速度大小 (m/s^2)。"""
    v = np.asarray(v_eci, dtype=float).reshape(3)
    r = np.asarray(r_eci, dtype=float).reshape(3)
    # 相对速度扣除地球自转
    omega = np.array([0.0, 0.0, 7.292115e-5], dtype=float)
    v_rel = v - np.cross(omega, r)
    vm = float(np.linalg.norm(v_rel))
    if vm < _EPS or rho <= 0.0 or mass_kg <= 0.0:
        return 0.0
    return 0.5 * rho * cd * area_m2 / mass_kg * vm * vm


@dataclass
class ReentryGuidance:
    """再入制导律：按剩余射程与能量调整倾侧角/攻角。

    Parameters
    ----------
    target_lat_deg, target_lon_deg:
        目标落点经纬度 (deg)。
    target_energy_j_kg:
        终端比机械能 (J/kg)，未设置时按海平面估算。
    max_bank_deg:
        最大倾侧角 (deg)。
    nominal_aoa_deg:
        标称攻角 (deg)。
    bank_gain:
        能量误差 -> 倾侧角的比例增益。
    moment_gain:
        6-DOF 控制力矩增益 (N·m/rad)。
    """

    target_lat_deg: float = 0.0
    target_lon_deg: float = 0.0
    target_energy_j_kg: Optional[float] = None
    max_bank_deg: float = 60.0
    nominal_aoa_deg: float = 10.0
    bank_gain: float = 1.0e-6
    moment_gain: float = 5.0e3
    failed: bool = field(default=False, init=False)

    def _target_energy(self) -> float:
        if self.target_energy_j_kg is not None:
            return float(self.target_energy_j_kg)
        # 默认：海平面圆轨道能量 -0.5*g0*R 量级，取近似终端能量
        return -0.5 * _G0 * WGS84_A

    def _range_to_go(self, r_eci: Any) -> float:
        """剩余射程：当前位置到目标点的大地线距离 (m)。"""
        r_ecef = eci_to_ecef(np.asarray(r_eci, dtype=float).reshape(3), 0.0)
        lat, lon, _ = ecef_to_geodetic(r_ecef)
        from ballistic_sim.frames import haversine_distance

        return haversine_distance(lat, lon, self.target_lat_deg, self.target_lon_deg)

    def _cross_track_error(self, r_eci: Any, v_eci: Any) -> float:
        """简单横程误差符号：目标在速度方向左侧为负、右侧为正。"""
        r = np.asarray(r_eci, dtype=float).reshape(3)
        v = np.asarray(v_eci, dtype=float).reshape(3)
        r_ecef = eci_to_ecef(r, 0.0)
        lat, lon, _ = ecef_to_geodetic(r_ecef)
        from ballistic_sim.frames import enu_basis

        e_hat, n_hat, u_hat = enu_basis(lat, lon)
        # ECI 速度 -> ECEF 速度，再投影到 ENU
        omega = np.array([0.0, 0.0, 7.292115e-5], dtype=float)
        v_ecef = eci_to_ecef(v, 0.0) - np.cross(omega, r_ecef)
        v_enu = np.array([
            float(np.dot(v_ecef, e_hat)),
            float(np.dot(v_ecef, n_hat)),
            float(np.dot(v_ecef, u_hat)),
        ])
        # 简化：用当前位置到目标的大圆方位与速度方位差
        from ballistic_sim.frames import initial_bearing

        bearing_now = initial_bearing(lat, lon, self.target_lat_deg, self.target_lon_deg)
        vel_bearing = (np.degrees(np.arctan2(v_enu[0], v_enu[1])) + 360.0) % 360.0
        err = (bearing_now - vel_bearing + 180.0) % 360.0 - 180.0
        return float(err)

    def command(
        self,
        t: float,
        r_eci: Any,
        v_eci: Any,
        rho: float = 0.0,
        dyn_ctx: Any = None,
    ) -> dict:
        """输出再入制导指令。

        Returns
        -------
        dict
            ``{"bank_deg": ..., "aoa_deg": ..., "normal_accel": ..., "failed": bool}``
        """
        _ = t
        r = np.asarray(r_eci, dtype=float).reshape(3)
        v = np.asarray(v_eci, dtype=float).reshape(3)
        if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v))):
            self.failed = True
            return {"bank_deg": 0.0, "aoa_deg": self.nominal_aoa_deg,
                    "normal_accel": 0.0, "failed": True}

        E_now = specific_energy(r, v)
        E_target = self._target_energy()
        dE = E_now - E_target  # 正值表示能量过高，需增加阻力
        rng = self._range_to_go(r)

        # 能量误差经射程缩放后映射为倾侧角大小
        # 剩余射程越小，对能量偏差越敏感
        scale = max(rng, 1.0e3)
        bank_mag = self.bank_gain * abs(dE) / scale * RAD2DEG
        bank_mag = float(np.clip(bank_mag, 0.0, self.max_bank_deg))

        # 横程误差决定倾侧角符号（左右转弯）
        cross_err = self._cross_track_error(r, v)
        bank_sign = -1.0 if cross_err > 0.0 else 1.0
        bank_deg = bank_sign * bank_mag

        # 攻角：能量偏高时适度增大攻角以提升阻力
        aoa_deg = float(np.clip(self.nominal_aoa_deg + 0.1 * dE / scale, 1.0, 30.0))

        # 法向加速度指令（供点质量再入动力学使用）
        v_mag = float(np.linalg.norm(v))
        normal_accel = 0.0
        if v_mag > _EPS:
            normal_accel = bank_mag * DEG2RAD * _G0  # 简化：倾侧产生法向过载

        self.failed = not np.isfinite(bank_deg)
        return {
            "bank_deg": bank_deg,
            "aoa_deg": aoa_deg,
            "normal_accel": normal_accel,
            "failed": self.failed,
        }

    def control_moment(
        self,
        y: Any,
        alpha_rad: float = 0.0,
        beta_rad: float = 0.0,
    ) -> ControlMoment:
        """6-DOF 控制力矩接口。

        根据当前攻角/侧滑角与制导指令的偏差，输出俯仰/偏航力矩。
        再入制导主要控制攻角，因此俯仰力矩主导。
        """
        y_arr = np.asarray(y, dtype=float)
        if y_arr.size < 13:
            self.failed = True
            return ControlMoment()
        r = y_arr[0:3]
        v = y_arr[3:6]
        cmd = self.command(0.0, r, v)
        aoa_cmd = float(cmd.get("aoa_deg", self.nominal_aoa_deg)) * DEG2RAD
        # 简单 PD：力矩与攻角误差成正比
        error_alpha = alpha_rad - aoa_cmd
        error_beta = beta_rad  # 期望侧滑为 0
        return ControlMoment(
            L=0.0,
            M=-self.moment_gain * error_alpha,
            N=-self.moment_gain * error_beta,
        )

    def set_target(
        self,
        lat_deg: float,
        lon_deg: float,
        energy_j_kg: Optional[float] = None,
    ) -> "ReentryGuidance":
        """设置目标落点与能量。"""
        self.target_lat_deg = float(lat_deg)
        self.target_lon_deg = float(lon_deg)
        if energy_j_kg is not None:
            self.target_energy_j_kg = float(energy_j_kg)
        return self
