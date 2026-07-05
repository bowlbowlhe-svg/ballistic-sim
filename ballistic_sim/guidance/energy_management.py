"""能量管理制导 (Energy Management Guidance) 简化实现。

用于亚轨道/返回式飞行器，按目标能量剖面调整攻角/倾侧角或输出法向加速度。
同时提供 ``control_moment`` 接口供 6-DOF 消费。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from ballistic_sim.constants import DEG2RAD, GM_EARTH, WGS84_A
from ballistic_sim.frames import ecef_to_geodetic, eci_to_ecef
from ballistic_sim.guidance.control import ControlMoment

__all__ = [
    "EnergyManagementGuidance",
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


@dataclass
class EnergyManagementGuidance:
    """能量管理制导律。

    按当前比机械能与目标能量剖面的偏差，输出法向加速度或攻角/倾侧角指令。
    目标能量剖面默认按高度线性下降：``E_target(h) = E0 + slope * h``。

    Parameters
    ----------
    energy_target_j_kg:
        终端目标比机械能 (J/kg)。
    energy_slope_j_kg_m:
        能量随高度变化斜率 (J/kg / m)。负值表示随高度下降能量减小。
    max_normal_accel:
        最大法向加速度 (m/s^2)。
    max_aoa_deg:
        最大攻角 (deg)。
    kp:
        能量误差 -> 法向加速度的比例增益。
    moment_gain:
        6-DOF 控制力矩增益 (N·m/rad)。
    """

    energy_target_j_kg: float = -0.5 * _G0 * WGS84_A
    energy_slope_j_kg_m: float = -1.0
    max_normal_accel: float = 50.0
    max_aoa_deg: float = 20.0
    kp: float = 0.1
    moment_gain: float = 5.0e3
    energy_profile: Optional[Callable[[float], float]] = None
    failed: bool = field(default=False, init=False)

    def _target_energy_at_height(self, h_m: float) -> float:
        if self.energy_profile is not None:
            return float(self.energy_profile(h_m))
        return float(self.energy_target_j_kg) + self.energy_slope_j_kg_m * float(h_m)

    def command(
        self,
        t: float,
        r_eci: Any,
        v_eci: Any,
    ) -> dict:
        """输出能量管理指令。

        Returns
        -------
        dict
            ``{"normal_accel": ..., "aoa_deg": ..., "bank_deg": ..., "failed": bool}``
        """
        _ = t
        r = np.asarray(r_eci, dtype=float).reshape(3)
        v = np.asarray(v_eci, dtype=float).reshape(3)
        if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v))):
            self.failed = True
            return {"normal_accel": 0.0, "aoa_deg": 0.0, "bank_deg": 0.0, "failed": True}

        r_ecef = eci_to_ecef(r, 0.0)
        _, _, h_m = ecef_to_geodetic(r_ecef)
        E_now = specific_energy(r, v)
        E_target = self._target_energy_at_height(h_m)
        dE = E_now - E_target  # 正值表示能量过高，需增加法向机动以耗散能量

        # 法向加速度：能量偏高 -> 拉起（增大攻角）产生更大阻力/转弯
        normal_accel = float(np.clip(self.kp * dE, -self.max_normal_accel, self.max_normal_accel))

        # 攻角：按法向加速度与 g0 的比值粗略映射
        aoa_deg = float(np.clip(
            np.degrees(np.arctan2(abs(normal_accel), _G0)),
            -self.max_aoa_deg,
            self.max_aoa_deg,
        ))
        if normal_accel < 0.0:
            aoa_deg = -aoa_deg

        self.failed = not (np.isfinite(normal_accel) and np.isfinite(aoa_deg))
        return {
            "normal_accel": normal_accel,
            "aoa_deg": aoa_deg,
            "bank_deg": 0.0,
            "failed": self.failed,
        }

    def control_moment(
        self,
        y: Any,
        alpha_rad: float = 0.0,
        beta_rad: float = 0.0,
    ) -> ControlMoment:
        """6-DOF 控制力矩接口：按能量误差调整攻角。"""
        y_arr = np.asarray(y, dtype=float)
        if y_arr.size < 13:
            self.failed = True
            return ControlMoment()
        r = y_arr[0:3]
        v = y_arr[3:6]
        cmd = self.command(0.0, r, v)
        aoa_cmd = float(cmd.get("aoa_deg", 0.0)) * DEG2RAD
        error_alpha = alpha_rad - aoa_cmd
        error_beta = beta_rad  # 期望侧滑为 0
        return ControlMoment(
            L=0.0,
            M=-self.moment_gain * error_alpha,
            N=-self.moment_gain * error_beta,
        )

    def set_target_profile(
        self,
        energy_target_j_kg: float,
        slope_j_kg_m: float = -1.0,
    ) -> "EnergyManagementGuidance":
        """设置目标能量剖面。"""
        self.energy_target_j_kg = float(energy_target_j_kg)
        self.energy_slope_j_kg_m = float(slope_j_kg_m)
        return self
