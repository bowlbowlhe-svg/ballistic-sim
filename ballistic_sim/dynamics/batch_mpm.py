r"""批量 MPM 求解器（NumPy/CuPy 向量化）.

为 Monte Carlo 大规模样本优化：所有样本在同一数组上同步积分，
便于在 CPU（NumPy）或 GPU（CuPy）上并行执行。

当前限制：
- 仅支持标准 ISA 大气（无自定义探空廓线、无湿度修正）
- 仅支持 UniformWind（每个样本可独立）
- 忽略自转偏流与动态攻角（alpha=0），适合散布统计分析
- 固定步长 RK4，落地用线性插值
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.interpolate import PchipInterpolator

from ballistic_sim.constants import (
    DEG2RAD,
    G0_STANDARD,
    GAMMA_AIR,
    OMEGA_EARTH,
    P0_ISA,
    R_AIR,
    T0_ISA,
)
from ballistic_sim.dynamics.mpm import MPMOptions
from ballistic_sim.models.gravity import gravity_wgs84


@dataclass
class BatchMPMResult:
    """批量 MPM 结果."""

    range_m: np.ndarray
    cross_m: np.ndarray
    tof: np.ndarray
    v_impact: np.ndarray
    impact_angle: np.ndarray
    landed: np.ndarray
    n_samples: int


class BatchMPMModel:
    """批量修正质点模型.

    参数数组长度均为 n_samples，允许每个样本独立变化。
    状态向量 6 维 ``[E, N, U, vE, vN, vU]``。
    """

    # ISA 分层：位势高度底界 (m), 温度递减率 (K/m)
    _ISA_HB = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0])
    _ISA_LB = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002])
    _H_GEOPOT_MAX = 84852.0
    _H_GEOM_MAX = 86000.0
    _R0_GEOPOT = 6356766.0

    def __init__(
        self,
        mass_kg: np.ndarray,
        diameter_m: np.ndarray,
        form_factor: np.ndarray,
        v0: np.ndarray,
        theta_deg: np.ndarray,
        az_deg: np.ndarray,
        delta_t: np.ndarray,
        density_factor: np.ndarray,
        wind_e: np.ndarray,
        wind_n: np.ndarray,
        wind_u: Optional[np.ndarray] = None,
        lat_deg: float = 39.9,
        h0: float = 0.0,
        azimuth_deg: float = 0.0,
        drag_table: Optional[np.ndarray] = None,
        options: Optional[MPMOptions] = None,
        xp: Any = None,
        t_max: float = 400.0,
        dt: float = 0.05,
    ):
        """初始化批量模型.

        Parameters
        ----------
        mass_kg:
            质量数组 (kg)，长度 n_samples。
        diameter_m:
            直径数组 (m)，长度 n_samples 或标量广播。
        form_factor:
            阻力乘子数组，长度 n_samples。
        v0:
            初速数组 (m/s)。
        theta_deg:
            仰角数组 (deg)。
        az_deg:
            方位角数组 (deg)。
        delta_t:
            温度偏差数组 (K)。
        density_factor:
            密度修正因子数组。
        wind_e, wind_n, wind_u:
            风场 E/N/U 分量数组 (m/s)。
        lat_deg:
            发射点纬度 (deg)，用于重力和科氏力。
        h0:
            初始高度 (m)。
        azimuth_deg:
            发射基准方位角 (deg)，用于 range/cross 投影。
        drag_table:
            形状 (N, 2) 的 [Ma, Cd] 阻力表；未提供时使用 G1。
        options:
            MPM 选项；未提供时默认关闭 spin/dynamic_alpha。
        xp:
            数组后端，默认 numpy；可传入 cupy 以使用 GPU。
        t_max:
            最大积分时间 (s)。
        dt:
            固定积分步长 (s)。
        """
        self.options = options if options is not None else MPMOptions()
        self.xp = xp if xp is not None else np
        self.t_max = t_max
        self.dt = dt
        self.lat_deg = float(lat_deg)
        self.h0 = float(h0)
        self.azimuth_deg = float(azimuth_deg)

        self.n = int(np.asarray(mass_kg).shape[0])
        arrays = [
            mass_kg,
            form_factor,
            v0,
            theta_deg,
            az_deg,
            delta_t,
            density_factor,
            wind_e,
            wind_n,
        ]
        if wind_u is not None:
            arrays.append(wind_u)
        self._assert_shape(*arrays)

        self.mass_kg = self._as_xp(mass_kg)
        self.diameter_m = self._as_xp(diameter_m)
        self.form_factor = self._as_xp(form_factor)
        self.v0 = self._as_xp(v0)
        self.theta_deg = self._as_xp(theta_deg)
        self.az_deg = self._as_xp(az_deg)
        self.delta_t = self._as_xp(delta_t)
        self.density_factor = self._as_xp(density_factor)
        self.wind_e = self._as_xp(wind_e)
        self.wind_n = self._as_xp(wind_n)
        self.wind_u = self._as_xp(wind_u if wind_u is not None else np.zeros(self.n))

        self.area = 0.25 * np.pi * self.diameter_m**2

        # 准备阻力表
        self._drag_ma, self._drag_cd = self._prepare_drag_table(drag_table)

        # ISA 底层温度/压强（一次性计算）
        self._isa_Tb, self._isa_Pb = self._prepare_isa_baseline()

    def _as_xp(self, arr: np.ndarray) -> np.ndarray:
        """将数组转换到 xp 后端，标量广播到 (n_samples,)。"""
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(self.n, float(arr))
        elif arr.shape[0] != self.n:
            if arr.size == 1:
                arr = np.full(self.n, float(arr.flat[0]))
            else:
                raise ValueError(f"参数数组长度必须为 1 或 n_samples={self.n}，得到 {arr.shape[0]}")
        return self.xp.asarray(arr)

    def _assert_shape(self, *arrays: np.ndarray) -> None:
        for a in arrays:
            a_arr = np.asarray(a)
            if a_arr.ndim > 0 and a_arr.shape[0] != self.n:
                raise ValueError(f"参数数组长度必须等于 n_samples={self.n}，得到 {a_arr.shape[0]}")

    def _prepare_drag_table(
        self, drag_table: Optional[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """准备可向量化的 (ma, cd) 表。"""
        if drag_table is None:
            from ballistic_sim.models.aerodynamics import DRAG_G1

            interp = DRAG_G1.cd0_fn
            assert isinstance(interp, PchipInterpolator)
            ma = np.linspace(float(interp.x[0]), float(interp.x[-1]), 501)
            cd = np.asarray(interp(ma), dtype=np.float64)
            return ma, cd
        table = np.asarray(drag_table, dtype=np.float64)
        if table.ndim != 2 or table.shape[1] != 2 or table.shape[0] < 2:
            raise ValueError("drag_table 必须是形状 (N, 2) 且 N>=2 的数组")
        return table[:, 0].copy(), table[:, 1].copy()

    def _prepare_isa_baseline(self) -> tuple[np.ndarray, np.ndarray]:
        Tb = [float(T0_ISA)]
        Pb = [float(P0_ISA)]
        for k in range(len(self._ISA_HB) - 1):
            dH = self._ISA_HB[k + 1] - self._ISA_HB[k]
            Tn = Tb[k] + self._ISA_LB[k] * dH
            if abs(self._ISA_LB[k]) < 1e-12:
                p_next = Pb[k] * np.exp(-G0_STANDARD * dH / (R_AIR * Tb[k]))
            else:
                p_next = Pb[k] * (Tn / Tb[k]) ** (-G0_STANDARD / (R_AIR * self._ISA_LB[k]))
            Tb.append(float(Tn))
            Pb.append(float(p_next))
        return np.array(Tb, dtype=np.float64), np.array(Pb, dtype=np.float64)

    def _isa_batch(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """批量 ISA：返回 T, p, rho, c（未加湿度修正）."""
        xp = self.xp
        H = self._R0_GEOPOT * h / (self._R0_GEOPOT + h)
        H = xp.clip(H, 0.0, self._H_GEOPOT_MAX)
        isa_hb = self.xp.asarray(self._ISA_HB, dtype=np.float64)
        isa_tb = self.xp.asarray(self._isa_Tb, dtype=np.float64)
        isa_pb = self.xp.asarray(self._isa_Pb, dtype=np.float64)
        isa_lb = self.xp.asarray(self._ISA_LB, dtype=np.float64)
        idx = xp.searchsorted(isa_hb, H, side="right") - 1
        idx = xp.clip(idx, 0, len(self._ISA_HB) - 2)
        Tb = isa_tb[idx]
        Pb = isa_pb[idx]
        Hb = isa_hb[idx]
        Lb = isa_lb[idx]
        dH = H - Hb
        is_isothermal = xp.abs(Lb) < 1e-12
        T_grad = Tb + Lb * dH
        T = xp.where(is_isothermal, Tb, T_grad)
        p_iso = Pb * xp.exp(-G0_STANDARD * dH / (R_AIR * Tb))
        p_grad = Pb * (T_grad / Tb) ** (-G0_STANDARD / (R_AIR * xp.where(is_isothermal, 1.0, Lb)))
        p = xp.where(is_isothermal, p_iso, p_grad)
        T = T + self.delta_t
        rho = p / (R_AIR * T) * self.density_factor
        c = xp.sqrt(GAMMA_AIR * R_AIR * T)
        return T, p, rho, c

    def _interp_drag_cd(self, ma: np.ndarray) -> np.ndarray:
        """批量线性插值阻力系数."""
        xp = self.xp
        ma_arr = self.xp.asarray(self._drag_ma, dtype=np.float64)
        cd_arr = self.xp.asarray(self._drag_cd, dtype=np.float64)
        ma = xp.clip(ma, ma_arr[0], ma_arr[-1])
        idx = xp.searchsorted(ma_arr, ma, side="right") - 1
        idx = xp.clip(idx, 0, len(ma_arr) - 2)
        x0 = ma_arr[idx]
        x1 = ma_arr[idx + 1]
        y0 = cd_arr[idx]
        y1 = cd_arr[idx + 1]
        w = (ma - x0) / (x1 - x0 + 1e-15)
        return y0 + w * (y1 - y0)

    def _derivatives(self, y: np.ndarray) -> np.ndarray:
        """计算批量状态导数.

        y shape: (n_samples, 6) [E, N, U, vE, vN, vU]
        """
        xp = self.xp
        vel = y[:, 3:6]
        h = y[:, 2]

        _, _, rho, c = self._isa_batch(h)

        if self.options.use_wind:
            vr = vel - xp.stack([self.wind_e, self.wind_n, self.wind_u], axis=1)
        else:
            vr = vel
        Vr = xp.linalg.norm(vr, axis=1) + 1e-12

        Ma = Vr / c
        Cd = self._interp_drag_cd(Ma) * self.form_factor
        q = 0.5 * rho * Vr**2

        a = xp.zeros_like(vel)
        if self.options.use_drag:
            k = -q * Cd * self.area / self.mass_kg / Vr
            a = a + k[:, None] * vr

        # 重力
        g = gravity_wgs84(h, self.lat_deg)
        if not isinstance(g, xp.ndarray):
            g = xp.asarray(g)
        a[:, 2] -= g

        # 科氏力
        if self.options.use_coriolis:
            lat_rad = self.lat_deg * DEG2RAD
            Omega_n = OMEGA_EARTH * np.cos(lat_rad)
            Omega_u = OMEGA_EARTH * np.sin(lat_rad)
            a[:, 0] += -2.0 * (Omega_n * vel[:, 2] - Omega_u * vel[:, 1])
            a[:, 1] += -2.0 * (Omega_u * vel[:, 0])
            a[:, 2] += 2.0 * (Omega_n * vel[:, 0])

        return xp.concatenate([vel, a], axis=1)

    def simulate(self) -> BatchMPMResult:
        """批量积分."""
        xp = self.xp
        n = self.n
        theta = self.theta_deg * DEG2RAD
        az = self.az_deg * DEG2RAD
        vh = self.v0 * xp.cos(theta)

        y = xp.zeros((n, 6), dtype=np.float64)
        y[:, 2] = self.h0
        y[:, 3] = vh * xp.sin(az)
        y[:, 4] = vh * xp.cos(az)
        y[:, 5] = self.v0 * xp.sin(theta)

        landed = xp.zeros(n, dtype=bool)
        tof = xp.full(n, xp.nan)
        range_m = xp.full(n, xp.nan)
        cross_m = xp.full(n, xp.nan)
        v_impact = xp.full(n, xp.nan)
        impact_angle = xp.full(n, xp.nan)

        active = xp.ones(n, dtype=bool)
        t = 0.0
        dt = self.dt

        while t < self.t_max:
            k1 = self._derivatives(y)
            k2 = self._derivatives(y + 0.5 * dt * k1)
            k3 = self._derivatives(y + 0.5 * dt * k2)
            k4 = self._derivatives(y + dt * k3)
            y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            t_next = t + dt
            h0 = y[:, 2]
            h1 = y_next[:, 2]

            impact_mask = active & (h0 > 0.0) & (h1 <= 0.0)
            if xp.any(impact_mask):
                frac = h0[impact_mask] / (h0[impact_mask] - h1[impact_mask] + 1e-15)
                frac = xp.clip(frac, 0.0, 1.0)
                y_impact = y[impact_mask] + frac[:, None] * (y_next[impact_mask] - y[impact_mask])
                tof[impact_mask] = t + dt * frac
                range_m[impact_mask] = xp.linalg.norm(y_impact[:, :2], axis=1)
                cross_m[impact_mask] = y_impact[:, 0]
                v_impact[impact_mask] = xp.linalg.norm(y_impact[:, 3:6], axis=1)
                v_down = -y_impact[:, 5]
                v_horiz = xp.linalg.norm(y_impact[:, 3:5], axis=1)
                impact_angle[impact_mask] = xp.arctan2(v_down, v_horiz) / DEG2RAD
                landed[impact_mask] = True
                active = active & ~impact_mask

            y = y_next
            t = t_next

            if not bool(xp.any(active)):
                break

        return BatchMPMResult(
            range_m=self._to_numpy(range_m),
            cross_m=self._to_numpy(cross_m),
            tof=self._to_numpy(tof),
            v_impact=self._to_numpy(v_impact),
            impact_angle=self._to_numpy(impact_angle),
            landed=self._to_numpy(landed),
            n_samples=n,
        )

    def _to_numpy(self, arr: np.ndarray) -> np.ndarray:
        if hasattr(arr, "get"):
            return arr.get()
        return np.asarray(arr)
