"""共享 RHS 工具与上下文。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ballistic_sim.constants import GM_EARTH, OMEGA_EARTH, WGS84_A
from ballistic_sim.models.atmosphere import AtmosphereModel
from ballistic_sim.models.wind import WindModel


@dataclass
class AeroEnv:
    """某高度的气动环境。"""

    rho: float
    c: float
    p: float
    T: float
    q: float = 0.0
    Ma: float = 0.0


@dataclass
class DynamicContext:
    """RHS 内部使用的环境/模型上下文。"""

    cfg: Any
    atmosphere: AtmosphereModel
    wind: WindModel
    aero: Any
    terrain: Optional[Any] = None
    gravity_model: str = "j2"
    options: Dict[str, bool] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.options is None:
            self.options = {}


def dynamic_pressure(rho: float, v_rel: float) -> float:
    """动压 q = 0.5 * rho * v_rel^2 (Pa)。"""
    return 0.5 * float(rho) * float(v_rel) * float(v_rel)


def mach_number(v_rel: float, c: float) -> float:
    """马赫数 Ma = |v_rel| / c。"""
    return float(v_rel) / float(c) if float(c) > 0.0 else 0.0


def compute_alpha(
    v_rel: np.ndarray,
    body_axis: np.ndarray,
    max_deg: float = 90.0,
) -> float:
    """攻角 (rad)：相对速度与弹体纵轴夹角。"""
    v = np.asarray(v_rel, dtype=float).reshape(3)
    b = np.asarray(body_axis, dtype=float).reshape(3)
    vn = float(np.linalg.norm(v))
    bn = float(np.linalg.norm(b))
    if vn < 1e-12 or bn < 1e-12:
        return 0.0
    cos_a = float(np.clip(np.dot(v, b) / (vn * bn), -1.0, 1.0))
    return float(np.clip(np.arccos(cos_a), -np.deg2rad(max_deg), np.deg2rad(max_deg)))


def relative_velocity_eci(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
    wind_enu: Optional[np.ndarray] = None,
    lat_deg: float = 0.0,
    lon_deg: float = 0.0,
) -> np.ndarray:
    """ECI 相对气流速度 = v_eci - omega×r - wind_eci。"""
    from ballistic_sim.frames import enu_to_ecef_vec, ecef_to_eci

    v_rel = np.asarray(v_eci, dtype=float).copy()
    omega = np.array([0.0, 0.0, OMEGA_EARTH], dtype=float)
    v_rel -= np.cross(omega, np.asarray(r_eci, dtype=float))
    if wind_enu is not None and np.linalg.norm(wind_enu) > 0.0:
        w_ecef = enu_to_ecef_vec(wind_enu, lat_deg, lon_deg)
        v_rel -= ecef_to_eci(w_ecef, 0.0)
    return v_rel


def rv_to_oe(r: np.ndarray, v: np.ndarray) -> Dict[str, float]:
    """由 ECI 位置速度计算轨道根数（简化）。"""
    r = np.asarray(r, dtype=float).reshape(3)
    v = np.asarray(v, dtype=float).reshape(3)
    mu = GM_EARTH
    rm = float(np.linalg.norm(r))
    vm = float(np.linalg.norm(v))
    h_vec = np.cross(r, v)
    h = float(np.linalg.norm(h_vec))
    e_vec = ((vm * vm - mu / rm) * r - np.dot(r, v) * v) / mu
    e = float(np.linalg.norm(e_vec))
    a = 0.0
    if abs(e - 1.0) > 1e-12:
        a = -mu / (2.0 * (0.5 * vm * vm - mu / rm))
    else:
        a = h * h / mu
    i = float(np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0))) if h > 0.0 else 0.0
    return {
        "a": float(a),
        "e": float(e),
        "i_rad": float(i),
        "i_deg": float(np.rad2deg(i)),
        "h_peri_km": float((a * (1.0 - e) - WGS84_A) / 1e3) if a != 0.0 else 0.0,
        "h_apo_km": float((a * (1.0 + e) - WGS84_A) / 1e3) if a != 0.0 else 0.0,
    }
