"""再入热流与过载 (骨架实现)。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ballistic_sim.simulator import SimResult
from ballistic_sim.viz import altitude, speed


def plot(result: SimResult, r_nose_m: float = 0.3) -> Figure:
    """默认绘图函数: 再入热流与过载。"""
    return plot_reentry_heat(result, r_nose_m)


def _sutton_graves_heat_flux(
    rho: float,
    v: float,
    r_nose_m: float = 0.3,
) -> float:
    """Sutton-Graves 热流估算 (W/m²)。"""
    return 1.83e-4 * np.sqrt(rho / r_nose_m) * (v**3)


def plot_reentry_heat(result: SimResult, r_nose_m: float = 0.3) -> Figure:
    """绘制再入热流与过载随高度变化。"""
    h = altitude(result)
    v = speed(result)
    # 简化的密度模型 (指数近似)
    rho0 = 1.225
    h_scale = 8500.0
    rho = rho0 * np.exp(-h / h_scale)
    q = _sutton_graves_heat_flux(rho, v, r_nose_m)
    g0 = 9.80665
    # 简化过载: 速度变化率
    dvdt = np.gradient(v, result.t, edge_order=1)
    g_load = np.abs(dvdt) / g0

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(h / 1e3, q / 1e6, lw=1.5)
    axes[0].set_ylabel("Heat flux (MW/m²)")
    axes[0].set_title("Reentry Heat Flux")
    axes[0].grid(True)
    axes[0].invert_xaxis()

    axes[1].plot(h / 1e3, g_load, lw=1.5, color="orange")
    axes[1].set_xlabel("Altitude (km)")
    axes[1].set_ylabel("g-load")
    axes[1].set_title("Reentry Load Factor")
    axes[1].grid(True)
    axes[1].invert_xaxis()

    fig.tight_layout()
    return fig
