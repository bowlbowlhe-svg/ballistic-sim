"""风廓线与风偏贡献。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ballistic_sim.simulator import SimResult
from ballistic_sim.viz import altitude, downrange, enu_coords


def plot(result: SimResult) -> Figure:
    """默认绘图函数: 使用零风占位绘制弹道与风偏贡献。"""
    from ballistic_sim.viz import altitude

    h = altitude(result)
    heights = np.linspace(0, max(float(np.max(h)), 1.0), 50)
    wind_enu = np.zeros((len(heights), 3))
    return plot_wind_contribution(result, wind_enu, heights)


def plot_wind_profile(heights_m: np.ndarray, wind_enu: np.ndarray) -> Figure:
    """绘制风廓线 (高度 vs 东/北风速分量)。"""
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(wind_enu[:, 0], heights_m / 1e3, label="East", lw=1.5)
    ax.plot(wind_enu[:, 1], heights_m / 1e3, label="North", lw=1.5)
    ax.axvline(0.0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Wind Profile")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_wind_contribution(
    result: SimResult,
    wind_enu: np.ndarray,
    heights_m: np.ndarray,
) -> Figure:
    """绘制弹道与风偏贡献对比。"""
    e, n, u = enu_coords(result)
    rng = downrange(result)
    # 线性风偏估算
    drift_e = wind_enu[0, 0] * result.t
    drift_n = wind_enu[0, 1] * result.t
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(e / 1e3, n / 1e3, lw=1.5, label="Actual")
    axes[0].plot(
        (e - drift_e) / 1e3,
        (n - drift_n) / 1e3,
        lw=1.5,
        ls="--",
        label="No-wind estimate",
    )
    axes[0].set_xlabel("East (km)")
    axes[0].set_ylabel("North (km)")
    axes[0].set_title("Wind Drift (plan view)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(rng / 1e3, u / 1e3, lw=1.5)
    axes[1].set_xlabel("Downrange (km)")
    axes[1].set_ylabel("Altitude (km)")
    axes[1].set_title("Altitude-Range")
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True)

    fig.tight_layout()
    return fig
