"""高度-射程曲线。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ballistic_sim.simulator import SimResult
from ballistic_sim.viz import altitude, downrange, speed


def plot_altitude_range(result: SimResult) -> Figure:
    """绘制高度-射程曲线。"""
    rng = downrange(result)
    alt = altitude(result)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rng / 1e3, alt / 1e3, lw=1.5)
    ax.scatter([rng[0] / 1e3], [alt[0] / 1e3], color="green", label="Launch")
    ax.scatter([rng[-1] / 1e3], [alt[-1] / 1e3], color="red", label="Impact")
    ax.set_xlabel("Downrange (km)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Altitude vs Downrange")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot(result: SimResult) -> Figure:
    """默认绘图函数: 高度-射程曲线。"""
    return plot_altitude_range(result)


def plot_velocity_profile(result: SimResult) -> Figure:
    t = result.t
    alt = altitude(result)
    v = speed(result)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(t, alt / 1e3, lw=1.5)
    axes[0].set_ylabel("Altitude (km)")
    axes[0].set_title("Altitude Profile")
    axes[0].grid(True)
    axes[1].plot(t, v, lw=1.5, color="orange")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].set_title("Velocity Profile")
    axes[1].grid(True)
    fig.tight_layout()
    return fig
