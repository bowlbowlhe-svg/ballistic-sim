"""大气层内三维弹道 (ENU)。"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ballistic_sim.simulator import SimResult
from ballistic_sim.viz import downrange, enu_coords


def plot_trajectory3d(result: SimResult) -> Figure:
    """绘制 ENU 三维弹道曲线。"""
    e, n, u = enu_coords(result)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(e / 1e3, n / 1e3, u / 1e3, lw=1.5)
    ax.scatter([e[0] / 1e3], [n[0] / 1e3], [u[0] / 1e3], color="green", label="Launch")
    ax.scatter([e[-1] / 1e3], [n[-1] / 1e3], [u[-1] / 1e3], color="red", label="Impact")
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.set_zlabel("Altitude (km)")
    ax.set_title("3-D Trajectory (ENU)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_trajectory3d_topdown(result: SimResult) -> Figure:
    """绘制三维弹道的顶视图与侧视图组合。"""
    e, n, u = enu_coords(result)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(e / 1e3, n / 1e3, lw=1.5)
    axes[0].scatter([e[0] / 1e3], [n[0] / 1e3], color="green", label="Launch")
    axes[0].scatter([e[-1] / 1e3], [n[-1] / 1e3], color="red", label="Impact")
    axes[0].set_xlabel("East (km)")
    axes[0].set_ylabel("North (km)")
    axes[0].set_title("Top-down view")
    axes[0].axis("equal")
    axes[0].legend()

    rng = downrange(result)
    axes[1].plot(rng / 1e3, u / 1e3, lw=1.5)
    axes[1].scatter([rng[0] / 1e3], [u[0] / 1e3], color="green")
    axes[1].scatter([rng[-1] / 1e3], [u[-1] / 1e3], color="red")
    axes[1].set_xlabel("Downrange (km)")
    axes[1].set_ylabel("Altitude (km)")
    axes[1].set_title("Side view")
    axes[1].set_ylim(bottom=0.0)

    fig.tight_layout()
    return fig
