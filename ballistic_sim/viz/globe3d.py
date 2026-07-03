"""地球三维升轨轨迹。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ballistic_sim.constants import WGS84_A
from ballistic_sim.frames import eci_to_ecef
from ballistic_sim.simulator import SimResult


def plot_globe3d(result: SimResult) -> Figure:
    """绘制地球三维与 ECI/ECEF 轨迹。"""
    r = result.y[:, 0:3]
    r_ecef = np.array([eci_to_ecef(r[i], result.t[i]) for i in range(len(result.t))])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 地球球体
    u_sphere = np.linspace(0, 2 * np.pi, 40)
    v_sphere = np.linspace(0, np.pi, 20)
    x = WGS84_A * np.outer(np.cos(u_sphere), np.sin(v_sphere)) / 1e6
    y = WGS84_A * np.outer(np.sin(u_sphere), np.sin(v_sphere)) / 1e6
    z = WGS84_A * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere)) / 1e6
    ax.plot_surface(x, y, z, alpha=0.15, color="blue")

    ax.plot(r_ecef[:, 0] / 1e6, r_ecef[:, 1] / 1e6, r_ecef[:, 2] / 1e6, lw=1.5, color="red")
    ax.scatter(
        [r_ecef[0, 0] / 1e6],
        [r_ecef[0, 1] / 1e6],
        [r_ecef[0, 2] / 1e6],
        color="green",
        s=40,
        label="Launch",
    )
    ax.scatter(
        [r_ecef[-1, 0] / 1e6],
        [r_ecef[-1, 1] / 1e6],
        [r_ecef[-1, 2] / 1e6],
        color="red",
        s=40,
        label="End",
    )
    ax.set_xlabel("X (1000 km)")
    ax.set_ylabel("Y (1000 km)")
    ax.set_zlabel("Z (1000 km)")
    ax.set_title("Earth-Centered Ascent Trajectory")
    ax.legend()
    fig.tight_layout()
    return fig
