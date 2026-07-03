"""入轨椭圆/轨道根数可视化。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ballistic_sim.constants import WGS84_A
from ballistic_sim.dynamics.common import rv_to_oe
from ballistic_sim.simulator import SimResult


def _ellipse_points(a: float, e: float, n: int = 200) -> np.ndarray:
    """椭圆上点的坐标 (二维轨道平面内)。"""
    theta = np.linspace(0, 2 * np.pi, n)
    r = a * (1.0 - e**2) / (1.0 + e * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def plot_orbit_insertion(result: SimResult) -> Figure:
    """绘制轨道插入椭圆与轨道根数摘要。"""
    y = result.y[-1]
    r = y[0:3]
    v = y[3:6]
    oe = rv_to_oe(r, v)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]

    a = float(oe["a"])
    e = float(oe["e"])
    pts = _ellipse_points(a, e)
    ax.plot(pts[:, 0] / 1e6, pts[:, 1] / 1e6, lw=1.5, label="Orbit")
    # 地球
    earth = plt.Circle((0.0, 0.0), WGS84_A / 1e6, color="blue", alpha=0.3, label="Earth")
    ax.add_patch(earth)
    ax.scatter([np.linalg.norm(r) / 1e6], [0.0], color="red", label="Insertion point")
    ax.set_aspect("equal")
    ax.set_xlabel("In-plane X (1000 km)")
    ax.set_ylabel("In-plane Y (1000 km)")
    ax.set_title("Insertion Orbit")
    ax.legend()
    ax.grid(True)

    info = (
        f"a = {a / 1e3:.1f} km\n"
        f"e = {e:.4f}\n"
        f"i = {oe['i_deg']:.2f}°\n"
        f"h_peri = {oe['h_peri_km']:.1f} km\n"
        f"h_apo = {oe['h_apo_km']:.1f} km"
    )
    axes[1].text(0.1, 0.5, info, fontsize=14, family="monospace", va="center")
    axes[1].axis("off")
    axes[1].set_title("Orbital Elements")

    fig.tight_layout()
    return fig
