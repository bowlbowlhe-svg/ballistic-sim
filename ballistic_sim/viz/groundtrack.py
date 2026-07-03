"""地面航迹/落点可视化。"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ballistic_sim.frames import haversine_distance
from ballistic_sim.simulator import SimResult
from ballistic_sim.viz import geodetic_coords


def plot_groundtrack(result: SimResult) -> Figure:
    """绘制地面航迹 (lat/lon) 与落点。"""
    lat, lon, _ = geodetic_coords(result)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(lon, lat, lw=1.5, label="Ground track")
    ax.scatter([lon[0]], [lat[0]], color="green", label="Launch", zorder=5)
    ax.scatter([lon[-1]], [lat[-1]], color="red", label="Impact", zorder=5)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("Ground Track")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_impact_summary(result: SimResult) -> Figure:
    """绘制落点相对发射点的距离/方位摘要。"""
    lat, lon, _ = geodetic_coords(result)
    range_m = haversine_distance(lat[0], lon[0], lat[-1], lon[-1])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter([0.0], [0.0], color="green", s=80, label="Launch")
    ax.scatter([range_m / 1e3], [0.0], color="red", s=80, label="Impact")
    ax.annotate(
        f"{range_m / 1e3:.1f} km",
        xy=(range_m / 2e3, 0.0),
        ha="center",
        va="bottom",
    )
    ax.set_xlabel("Downrange (km)")
    ax.set_title("Impact Summary")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True)
    fig.tight_layout()
    return fig
