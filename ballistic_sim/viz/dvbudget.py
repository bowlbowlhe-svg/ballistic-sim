"""Δv 预算堆叠图。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ballistic_sim.simulator import SimResult
from ballistic_sim.viz import mass, speed


def _delta_v_ideal(v_end: float, m0: float, m_dry: float, isp: float) -> float:
    """理想速度增量 (Tsiolkovsky)。"""
    from ballistic_sim.constants import G0_STANDARD

    if m0 <= m_dry:
        return 0.0
    return isp * G0_STANDARD * np.log(m0 / m_dry)


def plot_dv_budget(result: SimResult, isp_s: float = 300.0) -> Figure:
    """绘制 Δv 预算堆叠图: 实际动能增量 vs 理想推进 Δv vs 损失。"""
    t = result.t
    v = speed(result)
    m = mass(result)
    valid_mass = not np.all(np.isnan(m))

    delta_v_actual = v - v[0]
    ideal_dv = 0.0
    if valid_mass:
        m0 = float(m[0])
        mdry = float(np.nanmin(m))
        ideal_dv = _delta_v_ideal(v[-1], m0, mdry, isp_s)
    loss = ideal_dv - float(delta_v_actual[-1]) if ideal_dv > 0 else 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, delta_v_actual / 1e3, lw=1.5, label="Actual Δv")
    if ideal_dv > 0:
        ax.axhline(ideal_dv / 1e3, color="green", ls="--", label="Ideal propulsive Δv")
        ax.bar(
            [t[-1] * 0.5],
            [ideal_dv / 1e3],
            width=t[-1] * 0.1,
            color="green",
            alpha=0.3,
            label="Budget",
        )
        ax.bar(
            [t[-1] * 0.6],
            [loss / 1e3],
            width=t[-1] * 0.1,
            color="red",
            alpha=0.3,
            label="Losses",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Δv (km/s)")
    ax.set_title("Δv Budget")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig
