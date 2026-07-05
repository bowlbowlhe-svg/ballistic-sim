"""实时指标监控面板：高度 / 速度 / 动压 / 马赫数。

当前基于 matplotlib 实现；plotly 环境可后续以相同接口扩展。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ballistic_sim.models.atmosphere import make_atmosphere
from ballistic_sim.simulator import SimResult
from ballistic_sim.viz import altitude, speed


def _compute_q_mach(result: SimResult) -> tuple[np.ndarray, np.ndarray]:
    """计算动压 (Pa) 与马赫数数组。"""
    h = altitude(result)
    v = speed(result)
    atm = make_atmosphere("isa")
    rho = np.array([atm.density(float(max(hi, 0.0))) for hi in h])
    c = np.array([atm.sound_speed(float(max(hi, 0.0))) for hi in h])
    q = 0.5 * rho * v * v
    mach = np.where(c > 0.0, v / c, 0.0)
    return q, mach


def plot_dashboard(
    result: SimResult,
    output_path: Optional[Union[str, Path]] = None,
) -> Figure:
    """绘制 2×2 监控面板。

    Parameters
    ----------
    result:
        仿真结果。
    output_path:
        若提供，保存 PNG 图片。

    Returns
    -------
    matplotlib.figure.Figure
    """
    if result.y.size == 0 or result.t.size == 0:
        raise ValueError("SimResult 为空，无法绘制监控面板")

    t = result.t
    h = altitude(result)
    v = speed(result)
    q, mach = _compute_q_mach(result)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    axes[0, 0].plot(t, h / 1e3, lw=1.5)
    axes[0, 0].set_ylabel("Altitude (km)")
    axes[0, 0].set_title("Altitude Profile")
    axes[0, 0].grid(True)

    axes[0, 1].plot(t, v, lw=1.5, color="orange")
    axes[0, 1].set_ylabel("Speed (m/s)")
    axes[0, 1].set_title("Velocity Profile")
    axes[0, 1].grid(True)

    axes[1, 0].plot(t, q / 1e3, lw=1.5, color="red")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Dynamic pressure (kPa)")
    axes[1, 0].set_title("Dynamic Pressure")
    axes[1, 0].grid(True)

    axes[1, 1].plot(t, mach, lw=1.5, color="purple")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Mach")
    axes[1, 1].set_title("Mach Number")
    axes[1, 1].grid(True)

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)

    return fig
