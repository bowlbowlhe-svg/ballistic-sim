"""蒙特卡洛落点散布可视化."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse


def plot_dispersion(result: Any) -> Figure:
    """绘制落点散布分析.

    Parameters
    ----------
    result:
        含 ``ranges``/``crosses``/``range_mean``/``cross_mean``/``cep50``/``cep90``/
        ``ellipse_major``/``ellipse_minor``/``ellipse_angle``/``range_std``/
        ``cross_std``/``samples`` 等字段。

    Returns
    -------
    matplotlib.figure.Figure
        2x2 Figure：散布图 + CEP/椭圆、射程直方图、偏流直方图、落角直方图。
    """
    ranges = np.asarray(result.ranges, dtype=float)
    crosses = np.asarray(result.crosses, dtype=float)
    r_mean = float(result.range_mean)
    c_mean = float(result.cross_mean)

    fig = Figure(figsize=(11.0, 8.0))
    ax_sc = fig.add_subplot(2, 2, 1)
    ax_r = fig.add_subplot(2, 2, 2)
    ax_c = fig.add_subplot(2, 2, 3)
    ax_a = fig.add_subplot(2, 2, 4)

    # 配色与旧项目一致
    colors = {
        "marker": "#1f77b4",
        "mach": "#ff7f0e",
        "angle": "#2ca02c",
        "alt": "#d62728",
        "q": "#9467bd",
        "grid": "#cccccc",
    }

    ax_sc.scatter(crosses, ranges, s=10, color=colors["marker"], alpha=0.4, label="落点")
    ax_sc.scatter([c_mean], [r_mean], s=60, color="#000000", marker="+", label="平均落点")

    for radius, ls, lbl in (
        (float(result.cep50), "-", f"CEP50={result.cep50:.0f}m"),
        (float(result.cep90), "--", f"CEP90={result.cep90:.0f}m"),
    ):
        ax_sc.add_patch(
            Circle(
                (c_mean, r_mean),
                radius,
                fill=False,
                color=colors["mach"],
                ls=ls,
                lw=1.3,
                label=lbl,
            )
        )

    ax_sc.add_patch(
        Ellipse(
            (c_mean, r_mean),
            width=2.0 * float(result.ellipse_major),
            height=2.0 * float(result.ellipse_minor),
            angle=float(result.ellipse_angle),
            fill=False,
            color=colors["angle"],
            lw=1.5,
            label="2σ 椭圆",
        )
    )
    ax_sc.set_title(f"落点散布 (N={result.samples})")
    ax_sc.set_xlabel("横向偏流 (m)")
    ax_sc.set_ylabel("射程 (m)")
    ax_sc.set_aspect("equal", adjustable="datalim")
    ax_sc.legend(loc="best", fontsize=8)

    ax_r.hist(ranges, bins=30, color=colors["alt"], alpha=0.8)
    ax_r.set_title(f"射程分布 (μ={r_mean:.0f}, σ={result.range_std:.1f})")
    ax_r.set_xlabel("射程 (m)")
    ax_r.set_ylabel("频次")

    ax_c.hist(crosses, bins=30, color=colors["q"], alpha=0.8)
    ax_c.set_title(f"偏流分布 (μ={c_mean:.1f}, σ={result.cross_std:.1f})")
    ax_c.set_xlabel("横向偏流 (m)")
    ax_c.set_ylabel("频次")

    ax_a.hist(
        np.asarray(result.impact_angles, dtype=float), bins=30, color=colors["angle"], alpha=0.8
    )
    ax_a.set_title("落角分布")
    ax_a.set_xlabel("落角 (deg)")
    ax_a.set_ylabel("频次")

    for ax in (ax_r, ax_c, ax_a):
        ax.grid(True, color=colors["grid"], ls="--", lw=0.5)

    fig.suptitle("蒙特卡洛散布分析", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
