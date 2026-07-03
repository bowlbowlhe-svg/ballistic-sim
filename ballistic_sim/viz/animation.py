"""动画生成骨架。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from ballistic_sim.simulator import SimResult
from ballistic_sim.viz import detect_frame, enu_coords


def animate_trajectory(
    result: SimResult,
    out_path: str,
    fps: int = 30,
    figsize: tuple[int, int] = (10, 7),
) -> str:
    """生成三维轨迹 GIF 动画。

    Parameters
    ----------
    result:
        仿真结果。
    out_path:
        输出 GIF 路径。
    fps:
        帧率。
    figsize:
        画布尺寸。

    Returns
    -------
    输出文件路径。
    """
    frame = detect_frame(result)
    if frame == "ENU":
        e, n, u = enu_coords(result)
    elif frame == "ECI":
        e, n, u = enu_coords(result)
    else:
        raise ValueError(f"Unsupported frame for animation: {frame}")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    (line,) = ax.plot([], [], [], lw=1.5)
    (head,) = ax.plot([], [], [], "ro", markersize=6)
    x_min, x_max = float(np.min(e)) / 1e3, float(np.max(e)) / 1e3
    y_min, y_max = float(np.min(n)) / 1e3, float(np.max(n)) / 1e3
    z_min = min(0.0, float(np.min(u)) / 1e3)
    z_max = max(float(np.max(u)) / 1e3, 1.0)
    if x_min == x_max:
        x_min, x_max = x_min - 1.0, x_max + 1.0
    if y_min == y_max:
        y_min, y_max = y_min - 1.0, y_max + 1.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.set_zlabel("Altitude (km)")
    ax.set_title("Trajectory Animation")

    n_frames = min(len(result.t), max(30, fps * 5))
    indices = np.linspace(0, len(result.t) - 1, n_frames, dtype=int)

    def _init():
        line.set_data_3d([], [], [])
        head.set_data_3d([], [], [])
        return line, head

    def _update(i: int):
        idx = indices[i]
        line.set_data_3d(e[:idx] / 1e3, n[:idx] / 1e3, u[:idx] / 1e3)
        head.set_data_3d([e[idx] / 1e3], [n[idx] / 1e3], [u[idx] / 1e3])
        return line, head

    anim = FuncAnimation(
        fig,
        _update,
        init_func=_init,
        frames=len(indices),
        interval=1000 // fps,
        blit=False,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)
    return out_path
