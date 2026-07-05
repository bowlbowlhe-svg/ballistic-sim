"""基于 Plotly 的交互式 3D 弹道可视化。

``plotly`` 作为可选依赖（``[viz3d]`` extra），所有导入均采用懒加载；缺失时抛出清晰
的 ``ImportError``。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

from ballistic_sim.constants import RE_EARTH_MEAN
from ballistic_sim.simulator import SimResult
from ballistic_sim.viz.earth import coastline_polylines, result_to_ecef, sphere_mesh


def _require_plotly() -> Any:
    """懒加载 plotly；缺失时提示安装 ``[viz3d]`` extra。"""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "3D 可视化需要可选依赖 'plotly'，"
            "请执行 pip install ballistic_sim[viz3d] 安装对应 extra。"
        ) from exc
    return go


def plot_trajectory_3d(
    result: SimResult,
    output_path: Optional[str | Path] = None,
    *,
    show_earth: bool = True,
    show_coastlines: bool = True,
) -> Any:
    """绘制交互式 3D 地球 + 轨迹图。

    Parameters
    ----------
    result:
        仿真结果，支持 ENU 或 ECI 状态。
    output_path:
        若提供，将 Plotly 图保存为 HTML 文件。
    show_earth:
        是否显示地球球面。
    show_coastlines:
        是否显示简化海岸线。

    Returns
    -------
    plotly.graph_objects.Figure
        生成的 Figure 对象。
    """
    go = _require_plotly()

    if result.y.size == 0 or result.t.size == 0:
        raise ValueError("SimResult 为空，无法绘制 3D 轨迹")

    x_m, y_m, z_m, alt_m = result_to_ecef(result)
    x_km = x_m / 1e3
    y_km = y_m / 1e3
    z_km = z_m / 1e3
    alt_km = alt_m / 1e3

    data: list[Any] = []

    if show_earth:
        xs, ys, zs = sphere_mesh(RE_EARTH_MEAN, n_lat=50, n_lon=100)
        data.append(
            go.Surface(
                x=xs / 1e3,
                y=ys / 1e3,
                z=zs / 1e3,
                colorscale=[[0.0, "rgb(30,90,150)"], [1.0, "rgb(30,90,150)"]],
                opacity=0.35,
                showscale=False,
                hoverinfo="skip",
                name="Earth",
            )
        )

    if show_coastlines:
        for poly in coastline_polylines():
            data.append(
                go.Scatter3d(
                    x=poly[:, 0] / 1e3,
                    y=poly[:, 1] / 1e3,
                    z=poly[:, 2] / 1e3,
                    mode="lines",
                    line={"color": "rgb(100,60,30)", "width": 2},
                    hoverinfo="skip",
                    name="Coastline",
                )
            )

    # 轨迹线（按高度着色）
    data.append(
        go.Scatter3d(
            x=x_km,
            y=y_km,
            z=z_km,
            mode="lines",
            line={
                "color": alt_km,
                "colorscale": "Viridis",
                "width": 4,
                "colorbar": {"title": "Altitude (km)", "thickness": 16},
            },
            name="Trajectory",
        )
    )

    # 起点/终点标记
    data.append(
        go.Scatter3d(
            x=[x_km[0], x_km[-1]],
            y=[y_km[0], y_km[-1]],
            z=[z_km[0], z_km[-1]],
            mode="markers",
            marker={"size": 6, "color": ["green", "red"]},
            name="Start / End",
        )
    )

    fig = go.Figure(data=data)
    fig.update_layout(
        title="3D Trajectory (ECEF)",
        scene={
            "xaxis_title": "X (km)",
            "yaxis_title": "Y (km)",
            "zaxis_title": "Z (km)",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )

    if output_path is not None:
        fig.write_html(str(output_path))

    return fig


def demo(output_path: Optional[str | Path] = "trajectory3d_demo.html") -> Any:
    """生成一个示例 3D 轨迹图并保存为 HTML。

    默认使用 CZ-2F 火箭预设生成一条飞出大气层的轨迹。
    """
    _require_plotly()
    from ballistic_sim.presets import cz2f_config, cz2f_phases
    from ballistic_sim.simulator import simulate
    from ballistic_sim.viz import attach_launch_lla

    cfg = cz2f_config()
    result = simulate(cfg, phases=cz2f_phases(cfg))
    attach_launch_lla(result, cfg.launch.lat_deg, cfg.launch.lon_deg, cfg.launch.alt_m)
    fig = plot_trajectory_3d(result, output_path=output_path)
    return fig


def _main() -> None:
    parser = argparse.ArgumentParser(description="Ballistic Sim 交互式 3D 轨迹可视化")
    parser.add_argument("--demo", action="store_true", help="生成示例 HTML")
    parser.add_argument(
        "--output",
        type=str,
        default="trajectory3d_demo.html",
        help="示例 HTML 输出路径",
    )
    args = parser.parse_args()

    if args.demo:
        fig = demo(output_path=args.output)
        print(f"已生成 3D 轨迹示例: {args.output}")
        print(f"Figure traces: {len(fig.data)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
