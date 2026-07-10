"""端对端测试：覆盖所有 mission / preset 组合。

验证：
- ``build_config_and_phases`` 能成功构造配置与阶段。
- ``simulate`` 运行完成且 ``stop_reason == "completed"``。
- 轨迹时间、状态维度一致。
- ``_compute_summary`` 输出的射程/飞行时间/坐标系与结果本身一致。
- ``plot_altitude_range`` 能生成图像。
"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.cli import _compute_summary
from ballistic_sim.gui.builder import MISSIONS, build_config_and_phases, list_presets
from ballistic_sim.simulator import simulate
from ballistic_sim.viz import attach_launch_lla, detect_frame
from ballistic_sim.viz.profile import plot_altitude_range


def _all_mission_presets() -> list[tuple[str, str | None]]:
    cases: list[tuple[str, str | None]] = []
    for mission in MISSIONS:
        presets = list_presets(mission)
        if presets:
            cases.extend((mission, p) for p in presets)
        else:
            cases.append((mission, None))
    return cases


@pytest.mark.slow
@pytest.mark.parametrize("mission,preset", _all_mission_presets())
def test_simulate_all_mission_presets(mission: str, preset: str | None) -> None:
    """所有 mission / preset 组合都能跑通并给出一致结果。"""
    cfg, phases = build_config_and_phases(mission, preset)
    assert phases, f"{mission}/{preset} 未生成任何 phase"

    result = simulate(cfg)
    assert result.stop_reason == "completed", f"{mission}/{preset} 仿真未完成: {result.stop_reason}"
    assert result.t.size > 0
    assert result.y.shape[0] == result.t.size
    assert result.y.ndim == 2

    # 后处理摘要一致性检查
    attach_launch_lla(result, cfg.launch.lat_deg, cfg.launch.lon_deg, cfg.launch.alt_m)
    summary = _compute_summary(cfg, result)
    assert summary["stop_reason"] == "completed"
    assert summary["t_end_s"] == pytest.approx(float(result.t[-1]), rel=1e-9)

    frame = detect_frame(result)
    if frame == "ENU":
        expected_range = float(np.hypot(result.y[-1, 0], result.y[-1, 1]))
        assert summary["range_m"] == pytest.approx(expected_range, rel=1e-6)
        assert summary["max_alt_m"] >= 0.0
    elif frame == "ECI":
        assert summary["v_end_m_s"] > 0.0
        assert summary["range_m"] >= 0.0
    else:
        pytest.fail(f"未知坐标系: {frame}")

    # 绘图函数不应抛出异常
    fig = plot_altitude_range(result)
    assert fig is not None


def test_summary_returns_basic_keys_for_projectile() -> None:
    """弹丸任务的后处理摘要包含预期字段。"""
    cfg, _ = build_config_and_phases("projectile", "M107")
    result = simulate(cfg)
    summary = _compute_summary(cfg, result)
    for key in ["mission", "stop_reason", "t_end_s", "n_points", "range_m", "max_alt_m"]:
        assert key in summary, f"摘要缺少 {key}"
