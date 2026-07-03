"""事件系统测试（阶段 2）。

测试目标：
- 落地事件检测；
- 远地点事件检测；
- 推进剂耗尽事件检测；
- 目标距离事件检测。

当前 ``ballistic_sim.phases`` 目录为空，事件系统尚未实现，因此：
1. 对真实事件 API 的测试标记为 ``pytest.mark.skip``；
2. 同时提供基于黄金轨迹数组的事件检测函数，作为回归基线，
   验证事件判据本身在数学上的正确性。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden"


# ==============================================================================
# 事件检测基线函数（不依赖 phases 模块）
# ==============================================================================


def detect_impact(
    t: np.ndarray,
    alt: np.ndarray,
    terrain_alt: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """检测首次从上方穿越地形/海平面的时刻并线性插值。

    Returns
    -------
    dict or None
        {"t": float, "alt": float, "index": int}
    """
    if alt.size < 2:
        return None
    above = alt > terrain_alt
    # 找到第一个“此前在上方、当前在或低于地面”的索引
    crossings = np.where(above[:-1] & (alt[1:] <= terrain_alt))[0]
    if crossings.size == 0:
        return None
    idx = int(crossings[0]) + 1
    t0, t1 = t[idx - 1], t[idx]
    h0, h1 = alt[idx - 1], alt[idx]
    if abs(h1 - h0) < 1e-12:
        return {"t": float(t1), "alt": float(h1), "index": idx}
    t_impact = t0 + (terrain_alt - h0) * (t1 - t0) / (h1 - h0)
    return {"t": float(t_impact), "alt": float(terrain_alt), "index": idx}


def detect_apogee(
    t: np.ndarray,
    alt: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """检测轨迹最高点（远地点/最高点）。"""
    if alt.size == 0:
        return None
    idx = int(np.argmax(alt))
    return {"t": float(t[idx]), "alt": float(alt[idx]), "index": idx}


def detect_propellant_depletion(
    t: np.ndarray,
    mass: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """检测质量停止下降的时刻（推进剂耗尽）。"""
    if mass.size < 2:
        return None
    dm = np.diff(mass)
    # 找到最后一个 dm < 0 的索引，再往后一列为耗尽点
    burning = np.where(dm < -1e-9)[0]
    if burning.size == 0:
        return None
    idx = int(burning[-1]) + 1
    return {"t": float(t[idx]), "mass_kg": float(mass[idx]), "index": idx}


def detect_target_distance(
    t: np.ndarray,
    pos: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> Optional[Dict[str, Any]]:
    """检测首次进入目标距离阈值的时刻。"""
    dist = np.linalg.norm(pos - np.asarray(target), axis=1)
    inside = dist <= threshold
    if not np.any(inside):
        return None
    idx = int(np.argmax(inside))
    return {"t": float(t[idx]), "distance_m": float(dist[idx]), "index": idx}


# ==============================================================================
# 基线测试（基于黄金文件）
# ==============================================================================


def _load_golden(name: str) -> Dict[str, Any]:
    with open(GOLDEN_DIR / name, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def golden_m107() -> Dict[str, Any]:
    return _load_golden("atmospheric_m107.json")


@pytest.fixture
def golden_cz2f() -> Dict[str, Any]:
    return _load_golden("exo_cz2f.json")


def test_baseline_impact_detection(golden_m107: Dict[str, Any]) -> None:
    """M107 黄金轨迹应能检测到落地事件。"""
    ts = golden_m107["timeseries"]
    t = np.asarray(ts["t_s"])
    alt = np.asarray(ts["alt_m"])
    event = detect_impact(t, alt)
    assert event is not None
    assert event["alt"] == pytest.approx(0.0, abs=1.0)
    assert event["t"] == pytest.approx(golden_m107["scalars"]["tof_s"], rel=0.05)


def test_baseline_apogee_detection(golden_m107: Dict[str, Any]) -> None:
    """M107 黄金轨迹应能检测到最高点。"""
    ts = golden_m107["timeseries"]
    t = np.asarray(ts["t_s"])
    alt = np.asarray(ts["alt_m"])
    event = detect_apogee(t, alt)
    assert event is not None
    assert event["alt"] == pytest.approx(golden_m107["scalars"]["max_alt_m"], rel=0.01)


def test_baseline_propellant_depletion_cz2f(golden_cz2f: Dict[str, Any]) -> None:
    """CZ-2F 黄金轨迹应能检测到 SECO（质量停止下降）。"""
    ts = golden_cz2f["timeseries"]
    t = np.asarray(ts["t_s"])
    m = np.asarray(ts["mass_kg"])
    event = detect_propellant_depletion(t, m)
    assert event is not None
    assert event["t"] == pytest.approx(golden_cz2f["scalars"]["t_seco_s"], rel=0.05)


def test_baseline_target_distance_detection(golden_m107: Dict[str, Any]) -> None:
    """构造一个目标点，验证距离事件检测。"""
    ts = golden_m107["timeseries"]
    t = np.asarray(ts["t_s"])
    rg = np.asarray(ts["range_ground_m"])
    alt = np.asarray(ts["alt_m"])
    pos = np.column_stack([rg, np.zeros_like(rg), alt])
    target = np.array([float(golden_m107["scalars"]["range_m"]), 0.0, 0.0])
    event = detect_target_distance(t, pos, target, threshold=100.0)
    assert event is not None
    assert event["distance_m"] <= 100.0


# ==============================================================================
# 真实事件 API 测试
# ==============================================================================


def test_event_api_impact() -> None:
    """落地事件应由 phases 事件系统正确触发。"""
    from ballistic_sim.phases.events import make_ground_event

    ev = make_ground_event(frame="ENU", h_target=0.0)
    # 在地面上方时事件值为正，落地后为负
    y_above = np.array([0.0, 0.0, 100.0, 0.0, 0.0, 0.0])
    y_below = np.array([0.0, 0.0, -10.0, 0.0, 0.0, 0.0])
    assert ev(0.0, y_above) > 0.0
    assert ev(0.0, y_below) < 0.0
    assert ev.terminal is True
    assert ev.direction == -1


def test_event_api_apogee() -> None:
    """远地点事件应由 phases 事件系统正确触发。"""
    from ballistic_sim.phases.events import make_apogee_event

    ev = make_apogee_event(frame="ENU")
    # 上升阶段竖直速度为正，下降阶段为负
    y_ascent = np.array([0.0, 0.0, 1000.0, 100.0, 0.0, 50.0])
    y_descent = np.array([0.0, 0.0, 1000.0, 100.0, 0.0, -50.0])
    assert ev(0.0, y_ascent) > 0.0
    assert ev(0.0, y_descent) < 0.0
    assert ev.terminal is False


def test_event_api_propellant_depletion() -> None:
    """推进剂耗尽事件应由 phases 事件系统正确触发。"""
    from ballistic_sim.phases.events import make_burnout_event

    ev = make_burnout_event(m_dry=1000.0)
    y_before = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1500.0])
    y_after = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 800.0])
    assert ev(0.0, y_before) > 0.0
    assert ev(0.0, y_after) < 0.0
    assert ev.terminal is True


def test_event_api_target_distance() -> None:
    """目标距离事件应由 phases 事件系统正确触发。"""
    from ballistic_sim.phases.events import make_target_distance_event

    target = np.array([1000.0, 0.0, 0.0])
    ev = make_target_distance_event(target_eci=target, distance_m=100.0)
    y_far = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    y_near = np.array([950.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert ev(0.0, y_far) > 0.0
    assert ev(0.0, y_near) < 0.0
    assert ev.terminal is True
