"""simulate() 显式 phases 参数弃用测试。"""

from __future__ import annotations

import pytest

from ballistic_sim.phases.builder import build_phases
from ballistic_sim.presets import m107_config
from ballistic_sim.simulator import simulate


def test_explicit_phases_emits_deprecation_warning() -> None:
    """显式传入 phases 应触发 DeprecationWarning。"""
    cfg = m107_config()
    phases = build_phases(cfg)
    with pytest.warns(DeprecationWarning, match="phases"):
        simulate(cfg, phases=phases)


def test_simulate_without_phases_builds_automatically() -> None:
    """不传 phases 时 simulate() 内部自动 build_phases 并正常运行。"""
    cfg = m107_config()
    result = simulate(cfg)
    assert result.stop_reason == "completed"


def test_explicit_empty_phases_builds_automatically_and_warns() -> None:
    """显式传入空列表时保留历史兼容，自动 build_phases 并触发弃用警告。"""
    cfg = m107_config()
    with pytest.warns(DeprecationWarning, match="phases"):
        result = simulate(cfg, phases=[])
    assert result.stop_reason == "completed"


def test_explicit_phases_warning_message_contains_version() -> None:
    """弃用警告消息应包含版本号与推荐用法。"""
    cfg = m107_config()
    phases = build_phases(cfg)
    with pytest.warns(DeprecationWarning, match="v0\\.5\\.0") as record:
        simulate(cfg, phases=phases)
    assert any("simulate(cfg)" in str(w.message) for w in record)
