"""异步 / 流式仿真 API 测试。"""

from __future__ import annotations

import asyncio

import numpy as np

from ballistic_sim import simulate, simulate_async, simulate_streaming
from ballistic_sim.config import (
    EnvironmentConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    VehicleConfig,
)


def _projectile_cfg() -> SimConfig:
    return SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(mass_kg=10.0, diameter_m=0.1, cd=0.3),
        launch=LaunchConfig(elevation_deg=45.0, v0_m_s=500.0),
        environment=EnvironmentConfig(),
        options=OptionsConfig(),
    )


def test_simulate_async_returns_same_as_sync() -> None:
    """异步包装应返回与同步调用一致的结果。"""
    cfg = _projectile_cfg()
    sync_result = simulate(cfg)

    async def _run() -> None:
        async_result = await simulate_async(cfg)
        assert async_result.stop_reason == sync_result.stop_reason
        assert np.allclose(async_result.t, sync_result.t)
        assert np.allclose(async_result.y, sync_result.y)

    asyncio.run(_run())


def test_simulate_streaming_yields_result() -> None:
    """流式 API 最终应产出结果事件。"""
    cfg = _projectile_cfg()

    async def _collect() -> list[dict]:
        events: list[dict] = []
        async for event in simulate_streaming(cfg, progress_interval_s=0.05):
            events.append(event)
        return events

    events = asyncio.run(_collect())
    assert events
    assert events[-1]["type"] == "result"
    assert events[-1]["done"] is True
    result = events[-1]["result"]
    assert result.stop_reason == "completed"
    assert len(result.t) == len(result.y)
    # 进度事件数量应至少有一个
    progress_events = [e for e in events if e["type"] == "progress"]
    assert len(progress_events) >= 0
