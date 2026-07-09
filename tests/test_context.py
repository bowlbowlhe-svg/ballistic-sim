"""上下文解析单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.config import (
    EnvironmentConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    VehicleConfig,
)
from ballistic_sim.context import (
    _resolve_dynamics_context,
    _resolve_terrain,
    _resolve_wind,
)
from ballistic_sim.dynamics.common import DynamicContext
from ballistic_sim.models.aerodynamics import (
    ConstantAeroModel,
    ProjectileAeroModel,
    RocketAeroModel,
)
from ballistic_sim.models.atmosphere import AtmosphereModel
from ballistic_sim.models.terrain import NullTerrainModel
from ballistic_sim.models.wind import (
    CompositeWind,
    DrydenGust,
    LogarithmicWind,
    PowerLawWind,
    ProfileWind,
    UniformWind,
)


def _make_cfg(
    mission: str = "projectile",
    env: EnvironmentConfig | None = None,
) -> SimConfig:
    return SimConfig(
        mission=mission,
        vehicle=VehicleConfig(mass_kg=10.0, diameter_m=0.1, cd=0.3),
        launch=LaunchConfig(),
        environment=env or EnvironmentConfig(),
        options=OptionsConfig(),
    )


def test_resolve_wind_uniform_defaults_to_wind_m_s() -> None:
    """uniform 风模型兼容现有 wind_m_s。"""
    cfg = _make_cfg(env=EnvironmentConfig(wind_m_s=[1.0, 2.0, 0.5]))
    wind = _resolve_wind(cfg)
    assert isinstance(wind, UniformWind)
    s = wind(100.0)
    assert s.e == 1.0
    assert s.n == 2.0
    assert s.u == 0.5


def test_resolve_wind_none() -> None:
    """none 风模型返回零风。"""
    cfg = _make_cfg(env=EnvironmentConfig(wind_model="none"))
    wind = _resolve_wind(cfg)
    assert np.allclose(wind(100.0).vector, [0.0, 0.0, 0.0])


def test_resolve_wind_log() -> None:
    """log 风模型解析参数。"""
    cfg = _make_cfg(
        env=EnvironmentConfig(
            wind_model="log",
            wind_u_ref=10.0,
            wind_z_ref=10.0,
            wind_z0=0.03,
            wind_direction_deg=0.0,
        )
    )
    wind = _resolve_wind(cfg)
    assert isinstance(wind, LogarithmicWind)


def test_resolve_wind_power() -> None:
    """power 风模型解析参数。"""
    cfg = _make_cfg(
        env=EnvironmentConfig(
            wind_model="power",
            wind_u_ref=10.0,
            wind_z_ref=10.0,
            wind_alpha=0.14,
            wind_direction_deg=90.0,
        )
    )
    wind = _resolve_wind(cfg)
    assert isinstance(wind, PowerLawWind)


def test_resolve_wind_profile_text() -> None:
    """profile 风模型从 wind_profile_text 解析。"""
    text = "# h e n u\n0 0 0 0\n1000 5 0 0"
    cfg = _make_cfg(env=EnvironmentConfig(wind_model="profile", wind_profile_text=text))
    wind = _resolve_wind(cfg)
    assert isinstance(wind, ProfileWind)
    assert wind(500.0).e > 0.0


def test_resolve_wind_composite() -> None:
    """composite 风模型组合边界层与廓线。"""
    text = "# h e n u\n0 0 0 0\n3000 10 10 0"
    cfg = _make_cfg(
        env=EnvironmentConfig(
            wind_model="composite",
            wind_profile_text=text,
            wind_u_ref=5.0,
        )
    )
    wind = _resolve_wind(cfg)
    assert isinstance(wind, CompositeWind)


def test_resolve_wind_dryden() -> None:
    """dryden 风模型包裹 uniform 基础风。"""
    cfg = _make_cfg(env=EnvironmentConfig(wind_model="dryden", wind_m_s=[1.0, 0.0, 0.0]))
    wind = _resolve_wind(cfg)
    assert isinstance(wind, DrydenGust)


def test_resolve_terrain_null_by_default() -> None:
    """默认地形为 NullTerrainModel。"""
    cfg = _make_cfg()
    terrain = _resolve_terrain(cfg)
    assert isinstance(terrain, NullTerrainModel)


def test_resolve_terrain_via_use_terrain_compat() -> None:
    """use_terrain=True 兼容开关仍启用 hilly。"""
    cfg = _make_cfg(
        env=EnvironmentConfig(
            use_terrain=True,
            terrain_model="hilly",
            terrain_extent=(0.0, 1.0, 0.0, 1.0),
            terrain_max_height=100.0,
        )
    )
    terrain = _resolve_terrain(cfg)
    assert not isinstance(terrain, NullTerrainModel)
    assert terrain.max_height() > 0.0


def test_resolve_terrain_use_terrain_defaults_to_hilly() -> None:
    """use_terrain=True 且 terrain_model 未指定时默认使用 hilly。"""
    cfg = _make_cfg(
        env=EnvironmentConfig(
            use_terrain=True,
            terrain_extent=(0.0, 1.0, 0.0, 1.0),
            terrain_max_height=50.0,
        )
    )
    terrain = _resolve_terrain(cfg)
    assert not isinstance(terrain, NullTerrainModel)
    assert terrain.max_height() <= 50.0 + 1e-6


def test_resolve_terrain_hilly_without_use_terrain() -> None:
    """terrain_model != null 自动视为启用。"""
    cfg = _make_cfg(
        env=EnvironmentConfig(
            terrain_model="hilly",
            terrain_extent=(0.0, 1.0, 0.0, 1.0),
            terrain_max_height=50.0,
        )
    )
    terrain = _resolve_terrain(cfg)
    assert not isinstance(terrain, NullTerrainModel)
    assert terrain.max_height() <= 50.0 + 1e-6


def test_resolve_terrain_numpy(tmp_path) -> None:
    """numpy 地形从临时文件加载。"""
    path = tmp_path / "dem.npy"
    elev = np.array([[0.0, 50.0], [50.0, 100.0]], dtype=float)
    np.save(path, elev)
    cfg = _make_cfg(
        env=EnvironmentConfig(
            terrain_model="numpy",
            terrain_path=str(path),
            terrain_extent=(0.0, 1.0, 0.0, 1.0),
        )
    )
    terrain = _resolve_terrain(cfg)
    # flip_y=True 默认：第 0 行对应 lat_max，故 lat=0, lon=1 对应 100
    assert terrain.height_at(0.0, 1.0) == 100.0


def test_resolve_terrain_image(tmp_path) -> None:
    """image 地形从临时 PNG 加载。"""
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "dem.png"
    img = Image.fromarray(np.array([[0, 128], [128, 255]], dtype=np.uint8), mode="L")
    img.save(path)
    cfg = _make_cfg(
        env=EnvironmentConfig(
            terrain_model="image",
            terrain_path=str(path),
            terrain_extent=(0.0, 1.0, 0.0, 1.0),
            terrain_max_height=1000.0,
        )
    )
    terrain = _resolve_terrain(cfg)
    assert terrain.max_height() > 0.0


def test_resolve_dynamics_context_contains_models() -> None:
    """DynamicContext 包含 atmosphere/wind/aero/terrain。"""
    cfg = _make_cfg(
        env=EnvironmentConfig(terrain_model="hilly", terrain_extent=(0.0, 1.0, 0.0, 1.0))
    )
    ctx = _resolve_dynamics_context(cfg)
    assert isinstance(ctx, DynamicContext)
    assert isinstance(ctx.atmosphere, AtmosphereModel)
    assert isinstance(ctx._wind_model, UniformWind)
    assert isinstance(ctx.aero, ProjectileAeroModel)
    assert ctx.terrain is not None
    assert not isinstance(ctx.terrain, NullTerrainModel)


@pytest.mark.parametrize("mission", ["projectile", "rocket", "missile", "icbm", "suborbital"])
def test_resolve_dynamics_context_for_all_missions(mission: str) -> None:
    """各任务类型均可构造动力学上下文且不抛异常。"""
    cfg = _make_cfg(mission=mission)
    ctx = _resolve_dynamics_context(cfg)
    assert isinstance(ctx, DynamicContext)
    assert ctx.wind is not None
    assert ctx.aero is not None
    if mission == "projectile":
        assert isinstance(ctx.aero, ProjectileAeroModel)
    else:
        assert isinstance(ctx.aero, RocketAeroModel)


def test_resolve_wind_unknown() -> None:
    """未知风模型应抛出 ValueError。"""
    cfg = _make_cfg()
    cfg.environment.wind_model = "unknown"
    with pytest.raises(ValueError) as exc_info:
        _resolve_wind(cfg)
    assert "未知风场模型" in str(exc_info.value)


def test_resolve_wind_composite_from_path(tmp_path) -> None:
    """composite 风模型仅提供 wind_profile_path 时也能正确读取。"""
    path = tmp_path / "wind_profile.txt"
    path.write_text("# h e n u\n0 0 0 0\n2000 4 0 0", encoding="utf-8")
    cfg = _make_cfg(
        env=EnvironmentConfig(
            wind_model="composite",
            wind_profile_path=str(path),
            wind_profile_text=None,
            wind_u_ref=3.0,
        )
    )
    wind = _resolve_wind(cfg)
    assert isinstance(wind, CompositeWind)


def test_resolve_terrain_unknown() -> None:
    """未知地形模型应抛出 ValueError。"""
    cfg = _make_cfg()
    cfg.environment.terrain_model = "unknown"
    with pytest.raises(ValueError) as exc_info:
        _resolve_terrain(cfg)
    assert "未知地形模型" in str(exc_info.value)


def test_resolve_dynamics_context_unknown_mission() -> None:
    """未知 mission 类型回退到 constant 气动模型。"""
    cfg = _make_cfg()
    cfg.mission = "unknown"
    ctx = _resolve_dynamics_context(cfg)
    assert isinstance(ctx, DynamicContext)
    assert isinstance(ctx.aero, ConstantAeroModel)
    assert ctx.aero.drag_coefficient(0.5) == pytest.approx(cfg.vehicle.cd)
