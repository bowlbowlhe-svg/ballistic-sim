"""仿真器上下文解析测试。"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from ballistic_sim.config import (
    EnvironmentConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    VehicleConfig,
)
from ballistic_sim.models.terrain import NullTerrainModel
from ballistic_sim.models.wind import (
    CompositeWind,
    DrydenGust,
    LogarithmicWind,
    PowerLawWind,
    ProfileWind,
    UniformWind,
)
from ballistic_sim.context import (
    _resolve_dynamics_context,
    _resolve_terrain,
    _resolve_wind,
)


def _make_cfg(env: EnvironmentConfig | None = None) -> SimConfig:
    return SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(mass_kg=10.0, diameter_m=0.1, cd=0.3),
        launch=LaunchConfig(),
        environment=env or EnvironmentConfig(),
        options=OptionsConfig(),
    )


def test_resolve_wind_uniform_defaults_to_wind_m_s() -> None:
    """uniform 风模型兼容现有 wind_m_s。"""
    cfg = _make_cfg(EnvironmentConfig(wind_m_s=[1.0, 2.0, 0.5]))
    wind = _resolve_wind(cfg)
    assert isinstance(wind, UniformWind)
    s = wind(100.0)
    assert s.e == 1.0
    assert s.n == 2.0
    assert s.u == 0.5


def test_resolve_wind_none() -> None:
    """none 风模型返回零风。"""
    cfg = _make_cfg(EnvironmentConfig(wind_model="none"))
    wind = _resolve_wind(cfg)
    assert np.allclose(wind(100.0).vector, [0.0, 0.0, 0.0])


def test_resolve_wind_log() -> None:
    """log 风模型解析参数。"""
    cfg = _make_cfg(
        EnvironmentConfig(
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
        EnvironmentConfig(
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
    cfg = _make_cfg(EnvironmentConfig(wind_model="profile", wind_profile_text=text))
    wind = _resolve_wind(cfg)
    assert isinstance(wind, ProfileWind)
    assert wind(500.0).e > 0.0


def test_resolve_wind_composite() -> None:
    """composite 风模型组合边界层与廓线。"""
    text = "# h e n u\n0 0 0 0\n3000 10 10 0"
    cfg = _make_cfg(
        EnvironmentConfig(
            wind_model="composite",
            wind_profile_text=text,
            wind_u_ref=5.0,
        )
    )
    wind = _resolve_wind(cfg)
    assert isinstance(wind, CompositeWind)


def test_resolve_wind_dryden() -> None:
    """dryden 风模型包裹 uniform 基础风。"""
    cfg = _make_cfg(EnvironmentConfig(wind_model="dryden", wind_m_s=[1.0, 0.0, 0.0]))
    wind = _resolve_wind(cfg)
    assert isinstance(wind, DrydenGust)


def test_resolve_terrain_null_by_default() -> None:
    """默认地形为 NullTerrainModel。"""
    cfg = _make_cfg(EnvironmentConfig())
    terrain = _resolve_terrain(cfg)
    assert isinstance(terrain, NullTerrainModel)


def test_resolve_terrain_via_use_terrain_compat() -> None:
    """use_terrain=True 兼容开关仍启用 hilly。"""
    cfg = _make_cfg(
        EnvironmentConfig(
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
        EnvironmentConfig(
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
        EnvironmentConfig(
            terrain_model="hilly",
            terrain_extent=(0.0, 1.0, 0.0, 1.0),
            terrain_max_height=50.0,
        )
    )
    terrain = _resolve_terrain(cfg)
    assert not isinstance(terrain, NullTerrainModel)
    assert terrain.max_height() <= 50.0 + 1e-6


def test_resolve_dynamics_context_contains_terrain() -> None:
    """DynamicContext 包含 wind 与 terrain。"""
    cfg = _make_cfg(EnvironmentConfig(terrain_model="hilly", terrain_extent=(0.0, 1.0, 0.0, 1.0)))
    ctx = _resolve_dynamics_context(cfg)
    assert ctx.wind is not None
    assert ctx.terrain is not None
    assert not isinstance(ctx.terrain, NullTerrainModel)


def test_resolve_terrain_numpy(tmp_path) -> None:
    """numpy 地形从临时文件加载。"""
    path = tmp_path / "dem.npy"
    elev = np.array([[0.0, 50.0], [50.0, 100.0]], dtype=float)
    np.save(path, elev)
    cfg = _make_cfg(
        EnvironmentConfig(
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
    pytest = __import__("pytest")
    Image = pytest.importorskip("PIL.Image")
    path = tmp_path / "dem.png"
    img = Image.fromarray(np.array([[0, 128], [128, 255]], dtype=np.uint8), mode="L")
    img.save(path)
    cfg = _make_cfg(
        EnvironmentConfig(
            terrain_model="image",
            terrain_path=str(path),
            terrain_extent=(0.0, 1.0, 0.0, 1.0),
            terrain_max_height=1000.0,
        )
    )
    terrain = _resolve_terrain(cfg)
    assert terrain.max_height() > 0.0


def test_resolve_terrain_srtm_tile_parsing(tmp_path) -> None:
    """SRTM .hgt 文件可被正确解析。"""
    from ballistic_sim.models.terrain import SRTMTile

    path = tmp_path / "N00E000.hgt"
    size = 3
    elev = np.zeros((size, size), dtype=">i2")
    elev[1, 1] = 123
    path.write_bytes(elev.tobytes())
    tile = SRTMTile.from_file(path)
    model = tile.to_model()
    # flipud 后第 0 行对应原数组末行
    assert model.max_height() == 123.0


def test_simulate_reports_integration_failure() -> None:
    """当 solve_ivp 返回 success=False 时，simulate 应标记 stop_reason。"""
    from ballistic_sim.simulator import simulate

    cfg = _make_cfg()
    fake_y = np.zeros((7, 1), dtype=float)
    fake_sol = type(
        "FakeSol",
        (),
        {"success": False, "t": np.array([0.0]), "y": fake_y, "t_events": None},
    )()
    with patch("ballistic_sim.simulator.solve_ivp", return_value=fake_sol):
        result = simulate(cfg)
    assert result.stop_reason.startswith("integration_failed")
