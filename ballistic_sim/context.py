"""上下文解析：由 SimConfig 构建风场、地形与动力学上下文。"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ballistic_sim.config import SimConfig, apply_overrides
from ballistic_sim.dynamics.common import DynamicContext
from ballistic_sim.models.aerodynamics import make_aero
from ballistic_sim.models.atmosphere import make_atmosphere
from ballistic_sim.models.terrain import NullTerrainModel, TerrainExtent, make_terrain
from ballistic_sim.models.wind import UniformWind, WindModel, make_wind, parse_wind_profile


def _resolve_wind(cfg: SimConfig) -> WindModel:
    """由 ``EnvironmentConfig`` 构建风场模型。"""
    env = cfg.environment
    name = env.wind_model.lower()

    if name in ("none", "null"):
        return make_wind("none")

    if name == "uniform":
        w = env.wind_m_s if env.wind_m_s else [0.0, 0.0, 0.0]
        return UniformWind(e=w[0], n=w[1], u=w[2])

    if name == "log":
        return make_wind(
            "log",
            u_ref=env.wind_u_ref,
            z_ref=env.wind_z_ref,
            z0=env.wind_z0,
            direction_deg=env.wind_direction_deg,
        )

    if name == "power":
        return make_wind(
            "power",
            u_ref=env.wind_u_ref,
            z_ref=env.wind_z_ref,
            alpha=env.wind_alpha,
            direction_deg=env.wind_direction_deg,
        )

    if name == "profile":
        text = env.wind_profile_text
        if text is None and env.wind_profile_path:
            with open(env.wind_profile_path, "r", encoding="utf-8") as f:
                text = f.read()
        if text is None:
            raise ValueError("profile 风模型需要提供 wind_profile_text 或 wind_profile_path")
        profile = parse_wind_profile(text)
        return make_wind("profile", profile=profile)

    if name == "composite":
        boundary = make_wind(
            "log",
            u_ref=env.wind_u_ref,
            z_ref=env.wind_z_ref,
            z0=env.wind_z0,
            direction_deg=env.wind_direction_deg,
        )
        profile_model: Optional[Any] = None
        if env.wind_profile_text or env.wind_profile_path:
            text = env.wind_profile_text
            if text is None and env.wind_profile_path:
                with open(env.wind_profile_path, "r", encoding="utf-8") as f:
                    text = f.read()
            if text is not None:
                profile_model = make_wind("profile", profile=parse_wind_profile(text))
        return make_wind("composite", boundary_layer=boundary, profile=profile_model)

    if name == "dryden":
        base = _resolve_wind(
            apply_overrides(
                cfg,
                {"environment.wind_model": "uniform"},
            )
        )
        return make_wind("dryden", base_wind=base, seed=env.wind_dryden_seed)

    raise ValueError(f"未知风场模型: {env.wind_model}")


def _resolve_terrain(cfg: SimConfig):
    """由 ``EnvironmentConfig`` 构建地形模型。"""
    env = cfg.environment
    if not env.terrain_enabled:
        return NullTerrainModel()

    name = env.terrain_model.lower()
    # 兼容旧版 use_terrain=True 但未指定 terrain_model 的配置，默认使用 hilly
    if name in ("null", "none") and env.use_terrain:
        name = "hilly"
    extent = None
    if env.terrain_extent is not None:
        lat_min, lat_max, lon_min, lon_max = env.terrain_extent
        extent = TerrainExtent(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)

    kwargs: Dict[str, Any] = {}
    if env.terrain_max_height is not None:
        kwargs["max_height"] = env.terrain_max_height

    path = env.terrain_path
    if name == "flat":
        return make_terrain("flat", altitude=env.terrain_flat_alt_m, extent=extent)
    if name == "hilly":
        return make_terrain("hilly", extent=extent or TerrainExtent(0.0, 1.0, 0.0, 1.0), **kwargs)
    if name in ("numpy", "image", "geotiff", "srtm_dir", "srtm_files"):
        if path is None:
            raise ValueError(f"{env.terrain_model} 地形模型需要提供 terrain_path")
        return make_terrain(name, path=path, extent=extent, **kwargs)

    raise ValueError(f"未知地形模型: {env.terrain_model}")


def _resolve_dynamics_context(cfg: SimConfig) -> DynamicContext:
    """由 ``SimConfig`` 构建动力学上下文。"""
    atm = make_atmosphere(
        cfg.environment.atmosphere,
        delta_t=cfg.environment.delta_t,
        density_factor=cfg.environment.density_factor,
    )
    wind = _resolve_wind(cfg)
    terrain = _resolve_terrain(cfg)
    if cfg.mission == "projectile":
        aero = make_aero("g1", cl_slope=0.0)
    elif cfg.mission in ("rocket", "icbm", "missile", "suborbital"):
        aero = make_aero("rocket")
    else:
        aero = make_aero("constant", cd=cfg.vehicle.cd)
    return DynamicContext(
        cfg=cfg,
        atmosphere=atm,
        wind=wind,
        aero=aero,
        terrain=terrain,
        gravity_model=cfg.environment.gravity_model,
        options={},
        use_cache=cfg.options.use_cache,
    )
