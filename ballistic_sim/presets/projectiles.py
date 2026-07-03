"""弹丸/火炮预设。"""

from __future__ import annotations

from ballistic_sim.config import (
    EnvironmentConfig,
    GuidanceConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    VehicleConfig,
)


def m107_config() -> SimConfig:
    """M107 155mm 榴弹典型配置。"""
    return SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(
            mass_kg=43.2,
            diameter_m=0.155,
            cd=0.78,
            area_ref_m2=None,
        ),
        launch=LaunchConfig(
            lat_deg=39.9,
            lon_deg=0.0,
            alt_m=0.0,
            azimuth_deg=90.0,
            elevation_deg=45.0,
            v0_m_s=684.0,
            t0_s=0.0,
        ),
        environment=EnvironmentConfig(
            atmosphere="isa",
            wind_m_s=[0.0, 0.0, 0.0],
            gravity_model="wgs84",
        ),
        guidance=GuidanceConfig(),
        options=OptionsConfig(
            integrator="DOP853",
            rtol=1e-6,
            atol=1e-9,
            max_step=1.0,
            terminate_impact=True,
        ),
    )
