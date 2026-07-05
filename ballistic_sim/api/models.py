"""API request/response pydantic models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ballistic_sim.config import (
    EnvironmentConfig,
    GuidanceConfig,
    LaunchConfig,
    OptionsConfig,
    PerturbationConfig,
    VehicleConfig,
)


class SimulateRequest(BaseModel):
    """``POST /simulate/{mission}`` 请求体。

    各字段与 :class:`ballistic_sim.config.SimConfig` 的子配置对应；
    未提供的段将使用任务预设默认值。
    """

    vehicle: Optional[VehicleConfig] = None
    launch: Optional[LaunchConfig] = None
    environment: Optional[EnvironmentConfig] = None
    guidance: Optional[GuidanceConfig] = None
    options: Optional[OptionsConfig] = None
    preset: Optional[str] = Field(default=None, description="预设名（projectile/missile/rocket）")
    include_trajectory: bool = Field(default=True, description="是否在响应中包含轨迹")


class SimulateResponse(BaseModel):
    """``POST /simulate/{mission}`` 响应体。"""

    mission: str
    stop_reason: str
    t_end_s: Optional[float] = None
    range_m: Optional[float] = None
    apogee_m: Optional[float] = None
    impact_lat: Optional[float] = None
    impact_lon: Optional[float] = None
    summary: Dict[str, Any]
    trajectory: Optional[Dict[str, List[float]]] = None


class MonteCarloRequest(BaseModel):
    """``POST /simulate/monte-carlo`` 请求体。"""

    mission: str = Field(
        default="projectile",
        pattern=r"^(projectile|missile|rocket|icbm|suborbital)$",
        description="任务类型",
    )
    vehicle: Optional[VehicleConfig] = None
    launch: Optional[LaunchConfig] = None
    environment: Optional[EnvironmentConfig] = None
    guidance: Optional[GuidanceConfig] = None
    options: Optional[OptionsConfig] = None
    preset: Optional[str] = Field(default=None, description="任务预设名")
    perturbations: Optional[PerturbationConfig] = None
    n_samples: int = Field(default=100, gt=0, description="Monte Carlo 样本数")
    backend: str = Field(
        default="auto",
        pattern=r"^(auto|process|batch|gpu)$",
        description="计算后端 auto/process/batch/gpu",
    )
    n_jobs: int = Field(default=-1, description="process 后端并行数，-1 为自动")
    seed: int = Field(default=42, description="随机种子")


class MonteCarloResponse(BaseModel):
    """``POST /simulate/monte-carlo`` 响应体。"""

    samples: int
    range_mean_m: float
    range_std_m: float
    cross_mean_m: float
    cross_std_m: float
    cep50_m: float
    cep90_m: float
    ellipse_major_m: float
    ellipse_minor_m: float
    ellipse_angle_deg: float


class FireControlRequest(BaseModel):
    """``POST /firecontrol/solve`` 请求体。"""

    mission: str = Field(
        default="projectile",
        pattern=r"^(projectile|missile|rocket|icbm|suborbital)$",
        description="任务类型",
    )
    vehicle: Optional[VehicleConfig] = None
    launch: Optional[LaunchConfig] = None
    environment: Optional[EnvironmentConfig] = None
    guidance: Optional[GuidanceConfig] = None
    options: Optional[OptionsConfig] = None
    preset: Optional[str] = Field(default=None, description="任务预设名")
    target_lat_deg: float = Field(..., ge=-90, le=90, description="目标纬度 (deg)")
    target_lon_deg: float = Field(..., ge=-180, le=180, description="目标经度 (deg)")


class FireControlResponse(BaseModel):
    """``POST /firecontrol/solve`` 响应体。"""

    mission: str
    qe_deg: Optional[float] = None
    az_deg: Optional[float] = None
    converged: bool
    target_lat_deg: float
    target_lon_deg: float
    target_range_m: Optional[float] = None
    bearing_deg: Optional[float] = None
    residual_range_m: Optional[float] = None
    residual_cross_m: Optional[float] = None
    iterations: Optional[int] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """``GET /health`` 响应体。"""

    status: str
    version: str
