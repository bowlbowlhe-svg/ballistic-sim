"""统一仿真配置层 (pydantic BaseModel)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DistributionConfig(BaseModel):
    """参数扰动分布配置。"""

    model_config = ConfigDict(extra="forbid")

    mean: float = Field(default=0.0, description="均值")
    std: float = Field(default=0.0, ge=0, description="标准差")
    low: Optional[float] = Field(default=None, description="下界（含）")
    high: Optional[float] = Field(default=None, description="上界（含）")


class PerturbationConfig(BaseModel):
    """Monte Carlo 输入参数扰动配置。"""

    model_config = ConfigDict(extra="forbid")

    mass_kg: DistributionConfig = Field(
        default_factory=lambda: DistributionConfig(std=0.0), description="质量扰动 (kg)"
    )
    form_factor: DistributionConfig = Field(
        default_factory=lambda: DistributionConfig(std=0.0), description="阻力系数/形制系数扰动"
    )
    v0_m_s: DistributionConfig = Field(
        default_factory=lambda: DistributionConfig(std=2.0), description="初速扰动 (m/s)"
    )
    elevation_deg: DistributionConfig = Field(
        default_factory=lambda: DistributionConfig(std=0.05), description="发射仰角扰动 (deg)"
    )
    azimuth_deg: DistributionConfig = Field(
        default_factory=lambda: DistributionConfig(std=0.05), description="发射方位角扰动 (deg)"
    )
    delta_t: DistributionConfig = Field(
        default_factory=lambda: DistributionConfig(std=2.0), description="大气温度偏差扰动 (K)"
    )
    density_factor: DistributionConfig = Field(
        default_factory=lambda: DistributionConfig(std=0.02),
        description="大气密度修正因子扰动 (相对值)",
    )
    wind_e: DistributionConfig = Field(
        default_factory=lambda: DistributionConfig(std=1.0), description="东风分量扰动 (m/s)"
    )
    wind_n: DistributionConfig = Field(
        default_factory=lambda: DistributionConfig(std=1.0), description="北风分量扰动 (m/s)"
    )


class MonteCarloConfig(BaseModel):
    """Monte Carlo 仿真选项。"""

    model_config = ConfigDict(extra="forbid")

    num_samples: int = Field(default=100, gt=0, description="样本数")
    backend: str = Field(
        default="auto",
        pattern=r"^(auto|process|batch|gpu)$",
        description="计算后端：auto/process/batch/gpu",
    )
    n_jobs: int = Field(default=-1, description="process 后端并行数，-1 表示自动")
    seed: int = Field(default=42, description="随机种子")
    perturbations: Optional[PerturbationConfig] = Field(default=None, description="参数扰动配置")


class VehicleConfig(BaseModel):
    """飞行器装配参数。

    阶段 0 采用通用字段占位；后续按 ``mission`` 自动选择弹丸/导弹/火箭装配模式。
    """

    model_config = ConfigDict(extra="forbid")

    mass_kg: float = Field(default=1.0, gt=0, description="飞行器质量 (kg)")
    diameter_m: float = Field(default=0.1, gt=0, description="参考直径 (m)")
    cd: float = Field(default=0.3, ge=0, description="阻力系数")
    area_ref_m2: Optional[float] = Field(default=None, gt=0, description="参考面积 (m²)")
    thrust_N: Optional[float] = Field(default=None, ge=0, description="推力 (N)")
    burn_time_s: Optional[float] = Field(default=None, gt=0, description="发动机工作时间 (s)")
    # 6-DOF 可选参数
    Ix: Optional[float] = Field(default=None, gt=0, description="轴向转动惯量 (kg·m²)")
    It: Optional[float] = Field(default=None, gt=0, description="横向转动惯量 (kg·m²)")
    x_cp_cg: Optional[float] = Field(
        default=None, description="压心距质心距离 (m)，正表示压心在质心前方"
    )
    twist_cal: Optional[float] = Field(default=None, gt=0, description="缠距 (caliber)")


class LaunchConfig(BaseModel):
    """发射/初始条件。"""

    model_config = ConfigDict(extra="forbid")

    lat_deg: float = Field(default=0.0, ge=-90, le=90, description="发射点纬度 (deg)")
    lon_deg: float = Field(default=0.0, ge=-180, le=180, description="发射点经度 (deg)")
    alt_m: float = Field(default=0.0, description="发射点海拔 (m)")
    azimuth_deg: float = Field(default=0.0, ge=0, le=360, description="发射方位角 (deg)")
    elevation_deg: float = Field(default=45.0, ge=-90, le=90, description="发射仰角 (deg)")
    v0_m_s: float = Field(default=0.0, ge=0, description="初始速度 (m/s)")
    t0_s: float = Field(default=0.0, description="起始时刻 (s)")


class EnvironmentConfig(BaseModel):
    """环境模型选项。"""

    model_config = ConfigDict(extra="forbid")

    atmosphere: str = Field(
        default="isa",
        pattern=r"^(isa|us76|none|gfs)$",
        description="大气模型",
    )
    wind_m_s: List[float] = Field(
        default=[0.0, 0.0, 0.0],
        description="恒定风矢量 [E, N, U] (m/s)",
    )
    gravity_model: str = Field(
        default="wgs84",
        pattern=r"^(wgs84|j2|point)$",
        description="引力模型",
    )
    use_terrain: bool = Field(default=False, description="是否启用地形")
    terrain_path: Optional[str] = Field(default=None, description="地形数据路径")
    delta_t: float = Field(default=0.0, description="大气温度偏差 (K)")
    density_factor: float = Field(default=1.0, gt=0, description="大气密度修正因子")


class GuidanceConfig(BaseModel):
    """制导律参数占位。"""

    model_config = ConfigDict(extra="forbid")

    guidance_law: str = Field(
        default="none",
        pattern=r"^(none|proportional|peg|energy)$",
        description="制导律类型",
    )
    kick_deg: float = Field(default=0.0, ge=-90, le=90, description="程序转弯角 (deg)")
    nav_constant: float = Field(default=3.0, gt=0, description="比例导引系数")
    target_lat_deg: Optional[float] = Field(default=None, ge=-90, le=90, description="目标纬度")
    target_lon_deg: Optional[float] = Field(default=None, ge=-180, le=180, description="目标经度")


class OptionsConfig(BaseModel):
    """积分器、事件与输出选项。"""

    model_config = ConfigDict(extra="forbid")

    integrator: str = Field(default="RK45", description="scipy.integrate.solve_ivp method")
    rtol: float = Field(default=1e-6, gt=0, description="相对容差")
    atol: float = Field(default=1e-9, gt=0, description="绝对容差")
    max_step: float = Field(default=1.0, gt=0, description="最大积分步长 (s)")
    terminate_apogee: bool = Field(default=True, description="到达远地点停止")
    terminate_impact: bool = Field(default=True, description="触地停止")
    save_trajectory: bool = Field(default=True, description="保存轨迹")
    output_dir: Optional[str] = Field(default=None, description="输出目录")
    verbose: bool = Field(default=False, description="是否打印详细信息")
    sixdof_reentry: bool = Field(default=False, description="再入段使用 6-DOF 高保真动力学")
    mpm_use_spin: bool = Field(default=True, description="MPM 是否启用自转偏流")
    mpm_use_dynamic_alpha: bool = Field(default=False, description="MPM 是否启用动态攻角")
    monte_carlo: Optional[MonteCarloConfig] = Field(default=None, description="Monte Carlo 选项")


class SimConfig(BaseModel):
    """统一的弹道仿真配置入口。"""

    model_config = ConfigDict(extra="forbid")

    mission: str = Field(
        ...,
        pattern=r"^(projectile|missile|rocket|icbm|suborbital)$",
        description="任务类型",
    )
    vehicle: VehicleConfig = Field(default_factory=VehicleConfig)
    launch: LaunchConfig = Field(default_factory=LaunchConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    guidance: GuidanceConfig = Field(default_factory=GuidanceConfig)
    options: OptionsConfig = Field(default_factory=OptionsConfig)


# ==============================================================================
# 配置覆盖与业务校验骨架
# ==============================================================================


def _nested_set(data: dict, path: str, value: Any) -> None:
    """按点路径 ``path`` 将 ``value`` 写入嵌套 dict。"""
    keys = path.split(".")
    for key in keys[:-1]:
        if key not in data or not isinstance(data[key], dict):
            data[key] = {}
        data = data[key]
    data[keys[-1]] = value


def apply_overrides(cfg: SimConfig, overrides: dict) -> SimConfig:
    """返回被 ``overrides`` 覆盖后的新 ``SimConfig``。

    ``overrides`` 支持点路径，例如 ``{"guidance.kick_deg": 15.0}``。
    """
    cfg_dict = cfg.model_dump(mode="json")
    for path, value in overrides.items():
        _nested_set(cfg_dict, path, value)
    return SimConfig(**cfg_dict)


@dataclass
class ValidationIssue:
    """跨字段业务校验问题。"""

    severity: Literal["ERROR", "WARNING"]
    path: str
    message: str


def validate_config(cfg: SimConfig) -> List[ValidationIssue]:
    """对 ``SimConfig`` 执行业务级跨字段校验。

    阶段 0 仅保留骨架：目前只返回简单的自洽性检查，后续阶段逐步完善
    （如 T/W > 1、推进剂守恒、目标可达性等）。
    """
    issues: List[ValidationIssue] = []

    if cfg.launch.elevation_deg < 0 and cfg.mission in ("projectile", "missile"):
        issues.append(
            ValidationIssue(
                severity="WARNING",
                path="launch.elevation_deg",
                message="大气层内任务采用负仰角可能导致初始下坠",
            )
        )

    if cfg.vehicle.thrust_N is not None and cfg.vehicle.burn_time_s is None:
        issues.append(
            ValidationIssue(
                severity="WARNING",
                path="vehicle.burn_time_s",
                message="设置了推力但未设置 burn_time_s，动力段时长未知",
            )
        )

    return issues
