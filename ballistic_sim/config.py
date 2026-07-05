"""统一仿真配置层 (pydantic BaseModel)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union

import yaml
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


class StageConfig(BaseModel):
    """单级火箭/导弹 stage 参数。

    用于 ``VehicleConfig.stages`` 列表；保留 ``extra="allow"`` 以兼容
    后续阶段可能加入的气动、分离阻尼等扩展字段。
    """

    model_config = ConfigDict(extra="allow")

    name: Optional[str] = Field(default=None, description="级名称")
    thrust_sl: float = Field(..., ge=0, description="海平面推力 (N)")
    thrust_vac: float = Field(..., ge=0, description="真空推力 (N)")
    isp_vac: float = Field(..., gt=0, description="真空比冲 (s)")
    m_prop: float = Field(..., ge=0, description="推进剂质量 (kg)")
    m_dry: float = Field(..., ge=0, description="级干质量 (kg)")
    diameter_m: Optional[float] = Field(default=None, gt=0, description="级直径 (m)")
    Aref: Optional[float] = Field(default=None, gt=0, description="参考面积 (m²)")
    burn_time: Optional[float] = Field(default=None, gt=0, description="额定工作时间 (s)")


class VehicleConfig(BaseModel):
    """飞行器装配参数。

    阶段 0 采用通用字段占位；后续按 ``mission`` 自动选择弹丸/导弹/火箭装配模式。
    若提供 ``stages``，builder 将按多级质量记账自动构造 phase 链。
    """

    model_config = ConfigDict(extra="forbid")

    mass_kg: float = Field(default=1.0, gt=0, description="飞行器质量 (kg)")
    diameter_m: float = Field(default=0.1, gt=0, description="参考直径 (m)")
    cd: float = Field(default=0.3, ge=0, description="阻力系数")
    area_ref_m2: Optional[float] = Field(default=None, gt=0, description="参考面积 (m²)")
    thrust_N: Optional[float] = Field(default=None, ge=0, description="推力 (N)")
    burn_time_s: Optional[float] = Field(default=None, gt=0, description="发动机工作时间 (s)")
    stages: Optional[List[StageConfig]] = Field(
        default=None, description="多级火箭 stage 列表"
    )
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
    wind_model: Literal[
        "uniform", "log", "power", "profile", "composite", "dryden", "none"
    ] = Field(default="uniform", description="风场模型类型")
    wind_m_s: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="恒定风矢量 [E, N, U] (m/s)",
    )
    wind_profile_path: Optional[str] = Field(default=None, description="风场廓线文件路径")
    wind_profile_text: Optional[str] = Field(default=None, description="风场廓线文本")
    wind_u_ref: float = Field(default=10.0, description="参考风速 (m/s)")
    wind_z_ref: float = Field(default=10.0, gt=0, description="参考高度 (m)")
    wind_z0: float = Field(default=0.03, gt=0, description="对数风粗糙长度 (m)")
    wind_alpha: float = Field(default=0.14, description="幂律风指数")
    wind_direction_deg: float = Field(default=0.0, description="风向 (deg，自北顺时针)")
    wind_dryden_seed: Optional[int] = Field(default=None, description="Dryden 阵风随机种子")
    gravity_model: str = Field(
        default="wgs84",
        pattern=r"^(wgs84|j2|point)$",
        description="引力模型",
    )
    use_terrain: bool = Field(default=False, description="是否启用地形（兼容开关）")
    terrain_model: Literal[
        "null", "hilly", "numpy", "image", "geotiff", "srtm_dir", "srtm_files"
    ] = Field(default="null", description="地形模型类型")
    terrain_path: Optional[str] = Field(default=None, description="地形文件或目录路径")
    terrain_extent: Optional[Tuple[float, float, float, float]] = Field(
        default=None,
        description="地形范围 (lat_min, lat_max, lon_min, lon_max)",
    )
    terrain_max_height: float = Field(default=1000.0, ge=0, description="地形最大高程 (m)")
    delta_t: float = Field(default=0.0, description="大气温度偏差 (K)")
    density_factor: float = Field(default=1.0, gt=0, description="大气密度修正因子")

    @property
    def terrain_enabled(self) -> bool:
        """是否启用地形：兼容 ``use_terrain`` 或 ``terrain_model != "null"``。"""
        return self.use_terrain or self.terrain_model.lower() not in ("null", "none")


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
    use_cache: bool = Field(default=True, description="是否启用动力学上下文模型缓存")
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


def load_config(path: Union[str, Path]) -> SimConfig:
    """从 YAML/JSON 文件加载 ``SimConfig``（按扩展名自动识别）。"""
    p = Path(path)
    suffix = p.suffix.lower()
    with open(p, "r", encoding="utf-8") as f:
        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        elif suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"不支持的配置格式: {suffix}（仅支持 .yaml/.yml/.json）")
    if not isinstance(data, dict):
        raise ValueError("配置文件顶层必须是字典")
    return SimConfig(**data)


def save_config(cfg: SimConfig, path: Union[str, Path]) -> None:
    """将 ``SimConfig`` 保存为 YAML/JSON（按扩展名自动识别）。"""
    p = Path(path)
    suffix = p.suffix.lower()
    data = cfg.model_dump(mode="json")
    with open(p, "w", encoding="utf-8") as f:
        if suffix in (".yaml", ".yml"):
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        elif suffix == ".json":
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的配置格式: {suffix}（仅支持 .yaml/.yml/.json）")


@dataclass
class ValidationIssue:
    """跨字段业务校验问题。"""

    severity: Literal["ERROR", "WARNING"]
    path: str
    message: str


def validate_config(cfg: SimConfig) -> List[ValidationIssue]:
    """对 ``SimConfig`` 执行业务级跨字段校验。"""
    issues: List[ValidationIssue] = []
    g0 = 9.80665
    powered_missions = ("rocket", "icbm", "missile", "suborbital")

    # 1. 发射仰角业务语义（pydantic 已做越界校验）
    if cfg.launch.elevation_deg < 0 and cfg.mission in ("projectile", "missile"):
        issues.append(
            ValidationIssue(
                severity="WARNING",
                path="launch.elevation_deg",
                message="大气层内任务采用负仰角可能导致初始下坠",
            )
        )
    if cfg.launch.elevation_deg > 85.0:
        issues.append(
            ValidationIssue(
                severity="WARNING",
                path="launch.elevation_deg",
                message="仰角过大，可能导致上升段阻力损失显著或积分困难",
            )
        )
    if cfg.launch.elevation_deg < 0.0:
        issues.append(
            ValidationIssue(
                severity="WARNING",
                path="launch.elevation_deg",
                message="负仰角发射，通常仅用于特殊机动场景",
            )
        )

    # 2. 推力 / 工作时间必须成对出现
    thrust_N = cfg.vehicle.thrust_N
    burn_time_s = cfg.vehicle.burn_time_s
    has_thrust = thrust_N is not None
    has_burn = burn_time_s is not None
    if has_thrust != has_burn and cfg.vehicle.stages is None:
        issues.append(
            ValidationIssue(
                severity="WARNING",
                path="vehicle",
                message="thrust_N 与 burn_time_s 应同时设置或同时缺失",
            )
        )
    if (
        has_thrust
        and has_burn
        and thrust_N == 0.0
        and burn_time_s is not None
        and burn_time_s > 0.0
    ):
        issues.append(
            ValidationIssue(
                severity="ERROR",
                path="vehicle.thrust_N",
                message="推力为零但 burn_time_s > 0，动力段定义矛盾",
            )
        )

    # 3. 推进剂质量与分离质量非负
    if cfg.vehicle.burn_time_s is not None and cfg.vehicle.burn_time_s <= 0.0:
        issues.append(
            ValidationIssue(
                severity="ERROR",
                path="vehicle.burn_time_s",
                message="发动机工作时间必须大于 0",
            )
        )

    # 4. 动力任务 T/W > 1
    if cfg.mission in powered_missions:
        if cfg.vehicle.stages is not None:
            if not cfg.vehicle.stages:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        path="vehicle.stages",
                        message="动力任务 stages 列表不能为空",
                    )
                )
            else:
                liftoff_mass = cfg.vehicle.mass_kg
                tw0 = cfg.vehicle.stages[0].thrust_sl / (liftoff_mass * g0)
                if tw0 <= 1.0:
                    issues.append(
                        ValidationIssue(
                            severity="ERROR",
                            path="vehicle.stages[0].thrust_sl",
                            message=f"起飞推力/重量比 T/W = {tw0:.3f} <= 1，无法离地",
                        )
                    )
        elif has_thrust and cfg.vehicle.thrust_N is not None:
            tw = cfg.vehicle.thrust_N / (cfg.vehicle.mass_kg * g0)
            if tw <= 1.0:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        path="vehicle.thrust_N",
                        message=f"推力/重量比 T/W = {tw:.3f} <= 1，动力任务无法起飞",
                    )
                )
        else:
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    path="vehicle.thrust_N",
                    message="未设置推力或 stages，无法校验 T/W",
                )
            )

    # 5. 多级 stage 自洽性
    if cfg.vehicle.stages is not None and cfg.vehicle.stages:
        total_dry = sum(s.m_dry for s in cfg.vehicle.stages)
        total_prop = sum(s.m_prop for s in cfg.vehicle.stages)
        if total_prop + total_dry > cfg.vehicle.mass_kg:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path="vehicle.mass_kg",
                    message="各级干质量与推进剂之和超过飞行器总质量",
                )
            )
        for i, s in enumerate(cfg.vehicle.stages):
            prefix = f"vehicle.stages[{i}]"
            if s.m_prop < 0.0:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        path=f"{prefix}.m_prop",
                        message="推进剂质量不能为负",
                    )
                )
            if s.m_dry < 0.0:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        path=f"{prefix}.m_dry",
                        message="分离质量/级干质量不能为负",
                    )
                )
            if s.thrust_vac <= 0.0:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        path=f"{prefix}.thrust_vac",
                        message="真空推力必须为正",
                    )
                )
            if s.isp_vac <= 0.0:
                issues.append(
                    ValidationIssue(
                        severity="ERROR",
                        path=f"{prefix}.isp_vac",
                        message="真空比冲必须为正",
                    )
                )
            if s.thrust_sl > s.thrust_vac:
                issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        path=f"{prefix}.thrust_sl",
                        message="海平面推力大于真空推力，与实际发动机特性不符",
                    )
                )

    # 6. 目标经纬度与任务类型匹配
    if cfg.mission in ("missile", "icbm"):
        if cfg.guidance.target_lat_deg is None or cfg.guidance.target_lon_deg is None:
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    path="guidance",
                    message="missile/icbm 任务建议设置目标经纬度以进行落点评估",
                )
            )
    if cfg.mission == "projectile":
        if cfg.guidance.target_lat_deg is not None or cfg.guidance.target_lon_deg is not None:
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    path="guidance",
                    message="projectile 任务通常无需目标经纬度",
                )
            )

    # 7. Monte Carlo 参数合理性
    if cfg.options.monte_carlo is not None:
        mc = cfg.options.monte_carlo
        if mc.num_samples <= 0:
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path="options.monte_carlo.num_samples",
                    message="Monte Carlo 样本数必须大于 0",
                )
            )
        if mc.backend not in ("auto", "process", "batch", "gpu"):
            issues.append(
                ValidationIssue(
                    severity="ERROR",
                    path="options.monte_carlo.backend",
                    message="backend 必须是 auto/process/batch/gpu 之一",
                )
            )
        if mc.num_samples < 10:
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    path="options.monte_carlo.num_samples",
                    message="样本数小于 10，统计意义有限",
                )
            )

    # 8. 制导律与任务类型粗略匹配
    if cfg.guidance.guidance_law == "proportional" and cfg.mission not in ("missile",):
        issues.append(
            ValidationIssue(
                severity="WARNING",
                path="guidance.guidance_law",
                message="proportional 制导律通常用于导弹末端拦截任务",
            )
        )

    return issues
