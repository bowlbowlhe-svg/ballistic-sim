"""Monte Carlo 散布与不确定性分析."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ballistic_sim.config import (
    DistributionConfig,
    PerturbationConfig,
    SimConfig,
    apply_overrides,
)
from ballistic_sim.dynamics.batch_mpm import BatchMPMModel
from ballistic_sim.dynamics.gpu_mpm import gpu_available
from ballistic_sim.dynamics.mpm import MPMOptions
from ballistic_sim.models.aerodynamics import (
    ConstantAeroModel,
    DragLaw,
    ProjectileAeroModel,
    RocketAeroModel,
)
from ballistic_sim.models.wind import UniformWind, WindModel
from ballistic_sim.phases.builder import build_phases
from ballistic_sim.simulator import SimResult, simulate

logger = logging.getLogger(__name__)


@dataclass
class Distribution:
    """参数扰动分布."""

    mean: float = 0.0
    std: float = 0.0
    low: Optional[float] = None
    high: Optional[float] = None

    def sample(self, rng: np.random.Generator) -> float:
        if self.std <= 0:
            return self.mean
        val = rng.normal(self.mean, self.std)
        if self.low is not None:
            val = max(val, self.low)
        if self.high is not None:
            val = min(val, self.high)
        return val

    def sample_array(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """批量采样."""
        if self.std <= 0:
            return np.full(size, self.mean, dtype=np.float64)
        vals = rng.normal(self.mean, self.std, size=size)
        if self.low is not None:
            vals = np.maximum(vals, self.low)
        if self.high is not None:
            vals = np.minimum(vals, self.high)
        return vals


@dataclass
class DispersionResult:
    """散布分析结果."""

    ranges: np.ndarray
    crosses: np.ndarray
    tofs: np.ndarray
    impact_angles: np.ndarray
    range_mean: float
    range_std: float
    cross_mean: float
    cross_std: float
    cep50: float
    cep90: float
    ellipse_major: float
    ellipse_minor: float
    ellipse_angle: float
    samples: int


def _config_to_perturbation(
    cfg_perturb: Optional[PerturbationConfig],
    perturb: Optional[PerturbationConfig],
) -> PerturbationConfig:
    """合并 SimConfig 中携带的扰动配置与函数参数。"""
    if perturb is not None:
        return perturb
    if cfg_perturb is not None:
        return cfg_perturb
    return PerturbationConfig()


def _dist_from_config(dc: DistributionConfig) -> Distribution:
    return Distribution(mean=dc.mean, std=dc.std, low=dc.low, high=dc.high)


def _extract_drag_table(aero: Any) -> np.ndarray:
    """从气动模型提取 [Ma, Cd] 阻力表供批量插值。"""
    if isinstance(aero, DragLaw):
        from scipy.interpolate import PchipInterpolator

        interp = aero.cd0_fn
        if isinstance(interp, PchipInterpolator):
            ma = np.linspace(float(interp.x[0]), float(interp.x[-1]), 501)
            return np.column_stack([ma, np.asarray(interp(ma), dtype=np.float64)])
        ma = np.linspace(0.0, 5.0, 501)
        return np.column_stack([ma, np.asarray(interp(ma), dtype=np.float64)])
    if isinstance(aero, ProjectileAeroModel):
        return _extract_drag_table(aero._drag_law)
    if isinstance(aero, RocketAeroModel):
        ma = np.linspace(float(aero._mach[0]), float(aero._mach[-1]), 501)
        return np.column_stack([ma, np.asarray([aero._cd_of_mach(m) for m in ma])])
    if isinstance(aero, ConstantAeroModel):
        cd = float(aero.drag_coefficient(0.0))
        return np.array([[0.0, cd], [5.0, cd]], dtype=np.float64)
    # 兜底：直接调用
    ma = np.linspace(0.0, 5.0, 501)
    return np.column_stack([ma, np.asarray([aero.drag_coefficient(m) for m in ma])])


def _build_dispersion_result(
    ranges: np.ndarray,
    crosses: np.ndarray,
    tofs: np.ndarray,
    impact_angles: np.ndarray,
    n_valid: int,
) -> DispersionResult:
    """由有效样本构建 DispersionResult."""
    mean_r = float(np.mean(ranges))
    mean_c = float(np.mean(crosses))
    distances = np.hypot(ranges - mean_r, crosses - mean_c)
    cep50 = float(np.percentile(distances, 50))
    cep90 = float(np.percentile(distances, 90))

    points = np.column_stack([ranges, crosses])
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    ellipse_major = float(2 * np.sqrt(eigvals[0]))
    ellipse_minor = float(2 * np.sqrt(eigvals[1]))
    ellipse_angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))

    return DispersionResult(
        ranges=ranges,
        crosses=crosses,
        tofs=tofs,
        impact_angles=impact_angles,
        range_mean=mean_r,
        range_std=float(np.std(ranges)),
        cross_mean=mean_c,
        cross_std=float(np.std(crosses)),
        cep50=cep50,
        cep90=cep90,
        ellipse_major=ellipse_major,
        ellipse_minor=ellipse_minor,
        ellipse_angle=ellipse_angle,
        samples=n_valid,
    )


def _resolve_wind(cfg: SimConfig) -> WindModel:
    from ballistic_sim.models.wind import make_wind

    w = cfg.environment.wind_m_s
    return make_wind("uniform", e=w[0], n=w[1], u=w[2])


def _can_use_batch(cfg: SimConfig) -> bool:
    """判断能否使用批量求解器."""
    if cfg.mission != "projectile":
        return False
    wind = _resolve_wind(cfg)
    if not isinstance(wind, UniformWind):
        return False
    if cfg.options.mpm_use_spin or cfg.options.mpm_use_dynamic_alpha:
        return False
    return True


def _perturb_cfg(cfg: SimConfig, perturb: PerturbationConfig, seed: int) -> SimConfig:
    """对 SimConfig 做一次扰动采样并返回新配置。"""
    rng = np.random.default_rng(seed)
    overrides: dict[str, Any] = {}

    if perturb.mass_kg.std > 0:
        overrides["vehicle.mass_kg"] = cfg.vehicle.mass_kg + _dist_from_config(
            perturb.mass_kg
        ).sample(rng)
    if perturb.form_factor.std > 0:
        overrides["vehicle.cd"] = cfg.vehicle.cd + _dist_from_config(perturb.form_factor).sample(
            rng
        )
    if perturb.v0_m_s.std > 0:
        overrides["launch.v0_m_s"] = cfg.launch.v0_m_s + _dist_from_config(perturb.v0_m_s).sample(
            rng
        )
    if perturb.elevation_deg.std > 0:
        overrides["launch.elevation_deg"] = cfg.launch.elevation_deg + _dist_from_config(
            perturb.elevation_deg
        ).sample(rng)
    if perturb.azimuth_deg.std > 0:
        overrides["launch.azimuth_deg"] = cfg.launch.azimuth_deg + _dist_from_config(
            perturb.azimuth_deg
        ).sample(rng)

    wind = cfg.environment.wind_m_s.copy()
    if perturb.wind_e.std > 0:
        wind[0] = wind[0] + _dist_from_config(perturb.wind_e).sample(rng)
    if perturb.wind_n.std > 0:
        wind[1] = wind[1] + _dist_from_config(perturb.wind_n).sample(rng)
    overrides["environment.wind_m_s"] = wind

    if perturb.delta_t.std > 0:
        overrides["environment.delta_t"] = cfg.environment.delta_t + _dist_from_config(
            perturb.delta_t
        ).sample(rng)
    if perturb.density_factor.std > 0:
        overrides["environment.density_factor"] = cfg.environment.density_factor * (
            1.0 + _dist_from_config(perturb.density_factor).sample(rng)
        )

    if not overrides:
        return cfg
    return apply_overrides(cfg, overrides)


def _run_single_process(
    cfg: SimConfig, seed: int, perturb: PerturbationConfig
) -> Optional[SimResult]:
    """process 后端：单样本仿真。"""
    try:
        cfg_p = _perturb_cfg(cfg, perturb, seed)
        phases = build_phases(cfg_p)
        result = simulate(cfg_p, phases=phases)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.debug("Monte Carlo 样本仿真失败 (seed=%s): %s", seed, exc)
        return None


def _extract_enu_impact(result: SimResult) -> Optional[tuple[float, float, float, float, float]]:
    """从 SimResult 提取落点 ENU 信息。"""
    if result.y.size == 0 or result.y.shape[1] < 6:
        return None
    e, n, u = result.y[-1, 0], result.y[-1, 1], result.y[-1, 2]
    if u > 1e-6:
        return None
    v = result.y[-1, 3:6]
    v_impact = float(np.linalg.norm(v))
    v_down = max(0.0, -v[2])
    v_horiz = float(np.linalg.norm(v[:2]))
    impact_angle = float(np.degrees(np.arctan2(v_down, v_horiz)))
    tof = float(result.t[-1])
    range_m = float(np.hypot(e, n))
    cross_m = float(e)
    return range_m, cross_m, tof, v_impact, impact_angle


def _monte_carlo_process(
    cfg: SimConfig,
    perturb: PerturbationConfig,
    n_samples: int,
    n_jobs: int,
    seed: int,
) -> DispersionResult:
    """多进程/串行 CPU Monte Carlo."""
    if n_jobs == 1:
        results = [_run_single_process(cfg, seed + i, perturb) for i in range(n_samples)]
    else:
        from concurrent.futures import ProcessPoolExecutor

        n_workers = n_jobs if n_jobs > 0 else None
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_run_single_process, cfg, seed + i, perturb)
                for i in range(n_samples)
            ]
            results = [f.result() for f in futures]

    valid = []
    for r in results:
        if r is None:
            continue
        impact = _extract_enu_impact(r)
        if impact is not None:
            valid.append(impact)

    if len(valid) < 10:
        raise RuntimeError(f"有效样本过少: {len(valid)}/{n_samples}")

    arr = np.array(valid, dtype=np.float64)
    return _build_dispersion_result(
        arr[:, 0],
        arr[:, 1],
        arr[:, 2],
        arr[:, 4],
        len(valid),
    )


def _monte_carlo_batch(
    cfg: SimConfig,
    perturb: PerturbationConfig,
    n_samples: int,
    seed: int,
    use_gpu: bool = False,
) -> DispersionResult:
    """向量化批量 Monte Carlo."""
    rng = np.random.default_rng(seed)

    base_mass = cfg.vehicle.mass_kg
    base_cd = cfg.vehicle.cd or 1.0
    base_v0 = cfg.launch.v0_m_s
    base_theta = cfg.launch.elevation_deg
    base_az = cfg.launch.azimuth_deg
    base_delta_t = cfg.environment.delta_t
    base_density = cfg.environment.density_factor
    wind = _resolve_wind(cfg)
    base_wind_e = wind.e if isinstance(wind, UniformWind) else 0.0
    base_wind_n = wind.n if isinstance(wind, UniformWind) else 0.0
    base_wind_u = wind.u if isinstance(wind, UniformWind) else 0.0

    mass = base_mass + _dist_from_config(perturb.mass_kg).sample_array(rng, n_samples)
    form_factor = base_cd + _dist_from_config(perturb.form_factor).sample_array(rng, n_samples)
    v0 = base_v0 + _dist_from_config(perturb.v0_m_s).sample_array(rng, n_samples)
    theta = base_theta + _dist_from_config(perturb.elevation_deg).sample_array(rng, n_samples)
    az = base_az + _dist_from_config(perturb.azimuth_deg).sample_array(rng, n_samples)
    delta_t = base_delta_t + _dist_from_config(perturb.delta_t).sample_array(rng, n_samples)
    density_factor = base_density * (
        1.0 + _dist_from_config(perturb.density_factor).sample_array(rng, n_samples)
    )
    wind_e = base_wind_e + _dist_from_config(perturb.wind_e).sample_array(rng, n_samples)
    wind_n = base_wind_n + _dist_from_config(perturb.wind_n).sample_array(rng, n_samples)

    opt = MPMOptions(
        use_drag=True,
        use_wind=bool(cfg.environment.wind_m_s),
        use_coriolis=True,
        use_spin=False,
        use_dynamic_alpha=False,
    )

    from ballistic_sim.models.aerodynamics import make_aero

    aero = make_aero("g1", cl_slope=0.0)
    drag_table = _extract_drag_table(aero)

    model: BatchMPMModel
    if use_gpu:
        from ballistic_sim.dynamics.gpu_mpm import GPUBatchMPMModel

        model = GPUBatchMPMModel(
            mass_kg=mass,
            diameter_m=np.full(n_samples, cfg.vehicle.diameter_m),
            form_factor=form_factor,
            v0=v0,
            theta_deg=theta,
            az_deg=az,
            delta_t=delta_t,
            density_factor=density_factor,
            wind_e=wind_e,
            wind_n=wind_n,
            wind_u=np.full(n_samples, base_wind_u),
            lat_deg=cfg.launch.lat_deg,
            h0=cfg.launch.alt_m,
            azimuth_deg=cfg.launch.azimuth_deg,
            drag_table=drag_table,
            options=opt,
        )
    else:
        model = BatchMPMModel(
            mass_kg=mass,
            diameter_m=np.full(n_samples, cfg.vehicle.diameter_m),
            form_factor=form_factor,
            v0=v0,
            theta_deg=theta,
            az_deg=az,
            delta_t=delta_t,
            density_factor=density_factor,
            wind_e=wind_e,
            wind_n=wind_n,
            wind_u=np.full(n_samples, base_wind_u),
            lat_deg=cfg.launch.lat_deg,
            h0=cfg.launch.alt_m,
            azimuth_deg=cfg.launch.azimuth_deg,
            drag_table=drag_table,
            options=opt,
        )

    res = model.simulate()
    valid_mask = res.landed & np.isfinite(res.range_m)
    if int(valid_mask.sum()) < 10:
        raise RuntimeError(f"有效样本过少: {int(valid_mask.sum())}/{n_samples}")
    return _build_dispersion_result(
        res.range_m[valid_mask],
        res.cross_m[valid_mask],
        res.tof[valid_mask],
        res.impact_angle[valid_mask],
        int(valid_mask.sum()),
    )


def monte_carlo_simulation(
    cfg: SimConfig,
    perturb: Optional[PerturbationConfig] = None,
    n_samples: int = 100,
    backend: str = "auto",
    n_jobs: int = -1,
    seed: int = 42,
) -> DispersionResult:
    """Monte Carlo 散布仿真.

    Parameters
    ----------
    cfg:
        仿真配置。
    perturb:
        参数扰动配置；若未提供，使用 ``cfg.options.monte_carlo.perturbations``
        或默认零扰动。
    n_samples:
        样本数。
    backend:
        计算后端，可选 ``auto`` / ``process`` / ``batch`` / ``gpu``。
        auto 策略：在可用前提下优先 gpu -> batch -> process。
    n_jobs:
        ``process`` 后端并行数，-1 表示自动，1 表示串行。
    seed:
        随机种子。

    .. note:: 后端物理保真度差异
        ``process`` 后端用统一仿真器（与单发仿真同一物理）。
        ``batch``/``gpu`` 后端为向量化**重新实现**，物理被简化：仅标准 ISA 大气
        （无自定义廓线/湿度）、**忽略自转偏流与动态攻角**、固定步长 RK4。仅在
        projectile 任务、UniformWind、且 MPM 未启用 spin/dynamic_alpha 时启用。
        在该适用域下两者射程/偏流差异 < 1%。
    """
    if backend not in ("auto", "process", "batch", "gpu"):
        raise ValueError(f"未知的 backend: {backend}")

    cfg_perturb = None
    if cfg.options.monte_carlo is not None:
        cfg_perturb = cfg.options.monte_carlo.perturbations
    perturb = _config_to_perturbation(cfg_perturb, perturb)

    use_batch = _can_use_batch(cfg)

    if backend == "gpu":
        if not use_batch:
            raise ValueError("当前参数配置不支持 GPU 后端（需要 projectile + UniformWind）")
        if not gpu_available():
            raise RuntimeError("CuPy/GPU 不可用")
        return _monte_carlo_batch(cfg, perturb, n_samples, seed, use_gpu=True)

    if backend == "batch":
        if not use_batch:
            raise ValueError("当前参数配置不支持 batch 后端（需要 projectile + UniformWind）")
        return _monte_carlo_batch(cfg, perturb, n_samples, seed, use_gpu=False)

    if backend == "process":
        return _monte_carlo_process(cfg, perturb, n_samples, n_jobs, seed)

    # auto
    if use_batch:
        if gpu_available():
            return _monte_carlo_batch(cfg, perturb, n_samples, seed, use_gpu=True)
        return _monte_carlo_batch(cfg, perturb, n_samples, seed, use_gpu=False)
    return _monte_carlo_process(cfg, perturb, n_samples, n_jobs, seed)


# 保持灵敏度分析接口


def sensitivity_analysis(
    cfg: SimConfig,
    perturb: PerturbationConfig,
    param_name: str,
    n_samples: int = 100,
    seed: int = 42,
) -> dict:
    """对单个参数做灵敏度分析."""
    dist = getattr(perturb, param_name, DistributionConfig())
    if dist.std <= 0:
        raise ValueError(f"参数 {param_name} 的标准差必须大于 0")

    perturb_copy = PerturbationConfig()
    for key in vars(perturb_copy):
        if key == param_name:
            setattr(perturb_copy, key, dist)
        else:
            setattr(perturb_copy, key, DistributionConfig(std=0.0))

    result = monte_carlo_simulation(
        cfg,
        perturb=perturb_copy,
        n_samples=n_samples,
        backend="auto",
        seed=seed,
    )
    return {
        "param": param_name,
        "range_std": result.range_std,
        "cross_std": result.cross_std,
        "range_sensitivity": result.range_std / max(dist.std, 1e-12),
        "cross_sensitivity": result.cross_std / max(dist.std, 1e-12),
    }
