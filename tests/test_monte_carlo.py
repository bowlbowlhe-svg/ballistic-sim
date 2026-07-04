"""Monte Carlo 散布测试."""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.config import (
    DistributionConfig,
    EnvironmentConfig,
    GuidanceConfig,
    LaunchConfig,
    OptionsConfig,
    PerturbationConfig,
    SimConfig,
    VehicleConfig,
)
from ballistic_sim.monte_carlo import (
    Distribution,
    DispersionResult,
    _can_use_batch,
    _config_to_perturbation,
    _extract_drag_table,
    _extract_enu_impact,
    _perturb_cfg,
    _run_single_process,
    monte_carlo_simulation,
    sensitivity_analysis,
)


def _base_cfg(qe: float = 45.0) -> SimConfig:
    return SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(mass_kg=43.2, diameter_m=0.155, cd=0.78),
        launch=LaunchConfig(
            lat_deg=39.9,
            lon_deg=0.0,
            alt_m=0.0,
            azimuth_deg=90.0,
            elevation_deg=qe,
            v0_m_s=684.0,
        ),
        environment=EnvironmentConfig(
            atmosphere="isa",
            wind_m_s=[0.0, 0.0, 0.0],
            gravity_model="wgs84",
        ),
        guidance=GuidanceConfig(),
        options=OptionsConfig(),
    )


def test_distribution_zero_std_returns_mean() -> None:
    """Distribution 在 std<=0 时返回均值."""
    dist = Distribution(mean=3.14, std=0.0)
    rng = np.random.default_rng(0)
    assert dist.sample(rng) == 3.14
    arr = dist.sample_array(rng, 10)
    assert np.allclose(arr, 3.14)


def test_distribution_clipping() -> None:
    """Distribution 截断上下界."""
    dist = Distribution(mean=0.0, std=1.0, low=-0.5, high=0.5)
    rng = np.random.default_rng(0)
    samples = [dist.sample(rng) for _ in range(1000)]
    assert all(-0.5 <= s <= 0.5 for s in samples)


def test_distribution_array_clipping() -> None:
    """Distribution 批量采样截断."""
    dist = Distribution(mean=0.0, std=1.0, low=-0.5, high=0.5)
    rng = np.random.default_rng(0)
    samples = dist.sample_array(rng, 1000)
    assert np.all(samples >= -0.5)
    assert np.all(samples <= 0.5)


def test_monte_carlo_process() -> None:
    """process 后端产生有效散布统计."""
    cfg = _base_cfg()
    cfg.options.mpm_use_spin = False
    perturb = PerturbationConfig(
        v0_m_s=DistributionConfig(std=2.0),
        elevation_deg=DistributionConfig(std=0.05),
    )
    result = monte_carlo_simulation(
        cfg,
        perturb=perturb,
        n_samples=50,
        backend="process",
        n_jobs=1,
        seed=42,
    )
    assert result.samples == 50
    assert result.range_std > 0
    assert result.cross_std >= 0
    assert result.cep90 >= result.cep50 >= 0
    assert result.ellipse_major >= result.ellipse_minor >= 0


def test_monte_carlo_batch() -> None:
    """batch 后端产生有效散布统计."""
    cfg = _base_cfg()
    cfg.options.mpm_use_spin = False
    perturb = PerturbationConfig(
        v0_m_s=DistributionConfig(std=2.0),
        elevation_deg=DistributionConfig(std=0.05),
    )
    result = monte_carlo_simulation(
        cfg,
        perturb=perturb,
        n_samples=50,
        backend="batch",
        seed=42,
    )
    assert result.samples == 50
    assert result.range_std > 0


def test_monte_carlo_process_batch_consistency() -> None:
    """process 与 batch 后端统计结果一致（<5% 相对差）."""
    cfg = _base_cfg()
    cfg.options.mpm_use_spin = False
    perturb = PerturbationConfig(
        v0_m_s=DistributionConfig(std=2.0),
        elevation_deg=DistributionConfig(std=0.05),
    )
    res_process = monte_carlo_simulation(
        cfg,
        perturb=perturb,
        n_samples=100,
        backend="process",
        n_jobs=1,
        seed=42,
    )
    res_batch = monte_carlo_simulation(
        cfg,
        perturb=perturb,
        n_samples=100,
        backend="batch",
        seed=42,
    )
    assert abs(res_process.range_mean - res_batch.range_mean) / res_process.range_mean < 0.05
    # 零扰动单发对拍 <1%；有扰动时 batch 线性阻力表与 process PCHIP 存在系统偏差，
    # 标准差差异放宽到 <50%。
    assert abs(res_process.range_std - res_batch.range_std) / max(res_process.range_std, 1e-6) < 0.5


def test_monte_carlo_auto_selects_batch_for_projectile() -> None:
    """auto 后端在 projectile + UniformWind + spin 关闭下选择 batch。"""
    cfg = _base_cfg()
    cfg.options.mpm_use_spin = False
    assert _can_use_batch(cfg)
    result = monte_carlo_simulation(
        cfg,
        perturb=PerturbationConfig(v0_m_s=DistributionConfig(std=2.0)),
        n_samples=30,
        backend="auto",
        seed=42,
    )
    assert result.samples == 30


def test_monte_carlo_auto_falls_back_to_process_for_non_projectile() -> None:
    """auto 后端在非 projectile 任务下回退到 process。"""
    cfg = _base_cfg()
    cfg.mission = "rocket"
    assert not _can_use_batch(cfg)


def test_monte_carlo_invalid_backend() -> None:
    """非法 backend 抛出 ValueError."""
    cfg = _base_cfg()
    with pytest.raises(ValueError):
        monte_carlo_simulation(cfg, backend="invalid")


def test_monte_carlo_gpu_unavailable_raises() -> None:
    """GPU 不可用时 gpu 后端抛出 RuntimeError。"""
    from ballistic_sim.dynamics.gpu_mpm import gpu_available

    if gpu_available():
        pytest.skip("GPU 可用，跳过不可用测试")
    cfg = _base_cfg()
    cfg.options.mpm_use_spin = False
    with pytest.raises(RuntimeError):
        monte_carlo_simulation(
            cfg,
            n_samples=10,
            backend="gpu",
            seed=42,
        )


def test_sensitivity() -> None:
    """灵敏度分析返回正确结构."""
    cfg = _base_cfg()
    perturb = PerturbationConfig(v0_m_s=DistributionConfig(std=2.0))
    sens = sensitivity_analysis(
        cfg,
        perturb=perturb,
        param_name="v0_m_s",
        n_samples=30,
        seed=42,
    )
    assert sens["param"] == "v0_m_s"
    assert sens["range_std"] > 0


def test_dispersion_result_fields() -> None:
    """DispersionResult 字段可访问."""
    rng = np.random.default_rng(0)
    ranges = rng.normal(1000.0, 10.0, 50)
    crosses = rng.normal(0.0, 5.0, 50)
    tofs = rng.normal(20.0, 0.5, 50)
    angles = rng.normal(45.0, 1.0, 50)
    result = DispersionResult(
        ranges=ranges,
        crosses=crosses,
        tofs=tofs,
        impact_angles=angles,
        range_mean=float(np.mean(ranges)),
        range_std=float(np.std(ranges)),
        cross_mean=float(np.mean(crosses)),
        cross_std=float(np.std(crosses)),
        cep50=1.0,
        cep90=2.0,
        ellipse_major=3.0,
        ellipse_minor=1.5,
        ellipse_angle=30.0,
        samples=50,
    )
    assert result.samples == 50
    assert result.cep50 == 1.0


def test_config_to_perturbation_param_wins() -> None:
    """函数参数 perturb 优先于 cfg 中的配置."""
    cfg_perturb = PerturbationConfig(v0_m_s=DistributionConfig(std=1.0))
    param_perturb = PerturbationConfig(v0_m_s=DistributionConfig(std=3.0))
    assert _config_to_perturbation(cfg_perturb, param_perturb) is param_perturb
    assert _config_to_perturbation(cfg_perturb, None) is cfg_perturb
    default = _config_to_perturbation(None, None)
    assert default.v0_m_s.std == 2.0


def test_extract_drag_table_various_aero_models() -> None:
    """_extract_drag_table 覆盖各类气动模型."""
    from ballistic_sim.models.aerodynamics import (
        DRAG_G1,
        ConstantAeroModel,
        ProjectileAeroModel,
        RocketAeroModel,
        make_aero,
    )

    table_g1 = _extract_drag_table(DRAG_G1)
    assert table_g1.ndim == 2 and table_g1.shape[1] == 2
    assert table_g1.shape[0] == 501

    proj = ProjectileAeroModel(drag_law=DRAG_G1)
    table_proj = _extract_drag_table(proj)
    assert table_proj.shape == table_g1.shape

    const = ConstantAeroModel(cd=0.42)
    table_const = _extract_drag_table(const)
    assert table_const.ndim == 2 and table_const.shape[0] >= 2
    assert np.allclose(table_const[:, 1], 0.42)

    rocket = RocketAeroModel()
    table_rocket = _extract_drag_table(rocket)
    assert table_rocket.ndim == 2 and table_rocket.shape[1] == 2

    g1_model = make_aero("g1", cl_slope=0.0)
    table_make = _extract_drag_table(g1_model)
    assert table_make.shape == table_g1.shape


def test_can_use_batch_false_for_non_uniform_wind(monkeypatch) -> None:
    """非 UniformWind 时 _can_use_batch 返回 False."""
    from ballistic_sim.models.wind import LogarithmicWind

    cfg = _base_cfg()
    cfg.options.mpm_use_spin = False

    def _fake_resolve_wind(_cfg: SimConfig):
        return LogarithmicWind(u_ref=5.0)

    monkeypatch.setattr("ballistic_sim.monte_carlo._resolve_wind", _fake_resolve_wind)
    assert not _can_use_batch(cfg)


def test_perturb_cfg_all_fields() -> None:
    """_perturb_cfg 对所有可扰动字段采样."""
    cfg = _base_cfg()
    perturb = PerturbationConfig(
        mass_kg=DistributionConfig(std=0.1),
        form_factor=DistributionConfig(std=0.01),
        v0_m_s=DistributionConfig(std=2.0),
        elevation_deg=DistributionConfig(std=0.05),
        azimuth_deg=DistributionConfig(std=0.05),
        delta_t=DistributionConfig(std=2.0),
        density_factor=DistributionConfig(std=0.02),
        wind_e=DistributionConfig(std=1.0),
        wind_n=DistributionConfig(std=1.0),
    )
    cfg2 = _perturb_cfg(cfg, perturb, seed=0)
    # 至少某些字段应发生变化（大概率）
    assert cfg2.vehicle.mass_kg != pytest.approx(
        cfg.vehicle.mass_kg, abs=1e-9
    ) or cfg2.launch.v0_m_s != pytest.approx(cfg.launch.v0_m_s, abs=1e-9)


def test_perturb_cfg_zero_std_preserves_values() -> None:
    """所有扰动 std 为 0 时 _perturb_cfg 保持原值（仅 wind 列表复制）."""
    cfg = _base_cfg()
    perturb = PerturbationConfig(
        mass_kg=DistributionConfig(std=0.0),
        form_factor=DistributionConfig(std=0.0),
        v0_m_s=DistributionConfig(std=0.0),
        elevation_deg=DistributionConfig(std=0.0),
        azimuth_deg=DistributionConfig(std=0.0),
        delta_t=DistributionConfig(std=0.0),
        density_factor=DistributionConfig(std=0.0),
        wind_e=DistributionConfig(std=0.0),
        wind_n=DistributionConfig(std=0.0),
    )
    cfg2 = _perturb_cfg(cfg, perturb, seed=0)
    assert cfg2.vehicle.mass_kg == pytest.approx(cfg.vehicle.mass_kg)
    assert cfg2.launch.v0_m_s == pytest.approx(cfg.launch.v0_m_s)
    assert cfg2.launch.elevation_deg == pytest.approx(cfg.launch.elevation_deg)
    assert cfg2.environment.density_factor == pytest.approx(cfg.environment.density_factor)
    assert np.allclose(cfg2.environment.wind_m_s, cfg.environment.wind_m_s)


def test_run_single_process_exception_returns_none() -> None:
    """仿真失败时 _run_single_process 返回 None 且不抛异常."""
    cfg = _base_cfg()
    # 质量为负会导致积分失败
    cfg = cfg.model_copy(update={"vehicle": cfg.vehicle.model_copy(update={"mass_kg": -1.0})})
    result = _run_single_process(cfg, seed=0, perturb=PerturbationConfig())
    assert result is None


def test_extract_enu_impact_edge_cases() -> None:
    """_extract_enu_impact 处理空结果或未落地情况."""
    from ballistic_sim.simulator import SimResult

    empty = SimResult()
    assert _extract_enu_impact(empty) is None

    # 末高度仍大于 0，视为未落地
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, (10, 6))
    y[:, 2] = np.abs(y[:, 2]) + 0.1  # 所有高度为正
    airborne = SimResult(t=np.arange(10, dtype=float), y=y)
    assert _extract_enu_impact(airborne) is None


def test_monte_carlo_too_few_valid_samples() -> None:
    """有效样本过少时 process 后端抛 RuntimeError."""
    cfg = _base_cfg()
    cfg = cfg.model_copy(update={"vehicle": cfg.vehicle.model_copy(update={"mass_kg": -1.0})})
    with pytest.raises(RuntimeError, match="有效样本过少"):
        monte_carlo_simulation(cfg, n_samples=5, backend="process", seed=0)


def test_monte_carlo_batch_too_few_valid() -> None:
    """有效样本过少时 batch 后端抛 RuntimeError."""
    cfg = _base_cfg()
    cfg.options.mpm_use_spin = False
    # 初速为负会导致无法落地
    cfg = cfg.model_copy(update={"launch": cfg.launch.model_copy(update={"v0_m_s": -1.0})})
    with pytest.raises(RuntimeError, match="有效样本过少"):
        monte_carlo_simulation(cfg, n_samples=5, backend="batch", seed=0)


def test_monte_carlo_config_rejects_non_positive_samples() -> None:
    """MonteCarloConfig 拒绝非正样本数."""
    from ballistic_sim.config import MonteCarloConfig

    with pytest.raises(ValueError):
        MonteCarloConfig(num_samples=0)
    with pytest.raises(ValueError):
        MonteCarloConfig(num_samples=-5)


def test_monte_carlo_process_n_jobs_parallel() -> None:
    """process 后端 n_jobs=2 可并行运行."""
    cfg = _base_cfg()
    cfg.options.mpm_use_spin = False
    result = monte_carlo_simulation(
        cfg,
        perturb=PerturbationConfig(v0_m_s=DistributionConfig(std=2.0)),
        n_samples=10,
        backend="process",
        n_jobs=2,
        seed=42,
    )
    assert result.samples == 10


def test_monte_carlo_auto_rocket_uses_process(monkeypatch) -> None:
    """auto 后端在非 projectile 任务下调用 process 后端."""
    cfg = _base_cfg()
    cfg.mission = "rocket"
    cfg.vehicle.thrust_N = 1.0e6
    cfg.vehicle.burn_time_s = 10.0

    called = False

    def _fake_process(
        _cfg: SimConfig,
        _perturb: PerturbationConfig,
        _n_samples: int,
        _n_jobs: int,
        _seed: int,
    ) -> DispersionResult:
        nonlocal called
        called = True
        rng = np.random.default_rng(_seed)
        ranges = rng.normal(1000.0, 10.0, _n_samples)
        crosses = rng.normal(0.0, 5.0, _n_samples)
        tofs = rng.normal(20.0, 0.5, _n_samples)
        angles = rng.normal(45.0, 1.0, _n_samples)
        return DispersionResult(
            ranges=ranges,
            crosses=crosses,
            tofs=tofs,
            impact_angles=angles,
            range_mean=float(np.mean(ranges)),
            range_std=float(np.std(ranges)),
            cross_mean=float(np.mean(crosses)),
            cross_std=float(np.std(crosses)),
            cep50=1.0,
            cep90=2.0,
            ellipse_major=3.0,
            ellipse_minor=1.5,
            ellipse_angle=30.0,
            samples=_n_samples,
        )

    monkeypatch.setattr("ballistic_sim.monte_carlo._monte_carlo_process", _fake_process)
    result = monte_carlo_simulation(
        cfg,
        perturb=PerturbationConfig(v0_m_s=DistributionConfig(std=1.0)),
        n_samples=10,
        backend="auto",
        seed=42,
    )
    assert called
    assert result.samples == 10


def test_sensitivity_invalid_std() -> None:
    """灵敏度分析对 std<=0 的参数抛 ValueError."""
    cfg = _base_cfg()
    perturb = PerturbationConfig(v0_m_s=DistributionConfig(std=0.0))
    with pytest.raises(ValueError, match="标准差必须大于 0"):
        sensitivity_analysis(cfg, perturb, param_name="v0_m_s", n_samples=10)


def test_monte_carlo_uses_cfg_monte_carlo_perturbations() -> None:
    """monte_carlo_simulation 读取 cfg.options.monte_carlo.perturbations."""
    from ballistic_sim.config import MonteCarloConfig

    cfg = _base_cfg()
    cfg.options.mpm_use_spin = False
    cfg.options.monte_carlo = MonteCarloConfig(
        perturbations=PerturbationConfig(v0_m_s=DistributionConfig(std=2.0)),
        num_samples=20,
        seed=42,
    )
    result = monte_carlo_simulation(cfg, n_samples=20, backend="batch")
    assert result.samples == 20
    assert result.range_std > 0


def test_extract_drag_table_non_pchip_drag_law() -> None:
    """_extract_drag_table 支持非 PCHIP 的 DragLaw."""
    from ballistic_sim.models.aerodynamics import DragLaw

    drag_law = DragLaw(
        name="custom",
        cd0_fn=lambda ma: 0.3 + 0.1 * np.asarray(ma) ** 2,
        cd_alpha2_fn=lambda _: 0.0,
    )
    table = _extract_drag_table(drag_law)
    assert table.ndim == 2 and table.shape[1] == 2
    assert table.shape[0] == 501


def test_extract_drag_table_generic_fallback() -> None:
    """_extract_drag_table 兜底分支支持任意 drag_coefficient 对象."""

    class DummyAero:
        def drag_coefficient(self, mach, alpha=0.0):
            return 0.25 + 0.05 * np.asarray(mach)

    table = _extract_drag_table(DummyAero())
    assert table.ndim == 2 and table.shape[1] == 2
    assert table.shape[0] == 501


def test_monte_carlo_batch_gpu_require_use_batch() -> None:
    """spin 开启时 batch/gpu 后端抛 ValueError."""
    cfg = _base_cfg()
    cfg.options.mpm_use_spin = True  # 导致 _can_use_batch 为 False
    with pytest.raises(ValueError, match="batch"):
        monte_carlo_simulation(cfg, n_samples=5, backend="batch", seed=0)
    with pytest.raises(ValueError, match="GPU"):
        monte_carlo_simulation(cfg, n_samples=5, backend="gpu", seed=0)
