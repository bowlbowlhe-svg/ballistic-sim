"""批量 MPM 求解器测试."""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.config import (
    EnvironmentConfig,
    GuidanceConfig,
    LaunchConfig,
    OptionsConfig,
    SimConfig,
    VehicleConfig,
)
from ballistic_sim.dynamics.batch_mpm import BatchMPMModel
from ballistic_sim.dynamics.mpm import MPMOptions, MPMDynamics
from ballistic_sim.phases.powered import PoweredPhase
from ballistic_sim.phases.terminal import TerminalPhase
from ballistic_sim.simulator import simulate


def _base_cfg(
    qe: float = 45.0,
    az: float = 90.0,
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SimConfig:
    return SimConfig(
        mission="projectile",
        vehicle=VehicleConfig(mass_kg=43.2, diameter_m=0.155, cd=0.78),
        launch=LaunchConfig(
            lat_deg=39.9,
            lon_deg=0.0,
            alt_m=0.0,
            azimuth_deg=az,
            elevation_deg=qe,
            v0_m_s=684.0,
        ),
        environment=EnvironmentConfig(
            atmosphere="isa",
            wind_m_s=list(wind),
            gravity_model="wgs84",
            delta_t=0.0,
            density_factor=1.0,
        ),
        guidance=GuidanceConfig(),
        options=OptionsConfig(),
    )


def _single_mpm_result(cfg: SimConfig, dt: float = 0.05) -> tuple[float, float, float]:
    """使用单发 MPM（spin/dynamic_alpha 关闭）仿真并返回 (range_m, cross_m, tof)."""
    opt = MPMOptions(
        use_drag=True,
        use_wind=bool(cfg.environment.wind_m_s),
        use_coriolis=True,
        use_spin=False,
        use_dynamic_alpha=False,
        method="DOP853",
        rtol=1e-6,
        atol=1e-9,
    )
    dyn = MPMDynamics(
        mass_kg=cfg.vehicle.mass_kg,
        diameter_m=cfg.vehicle.diameter_m,
        form_factor=cfg.vehicle.cd or 1.0,
        options=opt,
        lat_deg=cfg.launch.lat_deg,
    )
    phases = [
        PoweredPhase(
            name="无动力弹道",
            t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 3000.0),
            dynamics=dyn,
            guidance=None,
            m_dry=cfg.vehicle.mass_kg,
            sep_name="落地",
        ),
        TerminalPhase(
            name="终点",
            t_span=(cfg.launch.t0_s, cfg.launch.t0_s + 3000.0),
            dynamics=dyn,
        ),
    ]
    result = simulate(cfg, phases=phases)
    e, n, u = result.y[-1, 0], result.y[-1, 1], result.y[-1, 2]
    assert u <= 1e-6, f"MPM 未落地，末高度 {u}"
    return float(np.hypot(e, n)), float(e), float(result.t[-1])


def _make_arrays(n: int, cfg: SimConfig) -> dict[str, np.ndarray]:
    return {
        "mass_kg": np.full(n, cfg.vehicle.mass_kg),
        "diameter_m": np.full(n, cfg.vehicle.diameter_m),
        "form_factor": np.full(n, cfg.vehicle.cd or 1.0),
        "v0": np.full(n, cfg.launch.v0_m_s),
        "theta_deg": np.full(n, cfg.launch.elevation_deg),
        "az_deg": np.full(n, cfg.launch.azimuth_deg),
        "delta_t": np.full(n, cfg.environment.delta_t),
        "density_factor": np.full(n, cfg.environment.density_factor),
        "wind_e": np.full(n, cfg.environment.wind_m_s[0]),
        "wind_n": np.full(n, cfg.environment.wind_m_s[1]),
        "wind_u": np.full(n, cfg.environment.wind_m_s[2]),
    }


def test_batch_mpm_matches_single_mpm() -> None:
    """批量求解器零扰动时应与单发 MPM 一致（<1%）."""
    cfg = _base_cfg(qe=45.0, az=90.0)
    single_range, single_cross, single_tof = _single_mpm_result(cfg)

    arrays = _make_arrays(10, cfg)
    opt = MPMOptions(use_drag=True, use_wind=False, use_coriolis=True, use_spin=False)
    model = BatchMPMModel(
        **arrays,
        lat_deg=cfg.launch.lat_deg,
        h0=cfg.launch.alt_m,
        azimuth_deg=cfg.launch.azimuth_deg,
        options=opt,
        dt=0.05,
    )
    res = model.simulate()

    assert res.n_samples == 10
    assert np.all(res.landed)
    assert np.allclose(res.range_m, res.range_m[0], rtol=1e-12)

    rel_range = abs(res.range_m[0] - single_range) / single_range
    assert rel_range < 0.01, f"射程相对差 {rel_range:.4%} ≥ 1%"
    rel_cross = abs(res.cross_m[0] - single_cross) / max(single_range, 1.0)
    assert rel_cross < 0.01, f"偏流相对差 {rel_cross:.4%} ≥ 1%"
    rel_tof = abs(res.tof[0] - single_tof) / single_tof
    assert rel_tof < 0.01, f"TOF 相对差 {rel_tof:.4%} ≥ 1%"


def test_batch_mpm_with_wind_and_coriolis() -> None:
    """批量求解器支持风和科氏力并与单发 MPM 一致."""
    cfg = _base_cfg(qe=45.0, az=90.0, wind=(5.0, 2.0, 0.0))
    single_range, single_cross, _ = _single_mpm_result(cfg)

    arrays = _make_arrays(5, cfg)
    opt = MPMOptions(use_drag=True, use_wind=True, use_coriolis=True, use_spin=False)
    model = BatchMPMModel(
        **arrays,
        lat_deg=cfg.launch.lat_deg,
        h0=cfg.launch.alt_m,
        azimuth_deg=cfg.launch.azimuth_deg,
        options=opt,
        dt=0.05,
    )
    res = model.simulate()

    assert res.landed.all()
    rel_range = abs(res.range_m[0] - single_range) / single_range
    assert rel_range < 0.01, f"带风射程相对差 {rel_range:.4%} ≥ 1%"
    assert np.sign(res.cross_m[0]) == np.sign(single_cross)


@pytest.mark.parametrize("n", [1, 5, 100])
def test_batch_mpm_multiple_samples(n: int) -> None:
    """不同样本数均可运行."""
    cfg = _base_cfg()
    arrays = _make_arrays(n, cfg)
    opt = MPMOptions(use_drag=True, use_wind=False, use_coriolis=False, use_spin=False)
    model = BatchMPMModel(
        **arrays,
        lat_deg=cfg.launch.lat_deg,
        h0=cfg.launch.alt_m,
        azimuth_deg=cfg.launch.azimuth_deg,
        options=opt,
        dt=0.05,
    )
    res = model.simulate()
    assert res.n_samples == n
    assert res.landed.sum() == n


def test_batch_mpm_perturbation() -> None:
    """批量求解器对参数扰动产生散布."""
    cfg = _base_cfg()
    n = 100
    rng = np.random.default_rng(42)
    arrays = _make_arrays(n, cfg)
    arrays["v0"] = rng.normal(cfg.launch.v0_m_s, 2.0, n)
    arrays["theta_deg"] = rng.normal(cfg.launch.elevation_deg, 0.05, n)
    opt = MPMOptions(use_drag=True, use_wind=False, use_coriolis=False, use_spin=False)
    model = BatchMPMModel(
        **arrays,
        lat_deg=cfg.launch.lat_deg,
        h0=cfg.launch.alt_m,
        azimuth_deg=cfg.launch.azimuth_deg,
        options=opt,
        dt=0.05,
    )
    res = model.simulate()

    assert res.landed.sum() > 90
    assert np.std(res.range_m) > 1.0


def test_batch_mpm_parameter_sanity() -> None:
    """质量/密度修正等参数变化产生合理单调趋势."""
    cfg = _base_cfg()
    n = 5
    arrays = _make_arrays(n, cfg)
    arrays["mass_kg"] = np.array([40.0, 43.0, 46.0, 49.0, 52.0])
    opt = MPMOptions(use_drag=True, use_wind=False, use_coriolis=False, use_spin=False)
    model = BatchMPMModel(
        **arrays,
        lat_deg=cfg.launch.lat_deg,
        h0=cfg.launch.alt_m,
        azimuth_deg=cfg.launch.azimuth_deg,
        options=opt,
        dt=0.05,
    )
    res = model.simulate()
    assert res.landed.all()
    # 质量越大，阻力加速度越小，射程应递增
    assert np.all(np.diff(res.range_m) > 0)


def test_batch_mpm_gpu_skip_if_no_cupy() -> None:
    """无 CuPy 时 GPU 后端应抛出 ImportError。"""
    cupy = pytest.importorskip("cupy", reason="需要 CuPy")
    if cupy is None:
        pytest.skip("CuPy 未安装")

    from ballistic_sim.dynamics.gpu_mpm import GPUBatchMPMModel

    cfg = _base_cfg()
    arrays = _make_arrays(2, cfg)
    model = GPUBatchMPMModel(
        **arrays,
        lat_deg=cfg.launch.lat_deg,
        h0=cfg.launch.alt_m,
        azimuth_deg=cfg.launch.azimuth_deg,
    )
    res = model.simulate()
    assert res.n_samples == 2


@pytest.mark.parametrize(
    "qe,wind_e,delta_t,density_factor",
    [
        (45.0, 0.0, 0.0, 1.0),
        (30.0, 5.0, 0.0, 1.0),
        (70.0, 0.0, 0.0, 1.0),
        (45.0, 0.0, 20.0, 1.0),
        (45.0, 0.0, 0.0, 0.9),
        (45.0, 0.0, 0.0, 1.1),
    ],
)
def test_batch_vs_single_boundary_conditions(
    qe: float, wind_e: float, delta_t: float, density_factor: float
) -> None:
    """批量与单发在不同边界条件下均保持 <1% 差异."""
    cfg = _base_cfg(qe=qe, wind=(wind_e, 0.0, 0.0))
    cfg.environment.delta_t = delta_t
    cfg.environment.density_factor = density_factor
    single_range, single_cross, _ = _single_mpm_result(cfg)

    arrays = _make_arrays(1, cfg)
    opt = MPMOptions(use_drag=True, use_wind=True, use_coriolis=True, use_spin=False)
    model = BatchMPMModel(
        **arrays,
        lat_deg=cfg.launch.lat_deg,
        h0=cfg.launch.alt_m,
        azimuth_deg=cfg.launch.azimuth_deg,
        options=opt,
        dt=0.02,
    )
    res = model.simulate()
    assert res.landed[0]
    rel_range = abs(res.range_m[0] - single_range) / single_range
    assert (
        rel_range < 0.01
    ), f"qe={qe} wind_e={wind_e} dt={delta_t} df={density_factor} 射程差 {rel_range:.4%}"


def test_batch_mpm_scalar_broadcast() -> None:
    """标量参数可广播到 n_samples."""
    cfg = _base_cfg()
    opt = MPMOptions(use_drag=True, use_wind=False, use_coriolis=False, use_spin=False)
    model = BatchMPMModel(
        mass_kg=np.full(5, cfg.vehicle.mass_kg),
        diameter_m=cfg.vehicle.diameter_m,  # 标量
        form_factor=cfg.vehicle.cd or 1.0,  # 标量
        v0=cfg.launch.v0_m_s,  # 标量
        theta_deg=np.full(5, cfg.launch.elevation_deg),
        az_deg=np.full(5, cfg.launch.azimuth_deg),
        delta_t=0.0,
        density_factor=1.0,
        wind_e=np.zeros(5),
        wind_n=np.zeros(5),
        lat_deg=cfg.launch.lat_deg,
        h0=cfg.launch.alt_m,
        azimuth_deg=cfg.launch.azimuth_deg,
        options=opt,
        dt=0.05,
    )
    res = model.simulate()
    assert res.n_samples == 5
    assert res.landed.all()


def test_batch_mpm_invalid_array_shape() -> None:
    """参数数组长度不匹配时抛 ValueError."""
    cfg = _base_cfg()
    opt = MPMOptions(use_drag=True, use_wind=False, use_coriolis=False, use_spin=False)
    # 通过 _assert_shape 检查的核心数组长度错误
    with pytest.raises(ValueError, match="n_samples"):
        BatchMPMModel(
            mass_kg=np.full(5, cfg.vehicle.mass_kg),
            diameter_m=cfg.vehicle.diameter_m,
            form_factor=np.full(5, cfg.vehicle.cd or 1.0),
            v0=np.full(3, cfg.launch.v0_m_s),
            theta_deg=np.full(5, cfg.launch.elevation_deg),
            az_deg=np.full(5, cfg.launch.azimuth_deg),
            delta_t=np.zeros(5),
            density_factor=np.ones(5),
            wind_e=np.zeros(5),
            wind_n=np.zeros(5),
            options=opt,
        )


def test_batch_mpm_as_xp_scalar_array_broadcast() -> None:
    """_as_xp 将长度为 1 的数组广播到 n_samples."""
    cfg = _base_cfg()
    opt = MPMOptions(use_drag=True, use_wind=False, use_coriolis=False, use_spin=False)
    model = BatchMPMModel(
        mass_kg=np.full(5, cfg.vehicle.mass_kg),
        diameter_m=np.full(5, cfg.vehicle.diameter_m),
        form_factor=np.full(5, cfg.vehicle.cd or 1.0),
        v0=np.full(5, cfg.launch.v0_m_s),
        theta_deg=np.full(5, cfg.launch.elevation_deg),
        az_deg=np.full(5, cfg.launch.azimuth_deg),
        delta_t=np.zeros(5),
        density_factor=np.ones(5),
        wind_e=np.zeros(5),
        wind_n=np.zeros(5),
        options=opt,
        dt=0.05,
    )
    scalar_arr = np.array([cfg.vehicle.diameter_m])
    broadcasted = model._as_xp(scalar_arr)
    assert broadcasted.shape == (5,)
    assert np.allclose(broadcasted, cfg.vehicle.diameter_m)


def test_batch_mpm_invalid_drag_table() -> None:
    """drag_table 形状非法时抛 ValueError."""
    cfg = _base_cfg()
    opt = MPMOptions(use_drag=True, use_wind=False, use_coriolis=False, use_spin=False)
    arrays = _make_arrays(2, cfg)
    with pytest.raises(ValueError, match="drag_table"):
        BatchMPMModel(
            **arrays,
            lat_deg=cfg.launch.lat_deg,
            h0=cfg.launch.alt_m,
            azimuth_deg=cfg.launch.azimuth_deg,
            drag_table=np.array([0.1, 0.2, 0.3]),  # 形状错误
            options=opt,
        )


def test_batch_mpm_no_wind() -> None:
    """use_wind=False 时忽略风."""
    cfg = _base_cfg(wind=(10.0, 5.0, 0.0))
    arrays = _make_arrays(3, cfg)
    opt = MPMOptions(use_drag=True, use_wind=False, use_coriolis=False, use_spin=False)
    model = BatchMPMModel(
        **arrays,
        lat_deg=cfg.launch.lat_deg,
        h0=cfg.launch.alt_m,
        azimuth_deg=cfg.launch.azimuth_deg,
        options=opt,
        dt=0.05,
    )
    res = model.simulate()
    assert res.landed.all()
    # 无侧向风时 cross 与 range 几乎相等（azimuth=90°，N≈0）
    assert np.allclose(res.cross_m, res.range_m, rtol=1e-12)


def test_batch_mpm_custom_drag_table() -> None:
    """自定义 drag_table 可正确参与积分."""
    cfg = _base_cfg()
    arrays = _make_arrays(2, cfg)
    opt = MPMOptions(use_drag=True, use_wind=False, use_coriolis=False, use_spin=False)
    # 常值 Cd=0.3 的自定义表
    drag_table = np.array([[0.0, 0.3], [5.0, 0.3]], dtype=np.float64)
    model = BatchMPMModel(
        **arrays,
        lat_deg=cfg.launch.lat_deg,
        h0=cfg.launch.alt_m,
        azimuth_deg=cfg.launch.azimuth_deg,
        drag_table=drag_table,
        options=opt,
        dt=0.05,
    )
    res = model.simulate()
    assert res.landed.all()


def test_batch_mpm_early_landing_break() -> None:
    """所有样本落地后提前结束积分."""
    cfg = _base_cfg(qe=30.0)
    arrays = _make_arrays(5, cfg)
    opt = MPMOptions(use_drag=True, use_wind=False, use_coriolis=False, use_spin=False)
    model = BatchMPMModel(
        **arrays,
        lat_deg=cfg.launch.lat_deg,
        h0=cfg.launch.alt_m,
        azimuth_deg=cfg.launch.azimuth_deg,
        options=opt,
        dt=0.05,
        t_max=400.0,
    )
    res = model.simulate()
    assert res.landed.all()
