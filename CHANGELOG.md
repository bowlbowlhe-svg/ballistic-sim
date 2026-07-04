# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/lang/zh-CN/).

## [0.1.0] - 2026-07-03

### Added

- **统一配置层**：基于 pydantic 的 `SimConfig`，包含 `VehicleConfig`、`LaunchConfig`、`EnvironmentConfig`、`GuidanceConfig`、`OptionsConfig`。
- **配置覆盖与校验**：`apply_overrides` 支持点路径覆盖；`validate_config` 提供跨字段业务校验骨架。
- **物理模型层**：
  - 标准大气模型（ISA/US76）与工厂函数 `make_atmosphere`。
  - J2 / 点质量引力模型 `gravity_eci`。
  - 气动力模型：G1/G7 弹丸阻力表、火箭近似阻力。
  - 风模型：均匀风、对数风、幂律风、Dryden 阵风。
  - 推进模型 `PropulsionModel`：推力高度修正、质量流率、燃尽事件。
- **运动方程**：
  - `MPMDynamics`：8 维 MPM（含自旋、章动、攻角近似）。
  - `PointMassDynamics`：3-DOF ECI 质点。
  - `PoweredECIDynamics`：3-DOF ECI 动力上升（支持 J2、阻力、程序转弯）。
- **阶段驱动**：`Phase` 抽象基类 + `PoweredPhase` / `CoastingPhase` / `ReentryPhase` / `TerminalPhase`。
- **统一仿真器**：`simulator.simulate` 按 Phase 序列逐段调用 `solve_ivp`，支持段间状态投影与事件日志。
- **制导与火控**：
  - 开环程序转弯/重力转弯 `open_loop`。
  - 动力显式制导 `PEG` 骨架。
  - 火炮诸元解算 `firecontrol`。
  - 6-DOF 控制律接口 `control`（骨架）。
- **CLI**：`python -m ballistic_sim.cli` 统一入口，支持 `projectile` / `missile` / `rocket` / `icbm` / `suborbital`。
- **可视化**：纯函数设计，按 ENU/ECI 分域输出 PNG；包含 3D 轨迹、地面航迹、高度/速度剖面、风廓线、地球 3D、轨道插入、再入热流、GIF 动画等。
- **YAML 预设**：`presets/projectiles.yaml`、`presets/missiles.yaml`、`presets/rockets.yaml`，提供 M107、CZ-2F 等教学/调试参数。
- **坐标系转换层**：`frames.py` 统一实现 ENU ↔ ECEF ↔ ECI。
- **状态切换层**：`state_switch.py` 管理阶段间状态投影，禁止 3-DOF → 6-DOF 自动升维。
- **测试套件**：263 项 pytest 用例，覆盖模型、动力学、阶段、制导、CLI、可视化、预设工厂；含 golden 参考与冒烟测试；整体代码覆盖率 86%。
- **回归桥与 MVP 脚本**：`scripts/regression_bridge.py`、`scripts/mvp_demo.py` 用于与旧项目对拍。
- **文档**：`README.md`、`docs/architecture.md`、`docs/known_failures.md`、`docs/migration_scope.md`。

### Changed

- 统一物理常数来源为 `ballistic_sim.constants`，合并原 `外弹道方程` 与 `火箭仿真程序` 的常量定义。

### Known Limitations

- **GUI/Web 界面**：v0.1.0 未包含；`web/` 目录仅为占位，计划在 v1.0 之后评估是否启用。
- **6-DOF 控制闭环**：`guidance.control` 接口已提供，但 6-DOF 动力学闭环仅在稳定工况下启用；默认任务使用 MPM 或 3-DOF ECI。
- **部分 viz 模块为骨架实现**：`dvbudget.py` 部分逻辑待完善，`reentry.py` 热流模型为简化 Sutton-Graves 近似。
- **风场**：CLI 中风廓线使用占位零风；真实风剖面需通过配置文件或 API 注入。
- **地形**：`use_terrain=True` 接口已预留，默认使用 `NullTerrain`，真实 DEM 数据需自行准备。
- **覆盖率**：整体覆盖率 86%，达到 Stage 5 ≥80% 目标；核心物理模块（dynamics / models / guidance）多数 ≥90%。部分地形加载器、Web 占位代码覆盖率仍较低，属于可选 extras 与占位模块。

### Version Milestones

| Tag | Stage | Description |
|-----|-------|-------------|
| `v0.1.0-stage0` | 阶段 0 | 项目骨架、迁移准备、常量统一 |
| `v0.1.0-stage1` | 阶段 1 | 物理模型统一层（大气、引力、气动力、风） |
| `v0.1.0-stage2` | 阶段 2 | 运动方程与阶段驱动（MPM、PointMass、PoweredECI） |
| `v0.1.0-stage3` | 阶段 3 | 制导与火控（开环、PEG、火控、6-DOF 控制接口） |
| `v0.1.0-stage4` | 阶段 4 | 入口与可视化（CLI、viz 纯函数、YAML 预设） |
| `v0.1.0` | 阶段 5 | 验证、对拍与收尾（文档、CHANGELOG、bug 修复、flake8/mypy 全绿、覆盖率 86%） |

## [0.2.0] - Unreleased

### Added

- **6-DOF 刚体动力学**：新增 `ballistic_sim/dynamics/six_dof.py`，状态向量 13 维 `[r(3), v(3), q(4), omega_y, omega_z, p]`，支持轴对称弹、四元数姿态、气动力/力矩、控制矩注入。
- **6-DOF 状态切换**：`state_switch.py` 支持 6-DOF ↔ 3-DOF/MPM 显式投影，7→13 升维默认禁止，但允许 `allow_auto=True` 沿速度方向构造姿态（用于再入段）。
- **6-DOF 再入段集成**：`ReentryPhase` 支持 `fidelity="sixdof"`，`builder.py` 可通过 `OptionsConfig(sixdof_reentry=True)` 自动插入 sixdof 再入段。
- **6-DOF 测试套件**：新增 `tests/dynamics/test_six_dof*.py`、`tests/test_sixdof_vs_mpm.py`、`tests/test_sixdof_reentry_builder.py`，覆盖单元测试、四元数代数、控制律、已知失效回归、MPM 对拍、阶段链集成。
- **Monte Carlo 散布分析**：新增 `ballistic_sim/monte_carlo.py`，提供 `Distribution`、`PerturbationConfig`、`DispersionResult` 与 `monte_carlo_simulation`，支持 `auto/process/batch/gpu` 四种后端。
- **批量 MPM**：新增 `ballistic_sim/dynamics/batch_mpm.py`，6 维状态向量化 RK4 积分，自包含 ISA 大气与线性插值阻力表，支持 NumPy/CuPy 后端切换。
- **GPU 批量 MPM**：新增 `ballistic_sim/dynamics/gpu_mpm.py`，`GPUBatchMPMModel` 继承 `BatchMPMModel` 并使用 `require_cupy()` 导入守卫。
- **Monte Carlo 可视化**：新增 `ballistic_sim/viz/monte_carlo.py`，`plot_dispersion` 绘制落点云、CEP50/CEP90 圆与 2σ 散布椭圆。
- **CLI Monte Carlo 支持**：`ballistic_sim/cli.py` 新增 `--monte-carlo`、`--mc-backend`、`--mc-n-jobs`、`--mc-seed`、`--mc-samples`，输出 `mc_summary.json` 与 `montecarlo_plot.png`。
- **配置扩展**：`OptionsConfig` 新增 `mpm_use_spin`、`mpm_use_dynamic_alpha`、`monte_carlo`；`EnvironmentConfig` 新增 `delta_t`、`density_factor`；新增 `MonteCarloConfig`、`DistributionConfig`、`PerturbationConfig`。
- **Monte Carlo 与批量 MPM 测试**：新增 `tests/dynamics/test_batch_mpm.py`、`tests/test_monte_carlo.py`、`tests/viz/test_monte_carlo_plot.py`，并在 `tests/test_cli.py` 补充 MC 参数测试。
- **VehicleConfig 6-DOF 字段**：`Ix`、`It`、`x_cp_cg`、`twist_cal`。

### Changed

- `SixDOFControl` 下标约定适配 13 维状态。
- `OptionsConfig` 新增 `sixdof_reentry` 开关。
- `phases/builder.py` 的 MPM 相位构造现在读取 `cfg.options.mpm_use_spin` 与 `cfg.options.mpm_use_dynamic_alpha`。
- `simulator.py` 的 `make_atmosphere` 调用透传 `delta_t` 与 `density_factor`。

### Known Limitations

- 6-DOF 再入段默认采用零攻角、零横向角速度初始化；非零攻角再入需调用方显式提供 `quat`/`omega`。
- 高动态压力再入（>10 km/s）下积分步长可能过小，需进一步调优 `max_step` 与阻尼模型。
- batch/gpu 后端为统计近似模型，忽略自转偏流与动态攻角，仅支持 projectile 任务与 UniformWind。
- GPU 后端依赖 CuPy/CUDA，无对应环境时自动跳过或抛出 ImportError/RuntimeError。

### Version Milestones

| Tag | Stage | Description |
|-----|-------|-------------|
| `v0.2.0-stage1` | 阶段 2.1 | 6-DOF 闭环动力学与高保真再入（MVP） |
| `v0.2.0-stage2.2` | 阶段 2.2 | Monte Carlo 散布分析与 GPU/CPU 批量 MPM 仿真 |

[0.2.0]: https://github.com/bowlbowlhe-svg/ballistic-sim/releases/tag/v0.2.0
[0.1.0]: https://github.com/bowlbowlhe-svg/ballistic-sim/releases/tag/v0.1.0
