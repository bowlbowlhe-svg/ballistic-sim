# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/lang/zh-CN/).

## [0.6.2] - 2026-07-09

### Fixed

- 6-DOF 再入 builder 集成：当 `options.sixdof_reentry=True` 时，终点阶段使用与再入段同维度的 `SixDOFDynamics`，避免 `3-DOF ECI -> 6-DOF ENU` 状态切换错误。
- `simulate()` 现在支持不同阶段状态维度不一致的轨迹拼接（低维数组补零），使 3-DOF -> 6-DOF 任务链能够完整运行。

### Added

- 新增 `test_simulate_missile_with_sixdof_reentry`，用 `SRBM_600` 预设验证 6-DOF 再入全链路仿真。

## [0.6.1] - 2026-07-09

### Added

- 补充覆盖率缺口测试：
  - `presets/loader.py` 异常分支（文件不存在、发射场缺失、aero 表长度匹配）。
  - `context.py` 未知风/地形模型、`composite` 风剖面文件读取、未知 mission 回退分支。
  - `phases/events.py` 抛整流罩 ECI 分支与 `mode="q"`/无效模式分支。
  - `simulator.py` `integration_failed` 失败路径。

### Changed

- 全量测试覆盖率从 87.47% 提升到 87.99%。

## [0.6.0] - 2026-07-09

### Removed

- `simulate(..., phases=...)` 参数正式移除；`test_simulate_phases_deprecation.py` 删除。

### Changed

- 所有调用方必须改用 `simulate(cfg)`。

## [0.5.0] - 2026-07-06

### Removed

- 移除 6 个已弃用的 phase 工厂函数（P1 Stage 3）。

### Changed

- `simulate(cfg, phases=None)` 现在自动调用 `build_phases(cfg)` 构造阶段；显式传入 `phases` 自 v0.5.0 起已弃用，请改用 `simulate(cfg)`。
- CLI、API、Monte Carlo、制导 targeting、GUI 等内部调用方全面收敛到 `SimConfig + build_phases`，不再显式传递 phases。
- targeting 模块不再显式传 phases，阻力开关通过 GuidanceConfig.use_drag 表达。

### Added

- `VehicleConfig` 新增 `twist_cal` / `Ix` / `It` / `x_cp_cg` 默认值，builder 自动将其传入 `MPMDynamics`。

## [0.4.0] - 2026-07-05

### Added

- 新增 `ballistic_sim/context.py`，集中管理 `_resolve_wind`、`_resolve_terrain`、`_resolve_dynamics_context`，彻底解耦 `simulator.py` ↔ `phases/builder.py` ↔ `monte_carlo.py`。
- 新增 `tests/test_context.py` 与 `tests/test_public_api.py`，覆盖上下文解析与公共 API 入口。
- 新增 `tests/test_config_validation_integration.py`，验证 `simulate()` / CLI / API 在运行前对 ERROR 配置的拒绝行为。
- 新增 `tests/test_dynamics_point_mass.py` 与 `tests/golden/point_mass_rhs_baseline.json`，对 `PointMassDynamics.rhs` 做数值回归并验证风场/大气只查询一次。
- `VehicleConfig` 新增 `drag_law: Literal["G1", "G7"]` 可选字段，`build_phases` 对弹丸任务可自动选择标准阻力表。
- `OptionsConfig` 新增 `cache_maxsize` 字段，支持自定义 `ModelCache` 容量上限。

### Changed

- 统一公共 API 入口：`ballistic_sim/__init__.py` 暴露 `SimConfig`、`simulate`、`build_phases`、`validate_config`、`load_config`、`save_config`、`apply_overrides`、`PerturbationConfig`、`SimResult` 等核心符号，并维护 `__all__`。
- CLI、API、预设模块统一收敛到 `SimConfig + build_phases(cfg)`；`m107_phases()` / `projectile_phases()` 等函数保留签名但改为调用 `build_phases(cfg)` 并标记 deprecated。
- `simulate()` / CLI `main()` / API 仿真端点在运行前统一调用 `validate_config(cfg)`，ERROR 级 issue 拒绝运行并给出清晰字段路径提示。
- `ModelCache` 改用基于 `OrderedDict` 的 `LRUDict`，默认容量上限 10000，避免长时间积分或大样本 Monte Carlo 导致 OOM。
- `PointMassDynamics.rhs` 引入 `_aero_env_full()`，一次性完成坐标转换、风场/大气查询与相对速度计算，消除重复查表；数值结果与重构前保持一致。
- 版本号统一升级为 `0.4.0`：`ballistic_sim/__init__.py`、`pyproject.toml`、`ballistic_sim/api/main.py`、`tests/test_version.py`、`scripts/release_check.py`。

## [0.3.1] - 2026-07-05

### Fixed

- Monte Carlo `process` 后端在 pytest/CLI 嵌套调用下的稳定性：当检测到 pytest 收集器内部运行时自动降级为 `batch`（如适用）或串行 `process`，避免 Windows spawn 子进程递归挂起。
- 提升 `ballistic_sim/guidance/proportional_navigation.py` 测试覆盖率，补充 `make_static_target_provider`、generalized 零速度、零视线速率、异常输入等分支测试。
- 降低 matplotlib 中文字体缺失警告：在 `ballistic_sim/viz/__init__.py` 配置常见中文字体列表并关闭 Unicode 负号，实现有字体时显示中文、无字体时单次回退警告。
- 补强 `ballistic_sim/viz/interactive3d.py` 缺失分支测试，覆盖 plotly 缺失时 `plot_trajectory_3d`/`demo` 的 `ImportError`、空结果保护、HTML 写出失败路径。

### Changed

- 版本号统一更新为 `0.3.1`：`ballistic_sim/__init__.py`、`ballistic_sim/api/main.py`、`pyproject.toml`、`tests/test_version.py`、`scripts/release_check.py`。

[0.3.1]: https://github.com/bowlbowlhe-svg/ballistic-sim/releases/tag/v0.3.1

## [0.3.0] - 2026-07-05

### Added

- **性能优化与工程基座（阶段 1）**：
  - 新增 `ballistic_sim/dynamics/common.py::ModelCache`，缓存大气、风、气动系数插值结果，减少 RHS 重复计算。
  - `simulator.py` 引入 `_resolve_dynamics_context` 缓存与 `reuse_context` 开关。
  - `monte_carlo.py` batch/process 后端支持更粗粒度并行与 `maxtasksperchild` 防内存泄漏。
  - 新增 `scripts/benchmark.py` 性能基准脚本，输出 `benchmark_results.json`。
  - 新增 `tests/test_performance.py`、`tests/test_cache.py` 覆盖缓存与性能断言。
- **配置驱动与业务校验强化（阶段 2）**：
  - `SimConfig` 扩展跨字段校验：T/W > 1、burn_time 与推力匹配、目标经纬度与任务类型匹配、Monte Carlo 参数合理性等。
  - 新增 `load_config` / `save_config` 支持 YAML/JSON 配置文件作为一等公民。
  - `phases/builder.py` 重构支持多级 stage 列表，自动插入分离、滑行、再入阶段。
  - CLI 新增 `--config` 参数；API 新增 `POST /config/validate` 端点。
  - 新增 `configs/sample_icbm.yaml` 与多份配置示例。
- **制导律扩展（阶段 3）**：
  - 新增闭环比例导引 `ballistic_sim/guidance/proportional_navigation.py`（TPN/GPN）。
  - 新增增广显式制导 `ballistic_sim/guidance/aag.py`。
  - 新增再入制导 `ballistic_sim/guidance/reentry_guidance.py`（阻力加速度剖面/弹道系数）。
  - 新增能量管理制导 `ballistic_sim/guidance/energy_management.py`。
  - `GuidanceConfig.guidance_law` 扩展为 `none | proportional | peg | aag | reentry | energy`。
- **导弹/火箭预设链完整化（阶段 4）**：
  - `presets/missiles.yaml`、`presets/rockets.yaml` 扩展多级推力、级间分离、整流罩抛罩、再入飞行器释放。
  - `phases/events.py` 新增 `make_stage_separation_event`、`make_fairing_jettison_event`。
  - `state_switch.py` 优化 ECI → ENU → 6-DOF 投影，支持再入点姿态初始化。
  - 新增完整 ICBM 任务链测试与 CZ 火箭多级分离测试。
- **实时三维可视化与交互监控（阶段 5）**：
  - 新增 `ballistic_sim/viz/interactive3d.py`（基于 Plotly）、`viz/earth.py`、`viz/dashboard.py`。
  - GUI 右侧结果区增加 3D 视图 tab；`web/index.html` 增加 Plotly 3D 地球与轨迹展示。
  - API 新增 `POST /viz/trajectory3d` 端点，支持 JSON/ECEF 点序列或 HTML 输出。
- **CI/CD、文档与发布（阶段 6）**：
  - 新增 `.github/workflows/ci.yml`：Python 3.10/3.11/3.12 矩阵，flake8、pytest（`MPLBACKEND=Agg`）、mypy、coverage（fail-under=80）。
  - 新增 `.github/workflows/benchmark.yml`：运行 `scripts/benchmark.py` 并上传 artifact。
  - 新增 `docs/user_guide.md`、`docs/guidance_laws.md`、`docs/performance.md`。
  - 新增 `scripts/release_check.py` 发布前检查脚本。
  - 新增 `tests/test_version.py` 统一断言版本号。

### Changed

- `__version__`、`pyproject.toml` 版本号、`FastAPI` 版本号统一为 `0.3.0`。
- 全局测试基线从 585 提升至 ≥602，整体覆盖率保持 ≥85%。
- CI 中显式跳过 GPU/显示环境相关测试，避免无 GPU/显示服务时失败。

### Version Milestones

| Tag | Stage | Description |
|-----|-------|-------------|
| `v0.3.0-stage1` | 阶段 1 | 性能优化与工程基座 |
| `v0.3.0-stage2` | 阶段 2 | 配置驱动与业务校验 |
| `v0.3.0-stage3` | 阶段 3 | 制导律扩展 |
| `v0.3.0-stage4` | 阶段 4 | 导弹/火箭预设链完整化 |
| `v0.3.0-stage5` | 阶段 5 | 实时 3D 可视化 |
| `v0.3.0` | 阶段 6 | CI/CD、文档与正式发布 |

[0.3.0]: https://github.com/bowlbowlhe-svg/ballistic-sim/releases/tag/v0.3.0

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

## [0.2.0] - 2026-07-05

### Added

- **6-DOF 刚体动力学**：新增 `ballistic_sim/dynamics/six_dof.py`，状态向量 13 维 `[r(3), v(3), q(4), omega_y, omega_z, p]`，支持轴对称弹、四元数姿态、气动力/力矩、控制矩注入。
- **6-DOF 推力接口**：`SixDOFDynamics` 新增 `thrust_N` 与 `burn_time_s` 字段；`options["thrust"]=True` 时沿弹轴施加轴向推力，`telemetry()` 返回 `thrust_N`。
- **导弹弹道整形**：`ballistic_sim/guidance/targeting.py` 实现 `max_range_missile`、`solve_missile_range`、`solve_missile_target`，基于 YAML 导弹预设与 PoweredECI 装配链反解 kick/az，支持 minenergy / loft / depressed 分支与 cross-track 误差校验。
- **后处理补充**：`simulator.py` 的 `_postprocess` 现在输出 `lat0`、`lon0`、`r0_ecef`，为射程 sanity check 与地理 targeting 提供发射点基准。
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
- **真实风场与地形集成（阶段 2.3）**：
  - `EnvironmentConfig` 扩展 `wind_model`、`wind_profile_path/text`、对数/幂律/Dryden 参数；
    `terrain_model`、`terrain_extent`、`terrain_max_height` 与 `terrain_enabled` 兼容开关。
  - `simulator.py` 新增 `_resolve_wind` / `_resolve_terrain`，按配置自动装配风场与地形模型。
  - `DynamicContext` 新增 `terrain` 字段；`make_ground_event` 支持 ENU/ECI 地形高程参与落地判据。
  - `phases/builder.py` 与 `PoweredPhase` / `CoastingPhase` / `ReentryPhase` / `TerminalPhase`
    接入地形模型，避免循环导入。
  - 修复 `PointMassDynamics` ECI 模式下风场固定使用 `h=0` 的 bug，改为按几何高度 `h` 查询。
  - CLI 新增 `--wind-model`、`--wind-profile`、`--terrain-model`、`--terrain-path`、
    `--terrain-extent` 参数。
  - `weather` extra 增加 `cfgrib>=0.9`、`xarray>=2023.0`（懒加载，不强制安装）。
  - 新增/扩展测试：`tests/test_simulator.py`、`tests/phases/test_events_terrain.py`、
    `tests/test_cli_wind_terrain.py`、`tests/test_terrain_integration.py`、
    `tests/dynamics/test_point_mass.py`，以及验收阶段补充的
    `tests/test_wind_terrain_boundary.py`。
- **GUI 可视化入口（阶段 2.4）**：
  - 新增 `ballistic_sim/gui/` 包：tkinter 主应用 `BallisticGuiApp`、
    表单自动生成 `fields.py`、后台仿真线程 `runner.py`、配置/预设构造 `builder.py`。
  - 顶部工具栏支持任务/预设切换、Run、Load YAML、Save YAML。
  - 左侧 `ttk.Notebook` 参数面板：Vehicle / Launch / Environment / Guidance /
    Options / Monte Carlo，按 pydantic 字段自动生成控件。
  - 右侧结果面板：文本摘要 + matplotlib `FigureCanvasTkAgg` 嵌入高度-射程曲线。
  - 新增顶层入口 `launch_gui.py`；`ballistic_sim/cli.py` 新增 `--gui` 参数，
    启动 GUI 而非 CLI 运行。GUI 导入采用懒加载，缺失 tkinter 时不破坏 CLI。
  - 新增测试 `tests/test_gui.py`，覆盖导入、表单生成、应用构造、YAML 往返、
    CLI `--gui` 集成；无显示环境时自动 skip。
- **Web API 与服务（阶段 2.5）**：
  - 新增 `ballistic_sim/api/` 包：
    `models.py` 定义请求/响应 pydantic 模型；
    `dependencies.py` 提供 `require_fastapi()` / `require_uvicorn()` 导入守卫；
    `main.py` 构造 FastAPI 应用并注册路由；
    `__main__.py` 支持 `python -m ballistic_sim.api` 启动 uvicorn；
    `__init__.py` 导出 `create_app()` 与全局 `app` 实例。
  - API 端点：
    `GET /health` 健康检查；
    `POST /simulate/{mission}` 单发仿真（projectile/missile/rocket/icbm/suborbital）；
    `POST /simulate/monte-carlo` Monte Carlo 散布分析；
    `POST /firecontrol/solve` 火控诸元反解（projectile/missile）。
  - 静态文件：`web/index.html` 最小前端表单，调用 API 并展示 JSON 结果；
    FastAPI 通过 `StaticFiles` 挂载 `web/` 到 `/`。
  - CORS 默认允许全部来源，便于本地前端联调。
  - `ballistic_sim/cli.py` 新增 `--serve`、`--host`、`--port`、`--reload`，
    可通过 `ballistic-sim --serve` 一键启动 Web 服务；API 模块采用懒加载，
    未安装 fastapi 时不破坏 CLI/GUI。
  - 静态文件路径基于 `__file__` 解析为绝对路径，确保从任意目录启动都能找到 `web/`。
  - 新增 `tests/test_api.py`，使用 `fastapi.testclient.TestClient` 覆盖各端点；
    fastapi 缺失时自动 skip。
  - 为支持 `TestClient`，在 `.venv` 中安装 `httpx`（测试依赖）。

### Changed

- `SixDOFControl` 下标约定适配 13 维状态。
- `OptionsConfig` 新增 `sixdof_reentry` 开关。
- `phases/builder.py` 的 MPM 相位构造现在读取 `cfg.options.mpm_use_spin` 与 `cfg.options.mpm_use_dynamic_alpha`。
- `simulator.py` 的 `make_atmosphere` 调用透传 `delta_t` 与 `density_factor`。
- `--mission` 改为可选参数；未提供且未使用 `--gui` 时给出明确错误提示。

### Known Limitations

- 6-DOF 再入段默认采用零攻角、零横向角速度初始化；非零攻角再入需调用方显式提供 `quat`/`omega`。
- 高动态压力再入（>10 km/s）下积分步长可能过小，需进一步调优 `max_step` 与阻尼模型。
- batch/gpu 后端为统计近似模型，忽略自转偏流与动态攻角，仅支持 projectile 任务与 UniformWind。
- GPU 后端依赖 CuPy/CUDA，无对应环境时自动跳过或抛出 ImportError/RuntimeError。
- GRIB2 空间插值在本阶段未完整实现，`GRIB2WindModel` 仅提供最近高度层兜底。
- GUI 为阶段 2.4 MVP：当前仅嵌入高度-射程曲线，后续可扩展 3D 轨迹、地面航迹、Monte Carlo 落点图等切换视图。
- Web API 为阶段 2.5 MVP：当前仅提供核心仿真/火控/Monte Carlo 端点，
  后续可扩展 YAML 上传下载、任务队列、WebSocket 实时推送等。

### Version Milestones

| Tag | Stage | Description |
|-----|-------|-------------|
| `v0.2.0-stage1` | 阶段 2.1 | 6-DOF 闭环动力学与高保真再入（MVP） |
| `v0.2.0-stage2.2` | 阶段 2.2 | Monte Carlo 散布分析与 GPU/CPU 批量 MPM 仿真 |
| `v0.2.0-stage2.3` | 阶段 2.3 | 真实风场与地形集成 |
| `v0.2.0-stage2.4` | 阶段 2.4 | GUI 可视化入口 |
| `v0.2.0-stage2.5` | 阶段 2.5 | Web API 与服务 |
| `v0.2.0` | 阶段 2 收尾 | 占位功能补齐、导弹弹道整形、6-DOF 推力接口、版本发布 |

[0.6.0]: https://github.com/bowlbowlhe-svg/ballistic-sim/releases/tag/v0.6.0
[0.5.0]: https://github.com/bowlbowlhe-svg/ballistic-sim/releases/tag/v0.5.0
[0.2.0]: https://github.com/bowlbowlhe-svg/ballistic-sim/releases/tag/v0.2.0
[0.1.0]: https://github.com/bowlbowlhe-svg/ballistic-sim/releases/tag/v0.1.0
