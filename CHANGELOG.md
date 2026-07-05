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

[0.2.0]: https://github.com/bowlbowlhe-svg/ballistic-sim/releases/tag/v0.2.0
[0.1.0]: https://github.com/bowlbowlhe-svg/ballistic-sim/releases/tag/v0.1.0
