# ballistic_sim v0.4.0 分阶段实施计划

**版本**：0.3.1 → 0.4.0  
**基线提交**：`652e8a9`（main）  
**质量基线**：pytest 614 passed / 2 skipped，覆盖率 86.84%，flake8 / mypy / black 全绿  
**计划目标**：完成审查报告中的 P0 项，并为 P1/P2/P3/P4 指明方向。

---

## 1. P0 总体策略

P0 共 6 项，按依赖关系拆为 **4 个阶段**，全部顺序执行：

| 阶段 | 主题 | 对应 P0 项 | 前置阶段 |
|------|------|-----------|---------|
| Phase 1 | 解耦 simulator ↔ builder，提取 context 模块 | #2 | 无 |
| Phase 2 | 统一公共 API 入口与版本号 | #1 | Phase 1 |
| Phase 3 | 统一配置→阶段入口与运行前业务校验 | #3、#4 | Phase 2 |
| Phase 4 | ModelCache 容量上限与 RHS 重复计算消除 | #5、#6 | Phase 3 |

所有阶段共享统一的验收标准：

- `pytest` 全绿（至少维持 614 passed / 2 skipped）；
- 覆盖率 **不下降**（≥ 86.84%）；
- `flake8`、`mypy`、`black` 全绿；
- CLI/API smoke 测试通过：`projectile`、`rocket`、`missile`、`icbm`、`suborbital` 至少各跑一单发；
- 不允许引入循环导入或破坏现有公开函数签名。

---

## 2. Phase 1：解耦 simulator ↔ builder，提取 context 模块

### 2.1 阶段目标
将 `simulator.py` 中的 `_resolve_wind`、`_resolve_terrain`、`_resolve_dynamics_context` 迁移到独立的 `ballistic_sim/context.py`，切断 `simulator.py` 与 `phases/builder.py`、`monte_carlo.py` 之间的交叉导入。

### 2.2 涉及文件/模块
- **新增**：`ballistic_sim/context.py`
- **修改**：
  - `ballistic_sim/simulator.py`（删除三个函数，改为从 context 导入）
  - `ballistic_sim/phases/builder.py`（`_resolve_terrain` 来源改为 context）
  - `ballistic_sim/monte_carlo.py`（`_resolve_dynamics_context` 来源改为 context）
  - `tests/test_simulator.py`（导入路径更新）
- **新增测试**：`tests/test_context.py`

### 2.3 开发任务（Developer Agent）
1. 创建 `ballistic_sim/context.py`，从 `simulator.py` 原样迁移：
   - `_resolve_wind(cfg: SimConfig) -> WindModel`
   - `_resolve_terrain(cfg: SimConfig) -> TerrainModel`
   - `_resolve_dynamics_context(cfg: SimConfig) -> DynamicContext`
2. 在 `context.py` 中补全所需导入：`SimConfig`、`apply_overrides`、`make_wind`、`parse_wind_profile`、`make_terrain`、`DynamicContext`、`make_atmosphere`、`make_aero`。
3. `simulator.py`：
   - 删除上述三个函数定义；
   - 在文件顶部 `from ballistic_sim.context import _resolve_dynamics_context`；
   - 保留 `simulate()` 中 `dyn_ctx = _resolve_dynamics_context(cfg)` 的调用不变。
4. `phases/builder.py`：
   - 三处 `from ballistic_sim.simulator import _resolve_terrain` 改为 `from ballistic_sim.context import _resolve_terrain`。
5. `monte_carlo.py`：
   - `from ballistic_sim.simulator import _resolve_dynamics_context` 改为 `from ballistic_sim.context import _resolve_dynamics_context`。
6. 检查 `ballistic_sim/__init__.py` 当前不加载 context，避免引入新循环导入。

### 2.4 测试任务（Tester Agent）
1. 更新 `tests/test_simulator.py` 中的导入路径为 `ballistic_sim.context`。
2. 新增 `tests/test_context.py`，覆盖：
   - 全部风模型解析（uniform/none/log/power/profile/composite/dryden）；
   - 地形解析（null/hilly/numpy/image，mock 文件即可）；
   - `_resolve_dynamics_context` 返回的 `DynamicContext` 包含 atmosphere/wind/aero/terrain；
   - 多 mission（projectile/rocket/missile/icbm/suborbital）context 构造不抛异常。
3. 运行完整回归：`pytest`、`coverage report`、`flake8 ballistic_sim tests`、`mypy ballistic_sim`、`black --check .`。
4. Smoke 测试：CLI 单发 `--mission projectile --preset M107` 与 `--mission rocket --rocket CZ-2F` 能生成结果。

### 2.5 验收标准
- [ ] `python -c "from ballistic_sim import simulator; from ballistic_sim.phases import builder; from ballistic_sim import monte_carlo"` 无循环导入错误。
- [ ] `tests/test_simulator.py` 与新增 `tests/test_context.py` 全部通过。
- [ ] pytest ≥ 614 passed / 2 skipped。
- [ ] 覆盖率 ≥ 86.84%。
- [ ] flake8 / mypy / black 全绿。
- [ ] CLI smoke 测试通过。

### 2.6 风险点
- **循环导入回弹**：`context.py` 若反向导入 `simulator.py` 会重新引入循环。必须严格只从 `config`、`models.*`、`dynamics.common` 导入。
- **私有函数被外部引用**：grep 确认仓库内只有 `builder.py`、`monte_carlo.py`、`tests/test_simulator.py` 引用这三个函数。

---

## 3. Phase 2：统一公共 API 入口与版本号

### 3.1 阶段目标
在 `ballistic_sim/__init__.py` 暴露核心符号并维护 `__all__`，同时把版本号从 `0.3.1` 提升到 `0.4.0`，并同步所有版本断言点。

### 3.2 涉及文件/模块
- `ballistic_sim/__init__.py`
- `pyproject.toml`
- `ballistic_sim/api/main.py`
- `tests/test_version.py`
- `scripts/release_check.py`
- `CHANGELOG.md`

### 3.3 开发任务（Developer Agent）
1. `ballistic_sim/__init__.py`：
   ```python
   from ballistic_sim.config import (
       PerturbationConfig,
       SimConfig,
       apply_overrides,
       load_config,
       save_config,
       validate_config,
   )
   from ballistic_sim.phases.builder import build_phases
   from ballistic_sim.simulator import SimResult, simulate

   __version__ = "0.4.0"
   __all__ = [
       "__version__",
       "SimConfig",
       "simulate",
       "build_phases",
       "validate_config",
       "load_config",
       "save_config",
       "apply_overrides",
       "PerturbationConfig",
       "SimResult",
   ]
   ```
   注意：先完成 Phase 1 的 context 提取后再做此步，避免 `__init__.py` 触发循环导入。
2. `pyproject.toml`：`version = "0.4.0"`。
3. `ballistic_sim/api/main.py`：FastAPI `version="0.4.0"`。
4. `tests/test_version.py`：`TARGET_VERSION = "0.4.0"`。
5. `scripts/release_check.py`：`TARGET_VERSION = "0.4.0"`。
6. `CHANGELOG.md`：在顶部新增 `## [0.4.0] - 2026-07-05` 章节（可先留空，待 P0 全部完成后补充条目）。

### 3.4 测试任务（Tester Agent）
1. 新增/更新 `tests/test_public_api.py`：
   - `from ballistic_sim import SimConfig, simulate, build_phases, validate_config, load_config, save_config, apply_overrides, PerturbationConfig, SimResult` 成功；
   - `import ballistic_sim` 后 `set(ballistic_sim.__all__)` 与 `dir(ballistic_sim)` 暴露集合一致；
   - `ballistic_sim.__version__ == "0.4.0"`。
2. 运行 `tests/test_version.py` 与 `tests/test_api.py`（FastAPI health/version 断言）。
3. 运行完整回归与 lint。

### 3.5 验收标准
- [ ] `from ballistic_sim import *` 仅暴露 `__all__` 中列出的符号。
- [ ] `ballistic_sim.__version__`、`pyproject.toml`、`FastAPI app.version`、`tests/test_version.py` 四者一致为 `0.4.0`。
- [ ] pytest / coverage / lint 全绿。
- [ ] API `/health` 返回 `version: "0.4.0"`。

### 3.6 风险点
- **循环导入**：`__init__.py` 同时导入 `simulator`、`phases.builder`、`config`，若 Phase 1 未彻底解耦，此处会引爆循环导入。必须等 Phase 1 验收通过。

---

## 4. Phase 3：统一配置→阶段入口与运行前业务校验

### 4.1 阶段目标
CLI、API、预设模块最终都收敛到“拿到 `SimConfig` 后调用 `build_phases(cfg)`”；`simulate()` 与 CLI `main()` 在运行前统一调用 `validate_config()`，遇到 ERROR 级 issue 拒绝运行并给出清晰提示。

### 4.2 涉及文件/模块
- `ballistic_sim/cli.py`
- `ballistic_sim/api/main.py`
- `ballistic_sim/simulator.py`
- `ballistic_sim/config.py`
- `ballistic_sim/phases/builder.py`
- `ballistic_sim/presets/projectiles.py`
- `ballistic_sim/presets/missiles.py`
- `ballistic_sim/presets/rockets.py`
- `tests/test_cli.py`
- `tests/test_api.py`
- `tests/test_simulator.py`
- 新增 `tests/test_config_validation.py`

### 4.3 开发任务（Developer Agent）

#### A. 补齐 projectile 预设到 `build_phases` 的缺口
当前 `projectile_phases()` 手动构造 `MPMDynamics` 并指定 G1/G7 阻力表，而 `build_phases` 的 `_build_projectile_phases` 不识别阻力表。

1. 在 `VehicleConfig` 中增加可选字段：
   ```python
   drag_law: Optional[Literal["G1", "G7"]] = Field(default=None, description="弹丸阻力定律 G1/G7")
   ```
   `VehicleConfig` 当前 `extra="forbid"`，新增字段不会破坏 pydantic 校验，但需同步更新所有手写 `VehicleConfig(...)` 调用处（若无该字段则默认 `None`，保持原行为）。
2. 修改 `phases/builder.py::_build_projectile_phases`：
   - 根据 `cfg.vehicle.drag_law` 选择 `DRAG_G1` 或 `DRAG_G7`；
   - 若 `drag_law is None`，保持现有默认行为（form_factor 作为阻力系数）。
3. 修改 `presets/projectiles.py::_projectile_config_from_preset`：
   - 读取 `p.get("drag", "G1")` 并设置 `vehicle=VehicleConfig(..., drag_law=...)`。

#### B. 收敛 CLI 与 API 的 preset 调用
目标：CLI/API 中所有分支最终都拿到 `SimConfig`，再统一 `build_phases(cfg)`。

1. `ballistic_sim/cli.py`：
   - `_build_projectile_config`：返回 `cfg` 后调用 `build_phases(cfg)`，不再直接调用 `m107_phases()` / `projectile_phases()`。
   - `_build_missile_config`：改为使用 `missile_full_config(name)`（已含 stages）+ `build_phases(cfg)`；原 `missile_config` / `missile_phases` 保留兼容但标记 deprecated。
   - `_build_rocket_config`：改为使用 `rocket_full_config(name)` + `build_phases(cfg)`。
   - `_build_icbm_config`：YAML 预设路径已返回 `missile_full_config` + `missile_full_chain`，改为 `missile_full_config` + `build_phases`；占位单级链暂时保留，但内部也应由 `build_phases(cfg)` 生成（可先完成 YAML 路径收敛，占位链作为 P1 遗留）。
   - `_build_suborbital_config`：占位 builder 直接构造 phases，改为先构造 `SimConfig` 再 `build_phases(cfg)`（可能需要扩展 `build_phases` 支持 suborbital 单级，若改动大则允许本阶段保留占位链并在 P1 彻底迁移）。
   - `_build_config_and_phases`：最终统一为 `cfg = ...; phases = build_phases(cfg)`。

2. `ballistic_sim/api/main.py`：
   - `_build_config_and_phases` 中所有分支统一改为 `cfg = ...; phases = build_phases(cfg)`；
   - 删除对 `m107_phases()`、`missile_phases()`、`rocket_phases()` 的直接调用。
   - `_run_simulation` 与 `_run_monte_carlo` 在调用 `simulate` 前执行 `validate_config(cfg)`。

#### C. 运行前业务校验
1. `ballistic_sim/simulator.py` 的 `simulate(cfg, phases, ...)`：
   - 函数入口调用 `validate_config(cfg)`；
   - 若存在 `severity == "ERROR"` 的 issue，抛出 `ValueError`，消息格式：
     ```text
     SimConfig validation failed:
       [ERROR] vehicle.thrust_N: 推力/重量比 T/W = 0.512 <= 1，动力任务无法起飞
     ```
2. `ballistic_sim/cli.py` 的 `main()`：
   - 在 `cfg, phases = _build_config_and_phases(args)` 之后调用 `validate_config(cfg)`；
   - 遇 ERROR 打印到 `sys.stderr` 并 `raise SystemExit(1)`。
3. `ballistic_sim/api/main.py`：
   - `_run_simulation`、`_run_monte_carlo`、`_solve_fire_control` 在运行前调用 `validate_config(cfg)`；
   - 遇 ERROR 抛出 `HTTPException(status_code=400, detail=formatted_message)`。

#### D. 预设函数 deprecation 标记
- `m107_phases()`、`projectile_phases()`、`missile_phases()`、`rocket_phases()`、`missile_full_chain()`、`rocket_full_chain()` 保留签名，但在 docstring 中标注：
  > Deprecated: v0.4.0 后 CLI/API 统一使用 `build_phases(cfg)`，本函数将在 v0.5.0 移除。
- 内部实现可改为 `return build_phases(cfg)`（对仍能拿到 cfg 的函数）或保持原样（对需要额外参数的函数，如 `rocket_phases(cfg, name=name)`）。

### 4.4 测试任务（Tester Agent）
1. 新增 `tests/test_config_validation.py`：
   - `validate_config` 识别 T/W <= 1、推力与工作时间矛盾、制导律缺少目标经纬度等 ERROR；
   - `simulate()` 遇到 ERROR 配置抛出 `ValueError` 且消息包含 issue path；
   - 纯 WARNING 配置仍可运行。
2. 更新 `tests/test_cli.py`：
   - 增加一个 ERROR 配置 CLI 拒绝运行的测试（例如 `--mission rocket --config configs/sample_icbm.yaml` 若配置合法则用另一个手工构造的非法 YAML）；
   - 验证现有 smoke 测试仍通过。
3. 更新 `tests/test_api.py`：
   - 测试 `POST /simulate/rocket` 带 ERROR 配置返回 400；
   - 测试 `POST /config/validate` 行为不变。
4. 回归测试：所有 mission 的 CLI 与 API smoke 用例通过。

### 4.5 验收标准
- [ ] CLI `--config`、预设分支最终都调用 `build_phases(cfg)`。
- [ ] API 所有 `/simulate/{mission}`、`/simulate/monte-carlo` 分支都调用 `build_phases(cfg)`。
- [ ] `simulate()` 入口对 ERROR 配置抛出 `ValueError`。
- [ ] CLI 对 ERROR 配置退出码非 0，并在 stderr 打印清晰提示。
- [ ] API 对 ERROR 配置返回 HTTP 400。
- [ ] WARNING-only 配置仍可正常运行。
- [ ] pytest / coverage / lint 全绿，覆盖率不下降。

### 4.6 风险点
- **suborbital / ICBM 占位 builder 迁移不完全**：这两个分支当前手动拼 phase。若强行在 P0 全部迁移，可能引入过多新逻辑。建议本阶段先把 `--config`、projectile、missile、rocket 收敛；suborbital/icbm 占位 builder 允许保留，但内部尽量用 `build_phases`，剩余工作量放入 P1。
- **drag_law 字段的兼容**：新增 `VehicleConfig.drag_law` 后，所有已有 YAML/JSON 配置文件无需改动（默认 `None`），但手写 `VehicleConfig(...)` 的位置需确认无 `extra="forbid"` 冲突。
- **validate_config 与现有测试冲突**：部分现有测试可能使用 T/W <= 1 的配置。需要检查并修正这些测试，或在测试中显式关闭校验（不推荐），优先修正配置使其合理。

---

## 5. Phase 4：ModelCache 容量上限与 RHS 重复计算消除

### 5.1 阶段目标
- 给 `ModelCache` 增加容量上限，避免无界增长；
- 消除 `PointMassDynamics.rhs` 中 `_aero_env` 与 `rhs` 重复查大气/风/坐标转换的问题。

### 5.2 涉及文件/模块
- `ballistic_sim/dynamics/common.py`
- `ballistic_sim/dynamics/point_mass.py`
- `tests/test_cache.py`
- 新增 `tests/test_dynamics_point_mass.py`

### 5.3 开发任务（Developer Agent）

#### A. ModelCache 容量上限
策略：使用标准库实现 bounded LRU，不引入新依赖（避免 `cachetools`）。

1. 在 `ballistic_sim/dynamics/common.py` 中新增轻量 `LRUDict`：
   ```python
   class LRUDict:
       def __init__(self, maxsize: int = 10000):
           self._maxsize = max(maxsize, 1)
           self._data: OrderedDict[Any, Any] = OrderedDict()

       def get(self, key: Any) -> Any:
           if key not in self._data:
               return None
           self._data.move_to_end(key)
           return self._data[key]

       def put(self, key: Any, value: Any) -> None:
           self._data[key] = value
           self._data.move_to_end(key)
           if len(self._data) > self._maxsize:
               self._data.popitem(last=False)
   ```
2. `ModelCache.__init__(self, enabled: bool = True, maxsize: int = 10000)`：
   - 将 `_atm`、`_wind`、`_aero`、`_coeff`、`_generic` 改为 `LRUDict(maxsize)`。
   - `stats()` 仍返回各桶条目数。
3. `DynamicContext.__post_init__` 中 `self.cache = ModelCache(enabled=self.use_cache, maxsize=...)`，默认容量 10000；可通过 `options.cache_maxsize` 覆盖（可选，若增加该字段需在 `OptionsConfig` 中加 `cache_maxsize: int = 10000`）。

#### B. RHS 重复计算消除
目标：`_aero_env` 计算一次并返回可复用的 `v_rel`、`h`、`wind_enu`。

1. 在 `PointMassDynamics` 中新增私有方法：
   ```python
   def _aero_env_full(self, ctx, r, v, lat0, lon0):
       """返回 (AeroEnv, v_rel, h, wind_enu_or_None)。"""
       ...
   ```
   - ENU 分支：`h = float(r[2])`，`v_rel = v.copy()`，`wind_enu = dyn_ctx.wind(h) if dyn_ctx.wind else None`。
   - ECI 分支：计算 `r_ecef`、`_h`、wind、`_aero_env` 所需的全部量。
2. 保留 `_aero_env(...)` 作为包装，仅返回 `AeroEnv`，供 `telemetry()` 等外部调用者使用。
3. 修改 `rhs()`：
   - 调用 `_aero_env_full(...)` 得到 `(env, v_rel, h, wind_enu)`；
   - ENU 分支直接使用 `v_rel` 与 `env`；
   - ECI 分支直接使用 `v_rel` 与 `env`，不再重复 `eci_to_ecef`、`ecef_to_geodetic`、`ctx.wind(h)`、`relative_velocity_eci`。
4. 确保数值结果与重构前一致（通过回归测试保证）。

### 5.4 测试任务（Tester Agent）
1. 更新 `tests/test_cache.py`：
   - `test_cache_capacity_bound`：连续插入 10010 个不同高度，断言 `cache.stats()["atmosphere"] <= 10000`；
   - `test_cache_lru_eviction`：插入两个 key，访问第一个，再大量插入，断言最近访问的 key 仍在；
   - `test_cache_clear_with_bounded_cache`：`clear()` 后所有桶为 0。
2. 新增 `tests/test_dynamics_point_mass.py`：
   - `test_rhs_numerical_regression`：使用固定 cfg 跑 `simulate`，与基线 trajectory 在 `1e-9` 相对容差内一致；
   - `test_aero_env_full_reuse`：mock `ctx.atmosphere` / `ctx.wind`，断言 `_aero_env_full` 只调用一次模型（rhs 内部不再重复调用）；
   - ENU 与 ECI 两种 frame 的 smoke 测试。
3. 使用 `scripts/benchmark.py` 或新增微基准验证 RHS 调用次数减少 / 耗时下降（定性即可，不强求 CI）。

### 5.5 验收标准
- [ ] `ModelCache` 各桶条目数不超过 `maxsize`。
- [ ] `DynamicContext.clear_cache()` 仍清空所有缓存。
- [ ] `PointMassDynamics.rhs` 不再重复调用 `eci_to_ecef` / `ecef_to_geodetic` / `ctx.wind`。
- [ ] 数值回归：相同配置下 trajectory 与基线差异在容差范围内。
- [ ] pytest / coverage / lint 全绿，覆盖率 ≥ 86.84%。

### 5.6 风险点
- **LRUDict 性能**：使用 `OrderedDict` 在大量查询下可能略慢于无界 dict，但容量上限带来的内存收益远大于此。若性能敏感，后续可替换为 `cachetools.LRUCache`。
- **`_aero_env` 签名变更影响外部调用者**：grep 确认外部只通过 `rhs` 与 `telemetry` 调用。`telemetry` 仍使用 `_aero_env`（返回 `AeroEnv`），签名不变。
- **ModelCache maxsize 默认值选择**：10000 对单发足够；MC process 后端每个子进程独立，10000 也足够。若需更大，可通过 `OptionsConfig.cache_maxsize` 调整。

---

## 6. P1 / P2 / P3 / P4 后续方向（概要）

| 优先级 | 方向 | 内容 |
|--------|------|------|
| **P1** | 完成配置→阶段全面收敛 | 1. 移除 `m107_phases`、`projectile_phases`、`missile_phases`、`rocket_phases` 等 deprecated 函数；<br>2. ICBM / suborbital CLI 占位 builder 完全改为 `SimConfig + build_phases`；<br>3. `VehicleConfig` 中补充更多预设所需字段（如 `twist_cal`、`Ix`、`It` 默认值），减少 `_projectile_dynamics_from_preset` 这类绕过 build_phases 的函数；<br>4. 将 `simulate(cfg, phases)` 的 `phases` 参数改为可选（默认 `None`），并最终标记为 deprecated。 |
| **P2** | API 与事件系统重构 | 1. 统一 `SimResult` 元数据与事件格式；<br>2. 引入结构化日志替代 print；<br>3. 改进 `validate_config` 错误信息的国际化/字段路径可读性；<br>4. 评估 `simulate` 返回 dataclass 还是 pydantic model，统一 CLI/API/测试序列化。 |
| **P3** | 文档与示例 | 1. 更新 `docs/user_guide.md`，以 `SimConfig + build_phases + simulate` 为主示例；<br>2. 新增 Jupyter notebook 示例；<br>3. 完善 API 文档与 OpenAPI schema。 |
| **P4** | 体验增强 | 1. GUI 预设选择器直接返回 `SimConfig`；<br>2. 可视化后端解耦（matplotlib/plotly 可选）；<br>3. 自动化性能基准与回归告警。 |

---

## 7. 总体时间线建议

| 阶段 | 预估工作量 | 关键产出 |
|------|-----------|---------|
| Phase 1 | 0.5 ~ 1 d | `context.py` + 导入修正 + 回归绿 |
| Phase 2 | 0.5 d | `__init__.py` 公共 API + 版本号 `0.4.0` |
| Phase 3 | 1.5 ~ 2 d | CLI/API 收敛到 `build_phases` + 校验前置 |
| Phase 4 | 1 ~ 1.5 d | Bounded cache + RHS 去重 + 回归绿 |
| 合计 | **3.5 ~ 5 d** | v0.4.0 发布就绪 |

---

## 8. 附录：P0 快速检查清单

- [ ] `from ballistic_sim import SimConfig, simulate, build_phases, validate_config, load_config, save_config, apply_overrides, PerturbationConfig` 成功。
- [ ] `ballistic_sim.__all__` 维护且与 `dir()` 一致。
- [ ] `simulator.py` 不再包含 `_resolve_wind` / `_resolve_terrain` / `_resolve_dynamics_context`。
- [ ] `phases/builder.py` 与 `monte_carlo.py` 从 `context.py` 导入上述函数。
- [ ] CLI `--config` 与所有预设路径最终调用 `build_phases(cfg)`。
- [ ] API 所有仿真入口最终调用 `build_phases(cfg)`。
- [ ] `simulate()` 与 CLI `main()` 对 ERROR 配置拒绝运行。
- [ ] `ModelCache` 配置容量上限，stats 中各桶不超限。
- [ ] `PointMassDynamics.rhs` 数值结果与基线一致。
- [ ] pytest 全绿、覆盖率 ≥ 86.84%、flake8/mypy/black 全绿。
