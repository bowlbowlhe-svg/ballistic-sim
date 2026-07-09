# 后续计划（Next Steps）

> 本文件记录 v0.5.0 之后的潜在改进方向，按需继续。

## 短期（v0.5.x）

- [x] **清理 phase 显式传参的遗留例外**：
  - 已在 `GuidanceConfig` 新增 `use_drag` 与 gamma 整形字段，`builder.py` 构造 `PoweredECIDynamics` 时读取；`targeting.py` 改为 `simulate(cfg)`。
  - 当前 `ballistic_sim/` 内部已无 `simulate(cfg, phases=...)` 调用。
- [ ] **把占位测试迁移到 builder**：
  - `tests/test_icbm.py::_placeholder_icbm_phases()` 仍手工构造 phase 链。
  - 等 ICBM 预设与 builder 成熟后，改为 `rocket_full_config` / `missile_full_config` + `build_phases(cfg)`。
- [x] **提升慢测试运行效率**：
  - 已安装 `pytest-xdist`，`pyproject.toml` 默认 `addopts = "... -n auto"`。
  - 全量 pytest 从 ~6 分钟降到 ~1 分 20 秒（675 passed / 2 skipped）。

## 中期（v0.6.0）

- [x] **正式移除 `simulate(..., phases=...)` 参数**：
  - 已在 v0.6.0 移除 `phases` 参数；`simulate(cfg)` 内部始终调用 `build_phases(cfg)`。
- [x] **terrain / wind 配置化**：
  - `EnvironmentConfig` 已支持 `terrain_model="flat"` 与 `terrain_flat_alt_m`，由 `_resolve_terrain` / `make_terrain` 自动构造 `FlatTerrainModel`。
  - `builder.py` 为 `rocket` / `suborbital` 等任务的 coast 段自动注入 `落地` 事件，无需再手工构造 phase。
- [x] **6-DOF 再入 builder 集成**：
  - `options.sixdof_reentry=True` 可生成 `SixDOFDynamics` 再入段，并通过 `SRBM_600` 全链路仿真测试。
  - `simulate()` 已支持不同阶段状态维度不一致的轨迹拼接。
  - 配平攻角、热流计算等高级再入特性仍待后续建模，可单独立项。
- [x] **补充覆盖率缺口**：
  - 已补充 `presets/loader.py`、`context.py`、`phases/events.py`、`simulator.py` 等异常/失败分支测试。
  - 全量覆盖率从 87.47% 提升到 87.99%。
  - `viz/*` 可选依赖分支与部分 `guidance/*` 分支留待后续按需补强。

## 长期 / 探索性

- [ ] **异步 / 流式仿真 API**：
  - 为 GUI 与 Web API 提供 `simulate_async(cfg)` 或生成器，支持进度回调与中途取消。
- [ ] **GPU batch 后端完善**：
  - `BatchMPMModel` 已有 GPU 变体，但测试覆盖率与跨平台兼容性仍需加强。
- [x] **文档与用户指南**：
  - 已新增 `docs/user_guide.md`，包含任务配置、预设、CLI/API、环境自定义、Monte Carlo 等示例。
- [ ] **持续性能基准**：
  - 把 `scripts/benchmark.py` 接入 CI，监控单发 M107、100 样本 MC、CZ-2F 端到端的耗时回归。
