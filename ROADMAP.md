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

- [ ] **正式移除 `simulate(..., phases=...)` 参数**：
  - 在 v0.5.0 已标记 `DeprecationWarning`，建议至少保留一个次要版本后于 v0.6.0 移除。
- [ ] ** terrain / wind 配置化**：
  - 目前自定义地形/风场需手工注入 phase，考虑在 `EnvironmentConfig` 中支持 `terrain_file` / `wind_profile` 等字段，由 builder 自动构造对应模型。
- [ ] **6-DOF 再入 builder 完整集成**：
  - `options.sixdof_reentry=True` 已可生成 `SixDOFDynamics`，但制导律、配平攻角、热流计算仍需补充测试与文档。
- [ ] **补充覆盖率缺口**：
  - `ballistic_sim/presets/loader.py` 异常分支。
  - `ballistic_sim/viz/*` 中依赖可选依赖（plotly、pyvista）的分支。
  - `ballistic_sim/guidance/*` 中部分失败回退路径。

## 长期 / 探索性

- [ ] **异步 / 流式仿真 API**：
  - 为 GUI 与 Web API 提供 `simulate_async(cfg)` 或生成器，支持进度回调与中途取消。
- [ ] **GPU batch 后端完善**：
  - `BatchMPMModel` 已有 GPU 变体，但测试覆盖率与跨平台兼容性仍需加强。
- [ ] **文档与用户指南**：
  - 基于当前 `README.md` / `CHANGELOG.md` 补充任务配置教程、YAML 预设说明、API 示例。
- [ ] **持续性能基准**：
  - 把 `scripts/benchmark.py` 接入 CI，监控单发 M107、100 样本 MC、CZ-2F 端到端的耗时回归。
