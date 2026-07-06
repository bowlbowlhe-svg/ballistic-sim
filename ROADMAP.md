# 后续计划（Next Steps）

> 本文件记录 v0.5.0 之后的潜在改进方向，按需继续。

## 短期（v0.5.x）

- [ ] **清理 phase 显式传参的遗留例外**：
  - `ballistic_sim/guidance/targeting.py` 因需要对 phase 做 in-place 制导/阻力调整而保留 `simulate(cfg, phases=...)`。
  - 后续可通过扩展 `GuidanceConfig` 或新增 `post_build_phases_hook` 机制，把调整逻辑纳入 `SimConfig`，从而彻底移除显式 `phases` 参数。
- [ ] **把占位测试迁移到 builder**：
  - `tests/test_icbm.py::_placeholder_icbm_phases()` 仍手工构造 phase 链。
  - 等 ICBM 预设与 builder 成熟后，改为 `rocket_full_config` / `missile_full_config` + `build_phases(cfg)`。
- [ ] **提升慢测试运行效率**：
  - 当前全量 pytest 在 Windows 本地约 5 分钟。
  - 可考虑安装 `pytest-xdist` 并启用 `-n auto` 拆分慢测试。

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
