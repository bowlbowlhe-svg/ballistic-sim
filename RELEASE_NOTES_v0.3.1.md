# Release Notes v0.3.1

**发布日期：** 2026-07-05

## 修复摘要

1. **Monte Carlo `process` 后端稳定性**
   - 在 `ballistic_sim/monte_carlo.py` 的 `_monte_carlo_process` 中增加 pytest 检测：当 `"pytest" in sys.modules` 且 `PYTEST_CURRENT_TEST` 环境变量存在时，打印 warning 并自动回退到 `batch` 后端（若任务配置支持）；否则强制串行 `process`（`n_jobs=1`），避免 Windows spawn 导致子进程递归挂起。
   - CLI/API 显式指定 `--mc-backend process` 在非 pytest 环境下仍正常尝试 process 后端。

2. **制导律测试覆盖提升**
   - 补充 `tests/guidance/test_proportional_navigation.py`：覆盖 `make_static_target_provider`、GPN 零速度分支、零视线速率、异常输入、目标提供者调用、direction 零加速度失败路径，使 `ballistic_sim/guidance/proportional_navigation.py` 覆盖率提升至 ≥85%。

3. **matplotlib 中文字体警告优化**
   - 在 `ballistic_sim/viz/__init__.py` 设置 `matplotlib.rcParams['font.sans-serif']` 为 `["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]`，并设置 `axes.unicode_minus = False`。
   - 有中文字体时直接显示中文；无字体时 matplotlib 仅回退并产生一次警告，不影响功能。

4. **3D 可视化测试补强**
   - 在 `tests/viz/test_interactive3d.py` 中新增 plotly 缺失时 `plot_trajectory_3d`/`demo` 的 `ImportError` 路径测试，以及 HTML 写出失败异常传播测试，使 `ballistic_sim/viz/interactive3d.py` 覆盖率尽量 ≥85%。

## 已知仍存在的限制

- `ballistic_sim/viz/interactive3d.py` 生成的 HTML 依赖 Plotly.js CDN；离线环境首次加载可能失败或需要预下载 JS bundle。
- Monte Carlo `process` 后端在 pytest 收集器内部会自动降级；若任务不支持 batch 后端则退化为串行执行，性能较并行 process 有所下降。
- 中文字体回退机制依赖系统已安装字体；若全部字体缺失，matplotlib 仍会发出一次字体查找警告。
- GPU 后端 (`--mc-backend gpu`) 依赖 CuPy/CUDA，未安装时不可用；CI 与常规测试环境已跳过 GPU 相关断言。

## 安装 / 升级

```bash
# 从 PyPI 安装/升级
pip install -U ballistic-sim

# 或从源码安装
pip install -e .

# 需要 3D 可视化时安装 viz3d extra
pip install -U ballistic-sim[viz3d]
```

## 质量门禁

- `MPLBACKEND=Agg python -m pytest tests -q --cov=ballistic_sim --cov-report=term --cov-fail-under=80`：测试数 ≥603，整体覆盖率 ≥85%。
- `flake8 ballistic_sim tests`：通过。
- `mypy ballistic_sim`：通过。
- `python scripts/release_check.py`：通过（需外层 Coordinator 已创建 `v0.3.1` tag）。
