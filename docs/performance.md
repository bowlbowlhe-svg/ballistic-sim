# 性能报告与调优建议

---

## 基准结果

`scripts/benchmark.py` 运行三组典型负载并输出 `benchmark_results.json`：

```json
{
  "version": "0.3.0",
  "benchmarks": [
    {
      "name": "single_projectile",
      "n_samples": 1,
      "elapsed_s": 0.12,
      "stop_reason": "completed",
      "n_points": 152
    },
    {
      "name": "monte_carlo_process_100",
      "n_samples": 100,
      "elapsed_s": 18.5,
      "valid_samples": 100,
      "cep50_m": 12.3,
      "cep90_m": 28.7
    },
    {
      "name": "sixdof_reentry_10",
      "n_samples": 10,
      "elapsed_s": 45.2,
      "completed": 10,
      "avg_points": 320.5
    }
  ],
  "total_elapsed_s": 63.82
}
```

> 注：上表为示例数据，实际数值请以 `benchmark_results.json` 为准。

---

## 调优建议

### 积分器容差

- `options.rtol` / `options.atol` 默认 `1e-6` / `1e-9`。
- 快速扫点可放宽到 `1e-4` / `1e-6`，可节省 30% 以上时间。
- 高精度再入/6-DOF 建议保持默认或更严。

### 模型缓存

阶段 1 引入的 `ModelCache` 默认开启，缓存大气、风、气动系数插值结果。对重复查询场景（如 Monte Carlo）可显著降低耗时。

### Monte Carlo 后端选择

| 后端 | 适用场景 | 精度 |
|------|----------|------|
| `batch` | projectile 任务、大量样本 | 统计模型，最快 |
| `process` | 通用任务、CPU 多核 | 全精度 |
| `gpu` | 配备 CUDA 的大规模样本 | 统计模型 |
| `auto` | 由配置自动选择 | 按任务而定 |

### 6-DOF 再入

- 关闭 `options.sixdof_reentry` 时，再入段使用 3-DOF 质点模型，速度提升 5~10 倍。
- 仅在需要姿态/热流细节时启用 6-DOF。

---

## CI 性能回归

`.github/workflows/benchmark.yml` 在每次 PR/push 到 `main` 时运行 `scripts/benchmark.py`，并将结果作为 artifact 上传。未来版本将自动与上一 tag 的 `benchmark_results.json` 对比，下降超过 10% 时标红。
