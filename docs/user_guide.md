# 用户指南

本指南面向使用 `ballistic-sim` 进行弹道/轨道仿真的开发者与分析师，介绍最常见任务的 Python API、CLI 与 Web API 用法。

---

## 简介

`ballistic-sim` 是一款模块化六自由度（6-DOF）弹道飞行仿真程序，核心能力包括：

- 统一多阶段仿真器：`ballistic_sim.simulator.simulate`
- 可配置的大气、风、引力、地形环境模型
- 导弹/火箭/炮弹/洲际导弹/亚轨道等多任务类型
- 制导律库（程序转弯、比例导引、PEG、再入制导等）
- Monte Carlo 散布分析
- Web API（FastAPI）与命令行入口

主要模块：

- `ballistic_sim/models/` —— 大气、引力、风、地形、气动模型
- `ballistic_sim/dynamics/` —— 运动学与动力学方程（点质量、MPM、6-DOF）
- `ballistic_sim/phases/` —— 起飞、助推、滑行、再入、终端等阶段
- `ballistic_sim/guidance/` —— 制导律与导航算法
- `ballistic_sim/viz/` —— 可视化与动画生成
- `ballistic_sim/api/` —— Web 服务与对外接口
- `ballistic_sim/presets/` —— 弹丸、导弹、火箭预设

> 说明：`presets/` 中的参数为教学、本地调试和回归测试使用的近似/占位数据，不代表任何真实飞行器性能。真实任务请替换为实际数据。

---

## 快速开始

安装后，用 `m107_config()` 获取 M107 155 mm 榴弹预设，调用 `simulate()` 即可得到落点与飞行时间：

```python
from ballistic_sim.presets import m107_config
from ballistic_sim.simulator import simulate

cfg = m107_config()
result = simulate(cfg)

print("飞行时间:", result.post["t_end_s"], "s")
print("落点 ENU:", result.post["r_end_m"])
```

典型输出：

```text
飞行时间: 69.08082374213444 s
落点 ENU: [17997.27, -525.66, 0.0]
```

`result.post` 包含后处理结果；`result.t` 与 `result.y` 分别为时间序列与状态矩阵。

---

## 配置体系

`SimConfig` 是仿真的单一事实源，包含五大子配置：

| 子配置 | 说明 | 常用字段 |
|--------|------|----------|
| `vehicle` | 飞行器参数 | `mass_kg`、`diameter_m`、`cd`、`area_ref_m2`、`stages`、`drag_law`、`Ix`、`It`、`x_cp_cg`、`twist_cal` |
| `launch` | 发射/初始条件 | `lat_deg`、`lon_deg`、`alt_m`、`azimuth_deg`、`elevation_deg`、`v0_m_s` |
| `environment` | 环境模型 | `atmosphere`、`wind_model`、`wind_m_s`、`gravity_model`、`terrain_model`、`delta_t`、`density_factor` |
| `guidance` | 制导律参数 | `guidance_law`、`kick_deg`、`target_lat_deg`、`target_lon_deg`、`nav_constant` |
| `options` | 积分器与输出选项 | `integrator`、`rtol`、`atol`、`max_step`、`sixdof_reentry`、`monte_carlo` |

创建配置：

```python
from ballistic_sim.config import SimConfig, VehicleConfig, LaunchConfig, EnvironmentConfig, GuidanceConfig, OptionsConfig

cfg = SimConfig(
    mission="projectile",
    vehicle=VehicleConfig(mass_kg=45.0, diameter_m=0.155, cd=0.295),
    launch=LaunchConfig(elevation_deg=45.0, azimuth_deg=90.0, v0_m_s=684.0),
    environment=EnvironmentConfig(atmosphere="isa"),
    guidance=GuidanceConfig(),
    options=OptionsConfig(integrator="DOP853", terminate_impact=True),
)
```

---

## 预设使用

`ballistic_sim.presets` 提供常见弹丸、导弹与火箭的完整配置函数。

### 弹道导弹：SRBM_600

```python
from ballistic_sim.presets import missile_full_config
from ballistic_sim.simulator import simulate

cfg = missile_full_config("SRBM_600")
result = simulate(cfg)
print("飞行时间:", result.post["t_end_s"], "s")
print("落点 ENU:", result.post["r_end_m"])
```

### 运载火箭：CZ-2F

```python
from ballistic_sim.presets import rocket_full_config
from ballistic_sim.simulator import simulate

cfg = rocket_full_config("CZ2F")
result = simulate(cfg)
print("飞行时间:", result.post["t_end_s"], "s")
print("落点 ENU:", result.post["r_end_m"])
```

> `missile_full_config` 与 `rocket_full_config` 会构造带多级 `StageConfig` 的完整任务链，可直接交给 `simulate` 运行。

---

## CLI 使用

安装包后，命令行入口为 `ballistic-sim`：

```bash
ballistic-sim --help
```

常用命令示例（运行一发 M107，仰角 45°、射向 90°）：

```bash
ballistic-sim --mission projectile --preset M107 --qe 45 --az 90 --no-viz --out-dir out/m107_cli
```

输出示例：

```text
Mission : projectile
Stop    : completed
TOF     : 69.08082374213444 s
Range   : 18.00 km
Output  : out\m107_cli
```

---

## Web API 使用

使用 `uvicorn` 启动 Web 服务：

```bash
uvicorn ballistic_sim.api.main:app --host 127.0.0.1 --port 8000
```

启动后浏览器访问：

- `http://127.0.0.1:8000/docs` —— 自动生成的 Swagger/OpenAPI 文档
- `http://127.0.0.1:8000/redoc` —— ReDoc 风格文档
- `GET /health` —— 健康检查
- `POST /simulate/{mission}` —— 单发仿真
- `POST /simulate/monte-carlo` —— Monte Carlo 散布
- `POST /firecontrol/solve` —— 火控诸元反解

> 若从项目根目录启动，确保已安装 extras（`pip install -e ".[api]"` 或 `"[all]"`）。

---

## 异步 / 流式仿真 API

对于 GUI、Web 后端或需要并发调度多个仿真的场景，可使用异步包装：

```python
import asyncio
from ballistic_sim.presets import m107_config
from ballistic_sim import simulate_async, simulate_streaming

async def run():
    cfg = m107_config()

    # 方式一：等待仿真完成，不阻塞事件循环
    result = await simulate_async(cfg)
    print("飞行时间:", result.post["t_end_s"], "s")

    # 方式二：流式接收进度事件（最终事件包含结果）
    async for event in simulate_streaming(cfg, progress_interval_s=0.5):
        if event["type"] == "progress":
            print(f"已运行 {event['elapsed_s']:.2f} s")
        else:
            print("完成，结果:", event["result"].post)

asyncio.run(run())
```

`simulate_async` 把 CPU 密集的 `simulate` 放到执行器中运行；`simulate_streaming` 在此基础上定期产出 `progress` 字典，方便前端展示进度条。

---

## 自定义环境

通过 `EnvironmentConfig` 可切换风场与地形模型。下面使用对数风廓线与平坦地形：

```python
from ballistic_sim.presets import m107_config
from ballistic_sim.config import EnvironmentConfig
from ballistic_sim.simulator import simulate

cfg = m107_config()
cfg = cfg.model_copy(update={
    "environment": EnvironmentConfig(
        atmosphere="isa",
        wind_model="log",
        wind_u_ref=10.0,      # 参考高度处风速 (m/s)
        wind_z_ref=10.0,      # 参考高度 (m)
        wind_z0=0.03,         # 地面粗糙长度 (m)
        wind_direction_deg=270.0,  # 西风，自北顺时针
        terrain_model="flat",
        terrain_flat_alt_m=0.0,
    )
})

result = simulate(cfg)
print("飞行时间:", result.post["t_end_s"], "s")
print("落点 ENU:", result.post["r_end_m"])
```

支持的风场模型：`uniform`、`log`、`power`、`profile`、`composite`、`dryden`、`none`。
支持的地形模型：`null`、`flat`、`hilly`、`numpy`、`image`、`geotiff`、`srtm_dir`、`srtm_files`。

---

## 6-DOF 再入

对于导弹/火箭等再入任务，设置 `OptionsConfig(sixdof_reentry=True)` 可在再入段启用高保真 6-DOF 动力学（状态维度由 7 升至 13）：

```python
from ballistic_sim.presets import missile_full_config
from ballistic_sim.simulator import simulate

cfg = missile_full_config("SRBM_600")
cfg = cfg.model_copy(update={
    "options": cfg.options.model_copy(update={"sixdof_reentry": True})
})

result = simulate(cfg)
print("状态维度:", result.y.shape[1])
print("飞行时间:", result.post["t_end_s"], "s")
print("落点 ENU:", result.post["r_end_m"])
```

---

## Monte Carlo

使用 `monte_carlo_simulation()` 进行散布分析。`process` 后端使用与单发仿真完全一致的物理模型；`batch` 后端为向量化实现，仅适用于 `projectile` 任务且关闭 MPM 自转/动态攻角时。

```python
from ballistic_sim.presets import m107_config
from ballistic_sim.monte_carlo import monte_carlo_simulation, PerturbationConfig, DistributionConfig
from ballistic_sim.config import OptionsConfig

cfg = m107_config()
cfg = cfg.model_copy(update={
    "options": cfg.options.model_copy(update={
        "mpm_use_spin": False,
        "mpm_use_dynamic_alpha": False,
    })
})

perturb = PerturbationConfig(
    v0_m_s=DistributionConfig(std=5.0),
    elevation_deg=DistributionConfig(std=0.1),
    azimuth_deg=DistributionConfig(std=0.1),
)

result = monte_carlo_simulation(
    cfg,
    perturb=perturb,
    n_samples=100,
    backend="batch",
    seed=42,
)
print("平均射程:", result.range_mean)
print("射程标准差:", result.range_std)
print("CEP50:", result.cep50)
print("有效样本:", result.samples)
```

常用后端：

- `auto` —— 自动选择（projectile + UniformWind 优先 batch/gpu，否则 process）
- `process` —— 与单发仿真一致物理，支持 n_jobs 并行
- `batch` —— 向量化 CPU，仅限 projectile + UniformWind
- `gpu` —— 需要 CuPy

---

## 更多参考

- [架构说明](architecture.md)
- [制导律说明](guidance_laws.md)
- [性能调优](performance.md)
