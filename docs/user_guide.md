# 用户指南

本指南介绍如何使用 `ballistic-sim` 运行常见弹道/轨道任务，并说明核心配置字段。

---

## 快速开始

### 安装

```bash
# 基础依赖（NumPy/SciPy/matplotlib/PyYAML/pydantic）
pip install -e .

# 完整 extras（Web API、3D 可视化、天气/地形数据源）
pip install -e .[all]
```

### 运行一次仿真

```bash
# 使用 CLI 与内置预设
python -m ballistic_sim.cli --mission projectile --preset M107

# 或使用安装后的入口脚本
ballistic-sim --mission rocket --preset CZ2F
```

### 使用 Web API

```bash
ballistic-sim --serve --host 127.0.0.1 --port 8000
```

浏览器访问 `http://127.0.0.1:8000/` 可查看最小前端；API 端点包括：

- `GET /health` — 健康检查与版本号
- `POST /simulate/{mission}` — 单发仿真
- `POST /simulate/monte-carlo` — Monte Carlo 散布
- `POST /firecontrol/solve` — 火控诸元反解
- `POST /config/validate` — 业务校验

---

## 配置文件说明

`ballistic-sim` 使用 `SimConfig`（pydantic 模型）作为单一事实源，支持 YAML/JSON 加载与点路径覆盖。

### 顶层字段

| 字段 | 说明 |
|------|------|
| `mission` | 任务类型：`projectile` / `missile` / `rocket` / `icbm` / `suborbital` |
| `vehicle` | 飞行器参数：质量、推力、燃时、气动参考面积、惯量等 |
| `launch` | 发射点：经纬高、射向、仰角 |
| `environment` | 环境模型：大气、引力、风、地形 |
| `guidance` | 制导律参数：程序转弯角、导航常数、导引模式等 |
| `options` | 求解器选项：积分容差、 fidelity、6-DOF 开关、Monte Carlo 开关 |

### 加载与覆盖

```python
from ballistic_sim.config import load_config, SimConfig

cfg = load_config("configs/sample_icbm.yaml")
# 点路径覆盖优先级高于文件内容
cfg = cfg.apply_overrides({"launch.elevation_deg": 55.0})
```

---

## 任务示例

### 1. 炮弹 projectile（M107）

```bash
ballistic-sim --mission projectile --preset M107 \
  --elevation 45 --azimuth 90 \
  --output out/m107_result.json
```

### 2. 导弹 missile

```bash
ballistic-sim --mission missile --preset scud-b \
  --target-lat 30.0 --target-lon 120.0 \
  --output out/missile_result.json
```

### 3. 火箭 rocket（CZ-2F）

```bash
ballistic-sim --mission rocket --preset CZ2F \
  --output out/rocket_result.json
```

### 4. 洲际弹道导弹 ICBM

```bash
ballistic-sim --mission icbm \
  --target-lat 40.0 --target-lon -74.0 \
  --output out/icbm_result.json
```

### 5. 亚轨道 suborbital

```bash
ballistic-sim --mission suborbital \
  --apogee-km 120 \
  --output out/suborbital_result.json
```

---

## 输出与可视化

仿真结束后，CLI 会输出关键指标（射程、落点、飞行时间）。使用 Python API 可进一步绘图：

```python
from ballistic_sim.phases.builder import build_phases
from ballistic_sim.presets import m107_config
from ballistic_sim.simulator import simulate
from ballistic_sim.viz.profile import plot_altitude_profile

cfg = m107_config()
phases = build_phases(cfg)
result = simulate(cfg, phases=phases)
plot_altitude_profile(result, save_path="out/altitude.png")
```

---

## 常见问题

- **GUI 无法启动**：Windows 下无显示服务时自动 skip；CI 中请设置 `MPLBACKEND=Agg`。
- **GPU 后端报错**：`cupy` 未安装或 CUDA 版本不匹配，自动回退到 CPU。
- **风/地形文件缺失**：使用 `weather` / `terrain` extras 或关闭对应功能。
