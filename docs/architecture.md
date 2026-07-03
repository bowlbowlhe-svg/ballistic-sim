# Ballistic Sim 架构设计文档

> 版本：v0.1.0  
> 对应阶段：阶段 5（验证、对拍与收尾）  
> 适用范围：`ballistic_sim` Python 包及其 CLI/Web 扩展。

---

## 1. 项目定位与两大仿真域划分

`ballistic-sim` 是一个模块化六自由度（6-DOF）弹道飞行仿真程序，目标覆盖从大气层内弹丸到跨大气层火箭/导弹的连续飞行过程。为了兼容不同任务对计算精度、状态维度与坐标系的不同需求，项目按**状态维度**与**坐标系**划分为两大仿真域：

| 仿真域 | 典型任务 | 状态向量 | 原生坐标系 | 核心动力学 | 适用阶段 |
|---|---|---|---|---|---|
| **近程大气域（MPM/3-DOF ENU）** | 炮弹、榴弹、近程导弹 | 8 维：`[E,N,U,v_E,v_N,v_U,p,q]` 或 3-DOF 变体 | ENU（东北天，发射点局部） | `MPMDynamics`、`PointMassDynamics` | 起飞/无动力弹道/再入/终点 |
| **跨大气域（3-DOF ECI）** | 运载火箭、ICBM、亚轨道飞行器 | 7 维：`[r_eci, v_eci, m]` | ECI（地心惯性） | `PoweredECIDynamics` | 主动段/滑行/轨道插入 |

- **MPM（Modified Point Mass）**：在 3-DOF 质点基础上加入自旋、章动与攻角近似，适用于大气层内弹道，可复现 Magnus 效应、陀螺稳定等弹道现象。
- **3-DOF ECI**：采用 J2 引力与大气阻力，支持多级动力上升，用于入轨、再入与轨道力学分析。
- **6-DOF 骨架**：阶段 3 已完成控制律接口（`ballistic_sim.guidance.control`），但 6-DOF 闭环动力学仅在稳定工况下启用；默认任务仍使用 MPM 或 3-DOF ECI。

---

## 2. 整体架构图

### 2.1 目录结构

```
D:\程序\弹道模拟程序
├── ballistic_sim/           # 核心仿真包
│   ├── __init__.py
│   ├── cli.py               # 统一命令行入口
│   ├── config.py            # pydantic SimConfig 与校验
│   ├── constants.py         # 统一物理常数（WGS84、ISA、地球自转等）
│   ├── frames.py            # ENU/ECEF/ECI 坐标系转换
│   ├── simulator.py         # 统一多阶段积分主循环
│   ├── state_switch.py      # 阶段间状态投影（升维/降维/坐标变换）
│   ├── models/              # 环境/物理模型
│   │   ├── aerodynamics.py  # 阻力/升力模型（G1/G7/火箭近似）
│   │   ├── atmosphere.py    # ISA/US76 标准大气
│   │   ├── gravity.py       # 点质量/J2 引力
│   │   ├── propulsion.py    # 推进剂流量与推力高度修正
│   │   ├── terrain.py       # 地形接口（含 NullTerrain）
│   │   └── wind.py          # 均匀风/对数风/Dryden 阵风
│   ├── dynamics/            # 运动方程 RHS
│   │   ├── common.py        # 共享上下文、动压、Mach 数、轨道根数
│   │   ├── mpm.py           # MPM 8 维运动方程
│   │   ├── point_mass.py    # 3-DOF ECI 质点
│   │   └── powered_eci.py   # 动力上升 3-DOF ECI
│   ├── phases/              # 阶段抽象与实现
│   │   ├── base.py          # Phase / PhaseContext 抽象
│   │   ├── builder.py       # 由 SimConfig 自动生成 Phase 列表
│   │   ├── powered.py       # 动力/推进阶段
│   │   ├── coasting.py      # 无动力滑行阶段
│   │   ├── reentry.py       # 再入阶段
│   │   ├── terminal.py      # 终点/轨道插入阶段
│   │   └── events.py        # 事件工厂（落地、燃尽、远地点、靶距）
│   ├── guidance/            # 制导律与火控
│   │   ├── open_loop.py     # 开环程序转弯/重力转弯
│   │   ├── peg.py           # 动力显式制导（Powered Explicit Guidance）
│   │   ├── circularization.py # 圆化脉冲
│   │   ├── targeting.py     # 射表/落点反算
│   │   ├── firecontrol.py   # 火炮诸元解算
│   │   └── control.py       # 6-DOF 控制律接口（骨架）
│   ├── viz/                 # 可视化（纯函数）
│   │   ├── __init__.py      # 坐标/帧检测工具
│   │   ├── trajectory3d.py  # 3D 轨迹
│   │   ├── groundtrack.py   # 地面航迹与落点
│   │   ├── profile.py       # 高度/速度剖面
│   │   ├── wind_plot.py     # 风廓线与风偏贡献
│   │   ├── globe3d.py       # 地球 3D
│   │   ├── orbit_insertion.py # 入轨参数
│   │   ├── dvbudget.py      # 速度增量预算
│   │   ├── reentry.py       # 再入热流
│   │   └── animation.py     # GIF 动画
│   ├── presets/             # YAML 预设兼容层
│   │   ├── loader.py        # YAML 加载器
│   │   ├── projectiles.py   # M107 等弹丸预设
│   │   ├── missiles.py      # 导弹预设
│   │   └── rockets.py       # CZ-2F 等火箭预设
│   └── utils/
│       └── optional_imports.py  # 可选依赖导入守卫
├── presets/                 # YAML 数据文件
│   ├── projectiles.yaml
│   ├── missiles.yaml
│   └── rockets.yaml
├── tests/                   # pytest 测试套件
├── scripts/                 # MVP 演示、回归桥、golden 生成
├── docs/                    # 文档
│   ├── architecture.md      # 本文档
│   ├── known_failures.md    # 已知失效工况
│   └── migration_scope.md   # 迁移范围
├── web/                     # Web API 占位（v1.0 未启用）
├── pyproject.toml
└── CHANGELOG.md
```

### 2.2 模块关系图

```
┌─────────────────────────────────────────────────────────────┐
│                        用户入口层                            │
│  CLI (ballistic_sim.cli)  │  Web API (web/, v1.0 占位)       │
└────────────────────┬────────────────────────────────────────┘
                     │ SimConfig / Phase list
┌────────────────────▼────────────────────────────────────────┐
│                      配置与驱动层                            │
│  SimConfig (pydantic) ── apply_overrides / validate_config  │
│  simulator.simulate() ── 多阶段积分主循环                    │
└────────────────────┬────────────────────────────────────────┘
                     │ Phase 序列
┌────────────────────▼────────────────────────────────────────┐
│                       Phase 层                               │
│  Phase (base) → PoweredPhase / CoastingPhase / ReentryPhase │
│  events.py 提供统一事件工厂                                  │
└────────────────────┬────────────────────────────────────────┘
                     │ rhs(t, y, ctx)
┌────────────────────▼────────────────────────────────────────┐
│                      动力学层                                │
│  MPMDynamics (8-DOF ENU)                                    │
│  PointMassDynamics (3-DOF ECI)                              │
│  PoweredECIDynamics (3-DOF ECI + 推力 + J2)                 │
└────────────────────┬────────────────────────────────────────┘
                     │ 调用
┌────────────────────▼────────────────────────────────────────┐
│                      物理模型层                              │
│  atmosphere  │  gravity  │  aerodynamics  │  wind  │  propulsion  │  terrain
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 坐标系约定

### 3.1 ENU（东北天）

- **原点**：发射点局部，通常取 `geodetic_to_ecef(lat0, lon0, alt0)`。
- **基矢量**：
  - `E`：东向，沿纬线切线。
  - `N`：北向，沿经线切线指向真北。
  - `U`：天向，沿椭球法向向外。
- **用途**：大气层内弹道、落点/射表计算、风场输入默认使用 ENU。
- **限制**：ENU 为局部坐标系，远距离飞行需切换至 ECI/ECEF，否则平面近似会引入误差。

### 3.2 ECI（地心惯性）

- **原点**：地心。
- **Z 轴**：地球自转轴，指向北极。
- **X 轴**：指向春分点方向（简化为 `t=0` 时与 ECEF X 轴重合，仅通过 `OMEGA_EARTH * t` 旋转）。
- **用途**：火箭/ICBM 动力上升、轨道力学、J2 摄动、多级分离。
- **转换链**：`ENU ↔ ECEF ↔ ECI`，详见 `ballistic_sim.frames`。

### 3.3 阶段间投影

`ballistic_sim.state_switch.project_state` 负责阶段间状态映射：

- 同维度同坐标系：直接拷贝。
- 3-DOF ECI ↔ 3-DOF ENU：通过 `frames` 进行位置/速度旋转。
- 6-DOF ↔ MPM：降维时保留 `r,v,m` 并丢弃姿态/角速度；升维**禁止自动进行**（抛 `NoAutoDimensionUpgrade`），必须显式给出 13 维状态。

---

## 4. Phase 驱动流程

`simulator.simulate(cfg, phases)` 的执行流程：

1. **解析动力学上下文**：由 `SimConfig` 生成 `DynamicContext`（大气、风、气动力、引力模型）。
2. **构建 Phase 序列**：若未传入 `phases`，调用 `phases.builder.build_phases(cfg)`。
3. **初始状态**：由第一个 Phase 的 `dynamics.initial_state(...)` 生成。
4. **逐段积分**：对每一个非终端 Phase：
   - 计算本段绝对时间区间 `(t_start, t_end_max)`。
   - 装订制导参数（如 `t_us_start`、`gamma0_deg`）。
   - 调用 `scipy.integrate.solve_ivp` 积分。
   - 处理事件：`phase.process_events(sol)`。
   - 记录轨迹并拼接。
   - 阶段间调用 `project_state` 映射到下一 Phase 的状态空间。
5. **后处理**：计算终点位置、速度、轨道根数等，写入 `SimResult.post`。

### 4.1 Phase 生命周期

```python
# 1. 构造
ph = PoweredPhase(name="动力上升", t_span=(0, t_burn), dynamics=dyn, guidance=guid, m_dry=m_dry)

# 2. 事件注册（在 __post_init__ 中自动完成）
ph.events = [make_burnout_event(m_dry), make_ground_event(frame="ECI")]

# 3. 积分器调用
sol = solve_ivp(ph.rhs, (t_start, t_end_max), y0, events=ph.events, ...)

# 4. 事件解析
records = ph.process_events(sol)

# 5. 末状态提取与映射
y_end = ph.final_state(sol)
y0_next = project_state(y_end, src_dim=..., dst_dim=..., src_frame=..., dst_frame=...)
```

---

## 5. 配置层设计

### 5.1 SimConfig（pydantic）

`ballistic_sim.config.SimConfig` 是统一配置入口，采用 `pydantic.BaseModel` 定义：

```python
class SimConfig(BaseModel):
    mission: str  # projectile | missile | rocket | icbm | suborbital
    vehicle: VehicleConfig      # 质量、直径、Cd、参考面积、推力、燃烧时间
    launch: LaunchConfig        # 经纬高、方位角、仰角、初速
    environment: EnvironmentConfig  # 大气、风、引力、地形
    guidance: GuidanceConfig    # 制导律、程序角、目标点
    options: OptionsConfig      # 积分器、容差、事件开关
```

- 使用 `ConfigDict(extra="forbid")` 防止拼写错误。
- 字段通过 `Field(pattern=...)` 限制合法取值。
- 支持 `.model_dump(mode="json")` 序列化。

### 5.2 Overrides

`apply_overrides(cfg, {"guidance.kick_deg": 15.0})` 按点路径覆盖嵌套字段，实现 CLI 参数到配置的映射。支持任意层级：

```python
cfg = apply_overrides(cfg, {
    "launch.elevation_deg": 45.0,
    "environment.wind_m_s": [5.0, 0.0, 0.0],
})
```

### 5.3 validate_config

`validate_config(cfg)` 执行业务级跨字段校验，返回 `ValidationIssue` 列表。阶段 0/1 已实现简单校验：

- 大气层内任务负仰角警告。
- 设置推力但未设置 `burn_time_s` 警告。

后续可扩展：T/W > 1 检查、推进剂守恒、目标可达性、安全落区等。

---

## 6. 可视化设计

### 6.1 纯函数约定

所有 `viz` 模块的绘图函数均为纯函数：

```python
def plot_<name>(result: SimResult, ...) -> Figure:
    ...
    return fig
```

- 不修改 `SimResult`。
- 不依赖全局 matplotlib 后端；CLI 中显式切换为 `Agg`。
- 调用方负责 `savefig` / `close`。

### 6.2 分域输出

`viz.__init__.detect_frame(result)` 根据状态向量维度自动判断坐标系：

- `ncol == 8` → ENU（MPM）。
- `ncol == 7` → ECI（3-DOF 动力/轨道）。
- 其他 → UNKNOWN。

CLI `_save_visualizations` 按坐标系分域调用不同 viz 函数：

- **ENU 域**：`trajectory3d`、`groundtrack`、`profile`、`wind_plot`、`impact_summary`。
- **ECI 域**：`globe3d`、`groundtrack`、`profile`、`orbit_insertion`、`dv_budget`、`reentry_heat`、`trajectory3d`。

### 6.3 坐标工具

`viz.__init__` 提供统一坐标抽取：

- `enu_coords(result)`：返回 `(E, N, U)`。
- `geodetic_coords(result)`：返回 `(lat, lon, alt)`。
- `downrange(result)`、`altitude(result)`、`speed(result)`、`mass(result)`。
- `attach_launch_lla(result, lat, lon, alt)`：附加发射点信息（用于 ENU↔大地坐标）。

---

## 7. 可扩展点

### 7.1 新增 mission

1. 在 `SimConfig.mission` 的 `pattern` 中增加新类型。
2. 在 `phases.builder.build_phases` 中增加该 mission 的 Phase 构造逻辑。
3. 在 `cli.py` 的 `_build_config_and_phases` 中增加 CLI 参数解析分支。
4. 在 `simulator._resolve_dynamics_context` 中选择合适的气动力模型。
5. 补充测试：`tests/test_<mission>.py`。

### 7.2 新增模型

- **大气模型**：在 `models.atmosphere` 中实现 `AtmosphereModel` 协议，注册到 `make_atmosphere` 工厂。
- **气动力模型**：在 `models.aerodynamics` 中实现 `AeroModel` 接口，注册到 `make_aero` 工厂。
- **风模型**：在 `models.wind` 中实现 `WindModel`，注册到 `make_wind` 工厂。
- **引力模型**：在 `models.gravity` 中实现新函数，并在 `gravity_eci` / `gravity_enu` 中分发。

### 7.3 新增 viz

1. 在 `ballistic_sim/viz/` 下新增模块，实现 `plot_<name>(result, ...) -> Figure`。
2. 在 `viz/__init__.py` 暴露必要的坐标工具（如需）。
3. 在 `cli.py` 的 `_save_visualizations` 中按坐标系分域调用。
4. 在 `tests/viz/test_viz_smoke.py` 中补充冒烟测试。

### 7.4 新增制导律

1. 在 `guidance/` 下实现制导函数或类。
2. 在 `GuidanceConfig.guidance_law` 的 `pattern` 中注册。
3. 在 `PoweredECIDynamics._thrust_dir` 或 `control.py` 中分发。

### 7.5 6-DOF 闭环集成

当前 6-DOF 控制律接口（`guidance.control`）已完成，但 6-DOF 动力学闭环需在以下位置接入：

- 新增 `dynamics/six_dof.py` 实现 13 维 RHS。
- 在 `phases` 中新增 `SixDOFPhase`。
- 在 `state_switch` 中补充 6-DOF ↔ 3-DOF/MPM 的投影规则。
- 默认仍使用 3-DOF/MPM，仅在 `options.use_6dof = True` 且初始姿态显式给出时启用。

---

## 8. 与旧项目的关系

本项目逐步迁移并重构原有两个旧项目中的可用模块：

- `D:\Pycharm\游戏与兴趣\外弹道方程`：提供 MPM 动力学、气动力系数、射表算法。
- `D:\Pycharm\游戏与兴趣\火箭仿真程序`：提供 ECI 动力学、坐标系转换、程序转弯制导。

迁移原则：

1. **不修改旧项目源码**；仅做读取与适配。
2. 旧项目常量合并到 `ballistic_sim.constants`。
3. 旧项目测试优先转为 `tests/` 下的 pytest 用例。
4. 已知发散工况登记在 `docs/known_failures.md`。
