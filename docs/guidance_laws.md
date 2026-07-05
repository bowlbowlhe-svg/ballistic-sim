# 制导律说明

`ballistic-sim` v0.3.0 集成了多种制导律，分别适用于不同飞行阶段与任务类型。

---

## 1. 比例导引（PN）

比例导引（Proportional Navigation, PN）是导弹末端拦截最常用的闭环制导律。

### 原理

加速度指令正比于视线角速度：

```
a_cmd = N' * v_c * dλ/dt
```

其中：

- `N'`：有效导航比，通常取 3~5。
- `v_c`：弹目接近速度。
- `dλ/dt`：视线角速度矢量。

### 模式

- **真比例导引（TPN）**：加速度垂直于视线。
- **广义比例导引（GPN）**：加速度投影到垂直于导弹速度平面，更适合高机动目标。

### 参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `guidance.guidance_law` | `"proportional"` | — |
| `guidance.nav_constant` | 导航比 N' | 3.0 ~ 5.0 |
| `guidance.pro_nav_mode` | `"true"` / `"generalized"` | `"true"` |
| `vehicle.max_lateral_accel_m_s2` | 最大可用横向加速度 | 50 ~ 300 m/s² |

---

## 2. 增广显式制导（AAG）

增广显式制导（Augmented Explicit Guidance, AAG）用于火箭上升段，将飞行器导引到目标终端状态 `(r_T, v_T, γ_T)`。

### 原理

在飞行时间剩余 `t_go` 内，通过解析预测终端状态误差，构造推力方向指令，使终端位置、速度、航迹角同时收敛。

### 参数

| 参数 | 说明 |
|------|------|
| `guidance.guidance_law` | `"aag"` |
| `guidance.target_apogee_m` | 目标远地点半径/高度 |
| `guidance.target_inclination_deg` | 目标轨道倾角 |
| `guidance.aag_iterations` | 迭代显式预测次数 |

---

## 3. 再入制导

再入制导基于阻力加速度剖面或弹道系数，控制倾侧角/攻角，使再入飞行器满足热流、动压、落点约束。

### 原理

- 通过调节倾侧角控制纵向航程。
- 通过反向倾侧控制横程误差。
- 能量管理：在较高高度耗散多余能量，避免过低高度速度过大。

### 参数

| 参数 | 说明 |
|------|------|
| `guidance.guidance_law` | `"reentry"` |
| `guidance.reentry_max_g` | 最大过载约束 |
| `guidance.reentry_max_q` | 最大动压约束 |
| `guidance.reentry_lifting` | 是否启用升力控制 |

---

## 4. 能量管理制导

能量管理制导（Energy Management）适用于亚轨道/返回式飞行器，使飞行器按目标能量剖面到达指定高度/速度。

### 原理

定义总机械能 `E = v²/2 + g*h`，通过俯仰角/推力调节耗散率，使实际能量剖面跟踪目标剖面：

```
ΔE = E_target(h) - E_actual
a_cmd = k_p * ΔE + k_d * d(ΔE)/dt
```

### 参数

| 参数 | 说明 |
|------|------|
| `guidance.guidance_law` | `"energy"` |
| `guidance.energy_target_profile` | 目标高度-能量剖面 |
| `guidance.energy_kp` / `guidance.energy_kd` | PD 控制增益 |

---

## 制导律选择建议

| 任务阶段 | 推荐制导律 |
|----------|-----------|
| 火箭上升段 | AAG / PEG |
| 导弹中段/末段 | PN / GPN |
| 再入段 | 再入制导 |
| 亚轨道返回 | 能量管理制导 |
| 无控弹道 | `none` / `open_loop` |
