# ballistic_sim v0.5.0 P1 实施计划

**版本**：0.4.0 → 0.5.0  
**基线提交**：`main` 上 v0.4.0 tag/release 已完成  
**质量基线**：pytest 全绿（至少维持 v0.4.0 基线），覆盖率不下降，flake8 / mypy / black 全绿  
**计划目标**：完成 `PLAN_v0.4.0.md` 中 P1 项，全面收敛到 `SimConfig + build_phases(cfg)`。

---

## 1. P1 总体目标（来自 PLAN_v0.4.0 P1）

1. 移除 `m107_phases`、`projectile_phases`、`missile_phases`、`rocket_phases`、`missile_full_chain`、`rocket_full_chain` 等 deprecated 函数。
2. ICBM / suborbital CLI 占位 builder 完全改为 `SimConfig + build_phases(cfg)`。
3. `VehicleConfig` 中补充更多预设所需字段（如 `twist_cal`、`Ix`、`It` 默认值），减少绕过 `build_phases` 的函数。
4. `simulate(cfg, phases)` 的 `phases` 参数改为可选（默认 `None`），内部自动 `build_phases(cfg)`，并标记显式传 phases 为 deprecated。

---

## 2. P1 总体策略

按依赖关系拆为 **4 个顺序阶段**：

| 阶段 | 主题 | 前置阶段 | 关键产出 |
|------|------|---------|---------|
| Stage 1 | 补齐 `VehicleConfig` 默认值与 projectile builder | 无 | `VehicleConfig` 可携带 preset 所需 6-DOF/MPM 参数；`_build_projectile_phases` 读取这些参数 |
| Stage 2 | 迁移内部调用方到 `SimConfig + build_phases` | Stage 1 | CLI/API/GUI/targeting/脚本/测试不再调用 deprecated phase 工厂 |
| Stage 3 | 移除 deprecated phase 工厂与导出 | Stage 2 | `presets/*.py` 删除 6 个函数；`presets/__init__.py` 更新 |
| Stage 4 | `simulate(cfg, phases=None)` + 版本号升级到 0.5.0 | Stage 3 | `phases` 可选并自动 build；所有内部调用改为 `simulate(cfg)`；版本号统一 |

所有阶段共享验收标准：

- `pytest` 全绿（至少维持 v0.4.0 基线通过数）；
- 覆盖率 **不下降**；
- `flake8`、`mypy`、`black` 全绿；
- CLI/API smoke 测试通过：`projectile`、`rocket`、`missile`、`icbm`、`suborbital` 至少各跑一单发；
- 不允许引入循环导入；
- `scripts/benchmark.py`、`scripts/comparison_demo.py`、`scripts/monte_carlo_demo.py`、`scripts/regression_bridge.py` 可正常跑完。

---

## 3. Stage 1：补齐 `VehicleConfig` 与 projectile builder

### 3.1 阶段目标
让 `VehicleConfig` 能完整承载弹丸预设里的 MPM/6-DOF 参数，使 `_build_projectile_phases` 不再使用硬编码默认值，并为后续移除 `_projectile_dynamics_from_preset` 打下基础。

### 3.2 涉及文件
- `ballistic_sim/config.py`
- `ballistic_sim/phases/builder.py`
- `ballistic_sim/presets/projectiles.py`
- `presets/projectiles.yaml`（已含数据，只需读取）
- `tests/phases/test_builder_advanced.py`（验证 builder 行为不变）
- 新增 `tests/test_vehicle_config_defaults.py`（验证新增字段默认值）

### 3.3 开发任务

#### A. `VehicleConfig` 新增/调整字段
在 `ballistic_sim/config.py::VehicleConfig` 中：

```python
twist_cal: Optional[float] = Field(default=20.0, gt=0, description="缠距 (caliber)")
Ix: Optional[float] = Field(default=0.1, gt=0, description="轴向转动惯量 (kg·m²)")
It: Optional[float] = Field(default=1.0, gt=0, description="横向转动惯量 (kg·m²)")
x_cp_cg: Optional[float] = Field(
    default=0.05, description="压心距质心距离 (m)，正表示压心在质心前方"
)
```

说明：
- 当前 `Ix/It/x_cp_cg/twist_cal` 均为 `Optional` 且默认 `None`，导致 `_build_projectile_phases` 与 6-DOF builder 需要写 `or` 回退。
- 改为显式默认值（与 `MPMDynamics` / `SixDOFDynamics` 默认值一致）后，旧配置/YAML 无需改动，builder 可直接使用。
- `VehicleConfig.extra="forbid"` 不会破坏 pydantic 校验，新增字段有默认值，手写 `VehicleConfig(...)` 调用处无需修改。

#### B. `builder.py::_build_projectile_phases` 读取 VehicleConfig 参数
修改 `ballistic_sim/phases/builder.py`：

```python
dyn = MPMDynamics(
    mass_kg=cfg.vehicle.mass_kg,
    diameter_m=cfg.vehicle.diameter_m,
    form_factor=cfg.vehicle.cd or 1.0,
    drag_law=drag_law,
    twist_cal=cfg.vehicle.twist_cal,
    Ix=cfg.vehicle.Ix,
    It=cfg.vehicle.It,
    options=opt,
    lat_deg=cfg.launch.lat_deg,
)
```

当前 builder 未传递 `twist_cal/Ix/It`，默认使用 `MPMDynamics` 的默认值；传递后数值行为一致，但允许 preset 覆盖。

#### C. `projectiles.py::_projectile_config_from_preset` 读取 YAML 中的 MPM 参数
修改 `ballistic_sim/presets/projectiles.py`：

```python
return SimConfig(
    mission="projectile",
    vehicle=VehicleConfig(
        mass_kg=float(p["m"]),
        diameter_m=float(p["d"]),
        cd=float(p["i"]),
        area_ref_m2=None,
        drag_law=drag_law,
        twist_cal=float(p.get("twist_cal", 20.0)),
        Ix=float(p.get("Ix", 0.1)),
        It=float(p.get("It", 1.0)),
    ),
    ...
)
```

#### D. 处理 `_projectile_dynamics_from_preset`
当前 `_projectile_dynamics_from_preset(name)` 直接构造 `MPMDynamics`，但 **仓库内没有任何调用方**。本阶段直接删除该函数，并从 `projectiles.py` 的 `__all__` 中移除。

### 3.4 测试任务
1. 新增 `tests/test_vehicle_config_defaults.py`：
   - `VehicleConfig()` 默认值：`twist_cal=20.0`、`Ix=0.1`、`It=1.0`、`x_cp_cg=0.05`。
   - 显式覆盖仍生效。
2. 更新/新增 `tests/phases/test_builder_projectile.py`：
   - `build_phases(m107_config())` 生成的 `MPMDynamics` 的 `twist_cal/Ix/It` 与 YAML 一致。
   - 行为与 Stage 1 之前数值一致（回归测试）。
3. 运行完整回归：`pytest`、`coverage report`、`flake8`、`mypy`、`black --check .`。

### 3.5 验收标准
- [ ] `VehicleConfig()` 新增字段有显式默认值。
- [ ] `_build_projectile_phases` 使用 `cfg.vehicle.twist_cal/Ix/It`。
- [ ] `_projectile_config_from_preset` 从 YAML 读取 `twist_cal/Ix/It`。
- [ ] `_projectile_dynamics_from_preset` 已删除。
- [ ] pytest / coverage / lint 全绿。

### 3.6 风险点
- **默认值变更风险**：若将 `x_cp_cg` 默认值从 `None` 改为 `0.05`，需确认 `SixDOFDynamics` 当前 `or 0.05` 回退路径移除后数值不变。
- **pydantic 序列化**：`VehicleConfig` 新增字段后，现有 YAML/JSON 配置文件无需改动；但需确认 `save_config` / `load_config` 往返正常。

---

## 4. Stage 2：迁移内部调用方到 `SimConfig + build_phases`

### 4.1 阶段目标
在删除 deprecated phase 工厂之前，先把 CLI、API、GUI、targeting、脚本、测试等所有内部调用方改为 `cfg = ...; phases = build_phases(cfg)`（或直接后续调用 `simulate(cfg)`）。

### 4.2 涉及文件
- `ballistic_sim/cli.py`
- `ballistic_sim/api/main.py`
- `ballistic_sim/gui/builder.py`
- `ballistic_sim/guidance/targeting.py`
- `ballistic_sim/presets/missiles.py`
- `ballistic_sim/presets/rockets.py`
- `ballistic_sim/monte_carlo.py`
- `scripts/regression_bridge.py`
- `scripts/monte_carlo_demo.py`
- `scripts/comparison_demo.py`
- `scripts/benchmark.py`
- `tests/presets/test_presets.py`
- `tests/presets/test_missile_full_chain.py`
- `tests/presets/test_rocket_full_chain.py`
- `tests/test_performance.py`
- `docs/user_guide.md`

### 4.3 开发任务

#### A. CLI 占位 builder 收敛
`ballistic_sim/cli.py` 中 `_build_icbm_config` 与 `_build_suborbital_config` 已返回 `SimConfig`，`_build_config_and_phases` 已统一调用 `build_phases(cfg)`。本阶段只需确认：

- `_build_icbm_config` 不再使用任何 `missile_full_chain`。
- `_build_suborbital_config` 构造的 `SimConfig` 能被 `build_phases` 正确处理（当前 `build_phases` 对 `suborbital` 走 `_build_rocket_phases` 分支，已支持）。
- 删除 `_build_config_and_phases` 中任何遗留的 deprecated 调用（当前已无）。

#### B. API 收敛
`ballistic_sim/api/main.py` 中 `_build_config_and_phases` 已统一调用 `build_phases(cfg)`。本阶段只需确认无 `m107_phases` / `missile_phases` / `rocket_phases` 残留。

#### C. GUI builder 收敛
修改 `ballistic_sim/gui/builder.py`：

```python
from ballistic_sim.presets import (
    list_missiles,
    list_projectiles,
    list_rockets,
    m107_config,
    missile_full_config,
    rocket_full_config,
    _projectile_config_from_preset,  # 或新增公开 projectile_config
)

def build_config_and_phases(mission, preset=None):
    ...
    if mission == "projectile":
        name = preset or "M107"
        cfg = m107_config() if name == "M107" else _projectile_config_from_preset(name)
        phases = build_phases(cfg)
        return cfg, phases

    if mission == "missile":
        name = preset or default_preset(mission) or list_missiles()[0]
        cfg = missile_full_config(name)
        phases = build_phases(cfg)
        return cfg, phases

    if mission == "rocket":
        name = preset or "CZ2F"
        cfg = rocket_full_config(name)
        phases = build_phases(cfg)
        return cfg, phases
    ...
```

同时从 import 列表中移除 `m107_phases`、`projectile_phases`、`missile_phases`、`rocket_phases`、`missile_config`、`rocket_config`、`cz2f_phases`。

#### D. targeting.py 收敛
修改 `ballistic_sim/guidance/targeting.py`：

`_simulate_shaped` 当前使用 `missile_config(name)` + `missile_phases(name)`，并手动 patching phase guidance dict。改为：

```python
from ballistic_sim.presets.missiles import missile_full_config

def _simulate_shaped(name, kick_deg, ..., overrides=None):
    cfg = missile_full_config(name)
    ov = {"guidance.kick_deg": float(kick_deg)}
    if az_deg is not None:
        ov["launch.azimuth_deg"] = float(az_deg)
    if overrides:
        ov.update(overrides)
    cfg = apply_overrides(cfg, ov)
    phases = build_phases(cfg)
    ...
    return simulate(cfg, phases=phases)
```

注意：`missile_full_config(name)` 的 `mission` 字段来自 YAML（可能是 `missile` 或 `icbm`）。 targeting 用于战术弹道导弹，应确保传入 name 对应的 YAML `mission` 为 `missile`；若存在 ICBM 预设被误用，可在本函数入口断言 `cfg.mission == "missile"`。

#### E. `missile_config` 与 `rocket_config` 调整
- `missile_config(name)` 当前返回无 stages 的简化配置，且被 targeting 间接依赖。本阶段将其改为委托给 `missile_full_config(name)` 并强制 `mission="missile"`：
  ```python
  def missile_config(name: str) -> SimConfig:
      cfg = missile_full_config(name)
      return cfg.model_copy(update={"mission": "missile"})
  ```
- `rocket_config(name)` 当前为简化配置，本阶段不强制删除，但所有需要 phase 的内部调用方应改用 `rocket_full_config(name)`。`rocket_config` 可保留给只需要 `SimConfig` 的调用方，或在 P2 进一步统一。

#### F. `monte_carlo.py` 调整
`monte_carlo.py` 中 `_run_single` 当前显式 `build_phases(cfg_p)` 后传入 `simulate`。Stage 4 会统一改为 `simulate(cfg_p)`，但 Stage 2 先保证它不再使用 deprecated 工厂函数（当前已无）。

#### G. 脚本与测试迁移
按以下映射统一替换：

| 原用法 | 新用法 |
|--------|--------|
| `m107_phases()` | `build_phases(m107_config())` |
| `projectile_phases(name)` | `build_phases(_projectile_config_from_preset(name))`（或新增 `projectile_config(name)`） |
| `missile_phases(name)` | `build_phases(missile_full_config(name))` |
| `rocket_phases(cfg, name=...)` | `build_phases(rocket_full_config(name))` 或 `build_phases(cfg)`（若 cfg 已含 stages） |
| `missile_full_chain(name)` | `build_phases(missile_full_config(name))` |
| `rocket_full_chain(name)` | `build_phases(rocket_full_config(name))` |

具体文件修改：
- `scripts/regression_bridge.py`：`m107_phases()` → `build_phases(m107_config())`。
- `scripts/monte_carlo_demo.py`：`m107_phases()` / `cz2f_phases()` → `build_phases(...)`。
- `scripts/comparison_demo.py`：`m107_phases()` → `build_phases(m107_config())`。
- `scripts/benchmark.py`：`m107_phases()` / `rocket_phases(cfg, name="CZ2F")` → `build_phases(...)`。
- `tests/presets/test_presets.py`：重写为验证 `m107_config` / `missile_full_config` / `rocket_full_config` + `build_phases`。
- `tests/presets/test_missile_full_chain.py`：`missile_full_chain("ICBM_8000")` → `build_phases(missile_full_config("ICBM_8000"))`。
- `tests/presets/test_rocket_full_chain.py`：`rocket_full_chain(name)` → `build_phases(rocket_full_config(name))`。
- `tests/test_performance.py`：`m107_phases()` → `build_phases(m107_config())`。
- `docs/user_guide.md`：示例改为 `from ballistic_sim import build_phases, simulate, SimConfig` 或 `from ballistic_sim.presets import m107_config; phases = build_phases(m107_config())`。

### 4.4 测试任务
1. 运行 `pytest tests/presets tests/test_cli.py tests/test_api.py tests/phases/test_builder_advanced.py`。
2. 运行 CLI smoke：`--mission projectile`、`--mission rocket`、`--mission missile`、`--mission icbm --preset ICBM_8000`、`--mission suborbital`。
3. 运行 API smoke：5 个 `/simulate/{mission}` 端点。
4. 运行脚本 smoke：`python scripts/comparison_demo.py`、`python scripts/monte_carlo_demo.py`（可接受慢标记）。
5. 运行完整回归与 lint。

### 4.5 验收标准
- [ ] `grep -R "m107_phases\|projectile_phases\|missile_phases\|rocket_phases\|missile_full_chain\|rocket_full_chain" ballistic_sim/ tests/ scripts/ docs/` 无内部调用（仅 `presets/*.py` 中函数定义本身除外）。
- [ ] GUI `build_config_and_phases` 对 5 个 mission 均返回 `SimConfig + build_phases(cfg)`。
- [ ] `targeting.py` 使用 `missile_full_config` + `build_phases`。
- [ ] pytest / coverage / lint 全绿。

### 4.6 风险点
- **`missile_full_config` 与 `missile_config` 语义差异**：`missile_full_config` 读取 YAML 的 `mission` 字段，ICBM 预设可能是 `icbm`。targeting 需要 `missile`，需通过 `model_copy(update={"mission": "missile"})` 覆盖。
- **GUI 火箭 preset 变更**：从 `rocket_config + rocket_phases` 改为 `rocket_full_config + build_phases`，入轨/再入行为可能略有不同，需 smoke 验证。

---

## 5. Stage 3：移除 deprecated phase 工厂

### 5.1 阶段目标
彻底删除 6 个 deprecated phase 工厂函数，并清理 `ballistic_sim/presets/__init__.py` 的导出与 `__all__`。

### 5.2 涉及文件
- `ballistic_sim/presets/projectiles.py`
- `ballistic_sim/presets/missiles.py`
- `ballistic_sim/presets/rockets.py`
- `ballistic_sim/presets/__init__.py`
- 可能受影响的导入：无（Stage 2 已迁移全部内部调用方）

### 5.3 开发任务

#### A. `ballistic_sim/presets/projectiles.py`
删除：
- `projectile_phases(name)`
- `m107_phases()`

保留：
- `m107_config()`
- `_projectile_config_from_preset(name)`（内部使用）
- `list_projectiles` 等

更新 `__all__`：
```python
__all__ = [
    "m107_config",
    "_projectile_config_from_preset",
    "list_projectiles",
]
```

#### B. `ballistic_sim/presets/missiles.py`
删除：
- `missile_phases(name)`
- `missile_full_chain(name)`

保留：
- `missile_config(name)`
- `missile_full_config(name)`
- `list_missiles` 等

更新 `__all__`：
```python
__all__ = [
    "missile_config",
    "missile_full_config",
    "list_missiles",
]
```

#### C. `ballistic_sim/presets/rockets.py`
删除：
- `rocket_phases(cfg, name=..., payload_mass_kg=...)`
- `rocket_full_chain(name, payload_mass_kg=...)`
- `cz2f_phases(cfg, payload_mass_kg=...)`

保留：
- `rocket_config(name, ...)`
- `rocket_full_config(name, ...)`
- `rocket_stages(name, ...)`
- `cz2f_config(...)`

更新 `__all__`：
```python
__all__ = [
    "cz2f_config",
    "rocket_config",
    "rocket_full_config",
    "rocket_stages",
    "list_rockets",
]
```

注意：`rocket_full_chain` 内部调用的 `_set_terminal_target` 若不再被使用，可一并删除或保留为私有辅助函数。

#### D. `ballistic_sim/presets/__init__.py`
从 import 与 `__all__` 中移除全部已删除的 phase 工厂。

更新后导出：
```python
__all__ = [
    "m107_config",
    "projectile_config",  # 若 Stage 2 新增
    "cz2f_config",
    "rocket_config",
    "rocket_full_config",
    "rocket_stages",
    "missile_config",
    "missile_full_config",
    "list_projectiles",
    "list_missiles",
    "list_rockets",
]
```

### 5.4 测试任务
1. 确认 `grep` 无残留调用。
2. 运行完整 pytest，确认无 `ImportError`。
3. 运行 lint。

### 5.5 验收标准
- [ ] `ballistic_sim/presets/projectiles.py` 不再含 `m107_phases`、`projectile_phases`。
- [ ] `ballistic_sim/presets/missiles.py` 不再含 `missile_phases`、`missile_full_chain`。
- [ ] `ballistic_sim/presets/rockets.py` 不再含 `rocket_phases`、`rocket_full_chain`、`cz2f_phases`。
- [ ] `ballistic_sim/presets/__init__.py` 不再导出上述函数。
- [ ] pytest / coverage / lint 全绿。

### 5.6 风险点
- **外部用户破坏**：这些函数在 v0.4.0 仅标记 deprecated 但未移除，v0.5.0 移除属于 breaking change。需在 CHANGELOG 中明确列出。
- **脚本/文档遗漏**：Stage 2 必须彻底 grep，否则 Stage 3 会触发 ImportError。

---

## 6. Stage 4：`simulate(cfg, phases=None)` 与版本号升级

### 6.1 阶段目标
让 `simulate()` 的 `phases` 参数可选，默认自动 `build_phases(cfg)`；显式传 phases 时发出 `DeprecationWarning`。同时把所有内部调用改为 `simulate(cfg)`，最后统一升级版本号到 `0.5.0`。

### 6.2 涉及文件
- `ballistic_sim/simulator.py`
- `ballistic_sim/cli.py`
- `ballistic_sim/api/main.py`
- `ballistic_sim/monte_carlo.py`
- `ballistic_sim/gui/builder.py`（若 GUI 直接调用 simulate）
- `ballistic_sim/guidance/targeting.py`
- `scripts/*.py`
- `tests/*.py`
- `ballistic_sim/__init__.py`
- `pyproject.toml`
- `ballistic_sim/api/main.py`
- `tests/test_version.py`
- `scripts/release_check.py`
- `CHANGELOG.md`

### 6.3 开发任务

#### A. 修改 `simulate` 签名
`ballistic_sim/simulator.py`：

```python
from typing import Optional
import warnings

def simulate(
    cfg: SimConfig,
    phases: Optional[List[Phase]] = None,
    reuse_context: bool = True,
) -> SimResult:
    """统一仿真主循环。

    Parameters
    ----------
    phases:
        Phase 列表。自 v0.5.0 起已弃用显式传入；缺省时内部自动调用
        ``build_phases(cfg)``。
    reuse_context:
        是否复用 ``cfg`` 上已绑定的动力学上下文。
    """
    issues = validate_config(cfg)
    ...
    if phases is None:
        phases = build_phases(cfg)
    else:
        warnings.warn(
            "显式传入 phases 自 v0.5.0 起已弃用，请改用 simulate(cfg)。",
            DeprecationWarning,
            stacklevel=2,
        )
        # 保留对空列表的历史兼容
        if not phases:
            phases = build_phases(cfg)
    ...
```

#### B. 迁移内部调用方
将以下调用统一改为 `simulate(cfg)`：

- `ballistic_sim/cli.py`：`result = simulate(cfg, phases=phases)` → `result = simulate(cfg)`
- `ballistic_sim/api/main.py`：
  - `_run_simulation` 中 `result = simulate(cfg, phases=phases)` → `result = simulate(cfg)`
  - `trajectory3d_endpoint` 中 `result = simulate(cfg, phases=phases)` → `result = simulate(cfg)`
- `ballistic_sim/monte_carlo.py`：`result = simulate(cfg_p, phases=phases, reuse_context=True)` → `result = simulate(cfg_p)`（simulate 内部会处理 context）
- `ballistic_sim/guidance/targeting.py`：`simulate(cfg, phases=phases)` → `simulate(cfg)`
- `ballistic_sim/gui/builder.py`：若 GUI 直接调用 simulate，同样改为无 phases。
- `scripts/*.py`：`simulate(cfg, phases=...)` → `simulate(cfg)`
- `tests/*.py`：smoke 测试改为 `simulate(cfg)`；保留 1~2 个显式测试验证 deprecation warning。

#### C. 保留显式 phases 的回归测试
在 `tests/test_simulator.py` 或新增 `tests/test_simulate_phases_deprecation.py` 中：

```python
import warnings
from ballistic_sim.presets import m107_config
from ballistic_sim.phases.builder import build_phases
from ballistic_sim.simulator import simulate

def test_explicit_phases_emits_deprecation_warning():
    cfg = m107_config()
    phases = build_phases(cfg)
    with pytest.warns(DeprecationWarning, match="phases"):
        simulate(cfg, phases=phases)
```

#### D. 版本号统一升级到 0.5.0
修改以下文件：
1. `ballistic_sim/__init__.py`：`__version__ = "0.5.0"`
2. `pyproject.toml`：`version = "0.5.0"`
3. `ballistic_sim/api/main.py`：`version="0.5.0"`
4. `tests/test_version.py`：`TARGET_VERSION = "0.5.0"`
5. `scripts/release_check.py`：`TARGET_VERSION = "0.5.0"`
6. `CHANGELOG.md`：在顶部新增 `## [0.5.0] - 2026-07-05` 章节，记录：
   - Removed: 6 个 deprecated phase 工厂函数
   - Changed: `simulate(cfg, phases=None)` 自动 build phases；显式传 phases 弃用
   - Changed: CLI/API/GUI/targeting 全面收敛到 `SimConfig + build_phases`
   - Added: `VehicleConfig` 新增 `twist_cal/Ix/It/x_cp_cg` 默认值

### 6.4 测试任务
1. 运行完整 pytest，确认无 `DeprecationWarning` 失败（除非测试本身断言 warning）。
2. 运行 `pytest -W error::DeprecationWarning tests/test_api.py tests/test_cli.py` 确认内部调用不再触发 warning。
3. CLI/API smoke 全过。
4. 运行 `scripts/release_check.py`（或至少执行其中版本/CHANGELOG/pytest 检查）。
5. `black --check .`、`flake8 ballistic_sim tests`、`mypy ballistic_sim`。

### 6.5 验收标准
- [ ] `simulate(cfg)` 单参数调用成功，内部自动 `build_phases(cfg)`。
- [ ] `simulate(cfg, phases=phases)` 仍成功，但抛出 `DeprecationWarning`。
- [ ] CLI/API/GUI/monte_carlo/targeting/scripts 均不再显式传 phases。
- [ ] `ballistic_sim.__version__`、`pyproject.toml`、`FastAPI version`、`tests/test_version.py`、`scripts/release_check.py` 一致为 `0.5.0`。
- [ ] `CHANGELOG.md` 含 `## [0.5.0]` 章节。
- [ ] pytest / coverage / lint 全绿。

### 6.6 风险点
- **Monte Carlo 性能**：`monte_carlo.py` 每个样本调用 `simulate(cfg_p)` 会重新 `build_phases`。若 profiling 显示瓶颈，可在后续版本缓存 phase 模板，但 P1 以保持语义正确优先。
- **外部用户代码**：显式传 phases 的用户会收到 deprecation warning。这是预期行为，文档中需说明。

---

## 7. Deprecated 函数迁移路径总表

| 函数 | 当前定义位置 | 内部调用方 | 迁移路径 | 是否删除 |
|------|-------------|-----------|---------|---------|
| `m107_phases()` | `presets/projectiles.py` | GUI、scripts、tests、docs | `build_phases(m107_config())` | 是 |
| `projectile_phases(name)` | `presets/projectiles.py` | GUI、tests | `build_phases(_projectile_config_from_preset(name))` 或新增 `projectile_config(name)` | 是 |
| `missile_phases(name)` | `presets/missiles.py` | GUI、targeting、tests | `build_phases(missile_full_config(name))` | 是 |
| `rocket_phases(cfg, name, ...)` | `presets/rockets.py` | GUI、tests、scripts | `build_phases(rocket_full_config(name))` 或 `build_phases(cfg)` | 是 |
| `missile_full_chain(name)` | `presets/missiles.py` | tests | `build_phases(missile_full_config(name))` | 是 |
| `rocket_full_chain(name)` | `presets/rockets.py` | tests | `build_phases(rocket_full_config(name))` | 是 |
| `_projectile_dynamics_from_preset(name)` | `presets/projectiles.py` | 无 | 直接删除 | 是 |
| `cz2f_phases(cfg, ...)` | `presets/rockets.py` | scripts、tests | `build_phases(rocket_full_config("CZ2F"))` | 是 |
| `missile_config(name)` | `presets/missiles.py` | targeting、tests | 保留，改为委托 `missile_full_config(name)` 并强制 `mission="missile"` | 否 |
| `rocket_config(name)` | `presets/rockets.py` | tests、scripts | 保留为简化配置；需要 phase 的调用方改用 `rocket_full_config` | 否 |

---

## 8. `simulate(cfg, phases=None)` 冲突评估与平滑迁移策略

### 8.1 当前状态
- `simulate(cfg, phases, reuse_context=True)` 要求必须传 `phases`。
- CLI、API、monte_carlo 等都在调用前显式 `build_phases(cfg)`。

### 8.2 变更后行为
- `simulate(cfg)`：内部自动 `build_phases(cfg)`，与外部显式 build 结果一致。
- `simulate(cfg, phases=phases)`：仍可工作，但发出 `DeprecationWarning`。

### 8.3 冲突评估
1. **内部调用冲突**：无。Stage 4 会把所有内部调用改为 `simulate(cfg)`。
2. **测试冲突**：部分旧测试显式传 phases，改为 `simulate(cfg)` 即可；保留少量 warning 测试。
3. **外部用户冲突**：外部用户若显式传 phases，会收到 deprecation warning，但功能不受影响，可平滑迁移。
4. **Monte Carlo 上下文绑定冲突**：`monte_carlo.py` 当前在调用前绑定 `_dynamics_context`。改为 `simulate(cfg_p)` 后，simulate 内部会检查并复用/创建 context，行为一致。

### 8.4 平滑迁移策略
- **阶段内**：先改签名并保留 warning，再迁移内部调用方，最后删除 deprecated 工厂。
- **对外**：在 CHANGELOG 中明确说明：
  - v0.5.0 移除 6 个 phase 工厂；
  - `simulate(cfg, phases=...)` 仍可用但已弃用；
  - 推荐写法：`phases = build_phases(cfg); result = simulate(cfg)` 或未来直接 `result = simulate(cfg)`。

---

## 9. 版本号处理

P1 完成后版本号升级为 `0.5.0`。需修改的文件：

| 文件 | 当前值 | 目标值 | 说明 |
|------|--------|--------|------|
| `ballistic_sim/__init__.py` | `__version__ = "0.4.0"` | `"0.5.0"` | 包版本 |
| `pyproject.toml` | `version = "0.4.0"` | `"0.5.0"` | 构建/安装版本 |
| `ballistic_sim/api/main.py` | `version="0.4.0"` | `"0.5.0"` | FastAPI 应用版本 |
| `tests/test_version.py` | `TARGET_VERSION = "0.4.0"` | `"0.5.0"` | 版本断言 |
| `scripts/release_check.py` | `TARGET_VERSION = "0.4.0"` | `"0.5.0"` | 发布检查 |
| `CHANGELOG.md` | `## [0.4.0]` | 顶部新增 `## [0.5.0]` | 变更日志 |

---

## 10. 全局验收标准

- [ ] `pytest` 全绿（至少维持 v0.4.0 基线）。
- [ ] 覆盖率 ≥ v0.4.0 基线，不下降。
- [ ] `flake8 ballistic_sim tests`、`mypy ballistic_sim`、`black --check .` 全绿。
- [ ] CLI 5 个 mission smoke 测试通过。
- [ ] API 5 个 `/simulate/{mission}` smoke 测试通过。
- [ ] `grep -R "m107_phases\|projectile_phases\|missile_phases\|rocket_phases\|missile_full_chain\|rocket_full_chain\|cz2f_phases" ballistic_sim/ tests/ scripts/ docs/` 无命中（历史文档/CHANGELOG 除外）。
- [ ] `simulate(cfg)` 单参数调用成功。
- [ ] `simulate(cfg, phases=phases)` 发出 `DeprecationWarning`。
- [ ] 版本号在 5 个关键文件一致为 `0.5.0`。

---

## 11. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| `missile_full_config` 的 `mission` 字段可能为 `icbm`，targeting 需要 `missile` | targeting 语义错误 | `missile_config` 委托时强制 `mission="missile"`；targeting 入口加断言 |
| GUI 火箭从 `rocket_config+rocket_phases` 切到 `rocket_full_config+build_phases` 后入轨/再入行为变化 | GUI smoke 失败 | Stage 2 单独跑 GUI smoke；必要时调整 `rocket_full_config` 默认 target orbit |
| Monte Carlo 每个样本重新 build_phases 带来性能下降 | 大样本 MC 变慢 | P1 以保证正确性优先；P2 可引入 phase 模板缓存 |
| 删除 `_projectile_dynamics_from_preset` 后，若未来需要直接构造 `MPMDynamics` 失去入口 | 低 | 该函数当前无调用方；如需要可通过 `build_phases(cfg)[0].dynamics` 获取 |
| 外部用户代码因移除 phase 工厂而破坏 | 中 | 在 CHANGELOG 显著位置列出 breaking changes；v0.5.0 作为 minor 版本符合语义化版本对 breaking change 的约定（0.x 版本允许） |

---

## 12. 时间线建议

| 阶段 | 预估工作量 | 关键产出 |
|------|-----------|---------|
| Stage 1 | 0.5 d | `VehicleConfig` 默认值 + projectile builder 读取 preset 参数 |
| Stage 2 | 1 ~ 1.5 d | CLI/API/GUI/targeting/scripts/tests 全部迁移到 `build_phases` |
| Stage 3 | 0.5 d | 删除 6 个 deprecated phase 工厂 + 清理导出 |
| Stage 4 | 0.5 d | `simulate(cfg)` 单参数 + deprecation warning + 版本号 0.5.0 |
| 合计 | **2.5 ~ 3 d** | v0.5.0 发布就绪 |
