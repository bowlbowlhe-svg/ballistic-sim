# 待迁移模块清单

本文件用于记录从旧项目向 `ballistic-sim` 迁移的模块与状态。
旧项目位置：

- `D:\Pycharm\游戏与兴趣\外弹道方程`
- `D:\Pycharm\游戏与兴趣\火箭仿真程序`

> 注意：仅做清单与迁移状态记录，不修改旧项目源文件。

## 模块清单

| 旧项目 | 旧文件/模块 | 目标位置 | 状态 | 备注 |
|--------|------------|----------|------|------|
| 外弹道方程 | `constants.py` | `ballistic_sim/constants.py` | ✅ 已完成 | v0.1.0 已合并 WGS84/ISA/单位常量 |
| 外弹道方程 | `atmosphere.py` | `ballistic_sim/models/atmosphere.py` | ✅ 已完成 | v0.1.0 ISA/US76 + 86km 指数延拓 |
| 外弹道方程 | `sixdof.py` | `ballistic_sim/dynamics/six_dof.py` | ✅ 已完成 | v0.2.0-stage1 迁移 13 维 6-DOF 刚体动力学 |
| 外弹道方程 | `mpm.py` | `ballistic_sim/dynamics/mpm.py` | ✅ 已完成 | v0.1.0-stage2 修正质点模型 |
| 外弹道方程 | `wind.py` / `wind_field.py` | `ballistic_sim/models/wind.py` | ✅ 已完成 | v0.1.0-stage1 多风模型 + GRIB2 导入守卫 |
| 外弹道方程 | `terrain.py` / `srtm.py` | `ballistic_sim/models/terrain.py` | ✅ 已完成 | v0.1.0-stage1 程序化/SRTM/GeoTIFF/Null 地形 |
| 外弹道方程 | `firecontrol.py` | `ballistic_sim/guidance/firecontrol.py` | ✅ 已完成 | v0.1.0-stage3 火炮诸元解算 |
| 外弹道方程 | `control.py` / 鸭舵/脉冲 | `ballistic_sim/guidance/control.py` | ✅ 已完成 | v0.1.0-stage3 接口 + v0.2.0-stage1 接入动力学 |
| 火箭仿真程序 | `rocket.py` / `propulsion.py` | `ballistic_sim/dynamics/powered_eci.py` / `models/propulsion.py` | ✅ 已完成 | v0.1.0-stage2/stage3 动力上升 + PEG |
| 火箭仿真程序 | `guidance.py` | `ballistic_sim/guidance/open_loop.py` / `peg.py` | ✅ 已完成 | v0.1.0-stage3 开环/PEG/圆化 |
| 火箭仿真程序 | `visualization.py` | `ballistic_sim/viz/` | ✅ 已完成 | v0.1.0-stage4 静态图 + v0.2.0-stage1 6-DOF 姿态扩展预留 |
| 外弹道方程 | `ballistics/batch_mpm.py` | `ballistic_sim/dynamics/batch_mpm.py` | ✅ 已完成 | v0.2.0-stage2.2 NumPy/CuPy 向量化批量 MPM |
| 外弹道方程 | `ballistics/gpu_mpm.py` | `ballistic_sim/dynamics/gpu_mpm.py` | ✅ 已完成 | v0.2.0-stage2.2 CuPy GPU 批量 MPM |
| 外弹道方程 | `ballistics/monte_carlo.py` | `ballistic_sim/monte_carlo.py` | ✅ 已完成 | v0.2.0-stage2.2 Monte Carlo 散布分析 |
| 外弹道方程 | `ballistics/viz/montecarlo_plot.py` | `ballistic_sim/viz/monte_carlo.py` | ✅ 已完成 | v0.2.0-stage2.2 蒙特卡洛落点散布图 |

## 迁移原则

1. 先迁移常量与配置，建立统一的 `SimConfig`。
2. 再迁移并抽象动力学模型，补充导入守卫（import guards）。
3. 旧项目的测试脚本优先转为 `tests/` 下的 pytest 用例。
4. 对旧项目中已知的 6-DOF 发散工况，登记在 `known_failures.md` 中。

## 状态图例

- ⬜ 待迁移
- 🔄 迁移中
- ✅ 已完成
- ❌ 不迁移 / 废弃
