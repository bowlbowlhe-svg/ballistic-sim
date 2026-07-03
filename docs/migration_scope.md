# 待迁移模块清单

本文件用于记录从旧项目向 `ballistic-sim` 迁移的模块与状态。
旧项目位置：

- `D:\Pycharm\游戏与兴趣\外弹道方程`
- `D:\Pycharm\游戏与兴趣\火箭仿真程序`

> 注意：仅做清单与迁移状态记录，不修改旧项目源文件。

## 模块清单

| 旧项目 | 旧文件/模块 | 目标位置 | 状态 | 备注 |
|--------|------------|----------|------|------|
| 外弹道方程 | `constants.py` | `ballistic_sim/models/constants.py` | ⬜ 待迁移 | 合并大气、地球、单位常量 |
| 外弹道方程 | `atmosphere.py` | `ballistic_sim/models/atmosphere.py` | ⬜ 待迁移 | 标准大气模型 |
| 火箭仿真程序 | `rocket.py` | `ballistic_sim/dynamics/six_dof.py` | ⬜ 待迁移 | 6-DOF 动力学 |
| 火箭仿真程序 | `guidance.py` | `ballistic_sim/guidance/` | ⬜ 待迁移 | 制导律 |
| 火箭仿真程序 | `visualization.py` | `ballistic_sim/viz/` | ⬜ 待迁移 | 可视化 |

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
