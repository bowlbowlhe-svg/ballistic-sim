# 已知失效清单

本文件记录会导致六自由度积分发散或结果异常的输入工况，供回归测试与调试参考。

> 由 Tester 在阶段 0 普查后填写。

## 阶段 0 测试执行摘要

- **测试范围**：``D:\Pycharm\游戏与兴趣\外弹道方程`` 中的
  ``tests/test_sixdof.py``、``tests/test_sixdof_quaternion.py``、
  ``tests/test_firecontrol.py``。
- **执行结果**：23 条用例全部通过，未观察到新的发散或崩溃。
- **环境**：``D:/Pycharm/游戏与兴趣/外弹道方程/venv/Scripts/python.exe``，
  通过 ``PYTHONPATH=.`` 运行。

## 6-DOF 发散工况

| 编号 | 配置/输入 | 现象 | 复现步骤 | 备注 |
|---|---|---|---|---|
| KF-001 | D30 炮弹，高射角 QE=50°，东侧风 wind_e=8 m/s，模型选 6-DOF | 历史上 RK45 步长坍缩，姿态四元数积分发散，``SixDOFModel.simulate`` 抛出 ``RuntimeError("6-DOF 仿真失败")`` | 1. ``cfg = config.build_config("D30", model="6dof", overrides={"qe": 50, "wind_e": 8})``<br>2. ``res = config.simulate(cfg)``<br>或运行 ``pytest tests/test_sixdof.py::test_sixdof_high_qe_crosswind_no_divergence`` | 原项目已知问题；根因为四元数运动学帧约定不一致（世界系角速度误用体系右乘）。已在当前代码修复，现作为回归用例保留。 |

## 待验证 / 风险区域

以下工况在阶段 0 的默认测试套件中未覆盖到，后续迁移 6-DOF 模块时应重点补充回归用例：

- 高攻角（>30°）下的气动力矩耦合。
- 低空高速飞行时的动压跳变。
- 推力偏心距较大时的姿态发散。
- 初始角速度过大导致的陀螺耦合发散。
- 静稳定距接近零（``Xcp ≈ Xcg``）时警告并退化为默认 0.05 m，可能影响稳定性结论。
