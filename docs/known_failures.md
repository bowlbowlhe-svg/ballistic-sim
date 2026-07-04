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
| KF-001 | D30 炮弹，高射角 QE=50°，东侧风 wind_e=8 m/s，模型选 6-DOF | 历史上 RK45 步长坍缩，姿态四元数积分发散，``SixDOFModel.simulate`` 抛出 ``RuntimeError("6-DOF 仿真失败")`` | 1. 构造 ``SixDOFDynamics``（D30 参数）<br>2. ``initial_state(v0=690, theta_deg=50, az_deg=90, h0=0)``<br>3. 设置 ``UniformWind(e=8)`` 并积分<br>或运行 ``pytest tests/dynamics/test_six_dof_known_failures.py::test_kf_001_d30_high_qe_crosswind_lands`` | **v0.2.0-stage1 已修复**：统一采用 scalar-last、世界系角速度左乘四元数导数，并加入四元数归一化。现为回归用例。 |

## 待验证 / 风险区域

以下工况在阶段 0 的默认测试套件中未覆盖到，后续迁移 6-DOF 模块时应重点补充回归用例：

- 高攻角（>30°）下的气动力矩耦合。
- 低空高速飞行时的动压跳变。
- 推力偏心距较大时的姿态发散。
- 初始角速度过大导致的陀螺耦合发散。
- 静稳定距接近零（``Xcp ≈ Xcg``）时警告并退化为默认 0.05 m，可能影响稳定性结论。

## 阶段 3 6-DOF 控制限制

阶段 3 完成 6-DOF 控制律接口（``ballistic_sim.guidance.control``）迁移，但明确限制以下行为，避免从当前 3-DOF/MPM 流程中隐式升维导致积分发散或结果不可解释：

| 编号 | 配置/输入 | 现象 | 复现步骤 | 备注 |
|---|---|---|---|---|
| KF-002 | 直接对 3-DOF（7 维）或 MPM（8 维）状态调用 ``SixDOFControl.control_moment`` | 抛出 ``NoAutoDimensionUpgrade``，提示须显式提供 13 维状态 | 1. ``ctrl = SixDOFControl()``<br>2. ``y = np.zeros(7)``<br>3. ``ctrl.control_moment(y)`` | 设计约束：禁止 3-DOF → 6-DOF 自动升维；姿态/角速度信息必须由用户显式给出。 |
| KF-003 | 提供 13 维状态但初始四元数模长为 0 | 抛出 ``NoAutoDimensionUpgrade``，视为未提供初始姿态 | 1. ``y = np.concatenate([np.zeros(6), [0,0,0,0], np.zeros(3)])``<br>2. ``SixDOFControl().control_moment(y)`` | 四元数零向量无法定义有效初始姿态。 |
| KF-004 | 在火箭/ICBM 动力上升段启用 6-DOF 控制 | 当前未接入 PoweredECIDynamics；控制矩接口被调用但不会影响 3-DOF 轨迹 | 1. 构建 ``PoweredECIDynamics``<br>2. 尝试注入 ``SixDOFControl`` | 阶段 3 仅完成控制律接口骨架；6-DOF 动力学闭环集成待阶段 4/5。 |

## 待验证 / 风险区域（阶段 3 新增）

- 6-DOF 控制律与真实气动力矩/惯量耦合后的数值稳定性。
- 再入段高动态压力下鸭舵饱和与脉冲发动机死区设计。
- 多阶段任务中 6-DOF 状态在阶段间投影（降维/升维）的守恒性。
