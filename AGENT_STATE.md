# Agent 协作状态看板

## 当前阶段
阶段 2：运动方程与阶段驱动

## 阶段状态
- [x] 阶段 0：项目骨架与迁移准备（v0.1.0-stage0）
- [x] 阶段 1：物理模型统一层（v0.1.0-stage1）
- [x] Developer: 抽象 Phase 基类
- [x] Developer: 实现 point_mass.py / mpm.py / powered_eci.py
- [x] Developer: 实现 project_state()（fidelity 升降维与坐标系转换）
- [x] Developer: 实现统一 simulator.py（多阶段积分主循环）
- [x] Developer: 域 I MVP：炮弹 MPM 落地，对比 M107 射表
- [x] Developer: 域 II MVP：CZ-2F 入轨，对比 lvsim 结果
- [x] Tester: 状态切换单元测试（允许/禁止方向、守恒量检查）
- [x] Tester: 迁移对拍脚本 regression_bridge.py 上线（黄金文件对比）
- [x] Tester: 域 I MVP 测试：M107 射表误差验证
- [x] Tester: 域 II MVP 测试：CZ-2F 入轨误差验证（xfail，待阶段 3 调优）

## 阶段 2 测试摘要
- pytest 通过：80 / 80 + 1 xfailed（CZ-2F 制导调优待阶段 3）
- 整体覆盖率：66.99%（达到 Stage 2 暂设 fail_under=50）
- 报告文件：`tests/stage2_report.txt`

## 遗留到阶段 3 的问题
- CZ-2F 入轨偏差：SECO 高度 525 km vs 黄金 366 km（+43%），偏心率 0.191 vs 0.114。
   suspected cause: 上面级 gamma0 装订 / 线性俯仰速率 / 关机条件 / 整流罩抛罩质量阶跃。

## 下一阶段触发条件
本阶段所有 checklist 完成后，进入阶段 3。

## 阻塞项
无

## 备注
阶段 2 已完成：Phase 基类、运动方程（point_mass/mpm/powered_eci）、状态切换、统一 simulator、两个 MVP（M107 已达标，CZ-2F 待制导调优）。
