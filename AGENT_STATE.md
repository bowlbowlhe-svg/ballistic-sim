# Agent 协作状态看板

## 当前阶段
阶段 1：物理模型统一层

## 阶段状态
- [x] 阶段 0：项目骨架与迁移准备（已提交并 tag v0.1.0-stage0）
- [x] Developer: 迁移并合并大气模型（含 86 km 以上指数延拓）
- [x] Developer: 迁移并合并引力模型
- [x] Developer: 迁移并合并阻力/气动模型
- [x] Developer: 迁移并合并风模型
- [x] Developer: 迁移地形模型（含内存缓存与两层落地检测）
- [x] Developer: 实现 optional 依赖导入守卫
- [x] Tester: 为每个物理模型编写单元测试
- [x] Tester: 大气模型 86 km 连续性与高空衰减测试
- [x] Tester: 坐标转换 ECI↔ECEF↔ENU 往返一致性测试
- [x] Tester: 地形缓存与两层检测一致性测试

## 阶段 1 测试摘要
- pytest 通过：56 / 56
- 整体覆盖率：53.12%（Stage 1 暂设 fail_under=50，Stage 5 目标 ≥80%）
- 报告文件：`tests/stage1_report.txt`

## 下一阶段触发条件
本阶段所有 checklist 完成后，进入阶段 2。

## 阻塞项
无

## 备注
阶段 1 已完成：物理模型（大气、引力、阻力/气动、风、地形）、坐标转换、导入守卫、56 项测试。
