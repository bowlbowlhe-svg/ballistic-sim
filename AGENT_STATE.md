# Agent 协作状态看板

## 当前阶段
阶段 4：入口与可视化

## 阶段状态
- [x] 阶段 0：项目骨架与迁移准备（v0.1.0-stage0）
- [x] 阶段 1：物理模型统一层（v0.1.0-stage1）
- [x] 阶段 2：运动方程与阶段驱动（v0.1.0-stage2）
- [x] 阶段 3：制导与火控（v0.1.0-stage3）
- [x] Developer: CLI 合并（支持 --mission 五种类型与 --viz 默认出图）
- [x] Developer: 大气层内可视化（三维弹道、地面航迹/落点、高度-射程、风偏）
- [x] Developer: 大气层外可视化（地球三维轨迹、地面轨迹、轨道插入、Δv 预算、再入热流）
- [x] Developer: 动画生成骨架（GIF/MP4）
- [x] Developer: 预设 YAML 迁移
- [ ] Tester: 可视化冒烟测试（每个 viz/ 函数返回非空 Figure）
- [ ] Tester: CLI 端到端测试（所有 mission 类型产出预期 PNG）
- [ ] Tester: 动画生成冒烟测试

## 阶段 4 范围说明
- v1 只承诺 CLI + 静态可视化；GUI 作为可选增强，可延至 v1.1。
- Web 前端明确放第二阶段。

## 下一阶段触发条件
本阶段所有 checklist 完成后，进入阶段 5。

## 阻塞项
无

## 备注
阶段 4 目标：提供统一 CLI 入口和分域可视化出图能力。
Developer 阶段 4 完成。
