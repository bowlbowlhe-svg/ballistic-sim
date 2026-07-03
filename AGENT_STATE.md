# Agent 协作状态看板

## 当前阶段
阶段 3：制导与火控

## 阶段状态
- [x] 阶段 0：项目骨架与迁移准备（v0.1.0-stage0）
- [x] 阶段 1：物理模型统一层（v0.1.0-stage1）
- [x] 阶段 2：运动方程与阶段驱动（v0.1.0-stage2）
- [x] Developer: 迁移 lvsim 开环/PEG/两脉冲圆化
- [x] Developer: 迁移 ballistics 火控反解与弹道整形
- [x] Developer: ICBM 跨大气层阶段链：主动段 → 中段 → 再入段
- [x] Developer: 6-DOF 控制律接口（默认禁止自动升维）
- [x] Tester: 制导律单元测试
- [x] Tester: ICBM 端到端测试（全程能量曲线检查）
- [x] Tester: 火控反解测试（目标点反解后仿真命中）
- [x] Tester: 更新 docs/known_failures.md 中 6-DOF 相关条目

## 阶段 3 重点关注
- CZ-2F 入轨偏差根因定位与修复（上面级 gamma0 装订 / 线性俯仰 / 关机条件 / 抛罩质量阶跃）。
- 迁移对拍脚本持续运行，确保阶段 2 已达标项不退化。

## 下一阶段触发条件
本阶段所有 checklist 完成后，进入阶段 4。

## 阻塞项
无

## 备注
阶段 3 目标：让火箭/ICBM 制导正确，火控反解可用，6-DOF 控制在稳定工况可启用。

Developer 阶段 3 完成：
- 修复 `PoweredECIDynamics._altitude` 为球近似高，与 lvsim 一致，使 CZ-2F SECO 高度/速度误差 <0.3%。
- 修复 `simulator.py` 事件日志绝对时刻重复累加 bug。
- 更新 `test_mvp_cz2f`：移除 xfail，使用球高与 lvsim golden 对比；倾角/偏心率 <1%，近/远地点因瞬态亚轨道 golden 允许 <5%。
- 完成开环/PEG/两脉冲圆化、火控/弹道整形、ICBM 阶段链、6-DOF 控制接口骨架迁移。

Tester 阶段 3 完成：
- 新增 `tests/guidance/test_open_loop.py`：覆盖 local_up/flight_path_angle/horizontal、三段式开环制导切换、上面级线性俯仰律。
- 新增 `tests/guidance/test_firecontrol.py`：M107 对 3 组目标经纬度反解射角/方位，正向仿真验证脱靶 < 50 m。
- 新增 `tests/test_icbm.py`：占位 ICBM 主动段→中段→再入段端到端，验证事件链、能量单调、出/再入、落点合理。
- 更新 `docs/known_failures.md`：追加 KF-002 ~ KF-004 6-DOF 控制限制说明。
- 全量测试：`pytest tests/` 94 项全部通过，覆盖率 59.42%，报告写入 `tests/stage3_report.txt`。
