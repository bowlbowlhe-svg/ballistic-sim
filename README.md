# Ballistic Sim

六自由度（6-DOF）弹道飞行仿真程序，采用模块化阶段（phases）设计，支持动力学、制导律、可视化与 Web API 扩展。

## 项目简介

本项目用于对火箭/导弹等飞行器的弹道进行六自由度数值仿真，计划逐步迁移并重构原有
`D:\Pycharm\游戏与兴趣\外弹道方程` 与 `D:\Pycharm\游戏与兴趣\火箭仿真程序` 中的可用模块。

主要模块：

- `ballistic_sim/models/` —— 飞行器、大气、地球等模型定义
- `ballistic_sim/dynamics/` —— 6-DOF 运动学与动力学方程
- `ballistic_sim/phases/` —— 起飞、助推、滑翔、再入等飞行阶段
- `ballistic_sim/guidance/` —— 制导律与导航算法
- `ballistic_sim/viz/` —— 可视化与动画生成
- `ballistic_sim/api/` —— Web 服务与对外接口

详细使用说明见 [docs/user_guide.md](docs/user_guide.md)。

## 安装说明

```bash
# 克隆仓库
git clone https://github.com/bowlbowlhe-svg/ballistic-sim.git
cd ballistic-sim

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows

# 安装核心依赖
pip install -e .

# 安装全部可选依赖
pip install -e ".[all]"
```

## 快速开始

```python
from ballistic_sim.presets import m107_config
from ballistic_sim.simulator import simulate

cfg = m107_config()
result = simulate(cfg)
print("飞行时间:", result.post["t_end_s"], "s")
print("落点 ENU:", result.post["r_end_m"])
```

更多示例与说明见 [docs/user_guide.md](docs/user_guide.md)。

## 教学与占位数据声明

本仓库包含的预设文件、示例数据与占位参数仅用于教学、本地调试和回归测试，不代表任何真实飞行器性能。使用真实任务数据前，请替换 `presets/` 与 `data/` 中的相应文件。

## 许可证

MIT License —— 详见 [LICENSE](LICENSE)。
