"""制导与火控子包 (阶段 3)。

包含开环/闭环制导律、弹道整形、火控反解与 6-DOF 控制接口。
"""

from ballistic_sim.guidance.aag import (
    AAGState,
    aag_acceleration,
    aag_replan,
    aag_thrust_dir,
    make_aag_state,
)
from ballistic_sim.guidance.energy_management import EnergyManagementGuidance
from ballistic_sim.guidance.open_loop import (
    flight_path_angle,
    local_horizontal,
    local_up,
    thrust_dir_eci,
    thrust_dir_upperstage,
)
from ballistic_sim.guidance.proportional_navigation import (
    ProNavGuidance,
    make_static_target_provider,
    pro_nav_acceleration,
)
from ballistic_sim.guidance.reentry_guidance import ReentryGuidance

__all__ = [
    # 开环
    "thrust_dir_eci",
    "thrust_dir_upperstage",
    "local_up",
    "local_horizontal",
    "flight_path_angle",
    # 比例导引
    "ProNavGuidance",
    "pro_nav_acceleration",
    "make_static_target_provider",
    # AAG
    "AAGState",
    "make_aag_state",
    "aag_replan",
    "aag_thrust_dir",
    "aag_acceleration",
    # 再入制导
    "ReentryGuidance",
    # 能量管理
    "EnergyManagementGuidance",
]
