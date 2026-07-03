"""两脉冲圆化入轨骨架单元测试。"""

from __future__ import annotations

import pytest

from ballistic_sim.guidance.circularization import (
    CircPlan,
    build_two_impulse_phases,
    make_circ_plan,
)

_TARGET = {"peri_km": 200.0, "apo_km": 350.0, "inc_deg": 42.0}


def test_make_circ_plan_defaults() -> None:
    """make_circ_plan 使用默认圆化级参数。"""
    plan = make_circ_plan(350.0, 200.0)
    assert isinstance(plan, CircPlan)
    assert plan.r_apo_circ_m == pytest.approx(6378137.0 + 350e3, rel=1e-6)
    assert plan.transfer_peri_km == 200.0
    assert plan.circ_stage["name"] == "圆化级"


def test_make_circ_plan_custom_stage() -> None:
    """make_circ_plan 接受自定义圆化级。"""
    custom = {"thrust_vac": 20e3, "m_prop": 1000.0}
    plan = make_circ_plan(350.0, 200.0, circ_stage=custom)
    assert plan.circ_stage["thrust_vac"] == 20e3
    assert plan.circ_stage["m_prop"] == 1000.0


def test_build_two_impulse_phases_extends_sequence() -> None:
    """build_two_impulse_phases 在原 phase 列表后追加滑行 + 圆化段。"""
    from ballistic_sim.dynamics.powered_eci import PoweredECIDynamics
    from ballistic_sim.phases.powered import PoweredPhase

    stage = {
        "name": " ascent",
        "thrust_vac": 1e5,
        "thrust_sl": 1e5,
        "isp_vac": 300.0,
        "m_prop": 1000.0,
        "m_dry": 500.0,
        "Aref": 1.0,
    }
    guid = {"lat_deg": 0.0, "lon_deg": 0.0, "azimuth_deg": 90.0}
    dyn = PoweredECIDynamics(stage=stage, guidance=guid)
    ascent = [
        PoweredPhase(
            name=" ascent",
            t_span=(0.0, 100.0),
            dynamics=dyn,
            guidance=guid,
            m_dry=500.0,
        )
    ]
    plan = make_circ_plan(350.0, 200.0)
    phases = build_two_impulse_phases(ascent, _TARGET, plan, guid)
    assert len(phases) == 3
    assert phases[0].name == " ascent"
    assert phases[1].name == "转移滑行"
    assert phases[2].name == "远地点圆化"


def test_build_two_impulse_phases_requires_dynamics() -> None:
    """ascent_phases 最后一个元素缺少 dynamics 时抛出 ValueError。"""

    class FakePhase:
        name = " ascent"

    plan = make_circ_plan(350.0, 200.0)
    with pytest.raises(ValueError):
        build_two_impulse_phases([FakePhase()], _TARGET, plan, {})
