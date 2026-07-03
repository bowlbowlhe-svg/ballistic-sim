"""补充大气模型测试（自定义廓线与解析）。"""

from __future__ import annotations

import numpy as np
import pytest

from ballistic_sim.models.atmosphere import StandardAtmosphere, make_atmosphere, parse_profile


def test_custom_profile_used_below_max_height() -> None:
    """自定义廓线在最大高度内被使用。"""
    profile = np.array(
        [
            [0.0, 300.0, 101325.0, 0.0],
            [2000.0, 290.0, 80000.0, 0.0],
        ]
    )
    atm = StandardAtmosphere(custom_profile=profile)
    state = atm(1000.0)
    # 插值温度应在 290–300 K 之间
    assert 290.0 < state.T < 300.0
    assert 80000.0 < state.p < 101325.0


def test_custom_profile_not_used_above_max_height() -> None:
    """超过自定义廓线最大高度后回退 ISA。"""
    profile = np.array(
        [
            [0.0, 288.15, 101325.0, 0.0],
            [1000.0, 281.65, 89874.0, 0.0],
        ]
    )
    atm = StandardAtmosphere(custom_profile=profile)
    state = atm(5000.0)
    # 5 km 处 ISA 压强约 54048 Pa
    assert state.p == pytest.approx(54048.0, rel=0.05)


def test_make_atmosphere_invalid() -> None:
    """未知大气模型抛出 ValueError。"""
    with pytest.raises(ValueError):
        make_atmosphere("mars")


def test_parse_profile_hpa_to_pa() -> None:
    """parse_profile 把 hPa 输入转换为 Pa。"""
    text = """
    # h(m) T(K) p(hPa) RH(0-1)
    0     288  1013   0.5
    1000  282  900    0.4
    """
    profile = parse_profile(text)
    assert profile[0, 2] == pytest.approx(101300.0, rel=1e-3)
    assert profile.shape == (2, 4)


def test_parse_profile_already_pa() -> None:
    """parse_profile 识别已是 Pa 的输入。"""
    text = """
    0     288  101325   0.5
    1000  282  90000    0.4
    """
    profile = parse_profile(text)
    assert profile[0, 2] == pytest.approx(101325.0, rel=1e-6)


def test_parse_profile_too_few_rows() -> None:
    """parse_profile 数据不足时抛出 ValueError。"""
    with pytest.raises(ValueError):
        parse_profile("0 288 1013 0.5")


def test_sound_speed_decreases_with_altitude() -> None:
    """声速随高度增加而降低（ISA 对流层）。"""
    atm = StandardAtmosphere()
    c0 = atm.sound_speed(0.0)
    c10k = atm.sound_speed(10000.0)
    assert c10k < c0
