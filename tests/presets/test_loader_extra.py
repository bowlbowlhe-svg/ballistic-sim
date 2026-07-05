"""预设加载器边界与错误分支测试。"""

from __future__ import annotations

import pytest

from ballistic_sim.presets.loader import (
    get_missile,
    get_projectile,
    get_rocket,
    list_missiles,
    list_projectiles,
    list_rockets,
    load_missiles,
    load_projectiles,
    load_rockets,
)


def test_list_projectiles_returns_known_names() -> None:
    """list_projectiles 应返回非空列表且包含 M107。"""
    names = list_projectiles()
    assert isinstance(names, list)
    assert "M107" in names


def test_list_missiles_returns_known_names() -> None:
    """list_missiles 应返回非空列表且包含 SRBM_600。"""
    names = list_missiles()
    assert isinstance(names, list)
    assert "SRBM_600" in names
    assert "MRBM_1500" in names
    assert "ICBM_8000" in names


def test_list_rockets_returns_known_names() -> None:
    """list_rockets 应返回非空列表且包含 CZ2F。"""
    names = list_rockets()
    assert isinstance(names, list)
    assert "CZ2F" in names


def test_get_unknown_projectile_raises() -> None:
    """未知弹丸预设应抛出 KeyError。"""
    with pytest.raises(KeyError):
        get_projectile("UNKNOWN_BULLET")


def test_get_unknown_missile_raises() -> None:
    """未知导弹预设应抛出 KeyError。"""
    with pytest.raises(KeyError):
        get_missile("DF-5")


def test_get_unknown_rocket_raises() -> None:
    """未知火箭预设应抛出 KeyError。"""
    with pytest.raises(KeyError):
        get_rocket("SATURN_V")


def test_load_missiles_structure() -> None:
    """load_missiles 返回的字典应包含 missiles 与 launch_sites。"""
    data = load_missiles()
    assert "missiles" in data
    assert "launch_sites" in data


def test_load_projectiles_structure() -> None:
    """load_projectiles 返回的字典应包含预设。"""
    data = load_projectiles()
    assert "M107" in data


def test_load_rockets_structure() -> None:
    """load_rockets 返回的字典应包含 rockets。"""
    data = load_rockets()
    assert "rockets" in data


def test_get_missile_returns_site() -> None:
    """get_missile 返回的字典应包含展开后的 _site。"""
    m = get_missile("SRBM_600")
    assert "_site" in m
    assert "lat" in m["_site"]


def test_get_projectile_has_aero() -> None:
    """弹丸预设应包含气动系数表。"""
    p = get_projectile("M107")
    assert "aero" in p
    assert "ma" in p["aero"]


def test_get_rocket_has_core1() -> None:
    """火箭预设应包含 core1 字段。"""
    r = get_rocket("CZ2F")
    assert "core1" in r
    assert "thrust_vac" in r["core1"]
