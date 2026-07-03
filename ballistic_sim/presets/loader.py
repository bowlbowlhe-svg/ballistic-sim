"""YAML 预设加载器。

从 ``<repo>/presets/*.yaml`` 加载弹丸/导弹/火箭数据, 供 Python 兼容层函数使用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PRESETS_DIR = _PROJECT_ROOT / "presets"


def _yaml_path(name: str) -> Path:
    return _PRESETS_DIR / f"{name}.yaml"


def _load_yaml(name: str) -> Dict[str, Any]:
    path = _yaml_path(name)
    if not path.exists():
        raise FileNotFoundError(f"预设文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_projectiles() -> Dict[str, Dict[str, Any]]:
    """加载弹丸预设字典。"""
    data = _load_yaml("projectiles")
    return data.get("presets", {})


def load_missiles() -> Dict[str, Dict[str, Any]]:
    """加载导弹预设字典 (含 launch_sites)。"""
    data = _load_yaml("missiles")
    return data


def load_rockets() -> Dict[str, Dict[str, Any]]:
    """加载火箭预设字典 (含 launch_sites)。"""
    data = _load_yaml("rockets")
    return data


def get_projectile(name: str) -> Dict[str, Any]:
    """获取指定弹丸预设。"""
    presets = load_projectiles()
    if name not in presets:
        raise KeyError(f"未知弹丸预设: {name}。可用: {list(presets.keys())}")
    return presets[name]


def get_missile(name: str) -> Dict[str, Any]:
    """获取指定导弹预设 (含发射场展开)。"""
    data = load_missiles()
    missiles = data.get("missiles", {})
    if name not in missiles:
        raise KeyError(f"未知导弹预设: {name}。可用: {list(missiles.keys())}")
    m = dict(missiles[name])
    site_name = m.get("launch_site")
    sites = data.get("launch_sites", {})
    if site_name not in sites:
        raise KeyError(f"导弹 {name} 引用未知发射场: {site_name}")
    m["_site"] = dict(sites[site_name])
    return m


def get_rocket(name: str) -> Dict[str, Any]:
    """获取指定火箭预设。"""
    data = load_rockets()
    rockets = data.get("rockets", {})
    if name not in rockets:
        raise KeyError(f"未知火箭预设: {name}。可用: {list(rockets.keys())}")
    return rockets[name]


def list_projectiles() -> List[str]:
    """列出弹丸预设名称。"""
    return list(load_projectiles().keys())


def list_missiles() -> List[str]:
    """列出导弹预设名称。"""
    return list(load_missiles().get("missiles", {}).keys())


def list_rockets() -> List[str]:
    """列出火箭预设名称。"""
    return list(load_rockets().get("rockets", {}).keys())


def make_aero_tables(preset: Dict[str, Any]) -> Dict[str, Optional[np.ndarray]]:
    """由预设 ``aero`` 字段构造 MPM 可用的系数表。"""
    aero = preset.get("aero", {})
    ma = np.array(aero.get("ma", []))
    out: Dict[str, Optional[np.ndarray]] = {
        "CMa_table": None,
        "CLa_table": None,
        "Clp_table": None,
    }
    if ma.size == 0:
        return out
    for key, out_key in [("CMa", "CMa_table"), ("CLa", "CLa_table"), ("Clp", "Clp_table")]:
        vals = aero.get(key)
        if vals is not None and len(vals) == len(ma):
            out[out_key] = np.column_stack([ma, np.array(vals)])
    return out
