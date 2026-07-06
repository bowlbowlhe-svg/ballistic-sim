"""公开 API 入口一致性测试。"""

from __future__ import annotations

import importlib
import types

import matplotlib.pyplot  # noqa: F401

import ballistic_sim


def test_public_api_imports() -> None:
    """``ballistic_sim.__all__`` 中所有符号均可从包根导入。"""
    for name in ballistic_sim.__all__:
        obj = getattr(ballistic_sim, name)
        assert obj is not None or name == "__version__"


def test_public_api_all_consistent() -> None:
    """``__all__`` 与 ``dir(ballistic_sim)`` 中的公开符号集合一致。"""
    public_dir = {name for name in dir(ballistic_sim) if not name.startswith("_")}
    # __version__ 以双下划线开头，会被 startswith("_") 过滤掉；
    # 因此比较时从 __all__ 中排除它。
    public_all = set(ballistic_sim.__all__) - {"__version__"}
    assert public_all == public_dir


def test_version_is_0_5_0() -> None:
    """包版本号为 0.5.0。"""
    assert ballistic_sim.__version__ == "0.5.0"


def test_public_api_star_import() -> None:
    """``from ballistic_sim import *`` 只导入 ``__all__`` 中的符号。"""
    module_code = "from ballistic_sim import *\n"
    module = types.ModuleType("ballistic_sim_star_import_test")
    exec(module_code, module.__dict__)  # noqa: S102
    imported = {name for name in dir(module) if not name.startswith("_") and name != "__builtins__"}
    # __version__ 以双下划线开头，from * 会导入它，但 startswith("_") 会过滤它。
    expected = set(ballistic_sim.__all__) - {"__version__"}
    assert imported == expected


def test_public_api_reimport() -> None:
    """重新导入包后公开符号仍然可用。"""
    reloaded = importlib.reload(ballistic_sim)
    assert reloaded.__version__ == "0.5.0"
    assert set(reloaded.__all__) == {
        "__version__",
        "SimConfig",
        "simulate",
        "build_phases",
        "validate_config",
        "load_config",
        "save_config",
        "apply_overrides",
        "PerturbationConfig",
        "SimResult",
    }
