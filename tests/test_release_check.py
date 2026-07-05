"""发布前检查脚本单元测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from ballistic_sim import __version__


def _load_release_check_module():
    """以 importlib 加载 scripts/release_check.py。"""
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "release_check_script", repo_root / "scripts" / "release_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture
def rc_module():
    return _load_release_check_module()


def test_target_version_matches_package(rc_module) -> None:
    """脚本目标版本与包版本一致。"""
    assert rc_module.TARGET_VERSION == __version__


def test_check_version_passes(rc_module) -> None:
    """当前包版本应通过版本检查。"""
    # 若失败会调用 sys.exit(1)，pytest 会捕获 SystemExit
    rc_module.check_version()


def test_check_changelog_passes(rc_module) -> None:
    """CHANGELOG.md 应包含目标版本章节。"""
    rc_module.check_changelog()


def test_match_glob(rc_module) -> None:
    """通配匹配辅助函数工作正常。"""
    assert rc_module._match_glob("out/foo.png", "out/*")
    assert rc_module._match_glob("benchmark_results.json", "benchmark_results.json")
    assert not rc_module._match_glob("src/main.py", "out/*")
