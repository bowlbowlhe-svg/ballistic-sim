"""发布前一致性检查脚本。

检查项：
1. ``ballistic_sim.__version__`` 与目标版本一致。
2. CHANGELOG.md 包含对应版本章节。
3. git 存在对应 tag。
4. 工作树干净（除输出文件外）。
5. pytest 全部通过。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from ballistic_sim import __version__

TARGET_VERSION = "0.4.0"


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def _ok(message: str) -> None:
    print(f"OK: {message}")


def check_version() -> None:
    """检查包版本号与目标版本一致。"""
    if __version__ != TARGET_VERSION:
        _fail(f"__version__={__version__!r} != {TARGET_VERSION!r}")
    _ok(f"__version__ == {TARGET_VERSION}")


def check_changelog() -> None:
    """检查 CHANGELOG 包含目标版本章节。"""
    changelog_path = Path("CHANGELOG.md")
    if not changelog_path.is_file():
        _fail("CHANGELOG.md not found")
    text = changelog_path.read_text(encoding="utf-8")
    pattern = re.compile(rf"^## \[{re.escape(TARGET_VERSION)}\]", re.MULTILINE)
    if not pattern.search(text):
        _fail(f"CHANGELOG.md missing [{TARGET_VERSION}] section")
    _ok(f"CHANGELOG.md contains [{TARGET_VERSION}]")


def check_git_tag() -> None:
    """检查本地存在目标 git tag。"""
    result = subprocess.run(
        ["git", "tag", "-l", f"v{TARGET_VERSION}"],
        capture_output=True,
        text=True,
        check=False,
    )
    tag = f"v{TARGET_VERSION}"
    if tag not in result.stdout:
        _fail(f"git tag {tag} not found")
    _ok(f"git tag {tag} exists")


def check_working_tree_clean() -> None:
    """检查工作树干净，允许忽略常见输出文件。"""
    allowed = {
        "benchmark_results.json",
        ".coverage",
        "*.png",
        "*.html",
        "out/*",
        "htmlcov*",
    }
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    dirty_lines = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        # git status --porcelain 前两字符为状态，后面为路径
        path = line[3:].strip()
        # 检查是否匹配任一允许模式
        matched = False
        for pattern in allowed:
            if _match_glob(path, pattern):
                matched = True
                break
        if not matched:
            dirty_lines.append(line)
    if dirty_lines:
        _fail("working tree is not clean:\n" + "\n".join(dirty_lines))
    _ok("working tree is clean (ignoring output files)")


def _match_glob(path: str, pattern: str) -> bool:
    """极简 glob 匹配，支持 * 通配。"""
    import fnmatch

    return fnmatch.fnmatch(path, pattern)


def run_pytest() -> None:
    """运行 pytest，失败则退出非 0。"""
    env = {**os.environ, "MPLBACKEND": "Agg"}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        env=env,
        check=False,
    )
    if result.returncode != 0:
        _fail("pytest failed")
    _ok("pytest passed")


def main() -> int:
    """运行所有发布检查。"""
    print(f"Release check for v{TARGET_VERSION}")
    print("-" * 40)
    check_version()
    check_changelog()
    check_git_tag()
    check_working_tree_clean()
    run_pytest()
    print("-" * 40)
    print(f"Release ready: v{TARGET_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
