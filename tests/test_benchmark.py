"""Benchmark 脚本结构测试。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from ballistic_sim import __version__


def _load_benchmark_module():
    """以 importlib 加载 scripts/benchmark.py（目录非包）。"""
    repo_root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "benchmark_script", repo_root / "scripts" / "benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture
def benchmark_module():
    return _load_benchmark_module()


def test_benchmark_module_has_main(benchmark_module) -> None:
    """benchmark 脚本暴露 main 入口。"""
    assert callable(benchmark_module.main)


def test_benchmark_single_shot_structure(benchmark_module) -> None:
    """单发 benchmark 返回预期字段。"""
    result = benchmark_module._benchmark_single_shot()
    assert result["name"] == "single_projectile"
    assert result["n_samples"] == 1
    assert result["elapsed_s"] >= 0.0
    assert "stop_reason" in result
    assert "n_points" in result


def test_benchmark_json_report_exists_after_main(tmp_path, monkeypatch, benchmark_module) -> None:
    """运行 main 后当前目录应生成 benchmark_results.json。"""
    monkeypatch.chdir(tmp_path)
    benchmark_module.main()
    out = tmp_path / "benchmark_results.json"
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["version"] == __version__
    assert "benchmarks" in data
    assert "total_elapsed_s" in data
    assert len(data["benchmarks"]) == 3
