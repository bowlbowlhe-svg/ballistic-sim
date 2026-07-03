"""可选依赖导入守卫。

将 weather/cfgrib、terrain/rasterio、srtm/requests、gpu/cupy、web/fastapi 等
重依赖声明为 optional extras，避免核心包在无网络/无 GPU 环境下硬失败。
"""

from __future__ import annotations

import importlib
from typing import Any


def try_import(module_name: str, extra_name: str) -> Any:
    """尝试导入可选模块，缺失时给出清晰的 pip extra 提示。

    Parameters
    ----------
    module_name: str
        需要导入的 Python 模块名（如 ``"rasterio"``）。
    extra_name: str
        对应的 ``pyproject.toml`` optional-dependency 名称（如 ``"terrain"``）。

    Returns
    -------
    module: module
        成功导入的模块对象。

    Raises
    ------
    ImportError
        模块缺失时抛出，提示用户执行 ``pip install ballistic_sim[<extra_name>]``。
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"功能需要可选依赖 '{module_name}'，"
            f"请执行 pip install ballistic_sim[{extra_name}] 安装对应 extra。"
        ) from exc


# -----------------------------------------------------------------------------
# 各能力域的守卫函数（示例，按需扩展）
# -----------------------------------------------------------------------------


def require_weather() -> Any:
    """高级气象数据读取（cfgrib / netCDF）。"""
    return try_import("cfgrib", "weather")


def require_netcdf4() -> Any:
    """NetCDF 气象数据读取（netCDF4）。"""
    return try_import("netCDF4", "weather")


def require_rasterio() -> Any:
    """地形栅格读取（rasterio）。"""
    return try_import("rasterio", "terrain")


def require_shapely() -> Any:
    """地形几何运算（shapely）。"""
    return try_import("shapely", "terrain")


def require_requests() -> Any:
    """SRTM / 在线数据下载（requests）。"""
    return try_import("requests", "srtm")


def require_srtm() -> Any:
    """SRTM 高程数据读取（srtm.py）。"""
    return try_import("srtm", "srtm")


def require_gpu() -> Any:
    """GPU 加速（cupy）。"""
    return try_import("cupy", "gpu")


def require_cupy() -> Any:
    """GPU 加速（cupy）别名。"""
    return try_import("cupy", "gpu")


def require_fastapi() -> Any:
    """Web API（fastapi / uvicorn）。"""
    return try_import("fastapi", "web")


def require_uvicorn() -> Any:
    """Web 服务（uvicorn）。"""
    return try_import("uvicorn", "web")
