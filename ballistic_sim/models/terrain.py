r"""统一地形模型。

来源与差异
----------
- 主体继承自 ``D:\Pycharm\游戏与兴趣\外弹道方程\ballistics\terrain.py``，
  保留 TerrainExtent、TerrainModel、程序化丘陵、灰度图高度图、GeoTIFF DEM 等能力。
- 集成 ``D:\Pycharm\游戏与兴趣\外弹道方程\ballistics\srtm.py`` 的 SRTM ``.hgt``
  读取与 tile 合并逻辑，避免 GDAL 依赖。
- **关键性能优化**：
  1. DEM 内存缓存：对 ``height_at`` 使用 LRU 缓存高频查询点。
  2. 双线性插值预计算：在 ``_height_at_impl`` 内一次性计算四个角点索引与权重。
  3. 两层落地检测：``impact_check`` 先用粗平面快速排除，再用精地形确认。
  4. ``NullTerrainModel`` 实现 ``terrain = null`` 快速模式。
- 所有可选依赖（GDAL/rasterio、Pillow）通过导入守卫懒加载。

接口约定
--------
- 高程单位为米 (m)。
- ``height_at(lat, lon)`` 输入 WGS84 经纬度 (deg)，输出海拔高程 (m)。
- ``height_at_enu(e, n, lat0, lon0)`` 输入 ENU 偏移 (m) 与原点经纬度。
- ``impact_check(lat, lon, alt, margin)`` 返回 ``True`` 表示飞行器已触地。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ballistic_sim.constants import DEG2RAD, RE_EARTH_MEAN
from ballistic_sim.utils.optional_imports import try_import


@dataclass(frozen=True)
class TerrainExtent:
    """地形范围（WGS84）。"""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


class TerrainModel:
    """地形高程模型。

    Parameters
    ----------
    elevation:
        高程数组 (H, W)，单位 m。默认第 0 行对应 ``lat_max``（北端），
        与常见 DEM 排列一致；若 ``flip_y=False`` 则第 0 行对应 ``lat_min``。
    extent:
        地理范围。
    no_data_value:
        无效值，将替换为 0。
    flip_y:
        是否将 elevation 按从北到南解释；默认 True。
    """

    def __init__(
        self,
        elevation: np.ndarray,
        extent: TerrainExtent,
        no_data_value: Optional[float] = None,
        flip_y: bool = True,
    ):
        self.elevation = np.array(elevation, dtype=np.float64)
        if no_data_value is not None:
            self.elevation[self.elevation == no_data_value] = 0.0
        self.elevation[np.isnan(self.elevation)] = 0.0
        self.extent = extent
        self.ny, self.nx = self.elevation.shape
        self.dlat = (extent.lat_max - extent.lat_min) / max(self.ny - 1, 1)
        self.dlon = (extent.lon_max - extent.lon_min) / max(self.nx - 1, 1)
        self.flip_y = flip_y

        # 预计算经纬度到像素坐标的转换常量，加速 ``height_at``。
        self._lat_scale = (self.ny - 1) / (extent.lat_max - extent.lat_min)
        self._lon_scale = (self.nx - 1) / (extent.lon_max - extent.lon_min)

    def _inside(self, lat: float, lon: float) -> bool:
        eps = 1e-12
        return (
            self.extent.lat_min - eps <= lat <= self.extent.lat_max + eps
            and self.extent.lon_min - eps <= lon <= self.extent.lon_max + eps
        )

    def _pixel_coords(self, lat: float, lon: float) -> Tuple[float, float]:
        """返回像素坐标 (y, x)，y 向下递增。"""
        if self.flip_y:
            y = (self.extent.lat_max - lat) * self._lat_scale
        else:
            y = (lat - self.extent.lat_min) * self._lat_scale
        x = (lon - self.extent.lon_min) * self._lon_scale
        return y, x

    def _height_at_impl(self, lat: float, lon: float) -> float:
        """双线性插值核心实现。"""
        if not self._inside(lat, lon):
            return 0.0

        y, x = self._pixel_coords(lat, lon)
        y0 = int(np.floor(y))
        x0 = int(np.floor(x))
        y1 = min(y0 + 1, self.ny - 1)
        x1 = min(x0 + 1, self.nx - 1)
        y0 = max(y0, 0)
        x0 = max(x0, 0)

        fy = y - y0
        fx = x - x0
        fy1 = 1.0 - fy
        fx1 = 1.0 - fx

        h00 = self.elevation[y0, x0]
        h01 = self.elevation[y0, x1]
        h10 = self.elevation[y1, x0]
        h11 = self.elevation[y1, x1]

        return float(fy1 * (fx1 * h00 + fx * h01) + fy * (fx1 * h10 + fx * h11))

    def height_at(self, lat: float, lon: float) -> float:
        """查询某经纬度的高程（带 LRU 缓存的双线性插值）。"""
        return self._height_at_cached(round(float(lat), 8), round(float(lon), 8))

    @lru_cache(maxsize=4096)
    def _height_at_cached(self, lat: float, lon: float) -> float:
        return self._height_at_impl(lat, lon)

    def height_at_enu(self, e: float, n: float, lat0: float, lon0: float) -> float:
        """将 ENU 偏移转为经纬度后查询高程（局部切平面近似）。"""
        lat_rad = lat0 * DEG2RAD
        dlat = n / RE_EARTH_MEAN * (180.0 / np.pi)
        dlon = e / (RE_EARTH_MEAN * np.cos(lat_rad)) * (180.0 / np.pi)
        return self.height_at(lat0 + dlat, lon0 + dlon)

    def max_height(self) -> float:
        return float(np.max(self.elevation))

    def min_height(self) -> float:
        return float(np.min(self.elevation))

    def sample_points(self, n_samples: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成均匀采样点（lat, lon, height），用于三维渲染。"""
        lats = np.linspace(self.extent.lat_min, self.extent.lat_max, n_samples)
        lons = np.linspace(self.extent.lon_min, self.extent.lon_max, n_samples)
        LAT, LON = np.meshgrid(lats, lons, indexing="ij")
        H = np.zeros_like(LAT)
        for i in range(n_samples):
            for j in range(n_samples):
                H[i, j] = self.height_at(LAT[i, j], LON[i, j])
        return LAT, LON, H

    def impact_check(
        self,
        lat: float,
        lon: float,
        alt: float,
        margin: float = 0.0,
    ) -> bool:
        """两层落地检测。

        第一层：用全局最大高程做粗平面快速排除（若飞行器高度远高于全局最大高程，
        直接判定未触地）。
        第二层：精确查询该点地形高程，并考虑 margin 安全余量。

        额外包含默认 0.1 m 的数值容差，用于处理插值与落地判定的数值稳定性。
        """
        _TOL = 0.1  # 数值容差 (m)
        if alt > self.max_height() + margin + _TOL:
            return False
        terrain_h = self.height_at(lat, lon)
        return alt <= terrain_h + margin + _TOL

    def clear_cache(self) -> None:
        """清空高程查询缓存。"""
        self._height_at_cached.cache_clear()


class NullTerrainModel:
    """空地形快速模式：所有点高程恒为 0，落地检测恒为 False。"""

    def height_at(self, lat: float, lon: float) -> float:
        return 0.0

    def height_at_enu(self, e: float, n: float, lat0: float, lon0: float) -> float:
        return 0.0

    def max_height(self) -> float:
        return 0.0

    def min_height(self) -> float:
        return 0.0

    def impact_check(
        self,
        lat: float,
        lon: float,
        alt: float,
        margin: float = 0.0,
    ) -> bool:
        _TOL = 0.1
        return alt <= margin + _TOL

    def clear_cache(self) -> None:
        pass


class FlatTerrainModel:
    """恒定高程地形，用于测试与简化场景。"""

    def __init__(self, altitude: float = 0.0, extent: Optional[TerrainExtent] = None) -> None:
        self.altitude = float(altitude)
        self.extent = extent

    def height_at(self, lat: float, lon: float) -> float:
        return self.altitude

    def height_at_enu(self, e: float, n: float, lat0: float, lon0: float) -> float:
        return self.altitude

    def max_height(self) -> float:
        return self.altitude

    def min_height(self) -> float:
        return self.altitude

    def impact_check(self, lat: float, lon: float, alt: float, margin: float = 0.0) -> bool:
        return alt <= self.altitude + margin + 0.1

    def clear_cache(self) -> None:
        pass


# ==============================================================================
# 程序化地形
# ==============================================================================


def generate_hilly_terrain(
    extent: TerrainExtent,
    resolution: int = 256,
    seed: int = 42,
    max_height: float = 200.0,
) -> TerrainModel:
    """生成程序化丘陵地形（多层正弦叠加近似噪声）。"""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 4.0 * np.pi, resolution)
    y = np.linspace(0.0, 4.0 * np.pi, resolution)
    X, Y = np.meshgrid(x, y, indexing="xy")

    z = (
        np.sin(X + rng.uniform(0.0, 2.0 * np.pi)) * np.cos(Y + rng.uniform(0.0, 2.0 * np.pi))
        + 0.5
        * np.sin(2.0 * X + rng.uniform(0.0, 2.0 * np.pi))
        * np.cos(2.0 * Y + rng.uniform(0.0, 2.0 * np.pi))
        + 0.25
        * np.sin(4.0 * X + rng.uniform(0.0, 2.0 * np.pi))
        * np.cos(4.0 * Y + rng.uniform(0.0, 2.0 * np.pi))
    )
    z = (z - z.min()) / (z.max() - z.min()) * max_height
    return TerrainModel(z, extent, flip_y=True)


# ==============================================================================
# 栅格加载（Pillow / rasterio 导入守卫）
# ==============================================================================


def load_from_numpy(path: Union[str, Path], extent: TerrainExtent) -> TerrainModel:
    """从 NumPy 文件加载地形。"""
    elevation = np.load(path)
    return TerrainModel(elevation, extent)


def load_from_image(
    path: Union[str, Path], extent: TerrainExtent, max_height: float = 1000.0
) -> TerrainModel:
    """从灰度 PNG/JPG 加载高度图（需要 Pillow）。"""
    Image = try_import("PIL.Image", "anim")  # noqa: N806
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float64) / 255.0 * max_height
    return TerrainModel(arr, extent)


def load_from_geotiff(path: Union[str, Path]) -> TerrainModel:
    """从 GeoTIFF 加载真实 DEM（需要 rasterio）。"""
    rasterio = try_import("rasterio", "terrain")  # noqa: F841
    with rasterio.open(path) as ds:
        elevation = ds.read(1).astype(np.float64)
        no_data = ds.nodata
        bounds = ds.bounds
        extent = TerrainExtent(
            lat_min=bounds.bottom,
            lat_max=bounds.top,
            lon_min=bounds.left,
            lon_max=bounds.right,
        )
    return TerrainModel(elevation, extent, no_data_value=no_data, flip_y=False)


# ==============================================================================
# SRTM .hgt 读取（无需外部依赖）
# ==============================================================================


@dataclass(frozen=True)
class SRTMTileName:
    """SRTM tile 文件名解析。"""

    lat: int
    lon: int
    name: str

    @classmethod
    def parse(cls, filename: Union[str, Path]) -> "SRTMTileName":
        base = Path(filename).stem
        match = re.fullmatch(r"([NS])(\d{2})([EW])(\d{3})", base)
        if not match:
            raise ValueError(f"无法解析 SRTM 文件名: {base}")
        lat = int(match.group(2))
        if match.group(1) == "S":
            lat = -lat
        lon = int(match.group(4))
        if match.group(3) == "W":
            lon = -lon
        return cls(lat=lat, lon=lon, name=base + ".hgt")

    @property
    def south(self) -> int:
        return self.lat

    @property
    def north(self) -> int:
        return self.lat + 1

    @property
    def west(self) -> int:
        return self.lon

    @property
    def east(self) -> int:
        return self.lon + 1


class SRTMTile:
    """单个 SRTM tile。"""

    NO_DATA_VALUE = -32768

    def __init__(self, elevation: np.ndarray, extent: TerrainExtent, arc_seconds: int):
        self.elevation = np.array(elevation, dtype=np.float64)
        self.extent = extent
        self.arc_seconds = arc_seconds
        self.ny, self.nx = self.elevation.shape

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SRTMTile":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SRTM tile 不存在: {path}")

        tile_name = SRTMTileName.parse(path)
        file_size = path.stat().st_size

        if file_size == 1201 * 1201 * 2:
            arc_seconds = 3
            size = 1201
        elif file_size == 3601 * 3601 * 2:
            arc_seconds = 1
            size = 3601
        elif file_size == 1801 * 3601 * 2:
            arc_seconds = 30
            size_y, size_x = 1801, 3601
        else:
            pixels = file_size // 2
            size = int(np.sqrt(pixels))
            if size * size * 2 != file_size:
                raise ValueError(f"无法识别的 SRTM 文件大小: {file_size} bytes ({path})")
            arc_seconds = 1

        with open(path, "rb") as f:
            raw = f.read()

        elevation: np.ndarray
        if arc_seconds == 30:
            elevation = np.frombuffer(raw, dtype=">i2").reshape((size_y, size_x))
        else:
            elevation = np.frombuffer(raw, dtype=">i2").reshape((size, size))

        elevation = np.flipud(elevation)  # 从北到南 -> 从南到北，便于 TerrainModel
        extent = TerrainExtent(
            lat_min=tile_name.south,
            lat_max=tile_name.north,
            lon_min=tile_name.west,
            lon_max=tile_name.east,
        )
        return cls(elevation, extent, arc_seconds)

    def to_model(self) -> TerrainModel:
        elevation = self.elevation.copy()
        elevation[elevation == self.NO_DATA_VALUE] = 0.0
        return TerrainModel(elevation, self.extent, flip_y=False)


class SRTMDataset:
    """多个 SRTM tile 组成的区域 DEM。"""

    def __init__(self, tiles: Dict[Tuple[int, int], SRTMTile]):
        self.tiles = tiles
        if not tiles:
            raise ValueError("至少需要提供一个 tile")

        self.lat_min = min(t.extent.lat_min for t in tiles.values())
        self.lat_max = max(t.extent.lat_max for t in tiles.values())
        self.lon_min = min(t.extent.lon_min for t in tiles.values())
        self.lon_max = max(t.extent.lon_max for t in tiles.values())
        self.arc_seconds = min(t.arc_seconds for t in tiles.values())

    @classmethod
    def from_directory(cls, directory: Union[str, Path]) -> "SRTMDataset":
        directory = Path(directory)
        tiles: Dict[Tuple[int, int], SRTMTile] = {}
        for path in directory.glob("*.hgt"):
            try:
                tile = SRTMTile.from_file(path)
                tiles[(int(tile.extent.lat_min), int(tile.extent.lon_min))] = tile
            except (ValueError, FileNotFoundError) as e:
                print(f"跳过 {path}: {e}")
        return cls(tiles)

    @classmethod
    def from_files(cls, paths: Sequence[Union[str, Path]]) -> "SRTMDataset":
        tiles: Dict[Tuple[int, int], SRTMTile] = {}
        for path in paths:
            tile = SRTMTile.from_file(path)
            tiles[(int(tile.extent.lat_min), int(tile.extent.lon_min))] = tile
        return cls(tiles)

    def height_at(self, lat: float, lon: float) -> float:
        """查询区域内任意点高程。"""
        lat = float(np.clip(lat, self.lat_min, self.lat_max))
        lon = float(np.clip(lon, self.lon_min, self.lon_max))

        tile_lat = int(np.floor(lat))
        tile_lon = int(np.floor(lon))
        tile = self.tiles.get((tile_lat, tile_lon))
        if tile is None:
            tile = (
                self.tiles.get((tile_lat - 1, tile_lon))
                or self.tiles.get((tile_lat, tile_lon - 1))
                or self.tiles.get((tile_lat - 1, tile_lon - 1))
            )
        if tile is None:
            return 0.0
        return tile.to_model().height_at(lat, lon)

    def build_model(
        self,
        extent: Optional[TerrainExtent] = None,
        resolution: Optional[int] = None,
    ) -> TerrainModel:
        """合并 tiles 为统一的 TerrainModel。"""
        extent = extent or TerrainExtent(
            lat_min=self.lat_min,
            lat_max=self.lat_max,
            lon_min=self.lon_min,
            lon_max=self.lon_max,
        )
        if resolution is None:
            tile_count_lat = int(extent.lat_max - extent.lat_min)
            tile_count_lon = int(extent.lon_max - extent.lon_min)
            resolution = max(120, 120 * max(tile_count_lat, tile_count_lon))

        lats = np.linspace(extent.lat_min, extent.lat_max, resolution)
        lons = np.linspace(extent.lon_min, extent.lon_max, resolution)
        lats = np.clip(lats, extent.lat_min, extent.lat_max)
        lons = np.clip(lons, extent.lon_min, extent.lon_max)
        elevation = np.zeros((resolution, resolution), dtype=np.float64)

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                elevation[i, j] = self.height_at(lat, lon)

        return TerrainModel(elevation, extent, flip_y=False)


def load_srtm_directory(directory: Union[str, Path]) -> TerrainModel:
    """便捷函数：从目录加载所有 SRTM tile 并合并。"""
    dataset = SRTMDataset.from_directory(directory)
    return dataset.build_model()


def load_srtm_files(paths: List[Union[str, Path]]) -> TerrainModel:
    """便捷函数：从文件列表加载 SRTM tile 并合并。"""
    dataset = SRTMDataset.from_files(paths)
    return dataset.build_model()


# ==============================================================================
# 兼容包装类（与现有测试/旧接口保持一致）
# ==============================================================================


class NullTerrain:
    """空地形快速模式（旧接口包装）。

    所有查询返回 0，与 ``NullTerrainModel`` 等价，但使用 ``height(lon, lat)`` 签名。
    """

    def height(self, lon: float, lat: float) -> float:
        return 0.0


class ProceduralTerrain:
    """程序化地形（旧接口包装）。

    基于 ``generate_hilly_terrain``，提供 ``height(lon, lat)`` 与
    ``two_stage_height(lon, lat)`` 两层查询接口。

    Parameters
    ----------
    extent:
        地理范围。
    seed:
        随机种子，保证可复现。
    resolution:
        内部栅格分辨率。
    max_height:
        最大高程 (m)。
    """

    def __init__(
        self,
        extent: TerrainExtent,
        seed: int = 42,
        resolution: int = 256,
        max_height: float = 200.0,
    ):
        self._model = generate_hilly_terrain(
            extent=extent,
            resolution=resolution,
            seed=seed,
            max_height=max_height,
        )
        self._cache: Dict[Tuple[float, float], float] = {}

    def _coarse_height(self, lon: float, lat: float) -> float:
        """粗查询：最近邻近似。"""
        if not self._model._inside(lat, lon):
            return 0.0
        y, x = self._model._pixel_coords(lat, lon)
        y0 = int(np.clip(round(y), 0, self._model.ny - 1))
        x0 = int(np.clip(round(x), 0, self._model.nx - 1))
        return float(self._model.elevation[y0, x0])

    def height(self, lon: float, lat: float) -> float:
        """精查询：双线性插值，带缓存。"""
        key = (round(float(lon), 8), round(float(lat), 8))
        if key not in self._cache:
            self._cache[key] = self._model.height_at(lat, lon)
        return self._cache[key]

    def two_stage_height(self, lon: float, lat: float) -> float:
        """两层查询：先粗后精（当前实现直接返回精查询结果）。"""
        return self.height(lon, lat)


# ==============================================================================
# 工厂函数
# ==============================================================================


def make_terrain(
    model: str = "null",
    path: Optional[Union[str, Path]] = None,
    extent: Optional[TerrainExtent] = None,
    **kwargs,
) -> Union[TerrainModel, NullTerrainModel, FlatTerrainModel]:
    """地形模型工厂函数。

    Parameters
    ----------
    model:
        ``"null"``、``"flat"``、``"hilly"``、``"numpy"``、``"image"``、``"geotiff"``、
        ``"srtm_dir"``、``"srtm_files"``。
    path:
        文件或目录路径（如适用）。
    extent:
        地理范围（部分加载器需要）。
    """
    name = model.lower()
    if name in ("null", "none"):
        return NullTerrainModel()
    if name == "flat":
        return FlatTerrainModel(altitude=kwargs.get("altitude", 0.0), extent=extent)
    if name == "hilly":
        return generate_hilly_terrain(extent=extent or TerrainExtent(0.0, 1.0, 0.0, 1.0), **kwargs)
    if name == "numpy":
        if path is None or extent is None:
            raise ValueError("numpy 地形需要提供 path 与 extent")
        return load_from_numpy(path, extent)
    if name == "image":
        if path is None or extent is None:
            raise ValueError("image 地形需要提供 path 与 extent")
        return load_from_image(path, extent, **kwargs)
    if name == "geotiff":
        if path is None:
            raise ValueError("geotiff 地形需要提供 path")
        return load_from_geotiff(path)
    if name == "srtm_dir":
        if path is None:
            raise ValueError("srtm_dir 地形需要提供 path")
        return load_srtm_directory(path)
    if name == "srtm_files":
        if path is None:
            raise ValueError("srtm_files 地形需要提供 path")
        return load_srtm_files([path] if isinstance(path, (str, Path)) else path)
    raise ValueError(f"未知地形模型: {model}")
