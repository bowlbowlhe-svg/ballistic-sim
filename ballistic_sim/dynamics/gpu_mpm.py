"""GPU 批量 MPM 求解器（基于 CuPy）.

如果没有安装 CuPy 或没有可用 GPU，导入时会抛出 ImportError，
调用方可以回退到 CPU 批量求解器。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ballistic_sim.dynamics.batch_mpm import BatchMPMModel, BatchMPMResult
from ballistic_sim.dynamics.mpm import MPMOptions
from ballistic_sim.utils.optional_imports import require_cupy


def gpu_available() -> bool:
    """检查 CuPy 与 GPU 是否可用."""
    try:
        cp = require_cupy()
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pragma: no cover
        return False


class GPUBatchMPMModel(BatchMPMModel):
    """CuPy 后端批量 MPM.

    继承 BatchMPMModel，仅将数组后端切换为 cupy。
    """

    def __init__(
        self,
        mass_kg: np.ndarray,
        diameter_m: np.ndarray,
        form_factor: np.ndarray,
        v0: np.ndarray,
        theta_deg: np.ndarray,
        az_deg: np.ndarray,
        delta_t: np.ndarray,
        density_factor: np.ndarray,
        wind_e: np.ndarray,
        wind_n: np.ndarray,
        wind_u: Optional[np.ndarray] = None,
        lat_deg: float = 39.9,
        h0: float = 0.0,
        azimuth_deg: float = 0.0,
        drag_table: Optional[np.ndarray] = None,
        options: Optional[MPMOptions] = None,
        t_max: float = 400.0,
        dt: float = 0.05,
    ):
        cp = require_cupy()
        super().__init__(
            mass_kg=mass_kg,
            diameter_m=diameter_m,
            form_factor=form_factor,
            v0=v0,
            theta_deg=theta_deg,
            az_deg=az_deg,
            delta_t=delta_t,
            density_factor=density_factor,
            wind_e=wind_e,
            wind_n=wind_n,
            wind_u=wind_u,
            lat_deg=lat_deg,
            h0=h0,
            azimuth_deg=azimuth_deg,
            drag_table=drag_table,
            options=options,
            xp=cp,
            t_max=t_max,
            dt=dt,
        )

    def simulate(self) -> BatchMPMResult:
        """在 GPU 上执行并返回 NumPy 结果."""
        cp = require_cupy()
        with cp.cuda.Device(0):
            return super().simulate()


def try_gpu_model(*args: Any, **kwargs: Any) -> BatchMPMModel:
    """尝试创建 GPU 批量模型，失败时抛出 ImportError。"""
    return GPUBatchMPMModel(*args, **kwargs)
