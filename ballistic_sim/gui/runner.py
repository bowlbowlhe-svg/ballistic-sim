"""在后台线程运行仿真并通过 queue 返回结果。"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Optional

from ballistic_sim.config import SimConfig
from ballistic_sim.simulator import SimResult, simulate
from ballistic_sim.viz import attach_launch_lla


class SimulationRunner:
    """将 ``simulate`` 封装到后台线程，通过 ``queue.Queue`` 异步返回结果。"""

    def __init__(self) -> None:
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None

    @property
    def result_queue(self) -> queue.Queue[Any]:
        return self._queue

    def run(
        self,
        cfg: SimConfig,
        on_done: Optional[Callable[[SimResult], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """启动后台线程执行仿真。"""
        if self._thread is not None and self._thread.is_alive():
            return

        def _target() -> None:
            try:
                result = simulate(cfg)
                attach_launch_lla(result, cfg.launch.lat_deg, cfg.launch.lon_deg, cfg.launch.alt_m)
                self._queue.put(("ok", result))
                if on_done is not None:
                    on_done(result)
            except Exception as exc:  # noqa: BLE001
                self._queue.put(("error", exc))
                if on_error is not None:
                    on_error(exc)

        self._thread = threading.Thread(target=_target, daemon=True)
        self._thread.start()

    def is_alive(self) -> bool:
        """后台线程是否仍在运行。"""
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: Optional[float] = None) -> None:
        """等待后台线程结束。"""
        if self._thread is not None:
            self._thread.join(timeout=timeout)
