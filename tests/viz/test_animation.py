"""Animation generation smoke test for stage 4."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import matplotlib
import pytest

matplotlib.use("Agg", force=True)

from ballistic_sim.presets import m107_config  # noqa: E402
from ballistic_sim.simulator import SimResult, simulate  # noqa: E402


def _atmospheric_result() -> SimResult:
    """Return a minimal atmospheric SimResult (M107 projectile)."""
    return simulate(m107_config())


def test_animate_trajectory_gif(tmp_path: Path) -> None:
    """``animate_trajectory`` should write a non-empty GIF file."""
    mod = pytest.importorskip(
        "ballistic_sim.viz.animation",
        reason="animation module not implemented",
    )
    pytest.importorskip("PIL", reason="Pillow required for GIF output")

    animate_fn = getattr(mod, "animate_trajectory", None)
    if animate_fn is None or not callable(animate_fn):
        pytest.skip("animate_trajectory not found in animation module")

    result = _atmospheric_result()
    output_path = tmp_path / "trajectory.gif"

    sig = inspect.signature(animate_fn)
    kwargs: dict[str, Any] = {}
    if "num_frames" in sig.parameters:
        kwargs["num_frames"] = 5
    if "fps" in sig.parameters:
        kwargs["fps"] = 2

    animate_fn(result, str(output_path), **kwargs)

    assert output_path.exists(), f"Animation file not created: {output_path}"
    assert output_path.stat().st_size > 0, "Animation file is empty"
