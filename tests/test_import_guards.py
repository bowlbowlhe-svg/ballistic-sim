"""Tests for optional-dependency import guards."""

from __future__ import annotations

import sys

import pytest

from ballistic_sim.utils.optional_imports import (
    require_fastapi,
    require_gpu,
    require_rasterio,
    require_requests,
    require_weather,
    try_import,
)


@pytest.mark.parametrize(
    "call, module",
    [
        (require_weather, "cfgrib"),
        (require_rasterio, "rasterio"),
        (require_requests, "requests"),
        (require_gpu, "cupy"),
        (require_fastapi, "fastapi"),
        (lambda: try_import("PIL", "anim"), "PIL"),
        (lambda: try_import("uvicorn", "web"), "uvicorn"),
    ],
)
def test_guard_raises_friendly_import_error(call, module, monkeypatch) -> None:
    """When an optional dependency is missing the guard must raise ImportError."""
    # Setting sys.modules[name] = None makes import behave as if the module
    # is not installed (ModuleNotFoundError).
    monkeypatch.setitem(sys.modules, module, None)

    with pytest.raises(ImportError) as excinfo:
        call()

    message = str(excinfo.value)
    assert "pip install ballistic_sim[" in message
    assert module in message


def test_try_import_unknown_module() -> None:
    """try_import should raise ImportError for a deliberately missing module."""
    with pytest.raises(ImportError, match="not_a_real_module"):
        try_import("not_a_real_module", "test_extra")
