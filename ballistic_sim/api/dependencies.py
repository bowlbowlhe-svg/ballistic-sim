"""API optional dependency guards.

Re-export the web extra guards so callers can require fastapi/uvicorn
with a friendly ``ImportError`` message.
"""

from __future__ import annotations

from ballistic_sim.utils.optional_imports import require_fastapi, require_uvicorn

__all__ = ["require_fastapi", "require_uvicorn"]
