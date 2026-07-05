"""Web API package for ballistic_sim.

Importing this package requires the ``web`` extra (fastapi + uvicorn).
"""

from __future__ import annotations

from ballistic_sim.api.main import create_app

app = create_app()

__all__ = ["create_app", "app"]
