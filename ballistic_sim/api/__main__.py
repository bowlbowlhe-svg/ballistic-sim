"""Entry point: ``python -m ballistic_sim.api``."""

from __future__ import annotations

import uvicorn

from ballistic_sim.api import create_app
from ballistic_sim.api.dependencies import require_uvicorn

require_uvicorn()

uvicorn.run(create_app(), host="127.0.0.1", port=8000)
