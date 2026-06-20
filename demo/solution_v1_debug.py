"""Local debug entrypoint for the v1 heightmap cross-pit policy.

Cloud submission should normally use ``solution.py``. This variant is useful
for local Isaac Sim validation when ``scripts/play_atec_task.py`` attaches the
training-like D435 raycaster and forwards ``_debug_depth_ray_hits_w`` into the
solution observation.
"""

from __future__ import annotations

import os

os.environ.setdefault("ATEC_CROSS_PIT_BOX_CHECKPOINT", "cross_pit_box_model_19999.pt")
os.environ.setdefault("ATEC_CROSS_PIT_REQUIRE_DEPLOY_DEFAULTS", "0")
os.environ.setdefault("ATEC_CROSS_PIT_FORWARD_COMMAND", "0.755")
os.environ.setdefault("ATEC_CROSS_PIT_STABILIZE_STEPS", "2")
os.environ.setdefault("ATEC_PLAY_ATTACH_CROSS_PIT_DEPTH_SCANNER", "1")

try:
    from .solution import AlgSolution as _PushThenCrossSolution
except ImportError:  # pragma: no cover - supports server.py style imports
    from solution import AlgSolution as _PushThenCrossSolution


class AlgSolution(_PushThenCrossSolution):
    """Push-then-cross solution configured for the local v1 heightmap policy."""

