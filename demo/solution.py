import json
import os
import sys
from typing import Any


def _install_official_action_spec_workaround() -> None:
    """Official play_atec_task.py references action_spec_json without defining it."""
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "action_spec_json"):
        setattr(main_module, "action_spec_json", json.dumps({}))


_install_official_action_spec_workaround()


class AlgSolution:

    def __init__(self):
        try:
            from .boxpush_planner_solution import BoxPushPlannerSolution
        except ImportError:  # pragma: no cover - supports server.py style imports
            from boxpush_planner_solution import BoxPushPlannerSolution

        self._impl = BoxPushPlannerSolution()
        self._step_count = 0
        self._debug_interval = int(os.environ.get("ATEC_DEBUG_STEP_INTERVAL", "0"))

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        """Optional action customization.

        Return None to use official default action config.

        Allowed groups:
            - leg
            - arm
            - wheel

        Allowed fields:
            - mode: "position", "velocity", or "effort"
            - scale: positive float
            - clip: None or [min, max]
        """
        return self._impl.get_action_spec()

    def predicts(self, obs, current_score):
        self._step_count += 1
        if self._debug_interval > 0 and self._step_count % self._debug_interval == 0:
            print(f"[DEBUG] step={self._step_count}")
        return self._impl.predicts(obs, current_score)

    def reset(self, **kwargs):
        self._step_count = 0
        if hasattr(self._impl, "reset"):
            return self._impl.reset(**kwargs)
        return None

    def get_debug_snapshot(self):
        if hasattr(self._impl, "get_debug_snapshot"):
            return self._impl.get_debug_snapshot()
        return {}
