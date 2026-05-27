try:
    from solution_homie import HomieSolution
except ImportError:  # pragma: no cover - local package import fallback
    from demo.solution_homie import HomieSolution

from typing import Any


class AlgSolution:
    """Platform entrypoint kept intentionally thin for submission."""

    def __init__(self):
        self._impl = HomieSolution()

    def reset(self, **kwargs):
        return self._impl.reset(**kwargs)

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
        return {}

    def predicts(self, obs, current_score):
        return self._impl.predicts(obs, current_score)

    def set_keyboard_command(self, nav_cmd=None, height_cmd=None):
        return self._impl.set_keyboard_command(nav_cmd, height_cmd)

    def get_debug_snapshot(self):
        return self._impl.get_debug_snapshot()
