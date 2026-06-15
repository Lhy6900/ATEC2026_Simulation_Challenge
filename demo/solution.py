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
            from .vision_loco_policy import VisionLocoConfig, VisionLocoPolicy
        except ImportError:  # pragma: no cover - supports server.py style imports
            from boxpush_planner_solution import BoxPushPlannerSolution
            from vision_loco_policy import VisionLocoConfig, VisionLocoPolicy

        self._impl = BoxPushPlannerSolution()
        self._vision_cls = VisionLocoPolicy
        self._vision_cfg = VisionLocoConfig.from_env()
        self._vision_impl = None
        self._step_count = 0
        self._debug_interval = int(os.environ.get("ATEC_DEBUG_STEP_INTERVAL", "0"))
        self._last_stage = "boxpush"
        self._debug_snapshot: dict[str, Any] = {}

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
        elapsed_time = self._estimate_elapsed_time(obs)
        use_vision = elapsed_time >= float(self._vision_cfg.switch_time_s)
        if self._debug_interval > 0 and self._step_count % self._debug_interval == 0:
            print(f"[DEBUG] step={self._step_count} elapsed={elapsed_time:.3f} stage={self._last_stage}")

        if not use_vision:
            self._last_stage = "boxpush"
            resp = self._impl.predicts(obs, current_score)
            self._debug_snapshot = {
                "stage": self._last_stage,
                "step": self._step_count,
                "elapsed_time_s": elapsed_time,
            }
            return resp

        if self._vision_impl is None:
            self._vision_impl = self._vision_cls(config=self._vision_cfg)
            print(
                "[AlgSolution] Switch to vision loco policy at "
                f"elapsed={elapsed_time:.3f}s step={self._step_count}"
            )

        self._last_stage = "vision_loco"
        action = self._vision_impl.predict(obs, elapsed_time_s=elapsed_time)
        self._debug_snapshot = {
            "stage": self._last_stage,
            "step": self._step_count,
            "elapsed_time_s": elapsed_time,
            "vision": self._vision_impl.get_debug_snapshot(),
        }
        return {"action": action.detach().cpu().tolist(), "giveup": False}

    def reset(self, **kwargs):
        self._step_count = 0
        self._last_stage = "boxpush"
        self._debug_snapshot = {}
        if self._vision_impl is not None:
            self._vision_impl.reset()
        if hasattr(self._impl, "reset"):
            return self._impl.reset(**kwargs)
        return None

    def get_debug_snapshot(self):
        snapshot = dict(self._debug_snapshot)
        if self._last_stage == "boxpush" and hasattr(self._impl, "get_debug_snapshot"):
            snapshot["boxpush"] = self._impl.get_debug_snapshot()
        return snapshot

    def _estimate_elapsed_time(self, obs) -> float:
        for key in ("Elapsed_Time", "elapsed_time", "time"):
            if isinstance(obs, dict) and key in obs:
                value = obs[key]
                try:
                    if hasattr(value, "detach"):
                        value = value.detach()
                    if hasattr(value, "flatten"):
                        value = value.flatten()[0]
                    if hasattr(value, "item"):
                        return float(value.item())
                    return float(value)
                except (TypeError, ValueError, IndexError):
                    pass
        return float(max(self._step_count - 1, 0)) * float(self._vision_cfg.nominal_step_dt)
