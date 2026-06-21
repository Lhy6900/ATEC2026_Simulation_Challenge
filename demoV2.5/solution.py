import json
import os
import sys
from typing import Any

import torch


def _install_official_action_spec_workaround() -> None:
    """Official play_atec_task.py references action_spec_json without defining it."""
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "action_spec_json"):
        setattr(main_module, "action_spec_json", json.dumps({}))


_install_official_action_spec_workaround()

try:
    from .boxpush_planner_solution import BoxPushPlannerSolution
    from .cross_pit_box_policy import CrossPitBoxPolicy
except ImportError:  # pragma: no cover - supports server.py style imports
    from boxpush_planner_solution import BoxPushPlannerSolution
    from cross_pit_box_policy import CrossPitBoxPolicy


class AlgSolution:

    def __init__(self, *, handoff_time_s: float | None = None, step_dt: float | None = None):
        self._boxpush = BoxPushPlannerSolution(
            planner_checkpoint="boxpush_planner_v2.pt",
            walk_policy="gr00t_walk.pt",
            balance_policy="gr00t_balance.pt",
        )
        self._cross_pit = None
        self._cross_pit_kwargs = {
            "model_path": os.environ.get("ATEC_CROSS_PIT_BOX_CHECKPOINT", "cross_pit_box_v2_5_blind_model_19999.pt"),
            "target_x": 7.8,
            "forward_command": float(os.environ.get("ATEC_CROSS_PIT_FORWARD_COMMAND", "0.6644027")),
            "stabilize_vel_gain": 0.0,
            "stabilize_steps": 0,
            "initial_last_action": os.environ.get("ATEC_CROSS_PIT_INITIAL_LAST_ACTION", "env_raw"),
            "handoff_blend_steps": 0,
            "handoff_max_delta": 0.0,
            "require_deploy_defaults": os.environ.get("ATEC_CROSS_PIT_REQUIRE_DEPLOY_DEFAULTS", "1")
            .strip()
            .lower()
            in ("1", "true", "yes", "on"),
        }
        self._impl = self._boxpush
        default_handoff_time = 9.5 if handoff_time_s is None else handoff_time_s
        default_step_dt = 0.02 if step_dt is None else step_dt
        self._handoff_time_s = float(os.environ.get("ATEC_CROSS_PIT_HANDOFF_TIME", default_handoff_time))
        self._step_dt = float(os.environ.get("ATEC_DEMO_STEP_DT", default_step_dt))
        self._walkout_enabled = os.environ.get("ATEC_WALKOUT_ENABLE", "0").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        self._walkout_trigger_x = float(os.environ.get("ATEC_WALKOUT_TRIGGER_X", "7.80"))
        self._walkout_trigger_z = float(os.environ.get("ATEC_WALKOUT_TRIGGER_Z", "0.45"))
        self._walkout_max_abs_y = float(os.environ.get("ATEC_WALKOUT_MAX_ABS_Y", "2.50"))
        self._walkout_command = [
            float(item)
            for item in os.environ.get("ATEC_WALKOUT_COMMAND", "1.0,0.0,0.0").replace(" ", "").split(",")
            if item != ""
        ]
        if len(self._walkout_command) != 3:
            raise ValueError("ATEC_WALKOUT_COMMAND must contain three comma-separated floats.")
        self._elapsed_time = 0.0
        self._handoff_done = False
        self._walkout_done = False
        self._step_count = 0
        self._debug_interval = int(os.environ.get("ATEC_DEBUG_STEP_INTERVAL", "50"))
        print(
            "[AlgSolution] push_then_cross "
            f"handoff_time={self._handoff_time_s:.2f}s step_dt={self._step_dt:.4f}s "
            f"walkout={int(self._walkout_enabled)}"
        )

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
        self._select_impl_for_current_time()
        if self._debug_interval > 0 and self._step_count % self._debug_interval == 0:
            print(f"[DEBUG] step={self._step_count} elapsed={self._elapsed_time:.2f} stage={self._stage_name()}")
        if self._stage_name() == "walkout":
            result = self._predict_walkout(obs)
        else:
            result = self._impl.predicts(obs, current_score)
        self._elapsed_time += self._step_dt
        return result

    def reset(self, **kwargs):
        self._step_count = 0
        self._elapsed_time = self._extract_elapsed_time(kwargs)
        self._handoff_done = False
        self._walkout_done = False
        self._impl = self._boxpush
        if hasattr(self._boxpush, "reset"):
            self._boxpush.reset(**kwargs)
        if self._cross_pit is not None and hasattr(self._cross_pit, "reset"):
            self._cross_pit.reset(**kwargs)
        self._select_impl_for_current_time()
        return None

    def get_debug_snapshot(self):
        snapshot = self._impl.get_debug_snapshot() if hasattr(self._impl, "get_debug_snapshot") else {}
        snapshot.update(
            {
                "stage": self._stage_name(),
                "step_count": self._step_count,
                "elapsed_time": self._elapsed_time,
                "handoff_time_s": self._handoff_time_s,
            }
        )
        return snapshot

    def _select_impl_for_current_time(self) -> None:
        if self._walkout_done:
            self._impl = self._boxpush
            return
        if self._elapsed_time + 1.0e-9 < self._handoff_time_s:
            self._impl = self._boxpush
            return
        self._ensure_cross_pit()
        if not self._handoff_done and hasattr(self._cross_pit, "reset"):
            self._cross_pit.reset()
            self._handoff_done = True
        if self._should_start_walkout():
            self._walkout_done = True
            self._impl = self._boxpush
            return
        self._impl = self._cross_pit

    def _stage_name(self) -> str:
        if self._walkout_done:
            return "walkout"
        return "cross_pit_box" if self._impl is self._cross_pit else "boxpush"

    def _ensure_cross_pit(self) -> None:
        if self._cross_pit is None:
            self._cross_pit = CrossPitBoxPolicy(**self._cross_pit_kwargs)

    def _should_start_walkout(self) -> bool:
        if not self._walkout_enabled or self._cross_pit is None or not hasattr(self._cross_pit, "get_debug_snapshot"):
            return False
        snapshot = self._cross_pit.get_debug_snapshot()
        local_x = snapshot.get("local_x")
        local_y = snapshot.get("local_y")
        local_z = snapshot.get("local_root_z")
        if local_x is None or local_y is None or local_z is None:
            return False
        return (
            float(local_x) >= self._walkout_trigger_x
            and float(local_z) >= self._walkout_trigger_z
            and abs(float(local_y)) <= self._walkout_max_abs_y
        )

    def _predict_walkout(self, obs: dict[str, Any]) -> dict[str, Any]:
        if not hasattr(self._boxpush, "low_level"):
            return self._boxpush.predicts(obs, 0.0)
        proprio = obs.get("proprio")
        if proprio is None:
            return self._boxpush.predicts(obs, 0.0)
        if not isinstance(proprio, torch.Tensor):
            proprio = torch.as_tensor(proprio, dtype=torch.float32)
        device = getattr(self._boxpush, "device", proprio.device)
        proprio = proprio.detach().to(device=device, dtype=torch.float32)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        command = torch.tensor(self._walkout_command, device=proprio.device, dtype=proprio.dtype).view(1, 3)
        command = command.expand(proprio.shape[0], -1)
        action = self._boxpush.low_level.predict({"proprio": proprio}, command)
        return {"action": action.detach().cpu().tolist(), "giveup": False}

    @staticmethod
    def _extract_elapsed_time(kwargs: dict[str, Any]) -> float:
        for key in ("elapsed_time", "Elapsed_Time", "time", "sim_time"):
            if key in kwargs:
                value = kwargs[key]
                if hasattr(value, "item"):
                    value = value.item()
                return float(value)
        return 0.0
