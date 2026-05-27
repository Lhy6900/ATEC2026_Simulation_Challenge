from typing import Any

try:
    from groot_policy_adapter import (
        ATEC_ENV_ACTION_SCALE,
        ATEC_ACTION_DIM,
        ATEC_G1_DEFAULT_33,
        GrootPolicyAdapter,
    )
except ImportError:  # pragma: no cover - local package import fallback
    from demo.groot_policy_adapter import (
        ATEC_ENV_ACTION_SCALE,
        ATEC_ACTION_DIM,
        ATEC_G1_DEFAULT_33,
        GrootPolicyAdapter,
    )


class HomieSolution:
    """Minimal submission-oriented wrapper around the GR00T adapter."""

    def __init__(
        self,
        load_runtime: bool = True,
        startup_zero_steps: int = 5,
        enable_home_stage: bool = False,
        home_gain: float = 1.5,
        home_damping: float = 0.15,
        home_tolerance: float = 0.08,
        home_hold_steps: int = 5,
        nav_ramp_steps: int = 80,
    ) -> None:
        self.adapter = GrootPolicyAdapter(load_runtime=load_runtime)
        self.startup_zero_steps = max(0, int(startup_zero_steps))
        self.enable_home_stage = bool(enable_home_stage)
        self.home_gain = float(home_gain)
        self.home_damping = float(home_damping)
        self.home_tolerance = float(home_tolerance)
        self.home_hold_steps = max(1, int(home_hold_steps))
        self.nav_ramp_steps = max(1, int(nav_ramp_steps))
        self._startup_step = 0
        self._home_done = False
        self._home_stable_steps = 0
        self._policy_step = 0
        self._keyboard_active = False
        self._stage = None
        self._debug_snapshot = {
            "stage": "init",
            "runtime_loaded": self.adapter._runtime is not None,
            "home_max_err": None,
            "home_stable_steps": 0,
            "policy_step": 0,
        }
        print(f"[HomieSolution] runtime_loaded={self.adapter._runtime is not None}")

    def reset(self, **kwargs: Any) -> None:
        self.adapter.reset()
        self._startup_step = 0
        self._home_done = False
        self._home_stable_steps = 0
        self._policy_step = 0
        self._keyboard_active = False
        self._stage = None
        self._debug_snapshot.update(
            {
                "stage": "reset",
                "home_max_err": None,
                "home_stable_steps": 0,
                "policy_step": 0,
            }
        )

    def _zero_action(self) -> list[list[float]]:
        return [[0.0] * ATEC_ACTION_DIM]

    def set_keyboard_command(
        self, nav_cmd: list[float] | None, height_cmd: float | None
    ) -> None:
        if nav_cmd is not None:
            self.adapter.set_navigation_command(nav_cmd)
            self.adapter.set_navigation_scale(1.0)
            self._keyboard_active = True
        if height_cmd is not None:
            self.adapter.set_base_height_cmd(height_cmd)

    def _set_stage(self, stage: str) -> None:
        if self._stage != stage:
            self._stage = stage
            self._debug_snapshot["stage"] = stage
            print(f"[HomieSolution] stage={stage}")

    def _compute_home_action(self, obs) -> list[list[float]]:
        joint_state = self.adapter.extract_joint_state(obs)
        q = joint_state["q_atec_33"]
        dq = joint_state["dq_atec_33"]
        action = [0.0] * ATEC_ACTION_DIM

        max_err = 0.0
        for idx, target in enumerate(ATEC_G1_DEFAULT_33):
            q_err = target - q[idx]
            max_err = max(max_err, abs(q_err))
            u = self.home_gain * q_err - self.home_damping * dq[idx]
            action[idx] = max(-1.0, min(1.0, u / ATEC_ENV_ACTION_SCALE))
        self._debug_snapshot["home_max_err"] = round(max_err, 6)

        if max_err < self.home_tolerance:
            self._home_stable_steps += 1
            if self._home_stable_steps >= self.home_hold_steps:
                self._home_done = True
            self._debug_snapshot["home_stable_steps"] = self._home_stable_steps
            return self._zero_action()

        self._home_stable_steps = 0
        self._debug_snapshot["home_stable_steps"] = 0

        return [action]

    def predicts(self, obs, current_score):
        if self._startup_step < self.startup_zero_steps:
            self._set_stage("startup_zero")
            self._startup_step += 1
            return {"action": self._zero_action(), "giveup": False}

        if self.enable_home_stage and not self._home_done:
            self._set_stage("home")
            return {"action": self._compute_home_action(obs), "giveup": False}

        self._set_stage("policy")
        self._policy_step += 1
        self._debug_snapshot["policy_step"] = self._policy_step
        if not self._keyboard_active:
            self.adapter.set_navigation_scale(min(1.0, self._policy_step / self.nav_ramp_steps))
        action = self.adapter.policy_inference(obs)
        return {"action": action, "giveup": False}

    def get_debug_snapshot(self) -> dict[str, Any]:
        snapshot = dict(self._debug_snapshot)
        snapshot.update(self.adapter.get_debug_snapshot())
        return snapshot
