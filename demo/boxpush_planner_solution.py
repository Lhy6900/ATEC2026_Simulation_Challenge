"""Official-demo entry policy for TaskD box pushing.

This module intentionally depends only on observations passed into
``AlgSolution.predicts``. It does not touch env_cfg or the Gym environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

try:
    from .boxpush_low_level import GrootLowLevelPolicy
    from .planner_inference_policy import load_planner_inference_policy
except ImportError:  # pragma: no cover - supports server.py style imports
    from boxpush_low_level import GrootLowLevelPolicy
    from planner_inference_policy import load_planner_inference_policy


NAV_CMD_MIN = torch.tensor([-1.5, -0.75, -1.0])
NAV_CMD_MAX = torch.tensor([1.5, 0.75, 1.0])
ACTION_SCALE = 0.25
CLIP_ACTIONS = torch.tensor([0.5, 0.3, 0.3])
PLANNER_DECIMATION = 5
PLANNER_OBS_DIM = 372
PLANNER_ACTION_DIM = 3
DEFAULT_PLANNER_CHECKPOINT = "boxpush_planner_10hz.pt"


def _as_2d_float_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        tensor = torch.as_tensor(value)
    tensor = tensor.to(device=device, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def compress_height_scan(height_scan: torch.Tensor) -> torch.Tensor:
    """Compress official TaskD LiDAR height scan from 5760D to planner 360D."""
    if height_scan.ndim == 1:
        height_scan = height_scan.unsqueeze(0)
    if height_scan.shape[-1] != 5760:
        raise ValueError(f"Expected extero height scan with 5760 values, got shape {tuple(height_scan.shape)}")
    h = height_scan.view(-1, 16, 360)
    h = h.view(-1, 4, 4, 360).mean(dim=2)
    h = h.view(-1, 4, 90, 4).mean(dim=-1)
    return h.clamp(-1.0, 1.0).view(-1, 360)


def update_nav_command(previous: torch.Tensor, planner_action: torch.Tensor) -> torch.Tensor:
    """Planner action is a relative velocity-command increment."""
    nav_min = NAV_CMD_MIN.to(device=previous.device, dtype=previous.dtype)
    nav_max = NAV_CMD_MAX.to(device=previous.device, dtype=previous.dtype)
    return torch.clamp(previous + ACTION_SCALE * planner_action, nav_min, nav_max)


class BoxPushPlannerSolution:
    """Planner checkpoint plus GR00T low-level policy packaged for official play."""

    def __init__(
        self,
        *,
        planner_checkpoint: str | Path | None = None,
        walk_policy: str | Path | None = None,
        balance_policy: str | Path | None = None,
        device: str | torch.device | None = None,
        compensate_official_dynamics: bool = True,
    ) -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.planner_checkpoint = self._resolve_path(
            planner_checkpoint,
            env_name="ATEC_BOXPUSH_PLANNER_CHECKPOINT",
            default_name=DEFAULT_PLANNER_CHECKPOINT,
        )
        self.walk_policy = self._resolve_path(
            walk_policy,
            env_name="ATEC_BOXPUSH_WALK_POLICY",
            default_name="gr00t_walk.pt",
        )
        self.balance_policy = self._resolve_path(
            balance_policy,
            env_name="ATEC_BOXPUSH_BALANCE_POLICY",
            default_name="gr00t_balance.pt",
        )

        self.planner = load_planner_inference_policy(
            self.planner_checkpoint,
            device=self.device,
            expected_obs_dim=PLANNER_OBS_DIM,
            expected_action_dim=PLANNER_ACTION_DIM,
            activation="elu",
        )
        self.low_level = GrootLowLevelPolicy(
            self.walk_policy,
            self.balance_policy,
            device=self.device,
            compensate_official_dynamics=compensate_official_dynamics,
        )

        self._num_envs = 0
        self._nav_cmd: torch.Tensor | None = None
        self._last_planner_action: torch.Tensor | None = None
        self._step = 0
        self._debug_snapshot: dict[str, Any] = {
            "stage": "init",
            "device": str(self.device),
            "planner_checkpoint": str(self.planner_checkpoint),
            "walk_policy": str(self.walk_policy),
            "balance_policy": str(self.balance_policy),
            "compensate_official_dynamics": bool(compensate_official_dynamics),
            "planner_decimation": PLANNER_DECIMATION,
        }
        print(
            "[BoxPushPlannerSolution] "
            f"device={self.device} planner={self.planner_checkpoint.name} "
            f"planner_decimation={PLANNER_DECIMATION} "
            f"official_dynamics_compensation={bool(compensate_official_dynamics)}"
        )

    def _resolve_path(self, explicit: str | Path | None, *, env_name: str, default_name: str) -> Path:
        value = explicit or os.environ.get(env_name) or default_name
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.base_dir / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required policy asset: {path}. "
                f"Set {env_name} or place {default_name} in {self.base_dir}."
            )
        return path

    def reset(self, **kwargs: Any) -> None:
        self._num_envs = 0
        self._nav_cmd = None
        self._last_planner_action = None
        self._step = 0
        self.low_level.reset(0)
        self._debug_snapshot.update({"stage": "reset", "step": 0, "planner_phase": 0})

    def get_action_spec(self) -> dict[str, dict[str, Any]]:
        return {}

    @torch.no_grad()
    def predicts(self, obs: dict[str, Any], current_score: float) -> dict[str, Any]:
        proprio = _as_2d_float_tensor(obs["proprio"], device=self.device)
        extero = _as_2d_float_tensor(obs["extero"], device=self.device)
        self._ensure_state(proprio.shape[0], proprio.dtype)

        planner_updated = self._step % PLANNER_DECIMATION == 0
        if planner_updated:
            planner_obs = self._build_planner_obs(proprio, extero)
            planner_action = self.planner.act_inference({"policy": planner_obs})
            clip_actions = CLIP_ACTIONS.to(device=self.device, dtype=planner_action.dtype)
            planner_action = torch.clamp(planner_action, -clip_actions, clip_actions)
            self._nav_cmd = update_nav_command(self._nav_cmd, planner_action)
            self._last_planner_action = planner_action.detach().clone()
        else:
            planner_action = self._last_planner_action

        low_action = self.low_level.predict({"proprio": proprio}, self._nav_cmd)
        self._step += 1
        self._debug_snapshot.update(
            {
                "stage": "planner_low_level",
                "step": self._step,
                "score": float(current_score),
                "planner_updated": bool(planner_updated),
                "planner_phase": self._step % PLANNER_DECIMATION,
                "nav_cmd": self._nav_cmd[0].detach().cpu().tolist(),
                "planner_action": planner_action[0].detach().cpu().tolist(),
                "action_head": low_action[0, :8].detach().cpu().tolist(),
            }
        )
        return {"action": low_action.detach().cpu().tolist(), "giveup": False}

    def _ensure_state(self, num_envs: int, dtype: torch.dtype) -> None:
        if self._num_envs == num_envs and self._nav_cmd is not None and self._last_planner_action is not None:
            return
        self._num_envs = int(num_envs)
        self._nav_cmd = torch.zeros(num_envs, 3, device=self.device, dtype=dtype)
        self._last_planner_action = torch.zeros(num_envs, 3, device=self.device, dtype=dtype)
        self._step = 0
        self.low_level.reset(num_envs)

    def _build_planner_obs(self, proprio: torch.Tensor, extero: torch.Tensor) -> torch.Tensor:
        lin_vel = proprio[:, 0:3]
        ang_vel = proprio[:, 3:6]
        gravity = proprio[:, 9:12]
        h_scan = compress_height_scan(extero)
        return torch.cat(
            [
                lin_vel,
                ang_vel,
                gravity,
                self._last_planner_action,
                h_scan,
            ],
            dim=-1,
        )

    def get_debug_snapshot(self) -> dict[str, Any]:
        return dict(self._debug_snapshot)
