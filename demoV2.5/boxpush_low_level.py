"""GR00T low-level locomotion wrapper used by the TaskD box-pushing demo."""

from __future__ import annotations

import math
from pathlib import Path

import torch


GR00T_LOWER_BODY_DEFAULT_15 = [
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    0.0, 0.0, 0.0,
]

# GR00T's internal standing reference. This is intentionally distinct from the
# official TaskD action offset below.
GR00T_G1_DEFAULT_33 = [
    -0.10, 0.0, 0.0, 0.30, -0.20, 0.0,
    -0.10, 0.0, 0.0, 0.30, -0.20, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.3, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, -0.3, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]

# Official ATEC TaskD-G1 default joint positions from UNITREE_G1_29DOF_DEX1_CFG.
TASKD_ENV_DEFAULT_33 = (
    -0.20, 0.0, 0.0, 0.42, -0.23, 0.0,
    -0.20, 0.0, 0.0, 0.42, -0.23, 0.0,
    0.0, 0.0, 0.0,
    0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
    0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
    0.024, 0.024, 0.024, 0.024,
)

TASKD_PLANNER_PD_GAINS_33 = (
    (150.0, 2.0), (150.0, 2.0), (150.0, 2.0), (200.0, 4.0), (40.0, 2.0), (40.0, 2.0),
    (150.0, 2.0), (150.0, 2.0), (150.0, 2.0), (200.0, 4.0), (40.0, 2.0), (40.0, 2.0),
    (250.0, 5.0), (250.0, 5.0), (250.0, 5.0),
    (100.0, 2.0), (100.0, 2.0), (50.0, 2.0), (50.0, 2.0), (40.0, 2.0), (40.0, 2.0), (40.0, 2.0),
    (100.0, 2.0), (100.0, 2.0), (50.0, 2.0), (50.0, 2.0), (40.0, 2.0), (40.0, 2.0), (40.0, 2.0),
    (800.0, 3.0), (800.0, 3.0), (800.0, 3.0), (800.0, 3.0),
)
OFFICIAL_DYNAMICS_PD_GAINS_33 = (
    (200.0, 5.0), (150.0, 5.0), (150.0, 5.0), (200.0, 5.0), (20.0, 2.0), (20.0, 2.0),
    (200.0, 5.0), (150.0, 5.0), (150.0, 5.0), (200.0, 5.0), (20.0, 2.0), (20.0, 2.0),
    (200.0, 5.0), (200.0, 5.0), (200.0, 5.0),
    (100.0, 2.0), (100.0, 2.0), (50.0, 2.0), (50.0, 2.0), (40.0, 2.0), (40.0, 2.0), (40.0, 2.0),
    (100.0, 2.0), (100.0, 2.0), (50.0, 2.0), (50.0, 2.0), (40.0, 2.0), (40.0, 2.0), (40.0, 2.0),
    (800.0, 3.0), (800.0, 3.0), (800.0, 3.0), (800.0, 3.0),
)

DEFAULT_UPPER_BODY_TARGET_29 = GR00T_G1_DEFAULT_33[:29]
GR00T_CMD_SCALE = [2.0, 2.0, 0.5]
GR00T_ANG_VEL_SCALE = 0.5
GR00T_DOF_VEL_SCALE = 0.05
GR00T_ACTION_SCALE = 0.25
GR00T_NUM_ACTIONS = 15
GR00T_SINGLE_OBS_DIM = 86
GR00T_HISTORY_LEN = 6
GR00T_NUM_OBS = GR00T_SINGLE_OBS_DIM * GR00T_HISTORY_LEN
ATEC_ENV_ACTION_SCALE = 0.5


def _pd_gain_tensors(gains: tuple[tuple[float, float], ...], *, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    kp = torch.tensor([gain[0] for gain in gains], device=device, dtype=dtype).view(1, -1)
    kd = torch.tensor([gain[1] for gain in gains], device=device, dtype=dtype).view(1, -1)
    return kp, kd


def compensate_action_for_official_dynamics(env_action: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
    """Convert boxpush-gain actions into torque-equivalent official-gain actions."""
    if env_action.shape[-1] != 33:
        raise ValueError(f"Expected 33-D env_action, got shape {tuple(env_action.shape)}")
    if proprio.shape[-1] < 78:
        raise ValueError(f"Expected proprio with at least 78 values, got shape {tuple(proprio.shape)}")

    device = env_action.device
    dtype = env_action.dtype
    proprio = proprio.to(device=device, dtype=dtype)
    default = torch.tensor(TASKD_ENV_DEFAULT_33, device=device, dtype=dtype).view(1, -1)
    planner_kp, planner_kd = _pd_gain_tensors(TASKD_PLANNER_PD_GAINS_33, device=device, dtype=dtype)
    official_kp, official_kd = _pd_gain_tensors(OFFICIAL_DYNAMICS_PD_GAINS_33, device=device, dtype=dtype)

    q = default + proprio[:, 12:45]
    qd = proprio[:, 45:78]
    planner_target = default + ATEC_ENV_ACTION_SCALE * env_action
    official_target = (
        q
        + (planner_kp / official_kp) * (planner_target - q)
        + ((official_kd - planner_kd) / official_kp) * qd
    )
    return (official_target - default) / ATEC_ENV_ACTION_SCALE


class GrootLowLevelPolicy:
    """Batched GR00T low-level policy for official TaskD playback."""

    def __init__(
        self,
        walk_path: str | Path,
        balance_path: str | Path,
        *,
        device: str | torch.device,
        compensate_official_dynamics: bool = True,
    ):
        self.device = torch.device(device)
        self.compensate_official_dynamics = bool(compensate_official_dynamics)
        self.walk = torch.jit.load(str(walk_path), map_location=self.device).eval()
        self.balance = torch.jit.load(str(balance_path), map_location=self.device).eval()

        self._cmd_scale = torch.tensor(GR00T_CMD_SCALE, device=self.device)
        self._lower_default = torch.tensor(GR00T_LOWER_BODY_DEFAULT_15, device=self.device)
        self._upper_default = torch.tensor(DEFAULT_UPPER_BODY_TARGET_29, device=self.device)
        self._gr00t_default = torch.tensor(GR00T_G1_DEFAULT_33, device=self.device)

        default_29 = torch.zeros(29, device=self.device)
        default_29[:15] = self._lower_default
        self._default_29 = default_29

        self._num_envs = 0
        self._obs_history: list[torch.Tensor] | None = None
        self._last_lower_action: torch.Tensor | None = None

    def reset(self, num_envs: int) -> None:
        self._num_envs = int(num_envs)
        self._obs_history = None
        self._last_lower_action = torch.zeros(self._num_envs, GR00T_NUM_ACTIONS, device=self.device)

    @torch.no_grad()
    def predict(self, obs_dict: dict, nav_cmd: torch.Tensor) -> torch.Tensor:
        proprio = obs_dict["proprio"].to(device=self.device, dtype=torch.float32)
        nav_cmd = nav_cmd.to(device=self.device, dtype=torch.float32)
        if self._last_lower_action is None or self._num_envs != proprio.shape[0]:
            self.reset(proprio.shape[0])

        single_obs = self._build_obs(proprio, nav_cmd)
        full_obs = self._stack_history(single_obs)

        nav_norm = torch.norm(nav_cmd, dim=-1)
        use_walk = nav_norm >= 0.05
        out_walk = self.walk(full_obs)
        out_balance = self.balance(full_obs)
        raw_action = torch.where(use_walk.unsqueeze(-1), out_walk, out_balance)

        self._last_lower_action = raw_action
        env_action = self._map_to_env_action(raw_action)
        if self.compensate_official_dynamics:
            env_action = compensate_action_for_official_dynamics(env_action, proprio)
        return env_action

    def _build_obs(self, proprio: torch.Tensor, nav_cmd: torch.Tensor) -> torch.Tensor:
        scaled_cmd = nav_cmd * self._cmd_scale
        num_envs = proprio.shape[0]
        height_cmd = proprio.new_full((num_envs, 1), 0.74)
        torso_rpy = self._compute_torso_rpy(proprio)
        ang_vel = proprio[:, 3:6] * GR00T_ANG_VEL_SCALE
        gravity = proprio[:, 9:12]

        joint_pos_rel = proprio[:, 12:45]
        joint_pos_abs = self._gr00t_default.unsqueeze(0) + joint_pos_rel
        scaled_pos = joint_pos_abs[:, :29] - self._default_29.unsqueeze(0)
        scaled_vel = proprio[:, 45:78][:, :29] * GR00T_DOF_VEL_SCALE

        return torch.cat(
            [
                scaled_cmd,
                height_cmd,
                torso_rpy,
                ang_vel,
                gravity,
                scaled_pos,
                scaled_vel,
                self._last_lower_action,
            ],
            dim=-1,
        )

    def _compute_torso_rpy(self, proprio: torch.Tensor) -> torch.Tensor:
        joint_pos_abs = self._gr00t_default.unsqueeze(0) + proprio[:, 12:45]
        waist_yaw = joint_pos_abs[:, 12]
        waist_roll = joint_pos_abs[:, 13]
        waist_pitch = joint_pos_abs[:, 14]

        cy, sy = torch.cos(waist_yaw), torch.sin(waist_yaw)
        cr, sr = torch.cos(waist_roll), torch.sin(waist_roll)
        cp, sp = torch.cos(waist_pitch), torch.sin(waist_pitch)

        r20 = -cr * sp
        pitch_out = torch.asin(torch.clamp(-r20, -1.0, 1.0))
        r21 = sr
        r22 = cr * cp
        roll_out = torch.atan2(r21, r22)
        r10 = sy * cp + cy * sr * sp
        r00 = cy * cp - sy * sr * sp
        yaw_out = torch.atan2(r10, r00)
        return torch.stack([roll_out, pitch_out, yaw_out], dim=-1)

    def _stack_history(self, single_obs: torch.Tensor) -> torch.Tensor:
        if self._obs_history is None:
            self._obs_history = []
        self._obs_history.append(single_obs.detach().clone())
        while len(self._obs_history) < GR00T_HISTORY_LEN:
            self._obs_history.insert(0, torch.zeros_like(single_obs))
        if len(self._obs_history) > GR00T_HISTORY_LEN:
            self._obs_history = self._obs_history[-GR00T_HISTORY_LEN:]
        return torch.cat(self._obs_history, dim=-1)

    def _map_to_env_action(self, raw_15: torch.Tensor) -> torch.Tensor:
        num_envs = raw_15.shape[0]
        lower_target = self._lower_default.unsqueeze(0) + GR00T_ACTION_SCALE * raw_15

        body_target_29 = self._upper_default.unsqueeze(0).expand(num_envs, -1).clone()
        body_target_29[:, :15] = lower_target

        env_action = torch.zeros(num_envs, 33, device=self.device)
        gr00t_default = self._gr00t_default[:29].unsqueeze(0)
        env_action[:, :29] = (body_target_29 - gr00t_default) / ATEC_ENV_ACTION_SCALE
        return env_action


def yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Small utility kept for future sensor-only handoff logic."""
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi
