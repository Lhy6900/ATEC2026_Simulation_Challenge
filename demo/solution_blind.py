import os
from typing import Any

import torch

try:
    from groot_policy_adapter import ATEC_ACTION_DIM
except ImportError:
    from demo.groot_policy_adapter import ATEC_ACTION_DIM


class BlindLocoSolution:
    """Blind locomotion policy (policy.pt) driven entirely by keyboard velocity commands."""

    POLICY_ACTION_DIM = 29
    HISTORY_LENGTH = 5

    TRAIN_ACTION_SCALE = 0.25
    COMPETITION_ACTION_SCALE = 0.5

    COMP_JOINTS = (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )

    LAB_JOINTS = (
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "waist_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "waist_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    )

    TRAIN_DEFAULT_COMP_29 = [
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.3, 0.25, 0.0, 0.97, 0.15, 0.0, 0.0,
        0.3, -0.25, 0.0, 0.97, -0.15, 0.0, 0.0,
    ]

    COMPETITION_DEFAULT_33 = [
        -0.20, 0.0, 0.0, 0.42, -0.23, 0.0,
        -0.20, 0.0, 0.0, 0.42, -0.23, 0.0,
        0.0, 0.0, 0.0,
        0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
        0.35, -0.18, 0.0, 0.87, 0.0, 0.0, 0.0,
        0.024, 0.024, 0.024, 0.024,
    ]

    # Velocity command limits for keyboard control
    VX_MIN, VX_MAX = -0.5, 1.5
    VY_MIN, VY_MAX = -0.75, 0.75
    VYAW_MIN, VYAW_MAX = -1.0, 1.0

    def __init__(self, load_runtime: bool = True) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.pt")
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        self.comp_to_lab = torch.tensor(
            [self.COMP_JOINTS.index(name) for name in self.LAB_JOINTS],
            dtype=torch.long,
            device=self.device,
        )
        self.lab_to_comp = torch.tensor(
            [self.LAB_JOINTS.index(name) for name in self.COMP_JOINTS],
            dtype=torch.long,
            device=self.device,
        )

        train_default_comp = torch.tensor(
            self.TRAIN_DEFAULT_COMP_29, dtype=torch.float32, device=self.device
        ).view(1, -1)
        self.competition_default = torch.tensor(
            self.COMPETITION_DEFAULT_33, dtype=torch.float32, device=self.device
        ).view(1, -1)
        self.train_default_lab = train_default_comp[:, self.comp_to_lab]

        self._history: dict[str, torch.Tensor] = {}
        self._last_policy_action: torch.Tensor | None = None
        self._velocity_command = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
        self._debug_snapshot: dict[str, Any] = {
            "policy": "blind",
            "nav_cmd": [0.0, 0.0, 0.0],
        }

    def reset(self, **kwargs) -> None:
        self._history.clear()
        self._last_policy_action = None
        self._velocity_command = torch.zeros(1, 3, dtype=torch.float32, device=self.device)

    def set_keyboard_command(self, nav_cmd: list[float] | None, height_cmd: float | None) -> None:
        if nav_cmd is not None:
            clamped = [
                max(self.VX_MIN, min(self.VX_MAX, nav_cmd[0])),
                max(self.VY_MIN, min(self.VY_MAX, nav_cmd[1])),
                max(self.VYAW_MIN, min(self.VYAW_MAX, nav_cmd[2])),
            ]
            self._velocity_command = torch.tensor(
                [clamped], dtype=torch.float32, device=self.device
            )
            self._debug_snapshot["nav_cmd"] = clamped

    def _as_tensor_2d(self, value) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _append_history(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if name not in self._history or self._history[name].shape[0] != value.shape[0]:
            self._history[name] = value.unsqueeze(1).repeat(1, self.HISTORY_LENGTH, 1)
        else:
            self._history[name] = torch.cat(
                [self._history[name][:, 1:, :], value.unsqueeze(1)], dim=1
            )
        return self._history[name].reshape(value.shape[0], -1)

    def _build_policy_obs(self, obs) -> tuple[torch.Tensor, int]:
        proprio = self._as_tensor_2d(obs["proprio"])
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        num_envs = proprio.shape[0]
        dtype = proprio.dtype

        idx = 0
        base_lin_vel = proprio[:, idx:idx + 3]; idx += 3
        base_ang_vel = proprio[:, idx:idx + 3]; idx += 3
        _env_velocity_commands = proprio[:, idx:idx + 3]; idx += 3
        projected_gravity = proprio[:, idx:idx + 3]; idx += 3
        joint_pos_rel = proprio[:, idx:idx + action_dim]; idx += action_dim
        joint_vel = proprio[:, idx:idx + action_dim]

        if self._last_policy_action is None or self._last_policy_action.shape[0] != num_envs:
            self._last_policy_action = torch.zeros(
                (num_envs, self.POLICY_ACTION_DIM), dtype=dtype, device=self.device
            )

        competition_default = self.competition_default.repeat(num_envs, 1)
        command = self._velocity_command.repeat(num_envs, 1)

        joint_pos_abs_comp = (
            joint_pos_rel[:, :self.POLICY_ACTION_DIM]
            + competition_default[:, :self.POLICY_ACTION_DIM]
        )
        joint_pos_abs_lab = joint_pos_abs_comp[:, self.comp_to_lab]
        joint_vel_lab = joint_vel[:, :self.POLICY_ACTION_DIM][:, self.comp_to_lab]

        terms = [
            ("base_lin_vel", base_lin_vel),
            ("base_ang_vel", base_ang_vel),
            ("velocity_commands", command),
            ("projected_gravity", projected_gravity),
            ("joint_pos", joint_pos_abs_lab),
            ("joint_vel", joint_vel_lab),
            ("actions", self._last_policy_action.to(dtype=dtype)),
        ]
        policy_obs = torch.cat([self._append_history(name, value) for name, value in terms], dim=-1)
        return policy_obs, action_dim

    def _map_policy_action_to_competition(
        self, policy_action: torch.Tensor, action_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_envs = policy_action.shape[0]
        dtype = policy_action.dtype
        train_default_lab = self.train_default_lab.repeat(num_envs, 1)
        competition_default = self.competition_default.repeat(num_envs, 1)

        train_target_lab = train_default_lab + self.TRAIN_ACTION_SCALE * policy_action
        train_target_comp = train_target_lab[:, self.lab_to_comp]

        env_action_29_comp = (
            train_target_comp - competition_default[:, :self.POLICY_ACTION_DIM]
        ) / self.COMPETITION_ACTION_SCALE

        env_action = torch.zeros((num_envs, action_dim), dtype=dtype, device=self.device)
        env_action[:, :self.POLICY_ACTION_DIM] = env_action_29_comp
        return env_action, policy_action.detach()

    def predicts(self, obs, current_score):
        policy_obs, action_dim = self._build_policy_obs(obs)

        with torch.inference_mode():
            policy_action = self.policy(policy_obs)

        policy_action = torch.as_tensor(policy_action, dtype=torch.float32, device=self.device)
        if policy_action.ndim == 1:
            policy_action = policy_action.unsqueeze(0)

        env_action, self._last_policy_action = self._map_policy_action_to_competition(
            policy_action, action_dim
        )
        return {"action": env_action.cpu().numpy().tolist(), "giveup": False}

    def get_debug_snapshot(self) -> dict[str, Any]:
        return dict(self._debug_snapshot)
