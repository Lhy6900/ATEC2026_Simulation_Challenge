from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

try:
    from .config import VisionLocoConfig
    from .height_memory import DEFAULT_HEIGHT_FILE, HeightMemory, HeightMemoryConfig
except ImportError:  # pragma: no cover - supports server.py style imports
    from config import VisionLocoConfig
    from height_memory import DEFAULT_HEIGHT_FILE, HeightMemory, HeightMemoryConfig

OFFICIAL_G1_JOINT_NAMES_33 = (
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
    "left_hand_Joint1_1",
    "left_hand_Joint2_1",
    "right_hand_Joint1_1",
    "right_hand_Joint2_1",
)

OFFICIAL_G1_DEFAULT_33 = (
    -0.20,
    0.0,
    0.0,
    0.42,
    -0.23,
    0.0,
    -0.20,
    0.0,
    0.0,
    0.42,
    -0.23,
    0.0,
    0.0,
    0.0,
    0.0,
    0.35,
    0.18,
    0.0,
    0.87,
    0.0,
    0.0,
    0.0,
    0.35,
    -0.18,
    0.0,
    0.87,
    0.0,
    0.0,
    0.0,
    0.024,
    0.024,
    0.024,
    0.024,
)

VISION_TRAIN_PD_GAINS_BY_NAME = {
    "left_hip_pitch_joint": (100.0, 2.0),
    "left_hip_roll_joint": (100.0, 2.0),
    "left_hip_yaw_joint": (100.0, 2.0),
    "left_knee_joint": (150.0, 4.0),
    "left_ankle_pitch_joint": (40.0, 2.0),
    "left_ankle_roll_joint": (40.0, 2.0),
    "right_hip_pitch_joint": (100.0, 2.0),
    "right_hip_roll_joint": (100.0, 2.0),
    "right_hip_yaw_joint": (100.0, 2.0),
    "right_knee_joint": (150.0, 4.0),
    "right_ankle_pitch_joint": (40.0, 2.0),
    "right_ankle_roll_joint": (40.0, 2.0),
    "waist_yaw_joint": (200.0, 5.0),
    "waist_roll_joint": (40.0, 5.0),
    "waist_pitch_joint": (40.0, 5.0),
    "left_shoulder_pitch_joint": (40.0, 10.0),
    "left_shoulder_roll_joint": (40.0, 10.0),
    "left_shoulder_yaw_joint": (40.0, 10.0),
    "left_elbow_joint": (40.0, 10.0),
    "left_wrist_roll_joint": (40.0, 10.0),
    "left_wrist_pitch_joint": (40.0, 10.0),
    "left_wrist_yaw_joint": (40.0, 10.0),
    "right_shoulder_pitch_joint": (40.0, 10.0),
    "right_shoulder_roll_joint": (40.0, 10.0),
    "right_shoulder_yaw_joint": (40.0, 10.0),
    "right_elbow_joint": (40.0, 10.0),
    "right_wrist_roll_joint": (40.0, 10.0),
    "right_wrist_pitch_joint": (40.0, 10.0),
    "right_wrist_yaw_joint": (40.0, 10.0),
}

VISION_TRAIN_EFFORT_LIMITS_BY_NAME = {
    "left_hip_pitch_joint": 88.0,
    "left_hip_roll_joint": 139.0,
    "left_hip_yaw_joint": 88.0,
    "left_knee_joint": 139.0,
    "left_ankle_pitch_joint": 25.0,
    "left_ankle_roll_joint": 25.0,
    "right_hip_pitch_joint": 88.0,
    "right_hip_roll_joint": 139.0,
    "right_hip_yaw_joint": 88.0,
    "right_knee_joint": 139.0,
    "right_ankle_pitch_joint": 25.0,
    "right_ankle_roll_joint": 25.0,
    "waist_yaw_joint": 88.0,
    "waist_roll_joint": 25.0,
    "waist_pitch_joint": 25.0,
    "left_shoulder_pitch_joint": 25.0,
    "left_shoulder_roll_joint": 25.0,
    "left_shoulder_yaw_joint": 25.0,
    "left_elbow_joint": 25.0,
    "left_wrist_roll_joint": 25.0,
    "left_wrist_pitch_joint": 5.0,
    "left_wrist_yaw_joint": 5.0,
    "right_shoulder_pitch_joint": 25.0,
    "right_shoulder_roll_joint": 25.0,
    "right_shoulder_yaw_joint": 25.0,
    "right_elbow_joint": 25.0,
    "right_wrist_roll_joint": 25.0,
    "right_wrist_pitch_joint": 5.0,
    "right_wrist_yaw_joint": 5.0,
}

OFFICIAL_PD_GAINS_BY_NAME = {
    "left_hip_pitch_joint": (200.0, 5.0),
    "left_hip_roll_joint": (150.0, 5.0),
    "left_hip_yaw_joint": (150.0, 5.0),
    "left_knee_joint": (200.0, 5.0),
    "left_ankle_pitch_joint": (20.0, 2.0),
    "left_ankle_roll_joint": (20.0, 2.0),
    "right_hip_pitch_joint": (200.0, 5.0),
    "right_hip_roll_joint": (150.0, 5.0),
    "right_hip_yaw_joint": (150.0, 5.0),
    "right_knee_joint": (200.0, 5.0),
    "right_ankle_pitch_joint": (20.0, 2.0),
    "right_ankle_roll_joint": (20.0, 2.0),
    "waist_yaw_joint": (200.0, 5.0),
    "waist_roll_joint": (200.0, 5.0),
    "waist_pitch_joint": (200.0, 5.0),
    "left_shoulder_pitch_joint": (100.0, 2.0),
    "left_shoulder_roll_joint": (100.0, 2.0),
    "left_shoulder_yaw_joint": (50.0, 2.0),
    "left_elbow_joint": (50.0, 2.0),
    "left_wrist_roll_joint": (40.0, 2.0),
    "left_wrist_pitch_joint": (40.0, 2.0),
    "left_wrist_yaw_joint": (40.0, 2.0),
    "right_shoulder_pitch_joint": (100.0, 2.0),
    "right_shoulder_roll_joint": (100.0, 2.0),
    "right_shoulder_yaw_joint": (50.0, 2.0),
    "right_elbow_joint": (50.0, 2.0),
    "right_wrist_roll_joint": (40.0, 2.0),
    "right_wrist_pitch_joint": (40.0, 2.0),
    "right_wrist_yaw_joint": (40.0, 2.0),
    "left_hand_Joint1_1": (800.0, 3.0),
    "left_hand_Joint2_1": (800.0, 3.0),
    "right_hand_Joint1_1": (800.0, 3.0),
    "right_hand_Joint2_1": (800.0, 3.0),
}


def _as_2d_float_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        tensor = torch.as_tensor(value)
    tensor = tensor.to(device=device, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class VisionLocoPolicy:
    policy_obs_dim = 2175
    policy_proprio_dim = 96
    height_dim = 33 * 21 * 3

    def __init__(
        self,
        *,
        config: VisionLocoConfig | None = None,
        policy_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.config = config or VisionLocoConfig.from_env()
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.policy_path = Path(policy_path or self.base_dir / "policy.pt").resolve()
        self.metadata_path = Path(metadata_path or self.base_dir / "sim2sim_metadata.json").resolve()
        memory_path = Path(self.config.height_memory_path)
        if not memory_path.is_absolute():
            memory_path = self.base_dir.parent / memory_path
        self.height_memory = HeightMemory(
            HeightMemoryConfig(
                record_path=memory_path,
                record_start_s=self.config.record_start_s,
                record_end_s=self.config.record_end_s,
                flat_after_end=self.config.flat_after_record_end,
            )
        )
        self._record_times_s: list[float] = []
        self._record_scans: list[torch.Tensor] = []

        with self.metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.train_joint_names = list(metadata["action_joint_names"])
        self.train_default = torch.tensor(
            metadata["default_joint_pos_action_order"],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)
        self.train_action_scale = float(metadata.get("action_scale", self.config.action_scale))
        if self.train_action_scale != self.config.action_scale:
            print(
                "[VisionLocoPolicy] metadata action_scale="
                f"{self.train_action_scale} overrides config action_scale={self.config.action_scale}"
            )

        self._validate_metadata()
        official_index = {name: idx for idx, name in enumerate(OFFICIAL_G1_JOINT_NAMES_33)}
        self.train_to_official = torch.tensor(
            [official_index[name] for name in self.train_joint_names],
            device=self.device,
            dtype=torch.long,
        )
        self.official_default = torch.tensor(OFFICIAL_G1_DEFAULT_33, device=self.device, dtype=torch.float32).view(1, -1)
        train_kp = [VISION_TRAIN_PD_GAINS_BY_NAME[name][0] for name in self.train_joint_names]
        train_kd = [VISION_TRAIN_PD_GAINS_BY_NAME[name][1] for name in self.train_joint_names]
        train_effort = [VISION_TRAIN_EFFORT_LIMITS_BY_NAME[name] for name in self.train_joint_names]
        official_kp = [OFFICIAL_PD_GAINS_BY_NAME[name][0] for name in self.train_joint_names]
        official_kd = [OFFICIAL_PD_GAINS_BY_NAME[name][1] for name in self.train_joint_names]
        self.train_kp = torch.tensor(train_kp, device=self.device, dtype=torch.float32).view(1, -1)
        self.train_kd = torch.tensor(train_kd, device=self.device, dtype=torch.float32).view(1, -1)
        self.train_effort_limits = torch.tensor(train_effort, device=self.device, dtype=torch.float32).view(1, -1)
        self.official_kp_train_order = torch.tensor(official_kp, device=self.device, dtype=torch.float32).view(1, -1)
        self.official_kd_train_order = torch.tensor(official_kd, device=self.device, dtype=torch.float32).view(1, -1)
        self.command = torch.tensor(self.config.command, device=self.device, dtype=torch.float32).view(1, 3)
        self.last_action_train: torch.Tensor | None = None
        self._step = 0
        self._vision_start_time_s: float | None = None
        self._last_debug: dict[str, Any] = {}

        self.policy = torch.jit.load(str(self.policy_path), map_location=self.device)
        self.policy.eval()
        print(
            "[VisionLocoPolicy] "
            f"device={self.device} policy={self.policy_path.name} command={list(self.config.command)}"
        )

    def reset(self) -> None:
        self.last_action_train = None
        self._step = 0
        self._vision_start_time_s = None
        self._last_debug = {}
        self._record_times_s = []
        self._record_scans = []

    @torch.no_grad()
    def predict(self, obs: dict[str, Any], *, elapsed_time_s: float | None = None) -> torch.Tensor:
        proprio = _as_2d_float_tensor(obs["proprio"], device=self.device)
        height_scan = self._get_height_scan(obs, elapsed_time_s=elapsed_time_s, num_envs=proprio.shape[0])
        num_envs = proprio.shape[0]
        action_dim = self._infer_action_dim(proprio)

        if action_dim != len(OFFICIAL_G1_JOINT_NAMES_33):
            raise ValueError(f"Vision loco expects TaskD-G1 33D action space, got action_dim={action_dim}")

        if self.last_action_train is None or self.last_action_train.shape[0] != num_envs:
            self.last_action_train = self._extract_last_action_train(proprio)

        command = self._get_ramped_command(proprio.shape[0], proprio.dtype, elapsed_time_s)
        policy_obs = self._build_policy_obs(proprio, height_scan, command)
        action_train = self.policy(policy_obs)
        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        action_train = action_train.to(device=self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)
        if action_train.shape != (num_envs, len(self.train_joint_names)):
            raise ValueError(
                "Vision loco policy output dim mismatch: "
                f"got {tuple(action_train.shape)}, expected {(num_envs, len(self.train_joint_names))}"
            )
        if torch.isnan(action_train).any() or torch.isinf(action_train).any():
            raise ValueError("Vision loco policy output contains NaN or Inf.")

        self.last_action_train = action_train.detach().clone()
        env_action = self._map_train_action_to_env_action(action_train, proprio)
        self._step += 1
        self._maybe_debug(elapsed_time_s, command, proprio, height_scan, policy_obs, action_train, env_action)
        return env_action

    def _validate_metadata(self) -> None:
        if len(self.train_joint_names) != 29:
            raise ValueError(f"Expected 29 train action joints, got {len(self.train_joint_names)}")
        if self.train_default.shape[-1] != 29:
            raise ValueError(f"Expected 29 train default positions, got {self.train_default.shape[-1]}")
        official_names = set(OFFICIAL_G1_JOINT_NAMES_33)
        missing = [name for name in self.train_joint_names if name not in official_names]
        if missing:
            raise ValueError(f"Vision loco metadata joints are not in official G1 joints: {missing}")
        excluded = [name for name in OFFICIAL_G1_JOINT_NAMES_33 if "hand_Joint" in name]
        if len(excluded) != 4:
            raise ValueError(f"Expected exactly 4 G1 hand joints to be excluded, got {excluded}")

    def _infer_action_dim(self, proprio: torch.Tensor) -> int:
        raw = int(proprio.shape[-1]) - 12
        if raw <= 0 or raw % 3 != 0:
            raise ValueError(f"Cannot infer action_dim from proprio shape {tuple(proprio.shape)}")
        return raw // 3

    def _get_height_scan(self, obs: dict[str, Any], *, elapsed_time_s: float | None, num_envs: int) -> torch.Tensor:
        key = self.config.height_obs_key
        if key in obs:
            value = obs[key]
            if isinstance(value, dict):
                if "height_map" not in value:
                    raise KeyError(f"Observation group '{key}' does not contain 'height_map'.")
                value = value["height_map"]
            height_scan = _as_2d_float_tensor(value, device=self.device)
            self._maybe_record_height_scan(height_scan, elapsed_time_s)
        else:
            if elapsed_time_s is None:
                elapsed_time_s = float(self._step) * float(self.config.nominal_step_dt)
            height_scan = self.height_memory.sample(float(elapsed_time_s), num_envs, self.device)

        if height_scan.shape[-1] != self.height_dim:
            raise ValueError(f"Expected {self.height_dim} height values, got shape {tuple(height_scan.shape)}")
        if torch.isnan(height_scan).any() or torch.isinf(height_scan).any():
            raise ValueError("Vision height scan contains NaN or Inf.")
        return height_scan

    def _maybe_record_height_scan(self, height_scan: torch.Tensor, elapsed_time_s: float | None) -> None:
        if not self.config.record_height_memory or elapsed_time_s is None:
            return
        elapsed_time_s = float(elapsed_time_s)
        if elapsed_time_s < float(self.config.record_start_s) or elapsed_time_s > float(self.config.record_end_s):
            return
        scan = height_scan[0].detach().cpu().clone()
        self._record_times_s.append(elapsed_time_s)
        self._record_scans.append(scan)
        self.height_memory.record(self._record_times_s, self._record_scans)

    def _get_ramped_command(
        self,
        num_envs: int,
        dtype: torch.dtype,
        elapsed_time_s: float | None,
    ) -> torch.Tensor:
        if elapsed_time_s is None:
            elapsed_time_s = float(self._step) * float(self.config.nominal_step_dt)
        elapsed_time_s = float(elapsed_time_s)
        if self._vision_start_time_s is None:
            self._vision_start_time_s = elapsed_time_s

        ramp_time = max(float(self.config.command_ramp_time_s), 0.0)
        if ramp_time <= 0.0:
            alpha = 1.0
        else:
            alpha = max(0.0, min(1.0, (elapsed_time_s - self._vision_start_time_s) / ramp_time))

        command = self.command.to(dtype=dtype) * alpha
        if num_envs > 1:
            command = command.repeat(num_envs, 1)
        return command

    def _build_policy_obs(self, proprio: torch.Tensor, height_scan: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        action_dim = self._infer_action_dim(proprio)
        idx = 0
        idx += 3  # base linear velocity is not part of this policy input.
        base_ang_vel = proprio[:, idx : idx + 3]
        idx += 3
        idx += 3  # use config command instead of env command.
        projected_gravity = proprio[:, idx : idx + 3]
        idx += 3
        joint_pos_rel_all = proprio[:, idx : idx + action_dim]
        idx += action_dim
        joint_vel_all = proprio[:, idx : idx + action_dim]

        joint_pos_abs_all = self.official_default.to(dtype=proprio.dtype) + joint_pos_rel_all
        joint_pos_abs_train = joint_pos_abs_all.index_select(1, self.train_to_official)
        joint_pos_train = joint_pos_abs_train - self.train_default.to(dtype=proprio.dtype)
        joint_vel_train = joint_vel_all.index_select(1, self.train_to_official)
        policy_obs = torch.cat(
            [
                base_ang_vel * 0.2,
                projected_gravity,
                command,
                joint_pos_train,
                joint_vel_train * 0.05,
                self.last_action_train.to(dtype=proprio.dtype),
                height_scan,
            ],
            dim=-1,
        )
        if policy_obs.shape[-1] != self.policy_obs_dim:
            raise ValueError(f"Expected {self.policy_obs_dim} vision policy obs, got {policy_obs.shape[-1]}")
        if torch.isnan(policy_obs).any() or torch.isinf(policy_obs).any():
            raise ValueError("Vision policy obs contains NaN or Inf.")
        return torch.clamp(policy_obs, min=-100.0, max=100.0)

    def _extract_last_action_train(self, proprio: torch.Tensor) -> torch.Tensor:
        action_dim = self._infer_action_dim(proprio)
        actions_all = proprio[:, 12 + 2 * action_dim : 12 + 3 * action_dim]
        action_target_abs = self.official_default.to(dtype=proprio.dtype) + float(self.config.env_action_scale) * actions_all
        action_target_train = action_target_abs.index_select(1, self.train_to_official)
        last_action_train = (action_target_train - self.train_default.to(dtype=proprio.dtype)) / self.train_action_scale
        if torch.isnan(last_action_train).any() or torch.isinf(last_action_train).any():
            raise ValueError("Extracted vision last_action contains NaN or Inf.")
        return torch.clamp(last_action_train, min=-100.0, max=100.0)

    def _map_train_action_to_env_action(self, action_train: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        train_target = self.train_default.to(dtype=action_train.dtype) + self.train_action_scale * action_train
        if self.config.enable_pd_compensation:
            q_all = self.official_default.to(dtype=action_train.dtype) + proprio[:, 12:45].to(dtype=action_train.dtype)
            qd_all = proprio[:, 45:78].to(dtype=action_train.dtype)
            q_train = q_all.index_select(1, self.train_to_official)
            qd_train = qd_all.index_select(1, self.train_to_official)
            official_kp = self.official_kp_train_order.to(dtype=action_train.dtype)
            official_kd = self.official_kd_train_order.to(dtype=action_train.dtype)
            train_torque = self.train_kp.to(dtype=action_train.dtype) * (train_target - q_train)
            train_torque -= self.train_kd.to(dtype=action_train.dtype) * qd_train
            if self.config.clamp_to_train_effort_limits:
                effort = self.train_effort_limits.to(dtype=action_train.dtype)
                train_torque = torch.clamp(train_torque, min=-effort, max=effort)
            train_target = q_train + (train_torque + official_kd * qd_train) / official_kp

        target_33 = self.official_default.to(dtype=action_train.dtype).repeat(action_train.shape[0], 1)
        target_33[:, self.train_to_official] = train_target
        env_action = (target_33 - self.official_default.to(dtype=action_train.dtype)) / float(self.config.env_action_scale)

        if env_action.shape[-1] != len(OFFICIAL_G1_JOINT_NAMES_33):
            raise ValueError(f"Expected 33D env action, got {tuple(env_action.shape)}")
        if torch.isnan(env_action).any() or torch.isinf(env_action).any():
            raise ValueError("Vision env action contains NaN or Inf.")
        return env_action

    def _maybe_debug(
        self,
        elapsed_time_s: float | None,
        command: torch.Tensor,
        proprio: torch.Tensor,
        height_scan: torch.Tensor,
        policy_obs: torch.Tensor,
        action_train: torch.Tensor,
        env_action: torch.Tensor,
    ) -> None:
        interval = int(self.config.debug_interval)
        if interval <= 0 or self._step % interval != 0:
            return
        height_min = float(height_scan.min().detach().cpu())
        height_max = float(height_scan.max().detach().cpu())
        obs_min = float(policy_obs.min().detach().cpu())
        obs_max = float(policy_obs.max().detach().cpu())
        q_rel_abs_max = float(proprio[:, 12:45].abs().max().detach().cpu())
        qd_abs_max = float(proprio[:, 45:78].abs().max().detach().cpu())
        env_action_abs_max = float(env_action.abs().max().detach().cpu())
        self._last_debug = {
            "stage": "vision_loco",
            "step": self._step,
            "elapsed_time_s": elapsed_time_s,
            "command": command[0].detach().cpu().tolist(),
            "target_command": list(self.config.command),
            "height_min": height_min,
            "height_max": height_max,
            "policy_obs_min": obs_min,
            "policy_obs_max": obs_max,
            "q_rel_abs_max": q_rel_abs_max,
            "qd_abs_max": qd_abs_max,
            "env_action_abs_max": env_action_abs_max,
            "action_train_head": action_train[0, :6].detach().cpu().tolist(),
            "env_action_head": env_action[0, :6].detach().cpu().tolist(),
        }
        print(
            "[VisionLocoPolicy] "
            f"stage=vision_loco step={self._step} elapsed={elapsed_time_s} "
            f"cmd={self._last_debug['command']} target_cmd={list(self.config.command)} "
            f"height=[{height_min:.3f}, {height_max:.3f}] "
            f"obs=[{obs_min:.3f}, {obs_max:.3f}] qrel_max={q_rel_abs_max:.3f} "
            f"qd_max={qd_abs_max:.3f} action_max={env_action_abs_max:.3f} "
            f"action_head={self._last_debug['env_action_head']}"
        )

    def get_debug_snapshot(self) -> dict[str, Any]:
        return dict(self._last_debug)
