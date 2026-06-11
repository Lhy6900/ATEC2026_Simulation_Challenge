"""Vision locomotion policy adapter for TaskD playback.

This adapter feeds the exported rough-terrain G1 policy with the same 96-D
proprioceptive terms and 33x21x3 elevation-map points used during training.
The elevation map is synthesized from the known TaskD ditch geometry instead of
using a live ray caster.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch


COMP_JOINTS_33 = (
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

COMP_DEFAULT_33 = (
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

TRAIN_PD_GAINS = {
    "left_hip_pitch_joint": (100.0, 2.0),
    "right_hip_pitch_joint": (100.0, 2.0),
    "waist_yaw_joint": (200.0, 5.0),
    "left_hip_roll_joint": (100.0, 2.0),
    "right_hip_roll_joint": (100.0, 2.0),
    "waist_roll_joint": (40.0, 5.0),
    "left_hip_yaw_joint": (100.0, 2.0),
    "right_hip_yaw_joint": (100.0, 2.0),
    "waist_pitch_joint": (40.0, 5.0),
    "left_knee_joint": (150.0, 4.0),
    "right_knee_joint": (150.0, 4.0),
    "left_shoulder_pitch_joint": (40.0, 10.0),
    "right_shoulder_pitch_joint": (40.0, 10.0),
    "left_ankle_pitch_joint": (40.0, 2.0),
    "right_ankle_pitch_joint": (40.0, 2.0),
    "left_shoulder_roll_joint": (40.0, 10.0),
    "right_shoulder_roll_joint": (40.0, 10.0),
    "left_ankle_roll_joint": (40.0, 2.0),
    "right_ankle_roll_joint": (40.0, 2.0),
    "left_shoulder_yaw_joint": (40.0, 10.0),
    "right_shoulder_yaw_joint": (40.0, 10.0),
    "left_elbow_joint": (40.0, 10.0),
    "right_elbow_joint": (40.0, 10.0),
    "left_wrist_roll_joint": (40.0, 10.0),
    "right_wrist_roll_joint": (40.0, 10.0),
    "left_wrist_pitch_joint": (40.0, 10.0),
    "right_wrist_pitch_joint": (40.0, 10.0),
    "left_wrist_yaw_joint": (40.0, 10.0),
    "right_wrist_yaw_joint": (40.0, 10.0),
}

TASKD_PD_GAINS = {
    "left_hip_pitch_joint": (150.0, 2.0),
    "left_hip_roll_joint": (150.0, 2.0),
    "left_hip_yaw_joint": (150.0, 2.0),
    "left_knee_joint": (200.0, 4.0),
    "left_ankle_pitch_joint": (40.0, 2.0),
    "left_ankle_roll_joint": (40.0, 2.0),
    "right_hip_pitch_joint": (150.0, 2.0),
    "right_hip_roll_joint": (150.0, 2.0),
    "right_hip_yaw_joint": (150.0, 2.0),
    "right_knee_joint": (200.0, 4.0),
    "right_ankle_pitch_joint": (40.0, 2.0),
    "right_ankle_roll_joint": (40.0, 2.0),
    "waist_yaw_joint": (250.0, 5.0),
    "waist_roll_joint": (250.0, 5.0),
    "waist_pitch_joint": (250.0, 5.0),
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
}

PLANNER_PIT_TARGET_X_RANGE = (-0.5, 0.5)
PLANNER_PIT_TARGET_Y_RANGE = (-2.0, -0.5)


class VisionLocoPolicy:
    """Run the exported G1 vision locomotion policy in the TaskD environment."""

    MAP_L = 33
    MAP_W = 21
    MAP_RESOLUTION = 0.05
    MAP_SIZE_X = 1.6
    MAP_SIZE_Y = 1.0
    TRAIN_ACTION_SCALE = 0.25
    TASKD_ACTION_SCALE = 0.5
    BASE_ANG_VEL_SCALE = 0.2
    JOINT_VEL_SCALE = 0.05

    def __init__(
        self,
        policy_path: str | Path,
        metadata_path: str | Path,
        *,
        device: str | torch.device,
        forward_velocity: float = 0.6,
        yaw_kp: float = 1.0,
        ditch_center_xy: tuple[float, float] | None = None,
        ditch_yaw: float = math.pi / 2.0,
        ditch_width: float = 0.95,
        ditch_length: float = 7.0,
        ditch_depth: float = 1.0,
        height_map_cfg: dict[str, Any] | None = None,
        action_compensation_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.policy = torch.jit.load(str(policy_path), map_location=self.device).eval()
        self.forward_velocity = float(forward_velocity)
        self.yaw_kp = float(yaw_kp)
        self.ditch_center_xy = (
            None
            if ditch_center_xy is None
            else torch.tensor(ditch_center_xy, dtype=torch.float32, device=self.device)
        )
        self.ditch_yaw = float(ditch_yaw)
        self.ditch_width = float(ditch_width)
        self.ditch_length = float(ditch_length)
        self.ditch_depth = float(ditch_depth)
        self.height_map_cfg = height_map_cfg or {}
        action_compensation_cfg = action_compensation_cfg or {}
        self.action_compensation_enabled = bool(action_compensation_cfg.get("enabled", False))
        self.action_compensation_blend = float(action_compensation_cfg.get("blend", 1.0))
        max_delta = action_compensation_cfg.get("max_extra_target_delta", 0.5)
        self.action_compensation_max_delta = None if max_delta is None else float(max_delta)
        force_ground_cfg = self._cfg_section("force_ground_behind")
        after_gap_cfg = self._cfg_section("after_gap")
        raise_cfg = self._cfg_section("in_gap_raise_forward_ground")
        box_fill_cfg = self._cfg_section("box_fill")
        self.force_ground_behind_enabled = bool(force_ground_cfg.get("enabled", False))
        self.force_ground_x_max = float(force_ground_cfg.get("x_max", 0.0))
        self.force_ground_y_min = float(force_ground_cfg.get("y_min", -0.4))
        self.force_ground_y_max = float(force_ground_cfg.get("y_max", 0.4))
        self.after_gap_enabled = bool(after_gap_cfg.get("enabled", False))
        self.after_gap_margin = float(after_gap_cfg.get("margin", 0.0))
        after_gap_vel_command = after_gap_cfg.get("vel_command")
        if after_gap_vel_command is None:
            self.after_gap_vel_command = None
        else:
            if not isinstance(after_gap_vel_command, (list, tuple)) or len(after_gap_vel_command) != 3:
                raise ValueError("height_map.after_gap.vel_command must be null or a 3-element list [vx, vy, wz]")
            self.after_gap_vel_command = torch.tensor(
                [float(item) for item in after_gap_vel_command],
                dtype=torch.float32,
                device=self.device,
            ).view(1, 3)
        self.raise_forward_ground_enabled = bool(raise_cfg.get("enabled", False))
        self.raise_forward_ground_x_min = float(raise_cfg.get("x_min", 0.0))
        self.raise_forward_ground_height = float(raise_cfg.get("height", 0.2))
        self.box_fill_enabled = bool(box_fill_cfg.get("enabled", True))
        self.box_fill_size_x = float(box_fill_cfg.get("size_x", 0.8))
        self.box_fill_size_y = float(box_fill_cfg.get("size_y", 1.0))
        self.box_fill_height = float(box_fill_cfg.get("height", 0.6))
        self.box_fill_margin = float(box_fill_cfg.get("margin", 0.05))
        self.box_fill_min_center_z = float(box_fill_cfg.get("min_center_z", -1.3))
        self.box_fill_max_center_z = float(box_fill_cfg.get("max_center_z", 0.2))

        metadata = json.loads(Path(metadata_path).read_text())
        train_joints = tuple(metadata["action_joint_names"])
        if len(train_joints) != 29:
            raise ValueError(f"Vision locomotion metadata must describe 29 action joints, got {len(train_joints)}")

        self.train_joints = train_joints
        self.comp_to_train = torch.tensor(
            [COMP_JOINTS_33.index(name) for name in train_joints],
            dtype=torch.long,
            device=self.device,
        )
        self.train_to_comp = torch.tensor(
            [train_joints.index(name) for name in COMP_JOINTS_33[:29]],
            dtype=torch.long,
            device=self.device,
        )
        self.train_kp_comp = torch.tensor(
            [TRAIN_PD_GAINS[name][0] for name in COMP_JOINTS_33[:29]],
            dtype=torch.float32,
            device=self.device,
        ).view(1, -1)
        self.train_kd_comp = torch.tensor(
            [TRAIN_PD_GAINS[name][1] for name in COMP_JOINTS_33[:29]],
            dtype=torch.float32,
            device=self.device,
        ).view(1, -1)
        self.taskd_kp_comp = torch.tensor(
            [TASKD_PD_GAINS[name][0] for name in COMP_JOINTS_33[:29]],
            dtype=torch.float32,
            device=self.device,
        ).view(1, -1)
        self.taskd_kd_comp = torch.tensor(
            [TASKD_PD_GAINS[name][1] for name in COMP_JOINTS_33[:29]],
            dtype=torch.float32,
            device=self.device,
        ).view(1, -1)

        self.train_default = torch.tensor(
            metadata["default_joint_pos_action_order"],
            dtype=torch.float32,
            device=self.device,
        ).view(1, -1)
        self.comp_default = torch.tensor(COMP_DEFAULT_33, dtype=torch.float32, device=self.device).view(1, -1)
        self._grid_xy = self._build_grid_xy()
        self._last_policy_action: torch.Tensor | None = None
        self._gap_crossed: torch.Tensor | None = None
        self._torso_body_id: int | None = None
        self._torso_body_lookup_done = False

    def _cfg_section(self, name: str) -> dict[str, Any]:
        section = self.height_map_cfg.get(name, {})
        if section is None:
            return {}
        if not isinstance(section, dict):
            raise TypeError(f"height_map.{name} must be a mapping")
        return section

    def reset(self, num_envs: int | None = None) -> None:
        if num_envs is None:
            self._last_policy_action = None
            self._gap_crossed = None
        else:
            self._last_policy_action = torch.zeros(num_envs, 29, dtype=torch.float32, device=self.device)
            self._gap_crossed = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def predict(
        self,
        env,
        obs: dict,
        command_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a TaskD 33-D action from a raw environment observation."""
        proprio = obs["proprio"].to(self.device)
        num_envs = proprio.shape[0]
        if self._last_policy_action is None or self._last_policy_action.shape[0] != num_envs:
            self.reset(num_envs)

        if command_override is None:
            command = self._forward_command(env, proprio)
        else:
            command = command_override.to(device=self.device, dtype=torch.float32)
            if command.ndim == 1:
                command = command.unsqueeze(0).expand(num_envs, -1)
        height_scan = self._synthetic_elevation_map(env)
        policy_obs = self._build_policy_obs(proprio, command, height_scan)
        policy_action = self.policy(policy_obs)
        if policy_action.ndim == 1:
            policy_action = policy_action.unsqueeze(0)
        policy_action = policy_action.to(dtype=torch.float32, device=self.device)

        env_action = self._map_policy_action(policy_action, proprio)
        self._last_policy_action = policy_action.detach()
        return env_action

    def _build_policy_obs(
        self,
        proprio: torch.Tensor,
        command: torch.Tensor,
        height_scan: torch.Tensor,
    ) -> torch.Tensor:
        base_ang_vel = proprio[:, 3:6] * self.BASE_ANG_VEL_SCALE
        projected_gravity = proprio[:, 9:12]
        joint_pos_rel_comp = proprio[:, 12:45]
        joint_vel_comp = proprio[:, 45:78]

        comp_default = self.comp_default.expand(proprio.shape[0], -1)
        joint_pos_abs_comp = comp_default[:, :29] + joint_pos_rel_comp[:, :29]
        joint_pos_abs_train = joint_pos_abs_comp[:, self.comp_to_train]
        joint_pos_rel_train = joint_pos_abs_train - self.train_default
        joint_vel_train = joint_vel_comp[:, :29][:, self.comp_to_train] * self.JOINT_VEL_SCALE

        return torch.cat(
            [
                base_ang_vel,
                projected_gravity,
                command,
                joint_pos_rel_train,
                joint_vel_train,
                self._last_policy_action,
                height_scan,
            ],
            dim=-1,
        ).clamp_(-100.0, 100.0)

    def _map_policy_action(self, policy_action: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        num_envs = policy_action.shape[0]
        train_target = self.train_default.expand(num_envs, -1) + self.TRAIN_ACTION_SCALE * policy_action
        comp_target_29 = train_target[:, self.train_to_comp]
        comp_default = self.comp_default.expand(num_envs, -1)
        if self.action_compensation_enabled:
            comp_target_29 = self._compensate_pd_target(comp_target_29, proprio, comp_default)

        env_action = torch.zeros(num_envs, 33, dtype=torch.float32, device=self.device)
        env_action[:, :29] = (comp_target_29 - comp_default[:, :29]) / self.TASKD_ACTION_SCALE
        return env_action

    def _compensate_pd_target(
        self,
        train_equivalent_target_comp: torch.Tensor,
        proprio: torch.Tensor,
        comp_default: torch.Tensor,
    ) -> torch.Tensor:
        q_comp = comp_default[:, :29] + proprio[:, 12:41]
        qd_comp = proprio[:, 45:74]

        # Match implicit PD torque:
        # kp_task * (q_cmd_task - q) - kd_task * qdot
        #   ~= kp_train * (q_target_train - q) - kd_train * qdot
        task_equivalent_target = (
            q_comp
            + (self.train_kp_comp / self.taskd_kp_comp) * (train_equivalent_target_comp - q_comp)
            + ((self.taskd_kd_comp - self.train_kd_comp) / self.taskd_kp_comp) * qd_comp
        )
        compensation_delta = task_equivalent_target - train_equivalent_target_comp
        if self.action_compensation_max_delta is not None:
            compensation_delta = compensation_delta.clamp(
                min=-self.action_compensation_max_delta,
                max=self.action_compensation_max_delta,
            )
        blend = max(0.0, min(1.0, self.action_compensation_blend))
        return train_equivalent_target_comp + blend * compensation_delta

    def _forward_command(self, env, proprio: torch.Tensor) -> torch.Tensor:
        robot = env.unwrapped.scene["robot"]
        yaw = self._yaw_from_quat(robot.data.root_quat_w.to(self.device))
        yaw_rate = torch.clamp(-self.yaw_kp * yaw, min=-1.0, max=1.0)
        command = torch.zeros(proprio.shape[0], 3, dtype=torch.float32, device=self.device)
        command[:, 0] = self.forward_velocity
        command[:, 2] = yaw_rate
        if self.after_gap_vel_command is not None:
            self._update_gap_crossed(self._ditch_pose_lidar(env))
            if self._gap_crossed is not None:
                after_gap_command = self.after_gap_vel_command.expand(proprio.shape[0], -1)
                command = torch.where(self._gap_crossed[:, None], after_gap_command, command)
        return command

    def _synthetic_elevation_map(self, env) -> torch.Tensor:
        local_xy, z, _, _, _ = self._synthetic_elevation_map_detail(env)
        scan = torch.cat([local_xy, z.unsqueeze(-1)], dim=-1)
        return scan.reshape(local_xy.shape[0], self.MAP_W * self.MAP_L * 3)

    def print_synthetic_elevation_map(
        self,
        env,
        *,
        step: int | None = None,
        env_id: int = 0,
        window: int = 9,
    ) -> None:
        """Print the center crop of the synthetic elevation map used by the policy."""
        local_xy, z, world_xy, terrain_z, in_ditch = self._synthetic_elevation_map_detail(env)
        ditch_pose_l = self._ditch_pose_lidar(env)
        ditch_center_xy_w, _, ditch_width, ditch_length, ditch_depth = self._ditch_geometry_w(env)
        env_id = max(0, min(int(env_id), local_xy.shape[0] - 1))
        scanner_pos, scanner_quat = self._scanner_pose_w(env)
        yaw = self._yaw_from_quat(scanner_quat)

        prefix = "[height_map]"
        if step is not None:
            prefix = f"[height_map step={step}]"
        print(
            f"{prefix} env={env_id} robot_base_w="
            f"({scanner_pos[env_id, 0].item():.4f}, {scanner_pos[env_id, 1].item():.4f}, {scanner_pos[env_id, 2].item():.4f}) "
            f"yaw={yaw[env_id].item():.4f} ditch_lidar="
            f"({ditch_pose_l[env_id, 0].item():.4f}, {ditch_pose_l[env_id, 1].item():.4f}, "
            f"{math.degrees(ditch_pose_l[env_id, 2].item()):.2f}deg) "
            f"ditch_center_w=({ditch_center_xy_w[env_id, 0].item():.4f}, "
            f"{ditch_center_xy_w[env_id, 1].item():.4f}) "
            f"ditch_width={ditch_width[env_id, 0].item():.4f} "
            f"ditch_length={ditch_length[env_id, 0].item():.4f} "
            f"ditch_depth={ditch_depth[env_id, 0].item():.4f}"
        )
        base_in_ditch = self._base_in_ditch(
            ditch_pose_l,
            ditch_width=ditch_width,
            ditch_length=ditch_length,
        )
        gap_crossed = (
            False
            if self._gap_crossed is None
            else bool(self._gap_crossed[env_id].detach().cpu().item())
        )
        z_env = z[env_id]
        print(
            f"{prefix} stats z_min={z_env.min().item():.4f} z_max={z_env.max().item():.4f} "
            f"flat_expected={(-scanner_pos[env_id, 2]).clamp(min=-1.2, max=0.0).item():.4f} "
            f"in_ditch_points={int(in_ditch[env_id].sum().item())}/{in_ditch.shape[1]} "
            f"base_in_ditch={bool(base_in_ditch[env_id].detach().cpu().item())} "
            f"gap_crossed={gap_crossed}"
        )
        window = max(1, min(int(window), self.MAP_L, self.MAP_W))
        if window % 2 == 0:
            window -= 1

        row_center = self.MAP_W // 2
        col_center = self.MAP_L // 2
        row_start = row_center - window // 2
        col_start = col_center - window // 2
        rows = range(row_start, row_start + window)
        cols = range(col_start, col_start + window)

        local_xy_grid = local_xy[env_id].reshape(self.MAP_W, self.MAP_L, 2).detach().cpu()
        z_grid = z[env_id].reshape(self.MAP_W, self.MAP_L).detach().cpu()

        print(f"{prefix} center {window}x{window} z_policy_m table")
        print(f"{prefix} each cell: coord=(row y_base_m, column x_base_m), value=z_policy_m")
        header = "y_base_m \\ x_base_m " + " ".join(
            f"{local_xy_grid[row_center, col, 0].item():>7.2f}" for col in cols
        )
        print(header)
        for row in rows:
            values = " ".join(f"{z_grid[row, col].item():>7.2f}" for col in cols)
            print(f"{local_xy_grid[row, col_center, 1].item():>18.2f} {values}")

    def _synthetic_elevation_map_detail(
        self, env
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scanner_pos, scanner_quat = self._scanner_pose_w(env)
        scanner_yaw = self._yaw_from_quat(scanner_quat)

        local_xy = self._grid_xy.unsqueeze(0).expand(scanner_pos.shape[0], -1, -1)
        ditch_pose_l = self._ditch_pose_lidar(env)
        _, _, ditch_width, ditch_length, ditch_depth = self._ditch_geometry_w(env)
        self._update_gap_crossed(ditch_pose_l)
        terrain_z, in_ditch = self._terrain_height_from_lidar_pose(
            local_xy,
            ditch_pose_l,
            ditch_width=ditch_width,
            ditch_length=ditch_length,
            ditch_depth=ditch_depth,
            return_mask=True,
        )
        terrain_z = self._apply_box_fill_height(env, local_xy, terrain_z, ditch_pose_l, ditch_width, ditch_length)
        z = (terrain_z - scanner_pos[:, 2:3]).clamp(min=-1.2, max=0.0)

        cos_yaw = torch.cos(scanner_yaw).unsqueeze(-1)
        sin_yaw = torch.sin(scanner_yaw).unsqueeze(-1)
        world_x = scanner_pos[:, 0:1] + cos_yaw * local_xy[..., 0] - sin_yaw * local_xy[..., 1]
        world_y = scanner_pos[:, 1:2] + sin_yaw * local_xy[..., 0] + cos_yaw * local_xy[..., 1]
        world_xy = torch.stack([world_x, world_y], dim=-1)
        return local_xy, z, world_xy, terrain_z, in_ditch

    def _terrain_height_from_lidar_pose(
        self,
        points_l: torch.Tensor,
        ditch_pose_l: torch.Tensor,
        *,
        ditch_width: torch.Tensor | float | None = None,
        ditch_length: torch.Tensor | float | None = None,
        ditch_depth: torch.Tensor | float | None = None,
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if ditch_width is None:
            ditch_width = self.ditch_width
        if ditch_length is None:
            ditch_length = self.ditch_length
        if ditch_depth is None:
            ditch_depth = self.ditch_depth
        ditch_width = self._as_env_column(ditch_width, points_l.shape[0], points_l.device, points_l.dtype)
        ditch_length = self._as_env_column(ditch_length, points_l.shape[0], points_l.device, points_l.dtype)
        ditch_depth = self._as_env_column(ditch_depth, points_l.shape[0], points_l.device, points_l.dtype)

        dx = points_l[..., 0] - ditch_pose_l[:, 0:1]
        dy = points_l[..., 1] - ditch_pose_l[:, 1:2]
        yaw = ditch_pose_l[:, 2:3]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        along = cos_yaw * dx + sin_yaw * dy
        across = -sin_yaw * dx + cos_yaw * dy
        in_ditch = (along.abs() <= 0.5 * ditch_length) & (across.abs() <= 0.5 * ditch_width)
        if self.force_ground_behind_enabled:
            force_ground = (
                (points_l[..., 0] <= self.force_ground_x_max)
                & (points_l[..., 1] >= self.force_ground_y_min)
                & (points_l[..., 1] <= self.force_ground_y_max)
            )
            in_ditch = in_ditch & (~force_ground)
        if self.after_gap_enabled and self._gap_crossed is not None:
            in_ditch = in_ditch & (~self._gap_crossed[:, None])
        terrain_z = torch.where(in_ditch, -ditch_depth.expand_as(dx), torch.zeros_like(dx))

        if self.raise_forward_ground_enabled:
            base_in_ditch = self._base_in_ditch(
                ditch_pose_l,
                ditch_width=ditch_width,
                ditch_length=ditch_length,
            )
            raised_ground = (
                base_in_ditch[:, None]
                & (points_l[..., 0] > self.raise_forward_ground_x_min)
                & (~in_ditch)
            )
            terrain_z = torch.where(
                raised_ground,
                terrain_z + self.raise_forward_ground_height,
                terrain_z,
            )
        if return_mask:
            return terrain_z, in_ditch
        return terrain_z

    def _apply_box_fill_height(
        self,
        env,
        points_l: torch.Tensor,
        terrain_z: torch.Tensor,
        ditch_pose_l: torch.Tensor,
        ditch_width: torch.Tensor,
        ditch_length: torch.Tensor,
    ) -> torch.Tensor:
        """Raise points covered by the pushed box to the box top height.

        The trained scanner observes terrain coordinates, so this keeps the
        same post-processing as ``elevation_map`` and only changes the
        synthesized terrain geometry after the box is physically in the ditch.
        """
        if not self.box_fill_enabled:
            return terrain_z
        try:
            box = env.unwrapped.scene["box"]
        except Exception:
            return terrain_z

        box_pos_w = box.data.root_pos_w.to(device=self.device, dtype=points_l.dtype)
        box_quat_w = box.data.root_quat_w.to(device=self.device, dtype=points_l.dtype)
        scanner_pos, scanner_quat = self._scanner_pose_w(env)
        scanner_yaw = self._yaw_from_quat(scanner_quat).to(dtype=points_l.dtype)
        box_yaw_w = self._yaw_from_quat(box_quat_w)
        box_yaw_l = self._wrap_to_pi(box_yaw_w - scanner_yaw)

        delta_w = box_pos_w[:, :2] - scanner_pos[:, :2]
        cos_yaw = torch.cos(scanner_yaw)
        sin_yaw = torch.sin(scanner_yaw)
        box_x_l = cos_yaw * delta_w[:, 0] + sin_yaw * delta_w[:, 1]
        box_y_l = -sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1]

        ditch_yaw = ditch_pose_l[:, 2]
        dx_ditch = box_x_l - ditch_pose_l[:, 0]
        dy_ditch = box_y_l - ditch_pose_l[:, 1]
        box_along_ditch = torch.cos(ditch_yaw) * dx_ditch + torch.sin(ditch_yaw) * dy_ditch
        box_across_ditch = -torch.sin(ditch_yaw) * dx_ditch + torch.cos(ditch_yaw) * dy_ditch
        box_in_ditch = (
            (box_pos_w[:, 2] >= self.box_fill_min_center_z)
            & (box_pos_w[:, 2] <= self.box_fill_max_center_z)
            & (box_along_ditch.abs() <= 0.5 * ditch_length[:, 0] + 0.5 * self.box_fill_size_y)
            & (box_across_ditch.abs() <= 0.5 * ditch_width[:, 0] + 0.5 * self.box_fill_size_x)
        )
        if not bool(box_in_ditch.any().item()):
            return terrain_z

        dx = points_l[..., 0] - box_x_l[:, None]
        dy = points_l[..., 1] - box_y_l[:, None]
        cos_box = torch.cos(box_yaw_l)[:, None]
        sin_box = torch.sin(box_yaw_l)[:, None]
        box_local_x = cos_box * dx + sin_box * dy
        box_local_y = -sin_box * dx + cos_box * dy
        on_box = (
            box_in_ditch[:, None]
            & (box_local_x.abs() <= 0.5 * self.box_fill_size_x + self.box_fill_margin)
            & (box_local_y.abs() <= 0.5 * self.box_fill_size_y + self.box_fill_margin)
        )
        box_top_z = (box_pos_w[:, 2:3] + 0.5 * self.box_fill_height).expand_as(terrain_z)
        return torch.where(on_box, torch.maximum(terrain_z, box_top_z), terrain_z)

    def _base_in_ditch(
        self,
        ditch_pose_l: torch.Tensor,
        *,
        ditch_width: torch.Tensor | float | None = None,
        ditch_length: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Return envs whose robot base is inside the ditch rectangle."""
        if ditch_width is None:
            ditch_width = self.ditch_width
        if ditch_length is None:
            ditch_length = self.ditch_length
        ditch_width = self._as_env_column(
            ditch_width, ditch_pose_l.shape[0], ditch_pose_l.device, ditch_pose_l.dtype
        )
        ditch_length = self._as_env_column(
            ditch_length, ditch_pose_l.shape[0], ditch_pose_l.device, ditch_pose_l.dtype
        )
        yaw = ditch_pose_l[:, 2]
        dx = -ditch_pose_l[:, 0]
        dy = -ditch_pose_l[:, 1]
        along = torch.cos(yaw) * dx + torch.sin(yaw) * dy
        across = -torch.sin(yaw) * dx + torch.cos(yaw) * dy
        return (along.abs() <= 0.5 * ditch_length[:, 0]) & (across.abs() <= 0.5 * ditch_width[:, 0])

    def _update_gap_crossed(self, ditch_pose_l: torch.Tensor) -> None:
        """Latch envs that have moved past the gap's far edge."""
        if self._gap_crossed is None or self._gap_crossed.shape[0] != ditch_pose_l.shape[0]:
            self._gap_crossed = torch.zeros(ditch_pose_l.shape[0], dtype=torch.bool, device=self.device)

        yaw = ditch_pose_l[:, 2]
        dx = -ditch_pose_l[:, 0]
        dy = -ditch_pose_l[:, 1]
        robot_across = -torch.sin(yaw) * dx + torch.cos(yaw) * dy
        crossed_now = robot_across < (-0.5 * self.ditch_width - self.after_gap_margin)
        self._gap_crossed |= crossed_now

    def _ditch_pose_lidar(self, env) -> torch.Tensor:
        """Return ditch x/y/yaw in the base yaw frame described by lidar_use_k7.md."""
        scanner_pos, scanner_quat = self._scanner_pose_w(env)
        root_yaw = self._yaw_from_quat(scanner_quat)
        ditch_center_xy_w, ditch_yaw_w, _, _, _ = self._ditch_geometry_w(env)

        delta_w = ditch_center_xy_w - scanner_pos[:, :2]
        cos_yaw = torch.cos(root_yaw)
        sin_yaw = torch.sin(root_yaw)
        ditch_x_l = cos_yaw * delta_w[:, 0] + sin_yaw * delta_w[:, 1]
        ditch_y_l = -sin_yaw * delta_w[:, 0] + cos_yaw * delta_w[:, 1]
        ditch_yaw_l = self._wrap_to_pi(ditch_yaw_w - root_yaw)
        return torch.stack([ditch_x_l, ditch_y_l, ditch_yaw_l], dim=-1)

    def _ditch_geometry_w(
        self, env
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_envs = env.unwrapped.scene["robot"].data.root_pos_w.shape[0]
        device = self.device
        dtype = torch.float32
        if self.ditch_center_xy is not None:
            center_xy = self.ditch_center_xy.unsqueeze(0).expand(num_envs, -1)
            yaw = torch.full((num_envs,), self.ditch_yaw, dtype=dtype, device=device)
            width = torch.full((num_envs, 1), self.ditch_width, dtype=dtype, device=device)
            length = torch.full((num_envs, 1), self.ditch_length, dtype=dtype, device=device)
            depth = torch.full((num_envs, 1), self.ditch_depth, dtype=dtype, device=device)
            return center_xy, yaw, width, length, depth

        scene = env.unwrapped.scene
        terrain = getattr(scene, "terrain", None)
        terrain_generator = getattr(terrain, "terrain_generator", None)
        cfg = getattr(terrain_generator, "cfg", None)
        sub_terrains = getattr(cfg, "sub_terrains", {}) if cfg is not None else {}
        pit_cfg = sub_terrains.get("pit_and_platform") if hasattr(sub_terrains, "get") else None
        if cfg is None or pit_cfg is None:
            center_xy = self._ditch_center_xy_w(env)
            yaw = torch.full((num_envs,), self.ditch_yaw, dtype=dtype, device=device)
            width = torch.full((num_envs, 1), self.ditch_width, dtype=dtype, device=device)
            length = torch.full((num_envs, 1), self.ditch_length, dtype=dtype, device=device)
            depth = torch.full((num_envs, 1), self.ditch_depth, dtype=dtype, device=device)
            return center_xy, yaw, width, length, depth

        size = getattr(cfg, "size", (12.0, 8.0))
        size_x = float(size[0])
        size_y = float(size[1])
        border_width = float(getattr(pit_cfg, "border_width", 1.0))
        pit_width_range = getattr(pit_cfg, "pit_width_range", (self.ditch_width, self.ditch_width))
        pit_depth = float(getattr(pit_cfg, "pit_depth", self.ditch_depth))

        platform_center_y = size_y * 0.75
        platform_extent_y = size_y * 0.5 - border_width
        pit_min_y = 0.5 * border_width
        platform_min_y = platform_center_y - 0.5 * platform_extent_y
        open_ditch_center_y = 0.5 * (pit_min_y + platform_min_y)
        open_ditch_length = max(platform_min_y - pit_min_y, 0.05)
        terrain_origin_y = size_y * 0.5
        center_local = torch.tensor(
            [size_x * 0.5 - size_x * 0.15, open_ditch_center_y - terrain_origin_y],
            dtype=dtype,
            device=device,
        )
        env_origins = getattr(scene, "env_origins", None)
        if env_origins is None:
            env_origin_xy = torch.zeros(num_envs, 2, dtype=dtype, device=device)
        else:
            env_origin_xy = env_origins.to(device=device, dtype=dtype)[:, :2]
        center_xy = env_origin_xy + center_local.unsqueeze(0)
        yaw = torch.full((num_envs,), self.ditch_yaw, dtype=dtype, device=device)
        width_value = float(pit_width_range[0])
        width = torch.full((num_envs, 1), width_value, dtype=dtype, device=device)
        length = torch.full((num_envs, 1), open_ditch_length, dtype=dtype, device=device)
        depth = torch.full((num_envs, 1), pit_depth, dtype=dtype, device=device)
        return center_xy, yaw, width, length, depth

    def _ditch_center_xy_w(self, env) -> torch.Tensor:
        """Return the TaskD open ditch center in world XY for each env.

        Keep this default aligned with PlannerEnv's box-in-pit target rectangle,
        because the planner checkpoint was trained to push the box into that
        world-frame rectangle.
        """
        num_envs = env.unwrapped.scene["robot"].data.root_pos_w.shape[0]
        if self.ditch_center_xy is not None:
            return self.ditch_center_xy.unsqueeze(0).expand(num_envs, -1)

        center_local = torch.tensor(
            [
                0.5 * (PLANNER_PIT_TARGET_X_RANGE[0] + PLANNER_PIT_TARGET_X_RANGE[1]),
                0.5 * (PLANNER_PIT_TARGET_Y_RANGE[0] + PLANNER_PIT_TARGET_Y_RANGE[1]),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return center_local.unsqueeze(0).expand(num_envs, -1)

    @staticmethod
    def _as_env_column(
        value: torch.Tensor | float,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=device, dtype=dtype)
            if tensor.ndim == 0:
                return tensor.reshape(1, 1).expand(num_envs, 1)
            if tensor.ndim == 1:
                return tensor.reshape(-1, 1)
            return tensor
        return torch.full((num_envs, 1), float(value), dtype=dtype, device=device)

    def _build_grid_xy(self) -> torch.Tensor:
        x = torch.linspace(
            -0.5 * self.MAP_SIZE_X,
            0.5 * self.MAP_SIZE_X,
            self.MAP_L,
            dtype=torch.float32,
            device=self.device,
        )
        y = torch.linspace(
            -0.5 * self.MAP_SIZE_Y,
            0.5 * self.MAP_SIZE_Y,
            self.MAP_W,
            dtype=torch.float32,
            device=self.device,
        )
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(self.MAP_W * self.MAP_L, 2)

    def _scanner_pose_w(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        """Pose of the torso/base scanner frame used by the loco policy.

        The trained loco policy's RayCaster is attached to ``torso_link`` with
        ``ray_alignment="yaw"``. Its 20 m z offset only moves ray starts; the
        observation still subtracts ``sensor.data.pos_w``, i.e. the torso/base
        link position.
        """
        robot = env.unwrapped.scene["robot"]
        data = robot.data
        body_pos_w = getattr(data, "body_pos_w", None)
        body_quat_w = getattr(data, "body_quat_w", None)
        torso_id = self._torso_body_index(robot)
        if body_pos_w is not None and body_quat_w is not None and torso_id is not None:
            return body_pos_w[:, torso_id, :].to(self.device), body_quat_w[:, torso_id, :].to(self.device)
        return data.root_pos_w.to(self.device), data.root_quat_w.to(self.device)

    def _torso_body_index(self, robot) -> int | None:
        if self._torso_body_lookup_done:
            return self._torso_body_id
        self._torso_body_lookup_done = True
        try:
            body_ids, _ = robot.find_bodies("torso_link", preserve_order=True)
        except Exception:
            body_ids = []
        if len(body_ids) > 0:
            self._torso_body_id = int(body_ids[0])
        return self._torso_body_id

    @staticmethod
    def _yaw_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
        qw, qx, qy, qz = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
        return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))
