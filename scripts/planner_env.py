"""PlannerEnv: wraps ATEC TaskD-G1 for high-level planner training.

Architecture:
    Planner (PPO, 50Hz) → nav_cmd → GR00T (GPU TorchScript, 50Hz) → env action → ATEC Env

The planner runs at the same frequency as the low-level policy (decimation=1).
Observations are all GPU tensors. Critic receives GT privileged information.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from tensordict import TensorDict

from scripts.low_level_policy import GrootLowLevelPolicy


class PlannerEnv:
    """High-level planner environment wrapping ATEC TaskD-G1.

    Parameters
    ----------
    env_cfg : object
        ATEC TaskD-G1 environment config (e.g. ``TaskDEnvG1Cfg``).
    low_level : GrootLowLevelPolicy
        GPU-parallelized GR00T low-level policy.
    """

    def __init__(self, env_cfg, low_level: GrootLowLevelPolicy):
        self.env = gym.make("ATEC-TaskD-G1", cfg=env_cfg)
        self.low_level = low_level
        self.device = self.env.device
        self.num_envs = self.env.num_envs

        # Fixed terrain positions (terrain size 12×8, seed=0, single sub-terrain)
        # Robot spawns at terrain-local (1.8, 4.0, 0.0)
        # Pit center at terrain-local (6.0, 4.0, -0.5)
        # Platform/obstacle at terrain-local (6.0, 6.0, ~0.5)
        self._spawn_offset = torch.tensor([1.8, 4.0, 0.0], device=self.device)
        self.pit_offset = torch.tensor([4.2, 0.0, -0.5], device=self.device)
        self.obstacle_offset = torch.tensor([4.2, 2.0, 0.5], device=self.device)

        # State
        self._current_obs: dict | None = None
        self._prev_box_pos_local: torch.Tensor | None = None
        self._box_in_pit_given: torch.Tensor | None = None
        self._last_planner_action: torch.Tensor | None = None

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def max_episode_length(self):
        return self.env.max_episode_length

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    @property
    def cfg(self):
        return self.env.cfg

    # ── Core interface ──────────────────────────────────────────────────────

    def reset(self) -> tuple[TensorDict, dict]:
        obs, info = self.env.reset()
        self._current_obs = obs
        self.low_level.reset(self.num_envs)
        self._box_in_pit_given = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_planner_action = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_box_pos_local = self._box_pos_local()
        return self._make_obs_dict(obs), info

    def step(self, planner_action: torch.Tensor):
        """Step the environment with a planner action.

        Parameters
        ----------
        planner_action : torch.Tensor
            Shape ``(num_envs, 3)`` — [vx, vy, vyaw].

        Returns
        -------
        obs_dict : TensorDict
        reward : torch.Tensor, shape ``(num_envs,)``
        terminated : torch.Tensor, shape ``(num_envs,)``
        truncated : torch.Tensor, shape ``(num_envs,)``
        info : dict
        """
        self._last_planner_action = planner_action.detach().clone()
        nav_cmd = planner_action

        # Decimation=1: one planner step = one env step
        low_action = self.low_level.predict(self._current_obs, nav_cmd)
        obs, _, terminated, truncated, info = self.env.step(low_action)
        self._current_obs = obs

        reward = self._compute_reward(terminated)
        self._prev_box_pos_local = self._box_pos_local()

        return self._make_obs_dict(obs), reward, terminated, truncated, info

    def get_observations(self) -> TensorDict:
        return self._make_obs_dict(self._current_obs)

    def close(self):
        self.env.close()

    # ── Observation construction ────────────────────────────────────────────

    def _make_obs_dict(self, obs: dict) -> TensorDict:
        planner_obs, privileged = self._build_obs(obs)
        return TensorDict(
            {"planner_obs": planner_obs, "privileged": privileged},
            batch_size=[self.num_envs],
        )

    def _build_obs(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Build actor obs (372D) and critic obs (386D = 372 + 14)."""
        proprio = obs["proprio"]  # (N, 111)
        height_scan = obs["extero"]  # (N, 5760)

        # ── Actor obs (372D) ──
        lin_vel = proprio[:, 0:3]
        ang_vel = proprio[:, 3:6]
        gravity = proprio[:, 9:12]
        h_scan = self._compress_height_scan(height_scan)

        planner_obs = torch.cat([
            lin_vel,                  # 3
            ang_vel,                  # 3
            gravity,                  # 3
            self._last_planner_action,  # 3
            h_scan,                   # 360
        ], dim=-1)  # 372D

        # ── Critic privileged (386D = planner_obs + 14D) ──
        robot_pos_local = self._robot_pos_local()
        robot_yaw = self._robot_yaw()
        box_pos_local = self._box_pos_local()
        pit_local = self.pit_offset[:2].unsqueeze(0).expand(self.num_envs, -1)
        obs_local = self.obstacle_offset[:2].unsqueeze(0).expand(self.num_envs, -1)

        box_to_pit = torch.norm(box_pos_local - pit_local, dim=-1, keepdim=True)
        robot_to_box = torch.norm(robot_pos_local - box_pos_local, dim=-1, keepdim=True).clamp(max=3.0)
        speed = torch.norm(lin_vel, dim=-1, keepdim=True)

        box_x = box_pos_local[:, 0:1]
        box_in_pit = (
            ((box_x >= -0.7) & (box_x <= 0.7))
            | ((box_x >= -1.4) & (box_x <= -0.7))
        ).float()

        box_to_obstacle = torch.norm(box_pos_local - obs_local, dim=-1, keepdim=True)

        priv = torch.cat([
            robot_pos_local,   # 3
            box_pos_local,     # 2
            pit_local,         # 2
            obs_local,         # 2
            box_to_pit,        # 1
            robot_to_box,      # 1
            speed,             # 1
            box_in_pit,        # 1
            box_to_obstacle,   # 1
        ], dim=-1)  # 14D

        privileged = torch.cat([planner_obs, priv], dim=-1)  # 386D
        return planner_obs, privileged

    def _compress_height_scan(self, h: torch.Tensor) -> torch.Tensor:
        """(N, 5760) → (N, 360)"""
        h = h.view(-1, 16, 360)
        h = h.view(-1, 4, 4, 360).mean(dim=2)
        h = h.view(-1, 4, 90, 4).mean(dim=-1)
        return h.clamp(-1.0, 1.0).view(-1, 360)

    # ── GT position helpers ─────────────────────────────────────────────────

    def _to_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World position → env-local (relative to env origin + spawn offset)."""
        return pos_w - self.env.scene.env_origins - self._spawn_offset.unsqueeze(0)

    def _robot_pos_local(self) -> torch.Tensor:
        pos_w = self.env.scene["robot"].data.root_pos_w
        return self._to_local(pos_w)

    def _box_pos_local(self) -> torch.Tensor:
        pos_w = self.env.scene["box"].data.root_pos_w
        return self._to_local(pos_w)

    def _robot_yaw(self) -> torch.Tensor:
        quat = self.env.scene["robot"].data.root_quat_w  # (N, 4) wxyz
        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return yaw.unsqueeze(-1)

    # ── Reward computation ──────────────────────────────────────────────────

    def _compute_reward(self, terminated: torch.Tensor) -> torch.Tensor:
        box_local = self._box_pos_local()
        pit_xy = self.pit_offset[:2].unsqueeze(0).expand(self.num_envs, -1)
        obs_xy = self.obstacle_offset[:2].unsqueeze(0).expand(self.num_envs, -1)

        # Box → pit distance
        dist = torch.norm(box_local[:, :2] - pit_xy, dim=-1)
        prev_dist = torch.norm(self._prev_box_pos_local[:, :2] - pit_xy, dim=-1)
        approach = (prev_dist - dist).clamp(min=0)

        # Robot → box distance
        robot_local = self._robot_pos_local()
        robot_box = torch.norm(robot_local[:, :2] - box_local[:, :2], dim=-1).clamp(max=3.0)

        # Box in pit (one-time)
        box_x = box_local[:, 0]
        in_range = ((box_x >= -0.7) & (box_x <= 0.7)) | ((box_x >= -1.4) & (box_x <= -0.7))
        box_in_pit = in_range & (~self._box_in_pit_given)
        self._box_in_pit_given |= in_range

        # Obstacle penalty
        robot_obs_dist = torch.norm(robot_local[:, :2] - obs_xy, dim=-1)
        box_obs_dist = torch.norm(box_local[:, :2] - obs_xy, dim=-1)
        obstacle_pen = torch.clamp(0.5 - robot_obs_dist, min=0) + torch.clamp(0.5 - box_obs_dist, min=0)

        # Fall
        robot_z = self.env.scene["robot"].data.root_pos_w[:, 2] - self.env.scene.env_origins[:, 2]
        fell = robot_z < 0.25

        return (
            -2.0 * dist
            + 1.5 * approach
            - 0.3 * robot_box
            + 50.0 * box_in_pit.float()
            - 1.0 * obstacle_pen
            - 0.02
            + 0.05
            - 10.0 * fell.float() * terminated.float()
        )


class PlannerRslRlWrapper:
    """Adapt PlannerEnv to RSL-RL's VecEnv interface.

    This wrapper satisfies the ``rsl_rl.env.VecEnv`` abstract interface so
    that ``OnPolicyRunner`` can drive training.
    """

    def __init__(self, planner_env: PlannerEnv):
        self.env = planner_env
        self.num_envs = planner_env.num_envs
        self.device = planner_env.device
        self.num_actions = 3  # [vx, vy, vyaw]
        self.num_obs = 372  # planner_obs dim
        self.num_privileged_obs = 386  # planner_obs + privileged
        self.max_episode_length = planner_env.max_episode_length

    @property
    def cfg(self):
        return self.env.cfg

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def get_observations(self) -> TensorDict:
        return self.env.get_observations()

    def reset(self) -> tuple[TensorDict, dict]:
        return self.env.reset()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        obs_dict, reward, terminated, truncated, info = self.env.step(actions)
        dones = (terminated | truncated).long()
        extras = {"time_outs": truncated}
        return obs_dict, reward, dones, extras

    def close(self):
        self.env.close()
