"""PlannerEnv: wraps ATEC TaskD-G1 for high-level planner training.

Architecture:
    Planner (PPO, 50Hz) → nav_cmd → GR00T (GPU TorchScript, 50Hz) → env action → ATEC Env

The planner runs at the same frequency as the low-level policy (decimation=1).
Observations are all GPU tensors. Critic receives GT privileged information.

PlannerEnv inherits from gym.Wrapper so that .unwrapped correctly resolves
to the underlying ManagerBasedRLEnv, matching IsaacLab's RslRlVecEnvWrapper
pattern for accessing scene, device, episode_length_buf, etc.

Action mode: relative velocity command
    nav_cmd[t] = clamp(nav_cmd[t-1] + 0.25 * action[t], nav_min, nav_max)
"""

from __future__ import annotations

import statistics
from collections import deque

import gymnasium as gym
import torch
from tensordict import TensorDict

from scripts.low_level_policy import GrootLowLevelPolicy

# nav_cmd clamp limits
NAV_CMD_MIN = torch.tensor([-1.5, -0.75, -1.0])
NAV_CMD_MAX = torch.tensor([1.5, 0.75, 1.0])
# action → nav_cmd change scale
ACTION_SCALE = 0.25
BOX_IN_PIT_Z_THRESHOLD = -0.2
BOX_DROP_FAILURE_Z_THRESHOLD = 0.0
PIT_TARGET_X_RANGE = (-0.5, 0.5)
PIT_TARGET_Y_RANGE = (-2.0, -0.5)
PIT_TARGET_CENTER_W = (
    0.5 * (PIT_TARGET_X_RANGE[0] + PIT_TARGET_X_RANGE[1]),
    0.5 * (PIT_TARGET_Y_RANGE[0] + PIT_TARGET_Y_RANGE[1]),
)
OBSTACLE_RECT_X_RANGE = (-0.7, 0.7)
OBSTACLE_RECT_Y_RANGE = (-0.5, 3.5)
OBSTACLE_RECT_CENTER_W = (
    0.5 * (OBSTACLE_RECT_X_RANGE[0] + OBSTACLE_RECT_X_RANGE[1]),
    0.5 * (OBSTACLE_RECT_Y_RANGE[0] + OBSTACLE_RECT_Y_RANGE[1]),
)
OBSTACLE_RECT_MARGIN = 0.5
REWARD_BOX_PIT_WEIGHT = 4.0
REWARD_BOX_PIT_DISTANCE_SCALE = 1.0  #2.0
BOX_PIT_GATE_FULL_DISTANCE = 0.8
BOX_PIT_GATE_ZERO_DISTANCE = 1.2
BOX_PIT_GATE_FLOOR = 0.1
BOX_XZ_PLANE_NEAR_PIT_FULL_DISTANCE = 0.5
BOX_XZ_PLANE_NEAR_PIT_ZERO_DISTANCE = 2.0
BOX_XZ_PLANE_METRIC_THRESHOLD = 0.8
POST_PIT_HOLD_STEPS = 150
ROBOT_STABLE_Z_THRESHOLD = 0.25
REWARD_APPROACH_WEIGHT = 35.0
REWARD_ROBOT_BOX_WEIGHT = 0.2   # 0.3
REWARD_ROBOT_BOX_DISTANCE_SCALE = 0.8  #0.9
REWARD_BOX_IN_PIT_WEIGHT = 500.0
REWARD_BOX_XZ_PLANE_WEIGHT = 1.0
REWARD_BOX_IN_PIT_ALIGN_BONUS_WEIGHT = 75.0
REWARD_STABLE_AFTER_PIT_WEIGHT = 1.0
OBSTACLE_PENALTY_WEIGHT = 0.001
REWARD_ALIVE = 0.01  #0.2
REWARD_TIME = -0.005
REWARD_ACTION_RATE_WEIGHT = -0.01
REWARD_NAV_CMD_CHANGE_WEIGHT = -0.005
REWARD_COMPONENT_KEYS = (
    "box_pit",
    "box_pit_gate",
    "approach",
    "robot_box",
    "box_in_pit",
    "box_y_axis_xz_plane",
    "box_in_pit_align_bonus",
    "stable_after_pit",
    "obstacle",
    "alive",
    "time",
    "action_rate",
    "nav_cmd_change",
    "total",
)


class PlannerEnv(gym.Wrapper):
    """High-level planner environment wrapping ATEC TaskD-G1.

    Parameters
    ----------
    env_cfg : object
        ATEC TaskD-G1 environment config (e.g. ``TaskDEnvG1Cfg``).
    low_level : GrootLowLevelPolicy
        GPU-parallelized GR00T low-level policy.
    """

    def __init__(self, env_cfg, low_level: GrootLowLevelPolicy):
        env = gym.make("ATEC-TaskD-G1", cfg=env_cfg)
        super().__init__(env)
        self.low_level = low_level
        self.device = self.unwrapped.device
        self.num_envs = self.unwrapped.num_envs

        self._nav_min = NAV_CMD_MIN.to(self.device)
        self._nav_max = NAV_CMD_MAX.to(self.device)
        self._pit_target_center_w = torch.tensor(PIT_TARGET_CENTER_W, device=self.device)
        self._obstacle_rect_center_w = torch.tensor(OBSTACLE_RECT_CENTER_W, device=self.device)

        # Fixed terrain positions (terrain size 12×8, seed=0, single sub-terrain)
        # Robot spawns at terrain-local (1.8, 4.0, 0.0)
        # Planner target pit rectangle is world x=[-0.5, 0.5], y=[-2.0, -0.5].
        # Planner obstacle rectangle is world x=[-0.7, 0.7], y=[-0.5, 3.5].
        self._spawn_offset = torch.tensor([1.8, 4.0, 0.0], device=self.device)

        # State
        self._current_obs: dict | None = None
        self._prev_box_pos_local: torch.Tensor | None = None
        self._prev_box_pos_w: torch.Tensor | None = None
        self._box_in_pit_given: torch.Tensor | None = None
        self._box_in_pit_latched: torch.Tensor | None = None
        self._post_pit_step_buf: torch.Tensor | None = None
        self._last_planner_action: torch.Tensor | None = None
        self._nav_cmd: torch.Tensor | None = None
        self._last_reward_components: dict[str, torch.Tensor] = {}
        self._post_reset_sync_compare_printed = False
        self._episode_step_buf: torch.Tensor | None = None
        self._done_reason_printed = False

        # Initial reset so get_observations() works before the runner calls reset()
        self.reset()

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def max_episode_length(self):
        return self.unwrapped.max_episode_length

    # ── Core interface ──────────────────────────────────────────────────────

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_obs = obs
        self.low_level.reset(self.num_envs)
        self._box_in_pit_given = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._box_in_pit_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._post_pit_step_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_planner_action = torch.zeros(self.num_envs, 3, device=self.device)
        self._nav_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self._episode_step_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_box_pos_local = self._box_pos_local()
        self._prev_box_pos_w = self.unwrapped.scene["box"].data.root_pos_w.detach().clone()
        self._post_reset_sync_compare_printed = False
        self._done_reason_printed = False
        return self._build_planner_obs(obs), info

    def step(self, planner_action: torch.Tensor):
        """Step the environment with a planner action.

        Parameters
        ----------
        planner_action : torch.Tensor
            Shape ``(num_envs, 3)`` — relative change [dvx, dvy, dvyaw].

        Returns
        -------
        obs_dict : dict with keys "policy" and "critic"
        reward : torch.Tensor, shape ``(num_envs,)``
        terminated : torch.Tensor, shape ``(num_envs,)``
        truncated : torch.Tensor, shape ``(num_envs,)``
        info : dict
        """
        # Relative action: nav_cmd[t] = clamp(nav_cmd[t-1] + 0.25 * action[t])
        prev_nav_cmd = self._nav_cmd.detach().clone()
        self._nav_cmd = torch.clamp(
            self._nav_cmd + ACTION_SCALE * planner_action,
            self._nav_min, self._nav_max,
        )
        self._episode_step_buf += 1
        nav_cmd = self._nav_cmd.detach().clone()
        prev_action = self._last_planner_action
        self._last_planner_action = planner_action.detach().clone()

        # Decimation=1: one planner step = one env step
        low_action = self.low_level.predict(self._current_obs, nav_cmd)

        obs, _, terminated, truncated, info = self.env.step(low_action)
        self._current_obs = obs

        # Planner-level termination: success after holding stable post-pit, or failure by falling elsewhere.
        box_in_pit_now = self._check_box_in_pit()
        _, post_pit_success_now = self._update_post_pit_state(box_in_pit_now)
        box_drop_failure_now = self._check_box_drop_failure()
        planner_done_now = post_pit_success_now | box_drop_failure_now
        raw_terminated = terminated.clone()
        planner_done_reset = self._planner_done_reset_mask(planner_done_now, raw_terminated, truncated)
        planner_success_reset = self._planner_success_reset_mask(post_pit_success_now, raw_terminated, truncated)
        terminated = terminated | planner_done_now

        # Compute reward BEFORE resetting per-env state (needs prev box pos)
        reward = self._compute_reward(prev_action, prev_nav_cmd, nav_cmd)

        # Reset per-env tracking state for environments that were auto-reset
        done = terminated | truncated
        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            info = dict(info)
            info["planner_done_metrics"] = self._compute_done_metrics(
                done_ids,
                post_pit_success_now,
                box_drop_failure_now=box_drop_failure_now,
                raw_terminated=raw_terminated,
                truncated=truncated,
                planner_success_reset=planner_success_reset,
                planner_done_reset=planner_done_reset,
            )
            if planner_done_reset.any():
                planner_done_reset_ids = planner_done_reset.nonzero(as_tuple=False).squeeze(-1)
                reset_obs, reset_info = self.unwrapped.reset(env_ids=planner_done_reset_ids)
                self._current_obs = reset_obs
                info["planner_done_reset_ids"] = planner_done_reset_ids
                info["planner_done_reset_info"] = reset_info
                planner_success_reset_ids = planner_success_reset.nonzero(as_tuple=False).squeeze(-1)
                if planner_success_reset_ids.numel() > 0:
                    info["planner_success_reset_ids"] = planner_success_reset_ids
            if not self._done_reason_printed:
                robot_z = self.unwrapped.scene["robot"].data.root_pos_w[:, 2]
                fell = robot_z < 0.25
                done_steps = self._episode_step_buf[done_ids]
                print("[DONE REASON COMPARE]")
                print(f"  done_count={done_ids.numel()}")
                print(f"  done_steps_unique={torch.unique(done_steps).tolist()}")
                print(f"  raw_terminated_count={raw_terminated[done_ids].sum().item()}")
                print(f"  truncated_count={truncated[done_ids].sum().item()}")
                print(f"  box_in_pit_count={box_in_pit_now[done_ids].sum().item()}")
                print(f"  post_pit_success_count={post_pit_success_now[done_ids].sum().item()}")
                print(f"  box_drop_failure_count={box_drop_failure_now[done_ids].sum().item()}")
                print(f"  fell_count={fell[done_ids].sum().item()}")
                print(f"  robot_z_min={robot_z[done_ids].min().item():.4f}")
                print(f"  robot_z_max={robot_z[done_ids].max().item():.4f}")
                print(f"  nav_cmd_mean={nav_cmd[done_ids].mean(dim=0).tolist()}")
                print(f"  nav_cmd_max_abs={nav_cmd[done_ids].abs().amax(dim=0).tolist()}")
                self._done_reason_printed = True
            if not self._post_reset_sync_compare_printed:
                current_proprio = self._current_obs["proprio"][done_ids].detach().clone()
                self.unwrapped.scene.write_data_to_sim()
                self.unwrapped.sim.forward()
                synced_obs = self.unwrapped.observation_manager.compute(update_history=False)
                synced_proprio = synced_obs["proprio"][done_ids]
                proprio_diff = (synced_proprio - current_proprio).abs().amax(dim=-1)
                print("[POST-RESET SYNC COMPARE]")
                print(f"  env_ids={done_ids.tolist()}")
                print(f"  proprio_max_abs_diff={proprio_diff.tolist()}")
                print(f"  current_proprio_head[0]={current_proprio[0, :16].tolist()}")
                print(f"  synced_proprio_head[0]={synced_proprio[0, :16].tolist()}")
                self._post_reset_sync_compare_printed = True
            self._box_in_pit_given[done_ids] = False
            self._box_in_pit_latched[done_ids] = False
            self._post_pit_step_buf[done_ids] = 0
            self._nav_cmd[done_ids] = 0.0
            self._last_planner_action[done_ids] = 0.0
            self._episode_step_buf[done_ids] = 0
            # Reset GR00T history for done envs to avoid stale OOD input
            proprio = self._current_obs["proprio"]
            single_obs = self.low_level._build_obs(proprio, torch.zeros_like(nav_cmd))
            self.low_level.reset_envs(done_ids, single_obs)
            self._prev_box_pos_local = self._box_pos_local()
            self._prev_box_pos_w = self.unwrapped.scene["box"].data.root_pos_w.detach().clone()
        else:
            self._prev_box_pos_local = self._box_pos_local()
            self._prev_box_pos_w = self.unwrapped.scene["box"].data.root_pos_w.detach().clone()

        return self._build_planner_obs(self._current_obs), reward, terminated, truncated, info

    def get_observations(self):
        return self._build_planner_obs(self._current_obs)

    # ── Observation construction ────────────────────────────────────────────

    def _build_planner_obs(self, obs: dict) -> dict:
        """Build observation dict with 'policy' and 'critic' groups.

        Returns
        -------
        dict
            "policy": (N, 372) — actor observations
            "critic": (N, 23) — privileged observations (critic-only part)
        """
        planner_obs, privileged = self._build_obs(obs)
        return {"policy": planner_obs, "critic": privileged}

    def _build_obs(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Build actor obs (372D) and critic-only privileged obs (23D)."""
        proprio = obs["proprio"]  # (N, 111)
        height_scan = obs["extero"]  # (N, 5760)

        # ── Actor obs (372D) ──
        lin_vel = proprio[:, 0:3]
        ang_vel = proprio[:, 3:6]
        gravity = proprio[:, 9:12]
        h_scan = self._compress_height_scan(height_scan)

        planner_obs = torch.cat([
            lin_vel,                    # 3
            ang_vel,                    # 3
            gravity,                    # 3
            self._last_planner_action,  # 3
            h_scan,                     # 360
        ], dim=-1)  # 372D

        # ── Critic privileged (23D) ──
        robot_pos_local = self._robot_pos_local()[:, :2]  # (N, 2)
        robot_yaw = self._robot_yaw()  # (N, 1)
        box_pos_local = self._box_pos_local()[:, :2]  # (N, 2)
        pit_center_local = self._pit_target_center_local()
        obs_local = self._obstacle_rect_center_local()
        box_xy_w = self.unwrapped.scene["box"].data.root_pos_w[:, :2]
        box_vel_xy_w = self.unwrapped.scene["box"].data.root_vel_w[:, :2]
        box_z_w = self.unwrapped.scene["box"].data.root_pos_w[:, 2:3]

        box_to_pit = self._box_to_pit_distance_xy(box_xy_w).unsqueeze(-1)
        robot_to_box = torch.norm(robot_pos_local - box_pos_local, dim=-1, keepdim=True).clamp(max=3.0)
        speed = torch.norm(lin_vel, dim=-1, keepdim=True)
        episode_progress = self._episode_step_buf.unsqueeze(-1).float() / float(self.max_episode_length)
        robot_fall_margin = self.unwrapped.scene["robot"].data.root_pos_w[:, 2:3] - 0.25
        box_motion_toward_pit = self._box_motion_toward_pit(box_xy_w, box_vel_xy_w)
        box_in_pit = self._check_box_in_pit().float().unsqueeze(-1)
        box_to_obstacle = self._obstacle_rect_distance_xy(box_xy_w).unsqueeze(-1)
        box_y_axis_xz_plane = self._box_y_axis_xz_plane_score().unsqueeze(-1)
        box_y_axis_xz_plane_error_deg = self._box_y_axis_xz_plane_error_deg().unsqueeze(-1) / 90.0

        priv = torch.cat([
            robot_pos_local,   # 2
            robot_yaw,         # 1
            box_pos_local,     # 2
            self._nav_cmd,     # 3
            pit_center_local,  # 2
            obs_local,         # 2
            box_to_pit,        # 1
            robot_to_box,      # 1
            speed,             # 1
            episode_progress,  # 1
            robot_fall_margin, # 1
            box_z_w,           # 1
            box_motion_toward_pit,  # 1
            box_in_pit,        # 1
            box_to_obstacle,   # 1
            box_y_axis_xz_plane,  # 1
            box_y_axis_xz_plane_error_deg,  # 1
        ], dim=-1)  # 23D

        return planner_obs, priv

    def _compress_height_scan(self, h: torch.Tensor) -> torch.Tensor:
        """(N, 5760) → (N, 360)"""
        h = h.view(-1, 16, 360)
        h = h.view(-1, 4, 4, 360).mean(dim=2)
        h = h.view(-1, 4, 90, 4).mean(dim=-1)
        return h.clamp(-1.0, 1.0).view(-1, 360)

    # ── GT position helpers ─────────────────────────────────────────────────

    def _check_box_in_pit(self) -> torch.Tensor:
        """Check if the box has fallen into the target pit in world-frame coordinates."""
        box_pos_w = self.unwrapped.scene["box"].data.root_pos_w
        return (box_pos_w[:, 2] < BOX_IN_PIT_Z_THRESHOLD) & self._box_xy_in_pit_rect(box_pos_w[:, :2])

    def _check_box_drop_failure(self) -> torch.Tensor:
        """Check if the box fell below ground outside the target pit rectangle."""
        box_pos_w = self.unwrapped.scene["box"].data.root_pos_w
        return (box_pos_w[:, 2] < BOX_DROP_FAILURE_Z_THRESHOLD) & (~self._box_xy_in_pit_rect(box_pos_w[:, :2]))

    def _box_xy_in_pit_rect(self, box_xy_w: torch.Tensor) -> torch.Tensor:
        """World-frame XY membership in the target pit rectangle."""
        x_min, x_max = PIT_TARGET_X_RANGE
        y_min, y_max = PIT_TARGET_Y_RANGE
        return (
            (box_xy_w[:, 0] >= x_min)
            & (box_xy_w[:, 0] <= x_max)
            & (box_xy_w[:, 1] >= y_min)
            & (box_xy_w[:, 1] <= y_max)
        )

    def _box_to_pit_distance_xy(self, box_xy_w: torch.Tensor) -> torch.Tensor:
        """World-frame XY distance from box position to the target pit rectangle."""
        x_min, x_max = PIT_TARGET_X_RANGE
        y_min, y_max = PIT_TARGET_Y_RANGE
        dx = torch.clamp(x_min - box_xy_w[:, 0], min=0.0) + torch.clamp(box_xy_w[:, 0] - x_max, min=0.0)
        dy = torch.clamp(y_min - box_xy_w[:, 1], min=0.0) + torch.clamp(box_xy_w[:, 1] - y_max, min=0.0)
        return torch.sqrt(dx * dx + dy * dy)

    def _box_pit_gate(self, robot_box_distance: torch.Tensor) -> torch.Tensor:
        """Softly enable the box→pit shaping reward once the robot is close enough to push."""
        gate = torch.clamp(
            (BOX_PIT_GATE_ZERO_DISTANCE - robot_box_distance)
            / (BOX_PIT_GATE_ZERO_DISTANCE - BOX_PIT_GATE_FULL_DISTANCE),
            min=0.0,
            max=1.0,
        )
        return BOX_PIT_GATE_FLOOR + (1.0 - BOX_PIT_GATE_FLOOR) * gate

    def _box_y_axis_w(self) -> torch.Tensor:
        """Box local y-axis expressed in world coordinates."""
        quat = self.unwrapped.scene["box"].data.root_quat_w  # (N, 4) wxyz
        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return torch.stack(
            [
                2.0 * (qx * qy - qw * qz),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz + qw * qx),
            ],
            dim=-1,
        )

    def _box_y_axis_xz_plane_score(self) -> torch.Tensor:
        """Score 1 when box local y-axis lies in world xz-plane, 0 when parallel to world y."""
        box_y_axis_w = self._box_y_axis_w()
        return torch.clamp(box_y_axis_w[:, 0] ** 2 + box_y_axis_w[:, 2] ** 2, min=0.0, max=1.0)

    def _box_y_axis_xz_plane_error_deg(self) -> torch.Tensor:
        """Smallest angle in degrees from box local y-axis to the world xz-plane."""
        box_y_axis_w = self._box_y_axis_w()
        score = torch.clamp(1.0 - box_y_axis_w[:, 1] ** 2, min=0.0, max=1.0)
        return torch.acos(torch.sqrt(score)) * (180.0 / torch.pi)

    def _box_xz_plane_near_pit_gate(self, box_to_pit_distance: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (BOX_XZ_PLANE_NEAR_PIT_ZERO_DISTANCE - box_to_pit_distance)
            / (BOX_XZ_PLANE_NEAR_PIT_ZERO_DISTANCE - BOX_XZ_PLANE_NEAR_PIT_FULL_DISTANCE),
            min=0.0,
            max=1.0,
        )

    def _box_motion_toward_pit(self, box_xy_w: torch.Tensor, box_vel_xy_w: torch.Tensor) -> torch.Tensor:
        """Velocity projected onto the direction from the box to the nearest pit point."""
        x_min, x_max = PIT_TARGET_X_RANGE
        y_min, y_max = PIT_TARGET_Y_RANGE
        nearest_x = torch.clamp(box_xy_w[:, 0], min=x_min, max=x_max)
        nearest_y = torch.clamp(box_xy_w[:, 1], min=y_min, max=y_max)
        nearest_xy = torch.stack((nearest_x, nearest_y), dim=-1)
        delta_xy = nearest_xy - box_xy_w
        delta_norm = torch.norm(delta_xy, dim=-1, keepdim=True)
        direction_xy = delta_xy / torch.clamp(delta_norm, min=1e-6)
        projection = torch.sum(box_vel_xy_w * direction_xy, dim=-1, keepdim=True)
        inside = self._box_xy_in_pit_rect(box_xy_w)[..., None]
        return torch.where(inside, torch.zeros_like(projection), projection)

    def _pit_target_center_local(self) -> torch.Tensor:
        """Target pit rectangle center in the same env-local frame used by critic obs."""
        pit_center_w = self._pit_target_center_w[None, :]
        return pit_center_w - self.unwrapped.scene.env_origins[:, :2] - self._spawn_offset[:2][None, :]

    def _obstacle_rect_center_local(self) -> torch.Tensor:
        """Obstacle rectangle center in the same env-local frame used by critic obs."""
        obstacle_center_w = self._obstacle_rect_center_w[None, :]
        return obstacle_center_w - self.unwrapped.scene.env_origins[:, :2] - self._spawn_offset[:2][None, :]

    def _obstacle_rect_distance_xy(self, point_xy_w: torch.Tensor) -> torch.Tensor:
        """World-frame XY distance from points to the obstacle rectangle exterior."""
        x_min, x_max = OBSTACLE_RECT_X_RANGE
        y_min, y_max = OBSTACLE_RECT_Y_RANGE
        dx = torch.clamp(x_min - point_xy_w[:, 0], min=0.0) + torch.clamp(point_xy_w[:, 0] - x_max, min=0.0)
        dy = torch.clamp(y_min - point_xy_w[:, 1], min=0.0) + torch.clamp(point_xy_w[:, 1] - y_max, min=0.0)
        return torch.sqrt(dx * dx + dy * dy)

    def _obstacle_rect_penalty_xy(self, point_xy_w: torch.Tensor) -> torch.Tensor:
        """Penalty amount for approaching or entering the obstacle rectangle in world XY."""
        x_min, x_max = OBSTACLE_RECT_X_RANGE
        y_min, y_max = OBSTACLE_RECT_Y_RANGE
        x = point_xy_w[:, 0]
        y = point_xy_w[:, 1]

        outside_dist = self._obstacle_rect_distance_xy(point_xy_w)
        inside = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        inside_depth = torch.minimum(
            torch.minimum(x - x_min, x_max - x),
            torch.minimum(y - y_min, y_max - y),
        )
        outside_penalty = torch.clamp(OBSTACLE_RECT_MARGIN - outside_dist, min=0.0)
        inside_penalty = OBSTACLE_RECT_MARGIN + inside_depth
        return torch.where(inside, inside_penalty, outside_penalty)

    def _planner_success_reset_mask(
        self,
        box_in_pit_now: torch.Tensor,
        raw_terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        """Env ids that PlannerEnv terminates and IsaacLab did not already reset."""
        return box_in_pit_now & (~raw_terminated) & (~truncated)

    def _planner_done_reset_mask(
        self,
        planner_done_now: torch.Tensor,
        raw_terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        """Planner-added success/failure dones that IsaacLab did not already reset."""
        return planner_done_now & (~raw_terminated) & (~truncated)

    def _update_post_pit_state(self, box_in_pit_now: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Latch pit entry and require a 150-step post-pit hold before success done."""
        first_entry = box_in_pit_now & (~self._box_in_pit_latched)
        self._box_in_pit_latched |= box_in_pit_now

        holding_after_pit = self._box_in_pit_latched & box_in_pit_now & (~first_entry)
        self._post_pit_step_buf[holding_after_pit] += 1
        post_pit_success = box_in_pit_now & (self._post_pit_step_buf >= POST_PIT_HOLD_STEPS)
        return first_entry, post_pit_success

    def _box_in_pit_success_reward(self) -> torch.Tensor:
        """One-time success bonus when the box falls below the pit threshold."""
        in_pit = self._check_box_in_pit()
        box_in_pit = in_pit & (~self._box_in_pit_given)
        self._box_in_pit_given |= in_pit
        if hasattr(box_in_pit, "float"):
            box_in_pit = box_in_pit.float()
        else:
            box_in_pit = box_in_pit.astype("float32")
        return REWARD_BOX_IN_PIT_WEIGHT * box_in_pit

    def _stable_after_pit_reward(self) -> torch.Tensor:
        """Reward standing on the ground after the box has entered the pit."""
        robot_z = self.unwrapped.scene["robot"].data.root_pos_w[:, 2]
        box_in_pit_now = self._check_box_in_pit()
        stable = (
            self._box_in_pit_latched
            & box_in_pit_now
            & (self._post_pit_step_buf > 0)
            & (robot_z >= ROBOT_STABLE_Z_THRESHOLD)
        )
        if hasattr(stable, "float"):
            stable = stable.float()
        else:
            stable = stable.astype("float32")
        return REWARD_STABLE_AFTER_PIT_WEIGHT * stable


    def _compute_done_metrics(
        self,
        done_ids: torch.Tensor,
        box_in_pit_now: torch.Tensor,
        box_drop_failure_now: torch.Tensor | None = None,
        raw_terminated: torch.Tensor | None = None,
        truncated: torch.Tensor | None = None,
        planner_success_reset: torch.Tensor | None = None,
        planner_done_reset: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute done-time task metrics in world-frame XY coordinates.

        IsaacLab auto-resets native fall/time-out dones inside ``env.step()`` before
        returning. Planner-added box-in-pit dones are measured at the success state;
        native dones may reflect the post-reset pose if IsaacLab already reset them.
        """
        if box_drop_failure_now is None:
            box_drop_failure_now = torch.zeros_like(box_in_pit_now)
        if raw_terminated is None:
            raw_terminated = torch.zeros_like(box_in_pit_now)
        if truncated is None:
            truncated = torch.zeros_like(box_in_pit_now)
        if planner_success_reset is None:
            planner_success_reset = torch.zeros_like(box_in_pit_now)
        if planner_done_reset is None:
            planner_done_reset = torch.zeros_like(box_in_pit_now)

        robot_xy_w = self.unwrapped.scene["robot"].data.root_pos_w[done_ids, :2]
        box_pos_w = self.unwrapped.scene["box"].data.root_pos_w[done_ids]
        box_xy_w = box_pos_w[:, :2]
        box_z_w = box_pos_w[:, 2]
        robot_box_distance = torch.norm(robot_xy_w - box_xy_w, dim=-1)
        box_pit_distance = self._box_to_pit_distance_xy(box_xy_w)
        box_y_axis_xz_plane = self._box_y_axis_xz_plane_score()[done_ids]
        box_y_axis_xz_plane_error_deg = self._box_y_axis_xz_plane_error_deg()[done_ids]
        box_entered_pit = self._box_in_pit_latched[done_ids]
        box_entered_pit_xz_plane_good = box_entered_pit & (
            box_y_axis_xz_plane >= BOX_XZ_PLANE_METRIC_THRESHOLD
        )
        box_in_pit_success = box_in_pit_now[done_ids]
        box_drop_failure = box_drop_failure_now[done_ids]
        box_xy_in_pit_rect = self._box_xy_in_pit_rect(box_xy_w)
        box_z_success_and_xy_in_rect = box_in_pit_success & box_xy_in_pit_rect
        if hasattr(box_in_pit_success, "float"):
            box_in_pit_success = box_in_pit_success.float()
            box_drop_failure = box_drop_failure.float()
            box_entered_pit = box_entered_pit.float()
            box_entered_pit_xz_plane_good = box_entered_pit_xz_plane_good.float()
        else:
            box_in_pit_success = box_in_pit_success.astype("float32")
            box_drop_failure = box_drop_failure.astype("float32")
            box_entered_pit = box_entered_pit.astype("float32")
            box_entered_pit_xz_plane_good = box_entered_pit_xz_plane_good.astype("float32")

        return {
            "box_in_pit_success": box_in_pit_success,
            "box_drop_failure": box_drop_failure,
            "robot_box_distance": robot_box_distance,
            "box_pit_distance": box_pit_distance,
            "raw_terminated": raw_terminated[done_ids],
            "truncated": truncated[done_ids],
            "planner_success_reset": planner_success_reset[done_ids],
            "planner_done_reset": planner_done_reset[done_ids],
            "box_xy_in_pit_rect": box_xy_in_pit_rect,
            "box_z_success_and_xy_in_rect": box_z_success_and_xy_in_rect,
            "box_z": box_z_w,
            "box_entered_pit": box_entered_pit,
            "box_entered_pit_xz_plane_good": box_entered_pit_xz_plane_good,
            "box_y_axis_xz_plane": box_y_axis_xz_plane,
            "box_y_axis_xz_plane_error_deg": box_y_axis_xz_plane_error_deg,
        }

    def _to_local(self, pos_w: torch.Tensor) -> torch.Tensor:
        """World position → env-local (relative to env origin + spawn offset)."""
        return pos_w - self.unwrapped.scene.env_origins - self._spawn_offset.unsqueeze(0)

    def _robot_pos_local(self) -> torch.Tensor:
        pos_w = self.unwrapped.scene["robot"].data.root_pos_w
        return self._to_local(pos_w)

    def _box_pos_local(self) -> torch.Tensor:
        pos_w = self.unwrapped.scene["box"].data.root_pos_w
        return self._to_local(pos_w)

    def _robot_yaw(self) -> torch.Tensor:
        quat = self.unwrapped.scene["robot"].data.root_quat_w  # (N, 4) wxyz
        qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return yaw.unsqueeze(-1)

    # ── Reward computation ──────────────────────────────────────────────────

    def _compute_reward(self, prev_action: torch.Tensor, prev_nav_cmd: torch.Tensor, nav_cmd: torch.Tensor) -> torch.Tensor:
        box_local = self._box_pos_local()

        # Robot → box distance (exp shaped, always positive)
        robot_local = self._robot_pos_local()
        robot_box = torch.norm(robot_local[:, :2] - box_local[:, :2], dim=-1).clamp(max=3.0)
        box_pit_gate = self._box_pit_gate(robot_box)

        # Box → target pit rectangle distance (exp shaped, always positive)
        box_xy_w = self.unwrapped.scene["box"].data.root_pos_w[:, :2]
        dist = self._box_to_pit_distance_xy(box_xy_w)
        box_pit_reward = box_pit_gate * REWARD_BOX_PIT_WEIGHT * torch.exp(-dist / REWARD_BOX_PIT_DISTANCE_SCALE)

        # Reward only positive progress toward the pit, matching the restored 9000-iteration run.
        prev_dist = self._box_to_pit_distance_xy(self._prev_box_pos_w[:, :2])
        progress = prev_dist - dist
        approach = progress.clamp(min=0)

        robot_box_reward = REWARD_ROBOT_BOX_WEIGHT * torch.exp(-robot_box / REWARD_ROBOT_BOX_DISTANCE_SCALE)

        # Box in pit (one-time reward)
        box_in_pit_reward = self._box_in_pit_success_reward()
        box_entry_mask = box_in_pit_reward / REWARD_BOX_IN_PIT_WEIGHT
        box_y_axis_xz_plane = self._box_y_axis_xz_plane_score()
        box_y_axis_xz_plane_reward = (
            REWARD_BOX_XZ_PLANE_WEIGHT
            * self._box_xz_plane_near_pit_gate(dist)
            * box_y_axis_xz_plane
        )
        box_in_pit_align_bonus = (
            REWARD_BOX_IN_PIT_ALIGN_BONUS_WEIGHT
            * box_entry_mask
            * box_y_axis_xz_plane
        )
        stable_after_pit_reward = self._stable_after_pit_reward()

        # Obstacle penalty
        robot_xy_w = self.unwrapped.scene["robot"].data.root_pos_w[:, :2]
        obstacle_pen = self._obstacle_rect_penalty_xy(robot_xy_w) + self._obstacle_rect_penalty_xy(box_xy_w)

        # Smoothness penalties
        action_rate = torch.sum((self._last_planner_action - prev_action) ** 2, dim=-1)
        nav_cmd_change = torch.sum((nav_cmd - prev_nav_cmd) ** 2, dim=-1)

        components = {
            "box_pit": box_pit_reward,
            "box_pit_gate": box_pit_gate,
            "approach": REWARD_APPROACH_WEIGHT * approach,
            "robot_box": robot_box_reward,
            "box_in_pit": box_in_pit_reward,
            "box_y_axis_xz_plane": box_y_axis_xz_plane_reward,
            "box_in_pit_align_bonus": box_in_pit_align_bonus,
            "stable_after_pit": stable_after_pit_reward,
            "obstacle": -OBSTACLE_PENALTY_WEIGHT * obstacle_pen,
            "alive": torch.full_like(box_pit_reward, REWARD_ALIVE),
            "time": torch.full_like(box_pit_reward, REWARD_TIME),
            "action_rate": REWARD_ACTION_RATE_WEIGHT * action_rate,
            "nav_cmd_change": REWARD_NAV_CMD_CHANGE_WEIGHT * nav_cmd_change,
        }
        total_reward = (
            box_pit_reward
            + REWARD_APPROACH_WEIGHT * approach
            + robot_box_reward
            + box_in_pit_reward                     # positive, success
            + box_y_axis_xz_plane_reward            # positive, keep long edge in world xz-plane near pit
            + box_in_pit_align_bonus                # positive, prefer xz-plane pit entry
            + stable_after_pit_reward               # positive, post-pit stability
            - OBSTACLE_PENALTY_WEIGHT * obstacle_pen
            + REWARD_ALIVE
            + REWARD_TIME
            + REWARD_ACTION_RATE_WEIGHT * action_rate
            + REWARD_NAV_CMD_CHANGE_WEIGHT * nav_cmd_change
        )
        components["total"] = total_reward
        self._last_reward_components = components
        return total_reward


class PlannerRslRlWrapper:
    """Adapt PlannerEnv to RSL-RL's VecEnv interface.

    Follows the same pattern as IsaacLab's RslRlVecEnvWrapper.
    Uses self.unwrapped (which resolves through gym.Wrapper to
    ManagerBasedRLEnv) for accessing env internals like
    episode_length_buf, cfg, and scene.
    """

    def __init__(self, planner_env: PlannerEnv):
        self.env = planner_env
        self.num_envs = planner_env.num_envs
        self.device = planner_env.device
        self.num_actions = 3  # [dvx, dvy, dvyaw]
        self.num_obs = 372  # planner_obs dim
        self.num_privileged_obs = 395  # planner_obs + privileged
        self.max_episode_length = planner_env.max_episode_length
        self.clip_actions = torch.tensor([0.5, 0.3, 0.3], device=self.device)
        self._init_done_metric_buffers()
        self._init_reward_component_accumulators()

        # Reset since OnPolicyRunner does not call reset
        self.env.reset()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def cfg(self):
        return self.unwrapped.cfg

    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf = value

    def get_observations(self) -> TensorDict:
        obs_dict = self.env.get_observations()
        return TensorDict(obs_dict, batch_size=[self.num_envs])

    def reset(self) -> tuple[TensorDict, dict]:
        obs_dict, info = self.env.reset()
        return TensorDict(obs_dict, batch_size=[self.num_envs]), info

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # actions = torch.zeros_like(actions)
        obs_dict, reward, terminated, truncated, info = self.env.step(actions)
        dones = (terminated | truncated).long()
        if not hasattr(self, "_dbg_steps"):
            self._dbg_steps = 0
        if self._dbg_steps < 5:
            print(
                f"[WRAP step {self._dbg_steps}] "
                f"dones={dones.sum().item()} "
                f"term={terminated.sum().item()} "
                f"trunc={truncated.sum().item()} "
                f"ep_len_min={self.unwrapped.episode_length_buf.min().item()} "
                f"ep_len_max={self.unwrapped.episode_length_buf.max().item()}"
            )
            self._dbg_steps += 1
        extras = {"time_outs": truncated}
        if "planner_done_metrics" in info:
            self._update_done_metric_buffers(info["planner_done_metrics"])
        self._update_reward_component_accumulators(self.env._last_reward_components)
        return TensorDict(obs_dict, batch_size=[self.num_envs]), reward, dones, extras

    def _init_done_metric_buffers(self) -> None:
        self._done_metric_buffers = {
            "box_in_pit_success": deque(maxlen=100),
            "box_drop_failure": deque(maxlen=100),
            "robot_box_distance": deque(maxlen=100),
            "box_pit_distance": deque(maxlen=100),
            "raw_terminated": deque(maxlen=100),
            "truncated": deque(maxlen=100),
            "planner_success_reset": deque(maxlen=100),
            "planner_done_reset": deque(maxlen=100),
            "box_xy_in_pit_rect": deque(maxlen=100),
            "box_z_success_and_xy_in_rect": deque(maxlen=100),
            "box_z": deque(maxlen=100),
            "box_entered_pit": deque(maxlen=100),
            "box_entered_pit_xz_plane_good": deque(maxlen=100),
            "box_y_axis_xz_plane": deque(maxlen=100),
            "box_y_axis_xz_plane_error_deg": deque(maxlen=100),
        }

    def _init_reward_component_accumulators(self) -> None:
        self._reward_component_sums = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
        self._reward_component_count = 0

    def _update_reward_component_accumulators(self, reward_components: dict[str, torch.Tensor]) -> None:
        if not reward_components:
            return
        for key in REWARD_COMPONENT_KEYS:
            values = reward_components[key]
            if hasattr(values, "detach"):
                mean_value = values.detach().mean().item()
            else:
                mean_value = values.mean().item()
            self._reward_component_sums[key] += float(mean_value)
        self._reward_component_count += 1

    def pop_reward_component_summary(self) -> dict[str, float]:
        if self._reward_component_count == 0:
            return {}
        summary = {
            key: self._reward_component_sums[key] / self._reward_component_count
            for key in REWARD_COMPONENT_KEYS
        }
        self._init_reward_component_accumulators()
        return summary

    def _extend_metric_buffer(self, key: str, values) -> None:
        if hasattr(values, "detach"):
            values = values.detach().cpu().reshape(-1).tolist()
        else:
            values = values.reshape(-1).tolist()
        self._done_metric_buffers[key].extend(float(value) for value in values)

    def _update_done_metric_buffers(self, done_metrics: dict[str, torch.Tensor]) -> None:
        for key in self._done_metric_buffers:
            if key in done_metrics:
                self._extend_metric_buffer(key, done_metrics[key])

    def get_done_metric_summary(self) -> dict[str, float]:
        if not self._done_metric_buffers["box_in_pit_success"]:
            return {}

        def _mean(key: str) -> float | None:
            values = self._done_metric_buffers[key]
            if not values:
                return None
            return statistics.mean(values)

        def _mean_for_success(key: str, success_value: bool) -> float | None:
            values = self._done_metric_buffers[key]
            success_values = self._done_metric_buffers["box_in_pit_success"]
            if not values or len(values) != len(success_values):
                return None
            selected = [
                value
                for value, success in zip(values, success_values)
                if bool(success >= 0.5) is success_value
            ]
            if not selected:
                return None
            return statistics.mean(selected)

        def _mean_for_entered_pit(key: str) -> float | None:
            values = self._done_metric_buffers[key]
            entered_values = self._done_metric_buffers["box_entered_pit"]
            if not values or len(values) != len(entered_values):
                return None
            selected = [
                value
                for value, entered in zip(values, entered_values)
                if entered >= 0.5
            ]
            if not selected:
                return None
            return statistics.mean(selected)

        summary = {
            "box_in_pit_success_rate": statistics.mean(self._done_metric_buffers["box_in_pit_success"]),
            "robot_box_distance": statistics.mean(self._done_metric_buffers["robot_box_distance"]),
            "box_pit_distance": statistics.mean(self._done_metric_buffers["box_pit_distance"]),
        }
        scalar_keys = (
            "box_y_axis_xz_plane",
            "box_y_axis_xz_plane_error_deg",
        )
        for key in scalar_keys:
            value = _mean(key)
            if value is not None:
                summary[key] = value
        rate_keys = {
            "native_terminated_rate": "raw_terminated",
            "timeout_rate": "truncated",
            "planner_success_reset_rate": "planner_success_reset",
            "planner_done_reset_rate": "planner_done_reset",
            "box_drop_failure_rate": "box_drop_failure",
            "box_entered_pit_rate": "box_entered_pit",
            "box_xy_in_pit_rect_rate": "box_xy_in_pit_rect",
            "box_z_success_and_xy_in_rect_rate": "box_z_success_and_xy_in_rect",
        }
        for summary_key, buffer_key in rate_keys.items():
            value = _mean(buffer_key)
            if value is not None:
                summary[summary_key] = value

        entered_pit_xz_plane_rate = _mean_for_entered_pit("box_entered_pit_xz_plane_good")
        if entered_pit_xz_plane_rate is not None:
            summary["entered_pit_xz_plane_rate"] = entered_pit_xz_plane_rate
        entered_pit_xz_plane = _mean_for_entered_pit("box_y_axis_xz_plane")
        if entered_pit_xz_plane is not None:
            summary["entered_pit_box_y_axis_xz_plane"] = entered_pit_xz_plane
        entered_pit_xz_plane_error_deg = _mean_for_entered_pit("box_y_axis_xz_plane_error_deg")
        if entered_pit_xz_plane_error_deg is not None:
            summary["entered_pit_box_y_axis_xz_plane_error_deg"] = entered_pit_xz_plane_error_deg

        split_keys = (
            "robot_box_distance",
            "box_pit_distance",
            "box_xy_in_pit_rect",
            "box_z",
            "box_y_axis_xz_plane",
            "box_y_axis_xz_plane_error_deg",
        )
        for key in split_keys:
            success_mean = _mean_for_success(key, True)
            failure_mean = _mean_for_success(key, False)
            if success_mean is not None:
                if key == "box_xy_in_pit_rect":
                    summary["success_xy_in_pit_rect_rate"] = success_mean
                else:
                    summary[f"success_{key}"] = success_mean
            if failure_mean is not None:
                if key == "box_xy_in_pit_rect":
                    summary["failure_xy_in_pit_rect_rate"] = failure_mean
                else:
                    summary[f"failure_{key}"] = failure_mean

        return summary

    def close(self):
        self.env.close()
