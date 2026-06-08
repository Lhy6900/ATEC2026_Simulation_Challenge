"""GR00T low-level locomotion policy for GPU-parallelized training.

Wraps the TorchScript-converted GR00T models (walk + balance) for batched
GPU inference. Reimplements the observation construction and action mapping
from ``demo/groot_policy_adapter.py`` using batched PyTorch operations.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ── Constants (from groot_policy_adapter.py) ────────────────────────────────

GR00T_LOWER_BODY_DEFAULT_15 = [
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    0.0, 0.0, 0.0,
]

ATEC_G1_DEFAULT_33 = [
    -0.10, 0.0, 0.0, 0.30, -0.20, 0.0,
    -0.10, 0.0, 0.0, 0.30, -0.20, 0.0,
    0.0, 0.0, 0.0,
    0.0, 0.3, 0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, -0.3, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]

DEFAULT_UPPER_BODY_TARGET_29 = ATEC_G1_DEFAULT_33[:29]

GR00T_CMD_SCALE = [2.0, 2.0, 0.5]
GR00T_ANG_VEL_SCALE = 0.5
GR00T_DOF_VEL_SCALE = 0.05
GR00T_ACTION_SCALE = 0.25
GR00T_NUM_ACTIONS = 15
GR00T_SINGLE_OBS_DIM = 86
GR00T_HISTORY_LEN = 6
GR00T_NUM_OBS = GR00T_SINGLE_OBS_DIM * GR00T_HISTORY_LEN

ATEC_ENV_ACTION_SCALE = 0.5
WAIST_CANONICAL_INDICES = [12, 13, 14]


class GrootLowLevelPolicy:
    """Batched GR00T low-level policy for GPU training.

    Parameters
    ----------
    walk_path : str
        Path to the walk TorchScript model.
    balance_path : str
        Path to the balance TorchScript model.
    device : str
        Torch device (e.g. ``"cuda:0"``).
    """

    def __init__(
        self,
        walk_path: str = "demo/gr00t_walk.pt",
        balance_path: str = "demo/gr00t_balance.pt",
        device: str = "cuda:0",
    ):
        self.device = device
        self.walk = torch.jit.load(walk_path, map_location=device).eval()
        self.balance = torch.jit.load(balance_path, map_location=device).eval()

        self._cmd_scale = torch.tensor(GR00T_CMD_SCALE, device=device)
        self._lower_default = torch.tensor(GR00T_LOWER_BODY_DEFAULT_15, device=device)
        self._upper_default = torch.tensor(DEFAULT_UPPER_BODY_TARGET_29, device=device)
        self._atec_default = torch.tensor(ATEC_G1_DEFAULT_33, device=device)

        # Pre-compute the 29-D default for joint pos centering
        default_29 = torch.zeros(29, device=device)
        default_29[:15] = self._lower_default
        self._default_29 = default_29

        self._num_envs: int = 0
        self._obs_history: list[torch.Tensor] | None = None
        self._last_lower_action: torch.Tensor | None = None
        self._baseline_single_obs: torch.Tensor | None = None
        self._baseline_full_obs: torch.Tensor | None = None
        self._pending_reset_env_ids: torch.Tensor | None = None
        self._pending_reset_compare_printed = False

    def reset(self, num_envs: int) -> None:
        self._num_envs = num_envs
        self._obs_history = None
        self._last_lower_action = torch.zeros(num_envs, GR00T_NUM_ACTIONS, device=self.device)
        self._baseline_single_obs = None
        self._baseline_full_obs = None
        self._pending_reset_env_ids = None
        self._pending_reset_compare_printed = False

    def reset_envs(self, env_ids: torch.Tensor, current_single_obs: torch.Tensor) -> None:
        """Reset GR00T history for specific environments (after auto-reset).

        After auto-reset, the next predict call will append a new obs frame.
        We need the history to look like: [0, 0, 0, 0, 0, <next_obs>].
        Since the next _stack_history will append the new obs and left-pad,
        we set all existing frames to zero for done envs so the result is
        [0, 0, 0, 0, 0, new_obs] (matching adapter's reset behavior).
        """
        if self._last_lower_action is not None:
            self._last_lower_action[env_ids] = 0.0
        if self._obs_history is not None:
            zero_frame = torch.zeros_like(self._obs_history[0])
            for i in range(len(self._obs_history)):
                self._obs_history[i][env_ids] = zero_frame[env_ids]
        self._pending_reset_env_ids = env_ids.detach().clone()

    def debug_compare_full_reset_equivalence(
        self, env_ids: torch.Tensor, next_single_obs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compare current partial-reset state against ideal full-reset semantics.

        Returns diagnostics for the env_ids only. This is read-only and intended
        for debugging reset behavior differences between full reset and auto-reset.
        """
        env_ids = env_ids.reshape(-1)
        if env_ids.numel() == 0:
            return {}

        if self._obs_history is None:
            current_hist = torch.zeros(
                env_ids.numel(), GR00T_NUM_OBS, device=self.device, dtype=next_single_obs.dtype
            )
        else:
            current_hist = torch.cat([frame[env_ids] for frame in self._obs_history], dim=-1)

        zero_frame = torch.zeros_like(next_single_obs[env_ids])
        ideal_frames = [zero_frame for _ in range(GR00T_HISTORY_LEN - 1)] + [next_single_obs[env_ids]]
        ideal_hist = torch.cat(ideal_frames, dim=-1)

        return {
            "env_ids": env_ids.detach().clone(),
            "current_hist_head": current_hist[:, :16].detach().clone(),
            "ideal_hist_head": ideal_hist[:, :16].detach().clone(),
            "history_max_abs_diff": (current_hist - ideal_hist).abs().amax(dim=-1).detach().clone(),
            "last_lower_action_norm": self._last_lower_action[env_ids].norm(dim=-1).detach().clone()
            if self._last_lower_action is not None
            else torch.zeros(env_ids.numel(), device=self.device, dtype=next_single_obs.dtype),
        }

    @torch.no_grad()
    def predict(self, obs_dict: dict, nav_cmd: torch.Tensor) -> torch.Tensor:
        """Run one step of the GR00T policy.

        Parameters
        ----------
        obs_dict : dict
            Environment observation dict with ``"proprio"`` key.
        nav_cmd : torch.Tensor
            Shape ``(num_envs, 3)`` — [vx, vy, vyaw] in m/s and rad/s.

        Returns
        -------
        torch.Tensor
            Shape ``(num_envs, 33)`` — ATEC env action.
        """
        proprio = obs_dict["proprio"]  # (N, 111)
        single_obs = self._build_obs(proprio, nav_cmd)  # (N, 86)
        full_obs = self._stack_history(single_obs)  # (N, 516)

        if self._baseline_single_obs is None:
            self._baseline_single_obs = single_obs.detach().clone()
            self._baseline_full_obs = full_obs.detach().clone()
        elif self._pending_reset_env_ids is not None and not self._pending_reset_compare_printed:
            env_ids = self._pending_reset_env_ids.reshape(-1)
            if env_ids.numel() > 0:
                baseline_single = self._baseline_single_obs[env_ids]
                baseline_full = self._baseline_full_obs[env_ids]
                single_diff = (single_obs[env_ids] - baseline_single).abs().amax(dim=-1)
                full_diff = (full_obs[env_ids] - baseline_full).abs().amax(dim=-1)
                print("[LOW-LEVEL RESET INPUT COMPARE]")
                print(f"  env_ids={env_ids.tolist()}")
                print(f"  single_obs_max_abs_diff={single_diff.tolist()}")
                print(f"  full_obs_max_abs_diff={full_diff.tolist()}")
                print(f"  baseline_single_head[0]={baseline_single[0, :16].tolist()}")
                print(f"  reset_single_head[0]={single_obs[env_ids][0, :16].tolist()}")
                print(f"  baseline_full_tail_head[0]={baseline_full[0, -16:].tolist()}")
                print(f"  reset_full_tail_head[0]={full_obs[env_ids][0, -16:].tolist()}")
            self._pending_reset_compare_printed = True
            self._pending_reset_env_ids = None

        # Select walk vs balance per-env
        nav_norm = torch.norm(nav_cmd, dim=-1)  # (N,)
        use_walk = nav_norm >= 0.05

        # Batched dual-model inference
        out_walk = self.walk(full_obs)
        out_balance = self.balance(full_obs)
        raw_action = torch.where(use_walk.unsqueeze(-1), out_walk, out_balance)

        self._last_lower_action = raw_action
        return self._map_to_env_action(raw_action)  # (N, 33)

    # ── Observation construction ────────────────────────────────────────────

    def _build_obs(self, proprio: torch.Tensor, nav_cmd: torch.Tensor) -> torch.Tensor:
        """Build the 86-D single-timestep GR00T observation.

        Proprio layout (111D):
            [0:3]   base_lin_vel
            [3:6]   base_ang_vel
            [6:9]   velocity_commands (3D)
            [9:12]  projected_gravity
            [12:45] joint_pos_rel (33)
            [45:78] joint_vel_rel (33)
            [78:111] last_env_action (33)
        """
        # Scaled nav command (3D)
        scaled_cmd = nav_cmd * self._cmd_scale  # (N, 3)

        # Base height command (1D) — fixed
        N = proprio.shape[0]
        height_cmd = proprio.new_full((N, 1), 0.74)

        # Torso RPY from waist joints (3D)
        torso_rpy = self._compute_torso_rpy(proprio)

        # Scaled angular velocity (3D)
        ang_vel = proprio[:, 3:6] * GR00T_ANG_VEL_SCALE

        # Projected gravity (3D)
        gravity = proprio[:, 9:12]

        # Joint positions (29D): absolute = default + relative, then subtract default
        joint_pos_rel = proprio[:, 12:45]  # (N, 33)
        joint_pos_abs = self._atec_default.unsqueeze(0) + joint_pos_rel
        joint_pos_29 = joint_pos_abs[:, :29]
        scaled_pos = joint_pos_29 - self._default_29.unsqueeze(0)

        # Joint velocities (29D)
        joint_vel_rel = proprio[:, 45:78]  # (N, 33)
        joint_vel_29 = joint_vel_rel[:, :29]
        scaled_vel = joint_vel_29 * GR00T_DOF_VEL_SCALE

        return torch.cat([
            scaled_cmd,       # 3
            height_cmd,       # 1
            torso_rpy,        # 3
            ang_vel,          # 3
            gravity,          # 3
            scaled_pos,       # 29
            scaled_vel,       # 29
            self._last_lower_action,  # 15
        ], dim=-1)  # 86D

    def _compute_torso_rpy(self, proprio: torch.Tensor) -> torch.Tensor:
        """Compute torso RPY from waist joint angles (analytical FK).

        Waist chain: pelvis -> yaw(z) -> roll(x) -> pitch(y) -> torso.
        The waist joints are at indices [12, 13, 14] in the 33-DoF vector.
        """
        joint_pos_rel = proprio[:, 12:45]  # (N, 33)
        joint_pos_abs = self._atec_default.unsqueeze(0) + joint_pos_rel

        waist_yaw = joint_pos_abs[:, 12]
        waist_roll = joint_pos_abs[:, 13]
        waist_pitch = joint_pos_abs[:, 14]

        # Rz(yaw) @ Rx(roll) @ Ry(pitch) — batched
        cy, sy = torch.cos(waist_yaw), torch.sin(waist_yaw)
        cr, sr = torch.cos(waist_roll), torch.sin(waist_roll)
        cp, sp = torch.cos(waist_pitch), torch.sin(waist_pitch)

        # Only need R[2,0], R[2,1], R[2,2], R[1,0], R[0,0] for RPY extraction
        # R = Rz @ Rx @ Ry
        # R[2,0] = -(cr*sp*cy + cr*cp*sy) ... simplified: -sin(pitch) in certain convention
        # Direct computation is cleaner:

        # Rz row 0: [cy, -sy, 0]; row 1: [sy, cy, 0]; row 2: [0, 0, 1]
        # Rx @ Ry:
        # Rx = [[1,0,0],[0,cr,-sr],[0,sr,cr]]
        # Ry = [[cp,0,sp],[0,1,0],[-sp,0,cp]]
        # Rx @ Ry = [[cp,0,sp],[sr*sp,cr,-sr*cp],[-cr*sp,sr,cr*cp]]
        # Rz @ (Rx @ Ry):
        # R[0,0] = cy*cp - sy*sr*sp
        # R[1,0] = sy*cp + cy*sr*sp
        # R[2,0] = -cr*sp
        # R[2,1] = sr
        # R[2,2] = cr*cp

        pitch_out = torch.asin(torch.clamp(-(-cr * sp), -1.0, 1.0))  # asin(cr*sp)
        # Wait, R[2,0] = -cr*sp, so -R[2,0] = cr*sp, asin(cr*sp) is wrong
        # Let me recalculate properly
        # R[2,0] = -(cr*sp) => asin(-R[2,0]) = asin(cr*sp)... no
        # Standard RPY: p = asin(-R[2,0])
        # R[2,0] = -cr*sp => -R[2,0] = cr*sp => p = asin(cr*sp)
        # Hmm, this doesn't simplify cleanly. Let me just do the full RPY extraction:

        # R[2,0] = -cr*sp  (should be -sin(p) for standard convention)
        # So p = asin(cr*sp) — this isn't standard. The reference uses asin(-R[2,0])
        # In the reference: p = asin(clip(-R[2,0], -1, 1))
        # -R[2,0] = cr*sp
        # But that gives p = asin(cr*sp) which is wrong when cr ≈ 1 (it's just sp ≈ pitch)

        # Actually, looking more carefully at the G1 waist chain convention:
        # The reference code compute_torso_rpy_from_upper_body does:
        # R = Rz(yaw) @ Rx(roll) @ Ry(pitch)
        # p = asin(-R[2,0])  where R[2,0] = -cr*sp
        # so p = asin(cr*sp)
        # When roll ≈ 0, p ≈ asin(sp) ≈ pitch. Good enough.

        r20 = -cr * sp
        pitch_out = torch.asin(torch.clamp(-r20, -1.0, 1.0))
        cp2 = torch.cos(pitch_out)

        # Roll and yaw
        r21 = sr  # R[2,1]
        r22 = cr * cp  # R[2,2]
        roll_out = torch.atan2(r21, r22)

        r10 = sy * cp + cy * sr * sp  # R[1,0]
        r00 = cy * cp - sy * sr * sp  # R[0,0]
        yaw_out = torch.atan2(r10, r00)

        return torch.stack([roll_out, pitch_out, yaw_out], dim=-1)

    # ── History stacking ────────────────────────────────────────────────────

    def _stack_history(self, single_obs: torch.Tensor) -> torch.Tensor:
        """Stack 6 frames of history → (N, 516).

        Matches the adapter's behavior: append current obs, then left-pad
        with zeros to fill the history buffer. This is critical because
        GR00T's estimator expects zero-padded initial history frames.
        """
        if self._obs_history is None:
            self._obs_history = []
        self._obs_history.append(single_obs.detach().clone())
        # Left-pad with zero frames (same as adapter's appendleft)
        while len(self._obs_history) < GR00T_HISTORY_LEN:
            self._obs_history.insert(0, torch.zeros_like(single_obs))
        # Keep only the last 6 frames
        if len(self._obs_history) > GR00T_HISTORY_LEN:
            self._obs_history = self._obs_history[-GR00T_HISTORY_LEN:]
        return torch.cat(self._obs_history, dim=-1)

    # ── Action mapping ─────────────────────────────────────────────────────

    def _map_to_env_action(self, raw_15: torch.Tensor) -> torch.Tensor:
        """Map 15-DoF raw policy output → 33-DoF ATEC env action.

        Steps:
        1. Rescale: target = lower_default + 0.25 * raw_15
        2. Assemble 29-DoF body target (upper body stays at default)
        3. Convert to env relative action: (target - atec_default) / 0.5
        """
        N = raw_15.shape[0]
        lower_target = self._lower_default.unsqueeze(0) + GR00T_ACTION_SCALE * raw_15  # (N, 15)

        body_target_29 = self._upper_default.unsqueeze(0).expand(N, -1).clone()
        body_target_29[:, :15] = lower_target

        env_action = torch.zeros(N, 33, device=self.device)
        env_action[:, :29] = (body_target_29 - self._atec_default[:29].unsqueeze(0)) / ATEC_ENV_ACTION_SCALE
        return env_action
