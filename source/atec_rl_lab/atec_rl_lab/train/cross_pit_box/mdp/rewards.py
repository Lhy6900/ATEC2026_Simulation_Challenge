from __future__ import annotations

import torch

from .frames import root_pos_env


def _robot(env, asset_cfg=None):
    return env.scene[getattr(asset_cfg, "name", "robot")]


def _resolve_success_x(env, success_x: float, success_x_attr: str | None = None) -> float:
    if success_x_attr is None:
        return success_x
    return float(getattr(env, success_x_attr, success_x))


def _stage_reward_unlocked(
    env,
    unlock_success_x: float | None,
    success_x: float | None,
    success_x_attr: str | None,
) -> bool:
    if unlock_success_x is None:
        return True
    active_success_x = _resolve_success_x(env, success_x if success_x is not None else unlock_success_x, success_x_attr)
    return active_success_x >= unlock_success_x


def _non_failure_mask(
    asset,
    pos: torch.Tensor,
    min_height: float | None = None,
    max_abs_gravity_xy: float | None = None,
) -> torch.Tensor:
    mask = torch.ones(pos.shape[0], device=pos.device, dtype=torch.bool)
    if min_height is not None:
        mask &= pos[:, 2] >= min_height
    if max_abs_gravity_xy is not None:
        mask &= torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=-1) <= max_abs_gravity_xy
    return mask


def crossing_progress(
    env,
    start_x: float,
    success_x: float,
    success_x_attr: str | None = None,
    asset_cfg=None,
) -> torch.Tensor:
    """Normalized base x progress from counter-650 start toward the far side."""
    asset = _robot(env, asset_cfg)
    success_x = _resolve_success_x(env, success_x, success_x_attr)
    denom = max(success_x - start_x, 1.0e-6)
    return torch.clamp((root_pos_env(env, asset)[:, 0] - start_x) / denom, 0.0, 1.0)


def _x_progress_potential(
    env,
    start_x: float,
    success_x: float,
    success_x_attr: str | None,
    potential_scale: float,
    target_y: float,
    safe_abs_y: float,
    min_upright: float,
    failure_min_height: float | None = None,
    failure_max_abs_gravity_xy: float | None = None,
    asset_cfg=None,
) -> torch.Tensor:
    asset = _robot(env, asset_cfg)
    pos = root_pos_env(env, asset)
    active_success_x = _resolve_success_x(env, success_x, success_x_attr)
    progress = torch.clamp((pos[:, 0] - start_x) / max(active_success_x - start_x, 1.0e-6), 0.0, 1.0)
    progress = torch.where(
        _non_failure_mask(asset, pos, min_height=failure_min_height, max_abs_gravity_xy=failure_max_abs_gravity_xy),
        progress,
        torch.zeros_like(progress),
    )
    centered = torch.clamp(1.0 - torch.abs(pos[:, 1] - target_y) / max(safe_abs_y, 1.0e-6), 0.0, 1.0)
    upright_raw = -asset.data.projected_gravity_b[:, 2]
    upright = torch.clamp((upright_raw - min_upright) / max(1.0 - min_upright, 1.0e-6), 0.0, 1.0)
    return float(potential_scale) * progress * centered * upright


def potential_based_x_progress(
    env,
    start_x: float,
    success_x: float,
    success_x_attr: str | None = None,
    gamma: float = 0.99,
    potential_scale: float = 50.0,
    target_y: float = 0.0,
    safe_abs_y: float = 0.55,
    min_upright: float = 0.65,
    failure_min_height: float | None = None,
    failure_max_abs_gravity_xy: float | None = None,
    unlock_success_x: float | None = None,
    cache_key: str = "_cross_pit_pbrs_x_phi",
    asset_cfg=None,
) -> torch.Tensor:
    """Potential-based x shaping: F(s, s') = gamma * Phi(s') - Phi(s).

    The cached Phi is reset on the first step of each episode to avoid leaking
    terminal-state potential into the next reset state.
    """
    phi = _x_progress_potential(
        env,
        start_x=start_x,
        success_x=success_x,
        success_x_attr=success_x_attr,
        potential_scale=potential_scale,
        target_y=target_y,
        safe_abs_y=safe_abs_y,
        min_upright=min_upright,
        failure_min_height=failure_min_height,
        failure_max_abs_gravity_xy=failure_max_abs_gravity_xy,
        asset_cfg=asset_cfg,
    )
    if not _stage_reward_unlocked(env, unlock_success_x, success_x, success_x_attr):
        zero = torch.zeros_like(phi)
        setattr(env, cache_key, zero.detach().clone())
        return zero

    prev_phi = getattr(env, cache_key, None)
    if prev_phi is None or tuple(prev_phi.shape) != tuple(phi.shape):
        setattr(env, cache_key, phi.detach().clone())
        return torch.zeros_like(phi)
    prev_phi = prev_phi.to(device=phi.device, dtype=phi.dtype)
    reward = float(gamma) * phi - prev_phi

    episode_length = getattr(env, "episode_length_buf", None)
    if episode_length is not None:
        first_step = episode_length.to(device=phi.device) <= 1
        reward = torch.where(first_step, torch.zeros_like(reward), reward)

    setattr(env, cache_key, phi.detach().clone())
    return reward


def incremental_crossing_progress(
    env,
    start_x: float,
    success_x: float,
    success_x_attr: str | None = None,
    max_step_progress: float = 0.02,
    min_height: float | None = None,
    max_abs_gravity_xy: float | None = None,
    asset_cfg=None,
) -> torch.Tensor:
    """Reward only forward progress made this step, so standing at a large x cannot harvest reward."""
    asset = _robot(env, asset_cfg)
    success_x = _resolve_success_x(env, success_x, success_x_attr)
    pos = root_pos_env(env, asset)
    in_course = (pos[:, 0] >= start_x) & (pos[:, 0] <= success_x)
    in_course &= _non_failure_mask(asset, pos, min_height=min_height, max_abs_gravity_xy=max_abs_gravity_xy)
    step_dt = float(getattr(env, "step_dt", 0.02))
    step_progress = torch.clamp(asset.data.root_lin_vel_w[:, 0] * step_dt, 0.0, max_step_progress)
    return torch.where(in_course, step_progress / max(max_step_progress, 1.0e-6), torch.zeros_like(step_progress))


def forward_velocity_reward(
    env,
    max_velocity: float = 1.0,
    min_height: float | None = None,
    max_abs_gravity_xy: float | None = None,
    asset_cfg=None,
) -> torch.Tensor:
    """Reward forward world-x velocity, clipped to avoid rewarding jumps or falls too much."""
    asset = _robot(env, asset_cfg)
    pos = root_pos_env(env, asset)
    reward = torch.clamp(asset.data.root_lin_vel_w[:, 0], 0.0, max_velocity) / max(max_velocity, 1.0e-6)
    return torch.where(
        _non_failure_mask(asset, pos, min_height=min_height, max_abs_gravity_xy=max_abs_gravity_xy),
        reward,
        torch.zeros_like(reward),
    )


def crossing_milestone_reward(
    env,
    down_step_x: float,
    box_x: float,
    far_lip_x: float,
    success_x: float,
    success_x_attr: str | None = None,
    margin: float = 0.15,
    min_height: float | None = None,
    max_abs_gravity_xy: float | None = None,
    asset_cfg=None,
) -> torch.Tensor:
    """Dense staged reward for down-step, box-support, far-lip, and far-side milestones."""
    asset = _robot(env, asset_cfg)
    success_x = _resolve_success_x(env, success_x, success_x_attr)
    pos = root_pos_env(env, asset)
    x = pos[:, 0]
    inv_margin = 1.0 / max(margin, 1.0e-6)
    stages = (
        torch.sigmoid((x - down_step_x) * inv_margin)
        + torch.sigmoid((x - box_x) * inv_margin)
        + torch.sigmoid((x - far_lip_x) * inv_margin)
        + torch.sigmoid((x - success_x) * inv_margin)
    )
    reward = stages / 4.0
    return torch.where(
        _non_failure_mask(asset, pos, min_height=min_height, max_abs_gravity_xy=max_abs_gravity_xy),
        reward,
        torch.zeros_like(reward),
    )


def box_support_region_reward(
    env,
    box_x: float,
    half_length: float,
    half_width: float,
    min_height: float,
    box_y: float = 0.0,
    success_x: float | None = None,
    success_x_attr: str | None = None,
    unlock_success_x: float | None = None,
    asset_cfg=None,
) -> torch.Tensor:
    """Reward being upright over the fixed box support region in the pit."""
    asset = _robot(env, asset_cfg)
    pos = root_pos_env(env, asset)
    if not _stage_reward_unlocked(env, unlock_success_x, success_x, success_x_attr):
        return torch.zeros(pos.shape[0], device=pos.device, dtype=pos.dtype)
    upright = -asset.data.projected_gravity_b[:, 2] > 0.8
    over_box = (torch.abs(pos[:, 0] - box_x) <= half_length) & (torch.abs(pos[:, 1] - box_y) <= half_width)
    high_enough = pos[:, 2] >= min_height
    return (over_box & upright & high_enough).float()


def box_gap_step_reward(
    env,
    near_lip_x: float,
    box_front_x: float,
    box_center_x: float,
    target_y: float = 0.0,
    min_forward_velocity: float = 0.55,
    max_abs_y: float = 0.35,
    max_up_velocity: float = 0.35,
    margin: float = 0.12,
    success_x: float | None = None,
    success_x_attr: str | None = None,
    unlock_success_x: float | None = None,
    asset_cfg=None,
) -> torch.Tensor:
    """Reward a committed step over the near-side gap onto the box front half."""
    asset = _robot(env, asset_cfg)
    pos = root_pos_env(env, asset)
    if not _stage_reward_unlocked(env, unlock_success_x, success_x, success_x_attr):
        return torch.zeros(pos.shape[0], device=pos.device, dtype=pos.dtype)
    vel = asset.data.root_lin_vel_w
    inv_margin = 1.0 / max(margin, 1.0e-6)
    after_lip = torch.sigmoid((pos[:, 0] - near_lip_x) * inv_margin)
    before_center = torch.sigmoid((box_center_x - pos[:, 0]) * inv_margin)
    box_front_band = torch.sigmoid((pos[:, 0] - box_front_x) * inv_margin) * before_center
    centered = torch.clamp(1.0 - torch.abs(pos[:, 1] - target_y) / max(max_abs_y, 1.0e-6), 0.0, 1.0)
    forward = torch.clamp((vel[:, 0] - min_forward_velocity) / max(min_forward_velocity, 1.0e-6), 0.0, 1.0)
    lift = torch.clamp(vel[:, 2] / max(max_up_velocity, 1.0e-6), 0.0, 1.0)
    return after_lip * box_front_band * centered * torch.maximum(forward, lift)


def box_stability_reward(
    env,
    box_x: float,
    half_length: float,
    half_width: float,
    min_height: float,
    box_y: float = 0.0,
    min_forward_velocity: float = 0.0,
    max_forward_velocity: float = 0.65,
    success_x: float | None = None,
    success_x_attr: str | None = None,
    unlock_success_x: float | None = None,
    asset_cfg=None,
) -> torch.Tensor:
    """Reward stable, centered support on the box before attempting the far-side climb."""
    asset = _robot(env, asset_cfg)
    pos = root_pos_env(env, asset)
    if not _stage_reward_unlocked(env, unlock_success_x, success_x, success_x_attr):
        return torch.zeros(pos.shape[0], device=pos.device, dtype=pos.dtype)
    vel = asset.data.root_lin_vel_w
    upright = -asset.data.projected_gravity_b[:, 2] > 0.82
    over_box = (torch.abs(pos[:, 0] - box_x) <= half_length) & (torch.abs(pos[:, 1] - box_y) <= half_width)
    high_enough = pos[:, 2] >= min_height
    controlled_forward = (vel[:, 0] >= min_forward_velocity) & (vel[:, 0] <= max_forward_velocity)
    return (over_box & upright & high_enough & controlled_forward).float()


def box_parking_penalty(
    env,
    box_front_x: float,
    target_y: float = 0.0,
    success_x: float | None = None,
    success_x_attr: str | None = None,
    unlock_success_x: float | None = None,
    target_margin: float = 0.12,
    max_forward_velocity: float = 0.08,
    max_abs_y: float = 0.35,
    asset_cfg=None,
) -> torch.Tensor:
    """Penalize parking after reaching the box/front stage without approaching the active target."""
    asset = _robot(env, asset_cfg)
    pos = root_pos_env(env, asset)
    if not _stage_reward_unlocked(env, unlock_success_x, success_x, success_x_attr):
        return torch.zeros(pos.shape[0], device=pos.device, dtype=pos.dtype)
    active_success_x = _resolve_success_x(env, success_x if success_x is not None else box_front_x, success_x_attr)
    after_box_front = pos[:, 0] >= box_front_x
    before_target_margin = pos[:, 0] < active_success_x - target_margin
    centered = torch.abs(pos[:, 1] - target_y) <= max_abs_y
    stalled = asset.data.root_lin_vel_w[:, 0] <= max_forward_velocity
    return (after_box_front & before_target_margin & centered & stalled).float()


def stage_progress_to_target(
    env,
    stage_start_x: float,
    success_x: float,
    success_x_attr: str | None = None,
    unlock_success_x: float | None = None,
    asset_cfg=None,
) -> torch.Tensor:
    """Reward hard-stage progress toward the active curriculum target."""
    asset = _robot(env, asset_cfg)
    pos = root_pos_env(env, asset)
    if not _stage_reward_unlocked(env, unlock_success_x, success_x, success_x_attr):
        return torch.zeros(pos.shape[0], device=pos.device, dtype=pos.dtype)
    active_success_x = _resolve_success_x(env, success_x, success_x_attr)
    denom = max(active_success_x - stage_start_x, 1.0e-6)
    return torch.clamp((pos[:, 0] - stage_start_x) / denom, 0.0, 1.0)


def stage_stall_penalty(
    env,
    stage_start_x: float,
    target_y: float = 0.0,
    success_x: float | None = None,
    success_x_attr: str | None = None,
    unlock_success_x: float | None = None,
    target_margin: float = 0.20,
    min_forward_velocity: float = 0.12,
    safe_abs_y: float = 0.45,
    min_upright: float = 0.75,
    asset_cfg=None,
) -> torch.Tensor:
    """Penalize centered, upright hard-stage states that stop before the active target."""
    asset = _robot(env, asset_cfg)
    pos = root_pos_env(env, asset)
    if not _stage_reward_unlocked(env, unlock_success_x, success_x, success_x_attr):
        return torch.zeros(pos.shape[0], device=pos.device, dtype=pos.dtype)
    active_success_x = _resolve_success_x(env, success_x if success_x is not None else stage_start_x, success_x_attr)
    in_hard_stage = pos[:, 0] >= stage_start_x
    before_target_margin = pos[:, 0] < active_success_x - target_margin
    centered = torch.abs(pos[:, 1] - target_y) <= safe_abs_y
    upright = -asset.data.projected_gravity_b[:, 2] >= min_upright
    stalled = asset.data.root_lin_vel_w[:, 0] < min_forward_velocity
    return (in_hard_stage & before_target_margin & centered & upright & stalled).float()


def far_platform_region_reward(
    env,
    far_lip_x: float,
    platform_x: float,
    half_width: float,
    min_height: float,
    margin: float = 0.20,
    target_y: float = 0.0,
    asset_cfg=None,
) -> torch.Tensor:
    """Reward upright, centered states after climbing onto the far platform."""
    asset = _robot(env, asset_cfg)
    pos = root_pos_env(env, asset)
    upright = -asset.data.projected_gravity_b[:, 2] > 0.8
    centered = torch.abs(pos[:, 1] - target_y) <= half_width
    high_enough = pos[:, 2] >= min_height
    after_lip = torch.sigmoid((pos[:, 0] - far_lip_x) / max(margin, 1.0e-6))
    on_platform = torch.clamp((pos[:, 0] - platform_x + margin) / max(margin, 1.0e-6), 0.0, 1.0)
    return after_lip * on_platform * (upright & centered & high_enough).float()


def far_side_commit_reward(
    env,
    far_lip_x: float,
    platform_x: float,
    target_y: float = 0.0,
    half_width: float = 0.45,
    min_height: float = 0.50,
    min_forward_velocity: float = 0.20,
    margin: float = 0.20,
    asset_cfg=None,
) -> torch.Tensor:
    """Reward committed motion beyond the far lip before the final platform is fully stabilized."""
    asset = _robot(env, asset_cfg)
    pos = root_pos_env(env, asset)
    vel = asset.data.root_lin_vel_w
    upright = -asset.data.projected_gravity_b[:, 2] > 0.8
    centered = torch.abs(pos[:, 1] - target_y) <= half_width
    after_far_lip = pos[:, 0] >= far_lip_x
    platform_progress = torch.clamp((pos[:, 0] - far_lip_x) / max(platform_x - far_lip_x, 1.0e-6), 0.0, 1.0)
    forward = torch.clamp((vel[:, 0] - min_forward_velocity) / max(min_forward_velocity, 1.0e-6), 0.0, 1.0)
    high_enough = torch.clamp((pos[:, 2] - min_height) / max(min_height, 1.0e-6), 0.0, 1.0)
    return after_far_lip.float() * platform_progress * torch.maximum(forward, platform_progress) * high_enough * (
        upright & centered
    ).float()


def crossing_success(
    env,
    success_x: float,
    success_x_attr: str | None = None,
    min_height: float = 0.55,
    max_abs_y: float | None = None,
    target_y: float = 0.0,
    asset_cfg=None,
) -> torch.Tensor:
    """One-step success reward after reaching the far side while still upright enough."""
    asset = _robot(env, asset_cfg)
    success_x = _resolve_success_x(env, success_x, success_x_attr)
    pos = root_pos_env(env, asset)
    stable = (pos[:, 2] >= min_height) & (-asset.data.projected_gravity_b[:, 2] > 0.75)
    if max_abs_y is not None:
        stable = stable & (torch.abs(pos[:, 1] - target_y) <= max_abs_y)
    return ((pos[:, 0] >= success_x) & stable).float()


def upright_reward(env, asset_cfg=None) -> torch.Tensor:
    """Small auxiliary reward for keeping the torso upright."""
    asset = _robot(env, asset_cfg)
    return torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)


def lateral_deviation_l2(env, target_y: float = 0.0, asset_cfg=None) -> torch.Tensor:
    """Squared lateral drift penalty away from the straight counter-650 crossing line."""
    asset = _robot(env, asset_cfg)
    return torch.square(root_pos_env(env, asset)[:, 1] - target_y)


def lateral_corridor_barrier(
    env,
    safe_abs_y: float = 0.45,
    max_abs_y: float = 1.25,
    target_y: float = 0.0,
    asset_cfg=None,
) -> torch.Tensor:
    """Soft barrier that grows only after the robot leaves the centered crossing corridor."""
    asset = _robot(env, asset_cfg)
    abs_y = torch.abs(root_pos_env(env, asset)[:, 1] - target_y)
    span = max(max_abs_y - safe_abs_y, 1.0e-6)
    normalized_excess = torch.clamp((abs_y - safe_abs_y) / span, min=0.0, max=1.0)
    return torch.square(normalized_excess)


def lateral_velocity_l2(env, asset_cfg=None) -> torch.Tensor:
    """Squared world-y velocity penalty to reduce drift before lateral termination."""
    asset = _robot(env, asset_cfg)
    return torch.square(asset.data.root_lin_vel_w[:, 1])


def crossing_failure_penalty(
    env,
    min_height: float = 0.30,
    max_abs_gravity_xy: float = 0.75,
    max_abs_y: float = 1.25,
    min_x: float = -2.10,
    target_y: float = 0.0,
    normalize_by_step_dt: bool = False,
    asset_cfg=None,
) -> torch.Tensor:
    """Penalty only for failure modes, not for successful crossing termination."""
    asset = _robot(env, asset_cfg)
    pos = root_pos_env(env, asset)
    low = pos[:, 2] < min_height
    tilt = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=-1) > max_abs_gravity_xy
    lateral = torch.abs(pos[:, 1] - target_y) > max_abs_y
    backward = pos[:, 0] < min_x
    penalty = (low | tilt | lateral | backward).float()
    if normalize_by_step_dt:
        penalty = penalty / max(float(getattr(env, "step_dt", 0.02)), 1.0e-6)
    return penalty


def time_penalty(env) -> torch.Tensor:
    """Per-step time pressure so timeout is not an attractive local optimum."""
    return torch.ones(env.num_envs, device=env.device)


def stage_time_penalty(
    env,
    success_x: float | None = None,
    success_x_attr: str | None = None,
    unlock_success_x: float | None = None,
) -> torch.Tensor:
    """Additional per-step pressure that turns on only after the active hard stage unlocks."""
    if not _stage_reward_unlocked(env, unlock_success_x, success_x, success_x_attr):
        return torch.zeros(env.num_envs, device=env.device)
    return torch.ones(env.num_envs, device=env.device)
