from __future__ import annotations

import torch

from .frames import root_pos_env


def _robot(env, asset_cfg=None):
    return env.scene[getattr(asset_cfg, "name", "robot")]


def _resolve_success_x(env, success_x: float, success_x_attr: str | None = None) -> float:
    if success_x_attr is None:
        return success_x
    return float(getattr(env, success_x_attr, success_x))


def _apply_affine_action_transform(term, actions: torch.Tensor) -> torch.Tensor:
    processed = actions
    if hasattr(term, "_scale"):
        processed = processed * term._scale
    if hasattr(term, "_offset"):
        processed = processed + term._offset
    if getattr(term, "cfg", None) is not None and getattr(term.cfg, "clip", None) is not None and hasattr(term, "_clip"):
        processed = torch.clamp(processed, min=term._clip[:, :, 0], max=term._clip[:, :, 1])
    return processed


def sync_handoff_action_state(env, action_name: str = "joint_pos") -> torch.Tensor:
    """Restore the captured handoff action into manager, action term, and actuator target state."""
    action = env.action_manager.action
    reset_action = getattr(env, "_cross_pit_reset_last_action", None)
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if reset_action is None or episode_length_buf is None:
        return action

    reset_action = reset_action.to(device=action.device, dtype=action.dtype)
    if reset_action.shape != action.shape:
        return action

    reset_mask = episode_length_buf.to(device=action.device) == 0
    if not torch.any(reset_mask):
        return action

    if hasattr(env.action_manager, "_action"):
        env.action_manager._action[reset_mask] = reset_action[reset_mask]
    if hasattr(env.action_manager, "_prev_action"):
        env.action_manager._prev_action[reset_mask] = reset_action[reset_mask]

    if action_name is None:
        return env.action_manager.action

    try:
        term = env.action_manager.get_term(action_name)
    except (KeyError, AttributeError):
        return env.action_manager.action
    if not hasattr(term, "_raw_actions") or term.raw_actions.shape != reset_action.shape:
        return env.action_manager.action

    term._raw_actions[reset_mask] = reset_action[reset_mask]
    if hasattr(term, "_processed_actions"):
        processed = _apply_affine_action_transform(term, reset_action)
        term._processed_actions[reset_mask] = processed[reset_mask]
        asset = getattr(term, "_asset", None)
        if asset is not None and hasattr(asset, "set_joint_position_target"):
            asset.set_joint_position_target(term._processed_actions, joint_ids=getattr(term, "_joint_ids", None))
    return env.action_manager.action


def base_pos_env(env, asset_cfg=None) -> torch.Tensor:
    """Privileged critic observation: base position in replicated env-local coordinates."""
    return root_pos_env(env, _robot(env, asset_cfg))


def last_action(env, action_name: str | None = None) -> torch.Tensor:
    """Use the captured handoff command for the first post-reset observation only."""
    if action_name is None:
        action = env.action_manager.action
    else:
        action = env.action_manager.get_term(action_name).raw_actions
    reset_action = getattr(env, "_cross_pit_reset_last_action", None)
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if reset_action is None or episode_length_buf is None or action_name is not None:
        return action

    action = sync_handoff_action_state(env)
    reset_action = getattr(env, "_cross_pit_reset_last_action", None)
    episode_length_buf = getattr(env, "episode_length_buf", None)
    if reset_action is None or episode_length_buf is None:
        return action

    reset_action = reset_action.to(device=action.device, dtype=action.dtype)
    if reset_action.shape != action.shape:
        return action
    reset_mask = episode_length_buf == 0
    if not torch.any(reset_mask):
        return action
    if hasattr(env.action_manager, "_action") and hasattr(env.action_manager, "_prev_action"):
        env.action_manager._action[reset_mask] = reset_action[reset_mask]
        env.action_manager._prev_action[reset_mask] = reset_action[reset_mask]
    return torch.where(reset_mask.unsqueeze(-1), reset_action, action)


def box_center_lateral_error(env, box_y: float = 0.0, asset_cfg=None) -> torch.Tensor:
    """Privileged critic observation: signed base-y offset from the box centerline."""
    return root_pos_env(env, _robot(env, asset_cfg))[:, 1:2] - box_y


def abs_lateral_error(env, box_y: float = 0.0, asset_cfg=None) -> torch.Tensor:
    """Privileged critic observation: absolute base-y offset from the box centerline."""
    return torch.abs(box_center_lateral_error(env, box_y=box_y, asset_cfg=asset_cfg))


def target_x_error(env, success_x: float, success_x_attr: str | None = None, asset_cfg=None) -> torch.Tensor:
    """Privileged critic observation: remaining env-local x distance to the active crossing target."""
    target_x = _resolve_success_x(env, success_x, success_x_attr)
    return target_x - root_pos_env(env, _robot(env, asset_cfg))[:, 0:1]


def crossing_stage_privileged(
    env,
    near_lip_x: float,
    box_front_x: float,
    box_center_x: float,
    box_back_x: float,
    far_lip_x: float,
    far_platform_x: float,
    success_x: float,
    success_x_attr: str | None = None,
    margin: float = 0.12,
    asset_cfg=None,
) -> torch.Tensor:
    """Privileged critic observation: smooth phase activations for the crossing stages."""
    asset = _robot(env, asset_cfg)
    x = root_pos_env(env, asset)[:, 0:1]
    success_x = _resolve_success_x(env, success_x, success_x_attr)
    inv_margin = 1.0 / max(margin, 1.0e-6)
    return torch.cat(
        (
            torch.sigmoid((x - near_lip_x) * inv_margin),
            torch.sigmoid((x - box_front_x) * inv_margin),
            torch.sigmoid((x - box_center_x) * inv_margin),
            torch.sigmoid((x - box_back_x) * inv_margin),
            torch.sigmoid((x - far_lip_x) * inv_margin),
            torch.sigmoid((x - far_platform_x) * inv_margin),
            torch.sigmoid((x - success_x) * inv_margin),
        ),
        dim=-1,
    )


def box_gap_privileged(
    env,
    near_lip_x: float,
    box_front_x: float,
    box_center_x: float,
    box_back_x: float,
    far_lip_x: float,
    far_platform_x: float,
    success_x: float,
    success_x_attr: str | None = None,
    asset_cfg=None,
) -> torch.Tensor:
    """Privileged critic observation: signed landmark distances around the gap and box."""
    asset = _robot(env, asset_cfg)
    x = root_pos_env(env, asset)[:, 0:1]
    success_x = _resolve_success_x(env, success_x, success_x_attr)
    span = max(success_x - near_lip_x, 1.0e-6)
    return torch.cat(
        (
            (x - near_lip_x) / span,
            (x - box_front_x) / span,
            (x - box_center_x) / span,
            (x - box_back_x) / span,
            (x - far_lip_x) / span,
            (x - far_platform_x) / span,
            (success_x - x) / span,
        ),
        dim=-1,
    )


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_vec = quat[:, 1:]
    quat_w = quat[:, 0:1]
    t = 2.0 * torch.cross(quat_vec, vec, dim=-1)
    return vec + quat_w * t + torch.cross(quat_vec, t, dim=-1)


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_inv = torch.cat((quat[:, 0:1], -quat[:, 1:]), dim=-1)
    return _quat_apply(quat_inv, vec)


def depth_image_to_heightmap(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_pos_w: torch.Tensor,
    camera_quat_w: torch.Tensor,
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
    *,
    grid_shape: tuple[int, int] = (24, 16),
    x_range: tuple[float, float] = (0.0, 2.4),
    y_range: tuple[float, float] = (-0.8, 0.8),
    z_clip: tuple[float, float] = (-1.0, 1.0),
    default_height: float = -1.0,
    sample_stride: int = 1,
) -> torch.Tensor:
    """Project world-frame camera depth into a compact base-frame height grid."""
    if depth.dim() == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.dim() != 3:
        raise ValueError(f"Expected depth shape (N, H, W[, 1]), got {tuple(depth.shape)}")
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")

    device = depth.device
    num_envs, height, width = depth.shape
    depth = depth[:, ::sample_stride, ::sample_stride]
    rows = torch.arange(0, height, sample_stride, device=device, dtype=depth.dtype)
    cols = torch.arange(0, width, sample_stride, device=device, dtype=depth.dtype)
    v, u = torch.meshgrid(rows, cols, indexing="ij")
    u = u.reshape(-1)
    v = v.reshape(-1)
    z = depth.reshape(num_envs, -1)

    fx = intrinsics[:, 0, 0].clamp_min(1.0e-6).unsqueeze(1)
    fy = intrinsics[:, 1, 1].clamp_min(1.0e-6).unsqueeze(1)
    cx = intrinsics[:, 0, 2].unsqueeze(1)
    cy = intrinsics[:, 1, 2].unsqueeze(1)

    x_cam = z
    y_cam = -(u.unsqueeze(0) - cx) / fx * z
    z_cam = -(v.unsqueeze(0) - cy) / fy * z
    points_cam = torch.stack((x_cam, y_cam, z_cam), dim=-1)

    points_world = _quat_apply(
        camera_quat_w[:, None, :].expand(-1, points_cam.shape[1], -1).reshape(-1, 4),
        points_cam.reshape(-1, 3),
    ).reshape(num_envs, -1, 3)
    points_world = points_world + camera_pos_w[:, None, :]

    points_base = _quat_apply_inverse(
        base_quat_w[:, None, :].expand(-1, points_world.shape[1], -1).reshape(-1, 4),
        (points_world - base_pos_w[:, None, :]).reshape(-1, 3),
    ).reshape(num_envs, -1, 3)

    nx, ny = grid_shape
    heightmap = torch.full((num_envs, nx, ny), default_height, device=device, dtype=depth.dtype)
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_span = max(x_max - x_min, 1.0e-6)
    y_span = max(y_max - y_min, 1.0e-6)

    for env_id in range(num_envs):
        pts = points_base[env_id]
        valid = (
            torch.isfinite(z[env_id])
            & torch.isfinite(pts).all(dim=-1)
            & (z[env_id] > 0.0)
            & (pts[:, 0] >= x_min)
            & (pts[:, 0] <= x_max)
            & (pts[:, 1] >= y_min)
            & (pts[:, 1] <= y_max)
        )
        if not torch.any(valid):
            continue

        valid_pts = pts[valid]
        ix = torch.clamp(((valid_pts[:, 0] - x_min) / x_span * nx).long(), 0, nx - 1)
        iy = torch.clamp(((valid_pts[:, 1] - y_min) / y_span * ny).long(), 0, ny - 1)
        heights = torch.clamp(valid_pts[:, 2], z_clip[0], z_clip[1])
        flat_index = ix * ny + iy
        flat = heightmap[env_id].reshape(-1)
        flat.scatter_reduce_(0, flat_index, heights, reduce="amax", include_self=True)

    return heightmap.reshape(num_envs, nx * ny)


def ray_hits_to_heightmap(
    ray_hits_w: torch.Tensor,
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
    *,
    grid_shape: tuple[int, int] = (24, 16),
    z_clip: tuple[float, float] = (-1.0, 1.0),
    default_height: float = -1.0,
) -> torch.Tensor:
    """Convert world-frame ray hit points into a flattened base-frame height grid."""
    if ray_hits_w.dim() != 3 or ray_hits_w.shape[-1] != 3:
        raise ValueError(f"Expected ray hits shape (N, R, 3), got {tuple(ray_hits_w.shape)}")
    expected_rays = grid_shape[0] * grid_shape[1]
    if ray_hits_w.shape[1] != expected_rays:
        raise ValueError(f"Expected {expected_rays} rays for grid shape {grid_shape}, got {ray_hits_w.shape[1]}")

    num_envs = ray_hits_w.shape[0]
    points_base = _quat_apply_inverse(
        base_quat_w[:, None, :].expand(-1, expected_rays, -1).reshape(-1, 4),
        (ray_hits_w - base_pos_w[:, None, :]).reshape(-1, 3),
    ).reshape(num_envs, expected_rays, 3)

    heights = torch.clamp(points_base[..., 2], z_clip[0], z_clip[1])
    valid = torch.isfinite(ray_hits_w).all(dim=-1) & torch.isfinite(heights)
    return torch.where(valid, heights, torch.full_like(heights, default_height))


def raycast_heightmap(
    env,
    sensor_cfg=None,
    asset_cfg=None,
    grid_shape: tuple[int, int] = (24, 16),
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_clip: tuple[float, float] = (-1.0, 1.0),
    default_height: float = -1.0,
) -> torch.Tensor:
    """Return a flattened base-frame heightmap from the counter-650 ray-cast scanner."""
    sensor_name = getattr(sensor_cfg, "name", "depth_scanner")
    asset_name = getattr(asset_cfg, "name", "robot")
    sensor = env.scene.sensors[sensor_name]
    asset = env.scene[asset_name]
    return ray_hits_to_heightmap(
        sensor.data.ray_hits_w,
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        grid_shape=grid_shape,
        z_clip=z_clip,
        default_height=default_height,
    )
