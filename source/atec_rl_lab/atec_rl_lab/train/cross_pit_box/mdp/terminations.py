from __future__ import annotations

import torch

from .frames import root_pos_env


def _resolve_success_x(env, success_x: float, success_x_attr: str | None = None) -> float:
    if success_x_attr is None:
        return success_x
    return float(getattr(env, success_x_attr, success_x))


def fallen_or_tilted(env, min_height: float = 0.30, max_abs_gravity_xy: float = 0.75, asset_cfg=None) -> torch.Tensor:
    """Terminate when the torso is too low or heavily tilted."""
    asset = env.scene[getattr(asset_cfg, "name", "robot")]
    low = root_pos_env(env, asset)[:, 2] < min_height
    tilt = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=-1) > max_abs_gravity_xy
    return low | tilt


def crossed_target(
    env,
    success_x: float,
    success_x_attr: str | None = None,
    min_height: float = 0.55,
    max_abs_y: float | None = None,
    target_y: float = 0.0,
    asset_cfg=None,
) -> torch.Tensor:
    """Terminate successful episodes after reaching the far side upright."""
    asset = env.scene[getattr(asset_cfg, "name", "robot")]
    success_x = _resolve_success_x(env, success_x, success_x_attr)
    pos = root_pos_env(env, asset)
    stable = (pos[:, 2] >= min_height) & (-asset.data.projected_gravity_b[:, 2] > 0.75)
    if max_abs_y is not None:
        stable = stable & (torch.abs(pos[:, 1] - target_y) <= max_abs_y)
    return (pos[:, 0] >= success_x) & stable


def lateral_out_of_bounds(
    env,
    max_abs_y: float = 1.25,
    target_y: float = 0.0,
    asset_cfg=None,
) -> torch.Tensor:
    """Terminate large lateral drift that no longer represents the counter-650 crossing attempt."""
    asset = env.scene[getattr(asset_cfg, "name", "robot")]
    return torch.abs(root_pos_env(env, asset)[:, 1] - target_y) > max_abs_y


def backtracked(env, min_x: float, asset_cfg=None) -> torch.Tensor:
    """Terminate policies that walk away from the fixed crossing scene."""
    asset = env.scene[getattr(asset_cfg, "name", "robot")]
    return root_pos_env(env, asset)[:, 0] < min_x
