from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils


def _sample_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose: tuple[float, ...],
    pose_noise: dict[str, tuple[float, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    device = env.device
    base = torch.tensor(pose, device=device, dtype=torch.float32)
    if base.numel() not in (4, 7):
        raise ValueError(f"Expected pose length 4 or 7, got {base.numel()}")

    ranges = [pose_noise.get(key, (0.0, 0.0)) for key in ("x", "y", "z")]
    limits = torch.tensor(ranges, device=device, dtype=torch.float32)
    pos_noise = math_utils.sample_uniform(limits[:, 0], limits[:, 1], (len(env_ids), 3), device=device)
    positions = base[:3].unsqueeze(0) + pos_noise + env.scene.env_origins[env_ids]

    yaw_range = pose_noise.get("yaw", (0.0, 0.0))
    yaw_noise = math_utils.sample_uniform(
        torch.tensor(yaw_range[0], device=device, dtype=torch.float32),
        torch.tensor(yaw_range[1], device=device, dtype=torch.float32),
        (len(env_ids),),
        device=device,
    )

    if base.numel() == 4:
        yaw = base[3] + yaw_noise
        zeros = torch.zeros_like(yaw)
        quats = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
        return positions, quats

    base_quat = base[3:7].unsqueeze(0).expand(len(env_ids), -1)
    if torch.allclose(yaw_noise, torch.zeros_like(yaw_noise)):
        return positions, base_quat

    zeros = torch.zeros_like(yaw_noise)
    delta_quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw_noise)
    quats = math_utils.quat_mul(delta_quat, base_quat)
    return positions, quats


def reset_robot_counter650_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose: tuple[float, ...],
    pose_noise: dict[str, tuple[float, float]],
    velocity_noise: dict[str, tuple[float, float]],
    root_velocity: tuple[float, ...] | None = None,
    joint_pos_overrides: dict[str, float] | None = None,
    joint_pos_noise: float = 0.03,
    joint_vel_overrides: dict[str, float] | None = None,
    joint_vel_noise: float = 0.0,
    last_action: tuple[float, ...] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset G1 near the counter-650 crossing state family."""
    asset: Articulation = env.scene[asset_cfg.name]
    positions, quats = _sample_pose(env, env_ids, pose, pose_noise)
    ranges = [velocity_noise.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")]
    limits = torch.tensor(ranges, device=asset.device, dtype=torch.float32)
    velocity_noise_samples = math_utils.sample_uniform(
        limits[:, 0],
        limits[:, 1],
        (len(env_ids), 6),
        device=asset.device,
    )
    if root_velocity is None:
        velocity_base = torch.zeros((1, 6), device=asset.device, dtype=torch.float32)
    else:
        velocity_base = torch.tensor(root_velocity, device=asset.device, dtype=torch.float32).reshape(1, -1)
        if velocity_base.shape[-1] != 6:
            raise ValueError(f"Expected root_velocity length 6, got {velocity_base.shape[-1]}")
    velocities = velocity_base + velocity_noise_samples
    asset.write_root_pose_to_sim(torch.cat([positions, quats], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    if joint_pos_overrides:
        for joint_expr, value in joint_pos_overrides.items():
            joint_ids, _ = asset.find_joints(joint_expr)
            if len(joint_ids) > 0:
                joint_pos[:, joint_ids] = float(value)
    if joint_pos_noise > 0.0:
        noise = math_utils.sample_uniform(
            -joint_pos_noise,
            joint_pos_noise,
            joint_pos.shape,
            device=asset.device,
        )
        joint_pos += noise
    joint_limits = asset.data.joint_pos_limits[env_ids]
    joint_pos = torch.clamp(joint_pos, joint_limits[..., 0], joint_limits[..., 1])
    joint_vel = torch.zeros_like(joint_pos)
    if joint_vel_overrides:
        for joint_expr, value in joint_vel_overrides.items():
            joint_ids, _ = asset.find_joints(joint_expr)
            if len(joint_ids) > 0:
                joint_vel[:, joint_ids] = float(value)
    if joint_vel_noise > 0.0:
        joint_vel += math_utils.sample_uniform(
            -joint_vel_noise,
            joint_vel_noise,
            joint_vel.shape,
            device=asset.device,
        )
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    if last_action is not None:
        action = torch.tensor(last_action, device=asset.device, dtype=torch.float32).reshape(1, -1)
        if action.shape[-1] < 1:
            raise ValueError("Expected last_action to contain at least one action value")
        stored_action = getattr(env, "_cross_pit_reset_last_action", None)
        expected_shape = (env.num_envs, action.shape[-1])
        if (
            stored_action is None
            or stored_action.shape != expected_shape
            or stored_action.device != asset.device
            or stored_action.dtype != torch.float32
        ):
            stored_action = torch.zeros(expected_shape, device=asset.device, dtype=torch.float32)
        stored_action[env_ids] = action.expand(len(env_ids), -1)
        env._cross_pit_reset_last_action = stored_action


def reset_fixed_box_support_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose: tuple[float, ...],
    pose_noise: dict[str, tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("box_support"),
) -> None:
    """Reset the already-pushed box as a fixed support in the pit."""
    asset: RigidObject = env.scene[asset_cfg.name]
    positions, quats = _sample_pose(env, env_ids, pose, pose_noise or {})
    velocities = torch.zeros((len(env_ids), 6), device=asset.device)
    asset.write_root_pose_to_sim(torch.cat([positions, quats], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
