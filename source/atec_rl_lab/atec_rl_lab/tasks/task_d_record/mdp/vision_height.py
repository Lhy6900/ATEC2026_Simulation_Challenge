from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def vision_elevation_map(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, noise: bool = False) -> torch.Tensor:
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    relative_pos_w = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    sensor_quat = sensor.data.quat_w
    num_envs, num_rays, _ = relative_pos_w.shape

    if getattr(sensor.cfg, "ray_alignment", "base") == "yaw":
        sensor_quat = yaw_quat(sensor_quat)

    sensor_quat_expanded = sensor_quat.unsqueeze(1).expand(num_envs, num_rays, 4).reshape(num_envs * num_rays, 4)
    sensor_coords = quat_apply_inverse(sensor_quat_expanded.to(torch.float), relative_pos_w.reshape(num_envs * num_rays, 3))
    sensor_coords = sensor_coords.reshape(num_envs, num_rays, 3)

    if torch.isnan(sensor_coords).any() or torch.isinf(sensor_coords).any():
        print(f"Warning: vision_elevation_map contains NaN or Inf: {sensor_coords}")
        sensor_coords = torch.nan_to_num(sensor_coords)

    if noise:
        height_noise = torch.randn_like(sensor_coords[..., 2]) * 0.03
        sensor_coords[..., 2] += height_noise
        offset_noise = torch.rand(num_envs, 1, device=env.device) * 0.1 - 0.05
        sensor_coords[..., 2] += offset_noise

    sensor_coords[..., 2] = torch.clamp(sensor_coords[..., 2], min=-1.2, max=0.0)
    return sensor_coords.reshape(num_envs, num_rays * 3)
