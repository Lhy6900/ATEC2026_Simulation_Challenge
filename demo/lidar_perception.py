from __future__ import annotations

from typing import Any

import torch

try:
    from demo.lidar_perception_prior import LidarPerceptionPrior
except ImportError:  # pragma: no cover - top-level demo import fallback
    from lidar_perception_prior import LidarPerceptionPrior

try:
    from demo.lidar_perception_torch import (
        TorchLidarPerceptionResult,
        TorchLidarPoseStabilizer,
        estimate_task_d_poses_from_lidar_torch,
    )
except ImportError:  # pragma: no cover - package import fallback
    from lidar_perception_torch import (
        TorchLidarPerceptionResult,
        TorchLidarPoseStabilizer,
        estimate_task_d_poses_from_lidar_torch,
    )


class TaskDLidarGpuPerception:
    """GPU-only Task D LiDAR perception adapter for policy/RL code."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        prior: LidarPerceptionPrior | None = None,
        init_samples: int = 5,
    ) -> None:
        self.device = torch.device(device)
        self.prior = prior or LidarPerceptionPrior()
        self.stabilizer = TorchLidarPoseStabilizer(
            num_envs=int(num_envs),
            device=self.device,
            init_samples=int(init_samples),
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.stabilizer.reset(env_ids)

    def update_from_obs(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        lidar = self._get_lidar_obs(obs)
        return self.update_from_lidar_data(lidar)

    def update_from_env(self, env: Any) -> dict[str, torch.Tensor]:
        return self.update_from_lidar_data(self._get_lidar_env_data(env))

    def estimate_from_lidar_data(self, lidar: dict[str, Any]) -> dict[str, torch.Tensor]:
        result = estimate_task_d_poses_from_lidar_torch(
            lidar["ray_hits_w"],
            lidar["pos_w"],
            lidar["quat_w"],
            prior=self.prior,
            max_distance=lidar.get("max_distance"),
        )
        return {
            "box_pose": result.box_pose,
            "box_valid": result.box_valid,
            "box_confidence": result.box_confidence,
            "ditch_pose": result.ditch_pose,
            "ditch_valid": result.ditch_valid,
            "ditch_confidence": result.ditch_confidence,
            "num_points": result.num_points,
            "num_ground_inliers": result.num_ground_inliers,
        }

    def update_from_lidar_data(self, lidar: dict[str, Any]) -> dict[str, torch.Tensor]:
        result_dict = self.estimate_from_lidar_data(lidar)
        result = TorchLidarPerceptionResult(**result_dict)
        stable = self.stabilizer.update(
            result,
            lidar_pos_w=lidar["pos_w"],
            lidar_quat_w=lidar["quat_w"],
        )
        return {
            "box_pose": stable.box_pose,
            "box_valid": stable.box_valid,
            "box_confidence": stable.box_confidence,
            "ditch_pose": stable.ditch_pose,
            "ditch_valid": stable.ditch_valid,
            "ditch_confidence": stable.ditch_confidence,
            "num_points": stable.num_points,
            "num_ground_inliers": stable.num_ground_inliers,
        }

    def _get_lidar_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        lidar = obs.get("lidar")
        if isinstance(lidar, dict):
            return lidar

        extero = obs.get("extero")
        if isinstance(extero, dict):
            lidar = extero.get("lidar")
            if isinstance(lidar, dict):
                return lidar

        sensors = obs.get("sensors")
        if isinstance(sensors, dict):
            lidar = sensors.get("lidar")
            if isinstance(lidar, dict):
                return lidar

        raise KeyError("LiDAR observation must be in obs['lidar'], obs['extero']['lidar'], or obs['sensors']['lidar']")

    def _get_lidar_env_data(self, env: Any) -> dict[str, Any]:
        scene = getattr(getattr(env, "unwrapped", env), "scene", None)
        if scene is None:
            raise KeyError("env.unwrapped.scene is required for LiDAR scene perception")
        sensors = getattr(scene, "sensors", {})
        lidar_sensor = sensors.get("lidar_sensor") if hasattr(sensors, "get") else None
        lidar_data = getattr(lidar_sensor, "data", None)
        ray_hits_w = getattr(lidar_data, "ray_hits_w", None)
        pos_w = getattr(lidar_data, "pos_w", None)
        quat_w = getattr(lidar_data, "quat_w", None)
        if ray_hits_w is None or pos_w is None or quat_w is None:
            raise KeyError("lidar_sensor.data must expose ray_hits_w, pos_w, and quat_w")
        return {
            "ray_hits_w": ray_hits_w,
            "pos_w": pos_w,
            "quat_w": quat_w,
            "max_distance": getattr(getattr(lidar_sensor, "cfg", None), "max_distance", None),
        }
