from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

BOX_LIDAR_Y_BIAS_M = 0.055

try:
    from lidar_perception import LidarPerceptionPrior
except ImportError:  # pragma: no cover - package import fallback for tests
    from scripts.lidar_perception import LidarPerceptionPrior


@dataclass
class TorchLidarPerceptionResult:
    box_pose: torch.Tensor
    box_valid: torch.Tensor
    box_confidence: torch.Tensor
    ditch_pose: torch.Tensor
    ditch_valid: torch.Tensor
    ditch_confidence: torch.Tensor
    num_points: torch.Tensor
    num_ground_inliers: torch.Tensor


class TorchLidarPoseStabilizer:
    """Batched CUDA-friendly temporal state for Task D LiDAR poses.

    This class intentionally keeps state as tensors so multi-env RL loops can
    update perception without copying estimates back to the CPU.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        init_samples: int = 5,
        max_step_xy: float = 0.35,
        max_yaw_step: float = math.radians(12.0),
    ):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.init_samples = max(1, int(init_samples))
        self.box_init_history_size = max(8, 4 * self.init_samples)
        self.max_step_xy = float(max_step_xy)
        self.max_yaw_step = float(max_yaw_step)
        self.box_pose = torch.full((self.num_envs, 3), float("nan"), dtype=torch.float32, device=self.device)
        self.box_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.box_confidence = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.box_init_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.box_yaw_anchor = torch.full((self.num_envs,), float("nan"), dtype=torch.float32, device=self.device)
        self.box_init_pose_samples = torch.full(
            (self.num_envs, self.box_init_history_size, 3),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        self.box_init_conf_samples = torch.zeros(
            (self.num_envs, self.box_init_history_size),
            dtype=torch.float32,
            device=self.device,
        )
        self.box_init_sample_valid = torch.zeros(
            (self.num_envs, self.box_init_history_size),
            dtype=torch.bool,
            device=self.device,
        )
        self.box_init_write_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.ditch_pose = torch.full((self.num_envs, 3), float("nan"), dtype=torch.float32, device=self.device)
        self.ditch_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ditch_confidence = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.ditch_init_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)
        self.box_pose[env_ids] = float("nan")
        self.box_valid[env_ids] = False
        self.box_confidence[env_ids] = 0.0
        self.box_init_count[env_ids] = 0
        self.box_yaw_anchor[env_ids] = float("nan")
        self.box_init_pose_samples[env_ids] = float("nan")
        self.box_init_conf_samples[env_ids] = 0.0
        self.box_init_sample_valid[env_ids] = False
        self.box_init_write_index[env_ids] = 0
        self.ditch_pose[env_ids] = float("nan")
        self.ditch_valid[env_ids] = False
        self.ditch_confidence[env_ids] = 0.0
        self.ditch_init_count[env_ids] = 0

    def update(self, result: TorchLidarPerceptionResult) -> TorchLidarPerceptionResult:
        measured_box_pose = result.box_pose.to(device=self.device, dtype=torch.float32)
        measured_box_confidence = result.box_confidence.to(device=self.device, dtype=torch.float32)
        measured_box_valid = (
            result.box_valid.to(device=self.device)
            & ~_is_startup_box_edge_outlier(
                measured_box_pose,
                measured_box_confidence,
                self.box_valid,
                self.box_init_count,
                self.init_samples,
            )
        )
        (
            box_pose,
            box_valid,
            box_confidence,
            self.box_init_count,
            self.box_yaw_anchor,
        ) = _stabilize_box_pose_batch(
            measured_pose=measured_box_pose,
            measured_valid=measured_box_valid,
            measured_confidence=measured_box_confidence,
            state_pose=self.box_pose,
            state_valid=self.box_valid,
            state_confidence=self.box_confidence,
            init_count=self.box_init_count,
            init_samples=self.init_samples,
            max_step_xy=self.max_step_xy,
            max_yaw_step=self.max_yaw_step,
            yaw_anchor=self.box_yaw_anchor,
            init_pose_samples=self.box_init_pose_samples,
            init_conf_samples=self.box_init_conf_samples,
            init_sample_valid=self.box_init_sample_valid,
            init_write_index=self.box_init_write_index,
            init_yaw_gate=math.radians(35.0),
            init_xy_gate=0.22,
        )
        self.box_pose = box_pose
        self.box_valid = box_valid
        self.box_confidence = box_confidence

        ditch_pose, ditch_valid, ditch_confidence, self.ditch_init_count = _stabilize_pose_batch(
            measured_pose=result.ditch_pose.to(device=self.device, dtype=torch.float32),
            measured_valid=result.ditch_valid.to(device=self.device),
            measured_confidence=result.ditch_confidence.to(device=self.device, dtype=torch.float32),
            state_pose=self.ditch_pose,
            state_valid=self.ditch_valid,
            state_confidence=self.ditch_confidence,
            init_count=self.ditch_init_count,
            init_samples=self.init_samples,
            max_step_xy=0.45,
            max_yaw_step=math.radians(20.0),
        )
        self.ditch_pose = ditch_pose
        self.ditch_valid = ditch_valid
        self.ditch_confidence = ditch_confidence

        return TorchLidarPerceptionResult(
            box_pose=_apply_box_output_bias(self.box_pose),
            box_valid=self.box_valid & (self.box_init_count >= self.init_samples),
            box_confidence=self.box_confidence,
            ditch_pose=self.ditch_pose,
            ditch_valid=self.ditch_valid & (self.ditch_init_count >= self.init_samples),
            ditch_confidence=self.ditch_confidence,
            num_points=result.num_points.to(device=self.device),
            num_ground_inliers=result.num_ground_inliers.to(device=self.device),
        )


def _is_startup_box_edge_outlier(
    measured_pose: torch.Tensor,
    measured_confidence: torch.Tensor,
    state_valid: torch.Tensor,
    init_count: torch.Tensor,
    init_samples: int,
) -> torch.Tensor:
    startup = init_count < int(init_samples)
    finite = torch.isfinite(measured_pose).all(dim=1)
    x = measured_pose[:, 0]
    y = measured_pose[:, 1]
    yaw = wrap_axis_yaw_torch(measured_pose[:, 2])
    side_edge = (x.abs() < 0.18) & (y.abs() > 0.90) & (torch.abs(torch.abs(yaw) - math.pi / 2.0) < math.radians(18.0))
    low_conf_axis_flip = (measured_confidence < 0.75) & (torch.abs(torch.abs(yaw) - math.pi / 2.0) < math.radians(35.0))
    return startup & finite & (side_edge | low_conf_axis_flip)


def _stabilize_pose_batch(
    measured_pose: torch.Tensor,
    measured_valid: torch.Tensor,
    measured_confidence: torch.Tensor,
    state_pose: torch.Tensor,
    state_valid: torch.Tensor,
    state_confidence: torch.Tensor,
    init_count: torch.Tensor,
    init_samples: int,
    max_step_xy: float,
    max_yaw_step: float,
    yaw_anchor: torch.Tensor | None = None,
    init_yaw_gate: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    measured_valid = measured_valid & torch.isfinite(measured_pose).all(dim=1)
    if yaw_anchor is not None and init_yaw_gate is not None:
        has_anchor = torch.isfinite(yaw_anchor)
        startup = init_count < int(init_samples)
        yaw_delta_anchor = torch.abs(wrap_axis_yaw_torch(measured_pose[:, 2] - yaw_anchor))
        measured_valid = measured_valid & ~(startup & has_anchor & (yaw_delta_anchor > float(init_yaw_gate)))
    initialized = state_valid & (init_count >= int(init_samples))
    uninitialized_accept = measured_valid & ~initialized
    new_pose = state_pose.clone()
    new_valid = state_valid.clone()
    new_confidence = state_confidence.clone()
    new_init_count = init_count.clone()

    new_pose = torch.where(uninitialized_accept[:, None], measured_pose, new_pose)
    new_valid = new_valid | uninitialized_accept
    new_confidence = torch.where(uninitialized_accept, measured_confidence, new_confidence)
    new_init_count = torch.where(
        uninitialized_accept,
        torch.minimum(new_init_count + 1, torch.full_like(new_init_count, int(init_samples))),
        new_init_count,
    )

    can_update = measured_valid & initialized
    target_yaw = align_axis_yaw_to_reference_torch(measured_pose[:, 2], state_pose[:, 2])
    delta_xy = measured_pose[:, :2] - state_pose[:, :2]
    dist = torch.linalg.norm(delta_xy, dim=1).clamp_min(1.0e-6)
    step_scale = (float(max_step_xy) / dist).clamp(max=1.0)
    limited_xy = state_pose[:, :2] + delta_xy * step_scale[:, None]
    yaw_delta = (target_yaw - state_pose[:, 2]).clamp(-float(max_yaw_step), float(max_yaw_step))
    limited_yaw = state_pose[:, 2] + yaw_delta
    limited_pose = torch.cat([limited_xy, limited_yaw[:, None]], dim=1)
    new_pose = torch.where(can_update[:, None], limited_pose, new_pose)
    new_valid = new_valid | can_update
    new_confidence = torch.where(can_update, torch.maximum(state_confidence * 0.85, measured_confidence), new_confidence)

    return new_pose, new_valid, new_confidence, new_init_count


def _stabilize_box_pose_batch(
    measured_pose: torch.Tensor,
    measured_valid: torch.Tensor,
    measured_confidence: torch.Tensor,
    state_pose: torch.Tensor,
    state_valid: torch.Tensor,
    state_confidence: torch.Tensor,
    init_count: torch.Tensor,
    init_samples: int,
    max_step_xy: float,
    max_yaw_step: float,
    yaw_anchor: torch.Tensor,
    init_pose_samples: torch.Tensor,
    init_conf_samples: torch.Tensor,
    init_sample_valid: torch.Tensor,
    init_write_index: torch.Tensor,
    init_yaw_gate: float,
    init_xy_gate: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    measured_valid = measured_valid & torch.isfinite(measured_pose).all(dim=1)
    initialized = state_valid & (init_count >= int(init_samples))
    startup = ~initialized

    anchor_for_measurement = _select_box_startup_anchor(
        samples=init_pose_samples,
        valid=init_sample_valid,
        confidence=init_conf_samples,
        fallback=yaw_anchor,
    )
    has_anchor = torch.isfinite(anchor_for_measurement)
    yaw_delta_anchor = torch.abs(wrap_axis_yaw_torch(measured_pose[:, 2] - anchor_for_measurement))
    measured_valid = measured_valid & ~(startup & has_anchor & (yaw_delta_anchor > float(init_yaw_gate)))

    duplicate_startup = _is_duplicate_startup_box_sample(
        measured_pose=measured_pose,
        samples=init_pose_samples,
        valid=init_sample_valid,
        yaw_epsilon=math.radians(0.1),
        xy_epsilon=0.002,
    )
    startup_accept = measured_valid & startup & ~duplicate_startup
    if init_pose_samples.shape[1] > 0:
        batch_idx = torch.arange(measured_pose.shape[0], device=measured_pose.device)
        write_slots = torch.remainder(init_write_index, init_pose_samples.shape[1])
        init_pose_samples[batch_idx, write_slots] = torch.where(
            startup_accept[:, None],
            measured_pose,
            init_pose_samples[batch_idx, write_slots],
        )
        init_conf_samples[batch_idx, write_slots] = torch.where(
            startup_accept,
            measured_confidence,
            init_conf_samples[batch_idx, write_slots],
        )
        init_sample_valid[batch_idx, write_slots] = init_sample_valid[batch_idx, write_slots] | startup_accept
        init_write_index[:] = torch.where(
            startup_accept,
            init_write_index + 1,
            init_write_index,
        )

    cluster_pose, cluster_confidence, cluster_count, cluster_yaw = _cluster_startup_box_samples(
        samples=init_pose_samples,
        confidence=init_conf_samples,
        valid=init_sample_valid,
        init_yaw_gate=float(init_yaw_gate),
        init_xy_gate=float(init_xy_gate),
    )
    cluster_ready = startup & (cluster_count >= int(init_samples)) & torch.isfinite(cluster_pose).all(dim=1)
    fast_ready = startup & _is_fast_reliable_box_init_sample(
        measured_pose=measured_pose,
        measured_valid=measured_valid,
        measured_confidence=measured_confidence,
    )
    cluster_pose = torch.where(fast_ready[:, None], measured_pose, cluster_pose)
    cluster_confidence = torch.where(fast_ready, measured_confidence, cluster_confidence)
    cluster_yaw = torch.where(fast_ready, wrap_axis_yaw_torch(measured_pose[:, 2]), cluster_yaw)
    cluster_ready = cluster_ready | fast_ready

    new_pose = state_pose.clone()
    new_valid = state_valid.clone()
    new_confidence = state_confidence.clone()
    new_init_count = init_count.clone()
    new_yaw_anchor = yaw_anchor.clone()

    new_pose = torch.where(cluster_ready[:, None], cluster_pose, new_pose)
    new_valid = new_valid | cluster_ready
    new_confidence = torch.where(cluster_ready, cluster_confidence, new_confidence)
    new_init_count = torch.where(
        cluster_ready,
        torch.full_like(new_init_count, int(init_samples)),
        new_init_count,
    )
    new_yaw_anchor = torch.where(cluster_ready, cluster_yaw, new_yaw_anchor)

    initialized_after = new_valid & (new_init_count >= int(init_samples))
    update_yaw_reference = torch.where(torch.isfinite(new_yaw_anchor), new_yaw_anchor, new_pose[:, 2])
    update_yaw_delta = torch.abs(wrap_axis_yaw_torch(measured_pose[:, 2] - update_yaw_reference))
    has_update_yaw_reference = torch.isfinite(update_yaw_reference)
    update_yaw_gate = min(float(init_yaw_gate), math.radians(20.0))
    can_update = (
        measured_valid
        & initialized_after
        & ~cluster_ready
        & ~(has_update_yaw_reference & (update_yaw_delta > update_yaw_gate))
    )
    target_yaw = align_axis_yaw_to_reference_torch(measured_pose[:, 2], new_pose[:, 2])
    delta_xy = measured_pose[:, :2] - new_pose[:, :2]
    dist = torch.linalg.norm(delta_xy, dim=1).clamp_min(1.0e-6)
    step_scale = (float(max_step_xy) / dist).clamp(max=1.0)
    limited_xy = new_pose[:, :2] + delta_xy * step_scale[:, None]
    yaw_delta = (target_yaw - new_pose[:, 2]).clamp(-float(max_yaw_step), float(max_yaw_step))
    limited_yaw = new_pose[:, 2] + yaw_delta
    limited_pose = torch.cat([limited_xy, limited_yaw[:, None]], dim=1)
    new_pose = torch.where(can_update[:, None], limited_pose, new_pose)
    new_valid = new_valid | can_update
    new_confidence = torch.where(can_update, torch.maximum(new_confidence * 0.85, measured_confidence), new_confidence)

    return new_pose, new_valid, new_confidence, new_init_count, new_yaw_anchor


def _select_box_startup_anchor(
    samples: torch.Tensor,
    valid: torch.Tensor,
    confidence: torch.Tensor,
    fallback: torch.Tensor,
) -> torch.Tensor:
    finite = valid & torch.isfinite(samples).all(dim=2)
    yaw = wrap_axis_yaw_torch(samples[..., 2])
    axis_flip_like = torch.abs(torch.abs(yaw) - math.pi / 2.0) < math.radians(35.0)
    reliable = finite & (confidence >= 0.75) & ~axis_flip_like
    score = torch.where(reliable, confidence, torch.full_like(confidence, -float("inf")))
    best_idx = torch.argmax(score, dim=1)
    best_score = score.gather(1, best_idx[:, None]).squeeze(1)
    best_yaw = yaw.gather(1, best_idx[:, None]).squeeze(1)
    return torch.where(torch.isfinite(best_score), best_yaw, fallback)


def _is_fast_reliable_box_init_sample(
    measured_pose: torch.Tensor,
    measured_valid: torch.Tensor,
    measured_confidence: torch.Tensor,
) -> torch.Tensor:
    finite = measured_valid & torch.isfinite(measured_pose).all(dim=1)
    yaw = wrap_axis_yaw_torch(measured_pose[:, 2])
    axis_flip_like = torch.abs(torch.abs(yaw) - math.pi / 2.0) < math.radians(35.0)
    near_box = torch.linalg.norm(measured_pose[:, :2], dim=1) <= 1.25
    return finite & near_box & (measured_confidence >= 0.80) & ~axis_flip_like


def _is_duplicate_startup_box_sample(
    measured_pose: torch.Tensor,
    samples: torch.Tensor,
    valid: torch.Tensor,
    yaw_epsilon: float,
    xy_epsilon: float,
) -> torch.Tensor:
    finite_samples = valid & torch.isfinite(samples).all(dim=2)
    finite_measured = torch.isfinite(measured_pose).all(dim=1)
    safe_samples = torch.where(finite_samples[..., None], samples, torch.zeros_like(samples))
    yaw_delta = torch.abs(wrap_axis_yaw_torch(safe_samples[..., 2] - measured_pose[:, None, 2]))
    xy_delta = torch.linalg.norm(safe_samples[..., :2] - measured_pose[:, None, :2], dim=2)
    duplicate = finite_samples & (yaw_delta <= float(yaw_epsilon)) & (xy_delta <= float(xy_epsilon))
    return finite_measured & duplicate.any(dim=1)


def _cluster_startup_box_samples(
    samples: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
    init_yaw_gate: float,
    init_xy_gate: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, history, _ = samples.shape
    device = samples.device
    finite = valid & torch.isfinite(samples).all(dim=2)
    safe_samples = torch.where(finite[..., None], samples, torch.zeros_like(samples))
    yaw = wrap_axis_yaw_torch(safe_samples[..., 2])
    yaw_delta = torch.abs(wrap_axis_yaw_torch(yaw[:, :, None] - yaw[:, None, :]))
    xy_delta = torch.linalg.norm(safe_samples[:, :, None, :2] - safe_samples[:, None, :, :2], dim=3)
    same_cluster = (
        finite[:, :, None]
        & finite[:, None, :]
        & (yaw_delta <= float(init_yaw_gate))
        & (xy_delta <= float(init_xy_gate))
    )
    cluster_weight = torch.where(same_cluster, confidence[:, None, :].clamp_min(0.05), torch.zeros_like(yaw_delta))
    cluster_count = same_cluster.sum(dim=2)
    cluster_score = cluster_weight.sum(dim=2)
    best_idx = torch.argmax(cluster_score, dim=1)
    best_members = same_cluster[
        torch.arange(batch, device=device),
        best_idx,
    ]
    best_count = cluster_count.gather(1, best_idx[:, None]).squeeze(1)
    best_weight = torch.where(best_members, confidence.clamp_min(0.05), torch.zeros_like(confidence))
    weight_sum = best_weight.sum(dim=1).clamp_min(1.0e-6)
    mean_xy = (safe_samples[..., :2] * best_weight[..., None]).sum(dim=1) / weight_sum[:, None]
    sin2 = (torch.sin(2.0 * yaw) * best_weight).sum(dim=1)
    cos2 = (torch.cos(2.0 * yaw) * best_weight).sum(dim=1)
    mean_yaw = wrap_axis_yaw_torch(0.5 * torch.atan2(sin2, cos2))
    pose = torch.cat([mean_xy, mean_yaw[:, None]], dim=1)
    conf = (cluster_score.gather(1, best_idx[:, None]).squeeze(1) / max(1.0, float(history))).clamp(0.0, 1.0)
    pose = torch.where((best_count > 0)[:, None], pose, torch.full_like(pose, float("nan")))
    conf = torch.where(best_count > 0, conf, torch.zeros_like(conf))
    return pose, conf, best_count, mean_yaw


def _apply_box_output_bias(box_pose: torch.Tensor) -> torch.Tensor:
    bias = torch.tensor([0.0, BOX_LIDAR_Y_BIAS_M, 0.0], dtype=box_pose.dtype, device=box_pose.device)
    finite = torch.isfinite(box_pose).all(dim=1)
    return torch.where(finite[:, None], box_pose + bias, box_pose)


def _as_torch(value: Any, device: torch.device | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def wrap_axis_yaw_torch(yaw: torch.Tensor) -> torch.Tensor:
    return torch.remainder(yaw + math.pi / 2.0, math.pi) - math.pi / 2.0


def align_axis_yaw_to_reference_torch(yaw: torch.Tensor, reference_yaw: torch.Tensor) -> torch.Tensor:
    candidates = torch.stack([yaw, yaw + math.pi, yaw - math.pi], dim=0)
    deltas = torch.abs(candidates - reference_yaw.unsqueeze(0))
    index = torch.argmin(deltas, dim=0)
    return candidates.gather(0, index.unsqueeze(0)).squeeze(0)


def quat_inverse_apply_torch(quat_wxyz: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    quat = quat_wxyz.to(dtype=vectors.dtype)
    inv_xyz = -quat[..., 1:4]
    inv_w = quat[..., 0:1]
    while inv_xyz.ndim < vectors.ndim:
        inv_xyz = inv_xyz.unsqueeze(-2)
        inv_w = inv_w.unsqueeze(-2)
    t = torch.cross(inv_xyz.expand_as(vectors), vectors, dim=-1) * 2.0
    return vectors + inv_w * t + torch.cross(inv_xyz.expand_as(vectors), t, dim=-1)


def world_points_to_lidar_frame_torch(
    points_w: Any,
    lidar_pos_w: Any,
    lidar_quat_w: Any,
) -> torch.Tensor:
    points = _as_torch(points_w)
    device = points.device
    points = points.to(dtype=torch.float32)
    pos = _as_torch(lidar_pos_w, device=device).to(dtype=torch.float32)
    quat = _as_torch(lidar_quat_w, device=device).to(dtype=torch.float32)
    if points.ndim == 2:
        points = points.unsqueeze(0)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    if quat.ndim == 1:
        quat = quat.unsqueeze(0)
    centered = points - pos[:, None, :3]
    return quat_inverse_apply_torch(quat[:, :4], centered)


def _pca_yaw_masked(points_xy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=points_xy.dtype)
    safe_points = torch.where(mask[..., None], points_xy, torch.zeros_like(points_xy))
    raw_count = weights.sum(dim=1)
    count = raw_count.clamp_min(1.0)
    mean = safe_points.sum(dim=1) / count[:, None]
    centered = torch.where(mask[..., None], points_xy - mean[:, None, :], torch.zeros_like(points_xy))
    denom = (count - 1.0).clamp_min(1.0)
    cov_xx = (centered[..., 0] * centered[..., 0]).sum(dim=1) / denom
    cov_yy = (centered[..., 1] * centered[..., 1]).sum(dim=1) / denom
    cov_xy = (centered[..., 0] * centered[..., 1]).sum(dim=1) / denom
    yaw = 0.5 * torch.atan2(2.0 * cov_xy, cov_xx - cov_yy)
    return torch.where(raw_count >= 2.0, wrap_axis_yaw_torch(yaw), torch.zeros_like(yaw))


def _oriented_pose_from_mask(
    points_xy: torch.Tensor,
    mask: torch.Tensor,
    yaw: torch.Tensor,
    expected_u: float | None = None,
    expected_v: float | None = None,
    infer_partial_center: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = points_xy.dtype
    inf = torch.tensor(float("inf"), dtype=dtype, device=points_xy.device)
    safe_points = torch.where(mask[..., None], points_xy, torch.zeros_like(points_xy))
    u = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
    v = torch.stack([-torch.sin(yaw), torch.cos(yaw)], dim=1)
    proj_u = (safe_points * u[:, None, :]).sum(dim=-1)
    proj_v = (safe_points * v[:, None, :]).sum(dim=-1)
    masked_u_min = torch.where(mask, proj_u, inf).amin(dim=1)
    masked_u_max = torch.where(mask, proj_u, -inf).amax(dim=1)
    masked_v_min = torch.where(mask, proj_v, inf).amin(dim=1)
    masked_v_max = torch.where(mask, proj_v, -inf).amax(dim=1)
    extent_u = masked_u_max - masked_u_min
    extent_v = masked_v_max - masked_v_min
    center_u = 0.5 * (masked_u_min + masked_u_max)
    center_v = 0.5 * (masked_v_min + masked_v_max)
    if infer_partial_center and expected_u is not None:
        center_u = _infer_axis_center_torch(proj_u, mask, float(expected_u), extent_u, center_u)
    if infer_partial_center and expected_v is not None:
        center_v = _infer_axis_center_torch(proj_v, mask, float(expected_v), extent_v, center_v)
    center = center_u[:, None] * u + center_v[:, None] * v
    return center, extent_u, extent_v


def _infer_axis_center_torch(
    projections: torch.Tensor,
    mask: torch.Tensor,
    expected_size: float,
    observed_extent: torch.Tensor,
    midpoint: torch.Tensor,
) -> torch.Tensor:
    dtype = projections.dtype
    inf = torch.tensor(float("inf"), dtype=dtype, device=projections.device)
    safe_projections = torch.where(mask, projections, torch.zeros_like(projections))
    min_p = torch.where(mask, safe_projections, inf).amin(dim=1)
    max_p = torch.where(mask, safe_projections, -inf).amax(dim=1)
    count = mask.to(dtype=dtype).sum(dim=1).clamp_min(1.0)
    mean_p = safe_projections.sum(dim=1) / count
    half = 0.5 * float(expected_size)
    inferred = torch.where(mean_p >= 0.0, min_p + half, max_p - half)
    return torch.where(observed_extent >= 0.90 * float(expected_size), midpoint, inferred)


def _dimension_score_torch(observed: torch.Tensor, expected: float) -> torch.Tensor:
    expected = max(float(expected), 1.0e-6)
    observed = observed.clamp_min(0.0)
    over = (observed - expected).clamp_min(0.0) / expected
    under = (expected - observed).clamp_min(0.0) / expected
    return 2.0 * over + 0.35 * under


def _choose_box_axis(
    points_xy: torch.Tensor,
    box_mask: torch.Tensor,
    base_yaw: torch.Tensor,
    length: float,
    width: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    yaw_a = base_yaw
    yaw_b = wrap_axis_yaw_torch(base_yaw + math.pi / 2.0)
    center_a, extent_u_a, extent_v_a = _oriented_pose_from_mask(
        points_xy,
        box_mask,
        yaw_a,
        expected_u=length,
        expected_v=width,
        infer_partial_center=True,
    )
    center_b, extent_u_b, extent_v_b = _oriented_pose_from_mask(
        points_xy,
        box_mask,
        yaw_b,
        expected_u=length,
        expected_v=width,
        infer_partial_center=True,
    )
    score_a = _dimension_score_torch(extent_u_a, length) + _dimension_score_torch(extent_v_a, width)
    score_b = _dimension_score_torch(extent_u_b, length) + _dimension_score_torch(extent_v_b, width)
    choose_b = score_b < score_a
    center = torch.where(choose_b[:, None], center_b, center_a)
    yaw = torch.where(choose_b, yaw_b, yaw_a)
    extent_u = torch.where(choose_b, extent_u_b, extent_u_a)
    extent_v = torch.where(choose_b, extent_v_b, extent_v_a)
    score = torch.where(choose_b, score_b, score_a)
    return center, yaw, extent_u, extent_v, score


def _gather_nearest_masked_points(
    points_xy: torch.Tensor,
    mask: torch.Tensor,
    max_points: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_points = torch.where(mask[..., None], points_xy, torch.zeros_like(points_xy))
    range_sq = (safe_points * safe_points).sum(dim=-1)
    sort_score = torch.where(mask, range_sq, torch.full_like(range_sq, float("inf")))
    selected_idx = torch.argsort(sort_score, dim=1)[:, : int(max_points)]
    selected_xy = safe_points.gather(1, selected_idx[..., None].expand(-1, -1, 2))
    selected_valid = mask.gather(1, selected_idx)
    return selected_xy, selected_valid


def _infer_axis_center_grid(
    projections: torch.Tensor,
    valid: torch.Tensor,
    expected_size: float,
    observed_extent: torch.Tensor,
    midpoint: torch.Tensor,
) -> torch.Tensor:
    dtype = projections.dtype
    inf = torch.tensor(float("inf"), dtype=dtype, device=projections.device)
    safe = torch.where(valid[:, None, :], projections, torch.zeros_like(projections))
    min_p = torch.where(valid[:, None, :], safe, inf).amin(dim=2)
    max_p = torch.where(valid[:, None, :], safe, -inf).amax(dim=2)
    count = valid.to(dtype=dtype).sum(dim=1).clamp_min(1.0)
    mean_p = safe.sum(dim=2) / count[:, None]
    inferred = torch.where(mean_p >= 0.0, min_p + 0.5 * float(expected_size), max_p - 0.5 * float(expected_size))
    return torch.where(observed_extent >= 0.90 * float(expected_size), midpoint, inferred)


def _choose_box_axis_grid(
    points_xy: torch.Tensor,
    box_mask: torch.Tensor,
    length: float,
    width: float,
    max_points: int = 512,
    yaw_bins: int = 37,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected_xy, selected_valid = _gather_nearest_masked_points(points_xy, box_mask, max_points=max_points)
    dtype = selected_xy.dtype
    device = selected_xy.device
    batch = selected_xy.shape[0]
    inf = torch.tensor(float("inf"), dtype=dtype, device=device)

    yaw_grid = torch.linspace(-math.pi / 2.0, math.pi / 2.0, int(yaw_bins) + 1, dtype=dtype, device=device)[:-1]
    u = torch.stack([torch.cos(yaw_grid), torch.sin(yaw_grid)], dim=1)
    v = torch.stack([-torch.sin(yaw_grid), torch.cos(yaw_grid)], dim=1)
    proj_u = torch.einsum("bnd,kd->bkn", selected_xy, u)
    proj_v = torch.einsum("bnd,kd->bkn", selected_xy, v)
    valid = selected_valid

    min_u = torch.where(valid[:, None, :], proj_u, inf).amin(dim=2)
    max_u = torch.where(valid[:, None, :], proj_u, -inf).amax(dim=2)
    min_v = torch.where(valid[:, None, :], proj_v, inf).amin(dim=2)
    max_v = torch.where(valid[:, None, :], proj_v, -inf).amax(dim=2)
    extent_u = max_u - min_u
    extent_v = max_v - min_v
    center_u_mid = 0.5 * (min_u + max_u)
    center_v_mid = 0.5 * (min_v + max_v)
    center_u = _infer_axis_center_grid(proj_u, valid, float(length), extent_u, center_u_mid)
    center_v = _infer_axis_center_grid(proj_v, valid, float(width), extent_v, center_v_mid)
    center = center_u[..., None] * u[None, :, :] + center_v[..., None] * v[None, :, :]

    enough = valid.sum(dim=1)[:, None] >= 12
    full_valid = (
        enough
        & (extent_u <= float(length) + 0.55)
        & (extent_v <= float(width) + 0.45)
        & (torch.maximum(extent_u, extent_v) >= 0.18)
        & (torch.minimum(extent_u, extent_v) >= 0.08)
    )
    full_score = _dimension_score_torch(extent_u, float(length)) + _dimension_score_torch(extent_v, float(width))
    full_score = torch.where(
        full_valid,
        full_score + 0.03 * torch.linalg.norm(center, dim=2),
        torch.full_like(full_score, float("inf")),
    )

    edge_center = center_u_mid[..., None] * u[None, :, :] + center_v_mid[..., None] * v[None, :, :]
    edge_norm = torch.linalg.norm(edge_center, dim=2).clamp_min(1.0e-6)
    edge_dir = edge_center / edge_norm[..., None]

    thin_u = (extent_u < 0.08) & (extent_v >= 0.08) & enough
    edge_center_v = _infer_axis_center_grid(proj_v, valid, float(width), extent_v, center_v_mid)
    cand_u_pos = (center_u_mid + 0.5 * float(length))[..., None] * u[None, :, :] + edge_center_v[..., None] * v[None, :, :]
    cand_u_neg = (center_u_mid - 0.5 * float(length))[..., None] * u[None, :, :] + edge_center_v[..., None] * v[None, :, :]
    use_u_pos = (cand_u_pos * edge_dir).sum(dim=2) >= (cand_u_neg * edge_dir).sum(dim=2)
    edge_u_center = torch.where(use_u_pos[..., None], cand_u_pos, cand_u_neg)
    edge_u_score = _dimension_score_torch(extent_v, float(width)) + 0.20 + 0.03 * torch.linalg.norm(edge_u_center, dim=2)
    edge_u_score = torch.where(thin_u, edge_u_score, torch.full_like(edge_u_score, float("inf")))

    thin_v = (extent_v < 0.08) & (extent_u >= 0.08) & enough
    edge_center_u = _infer_axis_center_grid(proj_u, valid, float(length), extent_u, center_u_mid)
    cand_v_pos = edge_center_u[..., None] * u[None, :, :] + (center_v_mid + 0.5 * float(width))[..., None] * v[None, :, :]
    cand_v_neg = edge_center_u[..., None] * u[None, :, :] + (center_v_mid - 0.5 * float(width))[..., None] * v[None, :, :]
    use_v_pos = (cand_v_pos * edge_dir).sum(dim=2) >= (cand_v_neg * edge_dir).sum(dim=2)
    edge_v_center = torch.where(use_v_pos[..., None], cand_v_pos, cand_v_neg)
    edge_v_score = _dimension_score_torch(extent_u, float(length)) + 0.20 + 0.03 * torch.linalg.norm(edge_v_center, dim=2)
    edge_v_score = torch.where(thin_v, edge_v_score, torch.full_like(edge_v_score, float("inf")))

    all_scores = torch.cat([full_score, edge_u_score, edge_v_score], dim=1)
    all_centers = torch.cat([center, edge_u_center, edge_v_center], dim=1)
    all_yaws = yaw_grid.repeat(3).reshape(1, -1).repeat(batch, 1)
    all_extent_u = extent_u.repeat(1, 3)
    all_extent_v = extent_v.repeat(1, 3)
    best_idx = torch.argmin(all_scores, dim=1)
    best_score = all_scores.gather(1, best_idx[:, None]).squeeze(1)
    best_center = all_centers.gather(1, best_idx[:, None, None].expand(-1, 1, 2)).squeeze(1)
    best_yaw = all_yaws.gather(1, best_idx[:, None]).squeeze(1)
    best_extent_u = all_extent_u.gather(1, best_idx[:, None]).squeeze(1)
    best_extent_v = all_extent_v.gather(1, best_idx[:, None]).squeeze(1)
    return best_center, wrap_axis_yaw_torch(best_yaw), best_extent_u, best_extent_v, best_score


def _select_box_window(
    points_xy: torch.Tensor,
    box_mask: torch.Tensor,
    length: float,
    width: float,
    max_points: int = 512,
    yaw_bins: int = 37,
) -> torch.Tensor:
    selected_xy, selected_valid = _gather_nearest_masked_points(points_xy, box_mask, max_points=max_points)
    dtype = selected_xy.dtype
    device = selected_xy.device
    batch = selected_xy.shape[0]
    inf = torch.tensor(float("inf"), dtype=dtype, device=device)

    yaw_grid = torch.linspace(-math.pi / 2.0, math.pi / 2.0, int(yaw_bins) + 1, dtype=dtype, device=device)[:-1]
    u = torch.stack([torch.cos(yaw_grid), torch.sin(yaw_grid)], dim=1)
    v = torch.stack([-torch.sin(yaw_grid), torch.cos(yaw_grid)], dim=1)
    proj_u = torch.einsum("bnd,kd->bkn", selected_xy, u)
    proj_v = torch.einsum("bnd,kd->bkn", selected_xy, v)

    du = torch.abs(proj_u[:, :, :, None] - proj_u[:, :, None, :])
    dv = torch.abs(proj_v[:, :, :, None] - proj_v[:, :, None, :])
    inside = (
        (du <= 0.5 * float(length) + 0.08)
        & (dv <= 0.5 * float(width) + 0.08)
        & selected_valid[:, None, None, :]
        & selected_valid[:, None, :, None]
    )
    count = inside.sum(dim=3).to(dtype=dtype)
    min_u = torch.where(inside, proj_u[:, :, None, :], inf).amin(dim=3)
    max_u = torch.where(inside, proj_u[:, :, None, :], -inf).amax(dim=3)
    min_v = torch.where(inside, proj_v[:, :, None, :], inf).amin(dim=3)
    max_v = torch.where(inside, proj_v[:, :, None, :], -inf).amax(dim=3)
    extent_u = max_u - min_u
    extent_v = max_v - min_v
    candidate_xy = selected_xy[:, None, :, :]
    shape_score = _dimension_score_torch(extent_u, float(length)) + _dimension_score_torch(extent_v, float(width))
    valid_window = (
        (count >= 12.0)
        & (extent_u <= float(length) + 0.25)
        & (extent_v <= float(width) + 0.25)
        & (torch.maximum(extent_u, extent_v) >= 0.18)
    )
    score = torch.where(
        valid_window,
        shape_score + 0.03 * torch.linalg.norm(candidate_xy, dim=3) - 0.008 * count,
        torch.full_like(count, float("inf")),
    )
    flat_score = score.reshape(batch, -1)
    best_flat = torch.argmin(flat_score, dim=1)
    best_yaw_idx = best_flat // selected_xy.shape[1]
    best_point_idx = best_flat % selected_xy.shape[1]
    best_inside = inside[
        torch.arange(batch, device=device),
        best_yaw_idx,
        best_point_idx,
    ]
    best_score = flat_score.gather(1, best_flat[:, None]).squeeze(1)

    selected_idx_score = torch.where(box_mask, (points_xy * points_xy).sum(dim=-1), torch.full_like(box_mask.to(dtype), float("inf")))
    selected_idx = torch.argsort(selected_idx_score, dim=1)[:, : int(max_points)]
    window_mask = torch.zeros((batch, points_xy.shape[1]), dtype=torch.bool, device=device)
    window_mask.scatter_(1, selected_idx, best_inside & torch.isfinite(best_score)[:, None])
    return torch.where((best_inside.sum(dim=1) >= 12)[:, None] & torch.isfinite(best_score)[:, None], window_mask, box_mask)


def _select_box_component(
    points_xy: torch.Tensor,
    box_mask: torch.Tensor,
    length: float,
    width: float,
    max_points: int = 512,
) -> torch.Tensor:
    batch, num_points, _ = points_xy.shape
    dtype = points_xy.dtype
    safe_points = torch.where(box_mask[..., None], points_xy, torch.zeros_like(points_xy))
    range_sq = (safe_points * safe_points).sum(dim=-1)
    sort_score = torch.where(box_mask, range_sq, torch.full_like(range_sq, float("inf")))
    selected_idx = torch.argsort(sort_score, dim=1)[:, : int(max_points)]
    selected_xy = safe_points.gather(1, selected_idx[..., None].expand(-1, -1, 2))
    selected_valid = box_mask.gather(1, selected_idx)
    selected_range = range_sq.gather(1, selected_idx)

    diff = selected_xy[:, :, None, :] - selected_xy[:, None, :, :]
    near = (diff * diff).sum(dim=-1) <= 0.30 * 0.30
    adjacent = near & selected_valid[:, :, None] & selected_valid[:, None, :]

    labels = torch.arange(int(max_points), dtype=torch.long, device=points_xy.device).reshape(1, -1).repeat(batch, 1)
    labels = torch.where(selected_valid, labels, torch.full_like(labels, int(max_points)))
    for _ in range(8):
        neighbor_labels = torch.where(adjacent, labels[:, None, :], torch.full_like(labels[:, None, :], int(max_points)))
        labels = torch.minimum(labels, neighbor_labels.amin(dim=2))

    same_label = labels[:, :, None] == labels[:, None, :]
    component = same_label & selected_valid[:, :, None] & selected_valid[:, None, :]
    component_count = component.sum(dim=2).to(dtype=dtype)
    inf = torch.tensor(float("inf"), dtype=dtype, device=points_xy.device)
    min_xy = torch.where(component[..., None], selected_xy[:, None, :, :], inf).amin(dim=2)
    max_xy = torch.where(component[..., None], selected_xy[:, None, :, :], -inf).amax(dim=2)
    extent_xy = max_xy - min_xy
    center_xy = 0.5 * (min_xy + max_xy)
    score_a = _dimension_score_torch(extent_xy[..., 0], float(length)) + _dimension_score_torch(extent_xy[..., 1], float(width))
    score_b = _dimension_score_torch(extent_xy[..., 0], float(width)) + _dimension_score_torch(extent_xy[..., 1], float(length))
    shape_score = torch.minimum(score_a, score_b)
    valid_component = (
        selected_valid
        & (component_count >= 12.0)
        & (torch.maximum(extent_xy[..., 0], extent_xy[..., 1]) >= 0.18)
        & (torch.minimum(score_a, score_b) <= 1.20)
    )
    component_score = torch.where(
        valid_component,
        shape_score + 0.03 * torch.linalg.norm(center_xy, dim=2) - 0.001 * component_count,
        torch.full_like(component_count, float("inf")),
    )
    best_idx = torch.argmin(component_score, dim=1)
    best_label = labels.gather(1, best_idx[:, None]).squeeze(1)
    selected_component = selected_valid & (labels == best_label[:, None]) & torch.isfinite(
        component_score.gather(1, best_idx[:, None]).squeeze(1)
    )[:, None]
    component_mask = torch.zeros((batch, num_points), dtype=torch.bool, device=points_xy.device)
    component_mask.scatter_(1, selected_idx, selected_component)
    return torch.where(selected_component.sum(dim=1)[:, None] >= 12, component_mask, box_mask)


def _estimate_box_torch(
    points_l: torch.Tensor,
    finite: torch.Tensor,
    prior: LidarPerceptionPrior,
    ground_plane: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    box_prior = prior.box
    z_rel = points_l[..., 2] - _plane_z_torch(ground_plane, points_l[..., :2])
    box_mask = (
        finite
        & (z_rel > max(0.12, 0.18 * float(box_prior.height)))
        & (z_rel < float(box_prior.height) + 0.45)
    )
    box_mask = _select_box_window(points_l[..., :2], box_mask, float(box_prior.length), float(box_prior.width))
    box_mask = _select_box_component(points_l[..., :2], box_mask, float(box_prior.length), float(box_prior.width))
    counts = box_mask.sum(dim=1)
    center, yaw, extent_u, extent_v, score = _choose_box_axis_grid(
        points_l[..., :2],
        box_mask,
        float(box_prior.length),
        float(box_prior.width),
    )
    valid = (
        (counts >= 12)
        & torch.isfinite(score)
    )
    size_bonus = (counts.to(dtype=points_l.dtype) / 80.0).clamp(max=1.0)
    compactness = 1.0 / (1.0 + score)
    confidence = (0.65 * compactness + 0.35 * size_bonus).clamp(0.0, 1.0)
    pose = torch.cat([center, yaw[:, None]], dim=1)
    pose = torch.where(valid[:, None], pose, torch.full_like(pose, float("nan")))
    confidence = torch.where(valid, confidence, torch.zeros_like(confidence))
    return pose, valid, confidence


def _estimate_ditch_torch(
    points_l: torch.Tensor,
    finite: torch.Tensor,
    prior: LidarPerceptionPrior,
    ground_plane: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ditch_prior = prior.ditch
    z_rel = points_l[..., 2] - _plane_z_torch(ground_plane, points_l[..., :2])
    ditch_mask = finite & (z_rel < -max(0.18, 0.18 * float(ditch_prior.depth))) & (points_l[..., 0] > -0.75)
    counts = ditch_mask.sum(dim=1)
    base_yaw = _pca_yaw_masked(points_l[..., :2], ditch_mask)
    center, extent_u, extent_v = _oriented_pose_from_mask(points_l[..., :2], ditch_mask, base_yaw)
    swap = extent_v > extent_u
    yaw = torch.where(swap, wrap_axis_yaw_torch(base_yaw + math.pi / 2.0), base_yaw)
    long_extent = torch.maximum(extent_u, extent_v)
    short_extent = torch.minimum(extent_u, extent_v)
    width_score = torch.abs(short_extent - float(ditch_prior.width)) / max(float(ditch_prior.width), 1.0e-6)
    front_bonus = (center[:, 0] / 3.0).clamp(0.0, 1.0)
    density_bonus = (counts.to(dtype=points_l.dtype) / 180.0).clamp(max=1.0)
    confidence = (0.45 * (1.0 / (1.0 + width_score)) + 0.25 * density_bonus + 0.30 * front_bonus).clamp(0.0, 1.0)
    valid = (
        (counts >= 10)
        & (short_extent >= 0.25)
        & (short_extent <= float(ditch_prior.width) + 0.7)
        & (long_extent >= torch.maximum(torch.full_like(short_extent, 0.45), 1.25 * short_extent))
    )
    pose = torch.cat([center, yaw[:, None]], dim=1)
    pose = torch.where(valid[:, None], pose, torch.full_like(pose, float("nan")))
    confidence = torch.where(valid, confidence, torch.zeros_like(confidence))
    return pose, valid, confidence


def _masked_quantile_torch(values: torch.Tensor, mask: torch.Tensor, quantile: float) -> torch.Tensor:
    inf = torch.tensor(float("inf"), dtype=values.dtype, device=values.device)
    sorted_values = torch.sort(torch.where(mask, values, inf), dim=1).values
    counts = mask.to(dtype=torch.long).sum(dim=1)
    safe_counts = counts.clamp_min(1)
    index = torch.floor((safe_counts.to(dtype=values.dtype) - 1.0) * float(quantile)).to(dtype=torch.long)
    selected = sorted_values.gather(1, index[:, None]).squeeze(1)
    return torch.where(counts > 0, selected, torch.zeros_like(selected))


def _weighted_plane_fit_torch(points_l: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dtype = points_l.dtype
    safe_points = torch.where(mask[..., None], points_l, torch.zeros_like(points_l))
    ones = torch.ones_like(safe_points[..., 0])
    design = torch.stack([safe_points[..., 0], safe_points[..., 1], ones], dim=-1)
    weights = mask.to(dtype=dtype)
    xtw = design.transpose(1, 2) * weights[:, None, :]
    xtx = torch.bmm(xtw, design)
    xtz = torch.bmm(xtw, safe_points[..., 2:3])
    eye = torch.eye(3, dtype=dtype, device=points_l.device).unsqueeze(0)
    coeff = torch.linalg.solve(xtx + 1.0e-5 * eye, xtz).squeeze(-1)

    count = weights.sum(dim=1)
    median_z = _masked_quantile_torch(points_l[..., 2], mask, 0.5)
    fallback = torch.stack([torch.zeros_like(median_z), torch.zeros_like(median_z), median_z], dim=1)
    return torch.where((count >= 3.0)[:, None], coeff, fallback)


def _plane_z_torch(coeff: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    return coeff[:, 0:1] * points_xy[..., 0] + coeff[:, 1:2] * points_xy[..., 1] + coeff[:, 2:3]


def _fit_ground_plane_torch(points_l: torch.Tensor, finite: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    safe_points = torch.where(finite[..., None], points_l, torch.zeros_like(points_l))
    z = safe_points[..., 2]
    counts = finite.to(dtype=torch.long).sum(dim=1)
    inf = torch.tensor(float("inf"), dtype=z.dtype, device=z.device)
    sorted_z = torch.sort(torch.where(finite, z, inf), dim=1).values
    window = 0.24
    valid_window = torch.isfinite(sorted_z) & torch.isfinite(sorted_z[:, :1])
    layer_counts = ((sorted_z[:, :, None] - sorted_z[:, None, :]).abs() <= window) & valid_window[:, :, None]
    layer_counts = layer_counts.sum(dim=2)
    layer_index = torch.argmax(layer_counts, dim=1)
    layer_center = sorted_z.gather(1, layer_index[:, None]).squeeze(1)
    mask = finite & (torch.abs(z - layer_center[:, None]) <= window)
    enough_layer = mask.sum(dim=1) >= 16
    low = _masked_quantile_torch(z, finite, 0.15)
    high = _masked_quantile_torch(z, finite, 0.85)
    central = finite & (z >= low[:, None]) & (z <= high[:, None])
    mask = torch.where(enough_layer[:, None], mask, central)
    enough_initial = mask.sum(dim=1) >= 16
    mask = torch.where(enough_initial[:, None], mask, finite)

    coeff = _weighted_plane_fit_torch(safe_points, mask)
    for _ in range(5):
        residuals = z - _plane_z_torch(coeff, safe_points[..., :2])
        median = _masked_quantile_torch(residuals, mask, 0.5)
        abs_dev = torch.abs(residuals - median[:, None])
        mad = _masked_quantile_torch(abs_dev, mask, 0.5)
        threshold = torch.maximum(torch.full_like(mad, 0.10), 2.5 * 1.4826 * mad)
        new_mask = finite & (torch.abs(residuals - median[:, None]) <= threshold[:, None])
        enough_new = new_mask.sum(dim=1) >= 16
        mask = torch.where(enough_new[:, None], new_mask, mask)
        coeff = _weighted_plane_fit_torch(safe_points, mask)

    inliers = finite & (torch.abs(z - _plane_z_torch(coeff, safe_points[..., :2])) <= 0.18)
    fallback_coeff = torch.zeros_like(coeff)
    return torch.where((counts >= 16)[:, None], coeff, fallback_coeff), inliers


def estimate_task_d_poses_from_lidar_torch(
    ray_hits_w: Any,
    lidar_pos_w: Any,
    lidar_quat_w: Any,
    prior: LidarPerceptionPrior | None = None,
    max_distance: float | None = None,
) -> TorchLidarPerceptionResult:
    prior = prior or LidarPerceptionPrior()
    points_w = _as_torch(ray_hits_w).to(dtype=torch.float32)
    device = points_w.device
    if points_w.ndim == 2:
        points_w = points_w.unsqueeze(0)
    points_w = points_w.reshape(points_w.shape[0], -1, 3)
    pos_w = _as_torch(lidar_pos_w, device=device).to(dtype=torch.float32)
    quat_w = _as_torch(lidar_quat_w, device=device).to(dtype=torch.float32)
    if pos_w.ndim == 1:
        pos_w = pos_w.unsqueeze(0)
    if quat_w.ndim == 1:
        quat_w = quat_w.unsqueeze(0)

    finite = torch.isfinite(points_w).all(dim=-1)
    if max_distance is not None:
        dist = torch.linalg.norm(points_w - pos_w[:, None, :3], dim=-1)
        finite = finite & (dist <= float(max_distance) + 1.0e-3)
    points_l = world_points_to_lidar_frame_torch(points_w, pos_w, quat_w)
    points_l = torch.where(finite[..., None], points_l, torch.full_like(points_l, float("nan")))
    finite_l = finite & torch.isfinite(points_l).all(dim=-1)

    ground_plane, ground_inliers = _fit_ground_plane_torch(points_l, finite_l)
    box_pose, box_valid, box_confidence = _estimate_box_torch(points_l, finite_l, prior, ground_plane)
    ditch_pose, ditch_valid, ditch_confidence = _estimate_ditch_torch(points_l, finite_l, prior, ground_plane)
    return TorchLidarPerceptionResult(
        box_pose=box_pose,
        box_valid=box_valid,
        box_confidence=box_confidence,
        ditch_pose=ditch_pose,
        ditch_valid=ditch_valid,
        ditch_confidence=ditch_confidence,
        num_points=finite_l.sum(dim=1),
        num_ground_inliers=ground_inliers.sum(dim=1),
    )


def task_d_lidar_torch_debug_stats(
    ray_hits_w: Any,
    lidar_pos_w: Any,
    lidar_quat_w: Any,
    prior: LidarPerceptionPrior | None = None,
    max_distance: float | None = None,
) -> dict[str, torch.Tensor]:
    prior = prior or LidarPerceptionPrior()
    points_w = _as_torch(ray_hits_w).to(dtype=torch.float32)
    device = points_w.device
    if points_w.ndim == 2:
        points_w = points_w.unsqueeze(0)
    points_w = points_w.reshape(points_w.shape[0], -1, 3)
    pos_w = _as_torch(lidar_pos_w, device=device).to(dtype=torch.float32)
    quat_w = _as_torch(lidar_quat_w, device=device).to(dtype=torch.float32)
    if pos_w.ndim == 1:
        pos_w = pos_w.unsqueeze(0)
    if quat_w.ndim == 1:
        quat_w = quat_w.unsqueeze(0)

    finite = torch.isfinite(points_w).all(dim=-1)
    if max_distance is not None:
        dist = torch.linalg.norm(points_w - pos_w[:, None, :3], dim=-1)
        finite = finite & (dist <= float(max_distance) + 1.0e-3)
    points_l = world_points_to_lidar_frame_torch(points_w, pos_w, quat_w)
    points_l = torch.where(finite[..., None], points_l, torch.full_like(points_l, float("nan")))
    finite_l = finite & torch.isfinite(points_l).all(dim=-1)
    ground_plane, ground_inliers = _fit_ground_plane_torch(points_l, finite_l)

    z_rel = points_l[..., 2] - _plane_z_torch(ground_plane, points_l[..., :2])
    box_prior = prior.box
    ditch_prior = prior.ditch
    box_mask = (
        finite_l
        & (z_rel > max(0.12, 0.18 * float(box_prior.height)))
        & (z_rel < float(box_prior.height) + 0.45)
    )
    box_component = _select_box_window(points_l[..., :2], box_mask, float(box_prior.length), float(box_prior.width))
    box_component = _select_box_component(
        points_l[..., :2],
        box_component,
        float(box_prior.length),
        float(box_prior.width),
    )
    box_mask_025 = finite_l & (z_rel > 0.25) & (z_rel < float(box_prior.height) + 0.45)
    box_mask_035 = finite_l & (z_rel > 0.35) & (z_rel < float(box_prior.height) + 0.45)
    box_mask_045 = finite_l & (z_rel > 0.45) & (z_rel < float(box_prior.height) + 0.45)
    ditch_mask = finite_l & (z_rel < -max(0.18, 0.18 * float(ditch_prior.depth))) & (points_l[..., 0] > -0.75)

    inf = torch.tensor(float("inf"), dtype=points_l.dtype, device=points_l.device)
    box_min_xy = torch.where(box_mask[..., None], points_l[..., :2], inf).amin(dim=1)
    box_max_xy = torch.where(box_mask[..., None], points_l[..., :2], -inf).amax(dim=1)
    box_component_min_xy = torch.where(box_component[..., None], points_l[..., :2], inf).amin(dim=1)
    box_component_max_xy = torch.where(box_component[..., None], points_l[..., :2], -inf).amax(dim=1)
    box_component_yaw = _pca_yaw_masked(points_l[..., :2], box_component)
    _, _, box_component_extent_u, box_component_extent_v, _ = _choose_box_axis(
        points_l[..., :2],
        box_component,
        box_component_yaw,
        float(box_prior.length),
        float(box_prior.width),
    )
    ditch_min_xy = torch.where(ditch_mask[..., None], points_l[..., :2], inf).amin(dim=1)
    ditch_max_xy = torch.where(ditch_mask[..., None], points_l[..., :2], -inf).amax(dim=1)
    return {
        "ground_plane": ground_plane,
        "num_points": finite_l.sum(dim=1),
        "num_ground_inliers": ground_inliers.sum(dim=1),
        "box_count": box_mask.sum(dim=1),
        "box_count_025": box_mask_025.sum(dim=1),
        "box_count_035": box_mask_035.sum(dim=1),
        "box_count_045": box_mask_045.sum(dim=1),
        "box_min_xy": box_min_xy,
        "box_max_xy": box_max_xy,
        "box_component_count": box_component.sum(dim=1),
        "box_component_min_xy": box_component_min_xy,
        "box_component_max_xy": box_component_max_xy,
        "box_component_extent": torch.stack([box_component_extent_u, box_component_extent_v], dim=1),
        "ditch_count": ditch_mask.sum(dim=1),
        "ditch_min_xy": ditch_min_xy,
        "ditch_max_xy": ditch_max_xy,
    }
