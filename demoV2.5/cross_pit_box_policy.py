"""Demo-local inference wrapper for the CrossPitBox locomotion policy."""

from __future__ import annotations

import os
import re
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


ACTOR_OBS_DIM = 492
BLIND_ACTOR_OBS_DIM = 108
ACTION_DIM = 33
HEIGHT_SCAN_DIM = 5760
HEIGHTMAP_DIM = 384
HEIGHTMAP_SHAPE = (24, 16)
HEIGHTMAP_X_RANGE = (0.15, 2.45)
HEIGHTMAP_Y_RANGE = (-0.75, 0.75)
LIDAR_CHANNELS = 16
LIDAR_HORIZONTAL_RAYS = 360
LIDAR_VERTICAL_FOV_DEG = (-20.0, 20.0)
LIDAR_HORIZONTAL_FOV_DEG = (-180.0, 180.0)
LIDAR_HEIGHT_SCAN_OFFSET = 0.5
DEFAULT_FORWARD_COMMAND = 0.6644027
DEFAULT_TARGET_X = 7.8
ACTION_CLIP = 6.0
DEFAULT_HANDOFF_BLEND_STEPS = 0
DEFAULT_HANDOFF_MAX_DELTA = 0.0
DEFAULT_PREALIGN_STEPS = 0
DEFAULT_HEIGHTMAP_SOURCE = "terrain_map"
DEFAULT_STEP_DT = 0.02
DEFAULT_STABILIZE_STEPS = 0
DEFAULT_BLIND_STABILIZE_STEPS = 0
DEFAULT_STABILIZE_VEL_GAIN = 0.0
DEFAULT_VELOCITY_OBS_WARMUP_STEPS = 0
DEFAULT_BASE_ANG_VEL_OBS_CLIP = 0.0
DEFAULT_JOINT_VEL_OBS_CLIP = 0.0
DEFAULT_HEADING_KP = 0.0
DEFAULT_HEADING_COMMAND_CLIP = 1.0
DEFAULT_ACTION_OFFSET_SCALE = 1.0
DEFAULT_INITIAL_LAST_ACTION = "env_raw"
DEFAULT_D435_LINK_OFFSET_B = (0.1885, 0.0052, 0.4331)
D435_WAIST_BASE_OFFSET_B = (-0.00063307, -0.00047869, 0.04387194)
D435_WAIST_LINK_OFFSET = (0.05463455, 0.01787839, 0.43122387)

CROSS_PIT_TRAINING_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hand_Joint1_1",
    "left_hand_Joint2_1",
    "right_hand_Joint1_1",
    "right_hand_Joint2_1",
)

# TaskD observation/action managers are configured with the Unitree G1 joint-name
# list above. Snapshot JSON files record the articulation's internal PhysX order,
# which is different and must not be used to interpret action/proprio vectors.
TASKD_JOINT_NAMES = CROSS_PIT_TRAINING_JOINT_NAMES

_TASKD_TO_TRAINING_INDEX = torch.tensor(
    [TASKD_JOINT_NAMES.index(name) for name in CROSS_PIT_TRAINING_JOINT_NAMES],
    dtype=torch.long,
)
_TRAINING_TO_TASKD_INDEX = torch.tensor(
    [CROSS_PIT_TRAINING_JOINT_NAMES.index(name) for name in TASKD_JOINT_NAMES],
    dtype=torch.long,
)

# IsaacLab's native CrossPitBox observation manager resolves the Unitree G1 joint
# regex into PhysX articulation order, while TaskD's public action/proprio vectors
# use UNITREE_G1_29DOF_DEX1_CFG.joint_names order.  Actor actions must stay in
# the public action order, but joint_pos/joint_vel observations have to be
# reordered into the native observation order below.
CROSS_PIT_OBSERVATION_JOINT_INDEX_IN_TASKD_ORDER = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
    29,
    30,
    31,
    32,
)
CROSS_PIT_OBSERVATION_JOINT_NAMES = tuple(
    TASKD_JOINT_NAMES[index] for index in CROSS_PIT_OBSERVATION_JOINT_INDEX_IN_TASKD_ORDER
)
_TASKD_TO_OBSERVATION_INDEX = torch.tensor(CROSS_PIT_OBSERVATION_JOINT_INDEX_IN_TASKD_ORDER, dtype=torch.long)

# Counter-950 TaskD handoff pose in the CrossPitBox-v2.5 training frame.  This
# blind policy was trained from the GUI-equivalent 9.5 s state, after the box is
# already settled in the pit and before the robot steps onto it.
CROSS_PIT_ENV_ORIGIN_X = -4.199999809265137
CROSS_PIT_HANDOFF_LOCAL_X = 3.4137089252471924
CROSS_PIT_HANDOFF_LOCAL_Y = -1.5012480020523071
CROSS_PIT_HANDOFF_ROOT_Z = 0.6975134611129761
CROSS_PIT_HANDOFF_YAW = -0.13635359439321332
CROSS_PIT_NEAR_LIP_X = 3.72
CROSS_PIT_FAR_LIP_X = 4.68
CROSS_PIT_BOX_FRONT_X = 4.0634
CROSS_PIT_BOX_BACK_X = 4.6635
CROSS_PIT_BOX_CENTER_Y = -0.7996934652328491
CROSS_PIT_BOX_HALF_Y = 0.50
CROSS_PIT_BOX_TOP_Z = -0.20
CROSS_PIT_PIT_BOTTOM_Z = -1.00
CROSS_PIT_PLATFORM_TOP_Z = 0.927
CROSS_PIT_PLATFORM_Y_MIN = 0.60
CROSS_PIT_PLATFORM_Y_MAX = 3.50

# Raw JointPositionAction command corresponding to the CrossPitBox-v2.5
# counter-950 reset joint pose when the official TaskD action term uses scale=0.5
# and the original G1 default joint position as offset. The CrossPitBox actor was
# trained with this counter-950 pose as the action-term default offset, so deploy
# adds this action-space offset back before sending commands to TaskD.
CROSS_PIT_RESET_ACTION = torch.tensor(
    [
        -0.5136534333229065,
        -0.17924976348876953,
        -0.1762511134147644,
        1.3571871852874757,
        -0.291397430896759,
        -0.009772847406566143,
        -0.7144568920135498,
        0.056261371821165085,
        0.08106247335672379,
        1.9136158561706544,
        -1.0266241216659546,
        0.03713797777891159,
        0.41322994232177734,
        0.17704898118972778,
        0.12416505068540573,
        -0.04283792972564693,
        -0.028904095888137804,
        -0.014778261072933674,
        0.0023859834671020597,
        -1.1894111594301648e-05,
        0.0030589147936552763,
        -0.003374680643901229,
        -0.022906374931335405,
        0.02222572326660155,
        0.013066630810499191,
        0.008685359954833993,
        -0.0001080234651453793,
        0.002745487494394183,
        0.0017184086609631777,
        3.207889199256797e-05,
        1.987484097480674e-05,
        -2.975720167160134e-05,
        0.00014003780484199424,
    ],
    dtype=torch.float32,
)
CROSS_PIT_OFFICIAL_JOINT_POS_OFFSET = 0.5 * CROSS_PIT_RESET_ACTION

CROSS_PIT_HANDOFF_LAST_ACTION = torch.tensor(
    [
        -0.5318777561187744,
        0.06515301764011383,
        -0.2943190634250641,
        1.0854837894439697,
        2.623720645904541,
        -0.05229632556438446,
        -0.6801775693893433,
        -0.058367013931274414,
        0.15366503596305847,
        1.7844688892364502,
        -0.28546908497810364,
        0.21682728826999664,
        0.39855700731277466,
        0.16573604941368103,
        0.09754028916358948,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=torch.float32,
)

_ACTOR_WEIGHT_RE = re.compile(r"^actor\.(\d+)\.weight$")
_LIDAR_DIRECTION_CACHE: dict[tuple[str, torch.dtype], torch.Tensor] = {}
_GRID_CACHE: dict[tuple[str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def _as_2d_float_tensor(value: Any, *, device: torch.device) -> torch.Tensor:
    if value is None:
        raise ValueError("Expected tensor-like value, got None")
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        tensor = torch.as_tensor(value)
    tensor = tensor.to(device=device, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _extract_model_state_dict(loaded: Any) -> dict[str, torch.Tensor]:
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        return loaded["model_state_dict"]
    if isinstance(loaded, dict):
        return loaded
    raise TypeError(f"Unsupported CrossPitBox checkpoint type: {type(loaded)!r}")


def _actor_layer_indices(state_dict: dict[str, torch.Tensor]) -> list[int]:
    indices: list[int] = []
    for key in state_dict:
        match = _ACTOR_WEIGHT_RE.match(key)
        if match is not None:
            indices.append(int(match.group(1)))
    if not indices:
        raise ValueError("Checkpoint does not contain actor.*.weight tensors")
    return sorted(indices)


def _build_actor_from_state_dict(state_dict: dict[str, torch.Tensor]) -> nn.Sequential:
    layers: list[nn.Module] = []
    indices = _actor_layer_indices(state_dict)
    for layer_pos, index in enumerate(indices):
        weight_key = f"actor.{index}.weight"
        bias_key = f"actor.{index}.bias"
        if bias_key not in state_dict:
            raise ValueError(f"Checkpoint is missing {bias_key}")
        weight = state_dict[weight_key]
        bias = state_dict[bias_key]
        if len(weight.shape) != 2 or len(bias.shape) != 1:
            raise ValueError(f"Invalid actor layer tensor shapes for actor.{index}")
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        layers.append(linear)
        if layer_pos != len(indices) - 1:
            layers.append(nn.ELU())
    actor = nn.Sequential(*layers)
    first_linear = next(module for module in actor if isinstance(module, nn.Linear))
    last_linear = next(module for module in reversed(actor) if isinstance(module, nn.Linear))
    if first_linear.in_features not in (ACTOR_OBS_DIM, BLIND_ACTOR_OBS_DIM):
        raise ValueError(
            f"CrossPitBox actor expects {first_linear.in_features} obs, "
            f"expected {ACTOR_OBS_DIM} or {BLIND_ACTOR_OBS_DIM}"
        )
    if last_linear.out_features != ACTION_DIM:
        raise ValueError(f"CrossPitBox actor outputs {last_linear.out_features} actions, expected {ACTION_DIM}")
    return actor


def _taskd_lidar_directions(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (str(device), dtype)
    cached = _LIDAR_DIRECTION_CACHE.get(key)
    if cached is not None:
        return cached

    vertical_angles = torch.linspace(
        LIDAR_VERTICAL_FOV_DEG[0],
        LIDAR_VERTICAL_FOV_DEG[1],
        LIDAR_CHANNELS,
        device=device,
        dtype=dtype,
    )
    num_horizontal_angles = (
        math.ceil((LIDAR_HORIZONTAL_FOV_DEG[1] - LIDAR_HORIZONTAL_FOV_DEG[0]) / 1.0) + 1
    )
    horizontal_angles = torch.linspace(
        LIDAR_HORIZONTAL_FOV_DEG[0],
        LIDAR_HORIZONTAL_FOV_DEG[1],
        num_horizontal_angles,
        device=device,
        dtype=dtype,
    )[:-1]

    v_angles, h_angles = torch.meshgrid(torch.deg2rad(vertical_angles), torch.deg2rad(horizontal_angles), indexing="ij")
    directions = torch.stack(
        (
            torch.cos(v_angles) * torch.cos(h_angles),
            torch.cos(v_angles) * torch.sin(h_angles),
            torch.sin(v_angles),
        ),
        dim=-1,
    ).reshape(-1, 3)
    _LIDAR_DIRECTION_CACHE[key] = directions
    return directions


def _cross_pit_grid_points(*, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    key = (str(device), dtype)
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached

    x_bins, y_bins = HEIGHTMAP_SHAPE
    x = torch.arange(
        start=-(x_bins - 1) * 0.1 * 0.5,
        end=(x_bins - 1) * 0.1 * 0.5 + 1.0e-9,
        step=0.1,
        device=device,
        dtype=dtype,
    ) + 1.3
    y = torch.arange(
        start=-(y_bins - 1) * 0.1 * 0.5,
        end=(y_bins - 1) * 0.1 * 0.5 + 1.0e-9,
        step=0.1,
        device=device,
        dtype=dtype,
    )
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    points = (grid_x.reshape(-1), grid_y.reshape(-1))
    _GRID_CACHE[key] = points
    return points


def _roll_pitch_from_projected_gravity(projected_gravity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gravity = projected_gravity
    pitch = torch.asin(torch.clamp(gravity[:, 0:1], -1.0, 1.0))
    cos_pitch = torch.cos(pitch).clamp_min(1.0e-6)
    roll = torch.asin(torch.clamp(-gravity[:, 1:2] / cos_pitch, -1.0, 1.0))
    return roll, pitch


def _rotation_matrix_zxy(angles: torch.Tensor) -> torch.Tensor:
    """Return Rz(yaw) * Rx(roll) * Ry(pitch) for batched waist angles."""
    if angles.ndim == 1:
        angles = angles.unsqueeze(0)
    yaw = angles[:, 0]
    roll = angles[:, 1]
    pitch = angles[:, 2]
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)

    rot = torch.empty((angles.shape[0], 3, 3), device=angles.device, dtype=angles.dtype)
    rot[:, 0, 0] = cy * cp - sy * sr * sp
    rot[:, 0, 1] = -sy * cr
    rot[:, 0, 2] = cy * sp + sy * sr * cp
    rot[:, 1, 0] = sy * cp + cy * sr * sp
    rot[:, 1, 1] = cy * cr
    rot[:, 1, 2] = sy * sp - cy * sr * cp
    rot[:, 2, 0] = -cr * sp
    rot[:, 2, 1] = sr
    rot[:, 2, 2] = cr * cp
    return rot


def estimate_d435_link_offset_b_from_waist(waist_yaw_roll_pitch: torch.Tensor) -> torch.Tensor:
    """Estimate native d435_link position in the base frame from waist joints.

    Native CrossPitBox trains the v1 heightmap on a raycaster attached to
    ``d435_link``. That link moves with the G1 waist, so a fixed offset quickly
    shifts the synthetic terrain map relative to the actor's training input.
    The constants here are the static torso-to-camera transform recovered from
    native rollout sensor poses; the rotation order matches the G1 waist chain.
    """
    waist = waist_yaw_roll_pitch
    if waist.ndim == 1:
        waist = waist.unsqueeze(0)
    if waist.shape[-1] != 3:
        raise ValueError(f"Expected waist_yaw_roll_pitch shape (N, 3), got {tuple(waist.shape)}")
    base = waist.new_tensor(D435_WAIST_BASE_OFFSET_B).view(1, 3)
    link = waist.new_tensor(D435_WAIST_LINK_OFFSET).view(1, 3, 1)
    return base + torch.bmm(_rotation_matrix_zxy(waist), link.expand(waist.shape[0], -1, -1)).squeeze(-1)


def _yaw_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    if quat.ndim == 1:
        quat = quat.unsqueeze(0)
    qw = quat[:, 0]
    qx = quat[:, 1]
    qy = quat[:, 2]
    qz = quat[:, 3]
    return torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_inv = torch.cat((quat[:, 0:1], -quat[:, 1:]), dim=-1)
    qvec = quat_inv[:, 1:]
    qw = quat_inv[:, 0:1]
    t = 2.0 * torch.cross(qvec, vec, dim=-1)
    return vec + qw * t + torch.cross(qvec, t, dim=-1)


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    qvec = quat[:, 1:]
    qw = quat[:, 0:1]
    t = 2.0 * torch.cross(qvec, vec, dim=-1)
    return vec + qw * t + torch.cross(qvec, t, dim=-1)


def ray_hits_to_heightmap(
    ray_hits_w: torch.Tensor,
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
    *,
    default_height: float = -1.0,
) -> torch.Tensor:
    """Convert training ray-hit points into the actor's base-frame heightmap."""
    if ray_hits_w.ndim != 3 or ray_hits_w.shape[-2:] != (HEIGHTMAP_DIM, 3):
        raise ValueError(f"Expected ray_hits_w shape (N, {HEIGHTMAP_DIM}, 3), got {tuple(ray_hits_w.shape)}")
    if base_pos_w.ndim == 1:
        base_pos_w = base_pos_w.unsqueeze(0)
    if base_quat_w.ndim == 1:
        base_quat_w = base_quat_w.unsqueeze(0)

    num_envs = ray_hits_w.shape[0]
    rel_points = (ray_hits_w - base_pos_w[:, None, :]).reshape(-1, 3)
    quats = base_quat_w[:, None, :].expand(num_envs, HEIGHTMAP_DIM, 4).reshape(-1, 4)
    points_base = _quat_apply_inverse(quats, rel_points).reshape(num_envs, HEIGHTMAP_DIM, 3)
    heights = torch.clamp(points_base[..., 2], -1.0, 1.0)
    valid = torch.isfinite(ray_hits_w).all(dim=-1) & torch.isfinite(heights)
    return torch.where(valid, heights, torch.full_like(heights, default_height))


def head_depth_to_heightmap(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_pos_w: torch.Tensor,
    camera_quat_w: torch.Tensor,
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
    *,
    sample_stride: int = 4,
    default_height: float = -1.0,
) -> torch.Tensor:
    """Project TaskD head depth into the CrossPitBox actor heightmap shape.

    This is a diagnostic bridge for v1 only: it needs camera pose/intrinsics
    attached by local debug runners, since the official solution API only passes
    image tensors.
    """
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Expected head_depth shape (N, H, W[, 1]), got {tuple(depth.shape)}")
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if camera_pos_w.ndim == 1:
        camera_pos_w = camera_pos_w.unsqueeze(0)
    if camera_quat_w.ndim == 1:
        camera_quat_w = camera_quat_w.unsqueeze(0)
    if base_pos_w.ndim == 1:
        base_pos_w = base_pos_w.unsqueeze(0)
    if base_quat_w.ndim == 1:
        base_quat_w = base_quat_w.unsqueeze(0)

    device = depth.device
    dtype = depth.dtype
    num_envs, height, width = depth.shape
    rows = torch.arange(0, height, sample_stride, device=device, dtype=dtype)
    cols = torch.arange(0, width, sample_stride, device=device, dtype=dtype)
    v, u = torch.meshgrid(rows, cols, indexing="ij")
    u = u.reshape(-1)
    v = v.reshape(-1)
    sampled_depth = depth[:, ::sample_stride, ::sample_stride].reshape(num_envs, -1)

    fx = intrinsics[:, 0, 0].clamp_min(1.0e-6).unsqueeze(1)
    fy = intrinsics[:, 1, 1].clamp_min(1.0e-6).unsqueeze(1)
    cx = intrinsics[:, 0, 2].unsqueeze(1)
    cy = intrinsics[:, 1, 2].unsqueeze(1)

    x_cam = sampled_depth
    y_cam = -(u.unsqueeze(0) - cx) / fx * sampled_depth
    z_cam = -(v.unsqueeze(0) - cy) / fy * sampled_depth
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

    x_bins, y_bins = HEIGHTMAP_SHAPE
    x_min, x_max = HEIGHTMAP_X_RANGE
    y_min, y_max = HEIGHTMAP_Y_RANGE
    x_span = x_max - x_min
    y_span = y_max - y_min
    heightmap = torch.full((num_envs, x_bins * y_bins), default_height, device=device, dtype=dtype)

    valid_depth = sampled_depth > 0.0
    for env_id in range(num_envs):
        pts = points_base[env_id]
        valid = (
            valid_depth[env_id]
            & torch.isfinite(sampled_depth[env_id])
            & torch.isfinite(pts).all(dim=-1)
            & (pts[:, 0] >= x_min)
            & (pts[:, 0] <= x_max)
            & (pts[:, 1] >= y_min)
            & (pts[:, 1] <= y_max)
        )
        if not torch.any(valid):
            continue
        valid_pts = pts[valid]
        x_idx = torch.clamp(((valid_pts[:, 0] - x_min) / x_span * x_bins).long(), 0, x_bins - 1)
        y_idx = torch.clamp(((valid_pts[:, 1] - y_min) / y_span * y_bins).long(), 0, y_bins - 1)
        flat_idx = y_idx * x_bins + x_idx
        heights = torch.clamp(valid_pts[:, 2], -1.0, 1.0)
        heightmap[env_id].scatter_reduce_(0, flat_idx, heights, reduce="amax", include_self=True)

    return heightmap.clamp(-1.0, 1.0)


def terrain_map_heightmap(
    local_x: torch.Tensor,
    local_y: torch.Tensor,
    root_z: torch.Tensor,
    yaw: torch.Tensor,
    projected_gravity: torch.Tensor | None = None,
    d435_link_offset_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Approximate the training d435_link downward raycast heightmap from fixed TaskD geometry."""
    if local_x.ndim == 1:
        local_x = local_x.unsqueeze(1)
    if local_y.ndim == 1:
        local_y = local_y.unsqueeze(1)
    if root_z.ndim == 1:
        root_z = root_z.unsqueeze(1)
    if yaw.ndim == 1:
        yaw = yaw.unsqueeze(1)

    device = local_x.device
    dtype = local_x.dtype
    rel_x, rel_y = _cross_pit_grid_points(device=device, dtype=dtype)
    rel_x = rel_x.unsqueeze(0)
    rel_y = rel_y.unsqueeze(0)
    if projected_gravity is None:
        roll = torch.zeros_like(yaw)
        pitch = torch.zeros_like(yaw)
    else:
        gravity = projected_gravity.to(device=device, dtype=dtype)
        if gravity.ndim == 1:
            gravity = gravity.unsqueeze(0)
        if gravity.shape[-1] != 3:
            raise ValueError(f"Expected projected_gravity shape (N, 3), got {tuple(gravity.shape)}")
        roll, pitch = _roll_pitch_from_projected_gravity(gravity)

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    cos_roll = torch.cos(roll)
    sin_roll = torch.sin(roll)
    sin_pitch = torch.sin(pitch)
    cos_pitch = torch.cos(pitch)

    if d435_link_offset_b is None:
        d435_offset = torch.zeros((local_x.shape[0], 3), device=device, dtype=dtype)
    else:
        d435_offset = d435_link_offset_b.to(device=device, dtype=dtype)
        if d435_offset.ndim == 1:
            d435_offset = d435_offset.unsqueeze(0)
        if d435_offset.shape[-1] != 3:
            raise ValueError(f"Expected d435_link_offset_b shape (N, 3), got {tuple(d435_offset.shape)}")
        if d435_offset.shape[0] == 1 and local_x.shape[0] > 1:
            d435_offset = d435_offset.expand(local_x.shape[0], -1)
        if d435_offset.shape[0] != local_x.shape[0]:
            raise ValueError(
                f"d435_link_offset_b batch size {d435_offset.shape[0]} does not match {local_x.shape[0]}"
            )

    # MultiMeshRayCasterCfg(ray_alignment="yaw") keeps the scan grid aligned
    # with heading, not full body roll/pitch. The sensor is attached to d435_link,
    # so the ray grid origin follows the camera link rather than the base/root.
    sensor_dx = cos_yaw * d435_offset[:, 0:1] - sin_yaw * d435_offset[:, 1:2]
    sensor_dy = sin_yaw * d435_offset[:, 0:1] + cos_yaw * d435_offset[:, 1:2]
    sensor_x = local_x + sensor_dx
    sensor_y = local_y + sensor_dy
    world_dx = cos_yaw * rel_x - sin_yaw * rel_y
    world_dy = sin_yaw * rel_x + cos_yaw * rel_y
    hit_x = sensor_x + world_dx
    hit_y = sensor_y + world_dy

    hit_z = torch.zeros_like(hit_x)
    in_pit_x = (hit_x >= CROSS_PIT_NEAR_LIP_X) & (hit_x <= CROSS_PIT_FAR_LIP_X)
    hit_z = torch.where(in_pit_x, torch.full_like(hit_z, CROSS_PIT_PIT_BOTTOM_Z), hit_z)

    on_platform = (
        (hit_x >= CROSS_PIT_NEAR_LIP_X)
        & (hit_x <= CROSS_PIT_FAR_LIP_X)
        & (hit_y >= CROSS_PIT_PLATFORM_Y_MIN)
        & (hit_y <= CROSS_PIT_PLATFORM_Y_MAX)
    )
    hit_z = torch.where(on_platform, torch.full_like(hit_z, CROSS_PIT_PLATFORM_TOP_Z), hit_z)

    on_box = (
        (hit_x >= CROSS_PIT_BOX_FRONT_X)
        & (hit_x <= CROSS_PIT_BOX_BACK_X)
        & (torch.abs(hit_y - CROSS_PIT_BOX_CENTER_Y) <= CROSS_PIT_BOX_HALF_Y)
    )
    hit_z = torch.where(on_box, torch.full_like(hit_z, CROSS_PIT_BOX_TOP_Z), hit_z)
    local_z = hit_z - root_z

    if projected_gravity is None:
        return torch.clamp(local_z, -1.0, 1.0)

    # The training raycast stores hit points in the robot base frame. Reconstruct
    # the base-frame z from the TaskD local terrain map using the projected gravity
    # vector, which captures the roll/pitch at handoff.
    base_z = (
        (cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll) * (sensor_dx + world_dx)
        + (sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll) * (sensor_dy + world_dy)
        + cos_pitch * cos_roll * local_z
    )
    return torch.clamp(base_z, -1.0, 1.0)


def compress_official_height_scan(height_scan: torch.Tensor, projected_gravity: torch.Tensor | None = None) -> torch.Tensor:
    """Project TaskD 16x360 LiDAR height scan into the CrossPitBox 24x16 heightmap slot.

    The trained CrossPitBox actor expects a flattened, forward-facing height grid. The
    official TaskD ``extero`` observation is a 360-degree lidar height scan, so a simple
    channel average destroys lateral locality. Reconstructing approximate local ray hit
    points from the known lidar angles preserves the 2-D structure the locomotion policy
    relies on at handoff.
    """
    height_scan = _as_2d_float_tensor(height_scan, device=height_scan.device if isinstance(height_scan, torch.Tensor) else torch.device("cpu"))
    if height_scan.shape[-1] != HEIGHT_SCAN_DIM:
        raise ValueError(f"Expected extero height scan with {HEIGHT_SCAN_DIM} values, got shape {tuple(height_scan.shape)}")

    num_envs = height_scan.shape[0]
    device = height_scan.device
    dtype = height_scan.dtype
    directions = _taskd_lidar_directions(device=device, dtype=dtype)
    scan = height_scan.reshape(num_envs, -1)

    vertical_drop = scan + LIDAR_HEIGHT_SCAN_OFFSET
    dir_z = directions[:, 2].unsqueeze(0)
    distance = vertical_drop / (-dir_z.clamp(max=-1.0e-6))
    points_x = distance * directions[:, 0].unsqueeze(0)
    points_y = distance * directions[:, 1].unsqueeze(0)
    points_z_world = -vertical_drop

    if projected_gravity is not None:
        gravity = _as_2d_float_tensor(projected_gravity, device=device).to(dtype=dtype)
        if gravity.shape[-1] != 3:
            raise ValueError(f"Expected projected_gravity shape (N, 3), got {tuple(gravity.shape)}")
        if gravity.shape[0] == 1 and num_envs > 1:
            gravity = gravity.expand(num_envs, -1)
        if gravity.shape[0] != num_envs:
            raise ValueError(
                f"projected_gravity batch size {gravity.shape[0]} does not match height scan batch size {num_envs}"
            )
        roll, pitch = _roll_pitch_from_projected_gravity(gravity)
        yaw = torch.zeros_like(roll)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        cos_roll = torch.cos(roll)
        sin_roll = torch.sin(roll)
        sin_pitch = torch.sin(pitch)
        cos_pitch = torch.cos(pitch)
        points_z = (
            (cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll) * points_x
            + (sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll) * points_y
            + cos_pitch * cos_roll * points_z_world
        )
    else:
        points_z = points_z_world

    x_bins, y_bins = HEIGHTMAP_SHAPE
    x_min, x_max = HEIGHTMAP_X_RANGE
    y_min, y_max = HEIGHTMAP_Y_RANGE
    x_span = x_max - x_min
    y_span = y_max - y_min

    valid = (
        torch.isfinite(scan)
        & torch.isfinite(distance)
        & (directions[:, 2].unsqueeze(0) < -1.0e-4)
        & (vertical_drop > 0.0)
        & (distance > 0.0)
        & (distance <= 10.0)
        & (points_x >= x_min)
        & (points_x <= x_max)
        & (points_y >= y_min)
        & (points_y <= y_max)
    )

    heightmap = torch.full((num_envs, x_bins * y_bins), -1.0, device=device, dtype=dtype)
    for env_id in range(num_envs):
        env_valid = valid[env_id]
        if not torch.any(env_valid):
            continue

        x_idx = torch.clamp(((points_x[env_id, env_valid] - x_min) / x_span * x_bins).long(), 0, x_bins - 1)
        y_idx = torch.clamp(((points_y[env_id, env_valid] - y_min) / y_span * y_bins).long(), 0, y_bins - 1)
        # GridPatternCfg(ordering="xy") flattens with y as the outer loop and x as the inner loop.
        flat_idx = y_idx * x_bins + x_idx
        heights = torch.clamp(points_z[env_id, env_valid], -1.0, 1.0)
        heightmap[env_id].scatter_reduce_(0, flat_idx, heights, reduce="amax", include_self=True)

    return heightmap.clamp(-1.0, 1.0)


class CrossPitBoxPolicy:
    """Actor-only wrapper for the trained 33-D G1 CrossPitBox checkpoint."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        device: str | torch.device | None = None,
        target_x: float = DEFAULT_TARGET_X,
        forward_command: float = DEFAULT_FORWARD_COMMAND,
        action_clip: float = ACTION_CLIP,
        handoff_blend_steps: int | None = None,
        handoff_max_delta: float | None = None,
        prealign_steps: int | None = None,
        stabilize_steps: int | None = None,
        stabilize_vel_gain: float | None = None,
        initial_last_action: str | None = None,
        require_deploy_defaults: bool | None = None,
    ) -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model_path = self._resolve_model_path(model_path)
        self.target_x = float(target_x)
        self.forward_command = float(forward_command)
        self.action_clip = float(action_clip)
        self.heightmap_source = os.environ.get("ATEC_CROSS_PIT_HEIGHTMAP_SOURCE", DEFAULT_HEIGHTMAP_SOURCE).strip().lower()
        self.step_dt = float(os.environ.get("ATEC_DEMO_STEP_DT", DEFAULT_STEP_DT))
        self.handoff_blend_steps = int(
            os.environ.get(
                "ATEC_CROSS_PIT_HANDOFF_BLEND_STEPS",
                DEFAULT_HANDOFF_BLEND_STEPS if handoff_blend_steps is None else handoff_blend_steps,
            )
        )
        self.handoff_max_delta = float(
            os.environ.get(
                "ATEC_CROSS_PIT_HANDOFF_MAX_DELTA",
                DEFAULT_HANDOFF_MAX_DELTA if handoff_max_delta is None else handoff_max_delta,
            )
        )
        self.prealign_steps = int(
            os.environ.get(
                "ATEC_CROSS_PIT_PREALIGN_STEPS",
                DEFAULT_PREALIGN_STEPS if prealign_steps is None else prealign_steps,
            )
        )
        self._requested_stabilize_steps = stabilize_steps
        self.stabilize_vel_gain = float(
            os.environ.get(
                "ATEC_CROSS_PIT_STABILIZE_VEL_GAIN",
                DEFAULT_STABILIZE_VEL_GAIN if stabilize_vel_gain is None else stabilize_vel_gain,
            )
        )
        self.velocity_obs_warmup_steps = int(
            os.environ.get("ATEC_CROSS_PIT_VELOCITY_OBS_WARMUP_STEPS", DEFAULT_VELOCITY_OBS_WARMUP_STEPS)
        )
        self.base_ang_vel_obs_clip = float(
            os.environ.get("ATEC_CROSS_PIT_BASE_ANG_VEL_OBS_CLIP", DEFAULT_BASE_ANG_VEL_OBS_CLIP)
        )
        self.joint_vel_obs_clip = float(
            os.environ.get("ATEC_CROSS_PIT_JOINT_VEL_OBS_CLIP", DEFAULT_JOINT_VEL_OBS_CLIP)
        )
        self.heading_kp = float(os.environ.get("ATEC_CROSS_PIT_HEADING_KP", DEFAULT_HEADING_KP))
        self.heading_command_clip = float(
            os.environ.get("ATEC_CROSS_PIT_HEADING_COMMAND_CLIP", DEFAULT_HEADING_COMMAND_CLIP)
        )
        self.action_offset_scale = float(
            os.environ.get("ATEC_CROSS_PIT_ACTION_OFFSET_SCALE", DEFAULT_ACTION_OFFSET_SCALE)
        )
        self.initial_last_action = os.environ.get(
            "ATEC_CROSS_PIT_INITIAL_LAST_ACTION",
            DEFAULT_INITIAL_LAST_ACTION if initial_last_action is None else initial_last_action,
        ).strip().lower()
        if self.initial_last_action not in ("zero", "env", "env_raw", "handoff_raw"):
            raise ValueError(
                "ATEC_CROSS_PIT_INITIAL_LAST_ACTION must be 'zero', 'env', 'env_raw', or 'handoff_raw', "
                f"got {self.initial_last_action!r}"
            )

        try:
            loaded = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
        except TypeError:
            loaded = torch.load(str(self.model_path), map_location="cpu")
        self.actor = _build_actor_from_state_dict(_extract_model_state_dict(loaded)).to(self.device).eval()
        first_linear = next(module for module in self.actor if isinstance(module, nn.Linear))
        self.actor_obs_dim = int(first_linear.in_features)
        self.uses_heightmap = self.actor_obs_dim == ACTOR_OBS_DIM
        self.stabilize_steps = self._resolve_stabilize_steps(self._requested_stabilize_steps)
        self.require_deploy_defaults = self._resolve_require_deploy_defaults(require_deploy_defaults)
        self._validate_deploy_defaults_if_requested()
        self._last_action: torch.Tensor | None = None
        self._last_training_action: torch.Tensor | None = None
        self._handoff_action: torch.Tensor | None = None
        self._current_env_last_action: torch.Tensor | None = None
        self._handoff_blend_step = 0
        self._prealign_start_action: torch.Tensor | None = None
        self._prealign_step = 0
        self._stabilize_step = 0
        self._current_joint_pos_rel: torch.Tensor | None = None
        self._current_joint_vel: torch.Tensor | None = None
        self._local_xy_yaw: torch.Tensor | None = None
        self._local_root_z: torch.Tensor | None = None
        self._policy_step_count = 0
        self._num_envs = 0
        self._debug_snapshot: dict[str, Any] = {
            "stage": "cross_pit_box_init",
            "model_path": str(self.model_path),
            "target_x": self.target_x,
            "forward_command": self.forward_command,
            "heightmap_source": self.heightmap_source,
            "stabilize_steps": self.stabilize_steps,
            "stabilize_vel_gain": self.stabilize_vel_gain,
            "handoff_blend_steps": self.handoff_blend_steps,
            "handoff_max_delta": self.handoff_max_delta,
            "prealign_steps": self.prealign_steps,
            "velocity_obs_warmup_steps": self.velocity_obs_warmup_steps,
            "base_ang_vel_obs_clip": self.base_ang_vel_obs_clip,
            "joint_vel_obs_clip": self.joint_vel_obs_clip,
            "heading_kp": self.heading_kp,
            "heading_command_clip": self.heading_command_clip,
            "action_offset_scale": self.action_offset_scale,
            "initial_last_action": self.initial_last_action,
            "require_deploy_defaults": self.require_deploy_defaults,
            "actor_obs_dim": self.actor_obs_dim,
            "uses_heightmap": self.uses_heightmap,
        }
        print(
            "[CrossPitBoxPolicy] "
            f"device={self.device} model={self.model_path.name} target_x={self.target_x:.4f} "
            f"actor_obs_dim={self.actor_obs_dim} "
            f"stabilize_steps={self.stabilize_steps} prealign_steps={self.prealign_steps} "
            f"handoff_blend_steps={self.handoff_blend_steps}"
        )

    def _validate_deploy_defaults_if_requested(self) -> None:
        if not self.require_deploy_defaults:
            return
        errors: list[str] = []
        if self.model_path.name != "cross_pit_box_v2_5_blind_model_19999.pt":
            errors.append(f"checkpoint={self.model_path.name!r}")
        if self.actor_obs_dim != BLIND_ACTOR_OBS_DIM:
            errors.append(f"actor_obs_dim={self.actor_obs_dim}")
        if self.uses_heightmap:
            errors.append("uses_heightmap=True")
        if self.stabilize_steps != DEFAULT_BLIND_STABILIZE_STEPS:
            errors.append(f"stabilize_steps={self.stabilize_steps}")
        if self.action_offset_scale != 1.0:
            errors.append(f"action_offset_scale={self.action_offset_scale}")
        if self.initial_last_action != "env_raw":
            errors.append(f"initial_last_action={self.initial_last_action!r}")
        if self.handoff_blend_steps != 0:
            errors.append(f"handoff_blend_steps={self.handoff_blend_steps}")
        if self.handoff_max_delta != 0.0:
            errors.append(f"handoff_max_delta={self.handoff_max_delta}")
        if errors:
            raise ValueError("CrossPitBox deploy defaults check failed: " + ", ".join(errors))

    def _resolve_model_path(self, explicit: str | Path | None) -> Path:
        value = (
            explicit
            or os.environ.get("ATEC_CROSS_PIT_BOX_CHECKPOINT")
            or "cross_pit_box_v2_5_blind_model_19999.pt"
        )
        path = Path(value).expanduser()
        candidates = [path] if path.is_absolute() else [Path.cwd() / path, self.base_dir / path]
        resolved_path = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
        if resolved_path is None:
            raise FileNotFoundError(
                f"Missing CrossPitBox checkpoint: {candidates[-1].resolve()}. "
                "Place cross_pit_box_v2_5_blind_model_19999.pt in demoV2.5/ or set ATEC_CROSS_PIT_BOX_CHECKPOINT."
            )
        return resolved_path

    def _resolve_stabilize_steps(self, explicit: int | None) -> int:
        env_value = os.environ.get("ATEC_CROSS_PIT_STABILIZE_STEPS")
        if env_value is not None:
            return int(env_value)
        if explicit is not None:
            return int(explicit)
        if self.actor_obs_dim == BLIND_ACTOR_OBS_DIM:
            return DEFAULT_BLIND_STABILIZE_STEPS
        return DEFAULT_STABILIZE_STEPS

    @staticmethod
    def _resolve_require_deploy_defaults(explicit: bool | None) -> bool:
        env_value = os.environ.get("ATEC_CROSS_PIT_REQUIRE_DEPLOY_DEFAULTS")
        if env_value is not None:
            return env_value.strip().lower() in ("1", "true", "yes", "on")
        return bool(explicit)

    def reset(self, **kwargs: Any) -> None:
        self._last_action = None
        self._last_training_action = None
        self._handoff_action = None
        self._current_env_last_action = None
        self._handoff_blend_step = 0
        self._prealign_start_action = None
        self._prealign_step = 0
        self._stabilize_step = 0
        self._current_joint_pos_rel = None
        self._current_joint_vel = None
        self._local_xy_yaw = None
        self._local_root_z = None
        self._policy_step_count = 0
        self._num_envs = 0
        self._debug_snapshot["stage"] = "cross_pit_box_reset"

    def predicts(self, obs: dict[str, Any], current_score: float) -> dict[str, Any]:
        action = self.predict(obs)
        return {"action": action.detach().cpu().tolist(), "giveup": False}

    @torch.no_grad()
    def predict(self, obs: dict[str, Any]) -> torch.Tensor:
        policy_obs = self.build_policy_obs(obs)
        raw_training_action = torch.clamp(self.actor(policy_obs), -self.action_clip, self.action_clip)
        raw_action = self._training_action_to_taskd_action(raw_training_action)
        action = self._stabilize_action(raw_action)
        if action is None:
            action = self._prealign_action(raw_action)
        if action is None:
            action = self._smooth_handoff_action(raw_action)
        action = self._rate_limit_action(action)
        self._last_action = action.detach().clone()
        self._last_training_action = self._taskd_action_to_training_action(action).detach().clone()
        self._debug_snapshot.update(
            {
                "stage": "cross_pit_box",
                "action_head": action[0, :8].detach().cpu().tolist(),
                "raw_action_head": raw_action[0, :8].detach().cpu().tolist(),
                "raw_training_action_head": raw_training_action[0, :8].detach().cpu().tolist(),
                "stabilize_step": self._stabilize_step,
                "prealign_step": self._prealign_step,
                "handoff_blend_step": self._handoff_blend_step,
                "velocity_obs_gain": self._velocity_obs_gain(),
                "velocity_commands": policy_obs[0, 6:9].detach().cpu().tolist(),
                "heightmap_mean": (
                    float(policy_obs[:, -HEIGHTMAP_DIM:].mean().detach().cpu()) if self.uses_heightmap else None
                ),
                "heightmap_min": (
                    float(policy_obs[:, -HEIGHTMAP_DIM:].min().detach().cpu()) if self.uses_heightmap else None
                ),
                "heightmap_max": (
                    float(policy_obs[:, -HEIGHTMAP_DIM:].max().detach().cpu()) if self.uses_heightmap else None
                ),
                "base_ang_vel_actor_norm": float(policy_obs[:, 0:3].norm(dim=-1).mean().detach().cpu()),
                "joint_pos_actor_norm": float(policy_obs[:, 9:42].norm(dim=-1).mean().detach().cpu()),
                "joint_vel_actor_norm": float(policy_obs[:, 42:75].norm(dim=-1).mean().detach().cpu()),
                "local_x": float(self._local_xy_yaw[0, 0].detach().cpu()) if self._local_xy_yaw is not None else None,
                "local_y": float(self._local_xy_yaw[0, 1].detach().cpu()) if self._local_xy_yaw is not None else None,
                "local_root_z": float(self._local_root_z[0].detach().cpu()) if self._local_root_z is not None else None,
            }
        )
        if os.environ.get("ATEC_CROSS_PIT_LOG_POLICY_OBS_FULL", "0").strip().lower() in ("1", "true", "yes", "on"):
            self._debug_snapshot["policy_obs_full"] = policy_obs[0].detach().cpu().tolist()
        self._policy_step_count += 1
        return action

    def build_policy_obs(self, obs: dict[str, Any]) -> torch.Tensor:
        proprio = _as_2d_float_tensor(obs["proprio"], device=self.device)
        if proprio.shape[-1] < 111:
            raise ValueError(f"Expected TaskD-G1 proprio with at least 111 values, got {tuple(proprio.shape)}")
        self._ensure_batch_size(proprio.shape[0])

        velocity_obs_gain = self._velocity_obs_gain()
        base_ang_vel = self._warmup_velocity_obs(
            proprio[:, 3:6],
            clip_abs=self.base_ang_vel_obs_clip,
            gain=velocity_obs_gain,
        )
        projected_gravity = proprio[:, 9:12]
        local_xy_yaw = self._update_local_xy_yaw(proprio)
        velocity_commands = proprio.new_zeros((proprio.shape[0], 3))
        velocity_commands[:, 0] = self.forward_command
        velocity_commands[:, 2] = self._heading_command(obs, proprio=proprio, local_xy_yaw=local_xy_yaw)
        taskd_joint_pos_rel = proprio[:, 12:45]
        handoff_joint_pos_rel = CROSS_PIT_OFFICIAL_JOINT_POS_OFFSET.to(device=proprio.device, dtype=proprio.dtype)
        joint_pos = self._taskd_to_observation(taskd_joint_pos_rel - handoff_joint_pos_rel.view(1, -1))
        raw_joint_vel = proprio[:, 45:78]
        training_joint_vel = self._taskd_to_observation(raw_joint_vel)
        joint_vel = self._warmup_velocity_obs(
            training_joint_vel * 0.1,
            clip_abs=self.joint_vel_obs_clip,
            gain=velocity_obs_gain,
        )
        self._current_joint_pos_rel = taskd_joint_pos_rel.detach().clone()
        self._current_joint_vel = raw_joint_vel.detach().clone()
        env_last_action = proprio[:, 78:111]
        self._current_env_last_action = env_last_action.detach().clone()
        last_action_training = self._last_training_action
        if last_action_training is None or last_action_training.shape[0] != proprio.shape[0]:
            if self.initial_last_action == "env":
                last_action_training = self._taskd_action_to_training_action(env_last_action)
            elif self.initial_last_action == "env_raw":
                last_action_training = self._taskd_to_training(env_last_action)
            elif self.initial_last_action == "handoff_raw":
                handoff_action = CROSS_PIT_HANDOFF_LAST_ACTION.to(device=proprio.device, dtype=proprio.dtype)
                handoff_action = handoff_action.view(1, -1).expand_as(env_last_action)
                last_action_training = self._taskd_to_training(handoff_action)
            else:
                last_action_training = torch.zeros_like(env_last_action)
        else:
            last_action_training = last_action_training.to(device=proprio.device, dtype=proprio.dtype)

        obs_terms = [
            base_ang_vel,
            projected_gravity,
            velocity_commands,
            joint_pos,
            joint_vel,
            last_action_training,
        ]
        if self.uses_heightmap:
            obs_terms.append(self._heightmap_from_obs(obs, proprio=proprio, local_xy_yaw=local_xy_yaw))
        policy_obs = torch.cat(obs_terms, dim=-1)
        if policy_obs.shape[-1] != self.actor_obs_dim:
            raise RuntimeError(
                f"Built CrossPitBox obs with {policy_obs.shape[-1]} dims, expected {self.actor_obs_dim}"
            )
        return policy_obs

    def _velocity_obs_gain(self) -> float:
        if self.velocity_obs_warmup_steps <= 0:
            return 1.0
        return min(1.0, float(self._policy_step_count) / float(self.velocity_obs_warmup_steps))

    @staticmethod
    def _warmup_velocity_obs(value: torch.Tensor, *, clip_abs: float, gain: float) -> torch.Tensor:
        if clip_abs > 0.0:
            value = torch.clamp(value, -clip_abs, clip_abs)
        if gain < 1.0:
            value = value * value.new_tensor(gain)
        return value

    def _heightmap_from_obs(
        self,
        obs: dict[str, Any],
        *,
        proprio: torch.Tensor,
        local_xy_yaw: torch.Tensor,
    ) -> torch.Tensor:
        debug_ray_hits = obs.get("_debug_depth_ray_hits_w")
        if debug_ray_hits is not None:
            root_pos_w = obs.get("_debug_root_pos_w")
            root_quat_w = obs.get("_debug_root_quat_w")
            if root_pos_w is not None and root_quat_w is not None:
                self._debug_snapshot["heightmap_source_actual"] = "debug_ray_hits"
                return ray_hits_to_heightmap(
                    _as_2d_float_tensor(debug_ray_hits, device=self.device).reshape(proprio.shape[0], HEIGHTMAP_DIM, 3),
                    _as_2d_float_tensor(root_pos_w, device=self.device),
                    _as_2d_float_tensor(root_quat_w, device=self.device),
                )

        if self.heightmap_source == "terrain_map":
            self._debug_snapshot["heightmap_source_actual"] = "terrain_map"
            debug_local_xy_yaw = self._debug_local_pose_from_obs(obs, proprio=proprio)
            if debug_local_xy_yaw is not None:
                local_xy_yaw = debug_local_xy_yaw
            if self._local_root_z is None:
                self._local_root_z = proprio.new_full((proprio.shape[0],), CROSS_PIT_HANDOFF_ROOT_Z)
            d435_link_offset_b = self._estimate_current_d435_link_offset_b()
            if d435_link_offset_b is None:
                d435_link_offset_b = proprio.new_tensor(DEFAULT_D435_LINK_OFFSET_B).view(1, 3)
            return terrain_map_heightmap(
                local_xy_yaw[:, 0],
                local_xy_yaw[:, 1],
                self._local_root_z.to(device=proprio.device, dtype=proprio.dtype),
                local_xy_yaw[:, 2],
                projected_gravity=proprio[:, 9:12],
                d435_link_offset_b=d435_link_offset_b.to(device=proprio.device, dtype=proprio.dtype),
            )

        if self.heightmap_source == "head_depth":
            image = obs.get("image")
            head_depth = image.get("head_depth") if isinstance(image, dict) else None
            intrinsics = obs.get("_debug_head_camera_intrinsics")
            camera_pos_w = obs.get("_debug_head_camera_pos_w")
            camera_quat_w = obs.get("_debug_head_camera_quat_w")
            root_pos_w = obs.get("_debug_root_pos_w")
            root_quat_w = obs.get("_debug_root_quat_w")
            if (
                head_depth is not None
                and intrinsics is not None
                and camera_pos_w is not None
                and camera_quat_w is not None
                and root_pos_w is not None
                and root_quat_w is not None
            ):
                self._debug_snapshot["heightmap_source_actual"] = "head_depth"
                return head_depth_to_heightmap(
                    _as_2d_float_tensor(head_depth, device=self.device).reshape(
                        proprio.shape[0], *head_depth.shape[-3:]
                    ),
                    _as_2d_float_tensor(intrinsics, device=self.device).reshape(proprio.shape[0], 3, 3),
                    _as_2d_float_tensor(camera_pos_w, device=self.device),
                    _as_2d_float_tensor(camera_quat_w, device=self.device),
                    _as_2d_float_tensor(root_pos_w, device=self.device),
                    _as_2d_float_tensor(root_quat_w, device=self.device),
                    sample_stride=int(os.environ.get("ATEC_CROSS_PIT_HEAD_DEPTH_SAMPLE_STRIDE", "4")),
                )

        extero = obs.get("extero")
        if extero is not None:
            self._debug_snapshot["heightmap_source_actual"] = "extero_lidar"
            return compress_official_height_scan(
                _as_2d_float_tensor(extero, device=self.device),
                projected_gravity=proprio[:, 9:12],
            )
        self._debug_snapshot["heightmap_source_actual"] = "constant"
        return torch.full((self._num_envs, HEIGHTMAP_DIM), -1.0, device=self.device, dtype=torch.float32)

    def _estimate_current_d435_link_offset_b(self) -> torch.Tensor | None:
        if self._current_joint_pos_rel is None:
            return None
        joint_pos_native = self._taskd_to_observation(self._current_joint_pos_rel)
        waist = torch.stack(
            (
                joint_pos_native[:, CROSS_PIT_OBSERVATION_JOINT_NAMES.index("waist_yaw_joint")],
                joint_pos_native[:, CROSS_PIT_OBSERVATION_JOINT_NAMES.index("waist_roll_joint")],
                joint_pos_native[:, CROSS_PIT_OBSERVATION_JOINT_NAMES.index("waist_pitch_joint")],
            ),
            dim=-1,
        )
        return estimate_d435_link_offset_b_from_waist(waist)

    def _heading_command(
        self,
        obs: dict[str, Any],
        *,
        proprio: torch.Tensor,
        local_xy_yaw: torch.Tensor,
    ) -> torch.Tensor:
        if self.heading_kp <= 0.0:
            return proprio.new_zeros(proprio.shape[0])

        yaw = local_xy_yaw[:, 2]
        root_quat_w = obs.get("_debug_root_quat_w")
        if root_quat_w is not None:
            root_quat = _as_2d_float_tensor(root_quat_w, device=self.device)
            if root_quat.shape[-1] == 4:
                yaw = _yaw_from_quat_wxyz(root_quat).to(device=proprio.device, dtype=proprio.dtype)

        command = -self.heading_kp * yaw.to(device=proprio.device, dtype=proprio.dtype)
        if self.heading_command_clip > 0.0:
            command = torch.clamp(command, -self.heading_command_clip, self.heading_command_clip)
        return command

    def _debug_local_pose_from_obs(self, obs: dict[str, Any], *, proprio: torch.Tensor) -> torch.Tensor | None:
        root_pos_w = obs.get("_debug_root_pos_w")
        root_quat_w = obs.get("_debug_root_quat_w")
        if root_pos_w is None or root_quat_w is None:
            return None

        root_pos = _as_2d_float_tensor(root_pos_w, device=self.device)
        root_quat = _as_2d_float_tensor(root_quat_w, device=self.device)
        if root_pos.shape[-1] != 3 or root_quat.shape[-1] != 4:
            return None

        local_x = root_pos[:, 0] - root_pos.new_tensor(CROSS_PIT_ENV_ORIGIN_X)
        local_y = root_pos[:, 1]
        yaw = _yaw_from_quat_wxyz(root_quat)
        self._local_xy_yaw = torch.stack((local_x, local_y, yaw), dim=-1).to(device=proprio.device, dtype=proprio.dtype)
        self._local_root_z = root_pos[:, 2].to(device=proprio.device, dtype=proprio.dtype)
        return self._local_xy_yaw

    def _update_local_xy_yaw(self, proprio: torch.Tensor) -> torch.Tensor:
        if self._local_xy_yaw is None or self._local_xy_yaw.shape[0] != proprio.shape[0]:
            self._local_xy_yaw = proprio.new_tensor(
                [CROSS_PIT_HANDOFF_LOCAL_X, CROSS_PIT_HANDOFF_LOCAL_Y, CROSS_PIT_HANDOFF_YAW]
            ).view(1, 3).repeat(proprio.shape[0], 1)
            self._local_root_z = proprio.new_full((proprio.shape[0],), CROSS_PIT_HANDOFF_ROOT_Z)
            return self._local_xy_yaw

        lin_vel_b = proprio[:, 0:3]
        ang_vel_b = proprio[:, 3:6]
        projected_gravity = proprio[:, 9:12]
        roll, pitch = _roll_pitch_from_projected_gravity(projected_gravity)
        roll = roll[:, 0]
        pitch = pitch[:, 0]
        yaw = self._local_xy_yaw[:, 2]
        cos_roll = torch.cos(roll)
        sin_roll = torch.sin(roll)
        cos_pitch = torch.cos(pitch).clamp_min(1.0e-6)
        sin_pitch = torch.sin(pitch)
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        vel_x_w = (
            cos_yaw * cos_pitch * lin_vel_b[:, 0]
            + (cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll) * lin_vel_b[:, 1]
            + (cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll) * lin_vel_b[:, 2]
        )
        vel_y_w = (
            sin_yaw * cos_pitch * lin_vel_b[:, 0]
            + (sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll) * lin_vel_b[:, 1]
            + (sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll) * lin_vel_b[:, 2]
        )
        vel_z_w = (
            -sin_pitch * lin_vel_b[:, 0]
            + cos_pitch * sin_roll * lin_vel_b[:, 1]
            + cos_pitch * cos_roll * lin_vel_b[:, 2]
        )
        dx = vel_x_w * self.step_dt
        dy = vel_y_w * self.step_dt
        self._local_xy_yaw[:, 0] = self._local_xy_yaw[:, 0] + dx
        self._local_xy_yaw[:, 1] = self._local_xy_yaw[:, 1] + dy
        if self._local_root_z is None or self._local_root_z.shape[0] != proprio.shape[0]:
            self._local_root_z = proprio.new_full((proprio.shape[0],), CROSS_PIT_HANDOFF_ROOT_Z)
        self._local_root_z = self._local_root_z + vel_z_w * self.step_dt
        yaw_rate = (sin_roll / cos_pitch) * ang_vel_b[:, 1] + (cos_roll / cos_pitch) * ang_vel_b[:, 2]
        self._local_xy_yaw[:, 2] = self._local_xy_yaw[:, 2] + yaw_rate * self.step_dt
        return self._local_xy_yaw

    @staticmethod
    def _taskd_to_training(value: torch.Tensor) -> torch.Tensor:
        index = _TASKD_TO_TRAINING_INDEX.to(device=value.device)
        return value.index_select(-1, index)

    @staticmethod
    def _taskd_to_observation(value: torch.Tensor) -> torch.Tensor:
        index = _TASKD_TO_OBSERVATION_INDEX.to(device=value.device)
        return value.index_select(-1, index)

    @staticmethod
    def _training_to_taskd(value: torch.Tensor) -> torch.Tensor:
        index = _TRAINING_TO_TASKD_INDEX.to(device=value.device)
        return value.index_select(-1, index)

    def _training_action_to_taskd_action(self, value: torch.Tensor) -> torch.Tensor:
        taskd_delta = self._training_to_taskd(value)
        if self.action_offset_scale == 0.0:
            return taskd_delta
        reset_action = CROSS_PIT_RESET_ACTION.to(device=value.device, dtype=value.dtype).view(1, -1)
        return taskd_delta + reset_action * value.new_tensor(self.action_offset_scale)

    def _taskd_action_to_training_action(self, value: torch.Tensor) -> torch.Tensor:
        if self.action_offset_scale == 0.0:
            return self._taskd_to_training(value)
        reset_action = CROSS_PIT_RESET_ACTION.to(device=value.device, dtype=value.dtype).view(1, -1)
        return self._taskd_to_training(value - reset_action * value.new_tensor(self.action_offset_scale))

    def _stabilize_action(self, raw_action: torch.Tensor) -> torch.Tensor | None:
        if self.stabilize_steps <= 0 or self._stabilize_step >= self.stabilize_steps:
            return None
        if self._current_joint_pos_rel is None or self._current_joint_vel is None:
            return None

        joint_pos_rel = self._current_joint_pos_rel.to(device=raw_action.device, dtype=raw_action.dtype)
        joint_vel = self._current_joint_vel.to(device=raw_action.device, dtype=raw_action.dtype)
        target_joint_pos_rel = joint_pos_rel - self.stabilize_vel_gain * joint_vel
        action = torch.clamp(2.0 * target_joint_pos_rel, -self.action_clip, self.action_clip)
        self._stabilize_step += 1
        if self._stabilize_step >= self.stabilize_steps:
            self._handoff_action = action.detach().clone()
            self._handoff_blend_step = 0
        return action

    def _prealign_action(self, raw_action: torch.Tensor) -> torch.Tensor | None:
        if self.prealign_steps <= 0 or self._prealign_step >= self.prealign_steps:
            return None

        if self._prealign_start_action is None or self._prealign_start_action.shape != raw_action.shape:
            if self._current_env_last_action is None or self._current_env_last_action.shape != raw_action.shape:
                self._prealign_start_action = torch.zeros_like(raw_action)
            else:
                self._prealign_start_action = self._current_env_last_action.detach().clone()

        alpha = float(self._prealign_step + 1) / float(max(self.prealign_steps, 1))
        start_action = self._prealign_start_action.to(device=raw_action.device, dtype=raw_action.dtype)
        target_action = self._training_action_to_taskd_action(torch.zeros_like(raw_action))
        target_action = target_action.expand_as(raw_action)
        action = start_action + alpha * (target_action - start_action)
        self._prealign_step += 1

        if self._prealign_step >= self.prealign_steps:
            self._handoff_action = action.detach().clone()
            self._handoff_blend_step = 0
        return action

    def _smooth_handoff_action(self, raw_action: torch.Tensor) -> torch.Tensor:
        if self.handoff_blend_steps <= 0:
            return raw_action

        if self._handoff_action is None or self._handoff_action.shape != raw_action.shape:
            if self._current_env_last_action is None or self._current_env_last_action.shape != raw_action.shape:
                self._handoff_action = torch.zeros_like(raw_action)
            else:
                self._handoff_action = self._current_env_last_action.detach().clone()
            self._handoff_blend_step = 0

        if self._handoff_blend_step >= self.handoff_blend_steps:
            return raw_action

        alpha = float(self._handoff_blend_step + 1) / float(max(self.handoff_blend_steps, 1))
        handoff_action = self._handoff_action.to(device=raw_action.device, dtype=raw_action.dtype)
        action = handoff_action + alpha * (raw_action - handoff_action)

        self._handoff_blend_step += 1
        return action

    def _rate_limit_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.handoff_max_delta <= 0.0:
            return action
        previous_action = self._last_action
        if previous_action is None or previous_action.shape != action.shape:
            previous_action = self._current_env_last_action
        if previous_action is None or previous_action.shape != action.shape:
            return action
        previous_action = previous_action.to(device=action.device, dtype=action.dtype)
        max_delta = torch.full_like(action, self.handoff_max_delta)
        return torch.maximum(torch.minimum(action, previous_action + max_delta), previous_action - max_delta)

    def _ensure_batch_size(self, num_envs: int) -> None:
        if self._num_envs == num_envs:
            return
        self._num_envs = int(num_envs)
        self._last_action = None
        self._last_training_action = None
        self._handoff_action = None
        self._current_env_last_action = None
        self._handoff_blend_step = 0
        self._prealign_start_action = None
        self._prealign_step = 0
        self._stabilize_step = 0
        self._current_joint_pos_rel = None
        self._current_joint_vel = None
        self._local_xy_yaw = None
        self._local_root_z = None
        self._policy_step_count = 0

    def get_debug_snapshot(self) -> dict[str, Any]:
        return dict(self._debug_snapshot)
