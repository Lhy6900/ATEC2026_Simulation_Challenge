from __future__ import annotations

from typing import Any

import numpy as np


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def finite_stats(value: Any) -> dict[str, float | int | None]:
    array = as_numpy(value).astype(np.float32, copy=False)
    finite = array[np.isfinite(array)]
    total = int(array.size)
    if total == 0 or finite.size == 0:
        return {"count": 0, "valid_ratio": 0.0, "min": None, "max": None, "mean": None}
    return {
        "count": int(finite.size),
        "valid_ratio": float(finite.size / total),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }


def depth_to_uint8(depth: Any, max_depth: float = 10.0) -> np.ndarray:
    return normalize_heightmap_to_uint8(depth, value_range=(0.0, max_depth), positive_only=True)


def squeeze_first_sensor_frame(value: Any) -> np.ndarray:
    array = as_numpy(value).astype(np.float32, copy=False)
    if array.ndim >= 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


def normalize_heightmap_to_uint8(
    heightmap: Any,
    value_range: tuple[float, float] | None = None,
    positive_only: bool = False,
) -> np.ndarray:
    height_array = squeeze_first_sensor_frame(heightmap)
    valid = np.isfinite(height_array)
    if positive_only:
        valid &= height_array > 0.0

    image = np.zeros(height_array.shape, dtype=np.uint8)
    if not np.any(valid):
        return image

    if value_range is None:
        low = float(height_array[valid].min())
        high = float(height_array[valid].max())
    else:
        low, high = value_range
    if high <= low:
        return image

    clipped = np.clip(height_array[valid], low, high)
    image[valid] = ((clipped - low) / (high - low) * 255.0).astype(np.uint8)
    return image


def depth_to_camera_heightmap(
    depth: Any,
    intrinsics: Any | None = None,
) -> np.ndarray:
    """Convert a depth image to camera-frame vertical coordinates.

    The returned value is -Y in the ROS optical camera frame, so larger values
    are visually treated as higher points relative to the camera.
    """
    depth_array = squeeze_first_sensor_frame(depth)
    valid = np.isfinite(depth_array) & (depth_array > 0.0)
    height, width = depth_array.shape

    if intrinsics is None:
        fy = float(max(height, 1))
        cy = float((height - 1) * 0.5)
    else:
        matrix = as_numpy(intrinsics).astype(np.float32, copy=False)
        if matrix.ndim == 3:
            matrix = matrix[0]
        fy = float(matrix[1, 1])
        cy = float(matrix[1, 2])
        if fy == 0.0:
            fy = float(max(height, 1))
            cy = float((height - 1) * 0.5)

    rows = np.arange(height, dtype=np.float32).reshape(height, 1)
    heightmap = -((rows - cy) / fy) * depth_array
    heightmap = heightmap.astype(np.float32, copy=False)
    heightmap[~valid] = np.nan
    return heightmap


def _first_matrix(value: Any) -> np.ndarray:
    matrix = as_numpy(value).astype(np.float32, copy=False)
    if matrix.ndim == 3:
        matrix = matrix[0]
    return matrix


def _first_vector(value: Any) -> np.ndarray:
    vector = as_numpy(value).astype(np.float32, copy=False)
    if vector.ndim == 2:
        vector = vector[0]
    return vector


def _quat_apply(quat_wxyz: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    xyz = quat_wxyz[1:4]
    t = np.cross(xyz, vectors) * 2.0
    return vectors + quat_wxyz[0] * t + np.cross(xyz, t)


def _yaw_only_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat_wxyz
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    return np.array([np.cos(yaw * 0.5), 0.0, 0.0, np.sin(yaw * 0.5)], dtype=np.float32)


def depth_to_world_points(
    depth: Any,
    intrinsics: Any,
    camera_pos_w: Any,
    camera_quat_w_ros: Any,
    stride: int = 24,
    max_depth: float | None = None,
) -> np.ndarray:
    depth_array = squeeze_first_sensor_frame(depth)
    height, width = depth_array.shape
    stride = max(1, int(stride))

    rows = np.arange(0, height, stride, dtype=np.float32)
    cols = np.arange(0, width, stride, dtype=np.float32)
    grid_cols, grid_rows = np.meshgrid(cols, rows)
    sampled_depth = depth_array[grid_rows.astype(np.int64), grid_cols.astype(np.int64)]

    valid = np.isfinite(sampled_depth) & (sampled_depth > 0.0)
    if max_depth is not None:
        valid &= sampled_depth <= float(max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    intrinsic = _first_matrix(intrinsics)
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    if fx == 0.0 or fy == 0.0:
        return np.empty((0, 3), dtype=np.float32)

    z = sampled_depth[valid].astype(np.float32, copy=False)
    u = grid_cols[valid] + 0.5
    v = grid_rows[valid] + 0.5
    points_camera_ros = np.stack(
        [
            (u - cx) / fx * z,
            (v - cy) / fy * z,
            z,
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    pos_w = _first_vector(camera_pos_w)
    quat_w_ros = _first_vector(camera_quat_w_ros)
    points_w = _quat_apply(quat_w_ros, points_camera_ros) + pos_w
    return points_w.astype(np.float32, copy=False)


def lidar_hits_to_world_points(
    ray_hits_w: Any,
    stride: int = 8,
    sensor_pos_w: Any | None = None,
    max_distance: float | None = None,
) -> np.ndarray:
    hits = as_numpy(ray_hits_w).astype(np.float32, copy=False)
    if hits.ndim == 3 and hits.shape[0] == 1:
        hits = hits[0]
    hits = hits.reshape(-1, 3)
    stride = max(1, int(stride))
    hits = hits[::stride]
    valid = np.all(np.isfinite(hits), axis=1)
    if sensor_pos_w is not None and max_distance is not None:
        origin = _first_vector(sensor_pos_w)
        distances = np.linalg.norm(hits - origin.reshape(1, 3), axis=1)
        valid &= distances <= float(max_distance) + 1.0e-3
    return hits[valid].astype(np.float32, copy=False)


def lidar_height_scan_to_world_points(
    scan: Any,
    ray_directions: Any,
    sensor_pos_w: Any,
    sensor_quat_w: Any | None = None,
    stride: int = 8,
    height_offset: float = 0.5,
    max_distance: float | None = None,
    yaw_only: bool = True,
) -> np.ndarray:
    """Reconstruct a yaw-aligned LiDAR height map as world-frame points.

    IsaacLab's height_scan observation is sensor_z - hit_z - offset. This
    routine keeps the horizontal grid attached to the sensor yaw so pitch/roll
    of the torso does not make the visualized height map slide forward/back.
    """
    scan_array = as_numpy(scan).astype(np.float32, copy=False)
    if scan_array.ndim == 2 and scan_array.shape[0] == 1:
        scan_array = scan_array[0]
    scan_array = scan_array.reshape(-1)

    directions = as_numpy(ray_directions).astype(np.float32, copy=False)
    if directions.ndim == 3 and directions.shape[0] == 1:
        directions = directions[0]
    directions = directions.reshape(-1, 3)

    count = min(scan_array.shape[0], directions.shape[0])
    if count == 0:
        return np.empty((0, 3), dtype=np.float32)
    stride = max(1, int(stride))
    indices = np.arange(0, count, stride, dtype=np.int64)
    heights = scan_array[indices]
    directions = directions[indices]

    origin = _first_vector(sensor_pos_w)
    if sensor_quat_w is not None:
        quat = _first_vector(sensor_quat_w)
        if yaw_only:
            quat = _yaw_only_quat(quat)
        directions = _quat_apply(quat, directions)

    hit_z = origin[2] - float(height_offset) - heights
    dz = directions[:, 2]
    valid = np.isfinite(heights) & np.all(np.isfinite(directions), axis=1) & (np.abs(dz) > 1.0e-4)
    distances = (hit_z - origin[2]) / dz
    valid &= np.isfinite(distances) & (distances > 0.0)
    if max_distance is not None:
        valid &= distances <= float(max_distance) + 1.0e-3
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    points = origin.reshape(1, 3) + directions * distances.reshape(-1, 1)
    points[:, 2] = hit_z
    return points[valid].astype(np.float32, copy=False)


def point_cloud_xy_report(points_w: Any, origin_w: Any | None = None) -> str:
    points = as_numpy(points_w).astype(np.float32, copy=False).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.size == 0:
        return "lidar_xy: no finite points"

    xy = points[:, :2]
    centroid_xy = xy.mean(axis=0)
    message = f"lidar_xy_center=({centroid_xy[0]:+.2f},{centroid_xy[1]:+.2f})"
    if origin_w is None:
        return message

    origin = _first_vector(origin_w)
    origin_xy = origin[:2]
    radii = np.linalg.norm(xy - origin_xy.reshape(1, 2), axis=1)
    centroid_offset = np.linalg.norm(centroid_xy - origin_xy)
    return (
        f"lidar_origin=({origin[0]:+.2f},{origin[1]:+.2f},{origin[2]:+.2f}) | "
        f"{message} | center_delta_xy={centroid_offset:.2f}m | "
        f"radius_xy=({radii.min():.2f},{radii.mean():.2f},{radii.max():.2f})m"
    )


def make_heightmap_overlay(
    heightmap: Any,
    value_range: tuple[float, float] | None = None,
    positive_only: bool = False,
    scatter_stride: int = 1,
) -> np.ndarray:
    base = normalize_heightmap_to_uint8(heightmap, value_range=value_range, positive_only=positive_only)
    overlay = np.repeat(base[..., None], 3, axis=2)
    height_array = squeeze_first_sensor_frame(heightmap)
    valid = np.isfinite(height_array)
    if positive_only:
        valid &= height_array > 0.0
    if scatter_stride > 1:
        scatter_mask = np.zeros_like(valid, dtype=bool)
        scatter_mask[::scatter_stride, ::scatter_stride] = True
        valid &= scatter_mask
    overlay[valid] = np.array([0, 0, 255], dtype=np.uint8)
    return overlay


def lidar_scan_to_heightmap(scan: Any, channels: int = 16) -> np.ndarray:
    scan_array = squeeze_first_sensor_frame(scan).reshape(-1)
    if scan_array.size % channels != 0:
        return scan_array.reshape(1, -1)
    return scan_array.reshape(channels, scan_array.size // channels)


def format_stats(name: str, stats: dict[str, float | int | None]) -> str:
    if stats["count"] == 0:
        return f"{name}: no finite samples"
    return (
        f"{name}: valid={stats['valid_ratio']:.1%}, "
        f"min={stats['min']:.3f}, max={stats['max']:.3f}, mean={stats['mean']:.3f}"
    )
