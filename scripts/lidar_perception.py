from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np

try:
    from sensor_vis_utils import as_numpy
except ImportError:  # pragma: no cover - package import fallback for tests
    from scripts.sensor_vis_utils import as_numpy


@dataclass(frozen=True)
class BoxPrior:
    length: float = 1.0
    width: float = 0.8
    height: float = 0.6


@dataclass(frozen=True)
class DitchPrior:
    length: float = 7.0
    width: float = 0.95
    depth: float = 1.0


@dataclass(frozen=True)
class LidarPerceptionPrior:
    box: BoxPrior = field(default_factory=BoxPrior)
    ditch: DitchPrior = field(default_factory=DitchPrior)


@dataclass
class Pose2DEstimate:
    label: str
    valid: bool = False
    x: float = math.nan
    y: float = math.nan
    yaw: float = math.nan
    confidence: float = 0.0
    num_points: int = 0
    extent_x: float = math.nan
    extent_y: float = math.nan
    message: str = "not detected"


@dataclass
class LidarPerceptionResult:
    box: Pose2DEstimate
    ditch: Pose2DEstimate
    num_points: int
    num_ground_inliers: int


def wrap_axis_yaw(yaw: float) -> float:
    """Wrap an unoriented rectangle axis to [-pi/2, pi/2)."""
    wrapped = (float(yaw) + math.pi / 2.0) % math.pi - math.pi / 2.0
    return wrapped


def quat_inverse_apply(quat_wxyz: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    inv_quat = np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float32)
    xyz = inv_quat[1:4]
    t = np.cross(xyz.reshape(1, 3), vectors) * 2.0
    return vectors + inv_quat[0] * t + np.cross(xyz.reshape(1, 3), t)


def world_points_to_lidar_frame(points_w: Any, lidar_pos_w: Any, lidar_quat_w: Any) -> np.ndarray:
    points = as_numpy(points_w).astype(np.float32, copy=False).reshape(-1, 3)
    origin = as_numpy(lidar_pos_w).astype(np.float32, copy=False)
    quat = as_numpy(lidar_quat_w).astype(np.float32, copy=False)
    if origin.ndim == 2:
        origin = origin[0]
    if quat.ndim == 2:
        quat = quat[0]
    valid = np.all(np.isfinite(points), axis=1)
    points = points[valid] - origin.reshape(1, 3)
    return quat_inverse_apply(quat, points).astype(np.float32, copy=False)


def fit_ground_plane(points_l: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit z = ax + by + c to the dominant terrain surface."""
    points = points_l[np.all(np.isfinite(points_l), axis=1)]
    if points.shape[0] < 16:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32), np.zeros(points_l.shape[0], dtype=bool)

    z = points[:, 2]
    low, high = np.percentile(z, [15.0, 85.0])
    mask = (z >= low) & (z <= high)
    if int(mask.sum()) < 16:
        mask = np.ones(points.shape[0], dtype=bool)

    coeff = np.array([0.0, 0.0, float(np.median(z[mask]))], dtype=np.float32)
    for _ in range(5):
        sample = points[mask]
        if sample.shape[0] < 16:
            break
        design = np.column_stack([sample[:, 0], sample[:, 1], np.ones(sample.shape[0], dtype=np.float32)])
        coeff, *_ = np.linalg.lstsq(design, sample[:, 2], rcond=None)
        residuals = points[:, 2] - plane_z(coeff, points[:, :2])
        median = float(np.median(residuals[mask]))
        mad = float(np.median(np.abs(residuals[mask] - median)))
        threshold = max(0.10, 2.5 * 1.4826 * mad)
        new_mask = np.abs(residuals - median) <= threshold
        if int(new_mask.sum()) < 16:
            break
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    residuals_all = points_l[:, 2] - plane_z(coeff, points_l[:, :2])
    inliers_all = np.isfinite(residuals_all) & (np.abs(residuals_all) <= 0.18)
    return coeff.astype(np.float32, copy=False), inliers_all


def plane_z(coeff: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return coeff[0] * xy[:, 0] + coeff[1] * xy[:, 1] + coeff[2]


def cluster_xy(points_xy: np.ndarray, radius: float, min_points: int) -> list[np.ndarray]:
    if points_xy.shape[0] == 0:
        return []

    radius = float(radius)
    cell_size = radius
    cell_map: dict[tuple[int, int], list[int]] = {}
    cells = np.floor(points_xy / cell_size).astype(np.int32)
    for idx, cell in enumerate(cells):
        cell_map.setdefault((int(cell[0]), int(cell[1])), []).append(idx)

    visited = np.zeros(points_xy.shape[0], dtype=bool)
    clusters: list[np.ndarray] = []
    radius_sq = radius * radius

    for start in range(points_xy.shape[0]):
        if visited[start]:
            continue
        visited[start] = True
        queue = [start]
        cluster = []
        while queue:
            idx = queue.pop()
            cluster.append(idx)
            cx, cy = cells[idx]
            for nx in range(int(cx) - 1, int(cx) + 2):
                for ny in range(int(cy) - 1, int(cy) + 2):
                    for nb in cell_map.get((nx, ny), []):
                        if visited[nb]:
                            continue
                        if float(np.sum((points_xy[nb] - points_xy[idx]) ** 2)) <= radius_sq:
                            visited[nb] = True
                            queue.append(nb)
        if len(cluster) >= min_points:
            clusters.append(np.asarray(cluster, dtype=np.int64))
    return clusters


def pca_yaw(points_xy: np.ndarray) -> float:
    centered = points_xy - points_xy.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2:
        return 0.0
    cov = centered.T @ centered / max(1, centered.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    return wrap_axis_yaw(math.atan2(float(axis[1]), float(axis[0])))


def oriented_bounds(points_xy: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
    v = np.array([-math.sin(yaw), math.cos(yaw)], dtype=np.float32)
    proj_u = points_xy @ u
    proj_v = points_xy @ v
    return u, v, proj_u, proj_v


def infer_rectangle_pose(
    points_xy: np.ndarray,
    yaw: float,
    length: float | None = None,
    width: float | None = None,
    infer_partial_center: bool = False,
) -> tuple[np.ndarray, float, float]:
    u, v, proj_u, proj_v = oriented_bounds(points_xy, yaw)
    min_u, max_u = float(np.min(proj_u)), float(np.max(proj_u))
    min_v, max_v = float(np.min(proj_v)), float(np.max(proj_v))
    extent_u = max_u - min_u
    extent_v = max_v - min_v
    center_u = 0.5 * (min_u + max_u)
    center_v = 0.5 * (min_v + max_v)

    if infer_partial_center and length is not None and width is not None:
        center_u = _infer_axis_center_from_visible_bounds(proj_u, length, extent_u, center_u)
        center_v = _infer_axis_center_from_visible_bounds(proj_v, width, extent_v, center_v)

    center = center_u * u + center_v * v
    return center.astype(np.float32, copy=False), extent_u, extent_v


def dimension_match_score(observed: float, expected: float) -> float:
    """Score how well an observed projected extent matches a known object size."""
    expected = max(float(expected), 1.0e-6)
    observed = max(0.0, float(observed))
    over = max(0.0, observed - expected) / expected
    under = max(0.0, expected - observed) / expected
    return 2.0 * over + 0.35 * under


def _infer_axis_center_from_visible_bounds(
    projections: np.ndarray,
    expected_size: float,
    observed_extent: float,
    midpoint: float,
) -> float:
    if observed_extent >= 0.65 * expected_size:
        return midpoint
    min_p = float(np.min(projections))
    max_p = float(np.max(projections))
    mean_p = float(np.mean(projections))
    half = 0.5 * expected_size
    if mean_p >= 0.0:
        return min_p + half
    return max_p - half


def estimate_box_pose(points_l: np.ndarray, ground_plane: np.ndarray, prior: BoxPrior) -> Pose2DEstimate:
    heights = points_l[:, 2] - plane_z(ground_plane, points_l[:, :2])
    valid = (
        np.all(np.isfinite(points_l), axis=1)
        & np.isfinite(heights)
        & (heights > max(0.12, 0.18 * prior.height))
        & (heights < prior.height + 0.45)
    )
    candidate_points = points_l[valid]
    clusters = cluster_xy(candidate_points[:, :2], radius=0.30, min_points=12)

    best: tuple[float, Pose2DEstimate] | None = None
    for cluster_idx in clusters:
        cluster = candidate_points[cluster_idx]
        base_yaw = pca_yaw(cluster[:, :2])
        for yaw in (base_yaw, wrap_axis_yaw(base_yaw + math.pi / 2.0)):
            center, extent_length_axis, extent_width_axis = infer_rectangle_pose(
                cluster[:, :2],
                yaw,
                length=prior.length,
                width=prior.width,
                infer_partial_center=True,
            )
            if extent_length_axis > prior.length + 0.55 or extent_width_axis > prior.width + 0.45:
                continue
            if max(extent_length_axis, extent_width_axis) < 0.18 or min(extent_length_axis, extent_width_axis) < 0.08:
                continue

            extent_score = dimension_match_score(extent_length_axis, prior.length) + dimension_match_score(
                extent_width_axis, prior.width
            )
            size_bonus = min(1.0, cluster.shape[0] / 80.0)
            compactness = 1.0 / (1.0 + extent_score)
            confidence = max(0.0, min(1.0, 0.65 * compactness + 0.35 * size_bonus))
            range_penalty = 0.03 * float(np.linalg.norm(center))
            score = extent_score - 0.01 * cluster.shape[0] + range_penalty
            estimate = Pose2DEstimate(
                label="box",
                valid=True,
                x=float(center[0]),
                y=float(center[1]),
                yaw=wrap_axis_yaw(yaw),
                confidence=confidence,
                num_points=int(cluster.shape[0]),
                extent_x=float(extent_length_axis),
                extent_y=float(extent_width_axis),
                message="ok",
            )
            if best is None or score < best[0]:
                best = (score, estimate)

    if best is None:
        return Pose2DEstimate(label="box", message=f"no compact high cluster ({candidate_points.shape[0]} high pts)")
    return best[1]


def estimate_ditch_pose(points_l: np.ndarray, ground_plane: np.ndarray, prior: DitchPrior) -> Pose2DEstimate:
    heights = points_l[:, 2] - plane_z(ground_plane, points_l[:, :2])
    valid = (
        np.all(np.isfinite(points_l), axis=1)
        & np.isfinite(heights)
        & (heights < -max(0.35, 0.35 * prior.depth))
    )
    candidate_points = points_l[valid]
    clusters = cluster_xy(candidate_points[:, :2], radius=0.45, min_points=24)

    best: tuple[float, Pose2DEstimate] | None = None
    for cluster_idx in clusters:
        cluster = candidate_points[cluster_idx]
        center_xy = cluster[:, :2].mean(axis=0)
        if center_xy[0] < -0.75:
            continue
        yaw = pca_yaw(cluster[:, :2])
        center, extent_u, extent_v = infer_rectangle_pose(cluster[:, :2], yaw)
        long_extent = max(extent_u, extent_v)
        short_extent = min(extent_u, extent_v)
        if short_extent < 0.25 or short_extent > prior.width + 0.7:
            continue
        if long_extent < max(0.45, 1.25 * short_extent):
            continue

        width_score = abs(short_extent - prior.width) / max(prior.width, 1.0e-6)
        front_bonus = max(0.0, min(1.0, center[0] / 3.0))
        density_bonus = min(1.0, cluster.shape[0] / 180.0)
        confidence = max(0.0, min(1.0, 0.45 * (1.0 / (1.0 + width_score)) + 0.25 * density_bonus + 0.30 * front_bonus))
        score = width_score - 0.002 * cluster.shape[0] - 0.1 * front_bonus
        estimate = Pose2DEstimate(
            label="ditch",
            valid=True,
            x=float(center[0]),
            y=float(center[1]),
            yaw=wrap_axis_yaw(yaw),
            confidence=confidence,
            num_points=int(cluster.shape[0]),
            extent_x=float(long_extent),
            extent_y=float(short_extent),
            message="ok",
        )
        if best is None or score < best[0]:
            best = (score, estimate)

    if best is None:
        return Pose2DEstimate(label="ditch", message=f"no low strip cluster ({candidate_points.shape[0]} low pts)")
    return best[1]


def estimate_task_d_poses_from_lidar(
    ray_hits_w: Any,
    lidar_pos_w: Any,
    lidar_quat_w: Any,
    prior: LidarPerceptionPrior | None = None,
    max_distance: float | None = None,
) -> LidarPerceptionResult:
    prior = prior or LidarPerceptionPrior()
    points_w = as_numpy(ray_hits_w).astype(np.float32, copy=False).reshape(-1, 3)
    pos_w = as_numpy(lidar_pos_w).astype(np.float32, copy=False)
    if pos_w.ndim == 2:
        pos_w = pos_w[0]
    valid = np.all(np.isfinite(points_w), axis=1)
    if max_distance is not None:
        valid &= np.linalg.norm(points_w - pos_w.reshape(1, 3), axis=1) <= float(max_distance) + 1.0e-3
    points_w = points_w[valid]

    if points_w.shape[0] < 32:
        invalid_box = Pose2DEstimate(label="box", message="not enough lidar hits")
        invalid_ditch = Pose2DEstimate(label="ditch", message="not enough lidar hits")
        return LidarPerceptionResult(invalid_box, invalid_ditch, int(points_w.shape[0]), 0)

    points_l = world_points_to_lidar_frame(points_w, pos_w, lidar_quat_w)
    ground_plane, ground_inliers = fit_ground_plane(points_l)
    box = estimate_box_pose(points_l, ground_plane, prior.box)
    ditch = estimate_ditch_pose(points_l, ground_plane, prior.ditch)
    return LidarPerceptionResult(box=box, ditch=ditch, num_points=int(points_l.shape[0]), num_ground_inliers=int(ground_inliers.sum()))


def format_pose_estimate(estimate: Pose2DEstimate) -> str:
    if not estimate.valid:
        return f"{estimate.label}=invalid({estimate.message})"
    yaw_deg = math.degrees(estimate.yaw)
    return (
        f"{estimate.label}=x:{estimate.x:+.2f} y:{estimate.y:+.2f} "
        f"yaw:{yaw_deg:+.1f}deg conf:{estimate.confidence:.2f} "
        f"pts:{estimate.num_points} ext:({estimate.extent_x:.2f},{estimate.extent_y:.2f})"
    )


def build_task_d_lidar_prior(env_cfg: Any | None) -> LidarPerceptionPrior:
    box_prior = BoxPrior()
    ditch_prior = DitchPrior()
    if env_cfg is None:
        return LidarPerceptionPrior(box=box_prior, ditch=ditch_prior)

    box_cfg = getattr(getattr(env_cfg, "scene", None), "box", None)
    box_size = getattr(getattr(box_cfg, "spawn", None), "size", None)
    if box_size is not None and len(box_size) >= 3:
        xy = sorted([float(box_size[0]), float(box_size[1])], reverse=True)
        box_prior = BoxPrior(length=xy[0], width=xy[1], height=float(box_size[2]))

    pit_width_range = getattr(env_cfg, "pit_width_range", None)
    if pit_width_range is not None and len(pit_width_range) >= 2:
        width = 0.5 * (float(pit_width_range[0]) + float(pit_width_range[1]))
    else:
        width = ditch_prior.width

    terrain_cfg = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    terrain_generator = getattr(terrain_cfg, "terrain_generator", None)
    size = getattr(terrain_generator, "size", None)
    sub_terrains = getattr(terrain_generator, "sub_terrains", {}) or {}
    pit_cfg = sub_terrains.get("pit_and_platform") if hasattr(sub_terrains, "get") else None
    length = ditch_prior.length
    depth = ditch_prior.depth
    if size is not None and len(size) >= 2:
        border_width = float(getattr(pit_cfg, "border_width", 1.0)) if pit_cfg is not None else 1.0
        length = max(0.1, float(size[1]) - border_width)
    if pit_cfg is not None:
        depth = float(getattr(pit_cfg, "pit_depth", depth))

    ditch_prior = DitchPrior(length=length, width=width, depth=depth)
    return LidarPerceptionPrior(box=box_prior, ditch=ditch_prior)
