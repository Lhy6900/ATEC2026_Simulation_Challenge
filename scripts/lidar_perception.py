from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import Any

import numpy as np

BOX_LIDAR_Y_BIAS_M = 0.055

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


@dataclass
class _TrackedWorldPose:
    xyz_w: np.ndarray
    yaw_w: float
    estimate: Pose2DEstimate


class LidarPoseStabilizer:
    """Temporal stabilizer for slowly moving LiDAR-relative object poses."""

    def __init__(
        self,
        init_samples: int = 5,
        max_step_xy: float = 0.8,
        max_yaw_step: float = math.radians(20.0),
        max_anchor_yaw_error: float = math.radians(15.0),
        ditch_world_z: float = -0.5,
    ):
        self.init_samples = max(1, int(init_samples))
        self.max_step_xy = float(max_step_xy)
        self.max_yaw_step = float(max_yaw_step)
        self.max_anchor_yaw_error = float(max_anchor_yaw_error)
        self.ditch_world_z = float(ditch_world_z)
        self._box_init: list[Pose2DEstimate] = []
        self._box: Pose2DEstimate | None = None
        self._box_yaw_anchor: float | None = None
        self._ditch: Pose2DEstimate | None = None
        self._ditch_yaw_anchor: float | None = None
        self._ditch_init: list[Pose2DEstimate] = []
        self._box_world: _TrackedWorldPose | None = None
        self._ditch_world: _TrackedWorldPose | None = None
        self._box_init_world: list[_TrackedWorldPose] = []
        self._ditch_init_world: list[_TrackedWorldPose] = []
        self._box_last_output_yaw: float | None = None
        self._ditch_last_output_yaw: float | None = None
        self._box_last_output: Pose2DEstimate | None = None
        self._ditch_last_output: Pose2DEstimate | None = None
        self._box_world_anchor_xyz: np.ndarray | None = None
        self._ditch_world_anchor_xyz: np.ndarray | None = None

    def update(
        self,
        result: LidarPerceptionResult,
        sensor_pos_w: Any | None = None,
        sensor_quat_w: Any | None = None,
    ) -> LidarPerceptionResult:
        sensor_pose = sensor_pose_numpy(sensor_pos_w, sensor_quat_w)
        if sensor_pose is not None:
            sensor_pos, sensor_quat = sensor_pose
            box = self._update_box_world(result.box, sensor_pos, sensor_quat)
            ditch = self._update_ditch_world(result.ditch, sensor_pos, sensor_quat)
            return LidarPerceptionResult(
                box=box,
                ditch=ditch,
                num_points=result.num_points,
                num_ground_inliers=result.num_ground_inliers,
            )

        box = self._update_box(result.box)
        ditch = self._update_ditch(result.ditch)
        return LidarPerceptionResult(
            box=box,
            ditch=ditch,
            num_points=result.num_points,
            num_ground_inliers=result.num_ground_inliers,
        )

    def _update_box_world(
        self,
        estimate: Pose2DEstimate,
        sensor_pos: np.ndarray,
        sensor_quat: np.ndarray,
    ) -> Pose2DEstimate:
        if not estimate.valid:
            return self._predict_from_world(self._box_world, sensor_pos, sensor_quat, "box", "predicted_invalid")

        measurement = estimate_to_world_track(estimate, sensor_pos, sensor_quat)
        swapped_axis = False
        if self._box_world is not None:
            measurement.yaw_w, swapped_axis = self._closest_box_world_yaw(
                self._box_world.yaw_w,
                measurement.yaw_w,
                return_axis_swap=True,
            )
        elif self._box_yaw_anchor is None:
            measurement.yaw_w, swapped_axis = self._closest_box_world_yaw(
                math.pi / 2.0,
                measurement.yaw_w,
                return_axis_swap=True,
            )

        if len(self._box_init_world) < self.init_samples:
            if swapped_axis:
                if self._box_world is not None:
                    return self._track_to_lidar_estimate(
                        self._box_world,
                        sensor_pos,
                        sensor_quat,
                        "measured_init_axis_swap",
                        confidence_scale=0.85,
                    )
                return Pose2DEstimate(label="box", message="init_axis_swap")
            self._box_init_world.append(measurement)
            if len(self._box_init_world) >= self.init_samples:
                self._box_world = self._world_median_track(self._box_init_world, estimate, "stabilized_init")
                self._box_yaw_anchor = self._box_world.yaw_w
                self._box_world_anchor_xyz = self._box_world.xyz_w.copy()
                return self._track_to_lidar_estimate(self._box_world, sensor_pos, sensor_quat, "stabilized_init")
            self._box_world = measurement
            return self._track_to_lidar_estimate(measurement, sensor_pos, sensor_quat, "measured_init")

        if self._box_world is None:
            self._box_world = measurement
            return self._track_to_lidar_estimate(measurement, sensor_pos, sensor_quat, "measured")

        prev = self._box_world
        pos_delta = float(np.linalg.norm(measurement.xyz_w[:2] - prev.xyz_w[:2]))
        anchor_xyz = self._box_world_anchor_xyz if self._box_world_anchor_xyz is not None else prev.xyz_w
        anchor_delta_xy = measurement.xyz_w[:2] - anchor_xyz[:2]
        yaw_reference = self._box_yaw_anchor if self._box_yaw_anchor is not None else prev.yaw_w
        yaw_delta_anchor = axis_yaw_error(measurement.yaw_w, yaw_reference)
        yaw_delta_prev = axis_yaw_error(measurement.yaw_w, prev.yaw_w)
        anchor_ok = abs(float(anchor_delta_xy[0])) <= 0.18 and abs(float(anchor_delta_xy[1])) <= 0.09
        pos_ok = pos_delta <= 0.75 and anchor_ok and not swapped_axis
        yaw_ok = yaw_delta_anchor <= self.max_anchor_yaw_error and yaw_delta_prev <= self.max_yaw_step

        # A bad box yaw normally means the visible patch was explained with the
        # wrong rectangle axis; the inferred center is then biased as well.
        if pos_ok and yaw_ok:
            pos_trust = 0.20 + 0.50 * max(0.0, min(1.0, estimate.confidence))
            xyz_w = prev.xyz_w + pos_trust * (measurement.xyz_w - prev.xyz_w)
        else:
            xyz_w = prev.xyz_w.copy()

        if yaw_ok:
            yaw_trust = min(0.10, 0.02 + 0.08 * max(0.0, min(1.0, estimate.confidence)))
            yaw_w = blend_continuous_axis_yaw(prev.yaw_w, measurement.yaw_w, yaw_trust)
        else:
            # Do not freeze to a bad single-frame yaw; stay on the world-frame track.
            yaw_w = prev.yaw_w
        if self._box_yaw_anchor is not None:
            yaw_w = blend_continuous_axis_yaw(yaw_w, self._box_yaw_anchor, 0.01)

        message = "stabilized" if pos_ok and yaw_ok else "predicted_outlier"
        confidence = estimate.confidence if pos_ok and yaw_ok else min(estimate.confidence, prev.estimate.confidence * 0.90)
        self._box_world = _TrackedWorldPose(
            xyz_w=xyz_w.astype(np.float32, copy=False),
            yaw_w=float(yaw_w),
            estimate=replace(estimate, confidence=confidence, message=message),
        )
        return self._track_to_lidar_estimate(self._box_world, sensor_pos, sensor_quat, message)

    def _update_ditch_world(
        self,
        estimate: Pose2DEstimate,
        sensor_pos: np.ndarray,
        sensor_quat: np.ndarray,
    ) -> Pose2DEstimate:
        if not estimate.valid:
            return self._predict_from_world(self._ditch_world, sensor_pos, sensor_quat, "ditch", "predicted_invalid")

        estimate = replace(estimate, yaw=canonical_axis_yaw_for_display(estimate.yaw))
        measurement = estimate_to_world_track_with_world_z(estimate, sensor_pos, sensor_quat, self.ditch_world_z)
        if self._ditch_world is not None:
            measurement.yaw_w = align_axis_yaw_to_reference(measurement.yaw_w, self._ditch_world.yaw_w)

        if len(self._ditch_init_world) < self.init_samples:
            self._ditch_init_world.append(measurement)
            provisional = self._ditch_world_cluster_track(
                self._ditch_init_world,
                estimate,
                "stabilizing_init" if len(self._ditch_init_world) > 1 else "measured_init",
            )
            if len(self._ditch_init_world) >= self.init_samples:
                self._ditch_world = self._ditch_world_cluster_track(
                    self._ditch_init_world,
                    estimate,
                    "stabilized_init",
                )
                self._ditch_yaw_anchor = self._ditch_world.yaw_w
                self._ditch_world_anchor_xyz = self._ditch_world.xyz_w.copy()
                return self._track_to_lidar_estimate(self._ditch_world, sensor_pos, sensor_quat, "stabilized_init")
            self._ditch_world = provisional
            return self._track_to_lidar_estimate(provisional, sensor_pos, sensor_quat, provisional.estimate.message)

        if self._ditch_world is None:
            self._ditch_world = measurement
            return self._track_to_lidar_estimate(measurement, sensor_pos, sensor_quat, "measured")

        prev = self._ditch_world
        delta_xy = measurement.xyz_w[:2] - prev.xyz_w[:2]
        anchor_xyz = self._ditch_world_anchor_xyz if self._ditch_world_anchor_xyz is not None else prev.xyz_w
        anchor_delta_xy = measurement.xyz_w[:2] - anchor_xyz[:2]
        prev_ok = abs(float(delta_xy[0])) <= 0.45 and abs(float(delta_xy[1])) <= 0.24
        anchor_ok = abs(float(anchor_delta_xy[0])) <= 0.28 and abs(float(anchor_delta_xy[1])) <= 0.12
        pos_ok = prev_ok and anchor_ok
        yaw_delta = axis_yaw_error(measurement.yaw_w, prev.yaw_w)

        if pos_ok:
            pos_trust = 0.04 + 0.10 * max(0.0, min(1.0, estimate.confidence))
            xyz_w = prev.xyz_w + pos_trust * (measurement.xyz_w - prev.xyz_w)
        else:
            xyz_w = prev.xyz_w.copy()

        if yaw_delta <= math.radians(25.0):
            yaw_trust = min(0.05, 0.01 + 0.04 * max(0.0, min(1.0, estimate.confidence)))
            yaw_w = blend_continuous_axis_yaw(prev.yaw_w, measurement.yaw_w, yaw_trust)
        else:
            yaw_w = prev.yaw_w

        message = "stabilized" if pos_ok else "predicted_xy_outlier"
        confidence = estimate.confidence if pos_ok else min(estimate.confidence, 0.60)
        self._ditch_world = _TrackedWorldPose(
            xyz_w=xyz_w.astype(np.float32, copy=False),
            yaw_w=float(yaw_w),
            estimate=replace(estimate, confidence=confidence, message=message),
        )
        return self._track_to_lidar_estimate(self._ditch_world, sensor_pos, sensor_quat, message)

    def _predict_from_world(
        self,
        track: _TrackedWorldPose | None,
        sensor_pos: np.ndarray,
        sensor_quat: np.ndarray,
        label: str,
        message: str,
    ) -> Pose2DEstimate:
        if track is None:
            return Pose2DEstimate(label=label, message="not initialized")
        return self._track_to_lidar_estimate(track, sensor_pos, sensor_quat, message, confidence_scale=0.92)

    def _track_to_lidar_estimate(
        self,
        track: _TrackedWorldPose,
        sensor_pos: np.ndarray,
        sensor_quat: np.ndarray,
        message: str,
        confidence_scale: float = 1.0,
    ) -> Pose2DEstimate:
        rel_xyz = quat_inverse_apply(sensor_quat, (track.xyz_w - sensor_pos).reshape(1, 3))[0]
        if track.estimate.label == "box":
            rel_xyz[1] += BOX_LIDAR_Y_BIAS_M
        yaw_l = track.yaw_w - yaw_from_quat_wxyz(sensor_quat)
        if track.estimate.label == "box":
            reference_yaw = self._box_last_output.yaw if self._box_last_output is not None else self._box_last_output_yaw
        elif track.estimate.label == "ditch":
            reference_yaw = self._ditch_last_output.yaw if self._ditch_last_output is not None else self._ditch_last_output_yaw
        else:
            reference_yaw = None
        if reference_yaw is None:
            yaw_l = canonical_axis_yaw_for_display(yaw_l)
        else:
            yaw_l = align_axis_yaw_to_reference(yaw_l, reference_yaw)
        estimate = replace(
            track.estimate,
            valid=True,
            x=float(rel_xyz[0]),
            y=float(rel_xyz[1]),
            yaw=float(yaw_l),
            confidence=max(0.0, min(1.0, track.estimate.confidence * confidence_scale)),
            message=message,
        )
        return self._limit_lidar_output_step(estimate)

    def _limit_lidar_output_step(self, estimate: Pose2DEstimate) -> Pose2DEstimate:
        if estimate.label == "box":
            prev = self._box_last_output
        elif estimate.label == "ditch":
            prev = self._ditch_last_output
        else:
            prev = None

        if prev is None or not prev.valid:
            self._store_lidar_output(estimate)
            return estimate

        max_xy_step = 0.25
        max_yaw_step = math.radians(7.0)
        dx = max(-max_xy_step, min(max_xy_step, float(estimate.x) - float(prev.x)))
        dy = max(-max_xy_step, min(max_xy_step, float(estimate.y) - float(prev.y)))
        target_yaw = align_axis_yaw_to_reference(estimate.yaw, prev.yaw)
        yaw_delta = max(-max_yaw_step, min(max_yaw_step, target_yaw - float(prev.yaw)))
        limited = replace(
            estimate,
            x=float(prev.x) + dx,
            y=float(prev.y) + dy,
            yaw=float(prev.yaw) + yaw_delta,
        )
        self._store_lidar_output(limited)
        return limited

    def _store_lidar_output(self, estimate: Pose2DEstimate) -> None:
        if estimate.label == "box":
            self._box_last_output = estimate
            self._box_last_output_yaw = float(estimate.yaw)
        elif estimate.label == "ditch":
            self._ditch_last_output = estimate
            self._ditch_last_output_yaw = float(estimate.yaw)

    def _world_median_track(
        self,
        samples: list[_TrackedWorldPose],
        template: Pose2DEstimate,
        message: str,
    ) -> _TrackedWorldPose:
        xyz_w = np.median(np.stack([sample.xyz_w for sample in samples], axis=0), axis=0).astype(np.float32)
        yaw_w, _ = dominant_axis_yaw([sample.yaw_w for sample in samples])
        confidence = float(np.mean([sample.estimate.confidence for sample in samples]))
        num_points = int(np.median([sample.estimate.num_points for sample in samples]))
        estimate = replace(template, confidence=max(template.confidence, confidence), num_points=num_points, message=message)
        return _TrackedWorldPose(xyz_w=xyz_w, yaw_w=float(yaw_w), estimate=estimate)

    def _ditch_world_cluster_track(
        self,
        samples: list[_TrackedWorldPose],
        template: Pose2DEstimate,
        message: str,
    ) -> _TrackedWorldPose:
        cluster = self._select_ditch_world_cluster(samples)
        return self._world_median_track(cluster, template, message)

    def _select_ditch_world_cluster(self, samples: list[_TrackedWorldPose]) -> list[_TrackedWorldPose]:
        if len(samples) <= 1:
            return samples

        first = samples[0]
        best_cluster: list[_TrackedWorldPose] = []
        best_key: tuple[int, int, float, float] | None = None
        for seed in samples:
            cluster = []
            for sample in samples:
                delta = sample.xyz_w[:2] - seed.xyz_w[:2]
                if abs(float(delta[0])) > 0.35 or abs(float(delta[1])) > 0.18:
                    continue
                if axis_yaw_error(sample.yaw_w, seed.yaw_w) > math.radians(28.0):
                    continue
                cluster.append(sample)
            if not cluster:
                cluster = [seed]

            cluster_y = np.array([sample.xyz_w[1] for sample in cluster], dtype=np.float32)
            spread_y = float(np.max(cluster_y) - np.min(cluster_y)) if cluster_y.size > 1 else 0.0
            contains_first = 1 if len(cluster) >= 2 and any(sample is first for sample in cluster) else 0
            mean_conf = float(np.mean([sample.estimate.confidence for sample in cluster]))
            key = (contains_first, len(cluster), mean_conf, -spread_y)
            if best_key is None or key > best_key:
                best_key = key
                best_cluster = cluster

        return best_cluster if best_cluster else [first]

    def _closest_box_world_yaw(
        self,
        reference_yaw: float,
        measured_yaw: float,
        return_axis_swap: bool = False,
    ) -> float | tuple[float, bool]:
        candidates = (
            float(measured_yaw),
            float(measured_yaw) + math.pi / 2.0,
            float(measured_yaw) - math.pi / 2.0,
        )
        aligned = [align_axis_yaw_to_reference(yaw, reference_yaw) for yaw in candidates]
        best_index = min(range(len(aligned)), key=lambda idx: abs(aligned[idx] - reference_yaw))
        best = aligned[best_index]
        if return_axis_swap:
            return best, best_index != 0
        return best

    def _update_box(self, estimate: Pose2DEstimate) -> Pose2DEstimate:
        if not estimate.valid:
            return self._box if self._box is not None else estimate

        if self._box is not None:
            estimate = replace(estimate, yaw=self._closest_box_yaw(self._box.yaw, estimate.yaw))
        else:
            estimate = replace(estimate, yaw=canonical_axis_yaw_for_display(estimate.yaw))

        if len(self._box_init) < self.init_samples:
            self._box_init.append(estimate)
            if len(self._box_init) < self.init_samples:
                self._box = self._provisional_box_estimate(estimate)
                return self._box
            anchor_yaw, _ = dominant_axis_yaw([item.yaw for item in self._box_init])
            anchor_yaw = canonical_axis_yaw_for_display(anchor_yaw)
            self._box_yaw_anchor = anchor_yaw
            anchor_xy = np.median(np.array([[item.x, item.y] for item in self._box_init], dtype=np.float32), axis=0)
            anchor_conf = float(np.mean([item.confidence for item in self._box_init]))
            anchor_pts = int(np.median([item.num_points for item in self._box_init]))
            self._box = replace(
                estimate,
                x=float(anchor_xy[0]),
                y=float(anchor_xy[1]),
                yaw=anchor_yaw,
                confidence=max(estimate.confidence, anchor_conf),
                num_points=anchor_pts,
                message="stabilized_init",
            )
            return self._box

        if self._box is None:
            self._box = estimate
            return estimate

        prev = self._box
        yaw_reference = self._box_yaw_anchor if self._box_yaw_anchor is not None else prev.yaw
        measured_yaw = self._closest_box_yaw(yaw_reference, estimate.yaw)
        measured_yaw = align_axis_yaw_to_reference(measured_yaw, prev.yaw)
        if axis_yaw_error(measured_yaw, yaw_reference) > self.max_anchor_yaw_error:
            measured_yaw = prev.yaw
        if abs(measured_yaw - prev.yaw) > self.max_yaw_step:
            measured_yaw = prev.yaw

        measured_xy = np.array([estimate.x, estimate.y], dtype=np.float32)
        prev_xy = np.array([prev.x, prev.y], dtype=np.float32)
        delta_xy = measured_xy - prev_xy
        dist = float(np.linalg.norm(delta_xy))
        if dist > self.max_step_xy:
            measured_xy = prev_xy + delta_xy * (self.max_step_xy / max(dist, 1.0e-6))

        trust = 0.20 + 0.45 * max(0.0, min(1.0, estimate.confidence))
        yaw_trust = min(0.08, 0.015 + 0.07 * max(0.0, min(1.0, estimate.confidence)))
        smoothed_xy = prev_xy + trust * (measured_xy - prev_xy)
        smoothed_yaw = blend_continuous_axis_yaw(prev.yaw, measured_yaw, yaw_trust)
        if self._box_yaw_anchor is not None:
            smoothed_yaw = blend_continuous_axis_yaw(smoothed_yaw, self._box_yaw_anchor, 0.02)
        self._box = replace(
            estimate,
            x=float(smoothed_xy[0]),
            y=float(smoothed_xy[1]),
            yaw=smoothed_yaw,
            confidence=max(prev.confidence * 0.85, estimate.confidence),
            message="stabilized",
        )
        return self._box

    def _closest_box_yaw(self, reference_yaw: float, measured_yaw: float) -> float:
        candidates = (
            float(measured_yaw),
            float(measured_yaw) + math.pi / 2.0,
            float(measured_yaw) - math.pi / 2.0,
        )
        aligned = [align_axis_yaw_to_reference(yaw, reference_yaw) for yaw in candidates]
        return min(aligned, key=lambda yaw: abs(yaw - reference_yaw))

    def _provisional_box_estimate(self, fallback: Pose2DEstimate) -> Pose2DEstimate:
        if len(self._box_init) < 2:
            return fallback
        anchor_yaw, count = dominant_axis_yaw([item.yaw for item in self._box_init])
        anchor_yaw = canonical_axis_yaw_for_display(anchor_yaw)
        if count < 2:
            return self._box if self._box is not None else fallback
        anchor_xy = np.median(np.array([[item.x, item.y] for item in self._box_init], dtype=np.float32), axis=0)
        anchor_conf = float(np.mean([item.confidence for item in self._box_init]))
        return replace(
            fallback,
            x=float(anchor_xy[0]),
            y=float(anchor_xy[1]),
            yaw=anchor_yaw,
            confidence=max(fallback.confidence, anchor_conf),
            message="stabilizing_init",
        )

    def _update_ditch(self, estimate: Pose2DEstimate) -> Pose2DEstimate:
        if not estimate.valid:
            return self._ditch if self._ditch is not None else estimate
        estimate = replace(estimate, yaw=canonical_axis_yaw_for_display(estimate.yaw))
        if self._ditch is None:
            self._ditch_init.append(estimate)
            if len(self._ditch_init) < self.init_samples:
                self._ditch = self._provisional_ditch_estimate(estimate)
                return self._ditch
            anchor_yaw, _ = dominant_axis_yaw([item.yaw for item in self._ditch_init])
            anchor_yaw = canonical_axis_yaw_for_display(anchor_yaw)
            self._ditch_yaw_anchor = anchor_yaw
            anchor_xy = np.median(np.array([[item.x, item.y] for item in self._ditch_init], dtype=np.float32), axis=0)
            anchor_conf = float(np.mean([item.confidence for item in self._ditch_init]))
            anchor_pts = int(np.median([item.num_points for item in self._ditch_init]))
            self._ditch = replace(
                estimate,
                x=float(anchor_xy[0]),
                y=float(anchor_xy[1]),
                yaw=anchor_yaw,
                confidence=max(estimate.confidence, anchor_conf),
                num_points=anchor_pts,
                message="stabilized_init",
            )
            return self._ditch

        prev = self._ditch
        measured_yaw = align_axis_yaw_to_reference(estimate.yaw, prev.yaw)
        if axis_yaw_error(measured_yaw, prev.yaw) > math.radians(25.0):
            estimate = replace(estimate, yaw=prev.yaw, confidence=min(estimate.confidence, 0.65))
        else:
            estimate = replace(estimate, yaw=measured_yaw)

        measured_xy = np.array([estimate.x, estimate.y], dtype=np.float32)
        prev_xy = np.array([prev.x, prev.y], dtype=np.float32)
        delta_xy = measured_xy - prev_xy
        max_dx = 0.45
        max_dy = 0.35
        if abs(float(delta_xy[0])) > max_dx or abs(float(delta_xy[1])) > max_dy:
            measured_xy = prev_xy
            estimate = replace(estimate, confidence=min(estimate.confidence, 0.60), message="held_xy_outlier")
        else:
            trust = 0.20 + 0.35 * max(0.0, min(1.0, estimate.confidence))
            measured_xy = prev_xy + trust * delta_xy

        self._ditch = replace(estimate, x=float(measured_xy[0]), y=float(measured_xy[1]))
        return self._ditch

    def _provisional_ditch_estimate(self, fallback: Pose2DEstimate) -> Pose2DEstimate:
        if len(self._ditch_init) < 2:
            return fallback
        anchor_yaw, count = dominant_axis_yaw([item.yaw for item in self._ditch_init], radius=math.radians(18.0))
        if count < 2:
            return self._ditch if self._ditch is not None else fallback
        anchor_xy = np.median(np.array([[item.x, item.y] for item in self._ditch_init], dtype=np.float32), axis=0)
        anchor_conf = float(np.mean([item.confidence for item in self._ditch_init]))
        return replace(
            fallback,
            x=float(anchor_xy[0]),
            y=float(anchor_xy[1]),
            yaw=canonical_axis_yaw_for_display(anchor_yaw),
            confidence=max(fallback.confidence, anchor_conf),
            message="stabilizing_init",
        )


def wrap_axis_yaw(yaw: float) -> float:
    """Wrap an unoriented rectangle axis to [-pi/2, pi/2)."""
    wrapped = (float(yaw) + math.pi / 2.0) % math.pi - math.pi / 2.0
    return wrapped


def canonical_axis_yaw_for_display(yaw: float) -> float:
    """Choose a stable display branch for an unoriented axis near +/-90 degrees."""
    yaw = wrap_axis_yaw(yaw)
    if yaw < 0.0 and abs(abs(yaw) - math.pi / 2.0) <= math.radians(8.0):
        return yaw + math.pi
    return yaw


def align_axis_yaw_to_reference(yaw: float, reference_yaw: float) -> float:
    """Return the equivalent axis yaw branch closest to a continuous reference."""
    yaw = float(yaw)
    reference_yaw = float(reference_yaw)
    candidates = (yaw, yaw + math.pi, yaw - math.pi)
    return min(candidates, key=lambda candidate: abs(candidate - reference_yaw))


def axis_yaw_error(a: float, b: float) -> float:
    return abs(align_axis_yaw_to_reference(float(a), float(b)) - float(b))


def mean_axis_yaw(yaws: list[float] | np.ndarray) -> float:
    if len(yaws) == 0:
        return 0.0
    values = np.asarray(yaws, dtype=np.float32)
    return wrap_axis_yaw(0.5 * math.atan2(float(np.sin(2.0 * values).mean()), float(np.cos(2.0 * values).mean())))


def blend_axis_yaw(current: float, measured: float, alpha: float) -> float:
    alpha = max(0.0, min(1.0, float(alpha)))
    current = float(current)
    measured = float(measured)
    sin2 = (1.0 - alpha) * math.sin(2.0 * current) + alpha * math.sin(2.0 * measured)
    cos2 = (1.0 - alpha) * math.cos(2.0 * current) + alpha * math.cos(2.0 * measured)
    return wrap_axis_yaw(0.5 * math.atan2(sin2, cos2))


def blend_continuous_axis_yaw(current: float, measured: float, alpha: float) -> float:
    alpha = max(0.0, min(1.0, float(alpha)))
    measured = align_axis_yaw_to_reference(measured, current)
    return float(current) + alpha * (measured - float(current))


def dominant_axis_yaw(yaws: list[float], radius: float = math.radians(15.0)) -> tuple[float, int]:
    if not yaws:
        return 0.0, 0
    best_cluster: list[float] = []
    for yaw in yaws:
        cluster = [candidate for candidate in yaws if axis_yaw_error(candidate, yaw) <= radius]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
    return mean_axis_yaw(best_cluster), len(best_cluster)


def quat_inverse_apply(quat_wxyz: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    inv_quat = np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float32)
    xyz = inv_quat[1:4]
    t = np.cross(xyz.reshape(1, 3), vectors) * 2.0
    return vectors + inv_quat[0] * t + np.cross(xyz.reshape(1, 3), t)


def quat_apply(quat_wxyz: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    vectors = np.asarray(vectors, dtype=np.float32).reshape(-1, 3)
    xyz = quat[1:4]
    t = np.cross(xyz.reshape(1, 3), vectors) * 2.0
    return vectors + quat[0] * t + np.cross(xyz.reshape(1, 3), t)


def yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    qw, qx, qy, qz = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(float(siny_cosp), float(cosy_cosp))


def sensor_pose_numpy(sensor_pos_w: Any | None, sensor_quat_w: Any | None) -> tuple[np.ndarray, np.ndarray] | None:
    if sensor_pos_w is None or sensor_quat_w is None:
        return None
    pos = as_numpy(sensor_pos_w).astype(np.float32, copy=False)
    quat = as_numpy(sensor_quat_w).astype(np.float32, copy=False)
    if pos.ndim == 2:
        pos = pos[0]
    if quat.ndim == 2:
        quat = quat[0]
    if pos.shape[0] < 3 or quat.shape[0] < 4:
        return None
    return pos[:3].copy(), quat[:4].copy()


def estimate_to_world_track(
    estimate: Pose2DEstimate,
    sensor_pos: np.ndarray,
    sensor_quat: np.ndarray,
) -> _TrackedWorldPose:
    local_xyz = np.array([[estimate.x, estimate.y, 0.0]], dtype=np.float32)
    xyz_w = quat_apply(sensor_quat, local_xyz)[0] + sensor_pos
    yaw_w = float(estimate.yaw) + yaw_from_quat_wxyz(sensor_quat)
    return _TrackedWorldPose(
        xyz_w=xyz_w.astype(np.float32, copy=False),
        yaw_w=float(yaw_w),
        estimate=estimate,
    )


def estimate_to_world_track_with_world_z(
    estimate: Pose2DEstimate,
    sensor_pos: np.ndarray,
    sensor_quat: np.ndarray,
    world_z: float,
) -> _TrackedWorldPose:
    rot_x = quat_apply(sensor_quat, np.array([[1.0, 0.0, 0.0]], dtype=np.float32))[0]
    rot_y = quat_apply(sensor_quat, np.array([[0.0, 1.0, 0.0]], dtype=np.float32))[0]
    rot_z = quat_apply(sensor_quat, np.array([[0.0, 0.0, 1.0]], dtype=np.float32))[0]
    numerator = float(world_z) - float(sensor_pos[2]) - float(estimate.x) * float(rot_x[2]) - float(estimate.y) * float(rot_y[2])
    local_z = numerator / max(float(rot_z[2]), 1.0e-6)
    local_xyz = np.array([[estimate.x, estimate.y, local_z]], dtype=np.float32)
    xyz_w = quat_apply(sensor_quat, local_xyz)[0] + sensor_pos
    xyz_w[2] = float(world_z)
    yaw_w = float(estimate.yaw) + yaw_from_quat_wxyz(sensor_quat)
    return _TrackedWorldPose(
        xyz_w=xyz_w.astype(np.float32, copy=False),
        yaw_w=float(yaw_w),
        estimate=estimate,
    )


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
        & (heights < -max(0.18, 0.18 * prior.depth))
    )
    candidate_points = points_l[valid]
    clusters = cluster_xy(candidate_points[:, :2], radius=0.55, min_points=10)

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

    edge_estimate = estimate_ditch_from_height_edges(points_l, heights, prior)
    if edge_estimate.valid:
        edge_score = 0.85 - edge_estimate.confidence - 0.001 * edge_estimate.num_points
        if best is None or edge_score < best[0]:
            best = (edge_score, edge_estimate)

    if best is None:
        return Pose2DEstimate(
            label="ditch",
            message=f"no low strip/edge cluster ({candidate_points.shape[0]} low pts)",
        )
    return best[1]


def estimate_ditch_from_height_edges(points_l: np.ndarray, heights: np.ndarray, prior: DitchPrior) -> Pose2DEstimate:
    """Detect a pit from the visible ground-height discontinuity at its front edge."""
    valid = np.all(np.isfinite(points_l), axis=1) & np.isfinite(heights) & (points_l[:, 0] > -0.5)
    candidates = points_l[valid]
    candidate_heights = heights[valid]
    if candidates.shape[0] < 32:
        return Pose2DEstimate(label="ditch", message="not enough edge candidates")

    bin_size = 0.18
    xy = candidates[:, :2]
    bins = np.floor(xy / bin_size).astype(np.int32)
    cell_min: dict[tuple[int, int], float] = {}
    cell_center: dict[tuple[int, int], np.ndarray] = {}
    for idx, cell in enumerate(bins):
        key = (int(cell[0]), int(cell[1]))
        height = float(candidate_heights[idx])
        if key not in cell_min or height < cell_min[key]:
            cell_min[key] = height
            cell_center[key] = (np.asarray(key, dtype=np.float32) + 0.5) * bin_size

    edge_points: list[tuple[np.ndarray, float]] = []
    for (cx, cy), height in cell_min.items():
        if height > -0.12:
            continue
        front_heights = []
        back_heights = []
        for dx in (-2, -1):
            neighbor = (cx + dx, cy)
            if neighbor in cell_min:
                front_heights.append(cell_min[neighbor])
        for dx in (1, 2):
            neighbor = (cx + dx, cy)
            if neighbor in cell_min:
                back_heights.append(cell_min[neighbor])
        if front_heights and np.median(front_heights) > -0.08:
            edge_points.append((cell_center[(cx, cy)], 1.0))
        if back_heights and np.median(back_heights) > -0.08:
            edge_points.append((cell_center[(cx, cy)], -1.0))

    if len(edge_points) < 6:
        return Pose2DEstimate(label="ditch", message=f"not enough edge points ({len(edge_points)})")

    low_xy = candidates[candidate_heights < -0.12, :2]
    best: tuple[float, Pose2DEstimate] | None = None
    for edge_sign in (1.0, -1.0):
        side_points = [xy for xy, sign in edge_points if sign == edge_sign]
        if len(side_points) < 6:
            continue
        edge_xy = np.asarray(side_points, dtype=np.float32)
        clusters = cluster_xy(edge_xy, radius=0.42, min_points=6)
        for cluster_idx in clusters:
            cluster = edge_xy[cluster_idx]
            if cluster.shape[0] < 6:
                continue
            yaw = pca_yaw(cluster)
            u, v, proj_u, proj_v = oriented_bounds(cluster, yaw)
            length_visible = float(np.max(proj_u) - np.min(proj_u))
            center = _ditch_center_from_low_extent_and_front_edge(cluster, low_xy, yaw, prior, edge_sign)
            if length_visible < 0.45:
                continue
            front_bonus = max(0.0, min(1.0, center[0] / 3.0))
            density_bonus = min(1.0, cluster.shape[0] / 40.0)
            edge_bonus = 0.05 if edge_sign > 0.0 else 0.0
            confidence = max(0.0, min(1.0, 0.35 + 0.30 * density_bonus + 0.25 * front_bonus + edge_bonus))
            score = -confidence - 0.001 * cluster.shape[0]
            estimate = Pose2DEstimate(
                label="ditch",
                valid=True,
                x=float(center[0]),
                y=float(center[1]),
                yaw=wrap_axis_yaw(yaw),
                confidence=confidence,
                num_points=int(cluster.shape[0]),
                extent_x=length_visible,
                extent_y=float(prior.width),
                message="edge_front" if edge_sign > 0.0 else "edge_back",
            )
            if best is None or score < best[0]:
                best = (score, estimate)

    if best is None:
        return Pose2DEstimate(label="ditch", message=f"no usable edge cluster ({len(edge_points)} edge pts)")
    return best[1]


def _ditch_center_from_low_extent_and_front_edge(
    edge_cluster_xy: np.ndarray,
    low_xy: np.ndarray,
    yaw: float,
    prior: DitchPrior,
    edge_sign: float,
) -> np.ndarray:
    u, v, edge_u, edge_v = oriented_bounds(edge_cluster_xy, yaw)
    width_axis = v if v[0] >= 0.0 else -v

    edge_width = edge_cluster_xy @ width_axis
    if low_xy.shape[0] >= 4:
        low_u = low_xy @ u
        low_extent = float(np.max(low_u) - np.min(low_u))
        if low_extent >= 0.45:
            if low_extent >= 0.65 * prior.length:
                low_min, low_max = np.percentile(low_u, [5.0, 95.0])
                center_u = 0.5 * (float(low_min) + float(low_max))
            else:
                low_min = float(np.min(low_u))
                low_max = float(np.max(low_u))
                edge_center_u = float(np.median(edge_u))
                if edge_center_u - low_min < low_max - edge_center_u:
                    center_u = low_min + 0.5 * prior.length
                else:
                    center_u = low_max - 0.5 * prior.length
            low_width = low_xy @ width_axis
            low_width_extent = float(np.max(low_width) - np.min(low_width))
            if low_width_extent >= 0.35 * prior.width:
                front_edge_width = float(np.percentile(low_width, 98.0))
                center_width = front_edge_width - 0.5 * prior.width
            else:
                center_width = float(np.median(edge_width)) + float(edge_sign) * 0.5 * prior.width
            return (center_u * u + center_width * width_axis).astype(np.float32, copy=False)

    center_u = float(np.median(edge_u))
    center_width = float(np.median(edge_width)) + float(edge_sign) * 0.5 * prior.width
    return (center_u * u + center_width * width_axis).astype(np.float32, copy=False)


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

    terrain_cfg = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    terrain_generator = getattr(terrain_cfg, "terrain_generator", None)
    pit_width_range = getattr(env_cfg, "pit_width_range", None)
    difficulty = _single_cell_terrain_difficulty(terrain_generator)
    if pit_width_range is not None and len(pit_width_range) >= 2:
        width_min, width_max = float(pit_width_range[0]), float(pit_width_range[1])
        if difficulty is None:
            width = 0.5 * (width_min + width_max)
        else:
            width = width_min + difficulty * (width_max - width_min)
    else:
        width = ditch_prior.width

    size = getattr(terrain_generator, "size", None)
    sub_terrains = getattr(terrain_generator, "sub_terrains", {}) or {}
    pit_cfg = sub_terrains.get("pit_and_platform") if hasattr(sub_terrains, "get") else None
    length = ditch_prior.length
    depth = ditch_prior.depth
    if size is not None and len(size) >= 2:
        border_width = float(getattr(pit_cfg, "border_width", 1.0)) if pit_cfg is not None else 1.0
        platform_extent_y = 0.5 * float(size[1]) - border_width
        platform_min_y = 0.75 * float(size[1]) - 0.5 * platform_extent_y
        pit_min_y = 0.5 * border_width
        length = max(0.1, platform_min_y - pit_min_y)
    if pit_cfg is not None:
        depth = float(getattr(pit_cfg, "pit_depth", depth))

    ditch_prior = DitchPrior(length=length, width=width, depth=depth)
    return LidarPerceptionPrior(box=box_prior, ditch=ditch_prior)


def _single_cell_terrain_difficulty(terrain_generator: Any | None) -> float | None:
    if terrain_generator is None:
        return None
    if int(getattr(terrain_generator, "num_rows", 1)) != 1 or int(getattr(terrain_generator, "num_cols", 1)) != 1:
        return None
    difficulty_range = getattr(terrain_generator, "difficulty_range", (0.0, 1.0))
    if difficulty_range is None or len(difficulty_range) < 2:
        return None
    lower, upper = float(difficulty_range[0]), float(difficulty_range[1])
    seed = getattr(terrain_generator, "seed", None)
    if seed is None:
        return None
    rng = np.random.default_rng(int(seed))
    sub_terrains = getattr(terrain_generator, "sub_terrains", {}) or {}
    if len(sub_terrains) > 0:
        proportions = np.array([getattr(sub_cfg, "proportion", 1.0) for sub_cfg in sub_terrains.values()], dtype=np.float64)
        proportions /= np.sum(proportions)
        rng.choice(len(proportions), p=proportions)
    return float(rng.uniform(lower, upper))
