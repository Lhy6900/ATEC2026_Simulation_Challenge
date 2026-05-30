# Created by skywoodsz on 2026/02/07.

import argparse
import csv
import math
import os
import time
import json
import sys
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play Atec Tasks (ENV only, no RL).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Enable debug prints for per-step reward/time metrics.",
)
parser.add_argument(
    "--debug_interval",
    type=int,
    default=30,
    help="Print debug reward/policy snapshots every N simulation steps when --debug is set.",
)
parser.add_argument(
    "--keyboard",
    action="store_true",
    default=False,
    help="Enable keyboard teleop (WASD=move, QE=yaw, ZX=height, R=reset).",
)
parser.add_argument(
    "--policy",
    type=str,
    default="gr00t",
    choices=["gr00t", "blind"],
    help="Locomotion policy: gr00t (GR00T WBC) or blind (policy.pt). Default: gr00t.",
)
parser.add_argument(
    "--sensor_vis",
    action="store_true",
    default=False,
    help="Show in-scene red marker points for Task D head depth and LiDAR hits.",
)
parser.add_argument(
    "--sensor_vis_interval",
    type=int,
    default=10,
    help="Print sensor range statistics every N simulation steps when --sensor_vis is set.",
)
parser.add_argument(
    "--depth_vis_max",
    type=float,
    default=10.0,
    help="Maximum depth in meters used for in-scene depth markers.",
)
parser.add_argument(
    "--keep_gpu_sim",
    action="store_true",
    default=False,
    help="Keep CUDA PhysX even when GUI camera visualization would require CPU PhysX for moving USD visuals.",
)
parser.add_argument(
    "--lidar_pose_log",
    type=str,
    default=os.path.join("logs", "lidar_perception", "taskd_lidar_pose.csv"),
    help="CSV file for Task D LiDAR-only box/ditch poses. Set to an empty string to disable.",
)
parser.add_argument(
    "--lidar_pose_log_interval",
    type=float,
    default=2.0,
    help="Write box/ditch LiDAR-relative pose estimates every N wall-clock seconds.",
)
parser.add_argument(
    "--lidar_pose_gt_log",
    type=str,
    default="",
    help="Optional dev CSV with LiDAR estimates and config-derived ground truth for Task D.",
)

# Isaac Sim / Kit args
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# If recording video or sensor views, need cameras enabled in IsaacLab/Kit.
if args_cli.video or args_cli.sensor_vis:
    args_cli.enable_cameras = True

# Fabric pose reads are broken in this local Isaac Sim / IsaacLab combination
# (missing usdrt.hierarchy), so keep camera runs on the USD backend.
if args_cli.enable_cameras and not args_cli.disable_fabric:
    print("[INFO] Cameras enabled: forcing --disable_fabric and enabling PhysX USD pose writeback.")
    args_cli.disable_fabric = True

# IsaacLab/Isaac Sim 4.5 cannot render live physics updates from GPU PhysX when
# Fabric is disabled. Fabric is disabled here because this local stack hits the
# Warp/usdrt Fabric failure, so use CPU PhysX for GUI/video camera debugging.
needs_live_usd_visuals = (not getattr(args_cli, "headless", False)) or args_cli.video
if (
    args_cli.enable_cameras
    and args_cli.disable_fabric
    and needs_live_usd_visuals
    and str(args_cli.device).startswith("cuda")
    and not args_cli.keep_gpu_sim
):
    print("[INFO] GUI/video camera run with --disable_fabric: using CPU PhysX so robot USD visuals update.")
    args_cli.device = "cpu"

# -----------------------------------------------------------------------------
# Launch Isaac Sim / Kit
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports AFTER simulation_app is created (IsaacLab pattern)
# -----------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    combine_frame_transforms,
    convert_camera_frame_orientation_convention,
    convert_quat,
)

import atec_rl_lab.tasks  # noqa: F401, E402 (register your tasks)
from isaaclab_tasks.utils import parse_env_cfg
from rl_utils import camera_follow
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec
from keyboard_teleop import KeyboardTeleop
from lidar_perception import (
    align_axis_yaw_to_reference,
    build_task_d_lidar_prior,
    estimate_task_d_poses_from_lidar,
    LidarPoseStabilizer,
    quat_inverse_apply,
    yaw_from_quat_wxyz,
)
from sensor_vis_utils import (
    depth_to_world_points,
    finite_stats,
    format_stats,
    lidar_height_scan_to_world_points,
    lidar_hits_to_world_points,
    point_cloud_xy_report,
)

from demo.solution import set_policy, AlgSolution
set_policy(args_cli.policy)
solution = AlgSolution()


class LidarPoseLogger:
    """Write LiDAR-relative Task D box/ditch pose estimates to a compact CSV log."""

    _FIELDS = (
        "wall_time_s",
        "sim_time_s",
        "step",
        "box_valid",
        "box_x_m",
        "box_y_m",
        "box_yaw_deg",
        "box_conf",
        "ditch_valid",
        "ditch_x_m",
        "ditch_y_m",
        "ditch_yaw_deg",
        "ditch_conf",
    )

    def __init__(self, path: str | None, interval_s: float):
        self.path = os.path.abspath(path) if path else ""
        self.interval_s = max(0.0, float(interval_s))
        self._file = None
        self._writer = None
        self._last_log_wall_time = None
        self._start_wall_time = time.time()
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._FIELDS)
        self._writer.writeheader()
        self._file.flush()
        print(f"[lidar_pose_log] Writing box/ditch poses every {self.interval_s:.1f}s to {self.path}")

    def maybe_write(self, result, step: int, sim_time_s: float | None = None) -> None:
        if self._writer is None or result is None:
            return
        wall_time_s = time.time() - self._start_wall_time
        if self._last_log_wall_time is not None and wall_time_s - self._last_log_wall_time < self.interval_s:
            return
        self._last_log_wall_time = wall_time_s
        row = {
            "wall_time_s": f"{wall_time_s:.3f}",
            "sim_time_s": f"{float(sim_time_s):.3f}" if sim_time_s is not None else "",
            "step": int(step),
        }
        row.update(self._pose_fields("box", result.box))
        row.update(self._pose_fields("ditch", result.ditch))
        self._writer.writerow(row)
        self._file.flush()

    def _pose_fields(self, prefix: str, estimate) -> dict[str, str | int]:
        if estimate is None or not estimate.valid:
            return {
                f"{prefix}_valid": 0,
                f"{prefix}_x_m": "",
                f"{prefix}_y_m": "",
                f"{prefix}_yaw_deg": "",
                f"{prefix}_conf": "0.000",
            }
        return {
            f"{prefix}_valid": 1,
            f"{prefix}_x_m": f"{estimate.x:.4f}",
            f"{prefix}_y_m": f"{estimate.y:.4f}",
            f"{prefix}_yaw_deg": f"{math.degrees(estimate.yaw):.3f}",
            f"{prefix}_conf": f"{estimate.confidence:.3f}",
        }

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


class LidarPoseGroundTruthLogger:
    """Write LiDAR-frame estimates against Task D ground truth for offline tuning."""

    _FIELDS = (
        "wall_time_s",
        "sim_time_s",
        "step",
        "box_valid",
        "box_x_m",
        "box_y_m",
        "box_yaw_deg",
        "box_gt_x_m",
        "box_gt_y_m",
        "box_gt_yaw_deg",
        "box_err_xy_m",
        "box_err_yaw_deg",
        "ditch_valid",
        "ditch_x_m",
        "ditch_y_m",
        "ditch_yaw_deg",
        "ditch_gt_x_m",
        "ditch_gt_y_m",
        "ditch_gt_yaw_deg",
        "ditch_err_xy_m",
        "ditch_err_yaw_deg",
    )

    def __init__(self, path: str | None, interval_s: float, env, lidar_prior):
        self.path = os.path.abspath(path) if path else ""
        self.interval_s = max(0.0, float(interval_s))
        self.env = env
        self.lidar_prior = lidar_prior
        self._file = None
        self._writer = None
        self._last_log_wall_time = None
        self._start_wall_time = time.time()
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._FIELDS)
        self._writer.writeheader()
        self._file.flush()
        print(f"[lidar_pose_gt_log] Writing estimate/ground-truth poses every {self.interval_s:.1f}s to {self.path}")

    def maybe_write(self, result, step: int, sim_time_s: float | None = None) -> None:
        if self._writer is None or result is None:
            return
        wall_time_s = time.time() - self._start_wall_time
        if self._last_log_wall_time is not None and wall_time_s - self._last_log_wall_time < self.interval_s:
            return
        gt = self._get_ground_truth()
        if gt is None:
            return
        self._last_log_wall_time = wall_time_s
        row = {
            "wall_time_s": f"{wall_time_s:.3f}",
            "sim_time_s": f"{float(sim_time_s):.3f}" if sim_time_s is not None else "",
            "step": int(step),
        }
        row.update(self._pose_gt_fields("box", result.box, gt["box"]))
        row.update(self._pose_gt_fields("ditch", result.ditch, gt["ditch"]))
        self._writer.writerow(row)
        self._file.flush()

    def _get_ground_truth(self):
        scene = getattr(getattr(self.env, "unwrapped", self.env), "scene", None)
        if scene is None:
            return None
        sensors = getattr(scene, "sensors", {})
        lidar_sensor = sensors.get("lidar_sensor") if hasattr(sensors, "get") else None
        lidar_data = getattr(lidar_sensor, "data", None)
        lidar_pos = self._first_array(getattr(lidar_data, "pos_w", None), 3)
        lidar_quat = self._first_array(getattr(lidar_data, "quat_w", None), 4)
        if lidar_pos is None or lidar_quat is None:
            return None

        box_pose = self._box_ground_truth(scene, lidar_pos, lidar_quat)
        ditch_pose = self._ditch_ground_truth(scene, lidar_pos, lidar_quat)
        if box_pose is None or ditch_pose is None:
            return None
        return {"box": box_pose, "ditch": ditch_pose}

    def _box_ground_truth(self, scene, lidar_pos, lidar_quat):
        try:
            box = scene["box"]
        except Exception:
            return None
        box_pos = self._first_array(getattr(box.data, "root_pos_w", None), 3)
        box_quat = self._first_array(getattr(box.data, "root_quat_w", None), 4)
        if box_pos is None or box_quat is None:
            return None
        return self._world_pose_to_lidar_pose(box_pos, yaw_from_quat_wxyz(box_quat) + math.pi / 2.0, lidar_pos, lidar_quat)

    def _ditch_ground_truth(self, scene, lidar_pos, lidar_quat):
        env_origin = self._first_array(getattr(scene, "env_origins", None), 3)
        if env_origin is None:
            env_origin = np.zeros(3, dtype=np.float32)
        terrain = getattr(scene, "terrain", None)
        terrain_generator = getattr(terrain, "terrain_generator", None)
        cfg = getattr(terrain_generator, "cfg", None)
        size = getattr(cfg, "size", (12.0, 8.0))
        pit_cfg = None
        sub_terrains = getattr(cfg, "sub_terrains", {}) if cfg is not None else {}
        if hasattr(sub_terrains, "get"):
            pit_cfg = sub_terrains.get("pit_and_platform")
        border_width = float(getattr(pit_cfg, "border_width", 1.0)) if pit_cfg is not None else 1.0
        platform_center_y = float(size[1]) * 0.75
        platform_extent_y = float(size[1]) * 0.5 - border_width
        pit_min_y = 0.5 * border_width
        platform_min_y = platform_center_y - 0.5 * platform_extent_y
        open_ditch_center_y = 0.5 * (pit_min_y + platform_min_y)
        terrain_origin_y = float(size[1]) * 0.5
        center_local = np.array(
            [
                float(size[0]) * 0.5 - float(size[0]) * 0.15,
                open_ditch_center_y - terrain_origin_y,
                -0.5 * float(self.lidar_prior.ditch.depth),
            ],
            dtype=np.float32,
        )
        center_w = env_origin + center_local
        return self._world_pose_to_lidar_pose(center_w, math.pi / 2.0, lidar_pos, lidar_quat)

    def _world_pose_to_lidar_pose(self, pos_w, yaw_w, lidar_pos, lidar_quat):
        rel_xyz = quat_inverse_apply(lidar_quat, (pos_w - lidar_pos).reshape(1, 3))[0]
        yaw_l = align_axis_yaw_to_reference(float(yaw_w) - yaw_from_quat_wxyz(lidar_quat), math.pi / 2.0)
        return float(rel_xyz[0]), float(rel_xyz[1]), float(yaw_l)

    def _pose_gt_fields(self, prefix: str, estimate, gt_pose) -> dict[str, str | int]:
        gt_x, gt_y, gt_yaw = gt_pose
        fields = {
            f"{prefix}_gt_x_m": f"{gt_x:.4f}",
            f"{prefix}_gt_y_m": f"{gt_y:.4f}",
            f"{prefix}_gt_yaw_deg": f"{math.degrees(gt_yaw):.3f}",
        }
        if estimate is None or not estimate.valid:
            fields.update(
                {
                    f"{prefix}_valid": 0,
                    f"{prefix}_x_m": "",
                    f"{prefix}_y_m": "",
                    f"{prefix}_yaw_deg": "",
                    f"{prefix}_err_xy_m": "",
                    f"{prefix}_err_yaw_deg": "",
                }
            )
            return fields
        err_xy = math.hypot(float(estimate.x) - gt_x, float(estimate.y) - gt_y)
        err_yaw = abs(align_axis_yaw_to_reference(float(estimate.yaw), gt_yaw) - gt_yaw)
        fields.update(
            {
                f"{prefix}_valid": 1,
                f"{prefix}_x_m": f"{estimate.x:.4f}",
                f"{prefix}_y_m": f"{estimate.y:.4f}",
                f"{prefix}_yaw_deg": f"{math.degrees(estimate.yaw):.3f}",
                f"{prefix}_err_xy_m": f"{err_xy:.4f}",
                f"{prefix}_err_yaw_deg": f"{math.degrees(err_yaw):.3f}",
            }
        )
        return fields

    @staticmethod
    def _first_array(value, size: int):
        if value is None:
            return None
        array = value.detach().cpu().numpy() if hasattr(value, "detach") else value
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 2:
            array = array[0]
        if array.shape[0] < size:
            return None
        return array[:size].copy()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None


class PhysicsUsdSynchronizer:
    """Keep USD/RTX views synchronized when Fabric is disabled."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._physx_interface = None
        self._disabled_reason = None
        self._settings_applied = False
        if enabled:
            self.apply_settings()

    def apply_settings(self) -> None:
        if not self.enabled or self._disabled_reason is not None:
            return
        try:
            import carb
            import omni.physx
            from isaacsim.core.simulation_manager import SimulationManager

            SimulationManager.enable_fabric(False)
            settings = carb.settings.get_settings()
            settings.set_bool("/physics/fabricEnabled", False)
            settings.set_bool("/physics/suppressReadback", False)
            settings.set_bool("/physics/updateToUsd", True)
            settings.set_bool("/physics/updateParticlesToUsd", True)
            settings.set_bool("/physics/updateVelocitiesToUsd", True)
            settings.set_bool("/physics/updateForceSensorsToUsd", True)
            settings.set_bool("/physics/outputVelocitiesLocalSpace", False)
            self._physx_interface = omni.physx.acquire_physx_interface()
        except Exception as exc:
            self._disabled_reason = str(exc)
            print(f"[physics_usd_sync] Disabled: {exc}")
            return
        if not self._settings_applied:
            print("[physics_usd_sync] PhysX -> USD transform writeback enabled for GUI/camera synchronization.")
            self._settings_applied = True

    def sync(self) -> None:
        if not self.enabled or self._disabled_reason is not None:
            return
        if self._physx_interface is None:
            self.apply_settings()
        if self._physx_interface is None:
            return
        try:
            self._physx_interface.update_transformations(False, True, False)
        except Exception as exc:
            self._disabled_reason = str(exc)
            print(f"[physics_usd_sync] Runtime sync disabled: {exc}")


class CameraPoseSynchronizer:
    """Move RTX camera prims to their current PhysX link poses before rendering."""

    _SENSOR_NAMES = ("head_camera", "ee_camera", "ee_dual_camera")

    def __init__(self, env, enabled: bool):
        self.env = env
        self.enabled = enabled
        self.targets = []
        self._disabled_reason = None
        self._printed_ready = False
        if enabled:
            self._resolve_targets()

    def _get_scene(self):
        return getattr(getattr(self.env, "unwrapped", self.env), "scene", None)

    def _get_sensor(self, sensor_name: str):
        scene = self._get_scene()
        if scene is None:
            return None
        sensors = getattr(scene, "sensors", {})
        return sensors.get(sensor_name) if hasattr(sensors, "get") else None

    def _resolve_targets(self) -> None:
        scene = self._get_scene()
        if scene is None:
            self._disabled_reason = "scene is unavailable"
            return
        try:
            self.robot = scene["robot"]
        except Exception as exc:
            self._disabled_reason = f"robot asset is unavailable: {exc}"
            return

        for sensor_name in self._SENSOR_NAMES:
            sensor = self._get_sensor(sensor_name)
            if sensor is None:
                continue
            body_name = self._body_name_from_sensor(sensor)
            if not body_name:
                print(f"[camera_sync] Skip {sensor_name}: cannot infer parent link from prim path.")
                continue
            try:
                body_ids, body_names = self.robot.find_bodies(body_name, preserve_order=True)
            except Exception as exc:
                print(f"[camera_sync] Skip {sensor_name}: cannot resolve body '{body_name}': {exc}")
                continue
            if not body_ids:
                print(f"[camera_sync] Skip {sensor_name}: body '{body_name}' not found.")
                continue
            offset = getattr(getattr(sensor, "cfg", None), "offset", None)
            convention = getattr(offset, "convention", "ros") if offset is not None else "ros"
            offset_pos = getattr(offset, "pos", (0.0, 0.0, 0.0)) if offset is not None else (0.0, 0.0, 0.0)
            offset_rot = getattr(offset, "rot", (1.0, 0.0, 0.0, 0.0)) if offset is not None else (1.0, 0.0, 0.0, 0.0)
            self.targets.append(
                {
                    "name": sensor_name,
                    "sensor": sensor,
                    "body_id": int(body_ids[0]),
                    "body_name": body_names[0],
                    "offset_pos": tuple(offset_pos),
                    "offset_rot": tuple(offset_rot),
                    "convention": convention,
                }
            )

        if self.targets:
            target_text = ", ".join(f"{item['name']}<-{item['body_name']}" for item in self.targets)
            print(f"[camera_sync] Syncing RTX cameras from PhysX link poses: {target_text}.")
            self._printed_ready = True
        else:
            self._disabled_reason = "no camera targets resolved"
            print("[camera_sync] No camera targets resolved; depth camera pose will not be manually synchronized.")

    def sync(self) -> None:
        if not self.enabled or self._disabled_reason is not None:
            return
        try:
            from isaacsim.core.simulation_manager import SimulationManager

            physics_sim_view = SimulationManager.get_physics_sim_view()
            if physics_sim_view is not None:
                physics_sim_view.update_articulations_kinematic()
            link_poses = self.robot.root_physx_view.get_link_transforms().clone()
            if link_poses.ndim == 2:
                link_poses = link_poses.view(self.robot.num_instances, self.robot.num_bodies, 7)
            link_poses[..., 3:7] = convert_quat(link_poses[..., 3:7], to="wxyz")

            for target in self.targets:
                body_pose = link_poses[:, target["body_id"], :]
                body_pos = body_pose[:, :3]
                body_quat = body_pose[:, 3:7]
                offset_pos = torch.tensor(
                    target["offset_pos"], dtype=body_pos.dtype, device=body_pos.device
                ).repeat(body_pos.shape[0], 1)
                offset_rot = torch.tensor(
                    target["offset_rot"], dtype=body_quat.dtype, device=body_quat.device
                ).repeat(body_quat.shape[0], 1)
                cam_pos, cam_quat = combine_frame_transforms(body_pos, body_quat, offset_pos, offset_rot)
                sensor = target["sensor"]
                convention = target["convention"]
                sensor.set_world_poses(cam_pos, cam_quat, convention=convention)
                self._update_camera_data_pose(sensor, cam_pos, cam_quat, convention)
        except Exception as exc:
            self._disabled_reason = str(exc)
            print(f"[camera_sync] Runtime sync disabled: {exc}")

    def _update_camera_data_pose(self, sensor, cam_pos: torch.Tensor, cam_quat: torch.Tensor, convention: str) -> None:
        camera_data = getattr(sensor, "_data", None)
        if camera_data is None:
            return
        if getattr(camera_data, "pos_w", None) is not None:
            camera_data.pos_w[:] = cam_pos
        if getattr(camera_data, "quat_w_world", None) is not None:
            camera_data.quat_w_world[:] = convert_camera_frame_orientation_convention(
                cam_quat, origin=convention, target="world"
            )

    def _body_name_from_sensor(self, sensor) -> str | None:
        prim_path = getattr(getattr(sensor, "cfg", None), "prim_path", None)
        if not prim_path:
            return None
        parent_path = prim_path.rstrip("/").rsplit("/", 1)[0]
        if not parent_path or parent_path == prim_path:
            return None
        return parent_path.rsplit("/", 1)[-1]


class RenderPreSyncHook:
    def __init__(self, env, *callbacks):
        self.env = env
        self.callbacks = [callback for callback in callbacks if callback is not None]
        self._sim = getattr(getattr(env, "unwrapped", env), "sim", None)
        self._original_render = None
        self._installed = False

    def install(self) -> None:
        if self._installed or self._sim is None:
            return
        self._original_render = self._sim.render

        def render_with_sync(*args, **kwargs):
            self.sync()
            return self._original_render(*args, **kwargs)

        self._sim.render = render_with_sync
        self._installed = True
        print("[render_sync] Installed pre-render PhysX/USD/camera synchronization hook.")

    def sync(self) -> None:
        for callback in self.callbacks:
            callback()

    def close(self) -> None:
        if self._installed and self._sim is not None and self._original_render is not None:
            self._sim.render = self._original_render
        self._installed = False


class SensorSceneMarkers:
    def __init__(
        self,
        env,
        enabled: bool,
        depth_max: float,
        interval: int,
        lidar_prior=None,
        lidar_pose_logger: LidarPoseLogger | None = None,
        lidar_pose_gt_logger: LidarPoseGroundTruthLogger | None = None,
    ):
        self.env = env
        self.enabled = enabled
        self.depth_max = depth_max
        self.interval = max(1, interval)
        self.depth_stride = 24
        self.lidar_stride = 1
        self.lidar_prior = lidar_prior
        self.lidar_pose_stabilizer = LidarPoseStabilizer()
        self.lidar_pose_logger = lidar_pose_logger
        self.lidar_pose_gt_logger = lidar_pose_gt_logger
        self._depth_marker = None
        self._lidar_marker = None
        self._disabled_reason = None
        self._last_lidar_report = None
        self._last_pose_report = None
        if not enabled:
            return
        self._depth_marker = self._make_red_sphere_marker("/Visuals/ATEC/depth_height_points", radius=0.018)
        self._lidar_marker = self._make_red_sphere_marker("/Visuals/ATEC/lidar_height_points", radius=0.025)
        print("[sensor_vis] In-scene red markers enabled for head depth and LiDAR hit points.")

    def update(self, obs, timestep: int, sim_time_s: float | None = None) -> None:
        if not self.enabled or self._disabled_reason is not None:
            return

        head_depth = None
        if isinstance(obs, dict):
            image_obs = obs.get("image")
            if isinstance(image_obs, dict):
                head_depth = image_obs.get("head_depth")

        lidar_scan = None
        if isinstance(obs, dict):
            lidar_scan = obs.get("extero")

        try:
            depth_points_count = self._update_depth_markers(head_depth)
            lidar_points_count = self._update_lidar_markers(lidar_scan, timestep, sim_time_s)
            self._last_pose_report = self._build_pose_report()
        except Exception as exc:
            self._disabled_reason = str(exc)
            self.close()
            print(f"[sensor_vis] In-scene marker update disabled: {exc}")
            return

        if timestep % self.interval == 0:
            messages = []
            if head_depth is not None:
                messages.append(format_stats("head_depth_m", finite_stats(head_depth)))
            if lidar_scan is not None:
                messages.append(format_stats("lidar_height_scan_m", finite_stats(lidar_scan)))
            messages.append(f"depth_markers={depth_points_count}")
            messages.append(f"lidar_markers={lidar_points_count}")
            if self._last_lidar_report is not None:
                messages.append(self._last_lidar_report)
            if self._last_pose_report is not None:
                messages.append(self._last_pose_report)
            if messages:
                print("[sensor_vis] " + " | ".join(messages))

    def close(self) -> None:
        if self._depth_marker is not None:
            self._depth_marker.set_visibility(False)
        if self._lidar_marker is not None:
            self._lidar_marker.set_visibility(False)

    def _make_red_sphere_marker(self, prim_path: str, radius: float) -> VisualizationMarkers:
        marker_cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "point": sim_utils.SphereCfg(
                    radius=radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        return VisualizationMarkers(marker_cfg)

    def _get_sensor(self, sensor_name: str):
        scene = getattr(getattr(self.env, "unwrapped", self.env), "scene", None)
        if scene is None:
            return None
        sensors = getattr(scene, "sensors", {})
        return sensors.get(sensor_name) if hasattr(sensors, "get") else None

    def _show_points(self, marker: VisualizationMarkers | None, points) -> int:
        if marker is None:
            return 0
        if points.shape[0] == 0:
            marker.set_visibility(False)
            return 0
        marker.set_visibility(True)
        marker.visualize(translations=points)
        return int(points.shape[0])

    def _update_depth_markers(self, head_depth) -> int:
        if head_depth is None:
            return self._show_points(self._depth_marker, torch.empty((0, 3)).cpu().numpy())
        head_camera = self._get_sensor("head_camera")
        camera_data = getattr(head_camera, "data", None)
        if camera_data is None:
            return 0
        depth_points = depth_to_world_points(
            head_depth,
            camera_data.intrinsic_matrices,
            camera_data.pos_w,
            camera_data.quat_w_ros,
            stride=self.depth_stride,
            max_depth=self.depth_max,
        )
        return self._show_points(self._depth_marker, depth_points)

    def _get_scene(self):
        return getattr(getattr(self.env, "unwrapped", self.env), "scene", None)

    def _update_lidar_markers(self, lidar_scan, timestep: int, sim_time_s: float | None) -> int:
        lidar_sensor = self._get_sensor("lidar_sensor")
        lidar_data = getattr(lidar_sensor, "data", None)
        sensor_pos_w = getattr(lidar_data, "pos_w", None)
        if sensor_pos_w is None:
            return 0
        max_distance = getattr(getattr(lidar_sensor, "cfg", None), "max_distance", None)
        ray_hits_w = getattr(lidar_data, "ray_hits_w", None)
        if ray_hits_w is not None:
            lidar_points = lidar_hits_to_world_points(
                ray_hits_w,
                stride=self.lidar_stride,
                sensor_pos_w=sensor_pos_w,
                max_distance=max_distance,
            )
            lidar_pose_result = self._estimate_lidar_relative_poses(
                ray_hits_w,
                sensor_pos_w,
                getattr(lidar_data, "quat_w", None),
                max_distance,
            )
            if self.lidar_pose_logger is not None:
                self.lidar_pose_logger.maybe_write(lidar_pose_result, timestep, sim_time_s)
            if self.lidar_pose_gt_logger is not None:
                self.lidar_pose_gt_logger.maybe_write(lidar_pose_result, timestep, sim_time_s)
        else:
            ray_directions = getattr(lidar_sensor, "ray_directions", None)
            sensor_quat_w = getattr(lidar_data, "quat_w", None)
            if lidar_scan is None or ray_directions is None:
                return 0
            lidar_points = lidar_height_scan_to_world_points(
                lidar_scan,
                ray_directions,
                sensor_pos_w,
                sensor_quat_w,
                stride=self.lidar_stride,
                height_offset=0.5,
                max_distance=max_distance,
                yaw_only=True,
            )
        self._last_lidar_report = point_cloud_xy_report(lidar_points, sensor_pos_w)
        return self._show_points(self._lidar_marker, lidar_points)

    def _estimate_lidar_relative_poses(self, ray_hits_w, sensor_pos_w, sensor_quat_w, max_distance):
        if sensor_quat_w is None:
            return None
        result = estimate_task_d_poses_from_lidar(
            ray_hits_w,
            sensor_pos_w,
            sensor_quat_w,
            prior=self.lidar_prior,
            max_distance=max_distance,
        )
        return self.lidar_pose_stabilizer.update(result, sensor_pos_w, sensor_quat_w)

    def _build_pose_report(self) -> str | None:
        scene = self._get_scene()
        if scene is None:
            return None

        def first_vec(value) -> list[float] | None:
            if value is None:
                return None
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "numpy"):
                value = value.numpy()
            if value.ndim == 2:
                value = value[0]
            return [float(value[0]), float(value[1]), float(value[2])]

        robot_pos = None
        try:
            robot_pos = first_vec(scene["robot"].data.root_pos_w)
        except Exception:
            robot_pos = None

        lidar_sensor = self._get_sensor("lidar_sensor")
        lidar_pos = first_vec(getattr(getattr(lidar_sensor, "data", None), "pos_w", None))
        head_camera = self._get_sensor("head_camera")
        head_pos = first_vec(getattr(getattr(head_camera, "data", None), "pos_w", None))

        parts = []
        if robot_pos is not None:
            parts.append(f"robot=({robot_pos[0]:+.2f},{robot_pos[1]:+.2f},{robot_pos[2]:+.2f})")
        if lidar_pos is not None:
            parts.append(f"lidar=({lidar_pos[0]:+.2f},{lidar_pos[1]:+.2f},{lidar_pos[2]:+.2f})")
        if head_pos is not None:
            parts.append(f"head=({head_pos[0]:+.2f},{head_pos[1]:+.2f},{head_pos[2]:+.2f})")
        return "pose " + " ".join(parts) if parts else None


def play() -> tuple[float, float]:
    if args_cli.task is None:
        raise ValueError("Please provide --task, e.g. --task ATEC-TaskA-G1")

    is_task_e = isinstance(args_cli.task, str) and args_cli.task.startswith("ATEC-TaskE")
    teleop = KeyboardTeleop() if args_cli.keyboard else None
    # -------------------------------------------------------------------------
    # Create env (plain Gym env)
    # -------------------------------------------------------------------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )

    if args_cli.sensor_vis:
        lidar_sensor = getattr(env_cfg.scene, "lidar_sensor", None)
        if lidar_sensor is not None:
            lidar_sensor.debug_vis = False
            print("[sensor_vis] LiDAR hit points will be visualized by in-scene red markers.")
        else:
            print("[sensor_vis] This task/robot has no lidar_sensor configured.")
    lidar_prior = build_task_d_lidar_prior(env_cfg)
    if args_cli.sensor_vis:
        print(
            "[lidar_perception] priors "
            f"box(L={lidar_prior.box.length:.2f}, W={lidar_prior.box.width:.2f}, H={lidar_prior.box.height:.2f}) "
            f"ditch(L={lidar_prior.ditch.length:.2f}, W={lidar_prior.ditch.width:.2f}, D={lidar_prior.ditch.depth:.2f})"
        )

    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    action_spec_json = json.dumps(action_spec)

    # New Feature: apply safe action spec to env config (e.g. for scaling/clipping actions from your solution)
    env_cfg = apply_safe_action_spec(env_cfg, action_spec_json)
    
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Convert MARL -> single agent if needed (kept from your original script)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # -------------------------------------------------------------------------
    # Optional: video wrapper
    # -------------------------------------------------------------------------
    if args_cli.video:
        # Put videos in ./logs/videos/play by default (edit as you like)
        video_kwargs = {
            "video_folder": os.path.abspath(os.path.join("logs", "videos", args_cli.task, "play")),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    physics_usd_sync = PhysicsUsdSynchronizer(enabled=args_cli.enable_cameras and args_cli.disable_fabric)

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------
    obs, _ = env.reset()
    camera_pose_sync = CameraPoseSynchronizer(env, enabled=args_cli.enable_cameras and args_cli.disable_fabric)
    render_sync_hook = RenderPreSyncHook(env, physics_usd_sync.sync, camera_pose_sync.sync)
    render_sync_hook.install()
    render_sync_hook.sync()
    lidar_pose_logger = LidarPoseLogger(
        args_cli.lidar_pose_log if args_cli.sensor_vis else "",
        args_cli.lidar_pose_log_interval,
    )
    lidar_pose_gt_logger = LidarPoseGroundTruthLogger(
        args_cli.lidar_pose_gt_log if args_cli.sensor_vis else "",
        args_cli.lidar_pose_log_interval,
        env,
        lidar_prior,
    )
    sensor_scene_markers = SensorSceneMarkers(
        env,
        enabled=args_cli.sensor_vis,
        depth_max=args_cli.depth_vis_max,
        interval=args_cli.sensor_vis_interval,
        lidar_prior=lidar_prior,
        lidar_pose_logger=lidar_pose_logger,
        lidar_pose_gt_logger=lidar_pose_gt_logger,
    )

    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
    timestep = 0

    # -------------------------------------------------------------------------
    # Play loop
    # -------------------------------------------------------------------------
    total_episode_reward = 0.0
    total_elapsed_time = 0.0
    while simulation_app.is_running():
        with torch.inference_mode():
            start_time = time.time()

            # ===== Your controller goes here =====
            if teleop is not None:
                nav_cmd, height_cmd, switch = teleop.advance()
                if switch is not None:
                    from demo.solution import set_policy, get_policy_name, _ensure_impl
                    if set_policy(switch if switch != "homie" else "gr00t"):
                        _ensure_impl()
                        print(f"[play] Switched to policy: {get_policy_name()}")
                if hasattr(solution, "set_keyboard_command"):
                    solution.set_keyboard_command(nav_cmd, height_cmd)
            resp = solution.predicts(obs, total_episode_reward)
            should_print_debug = args_cli.debug and timestep % max(1, args_cli.debug_interval) == 0
            if should_print_debug and hasattr(solution, "get_debug_snapshot"):
                print(solution.get_debug_snapshot())
            giveup = resp["giveup"]
            if giveup:
                break
            actions = resp["action"]
            actions = torch.tensor(actions, dtype=torch.float32, device=args_cli.device).view(args_cli.num_envs, -1)
            obs, reward, terminated, truncated, info = env.step(actions)
            render_sync_hook.sync()
            if not is_task_e:
                pass  # camera_follow(env)

            sim_dt = info["Step_dt"]
            if isinstance(reward, torch.Tensor):
                total_episode_reward += reward.mean().item() / sim_dt
            else:
                total_episode_reward += float(reward) / sim_dt

            if isinstance(info, dict) and "Elapsed_Time" in info:
                elapsed = info["Elapsed_Time"]  # simulation time from env as primary source
                total_elapsed_time = elapsed.item() if hasattr(elapsed, "item") else float(elapsed)
            elif dt is not None:
                total_elapsed_time += dt  # wall clock time as fallback

            if should_print_debug:
                print(f"total_episode_reward:{total_episode_reward: .2f}")
                print(f"total_elapsed_time:{total_elapsed_time: .2f}")

            sensor_scene_markers.update(obs, timestep, total_elapsed_time)

            done = (terminated.item() or truncated.item())
            if done:
                break

            timestep += 1
            # If recording one video, exit after video_length steps
            if args_cli.video and timestep >= args_cli.video_length:
                break

            # Real-time pacing
            if args_cli.real_time and dt is not None:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    sensor_scene_markers.close()
    lidar_pose_logger.close()
    lidar_pose_gt_logger.close()
    render_sync_hook.close()
    env.close()
    if teleop is not None:
        teleop.close()

    return total_episode_reward, total_elapsed_time


if __name__ == "__main__":
    score, elapsed_time = play()
    print(f"score: {score:.2f}, elapsed_time: {elapsed_time:.2f} seconds")

    # Finally, close the simulation app
    print("Closing simulation app...")
    simulation_app.close()
