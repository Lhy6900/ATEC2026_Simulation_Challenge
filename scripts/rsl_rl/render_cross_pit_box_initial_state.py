"""Render a CrossPitBox reset state without taking a physics step."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Render CrossPitBox reset state to a PNG.")
parser.add_argument("--task", type=str, default="ATEC-Isaac-TaskD-G1-CrossPitBox-v2")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument(
    "--snapshot_json",
    type=str,
    default=None,
    help="Optional TaskD handoff snapshot JSON to inject into the CrossPitBox reset events before rendering.",
)
parser.add_argument(
    "--view",
    choices=("reset", "oblique", "overview", "topdown"),
    default="reset",
    help=(
        "Camera preset. 'reset' matches the v1 reset overview style; 'oblique' shows lateral offset; "
        "'overview' is higher and wider; 'topdown' is for state verification."
    ),
)
parser.add_argument(
    "--no_reset_noise",
    action="store_true",
    default=False,
    help="Disable reset event noise for a nominal configuration render.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import atec_rl_lab.train  # noqa: F401, E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _add_render_camera(env_cfg) -> None:
    env_cfg.scene.lazy_sensor_update = False
    env_cfg.scene.render_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RenderCamera",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def _snapshot_pose(snapshot: dict, asset_name: str) -> tuple[float, ...]:
    asset = snapshot[asset_name]
    root_pos = asset["root_pos_w"]
    quat = asset["root_quat_w"]
    origin = snapshot["env_origin_w"]
    return (
        float(root_pos[0]) - float(origin[0]),
        float(root_pos[1]) - float(origin[1]),
        float(root_pos[2]) - float(origin[2]),
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    )


def _apply_snapshot_override(env_cfg, snapshot_path: str) -> None:
    snapshot = json.loads(Path(snapshot_path).read_text())
    robot_joint_names = snapshot["robot"]["joint_names"]
    robot_joint_pos = snapshot["robot"]["joint_pos"]
    joint_pos = {name: float(value) for name, value in zip(robot_joint_names, robot_joint_pos)}

    reset_robot = env_cfg.events.reset_robot
    reset_robot.params["pose"] = _snapshot_pose(snapshot, "robot")
    reset_robot.params["pose_noise"] = {}
    reset_robot.params["velocity_noise"] = {}
    reset_robot.params["joint_pos_overrides"] = joint_pos
    reset_robot.params["joint_pos_noise"] = 0.0

    reset_box = env_cfg.events.reset_box_support
    reset_box.params["pose"] = _snapshot_pose(snapshot, "box")
    reset_box.params["pose_noise"] = {}


def _wait_for_file(path: Path, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return
        simulation_app.update()
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for screenshot: {path}")


def _capture_viewport(path: Path) -> None:
    from omni.kit.viewport.utility import get_active_viewport
    import omni.renderer_capture

    path.parent.mkdir(parents=True, exist_ok=True)
    viewport = get_active_viewport()
    viewport.resolution = (1280, 720)
    viewport.resolution_scale = 1
    for _ in range(5):
        simulation_app.update()
    capture = omni.renderer_capture.acquire_renderer_capture_interface()
    capture.capture_next_frame_swapchain(str(path))
    for _ in range(3):
        simulation_app.update()
    capture.wait_async_capture()
    for _ in range(3):
        simulation_app.update()
    _wait_for_file(path)


def _scene_metadata(env, eye: torch.Tensor | None = None, target: torch.Tensor | None = None) -> dict:
    scene = env.unwrapped.scene
    origin = scene.env_origins[0].detach().cpu()
    robot_pos = scene["robot"].data.root_pos_w[0].detach().cpu()
    box_pos = scene["box_support"].data.root_pos_w[0].detach().cpu()
    metadata = {
        "view": args_cli.view,
        "env_origin_w": origin.tolist(),
        "robot_pos_w": robot_pos.tolist(),
        "box_pos_w": box_pos.tolist(),
        "robot_pos_env": (robot_pos - origin).tolist(),
        "box_pos_env": (box_pos - origin).tolist(),
    }
    if eye is not None:
        metadata["camera_eye_w"] = eye.detach().cpu().tolist()
    if target is not None:
        metadata["camera_target_w"] = target.detach().cpu().tolist()
    return metadata


def _write_sidecar_metadata(env, path: Path, eye: torch.Tensor, target: torch.Tensor) -> None:
    metadata = _scene_metadata(env, eye=eye, target=target)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def _camera_eye_target(env) -> tuple[torch.Tensor, torch.Tensor]:
    scene = env.unwrapped.scene
    robot_pos = scene["robot"].data.root_pos_w[0].detach().cpu()
    box_pos = scene["box_support"].data.root_pos_w[0].detach().cpu()
    center = 0.5 * (robot_pos + box_pos)

    if args_cli.view == "topdown":
        eye = center + torch.tensor([-0.05, -0.20, 6.0])
        target = center + torch.tensor([0.0, 0.0, 0.0])
    elif args_cli.view == "overview":
        eye = center + torch.tensor([-3.0, -3.8, 3.4])
        target = center + torch.tensor([0.25, 0.0, 0.05])
    elif args_cli.view == "oblique":
        eye = center + torch.tensor([-2.1, -2.9, 2.25])
        target = center + torch.tensor([0.35, 0.0, 0.15])
    else:
        eye = center + torch.tensor([-2.25, -2.15, 2.05])
        target = center + torch.tensor([0.25, 0.0, 0.10])

    return eye, target


def _set_camera(env) -> None:
    eye, target = _camera_eye_target(env)
    env.unwrapped.sim.set_camera_view(eye=eye.tolist(), target=target.tolist())
    if "render_cam" in env.unwrapped.scene.sensors:
        camera = env.unwrapped.scene["render_cam"]
        camera.set_world_poses_from_view(
            eyes=eye.to(env.unwrapped.device).unsqueeze(0),
            targets=target.to(env.unwrapped.device).unsqueeze(0),
        )


def _capture_sensor_camera(env, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    camera = env.unwrapped.scene["render_cam"]
    for _ in range(10):
        env.unwrapped.sim.render()
        env.unwrapped.scene.update(env.unwrapped.physics_dt)
        simulation_app.update()

    rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    rgb = np.asarray(rgb, dtype=np.uint8)
    if float(rgb[..., :3].std()) < 1.0:
        raise RuntimeError("Camera capture looks blank; refusing to write an unusable screenshot.")
    Image.fromarray(rgb[..., :3]).save(path)
    _wait_for_file(path)


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.log_dir = os.path.abspath(os.path.dirname(args_cli.output))
    _add_render_camera(env_cfg)
    if args_cli.snapshot_json:
        _apply_snapshot_override(env_cfg, args_cli.snapshot_json)

    if args_cli.no_reset_noise:
        reset_robot = env_cfg.events.reset_robot
        reset_robot.params["pose_noise"] = {}
        reset_robot.params["velocity_noise"] = {}
        reset_robot.params["joint_pos_noise"] = 0.0

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset()
        output_path = Path(args_cli.output)
        eye, target = _camera_eye_target(env)
        _set_camera(env)
        _write_sidecar_metadata(env, output_path, eye, target)
        _capture_sensor_camera(env, output_path)
        print(f"[INFO] wrote screenshot: {args_cli.output}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
