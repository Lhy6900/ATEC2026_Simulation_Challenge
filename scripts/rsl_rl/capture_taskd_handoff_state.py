"""Capture the official TaskD handoff state used to initialize CrossPitBox training."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


def quat_wxyz_to_yaw(quat: list[float] | tuple[float, float, float, float]) -> float:
    """Return yaw from a wxyz quaternion."""
    qw, qx, qy, qz = [float(value) for value in quat]
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _tensor_row_to_list(value: Any, row: int = 0) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[row]
    return [float(item) for item in value]


def pose_xyzyaw_from_asset(asset: Any, env_origin: Any, env_index: int = 0) -> list[float]:
    """Return env-local ``x, y, z, yaw`` from an IsaacLab asset."""
    pos_w = _tensor_row_to_list(asset.data.root_pos_w, env_index)
    quat_w = _tensor_row_to_list(asset.data.root_quat_w, env_index)
    origin = _tensor_row_to_list(env_origin)
    return [
        pos_w[0] - origin[0],
        pos_w[1] - origin[1],
        pos_w[2] - origin[2],
        quat_wxyz_to_yaw(quat_w),
    ]


def _asset_snapshot(asset: Any, env_index: int = 0) -> dict[str, Any]:
    data = asset.data
    snapshot = {
        "root_pos_w": _tensor_row_to_list(data.root_pos_w, env_index),
        "root_quat_w": _tensor_row_to_list(data.root_quat_w, env_index),
        "root_lin_vel_w": _tensor_row_to_list(data.root_lin_vel_w, env_index),
        "root_ang_vel_w": _tensor_row_to_list(data.root_ang_vel_w, env_index),
    }
    if hasattr(data, "joint_pos"):
        snapshot["joint_pos"] = _tensor_row_to_list(data.joint_pos, env_index)
    if hasattr(data, "joint_vel"):
        snapshot["joint_vel"] = _tensor_row_to_list(data.joint_vel, env_index)
    if hasattr(asset, "joint_names"):
        snapshot["joint_names"] = list(asset.joint_names)
    return snapshot


def _write_snapshot(path: str | None, snapshot: dict[str, Any]) -> None:
    text = json.dumps(snapshot, indent=2, sort_keys=True)
    print("=== TASKD_HANDOFF_STATE_JSON_BEGIN ===")
    print(text)
    print("=== TASKD_HANDOFF_STATE_JSON_END ===")
    if path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")
        print(f"[INFO] Wrote handoff snapshot to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture official TaskD robot/box state at a fixed step.")
    parser.add_argument("--task", type=str, default="ATEC-TaskD-G1")
    parser.add_argument("--step", type=int, default=650)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--screenshot_output", type=str, default=None)
    parser.add_argument(
        "--screenshot_view",
        choices=("play", "overview", "topdown"),
        default="play",
        help="Camera preset used with --screenshot_output.",
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--fast_headless_capture",
        action="store_true",
        default=False,
        help="Disable cameras and task terminations for state dumping at a fixed step.",
    )
    parser.add_argument(
        "--planner_decimation",
        type=int,
        default=None,
        help="Optional diagnostic override for demo.boxpush_planner_solution.PLANNER_DECIMATION.",
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    if args_cli.screenshot_output:
        args_cli.enable_cameras = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import numpy as np
    import torch
    from PIL import Image

    import atec_rl_lab.tasks  # noqa: F401
    from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec
    if args_cli.planner_decimation is not None:
        import demo.boxpush_planner_solution as boxpush_planner_solution

        boxpush_planner_solution.PLANNER_DECIMATION = int(args_cli.planner_decimation)
        print(f"[INFO] Overrode BoxPush planner decimation to {boxpush_planner_solution.PLANNER_DECIMATION}")

    from demo.solution import AlgSolution
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab.sensors import CameraCfg
    from isaaclab_tasks.utils import parse_env_cfg

    def _add_render_camera(cfg) -> None:
        cfg.scene.lazy_sensor_update = False
        cfg.scene.render_cam = CameraCfg(
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

    def _camera_eye_target(scene) -> tuple[torch.Tensor, torch.Tensor]:
        robot_pos = scene["robot"].data.root_pos_w[0].detach().cpu()
        box_pos = scene["box"].data.root_pos_w[0].detach().cpu()
        center = 0.5 * (robot_pos + box_pos)
        if args_cli.screenshot_view == "topdown":
            eye = center + torch.tensor([-0.05, -0.20, 6.0])
            target = center
        elif args_cli.screenshot_view == "overview":
            eye = center + torch.tensor([-3.0, -3.8, 3.4])
            target = center + torch.tensor([0.25, 0.0, 0.05])
        else:
            eye = center + torch.tensor([-2.25, -2.15, 2.05])
            target = center + torch.tensor([0.25, 0.0, 0.10])
        return eye, target

    def _write_screenshot(env, path_text: str, snapshot: dict[str, Any]) -> None:
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        scene = env.unwrapped.scene
        eye, target = _camera_eye_target(scene)
        env.unwrapped.sim.set_camera_view(eye=eye.tolist(), target=target.tolist())
        camera = scene["render_cam"]
        camera.set_world_poses_from_view(
            eyes=eye.to(env.unwrapped.device).unsqueeze(0),
            targets=target.to(env.unwrapped.device).unsqueeze(0),
        )
        for _ in range(10):
            env.unwrapped.sim.render()
            scene.update(env.unwrapped.physics_dt)
            simulation_app.update()
        rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        rgb = np.asarray(rgb, dtype=np.uint8)
        Image.fromarray(rgb[..., :3]).save(path)
        metadata = {
            "screenshot_view": args_cli.screenshot_view,
            "camera_eye_w": eye.tolist(),
            "camera_target_w": target.tolist(),
            "snapshot": snapshot,
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        print(f"[INFO] Wrote handoff screenshot to: {path}")

    solution = AlgSolution()
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.screenshot_output:
        _add_render_camera(env_cfg)
    if args_cli.fast_headless_capture:
        env_cfg.scene.head_camera = None
        env_cfg.scene.ee_camera = None
        env_cfg.scene.ee_dual_camera = None
        env_cfg.observations.image = None
        env_cfg.terminations.time_out = None
        env_cfg.terminations.fall = None
        env_cfg.terminations.x_reached = None
    env_cfg = apply_safe_action_spec(env_cfg, json.dumps(solution.get_action_spec() or {}))
    env_cfg.log_dir = os.path.abspath(os.path.join("logs", "taskd_handoff_capture"))
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "log_dir"):
        env_cfg.sim.log_dir = env_cfg.log_dir

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    obs, _ = env.reset()
    total_episode_reward = 0.0
    elapsed_time = 0.0
    terminated_at_step: int | None = None
    try:
        for step in range(1, args_cli.step + 1):
            with torch.inference_mode():
                resp = solution.predicts(obs, total_episode_reward)
                if resp.get("giveup", False):
                    terminated_at_step = step
                    break
                action = torch.tensor(resp["action"], dtype=torch.float32, device=env.unwrapped.device).view(
                    args_cli.num_envs, -1
                )
                obs, reward, terminated, truncated, info = env.step(action)

            sim_dt = float(info.get("Step_dt", env.unwrapped.step_dt))
            if isinstance(reward, torch.Tensor):
                total_episode_reward += reward.mean().item() / sim_dt
            else:
                total_episode_reward += float(reward) / sim_dt
            if "Elapsed_Time" in info:
                value = info["Elapsed_Time"]
                elapsed_time = value.item() if hasattr(value, "item") else float(value)
            else:
                elapsed_time += sim_dt
            if args_cli.debug and (step == 1 or step % 50 == 0 or step == args_cli.step):
                scene = env.unwrapped.scene
                origin = scene.env_origins[0]
                robot_pose = pose_xyzyaw_from_asset(scene["robot"], origin)
                box_pose = pose_xyzyaw_from_asset(scene["box"], origin)
                print(
                    "[CAPTURE_DEBUG] "
                    f"step={step} elapsed={elapsed_time:.3f} "
                    f"robot_xyzyaw={[round(v, 4) for v in robot_pose]} "
                    f"box_xyzyaw={[round(v, 4) for v in box_pose]}"
                )
            if bool(terminated[0].item() or truncated[0].item()):
                terminated_at_step = step
                break

        scene = env.unwrapped.scene
        env_origin = scene.env_origins[0]
        robot = scene["robot"]
        box = scene["box"]
        snapshot = {
            "task": args_cli.task,
            "requested_step": args_cli.step,
            "captured_step": step,
            "terminated_at_step": terminated_at_step,
            "elapsed_time_s": elapsed_time,
            "env_origin_w": _tensor_row_to_list(env_origin),
            "robot": _asset_snapshot(robot),
            "box": _asset_snapshot(box),
            "cross_pit_robot_pose_xyzyaw": pose_xyzyaw_from_asset(robot, env_origin),
            "cross_pit_box_pose_xyzyaw": pose_xyzyaw_from_asset(box, env_origin),
        }
        _write_snapshot(args_cli.output, snapshot)
        if args_cli.screenshot_output:
            _write_screenshot(env, args_cli.screenshot_output, snapshot)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
