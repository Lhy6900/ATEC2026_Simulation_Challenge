# Created by skywoodsz on 2026/02/07.

import argparse
import importlib
import importlib.util
import math
import os
import sys
import time
import json
from pathlib import Path
from typing import Any

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

# Isaac Sim / Kit args
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# If recording video, need cameras enabled in IsaacLab/Kit
if args_cli.video:
    args_cli.enable_cameras = True

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

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.sensors import MultiMeshRayCasterCfg, patterns  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402 (register your tasks)
from isaaclab_tasks.utils import parse_env_cfg
from rl_utils import camera_follow
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec


def _load_alg_solution_class():
    solution_file = os.environ.get("ATEC_SOLUTION_FILE")
    if solution_file:
        path = Path(solution_file).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"ATEC_SOLUTION_FILE does not exist: {path}")
        parent = str(path.parent.resolve())
        if parent not in sys.path:
            sys.path.insert(0, parent)
        spec = importlib.util.spec_from_file_location("atec_solution_runtime", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load solution file: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.AlgSolution

    module_name = os.environ.get("ATEC_SOLUTION_MODULE", "demo.solution")
    return importlib.import_module(module_name).AlgSolution


AlgSolution = _load_alg_solution_class()
solution = AlgSolution()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _make_cross_pit_depth_scanner_cfg() -> MultiMeshRayCasterCfg:
    """Build the v1 CrossPitBox training raycaster for local TaskD handoff diagnostics."""
    return MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/d435_link",
        update_period=0.02,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=(2.3000000000000003, 1.5),
            direction=(0.0, 0.0, -1.0),
            ordering="xy",
        ),
        max_distance=6.0,
        ray_alignment="yaw",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(1.3, 0.0, 2.5)),
        mesh_prim_paths=[
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Box",
                is_shared=True,
                track_mesh_transforms=True,
            ),
        ],
    )


def _attach_cross_pit_debug_obs(env: Any, obs: dict[str, Any]) -> dict[str, Any]:
    """Attach local-only scene/raycast diagnostics for the demo CrossPitBox wrapper."""
    try:
        scene = env.unwrapped.scene
        robot = scene["robot"]
        obs = dict(obs)
        obs["_debug_root_pos_w"] = robot.data.root_pos_w.detach().clone()
        obs["_debug_root_quat_w"] = robot.data.root_quat_w.detach().clone()
        depth_scanner = scene.sensors.get("depth_scanner")
        if depth_scanner is not None:
            obs["_debug_depth_ray_hits_w"] = depth_scanner.data.ray_hits_w.detach().clone()
        camera = scene.sensors.get("head_camera")
        if camera is not None:
            obs["_debug_head_camera_intrinsics"] = camera.data.intrinsic_matrices.detach().clone()
            obs["_debug_head_camera_pos_w"] = camera.data.pos_w.detach().clone()
            obs["_debug_head_camera_quat_w"] = camera.data.quat_w_world.detach().clone()
        return obs
    except Exception:
        return obs


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


def _quat_wxyz_to_yaw(quat: list[float] | tuple[float, float, float, float]) -> float:
    qw, qx, qy, qz = [float(value) for value in quat]
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


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


def _pose_xyzyaw_from_asset(asset: Any, env_origin: Any, env_index: int = 0) -> list[float]:
    pos_w = _tensor_row_to_list(asset.data.root_pos_w, env_index)
    quat_w = _tensor_row_to_list(asset.data.root_quat_w, env_index)
    origin = _tensor_row_to_list(env_origin)
    return [
        pos_w[0] - origin[0],
        pos_w[1] - origin[1],
        pos_w[2] - origin[2],
        _quat_wxyz_to_yaw(quat_w),
    ]


def _first_bool(value: Any) -> bool:
    if hasattr(value, "__getitem__"):
        value = value[0]
    if hasattr(value, "item"):
        value = value.item()
    return bool(value)


def _done_terms(env: Any) -> dict[str, bool]:
    try:
        termination_manager = env.unwrapped.termination_manager
        terms = {}
        for name, value in termination_manager.get_active_iterable_terms(env_idx=0):
            if hasattr(value, "detach"):
                active = bool(value.detach().any().item())
            elif isinstance(value, (list, tuple)):
                active = any(bool(item) for item in value)
            else:
                active = bool(value)
            terms[str(name)] = active
        return terms
    except Exception:
        return {}


def _json_ready(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _wait_for_capture_file(path: Path, timeout_s: float = 20.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return
        simulation_app.update()
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for screenshot: {path}")


def _capture_play_viewport_png(env: Any, output_path: str) -> None:
    from omni.kit.viewport.utility import get_active_viewport
    import omni.renderer_capture
    import torch

    scene = env.unwrapped.scene
    robot_pos = scene["robot"].data.root_pos_w[0].detach().cpu()
    box_pos = scene["box"].data.root_pos_w[0].detach().cpu()
    center = 0.5 * (robot_pos + box_pos)
    eye = center + torch.tensor([-3.0, -3.8, 3.2])
    target = center + torch.tensor([0.25, 0.0, 0.10])

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env.unwrapped.sim.set_camera_view(eye=eye.tolist(), target=target.tolist())

    viewport = get_active_viewport()
    viewport.resolution = (1280, 720)
    viewport.resolution_scale = 1
    for _ in range(8):
        simulation_app.update()

    capture = omni.renderer_capture.acquire_renderer_capture_interface()
    capture.capture_next_frame_swapchain(str(path))
    for _ in range(3):
        simulation_app.update()
    capture.wait_async_capture()
    for _ in range(3):
        simulation_app.update()
    _wait_for_capture_file(path)
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "camera_eye_w": eye.tolist(),
                "camera_target_w": target.tolist(),
                "robot_pos_w": robot_pos.tolist(),
                "box_pos_w": box_pos.tolist(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"[PLAY_CAPTURE] wrote viewport screenshot {path}")


def _maybe_write_play_capture(
    env: Any,
    output_path: str,
    *,
    task: str,
    step_index: int,
    elapsed_time: float,
    reward: Any,
    terminated: Any,
    truncated: Any,
    info: dict[str, Any],
    action: Any,
    png_output_path: str | None = None,
    capture_reason: str = "time",
) -> None:
    scene = env.unwrapped.scene
    env_origin = scene.env_origins[0]
    robot = scene["robot"]
    box = scene["box"]
    snapshot = {
        "task": task,
        "step_index": int(step_index),
        "elapsed_time_s": float(elapsed_time),
        "step_dt": float(info.get("Step_dt", env.unwrapped.step_dt)),
        "terminated": _first_bool(terminated),
        "truncated": _first_bool(truncated),
        "env_origin_w": _tensor_row_to_list(env_origin),
        "robot": _asset_snapshot(robot),
        "box": _asset_snapshot(box),
        "cross_pit_robot_pose_xyzyaw": _pose_xyzyaw_from_asset(robot, env_origin),
        "cross_pit_box_pose_xyzyaw": _pose_xyzyaw_from_asset(box, env_origin),
        "input_action": _json_ready(action),
        "reward": _json_ready(reward),
        "info_keys": sorted(str(key) for key in info.keys()),
        "done_terms": _done_terms(env),
        "capture_reason": capture_reason,
    }
    if hasattr(solution, "get_debug_snapshot"):
        snapshot["solution_debug"] = _json_ready(solution.get_debug_snapshot())

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    print(f"[PLAY_CAPTURE] wrote {path} at elapsed_time={elapsed_time:.3f}s step={step_index}")
    if png_output_path:
        _capture_play_viewport_png(env, png_output_path)


def _maybe_write_trace_record(
    env: Any,
    trace_file: Any,
    *,
    task: str,
    phase: str,
    step_index: int,
    elapsed_time: float,
    action: Any | None,
    obs: dict[str, Any],
    reward: Any | None = None,
    terminated: Any | None = None,
    truncated: Any | None = None,
    info: dict[str, Any] | None = None,
) -> None:
    if trace_file is None:
        return

    scene = env.unwrapped.scene
    env_origin = scene.env_origins[0]
    robot = scene["robot"]
    box = scene["box"]
    proprio = obs.get("proprio") if isinstance(obs, dict) else None
    proprio_head: dict[str, Any] = {}
    if proprio is not None:
        proprio_head = {
            "base_lin_vel": _json_ready(proprio[:, 0:3]),
            "base_ang_vel": _json_ready(proprio[:, 3:6]),
            "velocity_commands_env": _json_ready(proprio[:, 6:9]),
            "projected_gravity": _json_ready(proprio[:, 9:12]),
            "joint_pos_head": _json_ready(proprio[:, 12:20]),
            "joint_pos_norm": _json_ready(torch.linalg.norm(proprio[:, 12:45], dim=-1)),
            "joint_vel_head": _json_ready(proprio[:, 45:53]),
            "joint_vel_norm": _json_ready(torch.linalg.norm(proprio[:, 45:78], dim=-1)),
            "last_action_head": _json_ready(proprio[:, 78:86]),
            "last_action_norm": _json_ready(torch.linalg.norm(proprio[:, 78:111], dim=-1)),
        }

    record = {
        "task": task,
        "phase": phase,
        "step_index": int(step_index),
        "elapsed_time_s": float(elapsed_time),
        "robot": _asset_snapshot(robot),
        "box": _asset_snapshot(box),
        "cross_pit_robot_pose_xyzyaw": _pose_xyzyaw_from_asset(robot, env_origin),
        "cross_pit_box_pose_xyzyaw": _pose_xyzyaw_from_asset(box, env_origin),
        "proprio": proprio_head,
        "input_action": _json_ready(action) if action is not None else None,
        "reward": _json_ready(reward),
        "terminated": _json_ready(terminated),
        "truncated": _json_ready(truncated),
        "info_keys": sorted(str(key) for key in (info or {}).keys()),
        "done_terms": _done_terms(env),
    }
    if hasattr(solution, "get_debug_snapshot"):
        record["solution_debug"] = _json_ready(solution.get_debug_snapshot())
    trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    trace_file.flush()


def play() -> tuple[float, float]:
    if args_cli.task is None:
        raise ValueError("Please provide --task, e.g. --task ATEC-TaskA-G1")

    is_task_e = isinstance(args_cli.task, str) and args_cli.task.startswith("ATEC-TaskE")
    # -------------------------------------------------------------------------
    # Create env (plain Gym env)
    # -------------------------------------------------------------------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )
    attach_cross_pit_depth_scanner = _env_flag("ATEC_PLAY_ATTACH_CROSS_PIT_DEPTH_SCANNER")
    if attach_cross_pit_depth_scanner:
        env_cfg.scene.depth_scanner = _make_cross_pit_depth_scanner_cfg()

    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    action_spec_json = json.dumps(action_spec or {})

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


    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------
    obs, _ = env.reset()

    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
    timestep = 0
    capture_time_text = os.environ.get("ATEC_PLAY_CAPTURE_TIME")
    capture_json = os.environ.get("ATEC_PLAY_CAPTURE_JSON")
    capture_png = os.environ.get("ATEC_PLAY_CAPTURE_PNG")
    final_capture_json = os.environ.get("ATEC_PLAY_FINAL_CAPTURE_JSON")
    capture_exit = os.environ.get("ATEC_PLAY_CAPTURE_EXIT", "1").strip().lower() not in {"0", "false", "no"}
    trace_jsonl = os.environ.get("ATEC_PLAY_TRACE_JSONL")
    trace_start_s = float(os.environ.get("ATEC_PLAY_TRACE_START", "0.0"))
    trace_end_text = os.environ.get("ATEC_PLAY_TRACE_END")
    trace_end_s = float(trace_end_text) if trace_end_text is not None else None
    trace_max_records = int(os.environ.get("ATEC_PLAY_TRACE_MAX_RECORDS", "0"))
    trace_file = None
    trace_records = 0
    capture_time_s = None
    capture_written = False
    if capture_time_text is not None or capture_json is not None:
        if capture_time_text is None or capture_json is None:
            raise ValueError("Set both ATEC_PLAY_CAPTURE_TIME and ATEC_PLAY_CAPTURE_JSON for play capture.")
        capture_time_s = float(capture_time_text)
    if trace_jsonl is not None:
        trace_path = Path(trace_jsonl)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_file = trace_path.open("w", encoding="utf-8")

    # -------------------------------------------------------------------------
    # Play loop
    # -------------------------------------------------------------------------
    total_episode_reward = 0.0
    total_elapsed_time = 0.0
    while simulation_app.is_running():
        with torch.inference_mode():
            start_time = time.time()

            # ===== Your controller goes here =====
            obs_for_policy = _attach_cross_pit_debug_obs(env, obs) if attach_cross_pit_depth_scanner else obs
            resp = solution.predicts(obs_for_policy, total_episode_reward)
            giveup = resp["giveup"]
            if giveup:
                break
            actions = resp["action"]
            actions = torch.tensor(actions, dtype=torch.float32, device='cuda').view(1, -1)
            should_trace = (
                trace_file is not None
                and total_elapsed_time >= trace_start_s
                and (trace_end_s is None or total_elapsed_time <= trace_end_s)
                and (trace_max_records <= 0 or trace_records < trace_max_records)
            )
            if should_trace:
                _maybe_write_trace_record(
                    env,
                    trace_file,
                    task=args_cli.task,
                    phase="pre_step",
                    step_index=timestep,
                    elapsed_time=total_elapsed_time,
                    action=actions,
                    obs=obs,
                )
                trace_records += 1
            obs, reward, terminated, truncated, info = env.step(actions)
            # if not is_task_e:
            #     camera_follow(env)

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

            if args_cli.debug:
                print(f"total_episode_reward:{total_episode_reward: .2f}")
                print(f"total_elapsed_time:{total_elapsed_time: .2f}")

            should_trace = (
                trace_file is not None
                and total_elapsed_time >= trace_start_s
                and (trace_end_s is None or total_elapsed_time <= trace_end_s)
                and (trace_max_records <= 0 or trace_records < trace_max_records)
            )
            if should_trace:
                _maybe_write_trace_record(
                    env,
                    trace_file,
                    task=args_cli.task,
                    phase="post_step",
                    step_index=timestep + 1,
                    elapsed_time=total_elapsed_time,
                    action=actions,
                    obs=obs,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
                trace_records += 1

            if (
                capture_time_s is not None
                and capture_json is not None
                and not capture_written
                and total_elapsed_time >= capture_time_s
            ):
                _maybe_write_play_capture(
                    env,
                    capture_json,
                    task=args_cli.task,
                    step_index=timestep + 1,
                    elapsed_time=total_elapsed_time,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    action=actions,
                    png_output_path=capture_png,
                )
                capture_written = True
                if capture_exit:
                    break

            done = (terminated.item() or truncated.item())
            if done and final_capture_json is not None:
                _maybe_write_play_capture(
                    env,
                    final_capture_json,
                    task=args_cli.task,
                    step_index=timestep + 1,
                    elapsed_time=total_elapsed_time,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    action=actions,
                    capture_reason="done",
                )
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

    if trace_file is not None:
        trace_file.close()

    env.close()

    return total_episode_reward, total_elapsed_time


if __name__ == "__main__":
    score, elapsed_time = play()
    print(f"score: {score:.2f}, elapsed_time: {elapsed_time:.2f} seconds")

    # Finally, close the simulation app
    print("Closing simulation app...")
    simulation_app.close()
