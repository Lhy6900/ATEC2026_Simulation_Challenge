"""Headless diagnostic runner for TaskD push-then-cross handoff tuning."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "0")
os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TaskD push-then-cross headlessly and log handoff diagnostics.")
    parser.add_argument("--task", type=str, default="ATEC-TaskD-G1")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--log_path", type=str, default="logs/debug_push_then_cross_headless.jsonl")
    parser.add_argument("--trace_start_s", type=float, default=12.0)
    parser.add_argument("--trace_end_s", type=float, default=18.0)
    parser.add_argument("--print_interval", type=int, default=25)
    parser.add_argument(
        "--ignore_done_until_s",
        type=float,
        default=0.0,
        help="Keep stepping through done signals until this elapsed time for handoff diagnostics.",
    )
    parser.add_argument(
        "--inject_handoff_snapshot",
        type=str,
        default=None,
        help="Optional captured TaskD handoff JSON. If set, write robot/box state after reset and start at handoff.",
    )
    parser.add_argument(
        "--inject_elapsed_s",
        type=float,
        default=None,
        help="Elapsed time to use after injecting a snapshot. Defaults to snapshot elapsed_time_s, then handoff time.",
    )
    parser.add_argument(
        "--attach_debug_scene_pose",
        action="store_true",
        default=False,
        help="Attach scene root pose to observations for terrain-map heightmap diagnostics.",
    )
    parser.add_argument(
        "--attach_cross_pit_depth_scanner",
        action="store_true",
        default=False,
        help="Attach a debug-only v1 CrossPitBox d435 raycaster and pass its ray hits to the policy wrapper.",
    )
    parser.add_argument(
        "--freeze_box_after_inject",
        action="store_true",
        default=False,
        help="Diagnostic only: keep the TaskD box at the injected pose during rollout.",
    )
    parser.add_argument(
        "--disable_cameras",
        action="store_true",
        default=False,
        help="Disable TaskD image observations for faster local handoff diagnostics.",
    )
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True
    return args


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.sensors import MultiMeshRayCasterCfg, patterns  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec  # noqa: E402
from demo.solution import AlgSolution  # noqa: E402


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "mean"):
        value = value.mean()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _tensor_stats(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.float()
    return {
        "mean": float(tensor.mean().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "norm": float(torch.linalg.norm(tensor.reshape(tensor.shape[0], -1), dim=-1).mean().item())
        if tensor.ndim > 1
        else float(torch.linalg.norm(tensor).item()),
    }


def _image_stats(image_obs: Any) -> dict[str, Any]:
    if not isinstance(image_obs, dict):
        return {}
    stats: dict[str, Any] = {}
    for name, value in image_obs.items():
        if value is None:
            continue
        tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        term_stats = _tensor_stats(tensor)
        term_stats["shape"] = list(tensor.shape)
        stats[str(name)] = term_stats
    return stats


def _done_terms(env: gym.Env) -> dict[str, bool]:
    try:
        termination_manager = env.unwrapped.termination_manager
        terms = {}
        for name, value in termination_manager.get_active_iterable_terms(env_idx=0):
            if hasattr(value, "detach"):
                active = bool(value.detach().any().item())
            elif isinstance(value, (list, tuple)):
                active = any(bool(v) for v in value)
            else:
                active = bool(value)
            terms[name] = active
        return terms
    except Exception:
        return {}


def _robot_state(env: gym.Env) -> dict[str, float]:
    try:
        robot = env.unwrapped.scene["robot"]
        root_pos = robot.data.root_pos_w[0]
        env_origins = getattr(env.unwrapped.scene, "env_origins", None)
        if env_origins is None:
            env_origin = torch.zeros(3, device=root_pos.device, dtype=root_pos.dtype)
        else:
            env_origin = env_origins[0].to(device=root_pos.device, dtype=root_pos.dtype)
        root_pos_env = root_pos - env_origin
        root_vel = robot.data.root_vel_w[0]
        gravity = robot.data.projected_gravity_b[0]
        return {
            "root_x": float(root_pos[0].item()),
            "root_y": float(root_pos[1].item()),
            "root_z": float(root_pos[2].item()),
            "env_origin_x": float(env_origin[0].item()),
            "env_origin_y": float(env_origin[1].item()),
            "env_origin_z": float(env_origin[2].item()),
            "root_x_env": float(root_pos_env[0].item()),
            "root_y_env": float(root_pos_env[1].item()),
            "root_z_env": float(root_pos_env[2].item()),
            "root_vx": float(root_vel[0].item()),
            "root_vy": float(root_vel[1].item()),
            "root_vz": float(root_vel[2].item()),
            "gravity_x": float(gravity[0].item()),
            "gravity_y": float(gravity[1].item()),
            "gravity_z": float(gravity[2].item()),
        }
    except Exception:
        return {}


def _rigid_object_state(env: gym.Env, name: str) -> dict[str, float]:
    try:
        asset = env.unwrapped.scene[name]
        pos = asset.data.root_pos_w[0]
        quat = asset.data.root_quat_w[0]
        vel = asset.data.root_vel_w[0]
        return {
            "x": float(pos[0].item()),
            "y": float(pos[1].item()),
            "z": float(pos[2].item()),
            "qw": float(quat[0].item()),
            "qx": float(quat[1].item()),
            "qy": float(quat[2].item()),
            "qz": float(quat[3].item()),
            "vx": float(vel[0].item()),
            "vy": float(vel[1].item()),
            "vz": float(vel[2].item()),
        }
    except Exception:
        return {}


def _obs_stats(obs: dict[str, Any]) -> dict[str, Any]:
    proprio = obs.get("proprio")
    extero = obs.get("extero")
    image = obs.get("image")
    stats: dict[str, Any] = {}
    if proprio is not None:
        p = proprio.detach().float() if isinstance(proprio, torch.Tensor) else torch.as_tensor(proprio).float()
        stats["base_lin_vel"] = p[0, 0:3].detach().cpu().tolist()
        stats["base_ang_vel"] = p[0, 3:6].detach().cpu().tolist()
        stats["projected_gravity"] = p[0, 9:12].detach().cpu().tolist()
        stats["joint_pos_norm"] = float(torch.linalg.norm(p[0, 12:45]).item())
        stats["joint_vel_norm"] = float(torch.linalg.norm(p[0, 45:78]).item())
        stats["last_action_norm"] = float(torch.linalg.norm(p[0, 78:111]).item())
        stats["last_action_head"] = p[0, 78:86].detach().cpu().tolist()
    if extero is not None:
        stats["extero"] = _tensor_stats(extero)
    image_stats = _image_stats(image)
    if image_stats:
        stats["image"] = image_stats
    return stats


def _camera_sensor_stats(env: gym.Env) -> dict[str, Any]:
    try:
        camera = env.unwrapped.scene.sensors.get("head_camera")
    except Exception:
        return {}
    if camera is None:
        return {}
    try:
        data = camera.data
        stats: dict[str, Any] = {
            "pos_w": data.pos_w[0].detach().cpu().tolist(),
            "quat_w_world": data.quat_w_world[0].detach().cpu().tolist(),
            "intrinsic_matrix": data.intrinsic_matrices[0].detach().cpu().tolist(),
            "image_shape": list(data.image_shape) if data.image_shape is not None else None,
            "output_keys": sorted(str(key) for key in (data.output or {}).keys()),
        }
        if data.output and "depth" in data.output:
            depth = data.output["depth"]
            stats["depth"] = _tensor_stats(depth)
            stats["depth"]["shape"] = list(depth.shape)
            try:
                from atec_rl_lab.train.cross_pit_box.mdp.observations import depth_image_to_heightmap

                robot = env.unwrapped.scene["robot"]
                heightmap = depth_image_to_heightmap(
                    depth,
                    data.intrinsic_matrices,
                    data.pos_w,
                    data.quat_w_world,
                    robot.data.root_pos_w,
                    robot.data.root_quat_w,
                    grid_shape=(24, 16),
                    x_range=(0.15, 2.45),
                    y_range=(-0.75, 0.75),
                    z_clip=(-1.0, 1.0),
                    default_height=-1.0,
                    sample_stride=int(os.environ.get("ATEC_DEBUG_HEAD_DEPTH_SAMPLE_STRIDE", "4")),
                )
                stats["projected_heightmap"] = _tensor_stats(heightmap)
                stats["projected_heightmap"]["shape"] = list(heightmap.shape)
                stats["projected_heightmap"]["valid_cells"] = int((heightmap[0] > -0.999).sum().item())
            except Exception as exc:
                stats["projected_heightmap"] = {"error": repr(exc)}
        return stats
    except Exception as exc:
        return {"error": repr(exc)}


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


def _attach_debug_scene_pose(env: gym.Env, obs: dict[str, Any]) -> dict[str, Any]:
    """Attach scene-only pose fields for local diagnostics without changing official observations."""
    try:
        robot = env.unwrapped.scene["robot"]
        obs = dict(obs)
        obs["_debug_root_pos_w"] = robot.data.root_pos_w.detach().clone()
        obs["_debug_root_quat_w"] = robot.data.root_quat_w.detach().clone()
        depth_scanner = env.unwrapped.scene.sensors.get("depth_scanner")
        if depth_scanner is not None:
            obs["_debug_depth_ray_hits_w"] = depth_scanner.data.ray_hits_w.detach().clone()
        camera = env.unwrapped.scene.sensors.get("head_camera")
        if camera is not None:
            obs["_debug_head_camera_intrinsics"] = camera.data.intrinsic_matrices.detach().clone()
            obs["_debug_head_camera_pos_w"] = camera.data.pos_w.detach().clone()
            obs["_debug_head_camera_quat_w"] = camera.data.quat_w_world.detach().clone()
    except Exception:
        pass
    return obs


def _tensor_from_list(values: Any, *, device: str | torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=dtype)


def _reorder_named_values(
    values: list[float],
    source_names: list[str],
    target_names: list[str],
    *,
    device: str | torch.device,
) -> torch.Tensor:
    values_by_name = {name: values[index] for index, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in values_by_name]
    if missing:
        raise ValueError(f"Snapshot is missing joint values for current robot joints: {missing}")
    return torch.tensor([values_by_name[name] for name in target_names], device=device, dtype=torch.float32)


def _inject_handoff_snapshot(env: gym.Env, snapshot_path: str) -> None:
    """Write a captured TaskD handoff state into the current single-env scene."""
    snapshot = json.loads(Path(snapshot_path).read_text())
    scene = env.unwrapped.scene
    device = env.unwrapped.device
    env_ids = torch.tensor([0], device=device, dtype=torch.long)

    robot = scene["robot"]
    robot_snapshot = snapshot["robot"]
    robot_pos = _tensor_from_list(robot_snapshot["root_pos_w"], device=device).view(1, 3)
    robot_quat = _tensor_from_list(robot_snapshot["root_quat_w"], device=device).view(1, 4)
    robot_vel = torch.cat(
        (
            _tensor_from_list(robot_snapshot["root_lin_vel_w"], device=device),
            _tensor_from_list(robot_snapshot["root_ang_vel_w"], device=device),
        )
    ).view(1, 6)
    source_joint_names = list(robot_snapshot.get("joint_names") or [])
    target_joint_names = list(getattr(robot.data, "joint_names", []) or getattr(robot, "joint_names", []))
    if source_joint_names and target_joint_names and source_joint_names != target_joint_names:
        robot_joint_pos = _reorder_named_values(
            robot_snapshot["joint_pos"], source_joint_names, target_joint_names, device=device
        ).view(1, -1)
        robot_joint_vel = _reorder_named_values(
            robot_snapshot["joint_vel"], source_joint_names, target_joint_names, device=device
        ).view(1, -1)
    else:
        robot_joint_pos = _tensor_from_list(robot_snapshot["joint_pos"], device=device).view(1, -1)
        robot_joint_vel = _tensor_from_list(robot_snapshot["joint_vel"], device=device).view(1, -1)
    robot.write_root_pose_to_sim(torch.cat((robot_pos, robot_quat), dim=-1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(robot_vel, env_ids=env_ids)
    robot.write_joint_state_to_sim(robot_joint_pos, robot_joint_vel, env_ids=env_ids)

    box = scene["box"]
    box_snapshot = snapshot["box"]
    box_pos = _tensor_from_list(box_snapshot["root_pos_w"], device=device).view(1, 3)
    box_quat = _tensor_from_list(box_snapshot["root_quat_w"], device=device).view(1, 4)
    box_vel = torch.cat(
        (
            _tensor_from_list(box_snapshot["root_lin_vel_w"], device=device),
            _tensor_from_list(box_snapshot["root_ang_vel_w"], device=device),
        )
    ).view(1, 6)
    box.write_root_pose_to_sim(torch.cat((box_pos, box_quat), dim=-1), env_ids=env_ids)
    box.write_root_velocity_to_sim(box_vel, env_ids=env_ids)

    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()


def _freeze_box_from_snapshot(env: gym.Env, snapshot_path: str) -> None:
    """Keep TaskD's dynamic box fixed for local diagnosis of training-vs-TaskD dynamics gaps."""
    snapshot = json.loads(Path(snapshot_path).read_text())
    scene = env.unwrapped.scene
    device = env.unwrapped.device
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    box = scene["box"]
    box_snapshot = snapshot["box"]
    box_pos = _tensor_from_list(box_snapshot["root_pos_w"], device=device).view(1, 3)
    box_quat = _tensor_from_list(box_snapshot["root_quat_w"], device=device).view(1, 4)
    box_vel = torch.zeros((1, 6), device=device)
    box.write_root_pose_to_sim(torch.cat((box_pos, box_quat), dim=-1), env_ids=env_ids)
    box.write_root_velocity_to_sim(box_vel, env_ids=env_ids)


def _frame_record(
    env: gym.Env,
    obs: dict[str, Any],
    *,
    step: int,
    elapsed_time: float,
    solution_snapshot: dict[str, Any],
    reward_value: float,
    total_episode_reward: float,
    terminated: Any,
    truncated: Any,
    action: torch.Tensor | None,
) -> dict[str, Any]:
    terminated_value = bool(terminated.detach().any().item()) if hasattr(terminated, "detach") else bool(terminated)
    truncated_value = bool(truncated.detach().any().item()) if hasattr(truncated, "detach") else bool(truncated)
    record = {
        "step": step,
        "env_elapsed": elapsed_time,
        "solution": solution_snapshot,
        "reward": reward_value,
        "total_episode_reward": total_episode_reward,
        "terminated": terminated_value,
        "truncated": truncated_value,
        "done_terms": _done_terms(env),
        "robot": _robot_state(env),
        "box": _rigid_object_state(env, "box"),
        "obs": _obs_stats(obs),
        "head_camera": _camera_sensor_stats(env),
    }
    if action is not None:
        record["action"] = _tensor_stats(action)
        record["action_head"] = action[0, :8].detach().cpu().tolist()
    return record


def main() -> int:
    print("[DEBUG_RUNNER] entering main", flush=True)
    log_path = Path(args_cli.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("[DEBUG_RUNNER] constructing solution", flush=True)
    solution = AlgSolution()
    action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None

    print("[DEBUG_RUNNER] parsing env cfg", flush=True)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.disable_cameras:
        env_cfg.observations.image = None
        env_cfg.scene.head_camera = None
        env_cfg.scene.ee_camera = None
        env_cfg.scene.ee_dual_camera = None
    if args_cli.attach_cross_pit_depth_scanner:
        env_cfg.scene.depth_scanner = _make_cross_pit_depth_scanner_cfg()
    env_cfg = apply_safe_action_spec(env_cfg, json.dumps(action_spec or {}))
    print("[DEBUG_RUNNER] making env", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    print("[DEBUG_RUNNER] resetting env", flush=True)
    obs, _ = env.reset()
    if hasattr(solution, "reset"):
        if args_cli.inject_handoff_snapshot is not None:
            print(f"[DEBUG_RUNNER] injecting handoff snapshot: {args_cli.inject_handoff_snapshot}", flush=True)
            snapshot_data = json.loads(Path(args_cli.inject_handoff_snapshot).read_text())
            injected_elapsed = (
                args_cli.inject_elapsed_s
                if args_cli.inject_elapsed_s is not None
                else float(snapshot_data.get("elapsed_time_s", os.environ.get("ATEC_CROSS_PIT_HANDOFF_TIME", "13.0")))
            )
            _inject_handoff_snapshot(env, args_cli.inject_handoff_snapshot)
            obs = env.unwrapped.observation_manager.compute()
            solution.reset(elapsed_time=injected_elapsed)
            total_elapsed_time = injected_elapsed
        else:
            solution.reset()
    print("[DEBUG_RUNNER] starting loop", flush=True)

    total_episode_reward = 0.0
    total_elapsed_time = locals().get("total_elapsed_time", 0.0)
    use_synthetic_elapsed = args_cli.inject_handoff_snapshot is not None
    crossed = False
    start_wall = time.time()

    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(args_cli.max_steps):
            with torch.inference_mode():
                if args_cli.freeze_box_after_inject and args_cli.inject_handoff_snapshot is not None:
                    _freeze_box_from_snapshot(env, args_cli.inject_handoff_snapshot)
                obs_for_policy = _attach_debug_scene_pose(env, obs) if args_cli.attach_debug_scene_pose else obs
                pre_snapshot = solution.get_debug_snapshot() if hasattr(solution, "get_debug_snapshot") else {}
                pre_record = _frame_record(
                    env,
                    obs,
                    step=step,
                    elapsed_time=total_elapsed_time,
                    solution_snapshot=pre_snapshot,
                    reward_value=0.0,
                    total_episode_reward=total_episode_reward,
                    terminated=False,
                    truncated=False,
                    action=None,
                )
                resp = solution.predicts(obs_for_policy, total_episode_reward)
                action = torch.tensor(resp["action"], dtype=torch.float32, device=args_cli.device).view(args_cli.num_envs, -1)
                obs, reward, terminated, truncated, info = env.step(action)

            sim_dt = _to_float(info.get("Step_dt"), 0.02)
            reward_value = _to_float(reward)
            total_episode_reward += reward_value / sim_dt
            if use_synthetic_elapsed:
                total_elapsed_time += sim_dt
            elif isinstance(info, dict) and "Elapsed_Time" in info:
                total_elapsed_time = _to_float(info["Elapsed_Time"])
            else:
                total_elapsed_time += sim_dt

            raw_done = bool(terminated.detach().any().item() or truncated.detach().any().item())
            terms = _done_terms(env)
            crossed = crossed or bool(terms.get("x_reached", False))
            snapshot = solution.get_debug_snapshot() if hasattr(solution, "get_debug_snapshot") else {}
            done = raw_done and total_elapsed_time >= args_cli.ignore_done_until_s
            record = _frame_record(
                env,
                obs,
                step=step,
                elapsed_time=total_elapsed_time,
                solution_snapshot=snapshot,
                reward_value=reward_value,
                total_episode_reward=total_episode_reward,
                terminated=terminated,
                truncated=truncated,
                action=action,
            )
            record["raw_done"] = raw_done
            record["ignored_done"] = bool(raw_done and not done)
            should_trace = args_cli.trace_start_s <= total_elapsed_time <= args_cli.trace_end_s
            pre_should_trace = args_cli.trace_start_s <= pre_record["env_elapsed"] <= args_cli.trace_end_s
            should_print = raw_done or should_trace or (args_cli.print_interval > 0 and step % args_cli.print_interval == 0)
            if pre_should_trace:
                pre_record["phase"] = "pre_step"
                log_file.write(json.dumps(pre_record, ensure_ascii=False) + "\n")
                log_file.flush()
            if should_trace or done:
                record["phase"] = "post_step"
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()
            if should_print:
                robot = record["robot"]
                print(
                    "step={step:04d} env_t={time:.2f} stage={stage} "
                    "root=({x:.3f},{y:.3f},{z:.3f}) env=({ex:.3f},{ey:.3f},{ez:.3f}) "
                    "v=({vx:.3f},{vy:.3f},{vz:.3f}) action_norm={an:.3f} done={done} terms={terms}".format(
                        step=step,
                        time=total_elapsed_time,
                        stage=snapshot.get("stage"),
                        x=robot.get("root_x", float("nan")),
                        y=robot.get("root_y", float("nan")),
                        z=robot.get("root_z", float("nan")),
                        ex=robot.get("root_x_env", float("nan")),
                        ey=robot.get("root_y_env", float("nan")),
                        ez=robot.get("root_z_env", float("nan")),
                        vx=robot.get("root_vx", float("nan")),
                        vy=robot.get("root_vy", float("nan")),
                        vz=robot.get("root_vz", float("nan")),
                        an=record["action"].get("norm", float("nan")),
                        done=raw_done,
                        terms=terms,
                    )
                )
            if done:
                break

    env.close()
    elapsed_wall = time.time() - start_wall
    print(f"[RESULT] crossed={crossed} env_elapsed={total_elapsed_time:.3f} reward={total_episode_reward:.3f}")
    print(f"[RESULT] log_path={log_path} wall_time={elapsed_wall:.2f}s")
    return 0 if crossed else 2


if __name__ == "__main__":
    try:
        try:
            exit_code = main()
        except Exception:
            traceback.print_exc()
            raise
        raise SystemExit(exit_code)
    finally:
        simulation_app.close()
