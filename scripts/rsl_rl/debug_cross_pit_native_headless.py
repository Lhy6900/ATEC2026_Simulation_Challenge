"""Headless logger for native CrossPitBox checkpoint rollouts."""

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

import cli_args


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log native CrossPitBox rollout diagnostics.")
    parser.add_argument("--task", type=str, default="ATEC-Isaac-TaskD-G1-CrossPitBox-v1")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=320)
    parser.add_argument("--log_path", type=str, default="logs/debug_cross_pit_native_headless.jsonl")
    parser.add_argument("--cross_pit_success_x", type=float, default=5.8)
    parser.add_argument("--print_interval", type=int, default=20)
    parser.add_argument("--disable_fabric", action="store_true", default=False)
    parser.add_argument(
        "--inject_taskd_snapshot",
        type=str,
        default=None,
        help="Optional TaskD handoff snapshot JSON to write into the native CrossPitBox env before rollout.",
    )
    parser.add_argument(
        "--inject_zero_root_velocity",
        action="store_true",
        default=False,
        help="When injecting a snapshot, zero the robot root velocity before rollout.",
    )
    parser.add_argument(
        "--inject_zero_joint_velocity",
        action="store_true",
        default=False,
        help="When injecting a snapshot, zero the robot joint velocity before rollout.",
    )
    parser.add_argument(
        "--policy_source",
        choices=("rsl_rl", "demo"),
        default="rsl_rl",
        help="Use the native RSL-RL policy or the demo CrossPitBoxPolicy wrapper for actions.",
    )
    parser.add_argument(
        "--demo_checkpoint",
        type=str,
        default="demo/cross_pit_box_model_19999.pt",
        help="Checkpoint path for --policy_source demo.",
    )
    parser.add_argument(
        "--log_pre_step",
        action="store_true",
        default=False,
        help="Also log the policy observation and action before each environment step.",
    )
    parser.add_argument(
        "--log_policy_obs_full",
        action="store_true",
        default=False,
        help="Diagnostic only: include the full flattened policy observation in each JSONL record.",
    )
    parser.add_argument(
        "--demo_true_raycast",
        action="store_true",
        default=False,
        help="When --policy_source demo, feed the native depth_scanner ray hits into the demo wrapper.",
    )
    parser.add_argument(
        "--demo_native_order_proprio",
        action="store_true",
        default=False,
        help=(
            "When --policy_source demo, build the fake TaskD proprio in the native CrossPit observation/action order. "
            "Use together with ATEC_CROSS_PIT_ACTION_OFFSET_SCALE=0 for native-env wrapper isolation."
        ),
    )
    parser.add_argument(
        "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
    )
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args, hydra_args = parser.parse_known_args()
    args.headless = True
    args.enable_cameras = False
    sys.argv = [sys.argv[0]] + hydra_args
    return args


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import atec_rl_lab.train  # noqa: F401,E402


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


def _done_terms(env) -> dict[str, bool]:
    try:
        terms = {}
        for name, value in env.unwrapped.termination_manager.get_active_iterable_terms(env_idx=0):
            if hasattr(value, "detach"):
                terms[name] = bool(value.detach().any().item())
            elif isinstance(value, (list, tuple)):
                terms[name] = any(bool(item) for item in value)
            else:
                terms[name] = bool(value)
        return terms
    except Exception:
        return {}


def _current_success_x(env) -> float:
    return float(getattr(env.unwrapped, "cross_pit_success_x", float("nan")))


def _force_success_x(env, success_x: float) -> None:
    env.unwrapped.cross_pit_success_x = float(success_x)


def _robot_state(env) -> dict[str, float]:
    robot = env.unwrapped.scene["robot"]
    root_pos = robot.data.root_pos_w[0]
    env_origins = getattr(env.unwrapped.scene, "env_origins", None)
    if env_origins is None:
        env_origin = torch.zeros(3, device=root_pos.device, dtype=root_pos.dtype)
    else:
        env_origin = env_origins[0].to(device=root_pos.device, dtype=root_pos.dtype)
    root_pos_env = root_pos - env_origin
    root_quat = robot.data.root_quat_w[0]
    root_vel = robot.data.root_vel_w[0]
    gravity = robot.data.projected_gravity_b[0]
    joint_names = list(getattr(robot.data, "joint_names", []) or getattr(robot, "joint_names", []))
    joint_pos = robot.data.joint_pos[0]
    waist_terms = {}
    for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
        if name in joint_names:
            waist_terms[name] = float(joint_pos[joint_names.index(name)].item())
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
        "root_quat_w": root_quat.detach().cpu().tolist(),
        "root_vx": float(root_vel[0].item()),
        "root_vy": float(root_vel[1].item()),
        "root_vz": float(root_vel[2].item()),
        "gravity_x": float(gravity[0].item()),
        "gravity_y": float(gravity[1].item()),
        "gravity_z": float(gravity[2].item()),
        **waist_terms,
    }


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_inv = torch.cat((quat[:, 0:1], -quat[:, 1:]), dim=-1)
    qvec = quat_inv[:, 1:]
    qw = quat_inv[:, 0:1]
    t = 2.0 * torch.cross(qvec, vec, dim=-1)
    return vec + qw * t + torch.cross(qvec, t, dim=-1)


def _sensor_state(env) -> dict[str, Any]:
    robot = env.unwrapped.scene["robot"]
    try:
        sensor = env.unwrapped.scene.sensors["depth_scanner"]
    except Exception:
        return {}

    root_pos = robot.data.root_pos_w[0:1]
    root_quat = robot.data.root_quat_w[0:1]
    sensor_pos = sensor.data.pos_w[0:1]
    sensor_quat = sensor.data.quat_w[0:1]
    rel_base = _quat_apply_inverse(root_quat, sensor_pos - root_pos)[0]

    body_record: dict[str, Any] = {}
    try:
        for requested_name in ("pelvis", "torso_link", "d435_link"):
            body_ids, body_names = robot.find_bodies(requested_name)
            if not body_ids:
                continue
            body_id = int(body_ids[0])
            body_pos = robot.data.body_pos_w[0:1, body_id]
            body_quat = robot.data.body_quat_w[0:1, body_id]
            body_rel = _quat_apply_inverse(root_quat, body_pos - root_pos)[0]
            key = requested_name.replace("_link", "")
            body_record[f"{key}_body_name"] = body_names[0]
            body_record[f"{key}_body_pos_w"] = body_pos[0].detach().cpu().tolist()
            body_record[f"{key}_body_quat_w"] = body_quat[0].detach().cpu().tolist()
            body_record[f"{key}_body_rel_base"] = body_rel.detach().cpu().tolist()
    except Exception:
        body_record = {}

    return {
        "depth_sensor_pos_w": sensor_pos[0].detach().cpu().tolist(),
        "depth_sensor_quat_w": sensor_quat[0].detach().cpu().tolist(),
        "depth_sensor_rel_base": rel_base.detach().cpu().tolist(),
        "depth_ray_hits_stats": {
            "mean_z": float(sensor.data.ray_hits_w[0, :, 2].mean().item()),
            "min_z": float(sensor.data.ray_hits_w[0, :, 2].min().item()),
            "max_z": float(sensor.data.ray_hits_w[0, :, 2].max().item()),
        },
        **body_record,
    }


def _policy_obs_tensor(obs: Any) -> torch.Tensor:
    if isinstance(obs, torch.Tensor):
        return obs
    if isinstance(obs, dict):
        if "policy" in obs:
            return obs["policy"]
        if "observations" in obs and isinstance(obs["observations"], dict) and "policy" in obs["observations"]:
            return obs["observations"]["policy"]
    try:
        return obs["policy"]
    except Exception as exc:
        raise TypeError(f"Cannot extract policy observations from {type(obs)!r}") from exc


def _heightmap_stats(obs: Any) -> dict[str, float]:
    policy_obs = _policy_obs_tensor(obs)
    heightmap = policy_obs[:, -384:]
    stats = {
        "base_ang_vel": policy_obs[0, 0:3].detach().cpu().tolist(),
        "projected_gravity": policy_obs[0, 3:6].detach().cpu().tolist(),
        "velocity_commands": policy_obs[0, 6:9].detach().cpu().tolist(),
        "joint_pos_head": policy_obs[0, 9:17].detach().cpu().tolist(),
        "joint_pos_norm": float(torch.linalg.norm(policy_obs[0, 9:42]).item()),
        "joint_vel_head": policy_obs[0, 42:50].detach().cpu().tolist(),
        "joint_vel_norm": float(torch.linalg.norm(policy_obs[0, 42:75]).item()),
        "last_action_head": policy_obs[0, 75:83].detach().cpu().tolist(),
        "last_action_norm": float(torch.linalg.norm(policy_obs[0, 75:108]).item()),
        "heightmap_mean": float(heightmap.mean().item()),
        "heightmap_min": float(heightmap.min().item()),
        "heightmap_max": float(heightmap.max().item()),
    }
    if args_cli.log_policy_obs_full:
        stats["policy_obs_full"] = policy_obs[0].detach().cpu().tolist()
    return stats


def _official_like_obs_from_native_env(
    env, *, include_debug_ray_hits: bool = False, native_order_proprio: bool = False
) -> dict[str, torch.Tensor]:
    robot = env.unwrapped.scene["robot"]
    action = env.unwrapped.action_manager.action
    num_envs = robot.data.joint_pos.shape[0]
    proprio = torch.zeros(num_envs, 111, device=robot.device, dtype=torch.float32)
    proprio[:, 0:3] = robot.data.root_lin_vel_b
    proprio[:, 3:6] = robot.data.root_ang_vel_b
    proprio[:, 9:12] = robot.data.projected_gravity_b
    if native_order_proprio:
        from demo.cross_pit_box_policy import (
            CROSS_PIT_OBSERVATION_JOINT_INDEX_IN_TASKD_ORDER,
            CROSS_PIT_OFFICIAL_JOINT_POS_OFFSET,
        )

        inverse_index = torch.empty(len(CROSS_PIT_OBSERVATION_JOINT_INDEX_IN_TASKD_ORDER), device=robot.device, dtype=torch.long)
        inverse_index[
            torch.tensor(CROSS_PIT_OBSERVATION_JOINT_INDEX_IN_TASKD_ORDER, device=robot.device, dtype=torch.long)
        ] = torch.arange(len(CROSS_PIT_OBSERVATION_JOINT_INDEX_IN_TASKD_ORDER), device=robot.device)
        official_offset = CROSS_PIT_OFFICIAL_JOINT_POS_OFFSET.to(device=robot.device, dtype=torch.float32).view(1, -1)
        proprio[:, 12:45] = (robot.data.joint_pos - robot.data.default_joint_pos).index_select(1, inverse_index)
        proprio[:, 12:45] += official_offset
        proprio[:, 45:78] = robot.data.joint_vel.index_select(1, inverse_index)
        proprio[:, 78:111] = action
    else:
        proprio[:, 12:45] = robot.data.joint_pos - robot.data.default_joint_pos
        proprio[:, 45:78] = robot.data.joint_vel
        proprio[:, 78:111] = action
    obs = {
        "proprio": proprio,
        "extero": torch.full((num_envs, 5760), float("-inf"), device=robot.device, dtype=torch.float32),
        "_debug_root_pos_w": robot.data.root_pos_w,
        "_debug_root_quat_w": robot.data.root_quat_w,
    }
    if include_debug_ray_hits:
        try:
            obs["_debug_depth_ray_hits_w"] = env.unwrapped.scene.sensors["depth_scanner"].data.ray_hits_w
        except Exception:
            pass
    return obs


def _tensor_from_list(values: Any, *, device: str | torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(values, device=device, dtype=dtype)


def _reorder_named_values(values: list[float], source_names: list[str], target_names: list[str], *, device: str | torch.device) -> torch.Tensor:
    values_by_name = {name: values[index] for index, name in enumerate(source_names)}
    return torch.tensor([values_by_name[name] for name in target_names], device=device, dtype=torch.float32)


def _inject_taskd_snapshot(env, snapshot_path: str, *, zero_root_velocity: bool, zero_joint_velocity: bool) -> None:
    snapshot = json.loads(Path(snapshot_path).read_text())
    scene = env.unwrapped.scene
    device = env.unwrapped.device
    env_ids = torch.tensor([0], device=device, dtype=torch.long)

    robot = scene["robot"]
    robot_snapshot = snapshot["robot"]
    robot_pos = _tensor_from_list(robot_snapshot["root_pos_w"], device=device).view(1, 3)
    robot_quat = _tensor_from_list(robot_snapshot["root_quat_w"], device=device).view(1, 4)
    root_velocity = torch.cat(
        (
            _tensor_from_list(robot_snapshot["root_lin_vel_w"], device=device),
            _tensor_from_list(robot_snapshot["root_ang_vel_w"], device=device),
        )
    ).view(1, 6)
    if zero_root_velocity:
        root_velocity = torch.zeros_like(root_velocity)

    source_joint_names = list(robot_snapshot.get("joint_names") or [])
    target_joint_names = list(getattr(robot.data, "joint_names", []) or getattr(robot, "joint_names", []))
    if source_joint_names and target_joint_names and source_joint_names != target_joint_names:
        joint_pos = _reorder_named_values(robot_snapshot["joint_pos"], source_joint_names, target_joint_names, device=device).view(1, -1)
        joint_vel = _reorder_named_values(robot_snapshot["joint_vel"], source_joint_names, target_joint_names, device=device).view(1, -1)
    else:
        joint_pos = _tensor_from_list(robot_snapshot["joint_pos"], device=device).view(1, -1)
        joint_vel = _tensor_from_list(robot_snapshot["joint_vel"], device=device).view(1, -1)
    if zero_joint_velocity:
        joint_vel = torch.zeros_like(joint_vel)

    robot.write_root_pose_to_sim(torch.cat((robot_pos, robot_quat), dim=-1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    if "box_support" in scene.keys() and "box" in snapshot:
        box = scene["box_support"]
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

    scene.write_data_to_sim()
    env.unwrapped.sim.forward()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg) -> int:
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.curriculum.success_x = None

    log_path = Path(args_cli.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    resume_path = None
    if args_cli.policy_source == "rsl_rl":
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        env_cfg.log_dir = os.path.dirname(resume_path)
    else:
        env_cfg.log_dir = os.path.abspath(os.path.join("logs", "debug_cross_pit_native_headless"))
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "log_dir"):
        env_cfg.sim.log_dir = env_cfg.log_dir

    requested_success_x = float(args_cli.cross_pit_success_x)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    _force_success_x(env, requested_success_x)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # RslRlVecEnvWrapper resets the env during construction. Curriculum terms can update
    # cross_pit_success_x during that reset, so pin it again after wrapping.
    _force_success_x(wrapped_env, requested_success_x)
    obs = wrapped_env.get_observations()
    if args_cli.inject_taskd_snapshot is not None:
        _inject_taskd_snapshot(
            wrapped_env,
            args_cli.inject_taskd_snapshot,
            zero_root_velocity=args_cli.inject_zero_root_velocity,
            zero_joint_velocity=args_cli.inject_zero_joint_velocity,
        )
        _force_success_x(wrapped_env, requested_success_x)
        obs = wrapped_env.get_observations()
    demo_policy = None
    if args_cli.policy_source == "rsl_rl":
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        assert resume_path is not None
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
    else:
        from demo.cross_pit_box_policy import CrossPitBoxPolicy

        demo_policy = CrossPitBoxPolicy(
            model_path=args_cli.demo_checkpoint,
            device=wrapped_env.unwrapped.device,
            target_x=args_cli.cross_pit_success_x,
            handoff_blend_steps=0,
            handoff_max_delta=0.0,
            stabilize_steps=0,
        )
        policy = None
    start_wall = time.time()
    crossed = False
    last_pre_record: dict[str, Any] | None = None
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(args_cli.max_steps):
            with torch.inference_mode():
                if args_cli.policy_source == "demo":
                    actions = demo_policy.predict(
                        _official_like_obs_from_native_env(
                            wrapped_env,
                            include_debug_ray_hits=args_cli.demo_true_raycast,
                            native_order_proprio=args_cli.demo_native_order_proprio,
                        )
                    )
                else:
                    actions = policy(obs)
                pre_record = {
                    "phase": "pre_step",
                    "step": step,
                    "cross_pit_success_x": _current_success_x(wrapped_env),
                    "robot": _robot_state(wrapped_env),
                    "action_head": actions[0, :8].detach().cpu().tolist(),
                    "action_norm": float(torch.linalg.norm(actions[0]).item()),
                    "done": False,
                    "done_terms": _done_terms(wrapped_env),
                    "obs": _heightmap_stats(obs),
                    "sensor": _sensor_state(wrapped_env),
                }
                if demo_policy is not None:
                    pre_record["solution"] = demo_policy.get_debug_snapshot()
                last_pre_record = pre_record
                if args_cli.log_pre_step:
                    log_file.write(json.dumps(pre_record, ensure_ascii=False) + "\n")
                    log_file.flush()
                obs, _, dones, _ = wrapped_env.step(actions)
            terms = _done_terms(wrapped_env)
            crossed = crossed or bool(terms.get("crossed", False))
            record = {
                "phase": "post_step",
                "step": step,
                "cross_pit_success_x": _current_success_x(wrapped_env),
                "robot": _robot_state(wrapped_env),
                "action_head": actions[0, :8].detach().cpu().tolist(),
                "action_norm": float(torch.linalg.norm(actions[0]).item()),
                "done": bool(dones.detach().any().item()) if hasattr(dones, "detach") else bool(dones),
                "done_terms": terms,
                "obs": _heightmap_stats(obs),
                "sensor": _sensor_state(wrapped_env),
            }
            if record["done"]:
                record["post_step_state_after_auto_reset"] = True
                record["terminal_state_note"] = (
                    "ManagerBasedRLEnv resets terminated envs before returning observations/state; "
                    "robot/obs/sensor fields in this post_step record are after reset."
                )
                record["terminal_pre_step"] = last_pre_record
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()
            if args_cli.print_interval > 0 and (step % args_cli.print_interval == 0 or record["done"]):
                robot = record["robot"]
                print(
                    "step={step:04d} root=({x:.3f},{y:.3f},{z:.3f}) "
                    "v=({vx:.3f},{vy:.3f},{vz:.3f}) action_norm={an:.3f} done={done} terms={terms}".format(
                        step=step,
                        x=robot["root_x"],
                        y=robot["root_y"],
                        z=robot["root_z"],
                        vx=robot["root_vx"],
                        vy=robot["root_vy"],
                        vz=robot["root_vz"],
                        an=record["action_norm"],
                        done=record["done"],
                        terms=terms,
                    )
                )
            if record["done"]:
                break

    wrapped_env.close()
    print(f"[RESULT] crossed={crossed} log_path={log_path} wall_time={time.time() - start_wall:.2f}s")
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
