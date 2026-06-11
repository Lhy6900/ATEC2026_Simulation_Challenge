"""Play a trained planner checkpoint in Isaac Sim with GUI visualization.

Example:
    PYTHONPATH=. python scripts/play_planner_checkpoint.py \
        --checkpoint logs/planner_box_push/model_5800.pt --num_envs 1
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_scalar(value: str):
    value = value.strip()
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        items = value[1:-1].strip()
        if not items:
            return []
        return [_parse_scalar(item) for item in items.split(",")]
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def _load_simple_yaml(path: Path) -> dict:
    """Tiny fallback parser for this repo's simple nested mapping config."""
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"Invalid YAML line in {path}: {raw_line}")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _load_yaml_config(path: Path) -> dict:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data or {}
    except ModuleNotFoundError:
        return _load_simple_yaml(path)


def _section(cfg: dict, name: str) -> dict:
    value = cfg.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config section {name!r} must be a mapping")
    return value


def _resolve_cfg_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _parse_vec3(section: dict, name: str, default: list[float]) -> list[float]:
    value = section.get(name, default)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-element list: [vx, vy, wz]")
    return [float(item) for item in value]


def _configure_env(args):
    from source.atec_rl_lab.atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvG1Cfg
    from scripts.eval_planner_checkpoints import _configure_training_reset_ranges

    env_cfg = TaskDEnvG1Cfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.episode_length_s

    # Match planner training/eval task ownership.
    env_cfg.terminations.x_reached = None
    env_cfg.rewards.achieve = None
    env_cfg.rewards.box_in_target_x = None
    env_cfg.events.reset_robot_joints = None
    return _configure_training_reset_ranges(env_cfg, taskd_initpos=args.taskD_initpos)


def _print_done_metrics(info: dict, step: int) -> None:
    metrics = info.get("planner_done_metrics")
    if not metrics:
        return

    def _sum(key: str) -> float:
        values = metrics.get(key)
        if values is None:
            return 0.0
        if hasattr(values, "detach"):
            return float(values.detach().float().sum().item())
        return float(values.sum())

    done_count = int(metrics["box_in_pit_success"].numel())
    success_count = int(_sum("box_in_pit_success"))
    drop_count = int(_sum("box_drop_failure"))
    native_count = int(_sum("raw_terminated"))
    timeout_count = int(_sum("truncated"))
    print(
        f"[play step={step}] done={done_count} "
        f"success={success_count} drop_failure={drop_count} "
        f"native={native_count} timeout={timeout_count}"
    )


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Play a planner checkpoint with Isaac Sim visualization.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to model_*.pt checkpoint.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of visualized environments.")
    parser.add_argument("--episode_length_s", type=int, default=300)
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum planner steps before exiting.")
    parser.add_argument("--pause_on_done", action="store_true", help="Sleep briefly whenever an env reaches done.")
    parser.add_argument("--done_pause_s", type=float, default=1.0)
    parser.add_argument("--taskD_initpos", action="store_true", help="Use TaskD's fixed robot initial xy for reset.")
    parser.add_argument(
        "--vision_loco_cfg",
        type=Path,
        default=None,
        help="Optional YAML override for planner-to-vision-locomotion handoff. Defaults to scripts/vision_loco_taskd_config.py.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error("--checkpoint is required")

    args.headless = False
    args.enable_cameras = True
    args.disable_fabric = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import torch
        from scripts.eval_planner_checkpoints import CLIP_ACTIONS, _build_policy
        from scripts.low_level_policy import GrootLowLevelPolicy
        from scripts.planner_env import PlannerEnv
        from scripts.vision_loco_policy import VisionLocoPolicy
        from scripts.vision_loco_taskd_config import DEFAULT_VISION_LOCO_TASKD_CONFIG

        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        vision_cfg_path = None
        if args.vision_loco_cfg is None:
            vision_cfg = copy.deepcopy(DEFAULT_VISION_LOCO_TASKD_CONFIG)
        else:
            vision_cfg_path = args.vision_loco_cfg.expanduser().resolve()
            if not vision_cfg_path.exists():
                raise FileNotFoundError(f"Vision locomotion config not found: {vision_cfg_path}")
            vision_cfg = _load_yaml_config(vision_cfg_path)
        startup_cfg = _section(vision_cfg, "startup_motion")
        switch_cfg = _section(vision_cfg, "switch")
        loco_cfg = _section(vision_cfg, "loco")
        ditch_cfg = _section(vision_cfg, "ditch")
        height_map_cfg = _section(vision_cfg, "height_map")
        action_compensation_cfg = _section(vision_cfg, "action_compensation")
        success_cfg = _section(vision_cfg, "success")

        loco_policy_path = _resolve_cfg_path(loco_cfg.get("policy_path", "../loco_policy/policy.pt"))
        loco_metadata_path = _resolve_cfg_path(loco_cfg.get("metadata_path", "../loco_policy/sim2sim_metadata.json"))
        if not loco_policy_path.exists():
            raise FileNotFoundError(f"Vision locomotion policy not found: {loco_policy_path}")
        if not loco_metadata_path.exists():
            raise FileNotFoundError(f"Vision locomotion metadata not found: {loco_metadata_path}")
        if args.num_envs != 1:
            print("[play] Warning: planner->vision-loco switching is intended for --num_envs 1.")

        env_cfg = _configure_env(args)
        ditch_width = ditch_cfg.get("width")
        if ditch_width is None:
            ditch_width = float(env_cfg.pit_width_range[0])
        ditch_center_xy = ditch_cfg.get("center_xy")
        low_level = GrootLowLevelPolicy(device="cuda:0")
        env = PlannerEnv(env_cfg, low_level)
        policy = _build_policy(env, str(checkpoint), env.device)
        vision_loco = VisionLocoPolicy(
            loco_policy_path,
            loco_metadata_path,
            device=env.device,
            forward_velocity=float(loco_cfg.get("forward_velocity", 0.8)),
            yaw_kp=float(loco_cfg.get("yaw_kp", 1.0)),
            ditch_center_xy=ditch_center_xy,
            ditch_yaw=math.radians(float(ditch_cfg.get("yaw_deg", 90.0))),
            ditch_width=float(ditch_width),
            ditch_length=float(ditch_cfg.get("length", 7.0)),
            ditch_depth=float(ditch_cfg.get("depth", 1.0)),
            height_map_cfg=height_map_cfg,
            action_compensation_cfg=action_compensation_cfg,
        )
        clip_actions = torch.tensor(CLIP_ACTIONS, device=env.device)

        obs_dict, _ = env.reset()
        print(f"[play] Loaded checkpoint: {checkpoint}")
        print(f"[play] Loaded vision locomotion policy: {loco_policy_path}")
        if vision_cfg_path is None:
            print("[play] Loaded built-in vision locomotion config: scripts/vision_loco_taskd_config.py")
        else:
            print(f"[play] Loaded vision locomotion config override: {vision_cfg_path}")
        print("[play] Control mode: planner until box-in-ditch, then synthetic-height-map vision locomotion.")
        print("[play] Close the Isaac Sim window or wait for --max_steps to exit.")

        step = 0
        box_in_pit_hold_steps = 0
        using_vision_loco = False
        crossed_printed = False
        height_map_printed = False
        vision_loco_steps = 0
        startup_enabled = bool(startup_cfg.get("enabled", False))
        startup_stop_on_done = bool(startup_cfg.get("stop_on_done", False))
        startup_back_steps = max(0, int(startup_cfg.get("back_steps", 0)))
        startup_left_steps = max(0, int(startup_cfg.get("left_steps", 0)))
        startup_forward_steps = max(0, int(startup_cfg.get("forward_steps", 0)))
        startup_back_cmd = torch.tensor(
            _parse_vec3(startup_cfg, "back_command", [-0.4, 0.0, 0.0]),
            dtype=torch.float32,
            device=env.device,
        )
        startup_left_cmd = torch.tensor(
            _parse_vec3(startup_cfg, "left_command", [0.0, 0.3, 0.0]),
            dtype=torch.float32,
            device=env.device,
        )
        startup_forward_cmd = torch.tensor(
            _parse_vec3(startup_cfg, "forward_command", [0.4, 0.0, 0.0]),
            dtype=torch.float32,
            device=env.device,
        )
        switch_box_hold_steps = int(switch_cfg.get("box_in_pit_hold_steps", 5))
        start_with_vision_loco = bool(switch_cfg.get("start_with_vision_loco", False))
        start_with_vision_loco_settle_steps = max(0, int(switch_cfg.get("start_with_vision_loco_settle_steps", 0)))
        settle_steps_after_switch = max(0, int(switch_cfg.get("settle_steps_after_switch", 0)))
        settle_command = torch.tensor(
            _parse_vec3(switch_cfg, "settle_command", [0.0, 0.0, 0.0]),
            dtype=torch.float32,
            device=env.device,
        ).unsqueeze(0).expand(env.num_envs, -1)
        settle_steps_remaining = start_with_vision_loco_settle_steps if start_with_vision_loco else 0
        using_vision_loco = start_with_vision_loco
        settling_with_low_level = False
        if start_with_vision_loco:
            vision_loco.reset(env.num_envs)
            print(
                "[play] start_with_vision_loco=true: skipping startup motion and box-pushing planner; "
                f"settle_steps={settle_steps_remaining}."
            )
        no_height_map_print = bool(height_map_cfg.get("no_print", False))
        height_map_print_interval = int(height_map_cfg.get("print_interval", 0))
        height_map_print_window = int(height_map_cfg.get("print_window", 9))
        cross_x_threshold = float(success_cfg.get("cross_x_threshold", 3.5))
        raw_obs = env._current_obs
        with torch.no_grad():
            if startup_enabled and not start_with_vision_loco:
                startup_plan = (
                    ("back", startup_back_steps, startup_back_cmd),
                    ("left", startup_left_steps, startup_left_cmd),
                    ("forward", startup_forward_steps, startup_forward_cmd),
                )
                print(
                    "[play] Startup motion enabled: "
                    f"back={startup_back_steps}, left={startup_left_steps}, forward={startup_forward_steps} steps, "
                    f"stop_on_done={startup_stop_on_done}."
                )
                startup_done = False
                for phase_name, phase_steps, phase_cmd in startup_plan:
                    if phase_steps <= 0 or startup_done:
                        continue
                    nav_cmd = phase_cmd.unsqueeze(0).expand(env.num_envs, -1)
                    phase_executed_steps = 0
                    print(
                        f"[play step={step}] Startup phase {phase_name} begin: "
                        f"steps={phase_steps}, cmd={phase_cmd.detach().cpu().tolist()}"
                    )
                    for _ in range(phase_steps):
                        if step >= args.max_steps or not simulation_app.is_running():
                            break
                        low_action = env.low_level.predict(raw_obs, nav_cmd)
                        raw_obs, _, terminated, truncated, info = env.env.step(low_action)
                        env._current_obs = raw_obs
                        env.unwrapped.sim.render()
                        step += 1
                        phase_executed_steps += 1
                        done = terminated | truncated
                        if done.any():
                            print(
                                f"[play step={step}] raw env done during startup {phase_name}: "
                                f"terminated={terminated.detach().cpu().tolist()} "
                                f"truncated={truncated.detach().cpu().tolist()}"
                            )
                            if startup_stop_on_done:
                                startup_done = True
                                break
                    print(
                        f"[play step={step}] Startup phase {phase_name} end: "
                        f"executed_steps={phase_executed_steps}/{phase_steps}"
                    )
                    if step >= args.max_steps or not simulation_app.is_running():
                        break
                env._prev_box_pos_local = env._box_pos_local()
                env._prev_box_pos_w = env.unwrapped.scene["box"].data.root_pos_w.detach().clone()
                obs_dict = env._build_planner_obs(raw_obs)

            while simulation_app.is_running() and step < args.max_steps:
                stepped_with_low_level_settle = False
                if settling_with_low_level:
                    stepped_with_low_level_settle = True
                    low_action = env.low_level.predict(raw_obs, settle_command)
                    raw_obs, _, terminated, truncated, info = env.env.step(low_action)
                    env._current_obs = raw_obs
                    settle_steps_remaining -= 1
                    if settle_steps_remaining <= 0:
                        settling_with_low_level = False
                        using_vision_loco = True
                        vision_loco.reset(env.num_envs)
                        robot_xy = env.unwrapped.scene["robot"].data.root_pos_w[:, :2]
                        print(
                            f"[play step={step}] Low-level settle complete; switching to vision locomotion; "
                            f"robot_xy={robot_xy[0].detach().cpu().tolist()}"
                        )
                elif using_vision_loco:
                    if settle_steps_remaining > 0:
                        loco_action = vision_loco.predict(env, raw_obs, command_override=settle_command)
                        raw_obs, _, terminated, truncated, info = env.env.step(loco_action)
                        env._current_obs = raw_obs
                        settle_steps_remaining -= 1
                    else:
                        should_print_height_map = (
                            not no_height_map_print
                            and (
                                not height_map_printed
                                or (
                                    height_map_print_interval > 0
                                    and vision_loco_steps % height_map_print_interval == 0
                                )
                            )
                        )
                        if should_print_height_map:
                            vision_loco.print_synthetic_elevation_map(
                                env,
                                step=step,
                                env_id=0,
                                window=height_map_print_window,
                            )
                            height_map_printed = True
                        loco_action = vision_loco.predict(env, raw_obs)
                        raw_obs, _, terminated, truncated, info = env.env.step(loco_action)
                        env._current_obs = raw_obs
                        vision_loco_steps += 1
                else:
                    planner_action = policy.act_inference(obs_dict)
                    planner_action = torch.clamp(planner_action, -clip_actions, clip_actions)
                    obs_dict, _, terminated, truncated, info = env.step(planner_action)
                    raw_obs = env._current_obs

                    box_in_pit = env._check_box_in_pit()
                    if bool(box_in_pit.all().item()):
                        box_in_pit_hold_steps += 1
                    else:
                        box_in_pit_hold_steps = 0
                    if box_in_pit_hold_steps >= max(1, switch_box_hold_steps):
                        settle_steps_remaining = settle_steps_after_switch
                        robot_xy = env.unwrapped.scene["robot"].data.root_pos_w[:, :2]
                        if settle_steps_remaining > 0:
                            settling_with_low_level = True
                            print(
                                f"[play step={step}] Box in pit; settling with box-pushing low-level policy; "
                                f"robot_xy={robot_xy[0].detach().cpu().tolist()} "
                                f"settle_steps={settle_steps_remaining} "
                                f"settle_command={settle_command[0].detach().cpu().tolist()}"
                            )
                        else:
                            using_vision_loco = True
                            vision_loco.reset(env.num_envs)
                            print(
                                f"[play step={step}] Switching to vision locomotion; "
                                f"robot_xy={robot_xy[0].detach().cpu().tolist()} "
                                f"settle_steps={settle_steps_remaining}"
                            )

                done = terminated | truncated
                if done.any():
                    if stepped_with_low_level_settle:
                        print(
                            f"[play step={step}] raw env done during low-level settle: "
                            f"terminated={terminated.detach().cpu().tolist()} "
                            f"truncated={truncated.detach().cpu().tolist()}"
                        )
                    elif using_vision_loco:
                        print(
                            f"[play step={step}] raw env done during vision locomotion: "
                            f"terminated={terminated.detach().cpu().tolist()} "
                            f"truncated={truncated.detach().cpu().tolist()}"
                        )
                        vision_loco.reset(env.num_envs)
                    else:
                        _print_done_metrics(info, step)
                    if args.pause_on_done:
                        time.sleep(args.done_pause_s)

                if using_vision_loco and not crossed_printed:
                    robot_x = env.unwrapped.scene["robot"].data.root_pos_w[:, 0]
                    if bool((robot_x > cross_x_threshold).any().item()):
                        print(f"[play step={step}] Robot crossed x>{cross_x_threshold:.2f}.")
                        crossed_printed = True

                env.unwrapped.sim.render()
                step += 1

        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
