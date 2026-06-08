"""Play a trained planner checkpoint in Isaac Sim with GUI visualization.

Example:
    PYTHONPATH=. python scripts/play_planner_checkpoint.py \
        --checkpoint logs/planner_box_push/model_5800.pt --num_envs 1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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

        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        env_cfg = _configure_env(args)
        low_level = GrootLowLevelPolicy(device="cuda:0")
        env = PlannerEnv(env_cfg, low_level)
        policy = _build_policy(env, str(checkpoint), env.device)
        clip_actions = torch.tensor(CLIP_ACTIONS, device=env.device)

        obs_dict, _ = env.reset()
        print(f"[play] Loaded checkpoint: {checkpoint}")
        print("[play] Close the Isaac Sim window or wait for --max_steps to exit.")

        step = 0
        with torch.no_grad():
            while simulation_app.is_running() and step < args.max_steps:
                planner_action = policy.act_inference(obs_dict)
                planner_action = torch.clamp(planner_action, -clip_actions, clip_actions)
                obs_dict, _, terminated, truncated, info = env.step(planner_action)

                done = terminated | truncated
                if done.any():
                    _print_done_metrics(info, step)
                    if args.pause_on_done:
                        time.sleep(args.done_pause_s)

                env.unwrapped.sim.render()
                step += 1

        env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
