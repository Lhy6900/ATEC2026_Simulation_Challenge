"""Train a high-level planner for the box-pushing task.

Usage:
    PYTHONPATH=. python scripts/train_planner.py --num_envs 64 --headless

The planner outputs [vx, vy, vyaw] at 50Hz, which is fed directly to the
GR00T low-level locomotion policy (also running at 50Hz on GPU).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from rsl_rl.runners import OnPolicyRunner

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from source.atec_rl_lab.atec_rl_lab.tasks.task_d import TaskDEnvG1Cfg  # noqa: E402
from scripts.low_level_policy import GrootLowLevelPolicy  # noqa: E402
from scripts.planner_env import PlannerEnv, PlannerRslRlWrapper  # noqa: E402


TRAIN_CFG = {
    "seed": 42,
    "device": "cuda:0",
    "num_steps_per_env": 100,  # 100 steps × 0.02s = 2s per rollout
    "max_iterations": 10000,
    "save_interval": 200,
    "experiment_name": "planner_box_push",
    "empirical_normalization": None,
    "policy": {
        "class_name": "ActorCritic",
        "init_noise_std": 1.0,
        "noise_std_type": "scalar",
        "state_dependent_std": False,
        "actor_obs_normalization": True,
        "critic_obs_normalization": True,
        "actor_hidden_dims": [256, 128, 64],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
    },
    "algorithm": {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.01,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 5e-4,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
    },
    "obs_groups": {
        "policy": ["planner_obs"],
        "critic": ["planner_obs", "privileged"],
    },
    "logger": "tensorboard",
    "neptune_project": "isaaclab",
    "wandb_project": "isaaclab",
    "resume": False,
    "load_run": ".*",
    "load_checkpoint": "model_.*.pt",
}


def main():
    parser = argparse.ArgumentParser(description="Train planner for box-pushing task")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--episode_length_s", type=int, default=60)
    args = parser.parse_args()

    # Environment config
    env_cfg = TaskDEnvG1Cfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.episode_length_s

    # Low-level policy
    low_level = GrootLowLevelPolicy(device="cuda:0")

    # Planner environment
    planner_env = PlannerEnv(env_cfg, low_level)
    env = PlannerRslRlWrapper(planner_env)

    # Training config
    train_cfg = dict(TRAIN_CFG)
    if args.max_iterations is not None:
        train_cfg["max_iterations"] = args.max_iterations

    # Log directory
    log_root = os.path.join(PROJECT_ROOT, "logs", "planner_box_push")
    os.makedirs(log_root, exist_ok=True)

    # Create runner
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_root, device="cuda:0")

    # Train
    runner.learn(
        num_learning_iterations=train_cfg["max_iterations"],
        init_at_random_ep_len=True,
    )

    # Save final model
    final_path = os.path.join(log_root, "model_final.pt")
    torch.save(runner.alg.actor_critic.state_dict(), final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main()
