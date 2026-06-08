"""Train a high-level planner for the box-pushing task.

Usage:
    PYTHONPATH=. python scripts/train_planner.py --num_envs 1024 --headless

The planner outputs [vx, vy, vyaw] at 50Hz, which is fed directly to the
GR00T low-level locomotion policy (also running at 50Hz on GPU).
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from isaaclab.app import AppLauncher

DEFAULT_EXPERIMENT_NAME = "planner_box_push"

# ── CLI (must be before AppLauncher creation) ──────────────────────────────
parser = argparse.ArgumentParser(description="Train planner for box-pushing task")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--episode_length_s", type=int, default=300)
parser.add_argument("--experiment_name", type=str, default=DEFAULT_EXPERIMENT_NAME)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force headless for training (AppLauncher provides --headless)
args_cli.headless = True
# Cameras removed from scene — no need for rendering pipeline
args_cli.enable_cameras = False

# ── Launch Isaac Sim (must happen BEFORE any IsaacLab/ATEC imports) ────────
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports AFTER simulation_app is created (IsaacLab pattern) ─────────────
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from source.atec_rl_lab.atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvG1Cfg  # noqa: E402
from scripts.low_level_policy import GrootLowLevelPolicy  # noqa: E402
from scripts.planner_env import PlannerEnv, PlannerRslRlWrapper  # noqa: E402


TRAIN_CFG = {
    "seed": 42,
    "device": "cuda:0",
    "num_steps_per_env": 100,  # 100 steps × 0.02s = 2s per rollout
    "max_iterations": 10000,
    "save_interval": 200,
    "experiment_name": DEFAULT_EXPERIMENT_NAME,
    "empirical_normalization": None,
    "policy": {
        "class_name": "ActorCritic",
        "init_noise_std": 1.0,
        "noise_std_type": "scalar",
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
        "entropy_coef": 0.0025,  #0.01
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
        "policy": ["policy"],
        "critic": ["policy", "critic"],
    },
    "logger": "tensorboard",
    "neptune_project": "isaaclab",
    "wandb_project": "isaaclab",
    "resume": False,
    "load_run": ".*",
    "load_checkpoint": "model_.*.pt",
}


TASK_D_G1_ENV_ORIGIN = (-4.2, 0.0, 0.0)
TASK_D_G1_ROBOT_DEFAULT_ROOT_POS = (-3.0, 0.0, 0.8)
TASK_D_G1_BOX_DEFAULT_ROOT_POS = (-3.0, 1.6, 0.5)

ROBOT_RESET_WORLD_X_RANGE = (-4.5, -1.5)
ROBOT_RESET_WORLD_Y_RANGE = (-0.5, 0.5)
ROBOT_RESET_WORLD_Z = 0.8
BOX_RESET_WORLD_POS = (-3.0, 1.6, 0.5)


def _world_pose_range_to_reset_offsets(
    *,
    default_pos: tuple[float, float, float],
    env_origin: tuple[float, float, float],
    target_x: tuple[float, float],
    target_y: tuple[float, float],
    target_z: float,
) -> dict[str, tuple[float, float]]:
    """Convert absolute world spawn ranges into IsaacLab reset_root_state_uniform offsets."""
    base_x = default_pos[0] + env_origin[0]
    base_y = default_pos[1] + env_origin[1]
    base_z = default_pos[2] + env_origin[2]

    return {
        "x": (round(target_x[0] - base_x, 6), round(target_x[1] - base_x, 6)),
        "y": (round(target_y[0] - base_y, 6), round(target_y[1] - base_y, 6)),
        "z": (round(target_z - base_z, 6), round(target_z - base_z, 6)),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }


def _configure_training_reset_ranges(env_cfg):
    """Keep Task D training resets on the box-side ground patch."""
    env_cfg.events.reset_robot_root.params["pose_range"] = _world_pose_range_to_reset_offsets(
        default_pos=TASK_D_G1_ROBOT_DEFAULT_ROOT_POS,
        env_origin=TASK_D_G1_ENV_ORIGIN,
        target_x=ROBOT_RESET_WORLD_X_RANGE,
        target_y=ROBOT_RESET_WORLD_Y_RANGE,
        target_z=ROBOT_RESET_WORLD_Z,
    )
    env_cfg.events.reset_box_root.params["pose_range"] = _world_pose_range_to_reset_offsets(
        default_pos=TASK_D_G1_BOX_DEFAULT_ROOT_POS,
        env_origin=TASK_D_G1_ENV_ORIGIN,
        target_x=(BOX_RESET_WORLD_POS[0], BOX_RESET_WORLD_POS[0]),
        target_y=(BOX_RESET_WORLD_POS[1], BOX_RESET_WORLD_POS[1]),
        target_z=BOX_RESET_WORLD_POS[2],
    )
    return env_cfg


class PlannerOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner with planner-specific rolling done metrics in the console."""

    _PLANNER_METRIC_LABELS = {
        "box_in_pit_success_rate": "Mean box-in-pit success rate",
        "box_entered_pit_rate": "Mean box-entered-pit rate",
        "entered_pit_xz_plane_rate": "Mean entered-pit xz-plane rate",
        "entered_pit_box_y_axis_xz_plane": "Mean entered-pit box-y-axis xz-plane",
        "robot_box_distance": "Mean robot-box distance",
        "box_pit_distance": "Mean box-pit distance",
        "box_y_axis_xz_plane": "Mean box-y-axis xz-plane",
        "box_y_axis_xz_plane_error_deg": "Mean box-y-axis xz-plane error deg",
    }
    _REWARD_COMPONENT_LABELS = {
        "box_pit": "Mean reward/box_pit",
        "approach": "Mean reward/approach",
        "robot_box": "Mean reward/robot_box",
        "box_in_pit": "Mean reward/box_in_pit",
        "box_y_axis_xz_plane": "Mean reward/box_y_axis_xz_plane",
        "box_in_pit_align_bonus": "Mean reward/box_in_pit_align_bonus",
        "stable_after_pit": "Mean reward/stable_after_pit",
        "obstacle": "Mean reward/obstacle",
        "alive": "Mean reward/alive",
        "time": "Mean reward/time",
        "action_rate": "Mean reward/action_rate",
        "nav_cmd_change": "Mean reward/nav_cmd_change",
        "total": "Mean reward/total",
    }

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width=width, pad=pad)

        summary = getattr(self.env, "get_done_metric_summary", lambda: {})()
        metric_string = ""
        if summary:
            for key, value in summary.items():
                self.writer.add_scalar(f"Planner/{key}", value, locs["it"])
            for key, label in self._PLANNER_METRIC_LABELS.items():
                if key in summary:
                    metric_string += f"{label:>{pad}} {summary[key]:.4f}\n"

        pop_reward_component_summary = getattr(self.env, "pop_reward_component_summary", lambda: {})
        reward_summary = pop_reward_component_summary()
        for key, value in reward_summary.items():
            self.writer.add_scalar(f"Reward/{key}", value, locs["it"])
        if reward_summary:
            for key, label in self._REWARD_COMPONENT_LABELS.items():
                if key in reward_summary:
                    metric_string += f"{label:>{pad}} {reward_summary[key]:.4f}\n"

        if metric_string:
            print(metric_string, end="")


def main():
    # Environment config
    env_cfg = TaskDEnvG1Cfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.episode_length_s = args_cli.episode_length_s

    # Override x-based task terms; PlannerEnv owns success/reward using box_z and target rectangle.
    env_cfg.terminations.x_reached = None
    env_cfg.rewards.achieve = None
    env_cfg.rewards.box_in_target_x = None
    # Remove joint randomization so we can isolate training-path terminations.
    env_cfg.events.reset_robot_joints = None
    env_cfg = _configure_training_reset_ranges(env_cfg)

    # Remove camera sensors — planner only uses proprio + height_scan (RayCaster)
    # This allows running WITHOUT enable_cameras, massively speeding up training
    env_cfg.scene.head_camera = None
    env_cfg.scene.ee_camera = None
    env_cfg.scene.ee_dual_camera = None
    # Remove camera observation terms to match
    env_cfg.observations.image.head_rgb = None
    env_cfg.observations.image.head_depth = None
    env_cfg.observations.image.ee_rgb = None
    env_cfg.observations.image.ee_depth = None
    env_cfg.observations.image.ee_dual_rgb = None
    env_cfg.observations.image.ee_dual_depth = None

    # Low-level policy
    low_level = GrootLowLevelPolicy(device="cuda:0")

    # Planner environment
    planner_env = PlannerEnv(env_cfg, low_level)
    env = PlannerRslRlWrapper(planner_env)

    # Training config
    train_cfg = dict(TRAIN_CFG)
    train_cfg["experiment_name"] = args_cli.experiment_name
    if args_cli.max_iterations is not None:
        train_cfg["max_iterations"] = args_cli.max_iterations

    # Log directory
    log_root = os.path.join(PROJECT_ROOT, "logs", args_cli.experiment_name)
    os.makedirs(log_root, exist_ok=True)

    # Create runner
    runner = PlannerOnPolicyRunner(env, train_cfg, log_dir=log_root, device="cuda:0")

    # Train
    runner.learn(
        num_learning_iterations=train_cfg["max_iterations"],
        init_at_random_ep_len=False,
    )

    # Save final model
    final_path = os.path.join(log_root, "model_final.pt")
    torch.save(runner.alg.policy.state_dict(), final_path)
    print(f"Saved final model to {final_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
