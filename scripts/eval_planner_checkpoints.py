"""Evaluate planner checkpoints by replaying them in PlannerEnv headlessly.

Example:
    PYTHONPATH=. python scripts/eval_planner_checkpoints.py --checkpoint_steps 5800 8800 9000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TASK_D_G1_ENV_ORIGIN = (-4.2, 0.0, 0.0)
TASK_D_G1_ROBOT_DEFAULT_ROOT_POS = (-3.0, 0.0, 0.8)
TASK_D_G1_BOX_DEFAULT_ROOT_POS = (-3.0, 1.6, 0.5)

ROBOT_RESET_WORLD_X_RANGE = (-4.5, -1.5)
ROBOT_RESET_WORLD_Y_RANGE = (-0.5, 0.5)
ROBOT_RESET_WORLD_Z = 0.8
BOX_RESET_WORLD_POS = (-3.0, 1.6, 0.5)

POLICY_CFG = {
    "class_name": "ActorCritic",
    "init_noise_std": 1.0,
    "noise_std_type": "scalar",
    "actor_obs_normalization": True,
    "critic_obs_normalization": True,
    "actor_hidden_dims": [256, 128, 64],
    "critic_hidden_dims": [512, 256, 128],
    "activation": "elu",
}
OBS_GROUPS = {
    "policy": ["policy"],
    "critic": ["policy", "critic"],
}
CLIP_ACTIONS = (0.5, 0.3, 0.3)
PLANNER_NUM_ACTIONS = 3


class DoneMetricAccumulator:
    """Accumulate planner done metrics over an evaluation run."""

    _TRACKED_KEYS = (
        "box_in_pit_success",
        "box_drop_failure",
        "robot_box_distance",
        "box_pit_distance",
        "raw_terminated",
        "truncated",
        "planner_success_reset",
        "planner_done_reset",
        "box_xy_in_pit_rect",
        "box_z_success_and_xy_in_rect",
        "box_z",
        "box_entered_pit",
        "box_entered_pit_xz_plane_good",
        "box_y_axis_xz_plane",
        "box_y_axis_xz_plane_error_deg",
    )

    def __init__(self):
        self._buffers = {key: [] for key in self._TRACKED_KEYS}

    def update(self, done_metrics: dict) -> None:
        for key in self._TRACKED_KEYS:
            if key not in done_metrics:
                continue
            values = done_metrics[key]
            if hasattr(values, "detach"):
                values = values.detach().cpu().reshape(-1).tolist()
            else:
                values = values.reshape(-1).tolist()
            self._buffers[key].extend(float(value) for value in values)

    @property
    def done_count(self) -> int:
        return len(self._buffers["box_in_pit_success"])

    def _mean(self, key: str) -> float | None:
        values = self._buffers[key]
        if not values:
            return None
        return mean(values)

    def _mean_for_success(self, key: str, success_value: bool) -> float | None:
        values = self._buffers[key]
        success = self._buffers["box_in_pit_success"]
        if not values or len(values) != len(success):
            return None
        selected = [
            value
            for value, success_value_raw in zip(values, success)
            if bool(success_value_raw >= 0.5) is success_value
        ]
        if not selected:
            return None
        return mean(selected)

    def _mean_for_entered_pit(self, key: str) -> float | None:
        values = self._buffers[key]
        entered = self._buffers["box_entered_pit"]
        if not values or len(values) != len(entered):
            return None
        selected = [
            value
            for value, entered_value in zip(values, entered)
            if entered_value >= 0.5
        ]
        if not selected:
            return None
        return mean(selected)

    def summary(self) -> dict[str, float]:
        if self.done_count == 0:
            return {"done_count": 0}

        summary = {
            "done_count": float(self.done_count),
            "success_rate": mean(self._buffers["box_in_pit_success"]),
            "drop_failure_rate": mean(self._buffers["box_drop_failure"]),
            "native_terminated_rate": mean(self._buffers["raw_terminated"]),
            "timeout_rate": mean(self._buffers["truncated"]),
            "planner_success_reset_rate": mean(self._buffers["planner_success_reset"]),
            "planner_done_reset_rate": mean(self._buffers["planner_done_reset"]),
            "mean_robot_box_distance": mean(self._buffers["robot_box_distance"]),
            "mean_box_pit_distance": mean(self._buffers["box_pit_distance"]),
            "box_xy_in_pit_rect_rate": mean(self._buffers["box_xy_in_pit_rect"]),
            "box_z_success_and_xy_in_rect_rate": mean(self._buffers["box_z_success_and_xy_in_rect"]),
            "mean_box_z": mean(self._buffers["box_z"]),
            "box_entered_pit_rate": mean(self._buffers["box_entered_pit"]),
            "mean_box_y_axis_xz_plane": mean(self._buffers["box_y_axis_xz_plane"]),
            "mean_box_y_axis_xz_plane_error_deg": mean(self._buffers["box_y_axis_xz_plane_error_deg"]),
        }

        entered_pit_xz_plane_rate = self._mean_for_entered_pit("box_entered_pit_xz_plane_good")
        if entered_pit_xz_plane_rate is not None:
            summary["entered_pit_xz_plane_rate"] = entered_pit_xz_plane_rate
        entered_pit_xz_plane = self._mean_for_entered_pit("box_y_axis_xz_plane")
        if entered_pit_xz_plane is not None:
            summary["entered_pit_box_y_axis_xz_plane"] = entered_pit_xz_plane
        entered_pit_xz_plane_error_deg = self._mean_for_entered_pit("box_y_axis_xz_plane_error_deg")
        if entered_pit_xz_plane_error_deg is not None:
            summary["entered_pit_box_y_axis_xz_plane_error_deg"] = entered_pit_xz_plane_error_deg

        for key, summary_key_success, summary_key_failure in (
            ("robot_box_distance", "success_robot_box_distance", "failure_robot_box_distance"),
            ("box_pit_distance", "success_box_pit_distance", "failure_box_pit_distance"),
            ("box_z", "success_box_z", "failure_box_z"),
            ("box_y_axis_xz_plane", "success_box_y_axis_xz_plane", "failure_box_y_axis_xz_plane"),
            (
                "box_y_axis_xz_plane_error_deg",
                "success_box_y_axis_xz_plane_error_deg",
                "failure_box_y_axis_xz_plane_error_deg",
            ),
        ):
            success_mean = self._mean_for_success(key, True)
            failure_mean = self._mean_for_success(key, False)
            if success_mean is not None:
                summary[summary_key_success] = success_mean
            if failure_mean is not None:
                summary[summary_key_failure] = failure_mean

        success_xy_rate = self._mean_for_success("box_xy_in_pit_rect", True)
        failure_xy_rate = self._mean_for_success("box_xy_in_pit_rect", False)
        if success_xy_rate is not None:
            summary["success_xy_in_pit_rect_rate"] = success_xy_rate
        if failure_xy_rate is not None:
            summary["failure_xy_in_pit_rect_rate"] = failure_xy_rate

        return summary


def _world_pose_range_to_reset_offsets(
    *,
    default_pos: tuple[float, float, float],
    env_origin: tuple[float, float, float],
    target_x: tuple[float, float],
    target_y: tuple[float, float],
    target_z: float,
) -> dict[str, tuple[float, float]]:
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


def _configure_training_reset_ranges(env_cfg, *, taskd_initpos: bool = False):
    robot_target_x = (
        TASK_D_G1_ROBOT_DEFAULT_ROOT_POS[0],
        TASK_D_G1_ROBOT_DEFAULT_ROOT_POS[0],
    ) if taskd_initpos else ROBOT_RESET_WORLD_X_RANGE
    robot_target_y = (
        TASK_D_G1_ROBOT_DEFAULT_ROOT_POS[1],
        TASK_D_G1_ROBOT_DEFAULT_ROOT_POS[1],
    ) if taskd_initpos else ROBOT_RESET_WORLD_Y_RANGE
    env_cfg.events.reset_robot_root.params["pose_range"] = _world_pose_range_to_reset_offsets(
        default_pos=TASK_D_G1_ROBOT_DEFAULT_ROOT_POS,
        env_origin=TASK_D_G1_ENV_ORIGIN,
        target_x=robot_target_x,
        target_y=robot_target_y,
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


def _resolve_checkpoints(log_dir: Path, checkpoint_steps: list[int], checkpoints: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for checkpoint in checkpoints:
        resolved.append(Path(checkpoint).expanduser().resolve())
    for step in checkpoint_steps:
        resolved.append((log_dir / f"model_{step}.pt").resolve())

    deduped = []
    seen = set()
    for path in resolved:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)

    missing = [path for path in deduped if not path.exists()]
    if missing:
        missing_str = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing checkpoint(s):\n{missing_str}")
    return deduped


def _filter_compatible_state_dict(policy, state_dict):
    current_state = policy.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        current_value = current_state.get(key)
        if current_value is None or tuple(value.shape) != tuple(current_value.shape):
            skipped.append(key)
            continue
        compatible[key] = value
    return compatible, skipped


def _build_policy(env, checkpoint_path, device):
    obs = env.get_observations()
    from scripts.planner_inference_policy import load_planner_inference_policy

    return load_planner_inference_policy(
        checkpoint_path,
        device=device,
        expected_obs_dim=obs["policy"].shape[-1],
        expected_action_dim=PLANNER_NUM_ACTIONS,
        activation=POLICY_CFG["activation"],
    )


def _evaluate_checkpoint(env, policy, target_done_count: int, max_steps: int) -> dict[str, float]:
    import torch

    obs_dict, _ = env.reset()
    accumulator = DoneMetricAccumulator()
    clip_actions = torch.tensor(CLIP_ACTIONS, device=env.device)

    steps = 0
    start_time = time.time()
    with torch.no_grad():
        while accumulator.done_count < target_done_count and steps < max_steps:
            planner_action = policy.act_inference(obs_dict)
            planner_action = torch.clamp(planner_action, -clip_actions, clip_actions)
            obs_dict, _, terminated, truncated, info = env.step(planner_action)
            if "planner_done_metrics" in info:
                accumulator.update(info["planner_done_metrics"])
            steps += 1

    summary = accumulator.summary()
    summary["steps"] = float(steps)
    summary["eval_wall_time_s"] = time.time() - start_time
    return summary


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Evaluate planner checkpoints in PlannerEnv.")
    parser.add_argument("--log_dir", type=Path, default=PROJECT_ROOT / "logs" / "planner_box_push")
    parser.add_argument("--checkpoint_steps", type=int, nargs="*", default=[5800, 8800, 9000])
    parser.add_argument("--checkpoints", type=str, nargs="*", default=[])
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--episode_length_s", type=int, default=300)
    parser.add_argument("--target_done_count", type=int, default=2000)
    parser.add_argument("--max_steps", type=int, default=4000)
    parser.add_argument("--taskD_initpos", action="store_true", help="Use TaskD's fixed robot initial xy for reset.")
    parser.add_argument("--skip_app_close", action="store_true", help="Exit after printing results without closing Kit.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    args.headless = True
    args.enable_cameras = False

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import torch
        from source.atec_rl_lab.atec_rl_lab.tasks.task_d.env_cfg import TaskDEnvG1Cfg
        from scripts.low_level_policy import GrootLowLevelPolicy
        from scripts.planner_env import PlannerEnv

        env_cfg = TaskDEnvG1Cfg()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.episode_length_s = args.episode_length_s
        env_cfg.terminations.x_reached = None
        env_cfg.rewards.achieve = None
        env_cfg.rewards.box_in_target_x = None
        env_cfg.events.reset_robot_joints = None
        env_cfg = _configure_training_reset_ranges(env_cfg, taskd_initpos=args.taskD_initpos)

        env_cfg.scene.head_camera = None
        env_cfg.scene.ee_camera = None
        env_cfg.scene.ee_dual_camera = None
        env_cfg.observations.image.head_rgb = None
        env_cfg.observations.image.head_depth = None
        env_cfg.observations.image.ee_rgb = None
        env_cfg.observations.image.ee_depth = None
        env_cfg.observations.image.ee_dual_rgb = None
        env_cfg.observations.image.ee_dual_depth = None

        low_level = GrootLowLevelPolicy(device="cuda:0")
        env = PlannerEnv(env_cfg, low_level)
        checkpoint_paths = _resolve_checkpoints(args.log_dir, args.checkpoint_steps, args.checkpoints)

        print(f"Evaluating {len(checkpoint_paths)} checkpoint(s) with num_envs={args.num_envs}, target_done_count={args.target_done_count}")
        for checkpoint_path in checkpoint_paths:
            policy = _build_policy(env, str(checkpoint_path), env.device)
            summary = _evaluate_checkpoint(env, policy, args.target_done_count, args.max_steps)
            print("=" * 100)
            print(checkpoint_path.name)
            for key in sorted(summary):
                value = summary[key]
                if isinstance(value, float):
                    print(f"{key}: {value:.6f}")
                else:
                    print(f"{key}: {value}")
    finally:
        if not args.skip_app_close:
            simulation_app.close()


if __name__ == "__main__":
    main()
