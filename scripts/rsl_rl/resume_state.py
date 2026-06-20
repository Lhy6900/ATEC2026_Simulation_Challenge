from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResumeState:
    iteration: int
    common_step_counter: int
    total_timesteps: int


def sync_resume_training_state(env, runner) -> ResumeState | None:
    """Sync curriculum and logging counters after loading a checkpoint."""
    iteration = int(getattr(runner, "current_learning_iteration", 0))
    num_steps_per_env = getattr(runner, "num_steps_per_env", None)
    if iteration <= 0 or num_steps_per_env is None:
        return None

    base_env = getattr(env, "unwrapped", env)
    common_step_counter = iteration * int(num_steps_per_env)

    if hasattr(base_env, "common_step_counter"):
        base_env.common_step_counter = common_step_counter

    curriculum_manager = getattr(base_env, "curriculum_manager", None)
    if curriculum_manager is not None:
        curriculum_manager.compute()
        if hasattr(base_env, "extras"):
            base_env.extras.setdefault("log", {})
            base_env.extras["log"].update(curriculum_manager.reset())

    num_envs = int(getattr(env, "num_envs", getattr(base_env, "num_envs", 1)))
    world_size = int(getattr(runner, "gpu_world_size", 1))
    total_timesteps = iteration * int(num_steps_per_env) * num_envs * world_size
    runner.tot_timesteps = total_timesteps

    return ResumeState(
        iteration=iteration,
        common_step_counter=common_step_counter,
        total_timesteps=total_timesteps,
    )
