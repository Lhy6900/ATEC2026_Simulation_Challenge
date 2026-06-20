from __future__ import annotations

from collections.abc import Sequence

import torch


def _tensor_scalar(value: float, env) -> torch.Tensor:
    return torch.tensor(float(value), device=env.device)


def _termination_rate(env, name: str, env_ids: Sequence[int] | slice) -> float:
    termination_manager = getattr(env, "termination_manager", None)
    if termination_manager is None:
        return 0.0
    try:
        term = termination_manager.get_term(name)
    except (AttributeError, KeyError):
        return 0.0
    if not isinstance(env_ids, slice):
        term = term[env_ids]
    if term.numel() == 0:
        return 0.0
    return float(term.float().mean().item())


def cross_pit_success_x_value(
    env,
    start_x: float,
    final_x: float,
    start_iteration: int,
    end_iteration: int,
    steps_per_iteration: int = 32,
) -> float:
    """Linearly ramp the success target from the warmup line to the final far-side line."""
    if end_iteration <= start_iteration:
        raise ValueError("end_iteration must be greater than start_iteration")
    if steps_per_iteration <= 0:
        raise ValueError("steps_per_iteration must be positive")

    iteration = float(getattr(env, "common_step_counter", 0)) / float(steps_per_iteration)
    progress = (iteration - float(start_iteration)) / float(end_iteration - start_iteration)
    progress = min(max(progress, 0.0), 1.0)
    return float(start_x + progress * (final_x - start_x))


def update_cross_pit_success_x(
    env,
    env_ids: Sequence[int] | slice,
    start_x: float,
    final_x: float,
    start_iteration: int,
    end_iteration: int,
    steps_per_iteration: int = 32,
) -> dict[str, torch.Tensor]:
    """Update the environment-wide crossing target and return scalar state for TensorBoard."""
    target_x = cross_pit_success_x_value(
        env,
        start_x=start_x,
        final_x=final_x,
        start_iteration=start_iteration,
        end_iteration=end_iteration,
        steps_per_iteration=steps_per_iteration,
    )
    progress = (target_x - start_x) / max(final_x - start_x, 1.0e-6)
    env.cross_pit_success_x = target_x
    return {
        "target_x": _tensor_scalar(target_x, env),
        "progress": _tensor_scalar(progress, env),
    }


def update_cross_pit_success_x_adaptive(
    env,
    env_ids: Sequence[int] | slice,
    start_x: float,
    final_x: float,
    steps_per_iteration: int = 32,
    min_iteration: int = 0,
    update_interval_iterations: int = 10,
    advance_step: float = 0.05,
    fast_advance_step: float = 0.10,
    retreat_step: float = 0.025,
    success_threshold: float = 0.75,
    fast_success_threshold: float = 0.90,
    max_fallen_rate: float = 0.25,
    max_lateral_out_rate: float = 0.15,
    retreat_crossed_threshold: float = 0.35,
    retreat_failure_rate: float = 0.45,
    ema_alpha: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Adapt the success target from actual reset outcomes.

    IsaacLab computes curriculum terms during reset, after the termination manager
    has the current reset batch's term values. The raw rates are smoothed so a
    single noisy reset batch does not jerk the crossing target back and forth.
    """
    if final_x <= start_x:
        raise ValueError("final_x must be greater than start_x")
    if steps_per_iteration <= 0:
        raise ValueError("steps_per_iteration must be positive")
    if update_interval_iterations <= 0:
        raise ValueError("update_interval_iterations must be positive")
    if not 0.0 < ema_alpha <= 1.0:
        raise ValueError("ema_alpha must be in the range (0, 1]")

    target_x = float(getattr(env, "cross_pit_success_x", start_x))
    target_x = min(max(target_x, start_x), final_x)

    crossed_rate = _termination_rate(env, "crossed", env_ids)
    fallen_rate = _termination_rate(env, "fallen", env_ids)
    lateral_out_rate = _termination_rate(env, "lateral_out", env_ids)
    time_out_rate = _termination_rate(env, "time_out", env_ids)

    state = getattr(env, "_cross_pit_adaptive_success_x_state", None)
    if state is None:
        state = {
            "crossed_rate": crossed_rate,
            "fallen_rate": fallen_rate,
            "lateral_out_rate": lateral_out_rate,
            "time_out_rate": time_out_rate,
            "last_update_iteration": -update_interval_iterations,
        }
    else:
        state["crossed_rate"] = (1.0 - ema_alpha) * float(state["crossed_rate"]) + ema_alpha * crossed_rate
        state["fallen_rate"] = (1.0 - ema_alpha) * float(state["fallen_rate"]) + ema_alpha * fallen_rate
        state["lateral_out_rate"] = (1.0 - ema_alpha) * float(state["lateral_out_rate"]) + ema_alpha * lateral_out_rate
        state["time_out_rate"] = (1.0 - ema_alpha) * float(state.get("time_out_rate", 0.0)) + ema_alpha * time_out_rate

    iteration = int(float(getattr(env, "common_step_counter", 0)) / float(steps_per_iteration))
    last_update_iteration = int(state["last_update_iteration"])
    should_update = iteration >= min_iteration and iteration - last_update_iteration >= update_interval_iterations

    if should_update:
        stable_enough = state["fallen_rate"] <= max_fallen_rate and state["lateral_out_rate"] <= max_lateral_out_rate
        if state["crossed_rate"] >= fast_success_threshold and stable_enough:
            target_x += fast_advance_step
        elif state["crossed_rate"] >= success_threshold and stable_enough:
            target_x += advance_step
        elif (
            state["crossed_rate"] < retreat_crossed_threshold
            and (state["fallen_rate"] >= retreat_failure_rate or state["lateral_out_rate"] >= retreat_failure_rate)
        ):
            target_x -= retreat_step
        state["last_update_iteration"] = iteration

    target_x = min(max(target_x, start_x), final_x)
    progress = (target_x - start_x) / max(final_x - start_x, 1.0e-6)

    env.cross_pit_success_x = target_x
    env._cross_pit_adaptive_success_x_state = state
    return {
        "target_x": _tensor_scalar(target_x, env),
        "progress": _tensor_scalar(progress, env),
        "crossed_rate": _tensor_scalar(state["crossed_rate"], env),
        "fallen_rate": _tensor_scalar(state["fallen_rate"], env),
        "lateral_out_rate": _tensor_scalar(state["lateral_out_rate"], env),
        "time_out_rate": _tensor_scalar(state["time_out_rate"], env),
    }


def update_cross_pit_success_x_sticky_stage(
    env,
    env_ids: Sequence[int] | slice,
    start_x: float,
    final_x: float,
    retreat_floor_x: float,
    sticky_start_x: float | None = None,
    steps_per_iteration: int = 32,
    min_iteration: int = 0,
    update_interval_iterations: int = 10,
    advance_step: float = 0.05,
    fast_advance_step: float = 0.10,
    retreat_step: float = 0.025,
    success_threshold: float = 0.75,
    fast_success_threshold: float = 0.90,
    max_fallen_rate: float = 0.25,
    max_lateral_out_rate: float = 0.15,
    retreat_crossed_threshold: float = 0.35,
    retreat_failure_rate: float = 0.45,
    timeout_plateau_advance_step: float = 0.0,
    timeout_plateau_threshold: float = 0.9,
    timeout_plateau_max_crossed_rate: float = 0.05,
    timeout_plateau_max_fallen_rate: float = 0.10,
    timeout_plateau_unlock_x: float | None = None,
    report_relative_to_x: float | None = None,
    ema_alpha: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Adaptive success target with a sticky floor at the box stage.

    Once the target has reached the configured sticky start, failures should
    keep training on that hard transition instead of falling all the way back to
    the trivial near-lip target.
    """
    previous_target = float(getattr(env, "cross_pit_success_x", start_x))
    state = update_cross_pit_success_x_adaptive(
        env,
        env_ids=env_ids,
        start_x=start_x,
        final_x=final_x,
        steps_per_iteration=steps_per_iteration,
        min_iteration=min_iteration,
        update_interval_iterations=update_interval_iterations,
        advance_step=advance_step,
        fast_advance_step=fast_advance_step,
        retreat_step=retreat_step,
        success_threshold=success_threshold,
        fast_success_threshold=fast_success_threshold,
        max_fallen_rate=max_fallen_rate,
        max_lateral_out_rate=max_lateral_out_rate,
        retreat_crossed_threshold=retreat_crossed_threshold,
        retreat_failure_rate=retreat_failure_rate,
        ema_alpha=ema_alpha,
    )
    sticky_floor = min(max(retreat_floor_x, start_x), final_x)
    sticky_start = sticky_floor if sticky_start_x is None else min(max(sticky_start_x, start_x), final_x)
    sticky_reached = bool(getattr(env, "_cross_pit_sticky_stage_reached", False))
    sticky_reached = sticky_reached or previous_target >= sticky_start or env.cross_pit_success_x >= sticky_start
    if sticky_reached and env.cross_pit_success_x < sticky_floor:
        env.cross_pit_success_x = sticky_floor
        progress = (sticky_floor - start_x) / max(final_x - start_x, 1.0e-6)
        state["target_x"] = _tensor_scalar(sticky_floor, env)
        state["progress"] = _tensor_scalar(progress, env)
    env._cross_pit_sticky_stage_reached = sticky_reached

    plateau_unlock = sticky_floor if timeout_plateau_unlock_x is None else min(max(timeout_plateau_unlock_x, start_x), final_x)
    plateau_escape = (
        timeout_plateau_advance_step > 0.0
        and env.cross_pit_success_x >= plateau_unlock
        and float(state["time_out_rate"].item()) >= timeout_plateau_threshold
        and float(state["crossed_rate"].item()) <= timeout_plateau_max_crossed_rate
        and float(state["fallen_rate"].item()) <= timeout_plateau_max_fallen_rate
    )
    if plateau_escape:
        target_x = min(env.cross_pit_success_x + timeout_plateau_advance_step, final_x)
        env.cross_pit_success_x = target_x
        progress = (target_x - start_x) / max(final_x - start_x, 1.0e-6)
        state["target_x"] = _tensor_scalar(target_x, env)
        state["progress"] = _tensor_scalar(progress, env)

    stage = 0.0
    if sticky_reached or env.cross_pit_success_x >= sticky_floor:
        stage = 2.0
    if env.cross_pit_success_x >= final_x:
        stage = 4.0
    if report_relative_to_x is not None:
        state["target_x_abs"] = _tensor_scalar(env.cross_pit_success_x, env)
        state.pop("target_x", None)
        state["target_x_rel"] = _tensor_scalar(env.cross_pit_success_x - report_relative_to_x, env)
    state["stage"] = _tensor_scalar(stage, env)
    return state
