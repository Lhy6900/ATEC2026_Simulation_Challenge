from __future__ import annotations

import os
from dataclasses import dataclass


def _get_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return float(default)
    return float(value)


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class VisionLocoConfig:
    switch_time_s: float = 9.6
    command: tuple[float, float, float] = (1., 0.0, 0.3)
    action_scale: float = 0.25
    env_action_scale: float = 0.5
    height_obs_key: str = "vision_height_scan"
    height_memory_path: str = "vision_height_memory.npz"
    record_height_memory: bool = False
    record_start_s: float = 9.2
    record_end_s: float = 13.0
    flat_after_record_end: bool = False
    enable_pd_compensation: bool = True
    clamp_to_train_effort_limits: bool = True
    command_ramp_time_s: float = 0.5
    debug_interval: int = 0
    nominal_step_dt: float = 0.02

    @classmethod
    def from_env(cls) -> "VisionLocoConfig":
        return cls(
            switch_time_s=_get_float_env("ATEC_VISION_SWITCH_TIME", 9.2),
            command=(
                _get_float_env("ATEC_VISION_CMD_X", 0.6),
                _get_float_env("ATEC_VISION_CMD_Y", 0.0),
                _get_float_env("ATEC_VISION_CMD_YAW", 0.0),
            ),
            action_scale=_get_float_env("ATEC_VISION_ACTION_SCALE", 0.25),
            env_action_scale=_get_float_env("ATEC_VISION_ENV_ACTION_SCALE", 0.5),
            height_obs_key=os.environ.get("ATEC_VISION_HEIGHT_OBS_KEY", "vision_height_scan"),
            height_memory_path=os.environ.get("ATEC_VISION_HEIGHT_MEMORY", "vision_height_memory.npz"),
            record_height_memory=_get_bool_env("ATEC_VISION_RECORD_HEIGHT", False),
            record_start_s=_get_float_env("ATEC_VISION_RECORD_START", 9.2),
            record_end_s=_get_float_env("ATEC_VISION_RECORD_END", 13.0),
            flat_after_record_end=_get_bool_env("ATEC_VISION_FLAT_AFTER_RECORD_END", False),
            enable_pd_compensation=_get_bool_env("ATEC_VISION_PD_COMPENSATION", True),
            clamp_to_train_effort_limits=_get_bool_env("ATEC_VISION_CLAMP_TRAIN_EFFORT", True),
            command_ramp_time_s=_get_float_env("ATEC_VISION_CMD_RAMP_TIME", 0.5),
            debug_interval=int(_get_float_env("ATEC_VISION_DEBUG_INTERVAL", 0.0)),
            nominal_step_dt=_get_float_env("ATEC_VISION_STEP_DT", 0.02),
        )
