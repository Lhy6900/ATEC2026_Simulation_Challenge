import gymnasium as gym

gym.register(
    id="ATEC-TaskD-Record-G1",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskDRecordEnvG1Cfg",
    },
)

from .env_cfg import TaskDRecordEnvG1Cfg

__all__ = ["TaskDRecordEnvG1Cfg"]
