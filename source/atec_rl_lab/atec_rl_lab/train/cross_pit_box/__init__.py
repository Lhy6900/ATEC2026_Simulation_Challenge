"""Training task for G1 crossing TaskD pit using the fixed counter-650 box support."""

import gymnasium as gym

from . import agents


gym.register(
    id="ATEC-Isaac-TaskD-G1-CrossPitBox-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:CrossPitBoxG1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CrossPitBoxG1PPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-TaskD-G1-CrossPitBox-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_v1:CrossPitBoxG1V1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CrossPitBoxG1PPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-TaskD-G1-CrossPitBox-v1.5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_v1_5:CrossPitBoxG1V15EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CrossPitBoxG1PPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-TaskD-G1-CrossPitBox-v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_v2:CrossPitBoxG1V2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CrossPitBoxG1PPORunnerCfg",
    },
)

gym.register(
    id="ATEC-Isaac-TaskD-G1-CrossPitBox-v2.5",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_v2_5:CrossPitBoxG1V25EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CrossPitBoxG1PPORunnerCfg",
    },
)
