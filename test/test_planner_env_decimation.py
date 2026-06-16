import types
import unittest

import torch

from scripts import planner_env


class _FakeLowLevel:
    def __init__(self):
        self.nav_cmds = []

    def predict(self, obs, nav_cmd):
        self.nav_cmds.append(nav_cmd.detach().clone())
        return torch.zeros(nav_cmd.shape[0], 33)

    def reset_envs(self, env_ids, single_obs):
        pass

    def _build_obs(self, proprio, nav_cmd):
        return torch.zeros(proprio.shape[0], 86)


class _FakeEnv:
    def __init__(self):
        self.step_count = 0
        self.unwrapped = self
        self.scene = {
            "box": types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=torch.zeros(1, 3))),
        }

    def step(self, action):
        self.step_count += 1
        obs = {
            "proprio": torch.full((1, 111), float(self.step_count)),
            "extero": torch.zeros(1, 5760),
        }
        terminated = torch.tensor([False])
        truncated = torch.tensor([False])
        return obs, torch.zeros(1), terminated, truncated, {"substep": self.step_count}


class _PartialDoneFakeEnv:
    def __init__(self):
        self.step_count = 0
        self.unwrapped = self
        self.scene = {
            "box": types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=torch.zeros(2, 3))),
        }

    def step(self, action):
        self.step_count += 1
        obs = {
            "proprio": torch.full((2, 111), float(self.step_count)),
            "extero": torch.zeros(2, 5760),
        }
        terminated = torch.tensor([self.step_count == 2, False])
        truncated = torch.tensor([False, False])
        return obs, torch.zeros(2), terminated, truncated, {"substep": self.step_count}


def _minimal_planner_env(fake_env=None, num_envs=1):
    fake_env = fake_env or _FakeEnv()
    low_level = _FakeLowLevel()
    env = object.__new__(planner_env.PlannerEnv)
    env.env = fake_env
    env.low_level = low_level
    env.device = torch.device("cpu")
    env.num_envs = num_envs
    env._nav_min = planner_env.NAV_CMD_MIN.to(env.device)
    env._nav_max = planner_env.NAV_CMD_MAX.to(env.device)
    env._current_obs = {"proprio": torch.zeros(num_envs, 111), "extero": torch.zeros(num_envs, 5760)}
    env._prev_box_pos_w = torch.zeros(num_envs, 3)
    env._box_in_pit_given = torch.zeros(num_envs, dtype=torch.bool)
    env._box_in_pit_latched = torch.zeros(num_envs, dtype=torch.bool)
    env._post_pit_step_buf = torch.zeros(num_envs, dtype=torch.long)
    env._last_planner_action = torch.zeros(num_envs, 3)
    env._nav_cmd = torch.zeros(num_envs, 3)
    env._episode_step_buf = torch.zeros(num_envs, dtype=torch.long)
    env._last_reward_components = {}
    env._done_reason_printed = True
    env._post_reset_sync_compare_printed = True
    env._check_box_in_pit = lambda: torch.zeros(num_envs, dtype=torch.bool)
    env._check_box_drop_failure = lambda: torch.zeros(num_envs, dtype=torch.bool)
    env._update_post_pit_state = lambda box_in_pit_now: (
        torch.zeros(num_envs, dtype=torch.bool),
        torch.zeros(num_envs, dtype=torch.bool),
    )
    env._build_planner_obs = lambda obs: {"policy": obs["proprio"][:, :1], "critic": obs["proprio"][:, :1]}
    env._compute_done_metrics = lambda *args, **kwargs: {}

    def fake_compute_reward(prev_action, prev_nav_cmd, nav_cmd):
        value = torch.full((num_envs,), float(fake_env.step_count))
        env._last_reward_components = {key: value.clone() for key in planner_env.REWARD_COMPONENT_KEYS}
        return value

    env._compute_reward = fake_compute_reward
    return env, fake_env, low_level


class PlannerEnvDecimationTests(unittest.TestCase):
    def test_planner_step_runs_five_low_level_steps_and_sums_rewards(self):
        env, fake_env, low_level = _minimal_planner_env()

        obs, reward, terminated, truncated, info = env.step(torch.tensor([[0.5, 0.0, 0.0]]))

        self.assertEqual(planner_env.PLANNER_DECIMATION, 5)
        self.assertEqual(fake_env.step_count, 5)
        self.assertEqual(len(low_level.nav_cmds), 5)
        self.assertTrue(all(torch.allclose(nav_cmd, torch.tensor([[0.125, 0.0, 0.0]])) for nav_cmd in low_level.nav_cmds))
        self.assertEqual(env._episode_step_buf.tolist(), [1])
        self.assertTrue(torch.allclose(reward, torch.tensor([15.0])))
        self.assertTrue(torch.allclose(env._last_reward_components["total"], torch.tensor([15.0])))
        self.assertFalse(bool(terminated.item()))
        self.assertFalse(bool(truncated.item()))
        self.assertEqual(info["substep"], 5)
        self.assertTrue(torch.allclose(obs["policy"], torch.full((1, 1), 5.0)))

    def test_partial_env_done_does_not_shorten_other_envs_decimation_window(self):
        env, fake_env, low_level = _minimal_planner_env(fake_env=_PartialDoneFakeEnv(), num_envs=2)

        _, reward, terminated, truncated, info = env.step(
            torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
        )

        self.assertEqual(fake_env.step_count, planner_env.PLANNER_DECIMATION)
        self.assertEqual(len(low_level.nav_cmds), planner_env.PLANNER_DECIMATION)
        self.assertEqual(info["planner_decimation_substeps"], planner_env.PLANNER_DECIMATION)
        self.assertEqual(terminated.tolist(), [True, False])
        self.assertEqual(truncated.tolist(), [False, False])
        self.assertTrue(torch.allclose(reward, torch.tensor([3.0, 15.0])))
        self.assertTrue(
            all(torch.allclose(nav_cmd[0], torch.zeros(3)) for nav_cmd in low_level.nav_cmds[2:])
        )
        self.assertTrue(
            all(torch.allclose(nav_cmd[1], torch.tensor([0.125, 0.0, 0.0])) for nav_cmd in low_level.nav_cmds)
        )


if __name__ == "__main__":
    unittest.main()
