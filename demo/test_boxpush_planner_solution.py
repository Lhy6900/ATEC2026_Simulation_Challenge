import unittest

import torch

from demo import boxpush_planner_solution as solution


class _FakePlanner:
    def __init__(self):
        self.calls = 0

    def act_inference(self, obs):
        self.calls += 1
        return torch.tensor([[0.4, 0.2, 0.1]], dtype=torch.float32)


class _FakeLowLevel:
    def __init__(self):
        self.nav_cmds = []
        self.reset_calls = []

    def reset(self, num_envs):
        self.reset_calls.append(int(num_envs))

    def predict(self, obs, nav_cmd):
        self.nav_cmds.append(nav_cmd.detach().clone())
        return torch.zeros((nav_cmd.shape[0], 33), dtype=torch.float32)


class BoxPushPlannerSolutionTests(unittest.TestCase):
    def _make_solution(self):
        policy = object.__new__(solution.BoxPushPlannerSolution)
        policy.device = torch.device("cpu")
        policy.planner = _FakePlanner()
        policy.low_level = _FakeLowLevel()
        policy._num_envs = 0
        policy._nav_cmd = None
        policy._last_planner_action = None
        policy._step = 0
        policy._debug_snapshot = {}
        return policy

    def test_planner_runs_at_10hz_and_low_level_runs_every_step(self):
        policy = self._make_solution()
        obs = {
            "proprio": torch.zeros((1, 111), dtype=torch.float32),
            "extero": torch.zeros((1, 5760), dtype=torch.float32),
        }

        for _ in range(solution.PLANNER_DECIMATION + 1):
            policy.predicts(obs, current_score=0.0)

        self.assertEqual(policy.planner.calls, 2)
        self.assertEqual(len(policy.low_level.nav_cmds), solution.PLANNER_DECIMATION + 1)
        self.assertTrue(torch.allclose(policy.low_level.nav_cmds[0], torch.tensor([[0.1, 0.05, 0.025]])))
        self.assertTrue(torch.allclose(policy.low_level.nav_cmds[1], policy.low_level.nav_cmds[0]))
        self.assertTrue(torch.allclose(policy.low_level.nav_cmds[solution.PLANNER_DECIMATION - 1], policy.low_level.nav_cmds[0]))
        self.assertTrue(torch.allclose(policy.low_level.nav_cmds[solution.PLANNER_DECIMATION], torch.tensor([[0.2, 0.1, 0.05]])))
        self.assertEqual(policy.get_debug_snapshot()["planner_phase"], 1)

    def test_default_checkpoint_name_is_10hz_model(self):
        policy = object.__new__(solution.BoxPushPlannerSolution)
        policy.base_dir = solution.Path("/tmp/demo")

        with unittest.mock.patch.object(solution.Path, "exists", return_value=True):
            resolved = policy._resolve_path(
                None,
                env_name="ATEC_BOXPUSH_PLANNER_CHECKPOINT",
                default_name=solution.DEFAULT_PLANNER_CHECKPOINT,
            )

        self.assertEqual(resolved.name, "boxpush_planner_10hz.pt")


if __name__ == "__main__":
    unittest.main()
