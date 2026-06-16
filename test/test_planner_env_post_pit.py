import types
import unittest

import torch

from scripts import planner_env


class _FakeScene:
    def __init__(self, robot_z):
        robot_pos = torch.zeros(len(robot_z), 3)
        robot_pos[:, 2] = torch.tensor(robot_z, dtype=torch.float32)
        box_pos = torch.zeros(len(robot_z), 3)
        box_vel = torch.zeros(len(robot_z), 6)
        box_quat = torch.zeros(len(robot_z), 4)
        box_quat[:, 0] = 1.0
        self._assets = {
            "robot": types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=robot_pos)),
            "box": types.SimpleNamespace(
                data=types.SimpleNamespace(
                    root_pos_w=box_pos,
                    root_vel_w=box_vel,
                    root_quat_w=box_quat,
                )
            ),
        }

    def __getitem__(self, name):
        return self._assets[name]


def _minimal_planner_env(robot_z):
    env = object.__new__(planner_env.PlannerEnv)
    env.env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(scene=_FakeScene(robot_z)))
    return env


class PlannerEnvPostPitTests(unittest.TestCase):
    def test_post_pit_hold_requires_three_seconds_at_10hz(self):
        self.assertEqual(planner_env.POST_PIT_HOLD_STEPS, 30)

    def test_post_pit_hold_only_accumulates_while_robot_is_stable(self):
        env = _minimal_planner_env(robot_z=[0.3, 0.1])
        env._box_in_pit_latched = torch.tensor([True, True])
        env._post_pit_step_buf = torch.tensor([149, 149], dtype=torch.long)

        _, post_pit_success = env._update_post_pit_state(torch.tensor([True, True]))

        self.assertEqual(env._post_pit_step_buf.tolist(), [150, 0])
        self.assertEqual(post_pit_success.tolist(), [True, False])

    def test_stable_after_pit_reward_is_logged_and_added_to_total_reward(self):
        env = _minimal_planner_env(robot_z=[0.3, 0.1])
        env._prev_box_pos_w = torch.zeros(2, 3)
        env._last_planner_action = torch.zeros(2, 3)
        env._box_in_pit_latched = torch.tensor([True, True])
        env._post_pit_step_buf = torch.tensor([1, 1], dtype=torch.long)
        env._check_box_in_pit = lambda: torch.tensor([True, True])
        env._box_to_pit_distance_xy = lambda box_xy: torch.zeros(box_xy.shape[0])
        env._box_pit_gate = lambda robot_box_distance: torch.zeros_like(robot_box_distance)
        env._box_in_pit_success_reward = lambda: torch.zeros(2)
        env._box_y_axis_xz_plane_score = lambda: torch.zeros(2)
        env._box_xz_plane_near_pit_gate = lambda dist: torch.zeros_like(dist)
        env._obstacle_rect_penalty_xy = lambda point_xy: torch.zeros(point_xy.shape[0])

        reward = env._compute_reward(
            prev_action=torch.zeros(2, 3),
            prev_nav_cmd=torch.zeros(2, 3),
            nav_cmd=torch.zeros(2, 3),
        )

        self.assertIn("stable_after_pit", env._last_reward_components)
        self.assertTrue(torch.allclose(env._last_reward_components["stable_after_pit"], torch.tensor([1.0, 0.0])))
        self.assertAlmostEqual(float(reward[0] - reward[1]), planner_env.REWARD_STABLE_AFTER_PIT_WEIGHT)


if __name__ == "__main__":
    unittest.main()
