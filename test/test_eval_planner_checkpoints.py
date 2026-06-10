import sys
import types
import unittest
import ast
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.eval_planner_checkpoints import DoneMetricAccumulator, _build_policy, _configure_training_reset_ranges


EVAL_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_planner_checkpoints.py"


class DoneMetricAccumulatorTests(unittest.TestCase):
    def test_accumulator_summarizes_done_metrics(self):
        acc = DoneMetricAccumulator()
        acc.update(
            {
                "box_in_pit_success": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
                "box_drop_failure": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                "robot_box_distance": np.array([1.0, 10.0, 3.0, 20.0], dtype=np.float32),
                "box_pit_distance": np.array([0.0, 4.0, 2.0, 8.0], dtype=np.float32),
                "raw_terminated": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                "truncated": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "planner_success_reset": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
                "planner_done_reset": np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
                "box_xy_in_pit_rect": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "box_z_success_and_xy_in_rect": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "box_z": np.array([-0.3, 0.5, -0.4, 0.6], dtype=np.float32),
                "box_entered_pit": np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
                "box_y_axis_xz_plane": np.array([1.0, 0.0, 0.5, 0.25], dtype=np.float32),
                "box_y_axis_xz_plane_error_deg": np.array([0.0, 90.0, 45.0, 60.0], dtype=np.float32),
                "box_entered_pit_xz_plane_good": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            }
        )

        summary = acc.summary()

        self.assertEqual(summary["done_count"], 4)
        self.assertAlmostEqual(summary["success_rate"], 0.5)
        self.assertAlmostEqual(summary["drop_failure_rate"], 0.25)
        self.assertAlmostEqual(summary["native_terminated_rate"], 0.25)
        self.assertAlmostEqual(summary["timeout_rate"], 0.25)
        self.assertAlmostEqual(summary["planner_success_reset_rate"], 0.5)
        self.assertAlmostEqual(summary["planner_done_reset_rate"], 0.75)
        self.assertAlmostEqual(summary["mean_robot_box_distance"], 8.5)
        self.assertAlmostEqual(summary["mean_box_pit_distance"], 3.5)
        self.assertAlmostEqual(summary["success_robot_box_distance"], 2.0)
        self.assertAlmostEqual(summary["failure_robot_box_distance"], 15.0)
        self.assertAlmostEqual(summary["success_box_pit_distance"], 1.0)
        self.assertAlmostEqual(summary["failure_box_pit_distance"], 6.0)
        self.assertAlmostEqual(summary["success_box_z"], -0.35)
        self.assertAlmostEqual(summary["failure_box_z"], 0.55)
        self.assertAlmostEqual(summary["box_xy_in_pit_rect_rate"], 0.25)
        self.assertAlmostEqual(summary["success_xy_in_pit_rect_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_box_y_axis_xz_plane"], 0.4375)
        self.assertAlmostEqual(summary["success_box_y_axis_xz_plane"], 0.75)
        self.assertAlmostEqual(summary["failure_box_y_axis_xz_plane"], 0.125)
        self.assertAlmostEqual(summary["mean_box_y_axis_xz_plane_error_deg"], 48.75)
        self.assertAlmostEqual(summary["success_box_y_axis_xz_plane_error_deg"], 22.5)
        self.assertAlmostEqual(summary["failure_box_y_axis_xz_plane_error_deg"], 75.0)
        self.assertAlmostEqual(summary["box_entered_pit_rate"], 0.75)
        self.assertAlmostEqual(summary["entered_pit_xz_plane_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["entered_pit_box_y_axis_xz_plane"], 0.5)
        self.assertAlmostEqual(summary["entered_pit_box_y_axis_xz_plane_error_deg"], 45.0)


class ResetRangeConfigTests(unittest.TestCase):
    def _fake_env_cfg(self):
        return types.SimpleNamespace(
            events=types.SimpleNamespace(
                reset_robot_root=types.SimpleNamespace(params={"pose_range": {}}),
                reset_box_root=types.SimpleNamespace(params={"pose_range": {}}),
            )
        )

    def test_taskd_initpos_fixes_robot_xy_to_taskd_default(self):
        env_cfg = self._fake_env_cfg()

        _configure_training_reset_ranges(env_cfg, taskd_initpos=True)

        pose_range = env_cfg.events.reset_robot_root.params["pose_range"]
        self.assertEqual(pose_range["x"], (4.2, 4.2))
        self.assertEqual(pose_range["y"], (0.0, 0.0))
        self.assertEqual(pose_range["z"], (0.0, 0.0))

    def test_default_eval_reset_keeps_random_training_xy_range(self):
        env_cfg = self._fake_env_cfg()

        _configure_training_reset_ranges(env_cfg)

        pose_range = env_cfg.events.reset_robot_root.params["pose_range"]
        self.assertEqual(pose_range["x"], (2.7, 5.7))
        self.assertEqual(pose_range["y"], (-0.5, 0.5))


class BuildPolicyTests(unittest.TestCase):
    def test_build_policy_uses_repo_inference_loader_without_rsl_rl_modules(self):
        import torch

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.pt"
            state_dict = {
                "actor.0.weight": torch.zeros(4, 372),
                "actor.0.bias": torch.tensor([1.0, 2.0, 3.0, 4.0]),
                "actor.2.weight": torch.zeros(3, 4),
                "actor.2.bias": torch.tensor([0.5, -0.5, 0.25]),
            }
            torch.save({"model_state_dict": state_dict}, checkpoint_path)

            fake_env = types.SimpleNamespace(
                get_observations=lambda: {
                    "policy": torch.zeros(2, 372),
                    "critic": torch.zeros(2, 23),
                },
            )

            with patch.dict(sys.modules, {"rsl_rl.modules": None}):
                policy = _build_policy(fake_env, str(checkpoint_path), "cpu")

            action = policy.act_inference({"policy": torch.zeros(2, 372)})

        self.assertEqual(tuple(action.shape), (2, 3))
        self.assertTrue(torch.allclose(action[0], torch.tensor([0.5, -0.5, 0.25])))

    def test_eval_script_does_not_import_actorcritic_directly(self):
        source = EVAL_SCRIPT_PATH.read_text(encoding="utf-8")

        self.assertNotIn("from rsl_rl.modules import ActorCritic", source)
        self.assertIn("load_planner_inference_policy", source)


class EvalScriptCliTests(unittest.TestCase):
    def test_taskd_initpos_cli_controls_eval_reset_ranges(self):
        source = EVAL_SCRIPT_PATH.read_text(encoding="utf-8")
        module = ast.parse(source)
        main_fn = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        configure_fn = next(
            node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "_configure_training_reset_ranges"
        )

        added_args = [
            node.args[0].value
            for node in ast.walk(main_fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ]
        self.assertIn("--taskD_initpos", added_args)

        configure_source = ast.get_source_segment(source, configure_fn)
        main_source = ast.get_source_segment(source, main_fn)
        self.assertIn("taskd_initpos", configure_source)
        self.assertIn("TASK_D_G1_ROBOT_DEFAULT_ROOT_POS[0]", configure_source)
        self.assertIn("TASK_D_G1_ROBOT_DEFAULT_ROOT_POS[1]", configure_source)
        self.assertIn("taskd_initpos=args.taskD_initpos", main_source)

    def test_skip_app_close_cli_guard_exists(self):
        module = ast.parse(EVAL_SCRIPT_PATH.read_text(encoding="utf-8"))
        main_fn = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")

        added_args = [
            node.args[0].value
            for node in ast.walk(main_fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ]
        self.assertIn("--skip_app_close", added_args)

        close_calls = [
            node
            for node in ast.walk(main_fn)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "close"
        ]
        self.assertEqual(len(close_calls), 1)
        close_call = close_calls[0]
        parent_guards = [
            node
            for node in ast.walk(main_fn)
            if isinstance(node, ast.If)
            and any(child is close_call for child in ast.walk(node))
        ]
        self.assertTrue(
            any("skip_app_close" in ast.unparse(guard.test) for guard in parent_guards),
            "simulation_app.close() should be guarded by args.skip_app_close",
        )


if __name__ == "__main__":
    unittest.main()
