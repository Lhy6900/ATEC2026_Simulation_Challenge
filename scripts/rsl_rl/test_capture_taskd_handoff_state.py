import math

import pytest
import torch


def test_quat_wxyz_to_yaw_returns_heading_angle():
    from scripts.rsl_rl.capture_taskd_handoff_state import quat_wxyz_to_yaw

    quat = [math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)]

    assert quat_wxyz_to_yaw(quat) == pytest.approx(math.pi / 2.0)


def test_pose_xyzyaw_from_asset_uses_env_local_position():
    from scripts.rsl_rl.capture_taskd_handoff_state import pose_xyzyaw_from_asset

    class _Data:
        root_pos_w = torch.tensor([[7.25, -2.0, 0.8]])
        root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    class _Asset:
        data = _Data()

    env_origin = torch.tensor([8.0, -3.0, 0.0])

    assert pose_xyzyaw_from_asset(_Asset(), env_origin) == pytest.approx([-0.75, 1.0, 0.8, 0.0])


def test_capture_script_disables_unused_task_cameras_only_for_fast_headless_capture():
    from pathlib import Path

    script_text = (Path(__file__).with_name("capture_taskd_handoff_state.py")).read_text()

    assert 'parser.add_argument(\n        "--fast_headless_capture"' in script_text
    assert "if args_cli.fast_headless_capture:" in script_text
    assert "env_cfg.scene.head_camera = None" in script_text
    assert "env_cfg.scene.ee_camera = None" in script_text
    assert "env_cfg.scene.ee_dual_camera = None" in script_text
    assert "env_cfg.observations.image = None" in script_text


def test_capture_script_disables_task_terminations_only_for_fast_headless_capture():
    from pathlib import Path

    script_text = (Path(__file__).with_name("capture_taskd_handoff_state.py")).read_text()

    assert "if args_cli.fast_headless_capture:" in script_text
    assert "env_cfg.terminations.time_out = None" in script_text
    assert "env_cfg.terminations.fall = None" in script_text
    assert "env_cfg.terminations.x_reached = None" in script_text


def test_capture_script_accepts_play_debug_flag_for_cli_parity():
    from pathlib import Path

    script_text = (Path(__file__).with_name("capture_taskd_handoff_state.py")).read_text()

    assert 'parser.add_argument("--debug"' in script_text


def test_debug_push_runner_attaches_head_camera_pose_for_head_depth_heightmap():
    from pathlib import Path

    script_text = (Path(__file__).resolve().parents[1] / "debug_push_then_cross_headless.py").read_text()

    assert '"_debug_head_camera_intrinsics"' in script_text
    assert '"_debug_head_camera_pos_w"' in script_text
    assert '"_debug_head_camera_quat_w"' in script_text


def test_debug_push_runner_can_attach_cross_pit_training_raycast():
    from pathlib import Path

    script_text = (Path(__file__).resolve().parents[1] / "debug_push_then_cross_headless.py").read_text()

    assert "--attach_cross_pit_depth_scanner" in script_text
    assert "def _make_cross_pit_depth_scanner_cfg(" in script_text
    assert 'prim_path="{ENV_REGEX_NS}/Robot/d435_link"' in script_text
    assert "resolution=0.1" in script_text
    assert "size=(2.3000000000000003, 1.5)" in script_text
    assert "ordering=\"xy\"" in script_text
    assert "ray_alignment=\"yaw\"" in script_text
    assert "pos=(1.3, 0.0, 2.5)" in script_text
    assert '"_debug_depth_ray_hits_w"' in script_text


def test_debug_push_runner_crossed_result_uses_environment_x_reached_term():
    from pathlib import Path

    script_text = (Path(__file__).resolve().parents[1] / "debug_push_then_cross_headless.py").read_text()

    assert "def _is_x_reached(" not in script_text
    assert 'crossed = crossed or bool(terms.get("x_reached", False))' in script_text
    assert "return 0 if crossed else 2" in script_text


def test_debug_push_runner_logs_env_local_root_position():
    from pathlib import Path

    script_text = (Path(__file__).resolve().parents[1] / "debug_push_then_cross_headless.py").read_text()

    assert 'env_origins = getattr(env.unwrapped.scene, "env_origins", None)' in script_text
    assert "root_pos_env = root_pos - env_origin" in script_text
    assert '"root_x_env": float(root_pos_env[0].item())' in script_text


def test_native_cross_pit_debug_reapplies_success_x_after_wrapper_reset():
    from pathlib import Path

    script_text = (Path(__file__).with_name("debug_cross_pit_native_headless.py")).read_text()

    assert "def _force_success_x(" in script_text
    assert "requested_success_x = float(args_cli.cross_pit_success_x)" in script_text
    assert "RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)" in script_text
    assert "Curriculum terms can update" in script_text
    assert "_force_success_x(wrapped_env, requested_success_x)" in script_text
    assert '"cross_pit_success_x": _current_success_x(wrapped_env)' in script_text


def test_native_cross_pit_debug_done_record_marks_post_reset_state():
    from pathlib import Path

    script_text = (Path(__file__).with_name("debug_cross_pit_native_headless.py")).read_text()

    assert '"phase": "post_step"' in script_text
    assert 'record["post_step_state_after_auto_reset"] = True' in script_text
    assert "ManagerBasedRLEnv resets terminated envs before returning observations/state" in script_text
    assert 'record["terminal_pre_step"] = last_pre_record' in script_text


def test_native_cross_pit_debug_logs_env_local_root_position():
    from pathlib import Path

    script_text = (Path(__file__).with_name("debug_cross_pit_native_headless.py")).read_text()

    assert 'env_origins = getattr(env.unwrapped.scene, "env_origins", None)' in script_text
    assert "root_pos_env = root_pos - env_origin" in script_text
    assert '"root_x_env": float(root_pos_env[0].item())' in script_text
