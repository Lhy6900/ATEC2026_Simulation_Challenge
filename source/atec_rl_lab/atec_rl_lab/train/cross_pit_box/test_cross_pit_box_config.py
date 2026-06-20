import ast
import importlib.util
import math
from pathlib import Path

import gymnasium as gym
import pytest
import torch


def _literal_constants(path: Path) -> dict[str, object]:
    constants = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                continue
    return constants


def test_cross_pit_box_task_registers_rsl_rl_entry_points_and_replaces_box_bridge():
    import atec_rl_lab.train.cross_pit_box  # noqa: F401

    v0_spec = gym.spec("ATEC-Isaac-TaskD-G1-CrossPitBox-v0")
    v1_spec = gym.spec("ATEC-Isaac-TaskD-G1-CrossPitBox-v1")
    v1_5_spec = gym.spec("ATEC-Isaac-TaskD-G1-CrossPitBox-v1.5")
    v2_spec = gym.spec("ATEC-Isaac-TaskD-G1-CrossPitBox-v2")
    v2_5_spec = gym.spec("ATEC-Isaac-TaskD-G1-CrossPitBox-v2.5")

    assert v0_spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert v0_spec.kwargs["env_cfg_entry_point"].endswith("env_cfg:CrossPitBoxG1EnvCfg")
    assert v0_spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
        "agents.rsl_rl_ppo_cfg:CrossPitBoxG1PPORunnerCfg"
    )
    assert v1_spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert v1_spec.kwargs["env_cfg_entry_point"].endswith("env_cfg_v1:CrossPitBoxG1V1EnvCfg")
    assert v1_spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
        "agents.rsl_rl_ppo_cfg:CrossPitBoxG1PPORunnerCfg"
    )
    assert v1_5_spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert v1_5_spec.kwargs["env_cfg_entry_point"].endswith("env_cfg_v1_5:CrossPitBoxG1V15EnvCfg")
    assert v1_5_spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
        "agents.rsl_rl_ppo_cfg:CrossPitBoxG1PPORunnerCfg"
    )
    assert v2_spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert v2_spec.kwargs["env_cfg_entry_point"].endswith("env_cfg_v2:CrossPitBoxG1V2EnvCfg")
    assert v2_spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
        "agents.rsl_rl_ppo_cfg:CrossPitBoxG1PPORunnerCfg"
    )
    assert v2_5_spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert v2_5_spec.kwargs["env_cfg_entry_point"].endswith("env_cfg_v2_5:CrossPitBoxG1V25EnvCfg")
    assert v2_5_spec.kwargs["rsl_rl_cfg_entry_point"].endswith(
        "agents.rsl_rl_ppo_cfg:CrossPitBoxG1PPORunnerCfg"
    )
    assert importlib.util.find_spec("atec_rl_lab.train.box_bridge") is None


def test_cross_pit_box_backtrack_guard_leaves_reset_noise_margin():
    env_cfg_path = Path(__file__).with_name("env_cfg.py")
    constants = {}
    for node in ast.parse(env_cfg_path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue

    start_x = constants["COUNTER650_ROBOT_POSE"][0]
    min_x = constants["CROSS_PIT_MIN_X"]

    assert min_x <= start_x - 0.70


def test_cross_pit_box_v0_keeps_checkpoint_handoff_constants():
    env_cfg_path = Path(__file__).with_name("env_cfg.py")
    constants = {}
    for node in ast.parse(env_cfg_path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue

    assert constants["COUNTER650_ROBOT_POSE"] == (-1.75, 0.0, 0.80, 0.0)
    assert constants["COUNTER650_BOX_SUPPORT_POSE"] == (-0.85, 0.0, -0.42, 0.0)


def test_cross_pit_box_config_text_uses_success_curriculum_and_far_side_rewards():
    env_cfg_text = Path(__file__).with_name("env_cfg.py").read_text()
    env_cfg_v1_text = Path(__file__).with_name("env_cfg_v1.py").read_text()

    assert "CROSS_PIT_PLATFORM_X = 1.10" in env_cfg_text
    assert "CROSS_PIT_SUCCESS_START_X = 1.85" in env_cfg_text
    assert "CROSS_PIT_SUCCESS_FINAL_X = 2.55" in env_cfg_text
    assert "success_x = CurrTerm" in env_cfg_text
    assert "incremental_crossing_progress = RewTerm" in env_cfg_text
    assert "far_platform_region = RewTerm" in env_cfg_text
    assert "func=mdp.crossing_progress,\n        weight=3.0" in env_cfg_text
    assert "func=mdp.box_support_region_reward,\n        weight=0.8" in env_cfg_text
    assert "func=mdp.crossing_success,\n        weight=120.0" in env_cfg_text
    assert "func=mdp.lateral_deviation_l2, weight=-1.6" in env_cfg_text
    assert "lateral_corridor = RewTerm" not in env_cfg_text
    assert "lateral_velocity_l2 = RewTerm" not in env_cfg_text
    assert "lateral_corridor = RewTerm" in env_cfg_v1_text
    assert "lateral_velocity_l2 = RewTerm" in env_cfg_v1_text


def test_cross_pit_box_v0_observations_preserve_checkpoint_compatible_shape_contract():
    env_cfg_path = Path(__file__).with_name("env_cfg.py")
    tree = ast.parse(env_cfg_path.read_text())

    observations_cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ObservationsCfg")
    policy_cls = next(node for node in observations_cls.body if isinstance(node, ast.ClassDef) and node.name == "PolicyCfg")
    critic_cls = next(node for node in observations_cls.body if isinstance(node, ast.ClassDef) and node.name == "CriticCfg")

    def assigned_terms(cls):
        return {
            stmt.targets[0].id
            for stmt in cls.body
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
        }

    policy_terms = assigned_terms(policy_cls)
    critic_terms = assigned_terms(critic_cls)
    critic_only_terms = {"base_pos_env", "box_center_lateral_error", "abs_lateral_error", "target_x_error"}

    assert "base_lin_vel" in policy_terms
    assert critic_only_terms.isdisjoint(policy_terms)
    assert critic_only_terms.isdisjoint(critic_terms)
    assert [base.id for base in critic_cls.bases if isinstance(base, ast.Name)] == ["PolicyCfg"]


def test_cross_pit_box_v1_actor_observations_exclude_privileged_base_linear_velocity():
    env_cfg_path = Path(__file__).with_name("env_cfg_v1.py")
    tree = ast.parse(env_cfg_path.read_text())

    observations_cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ObservationsCfg")
    policy_cls = next(node for node in observations_cls.body if isinstance(node, ast.ClassDef) and node.name == "PolicyCfg")
    critic_cls = next(node for node in observations_cls.body if isinstance(node, ast.ClassDef) and node.name == "CriticCfg")

    def assigned_terms(cls):
        return {
            stmt.targets[0].id
            for stmt in cls.body
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
        }

    policy_terms = assigned_terms(policy_cls)
    critic_terms = assigned_terms(critic_cls)
    critic_only_terms = {
        "base_pos_env",
        "box_center_lateral_error",
        "abs_lateral_error",
        "target_x_error",
        "crossing_stage_privileged",
        "box_gap_privileged",
    }

    assert "base_lin_vel" not in policy_terms
    assert {"base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "depth_heightmap"} <= policy_terms
    assert "base_lin_vel" in critic_terms
    assert "depth_heightmap" in critic_terms
    assert critic_only_terms.isdisjoint(policy_terms)
    assert critic_only_terms <= critic_terms
    assert [base.id for base in critic_cls.bases if isinstance(base, ast.Name)] == ["ObsGroup"]


def test_cross_pit_box_v1_5_disables_heightmap_observations_for_blind_walking():
    env_cfg_path = Path(__file__).with_name("env_cfg_v1_5.py")
    tree = ast.parse(env_cfg_path.read_text())

    observations_cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BlindObservationsCfg")
    policy_cls = next(node for node in observations_cls.body if isinstance(node, ast.ClassDef) and node.name == "PolicyCfg")
    critic_cls = next(node for node in observations_cls.body if isinstance(node, ast.ClassDef) and node.name == "CriticCfg")

    def none_assigned_terms(cls):
        terms = set()
        for stmt in cls.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is None
            ):
                terms.add(stmt.targets[0].id)
        return terms

    cfg_text = env_cfg_path.read_text()

    assert "depth_heightmap" in none_assigned_terms(policy_cls)
    assert "depth_heightmap" in none_assigned_terms(critic_cls)
    assert "raycast_heightmap" not in cfg_text
    assert "self.scene.depth_scanner = None" in cfg_text


def test_cross_pit_box_v1_uses_gui_counter650_snapshot_and_robot_reset_window():
    env_cfg_path = Path(__file__).with_name("env_cfg_v1.py")
    constants = _literal_constants(env_cfg_path)
    cfg_text = env_cfg_path.read_text()

    robot_pose = constants["COUNTER650_ROBOT_POSE"]
    box_pose = constants["COUNTER650_BOX_SUPPORT_POSE"]

    assert robot_pose[:3] == pytest.approx((2.4567047357559204, -0.9189060926437378, 0.733408510684967))
    assert box_pose[:3] == pytest.approx((4.363436430692673, -0.7996934652328491, -0.6000000238418579))
    assert len(robot_pose) == 7
    assert len(box_pose) == 7
    assert robot_pose[3:] == pytest.approx(
        (0.9904746413230896, 0.033361710608005524, -0.053021665662527084, 0.12262003868818283)
    )
    assert box_pose[3:] == pytest.approx(
        (0.707106351852417, -1.4099019608693197e-05, 0.7071070671081543, 1.3773496903013438e-05)
    )

    assert '"pose_noise": {"x": (-0.10, 0.10), "y": (-0.10, 0.10), "yaw": (-0.2617993877991494, 0.2617993877991494)}' in cfg_text
    joint_pos = constants["COUNTER650_JOINT_POS"]
    assert joint_pos["left_hip_pitch_joint"] == pytest.approx(-0.2877374589443207)
    assert joint_pos["right_ankle_pitch_joint"] == pytest.approx(-0.6566326022148132)
    assert "rot=_pose_quat(COUNTER650_ROBOT_POSE)" in cfg_text
    assert "rot=_pose_quat(COUNTER650_BOX_SUPPORT_POSE)" in cfg_text


def test_cross_pit_box_v2_uses_verified_play_9p5s_snapshot_and_handoff_action():
    env_cfg_path = Path(__file__).with_name("env_cfg_v2.py")
    constants = _literal_constants(env_cfg_path)
    cfg_text = env_cfg_path.read_text()

    robot_pose = constants["COUNTER950_ROBOT_POSE"]
    box_pose = constants["COUNTER950_BOX_SUPPORT_POSE"]

    assert robot_pose[:3] == pytest.approx((3.4137089252471924, -1.5012480020523071, 0.6975134611129761))
    assert box_pose[:3] == pytest.approx((4.36343640089035, -0.7996934652328491, -0.5999999642372131))
    assert len(robot_pose) == 7
    assert len(box_pose) == 7
    assert robot_pose[3:] == pytest.approx(
        (0.9921770691871643, -0.00885490607470274, -0.1043638288974762, -0.06793048232793808)
    )
    assert box_pose[3:] == pytest.approx(
        (0.7071065902709961, -1.4027000361238606e-05, 0.70710688829422, 1.3851278708898462e-05)
    )
    assert constants["CROSS_PIT_TARGET_Y"] == pytest.approx(box_pose[1])
    assert constants["CROSS_PIT_START_X"] == pytest.approx(robot_pose[0])
    assert constants["CROSS_PIT_BOX_X"] == pytest.approx(box_pose[0])
    assert constants["CROSS_PIT_MIN_X"] == pytest.approx(robot_pose[0] - 0.75)
    assert constants["COUNTER950_ROBOT_ROOT_VELOCITY"] == pytest.approx(
        (
            0.25904640555381775,
            -0.4047621786594391,
            0.09794872254133224,
            0.8786299228668213,
            -1.0161986351013184,
            1.1399791240692139,
        )
    )
    assert '"joint_pos_noise": 0.06' in cfg_text
    assert '"pose_noise": {"x": (-0.10, 0.10), "y": (-0.12, 0.12), "yaw": (-0.2617993877991494, 0.2617993877991494)}' in cfg_text
    assert '"joint_vel_overrides": COUNTER950_JOINT_VEL' in cfg_text
    assert '"last_action": COUNTER950_LAST_ACTION' in cfg_text
    assert '"normalize_by_step_dt": True' in cfg_text
    assert cfg_text.count('"failure_min_height": 0.30') >= 2
    assert cfg_text.count('"failure_max_abs_gravity_xy": 0.75') >= 2
    assert cfg_text.count('"max_abs_gravity_xy": 0.75') >= 5

    joint_pos = constants["COUNTER950_JOINT_POS"]
    assert joint_pos["left_hip_pitch_joint"] == pytest.approx(-0.45682671666145325)
    assert joint_pos["right_hip_pitch_joint"] == pytest.approx(-0.5572284460067749)
    assert joint_pos["waist_yaw_joint"] == pytest.approx(0.20661497116088867)
    assert joint_pos["right_ankle_roll_joint"] == pytest.approx(0.018568988889455795)
    joint_vel = constants["COUNTER950_JOINT_VEL"]
    assert joint_vel["right_knee_joint"] == pytest.approx(-4.717284679412842)
    assert joint_vel["right_ankle_pitch_joint"] == pytest.approx(3.7889597415924072)
    last_action = constants["COUNTER950_LAST_ACTION"]
    assert len(last_action) == 33
    assert last_action[:8] == pytest.approx(
        (
            -0.5318777561187744,
            0.06515301764011383,
            -0.2943190634250641,
            1.0854837894439697,
            2.623720645904541,
            -0.05229632556438446,
            -0.6801775693893433,
            -0.058367013931274414,
        )
    )
    assert "rot=_pose_quat(COUNTER950_ROBOT_POSE)" in cfg_text
    assert "rot=_pose_quat(COUNTER950_BOX_SUPPORT_POSE)" in cfg_text


def test_cross_pit_box_v2_5_disables_heightmap_observations_for_blind_walking():
    env_cfg_path = Path(__file__).with_name("env_cfg_v2_5.py")
    tree = ast.parse(env_cfg_path.read_text())

    observations_cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BlindObservationsCfg")
    policy_cls = next(node for node in observations_cls.body if isinstance(node, ast.ClassDef) and node.name == "PolicyCfg")
    critic_cls = next(node for node in observations_cls.body if isinstance(node, ast.ClassDef) and node.name == "CriticCfg")

    def none_assigned_terms(cls):
        terms = set()
        for stmt in cls.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is None
            ):
                terms.add(stmt.targets[0].id)
        return terms

    cfg_text = env_cfg_path.read_text()

    assert "depth_heightmap" in none_assigned_terms(policy_cls)
    assert "depth_heightmap" in none_assigned_terms(critic_cls)
    assert "raycast_heightmap" not in cfg_text
    assert "self.scene.depth_scanner = None" in cfg_text
    assert "CrossPitBoxG1V2EnvCfg" in cfg_text


def test_cross_pit_box_robot_reset_clamps_noisy_joint_positions_to_hard_limits():
    events_text = Path(__file__).parent.joinpath("mdp", "events.py").read_text()

    assert "joint_pos = torch.clamp(joint_pos, joint_limits[..., 0], joint_limits[..., 1])" in events_text
    assert "asset.data.joint_pos_limits[env_ids]" in events_text


def test_cross_pit_box_robot_reset_supports_handoff_velocity_and_first_action():
    events_text = Path(__file__).parent.joinpath("mdp", "events.py").read_text()
    observations_text = Path(__file__).parent.joinpath("mdp", "observations.py").read_text()

    assert "root_velocity: tuple[float, ...] | None = None" in events_text
    assert "joint_vel_overrides: dict[str, float] | None = None" in events_text
    assert "last_action: tuple[float, ...] | None = None" in events_text
    assert 'env._cross_pit_reset_last_action = stored_action' in events_text
    assert 'episode_length_buf == 0' in observations_text
    assert 'torch.where(reset_mask.unsqueeze(-1), reset_action, action)' in observations_text
    assert "sync_handoff_action_state" in observations_text
    assert "_processed_actions[reset_mask]" in observations_text
    assert "set_joint_position_target" in observations_text


def test_cross_pit_box_handoff_action_sync_updates_manager_term_and_joint_target():
    from atec_rl_lab.train.cross_pit_box.mdp.observations import sync_handoff_action_state

    class DummyAsset:
        def __init__(self):
            self.target = None
            self.joint_ids = None

        def set_joint_position_target(self, target, joint_ids=None):
            self.target = target.clone()
            self.joint_ids = joint_ids

    class DummyTerm:
        action_dim = 3

        def __init__(self):
            self._raw_actions = torch.zeros(2, 3)
            self._processed_actions = torch.zeros(2, 3)
            self._scale = 0.5
            self._offset = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
            self._asset = DummyAsset()
            self._joint_ids = [4, 5, 6]

        @property
        def raw_actions(self):
            return self._raw_actions

        @property
        def processed_actions(self):
            return self._processed_actions

    class DummyActionManager:
        def __init__(self):
            self._action = torch.zeros(2, 3)
            self._prev_action = torch.zeros(2, 3)
            self.term = DummyTerm()

        @property
        def action(self):
            return self._action

        @property
        def prev_action(self):
            return self._prev_action

        def get_term(self, name):
            assert name == "joint_pos"
            return self.term

    class DummyEnv:
        pass

    env = DummyEnv()
    env.action_manager = DummyActionManager()
    env.episode_length_buf = torch.tensor([0, 2])
    env._cross_pit_reset_last_action = torch.tensor([[2.0, -2.0, 1.0], [9.0, 9.0, 9.0]])

    action = sync_handoff_action_state(env, action_name="joint_pos")

    assert action.tolist() == [[2.0, -2.0, 1.0], [0.0, 0.0, 0.0]]
    assert env.action_manager.prev_action.tolist() == [[2.0, -2.0, 1.0], [0.0, 0.0, 0.0]]
    assert env.action_manager.term.raw_actions.tolist() == [[2.0, -2.0, 1.0], [0.0, 0.0, 0.0]]
    assert env.action_manager.term.processed_actions.tolist() == [[2.0, 1.0, 3.5], [0.0, 0.0, 0.0]]
    assert env.action_manager.term._asset.target.tolist() == [[2.0, 1.0, 3.5], [0.0, 0.0, 0.0]]
    assert env.action_manager.term._asset.joint_ids == [4, 5, 6]


def test_cross_pit_box_v1_crossing_targets_are_ahead_of_box_and_centered_on_box_y():
    env_cfg_path = Path(__file__).with_name("env_cfg_v1.py")
    constants = _literal_constants(env_cfg_path)
    cfg_text = env_cfg_path.read_text()

    start_x = constants["CROSS_PIT_START_X"]
    down_step_x = constants["CROSS_PIT_DOWN_STEP_X"]
    box_x = constants["CROSS_PIT_BOX_X"]
    far_lip_x = constants["CROSS_PIT_FAR_LIP_X"]
    success_start_x = constants["CROSS_PIT_SUCCESS_START_X"]
    success_final_x = constants["CROSS_PIT_SUCCESS_FINAL_X"]
    target_y = constants["CROSS_PIT_TARGET_Y"]

    assert start_x < success_start_x < down_step_x < box_x < far_lip_x < success_final_x
    assert success_start_x == pytest.approx(3.0)
    assert down_step_x == pytest.approx(3.72, abs=0.02)
    assert far_lip_x == pytest.approx(4.68, abs=0.02)
    assert success_start_x < box_x < success_final_x
    assert success_final_x == pytest.approx(7.80)
    assert success_final_x - box_x >= 1.3
    assert constants["CROSS_PIT_MIN_X"] == pytest.approx(start_x - 0.75)
    assert target_y == pytest.approx(constants["COUNTER650_BOX_SUPPORT_POSE"][1])

    assert '"target_y": CROSS_PIT_TARGET_Y' in cfg_text
    assert '"box_y": CROSS_PIT_TARGET_Y' in cfg_text
    assert 'lateral_out = DoneTerm(func=mdp.lateral_out_of_bounds, params={"max_abs_y": 1.25, "target_y": CROSS_PIT_TARGET_Y}' in cfg_text
    assert 'CROSS_PIT_SUCCESS_START_X = 3.00' in cfg_text
    assert 'CROSS_PIT_SUCCESS_FINAL_X = 7.80' in cfg_text


def test_cross_pit_box_v2_crossing_targets_are_ahead_of_9p5s_box_and_centered_on_box_y():
    env_cfg_path = Path(__file__).with_name("env_cfg_v2.py")
    constants = _literal_constants(env_cfg_path)
    cfg_text = env_cfg_path.read_text()

    start_x = constants["CROSS_PIT_START_X"]
    down_step_x = constants["CROSS_PIT_DOWN_STEP_X"]
    box_x = constants["CROSS_PIT_BOX_X"]
    far_lip_x = constants["CROSS_PIT_FAR_LIP_X"]
    success_start_x = constants["CROSS_PIT_SUCCESS_START_X"]
    success_final_x = constants["CROSS_PIT_SUCCESS_FINAL_X"]
    target_y = constants["CROSS_PIT_TARGET_Y"]

    assert start_x < success_start_x <= box_x < far_lip_x < success_final_x
    assert success_start_x == pytest.approx(down_step_x)
    assert down_step_x == pytest.approx(3.72, abs=0.03)
    assert far_lip_x == pytest.approx(4.68, abs=0.03)
    assert success_start_x < box_x < success_final_x
    assert success_final_x == pytest.approx(7.80)
    assert success_final_x - down_step_x == pytest.approx(4.08, abs=0.03)
    assert success_final_x - box_x >= 1.3
    assert target_y == pytest.approx(constants["COUNTER950_BOX_SUPPORT_POSE"][1])

    assert '"target_y": CROSS_PIT_TARGET_Y' in cfg_text
    assert '"box_y": CROSS_PIT_TARGET_Y' in cfg_text
    assert 'CROSS_PIT_SUCCESS_START_X = 3.72' in cfg_text
    assert 'CROSS_PIT_SUCCESS_FINAL_X = 7.80' in cfg_text
    assert '"report_relative_to_x": CROSS_PIT_DOWN_STEP_X' in cfg_text


def test_cross_pit_box_v1_records_rotated_box_gap_geometry_for_stage_training():
    env_cfg_path = Path(__file__).with_name("env_cfg_v1.py")
    constants = _literal_constants(env_cfg_path)

    assert constants["CROSS_PIT_BOX_STAGE_TARGET_X"] == pytest.approx(constants["CROSS_PIT_BOX_X"], abs=0.02)
    assert constants["CROSS_PIT_NEAR_BOX_GAP_X"] == pytest.approx(0.34, abs=0.05)
    assert constants["CROSS_PIT_BOX_FRONT_X"] == pytest.approx(4.0634, abs=0.02)
    assert constants["CROSS_PIT_BOX_BACK_X"] == pytest.approx(4.6635, abs=0.02)
    assert constants["CROSS_PIT_BOX_TOP_Z"] == pytest.approx(-0.20, abs=0.02)
    assert constants["CROSS_PIT_BOX_FRONT_X"] - constants["CROSS_PIT_DOWN_STEP_X"] > 0.25
    assert constants["CROSS_PIT_FAR_LIP_X"] - constants["CROSS_PIT_BOX_BACK_X"] < 0.05


def test_cross_pit_box_v2_records_9p5s_box_gap_geometry_for_stage_training():
    env_cfg_path = Path(__file__).with_name("env_cfg_v2.py")
    constants = _literal_constants(env_cfg_path)

    assert constants["CROSS_PIT_BOX_STAGE_TARGET_X"] == pytest.approx(constants["CROSS_PIT_BOX_X"], abs=0.02)
    assert constants["CROSS_PIT_NEAR_BOX_GAP_X"] == pytest.approx(0.34, abs=0.05)
    assert constants["CROSS_PIT_BOX_FRONT_X"] == pytest.approx(4.0634, abs=0.03)
    assert constants["CROSS_PIT_BOX_BACK_X"] == pytest.approx(4.6635, abs=0.03)
    assert constants["CROSS_PIT_BOX_TOP_Z"] == pytest.approx(-0.20, abs=0.03)
    assert constants["CROSS_PIT_BOX_FRONT_X"] - constants["CROSS_PIT_DOWN_STEP_X"] > 0.25
    assert constants["CROSS_PIT_FAR_LIP_X"] - constants["CROSS_PIT_BOX_BACK_X"] < 0.05


def test_cross_pit_box_v2_sticky_stage_uses_9p5s_box_front_and_center():
    env_cfg_text = Path(__file__).with_name("env_cfg_v2.py").read_text()

    assert '"sticky_start_x": CROSS_PIT_BOX_FRONT_X' in env_cfg_text
    assert '"retreat_floor_x": CROSS_PIT_BOX_STAGE_TARGET_X' in env_cfg_text
    assert '"unlock_success_x": CROSS_PIT_BOX_STAGE_TARGET_X' in env_cfg_text


def test_cross_pit_box_v1_uses_adaptive_success_curriculum_from_termination_rates():
    env_cfg_text = Path(__file__).with_name("env_cfg_v1.py").read_text()

    assert "func=mdp.update_cross_pit_success_x_sticky_stage" in env_cfg_text
    assert '"update_interval_iterations": 10' in env_cfg_text
    assert '"success_threshold": 0.75' in env_cfg_text
    assert '"fast_success_threshold": 0.90' in env_cfg_text
    assert '"max_lateral_out_rate": 0.15' in env_cfg_text
    assert '"sticky_start_x": CROSS_PIT_BOX_FRONT_X' in env_cfg_text
    assert '"retreat_floor_x": CROSS_PIT_BOX_STAGE_TARGET_X' in env_cfg_text
    assert 'stage_time_pressure = RewTerm' in env_cfg_text
    assert 'pbrs_course_progress = RewTerm' in env_cfg_text
    assert 'pbrs_stage_progress = RewTerm' in env_cfg_text
    assert 'stage_stall = RewTerm' in env_cfg_text
    assert 'box_parking = RewTerm' in env_cfg_text
    assert '"timeout_plateau_advance_step": 0.20' in env_cfg_text
    assert '"timeout_plateau_threshold": 0.85' in env_cfg_text
    assert '"timeout_plateau_unlock_x": CROSS_PIT_BOX_STAGE_TARGET_X' in env_cfg_text
    assert '"unlock_success_x": CROSS_PIT_DOWN_STEP_X' in env_cfg_text
    assert env_cfg_text.count('"unlock_success_x": CROSS_PIT_BOX_FRONT_X') >= 2
    assert '"start_iteration"' not in env_cfg_text
    assert '"end_iteration"' not in env_cfg_text


def test_cross_pit_box_v1_hard_stage_rewards_directly_pressure_timeout_escape():
    env_cfg_text = Path(__file__).with_name("env_cfg_v1.py").read_text()

    assert "stage_time_pressure = RewTerm(\n        func=mdp.stage_time_penalty,\n        weight=-0.08" in env_cfg_text
    assert "box_stability = RewTerm(\n        func=mdp.box_stability_reward,\n        weight=0.5" in env_cfg_text
    assert "crossing_progress = None" in env_cfg_text
    assert "stage_progress_to_target = None" in env_cfg_text
    assert "pbrs_course_progress = RewTerm(\n        func=mdp.potential_based_x_progress,\n        weight=20.0" in env_cfg_text
    assert "pbrs_stage_progress = RewTerm(\n        func=mdp.potential_based_x_progress,\n        weight=15.0" in env_cfg_text
    assert '"gamma": 0.99' in env_cfg_text
    assert '"potential_scale": 50.0' in env_cfg_text
    assert "stage_stall = RewTerm(\n        func=mdp.stage_stall_penalty,\n        weight=-5.0" in env_cfg_text
    assert "far_side_commit = RewTerm(\n        func=mdp.far_side_commit_reward,\n        weight=6.0" in env_cfg_text
    assert '"min_forward_velocity": 0.12' in env_cfg_text
    assert '"target_margin": 0.20' in env_cfg_text


def test_success_x_curriculum_ramps_from_warmup_to_final_target():
    from atec_rl_lab.train.cross_pit_box.mdp.curriculums import (
        cross_pit_success_x_value,
        update_cross_pit_success_x,
    )

    class _Env:
        common_step_counter = 0
        device = "cpu"

    env = _Env()

    def target_at(iteration: int) -> float:
        env.common_step_counter = iteration * 32
        return cross_pit_success_x_value(
            env,
            start_x=1.85,
            final_x=2.55,
            start_iteration=1000,
            end_iteration=6000,
            steps_per_iteration=32,
        )

    assert target_at(0) == pytest.approx(1.85)
    assert target_at(1000) == pytest.approx(1.85)
    assert target_at(3500) == pytest.approx(2.20)
    assert target_at(6000) == pytest.approx(2.55)
    assert target_at(9000) == pytest.approx(2.55)

    env.common_step_counter = 3500 * 32
    state = update_cross_pit_success_x(
        env,
        env_ids=slice(None),
        start_x=1.85,
        final_x=2.55,
        start_iteration=1000,
        end_iteration=6000,
        steps_per_iteration=32,
    )

    assert env.cross_pit_success_x == pytest.approx(2.20)
    assert state["target_x"] == pytest.approx(2.20)
    assert state["progress"] == pytest.approx(0.5)


def test_adaptive_success_x_curriculum_advances_from_crossed_rate_and_pauses_on_failures():
    from atec_rl_lab.train.cross_pit_box.mdp.curriculums import update_cross_pit_success_x_adaptive

    class _TerminationManager:
        def __init__(self):
            self.rates = {"crossed": 0.0, "fallen": 0.0, "lateral_out": 0.0}

        def get_term(self, name: str) -> torch.Tensor:
            rate = self.rates[name]
            true_count = int(round(rate * 10))
            return torch.tensor([True] * true_count + [False] * (10 - true_count))

    class _Env:
        common_step_counter = 0
        device = "cpu"

        def __init__(self):
            self.termination_manager = _TerminationManager()

    env = _Env()

    def update(iteration: int, crossed: float, fallen: float, lateral_out: float) -> dict[str, torch.Tensor]:
        env.common_step_counter = iteration * 32
        env.termination_manager.rates = {"crossed": crossed, "fallen": fallen, "lateral_out": lateral_out}
        return update_cross_pit_success_x_adaptive(
            env,
            env_ids=slice(None),
            start_x=3.0,
            final_x=5.8,
            steps_per_iteration=32,
            min_iteration=0,
            update_interval_iterations=1,
            advance_step=0.05,
            fast_advance_step=0.10,
            retreat_step=0.025,
            success_threshold=0.75,
            fast_success_threshold=0.90,
            max_fallen_rate=0.25,
            max_lateral_out_rate=0.15,
            retreat_crossed_threshold=0.35,
            retreat_failure_rate=0.45,
            ema_alpha=1.0,
        )

    state = update(iteration=1, crossed=0.9, fallen=0.1, lateral_out=0.0)
    assert env.cross_pit_success_x == pytest.approx(3.05)
    assert state["target_x"] == pytest.approx(3.05)
    assert state["progress"] == pytest.approx((3.05 - 3.0) / (5.8 - 3.0))
    assert state["crossed_rate"] == pytest.approx(0.9)

    update(iteration=2, crossed=1.0, fallen=0.0, lateral_out=0.0)
    assert env.cross_pit_success_x == pytest.approx(3.15)

    update(iteration=3, crossed=1.0, fallen=0.0, lateral_out=0.5)
    assert env.cross_pit_success_x == pytest.approx(3.15)

    update(iteration=4, crossed=0.2, fallen=0.5, lateral_out=0.0)
    assert env.cross_pit_success_x == pytest.approx(3.125)


def test_adaptive_success_x_curriculum_uses_reset_env_subset_for_rates():
    from atec_rl_lab.train.cross_pit_box.mdp.curriculums import update_cross_pit_success_x_adaptive

    class _TerminationManager:
        def get_term(self, name: str) -> torch.Tensor:
            if name == "crossed":
                return torch.tensor([True] * 9 + [False] * 11)
            return torch.tensor([False] * 20)

    class _Env:
        common_step_counter = 32
        device = "cpu"
        termination_manager = _TerminationManager()

    env = _Env()
    state = update_cross_pit_success_x_adaptive(
        env,
        env_ids=torch.arange(10),
        start_x=3.0,
        final_x=5.8,
        update_interval_iterations=1,
        ema_alpha=1.0,
    )

    assert state["crossed_rate"] == pytest.approx(0.9)
    assert env.cross_pit_success_x == pytest.approx(3.05)


def test_sticky_stage_curriculum_does_not_retreat_below_box_stage_after_hard_failures():
    from atec_rl_lab.train.cross_pit_box.mdp.curriculums import update_cross_pit_success_x_sticky_stage

    class _TerminationManager:
        def __init__(self):
            self.rates = {"crossed": 0.0, "fallen": 1.0, "lateral_out": 0.0}

        def get_term(self, name: str) -> torch.Tensor:
            rate = self.rates[name]
            true_count = int(round(rate * 10))
            return torch.tensor([True] * true_count + [False] * (10 - true_count))

    class _Env:
        common_step_counter = 320
        device = "cpu"
        cross_pit_success_x = 4.8
        termination_manager = _TerminationManager()

    env = _Env()
    for iteration in range(10, 100, 10):
        env.common_step_counter = iteration * 32
        state = update_cross_pit_success_x_sticky_stage(
            env,
            env_ids=slice(None),
            start_x=3.0,
            final_x=5.8,
            retreat_floor_x=4.36,
            update_interval_iterations=1,
            retreat_step=0.10,
            ema_alpha=1.0,
        )

    assert env.cross_pit_success_x == pytest.approx(4.36)
    assert state["target_x"] == pytest.approx(4.36)
    assert state["stage"] >= 2.0


def test_sticky_stage_curriculum_latches_at_box_front_before_center_stage():
    from atec_rl_lab.train.cross_pit_box.mdp.curriculums import update_cross_pit_success_x_sticky_stage

    class _TerminationManager:
        def get_term(self, name: str) -> torch.Tensor:
            if name == "fallen":
                return torch.tensor([True] * 9 + [False])
            return torch.tensor([False] * 10)

    class _Env:
        common_step_counter = 320
        device = "cpu"
        cross_pit_success_x = 4.10
        termination_manager = _TerminationManager()

    env = _Env()
    for iteration in range(10, 100, 10):
        env.common_step_counter = iteration * 32
        state = update_cross_pit_success_x_sticky_stage(
            env,
            env_ids=slice(None),
            start_x=3.0,
            final_x=5.8,
            sticky_start_x=4.06,
            retreat_floor_x=4.06,
            update_interval_iterations=1,
            retreat_step=0.20,
            ema_alpha=1.0,
        )

    assert env.cross_pit_success_x == pytest.approx(4.06)
    assert state["target_x"] == pytest.approx(4.06)
    assert state["stage"] >= 2.0


def test_sticky_stage_curriculum_advances_out_of_timeout_parking_plateau():
    from atec_rl_lab.train.cross_pit_box.mdp.curriculums import update_cross_pit_success_x_sticky_stage

    class _TerminationManager:
        def get_term(self, name: str) -> torch.Tensor:
            if name == "time_out":
                return torch.tensor([True] * 10)
            if name in {"crossed", "fallen", "lateral_out"}:
                return torch.tensor([False] * 10)
            raise KeyError(name)

    class _Env:
        common_step_counter = 320
        device = "cpu"
        cross_pit_success_x = 4.36
        _cross_pit_sticky_stage_reached = True
        termination_manager = _TerminationManager()

    env = _Env()
    state = update_cross_pit_success_x_sticky_stage(
        env,
        env_ids=slice(None),
        start_x=3.0,
        final_x=5.8,
        sticky_start_x=4.06,
        retreat_floor_x=4.36,
        update_interval_iterations=1,
        timeout_plateau_advance_step=0.20,
        timeout_plateau_threshold=0.85,
        timeout_plateau_max_crossed_rate=0.05,
        timeout_plateau_max_fallen_rate=0.10,
        timeout_plateau_unlock_x=4.36,
        ema_alpha=1.0,
    )

    assert env.cross_pit_success_x == pytest.approx(4.56)
    assert state["target_x"] == pytest.approx(4.56)
    assert state["time_out_rate"] == pytest.approx(1.0)
    assert state["stage"] >= 2.0


def test_sticky_stage_curriculum_can_report_target_relative_to_handoff_line():
    from atec_rl_lab.train.cross_pit_box.mdp.curriculums import update_cross_pit_success_x_sticky_stage

    class _TerminationManager:
        def __init__(self, crossed_rate: float = 0.0):
            self.crossed_rate = crossed_rate

        def get_term(self, name: str) -> torch.Tensor:
            if name == "crossed":
                true_count = int(round(self.crossed_rate * 10))
                return torch.tensor([True] * true_count + [False] * (10 - true_count))
            return torch.tensor([False] * 10)

    class _Env:
        common_step_counter = 0
        device = "cpu"

        def __init__(self):
            self.termination_manager = _TerminationManager()

    env = _Env()
    state = update_cross_pit_success_x_sticky_stage(
        env,
        env_ids=slice(None),
        start_x=3.72,
        final_x=5.80,
        sticky_start_x=4.0634,
        retreat_floor_x=4.3634,
        report_relative_to_x=3.72,
        update_interval_iterations=1,
        ema_alpha=1.0,
    )

    assert env.cross_pit_success_x == pytest.approx(3.72)
    assert "target_x" not in state
    assert state["target_x_abs"] == pytest.approx(3.72)
    assert state["target_x_rel"] == pytest.approx(0.0)
    assert state["progress"] == pytest.approx(0.0)

    env.termination_manager.crossed_rate = 1.0
    env.common_step_counter = 32
    state = update_cross_pit_success_x_sticky_stage(
        env,
        env_ids=slice(None),
        start_x=3.72,
        final_x=5.80,
        sticky_start_x=4.0634,
        retreat_floor_x=4.3634,
        report_relative_to_x=3.72,
        update_interval_iterations=1,
        fast_success_threshold=0.9,
        fast_advance_step=0.10,
        ema_alpha=1.0,
    )

    assert env.cross_pit_success_x == pytest.approx(3.82)
    assert state["target_x_abs"] == pytest.approx(3.82)
    assert state["target_x_rel"] == pytest.approx(0.10)


def test_cross_pit_box_env_cfg_models_fixed_counter650_locomotion_scene():
    pytest.importorskip("pxr", reason="env config import requires Isaac/Omniverse runtime modules")

    from isaaclab.sensors import MultiMeshRayCasterCfg

    from atec_rl_lab.train.cross_pit_box.env_cfg import (
        COUNTER650_BOX_SUPPORT_POSE,
        COUNTER650_ROBOT_POSE,
        CROSS_PIT_BOX_GRID_SHAPE,
        CrossPitBoxG1EnvCfg,
    )

    cfg = CrossPitBoxG1EnvCfg()

    assert cfg.scene.robot.prim_path == "{ENV_REGEX_NS}/Robot"
    assert cfg.scene.box_support.prim_path == "{ENV_REGEX_NS}/BoxSupport"
    assert tuple(cfg.scene.box_support.init_state.pos) == COUNTER650_BOX_SUPPORT_POSE[:3]
    assert cfg.scene.box_support.spawn.rigid_props.kinematic_enabled is True
    assert tuple(cfg.scene.robot.init_state.pos) == COUNTER650_ROBOT_POSE[:3]
    assert cfg.observations.policy.depth_heightmap.params["grid_shape"] == CROSS_PIT_BOX_GRID_SHAPE
    assert isinstance(cfg.scene.depth_scanner, MultiMeshRayCasterCfg)
    assert cfg.scene.depth_scanner.mesh_prim_paths[0] == "/World/ground"
    assert cfg.rewards.track_lin_vel_xy_exp is None
    assert cfg.rewards.crossing_progress.weight > cfg.rewards.upright.weight
    assert cfg.terminations.crossed.params["success_x"] == 1.85
    assert cfg.terminations.crossed.params["success_x_attr"] == "cross_pit_success_x"


def test_cross_pit_box_support_uses_visible_material():
    env_cfg_text = Path(__file__).with_name("env_cfg.py").read_text()
    env_cfg_v1_text = Path(__file__).with_name("env_cfg_v1.py").read_text()

    assert "visual_material=sim_utils.PreviewSurfaceCfg" in env_cfg_text
    assert "diffuse_color=(0.85, 0.38, 0.10)" in env_cfg_text
    assert "visual_material=sim_utils.PreviewSurfaceCfg" in env_cfg_v1_text
    assert "diffuse_color=(0.85, 0.38, 0.10)" in env_cfg_v1_text


def test_stage_progress_rewards_down_box_far_side_milestones():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [-1.75, 0.0, 0.8],
                [-0.85, 0.0, 0.45],
                [0.40, 0.0, 0.75],
                [2.65, 0.0, 0.85],
            ]
        )
        root_lin_vel_w = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.4, 0.0, 0.0],
                [0.6, 0.0, 0.0],
                [0.1, 0.0, 0.0],
            ]
        )
        projected_gravity_b = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ]
        )

    class _Env:
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    env = _Env()
    progress = rewards.crossing_progress(env, start_x=-1.75, success_x=2.55)
    milestones = rewards.crossing_milestone_reward(
        env,
        down_step_x=-1.20,
        box_x=-0.85,
        far_lip_x=0.35,
        success_x=2.55,
        margin=0.15,
    )
    forward_vel = rewards.forward_velocity_reward(env, max_velocity=1.0)

    assert torch.all(progress[1:] > progress[:-1])
    assert torch.all(milestones[1:] > milestones[:-1])
    assert torch.allclose(forward_vel, torch.tensor([0.0, 0.4, 0.6, 0.1]))


def test_incremental_progress_rewards_forward_motion_without_position_harvesting():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [-1.70, 0.0, 0.8],
                [-0.80, 0.0, 0.8],
                [0.60, 0.0, 0.8],
                [1.20, 0.0, 0.8],
            ]
        )
        root_lin_vel_w = torch.tensor(
            [
                [0.20, 0.0, 0.0],
                [0.00, 0.0, 0.0],
                [0.50, 0.0, 0.0],
                [-0.10, 0.0, 0.0],
            ]
        )

    class _Env:
        step_dt = 0.02
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    reward = rewards.incremental_crossing_progress(_Env(), start_x=-1.75, success_x=2.55, max_step_progress=0.02)

    assert reward[0] > 0.0
    assert reward[1] == 0.0


def test_potential_based_x_progress_uses_gamma_phi_delta_and_resets_on_new_episode():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards

    class _AssetData:
        root_pos_w = torch.tensor([[2.0, -0.8, 0.8], [3.0, -0.8, 0.8]], dtype=torch.float32)
        projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=torch.float32)

    class _Env:
        num_envs = 2
        device = "cpu"
        episode_length_buf = torch.tensor([1, 1])
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    env = _Env()
    first = rewards.potential_based_x_progress(
        env,
        start_x=0.0,
        success_x=10.0,
        gamma=0.99,
        potential_scale=10.0,
        target_y=-0.8,
        safe_abs_y=0.5,
        min_upright=0.5,
        cache_key="_test_pbrs_phi",
    )
    assert torch.allclose(first, torch.zeros(2))

    env.episode_length_buf = torch.tensor([2, 2])
    _AssetData.root_pos_w = torch.tensor([[3.0, -0.8, 0.8], [3.0, -0.8, 0.8]], dtype=torch.float32)
    moved = rewards.potential_based_x_progress(
        env,
        start_x=0.0,
        success_x=10.0,
        gamma=0.99,
        potential_scale=10.0,
        target_y=-0.8,
        safe_abs_y=0.5,
        min_upright=0.5,
        cache_key="_test_pbrs_phi",
    )
    assert torch.allclose(moved, torch.tensor([0.97, -0.03]), atol=1.0e-5)

    parked = rewards.potential_based_x_progress(
        env,
        start_x=0.0,
        success_x=10.0,
        gamma=0.99,
        potential_scale=10.0,
        target_y=-0.8,
        safe_abs_y=0.5,
        min_upright=0.5,
        cache_key="_test_pbrs_phi",
    )
    assert torch.allclose(parked, torch.tensor([-0.03, -0.03]), atol=1.0e-5)

    env.episode_length_buf = torch.tensor([1, 1])
    _AssetData.root_pos_w = torch.tensor([[9.0, -0.8, 0.8], [1.0, -0.8, 0.8]], dtype=torch.float32)
    reset_first_step = rewards.potential_based_x_progress(
        env,
        start_x=0.0,
        success_x=10.0,
        gamma=0.99,
        potential_scale=10.0,
        target_y=-0.8,
        safe_abs_y=0.5,
        min_upright=0.5,
        cache_key="_test_pbrs_phi",
    )
    assert torch.allclose(reset_first_step, torch.zeros(2))


def test_potential_based_x_progress_zeroes_failed_state_potential():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards

    class _AssetData:
        root_pos_w = torch.tensor([[3.0, -0.8, 0.8], [3.0, -0.8, 0.8]], dtype=torch.float32)
        projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=torch.float32)

    class _Env:
        num_envs = 2
        device = "cpu"
        episode_length_buf = torch.tensor([1, 1])
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    env = _Env()
    first = rewards.potential_based_x_progress(
        env,
        start_x=0.0,
        success_x=10.0,
        gamma=0.99,
        potential_scale=10.0,
        target_y=-0.8,
        safe_abs_y=0.5,
        min_upright=0.5,
        failure_min_height=0.30,
        failure_max_abs_gravity_xy=0.75,
        cache_key="_test_failure_pbrs_phi",
    )
    assert torch.allclose(first, torch.zeros(2))

    env.episode_length_buf = torch.tensor([2, 2])
    _AssetData.root_pos_w = torch.tensor([[4.0, -0.8, 0.8], [4.0, -0.8, 0.8]], dtype=torch.float32)
    _AssetData.projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0], [0.8, 0.0, -0.6]], dtype=torch.float32)
    reward = rewards.potential_based_x_progress(
        env,
        start_x=0.0,
        success_x=10.0,
        gamma=0.99,
        potential_scale=10.0,
        target_y=-0.8,
        safe_abs_y=0.5,
        min_upright=0.5,
        failure_min_height=0.30,
        failure_max_abs_gravity_xy=0.75,
        cache_key="_test_failure_pbrs_phi",
    )

    assert torch.allclose(reward, torch.tensor([0.96, -3.0]), atol=1.0e-5)


def test_crossing_failure_penalty_can_avoid_step_dt_dilution():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards

    class _AssetData:
        root_pos_w = torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, 0.8]], dtype=torch.float32)
        projected_gravity_b = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=torch.float32)

    class _Env:
        step_dt = 0.02
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    penalty = rewards.crossing_failure_penalty(_Env(), min_height=0.30, normalize_by_step_dt=True)

    assert torch.allclose(penalty, torch.tensor([50.0, 0.0]))


def test_step_gap_and_box_stability_rewards_prefer_stride_to_box_over_shuffling():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [3.70, -0.80, 0.75],  # before lip, not stepping
                [4.05, -0.80, 0.32],  # crosses near gap with enough forward speed
                [4.36, -0.80, 0.32],  # upright on box
                [4.36, -0.20, 0.32],  # lateral miss on box
            ]
        )
        root_lin_vel_w = torch.tensor(
            [
                [0.20, 0.0, 0.0],
                [0.85, 0.0, 0.15],
                [0.20, 0.0, 0.0],
                [0.20, 0.0, 0.0],
            ]
        )
        projected_gravity_b = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ]
        )

    class _Env:
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    env = _Env()
    step_reward = rewards.box_gap_step_reward(
        env,
        near_lip_x=3.72,
        box_front_x=4.06,
        box_center_x=4.36,
        target_y=-0.80,
        min_forward_velocity=0.55,
        max_abs_y=0.35,
    )
    stability_reward = rewards.box_stability_reward(
        env,
        box_x=4.36,
        box_y=-0.80,
        half_length=0.32,
        half_width=0.45,
        min_height=0.20,
    )

    assert step_reward[1] > step_reward[0]
    assert step_reward[1] > step_reward[2]
    assert stability_reward[2] == pytest.approx(1.0)
    assert stability_reward[3] == pytest.approx(0.0)


def test_box_stage_rewards_are_locked_until_curriculum_reaches_gap_stage():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [4.05, -0.80, 0.32],
                [4.36, -0.80, 0.32],
            ]
        )
        root_lin_vel_w = torch.tensor(
            [
                [0.85, 0.0, 0.15],
                [0.20, 0.0, 0.0],
            ]
        )
        projected_gravity_b = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ]
        )

    class _Env:
        cross_pit_success_x = 3.0
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    env = _Env()

    locked_support = rewards.box_support_region_reward(
        env,
        box_x=4.36,
        box_y=-0.80,
        half_length=0.32,
        half_width=0.55,
        min_height=0.20,
        success_x=3.0,
        success_x_attr="cross_pit_success_x",
        unlock_success_x=4.06,
    )
    locked_step = rewards.box_gap_step_reward(
        env,
        near_lip_x=3.72,
        box_front_x=4.06,
        box_center_x=4.36,
        target_y=-0.80,
        success_x=3.0,
        success_x_attr="cross_pit_success_x",
        unlock_success_x=3.72,
    )
    locked_stability = rewards.box_stability_reward(
        env,
        box_x=4.36,
        box_y=-0.80,
        half_length=0.32,
        half_width=0.45,
        min_height=0.20,
        success_x=3.0,
        success_x_attr="cross_pit_success_x",
        unlock_success_x=4.06,
    )

    assert torch.allclose(locked_support, torch.zeros(2))
    assert torch.allclose(locked_step, torch.zeros(2))
    assert torch.allclose(locked_stability, torch.zeros(2))

    env.cross_pit_success_x = 4.10
    unlocked_support = rewards.box_support_region_reward(
        env,
        box_x=4.36,
        box_y=-0.80,
        half_length=0.32,
        half_width=0.55,
        min_height=0.20,
        success_x=3.0,
        success_x_attr="cross_pit_success_x",
        unlock_success_x=4.06,
    )
    unlocked_step = rewards.box_gap_step_reward(
        env,
        near_lip_x=3.72,
        box_front_x=4.06,
        box_center_x=4.36,
        target_y=-0.80,
        success_x=3.0,
        success_x_attr="cross_pit_success_x",
        unlock_success_x=3.72,
    )
    unlocked_stability = rewards.box_stability_reward(
        env,
        box_x=4.36,
        box_y=-0.80,
        half_length=0.32,
        half_width=0.45,
        min_height=0.20,
        success_x=3.0,
        success_x_attr="cross_pit_success_x",
        unlock_success_x=4.06,
    )

    assert unlocked_support[1] == pytest.approx(1.0)
    assert unlocked_step[0] > 0.0
    assert unlocked_stability[1] == pytest.approx(1.0)


def test_box_stability_reward_requires_controlled_forward_commitment_not_parking():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [4.36, -0.80, 0.32],  # parked on box
                [4.36, -0.80, 0.32],  # controlled forward continuation
                [4.36, -0.80, 0.32],  # too fast to count as stable
            ]
        )
        root_lin_vel_w = torch.tensor(
            [
                [0.02, 0.0, 0.0],
                [0.35, 0.0, 0.0],
                [0.95, 0.0, 0.0],
            ]
        )
        projected_gravity_b = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ]
        )

    class _Env:
        cross_pit_success_x = 4.36
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    reward = rewards.box_stability_reward(
        _Env(),
        box_x=4.36,
        box_y=-0.80,
        half_length=0.32,
        half_width=0.45,
        min_height=0.20,
        min_forward_velocity=0.15,
        max_forward_velocity=0.65,
        success_x=3.0,
        success_x_attr="cross_pit_success_x",
        unlock_success_x=4.06,
    )

    assert torch.allclose(reward, torch.tensor([0.0, 1.0, 0.0]))


def test_stage_time_penalty_turns_on_only_after_hard_stage_unlocks():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards

    class _Env:
        num_envs = 3
        device = "cpu"
        cross_pit_success_x = 3.0

    env = _Env()
    locked = rewards.stage_time_penalty(
        env,
        success_x=3.0,
        success_x_attr="cross_pit_success_x",
        unlock_success_x=4.06,
    )
    assert torch.allclose(locked, torch.zeros(3))

    env.cross_pit_success_x = 4.36
    unlocked = rewards.stage_time_penalty(
        env,
        success_x=3.0,
        success_x_attr="cross_pit_success_x",
        unlock_success_x=4.06,
    )
    assert torch.allclose(unlocked, torch.ones(3))


def test_stage_progress_to_target_tracks_active_curriculum_after_hard_stage_unlocks():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [4.05, -0.80, 0.32],  # still before the hard-stage start
                [4.36, -0.80, 0.32],  # controlled progress onto the box
                [5.80, -0.80, 0.80],  # current curriculum target reached
            ]
        )

    class _Env:
        cross_pit_success_x = 3.0
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    env = _Env()
    locked = rewards.stage_progress_to_target(
        env,
        stage_start_x=4.06,
        success_x=3.0,
        success_x_attr="cross_pit_success_x",
        unlock_success_x=4.06,
    )
    assert torch.allclose(locked, torch.zeros(3))

    env.cross_pit_success_x = 5.80
    progress = rewards.stage_progress_to_target(
        env,
        stage_start_x=4.06,
        success_x=3.0,
        success_x_attr="cross_pit_success_x",
        unlock_success_x=4.06,
    )

    assert progress[0] == pytest.approx(0.0)
    assert 0.0 < progress[1] < progress[2]
    assert progress[2] == pytest.approx(1.0)


def test_stage_stall_penalty_targets_centered_upright_timeout_behavior():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [4.36, -0.80, 0.32],  # centered and parked before target
                [4.36, -0.80, 0.32],  # moving forward enough
                [5.65, -0.80, 0.80],  # close enough to active target
                [4.36, -0.20, 0.32],  # lateral miss is handled by lateral terms
                [4.36, -0.80, 0.32],  # already falling/tilting
            ]
        )
        root_lin_vel_w = torch.tensor(
            [
                [0.03, 0.0, 0.0],
                [0.35, 0.0, 0.0],
                [0.03, 0.0, 0.0],
                [0.03, 0.0, 0.0],
                [0.03, 0.0, 0.0],
            ]
        )
        projected_gravity_b = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.8, 0.0, -0.4],
            ]
        )

    class _Env:
        cross_pit_success_x = 5.80
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    penalty = rewards.stage_stall_penalty(
        _Env(),
        stage_start_x=4.06,
        target_y=-0.80,
        success_x=3.0,
        success_x_attr="cross_pit_success_x",
        unlock_success_x=4.06,
        target_margin=0.20,
        min_forward_velocity=0.12,
        safe_abs_y=0.45,
        min_upright=0.75,
    )

    assert torch.allclose(penalty, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))


def test_box_parking_penalty_targets_stalled_box_states_before_active_goal():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [4.10, -0.80, 0.32],  # on/near box front but not close to target
                [4.34, -0.80, 0.32],  # close enough to active target
                [3.80, -0.80, 0.32],  # still before box front
                [4.10, -0.20, 0.32],  # lateral miss should not dominate this term
            ]
        )
        root_lin_vel_w = torch.tensor(
            [
                [0.03, 0.0, 0.0],
                [0.03, 0.0, 0.0],
                [0.03, 0.0, 0.0],
                [0.03, 0.0, 0.0],
            ]
        )

    class _Env:
        cross_pit_success_x = 4.36
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    penalty = rewards.box_parking_penalty(
        _Env(),
        box_front_x=4.06,
        target_y=-0.80,
        success_x=3.0,
        success_x_attr="cross_pit_success_x",
        unlock_success_x=4.06,
        target_margin=0.12,
        max_forward_velocity=0.08,
        max_abs_y=0.35,
    )

    assert torch.allclose(penalty, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_far_platform_reward_prefers_centered_upright_far_side_state():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [0.55, 0.00, 0.82],
                [1.30, 0.00, 0.86],
                [1.30, 0.75, 0.86],
                [1.30, 0.00, 0.30],
            ]
        )
        projected_gravity_b = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.8, 0.0, -0.4],
            ]
        )

    class _Env:
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    reward = rewards.far_platform_region_reward(
        _Env(),
        far_lip_x=0.35,
        platform_x=1.10,
        half_width=0.45,
        min_height=0.55,
        margin=0.20,
    )

    assert reward[1] > reward[0]
    assert reward[1] > 0.95
    assert reward[2] == 0.0
    assert reward[3] == 0.0


def test_far_side_commit_reward_activates_after_leaving_box_toward_platform():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [4.55, -0.80, 0.32],  # still on box / before far lip
                [4.85, -0.80, 0.58],  # committed past far lip
                [5.20, -0.80, 0.82],  # upright on far side
                [4.85, -0.20, 0.58],  # lateral miss
                [4.85, -0.80, 0.30],  # too low after lip
            ]
        )
        root_lin_vel_w = torch.tensor(
            [
                [0.20, 0.0, 0.0],
                [0.45, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.45, 0.0, 0.0],
                [0.45, 0.0, 0.0],
            ]
        )
        projected_gravity_b = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ]
        )

    class _Env:
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    reward = rewards.far_side_commit_reward(
        _Env(),
        far_lip_x=4.68,
        platform_x=5.10,
        target_y=-0.80,
        half_width=0.45,
        min_height=0.50,
        min_forward_velocity=0.20,
        margin=0.20,
    )

    assert reward[0] == pytest.approx(0.0)
    assert 0.0 < reward[1] < reward[2]
    assert reward[2] > 0.5
    assert reward[3] == pytest.approx(0.0)
    assert reward[4] == pytest.approx(0.0)


def test_crossing_terms_use_env_relative_positions_for_replicated_scenes():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards, terminations

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [8.25, 0.0, 0.80],
                [-7.35, 0.0, 0.80],
                [7.40, 0.0, 0.80],
            ]
        )
        root_lin_vel_w = torch.zeros(3, 3)
        projected_gravity_b = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ]
        )

    class _Scene(dict):
        env_origins = torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [-10.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ]
        )

    class _Env:
        scene = _Scene({"robot": type("_Asset", (), {"data": _AssetData})()})

    env = _Env()
    progress = rewards.crossing_progress(env, start_x=-1.75, success_x=2.55)
    backtracked = terminations.backtracked(env, min_x=-2.50)
    crossed = terminations.crossed_target(env, success_x=2.55, min_height=0.55)

    assert torch.allclose(progress, torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(backtracked, torch.tensor([False, False, True]))
    assert torch.equal(crossed, torch.tensor([False, True, False]))


def test_crossing_success_and_termination_use_curriculum_target_from_env():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards, terminations

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [2.00, 0.0, 0.80],
                [2.60, 0.0, 0.80],
            ]
        )
        root_lin_vel_w = torch.zeros(2, 3)
        projected_gravity_b = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ]
        )

    class _Env:
        cross_pit_success_x = 2.55
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    env = _Env()
    success_reward = rewards.crossing_success(
        env,
        success_x=1.85,
        success_x_attr="cross_pit_success_x",
        min_height=0.55,
    )
    crossed = terminations.crossed_target(
        env,
        success_x=1.85,
        success_x_attr="cross_pit_success_x",
        min_height=0.55,
    )
    progress = rewards.crossing_progress(
        env,
        start_x=-1.75,
        success_x=1.85,
        success_x_attr="cross_pit_success_x",
    )

    assert torch.equal(success_reward, torch.tensor([0.0, 1.0]))
    assert torch.equal(crossed, torch.tensor([False, True]))
    assert progress[0] < 1.0
    assert progress[1] == 1.0


def test_crossing_success_and_termination_require_centered_lateral_position():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards, terminations

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [2.60, 0.00, 0.80],
                [2.60, 0.54, 0.80],
                [2.60, 0.70, 0.80],
            ]
        )
        projected_gravity_b = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ]
        )

    class _Env:
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    env = _Env()
    success_reward = rewards.crossing_success(env, success_x=2.55, min_height=0.55, max_abs_y=0.55)
    crossed = terminations.crossed_target(env, success_x=2.55, min_height=0.55, max_abs_y=0.55)

    assert torch.equal(success_reward, torch.tensor([1.0, 1.0, 0.0]))
    assert torch.equal(crossed, torch.tensor([True, True, False]))


def test_crossing_success_and_termination_center_on_nonzero_box_line():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards, terminations

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [5.55, -0.80, 0.80],
                [5.55, 0.00, 0.80],
                [5.10, -0.80, 0.80],
                [5.55, -0.80, 0.40],
            ]
        )
        projected_gravity_b = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
            ]
        )

    class _Env:
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    env = _Env()
    success_reward = rewards.crossing_success(
        env,
        success_x=5.35,
        min_height=0.55,
        max_abs_y=0.55,
        target_y=-0.80,
        asset_cfg=None,
    )
    crossed = terminations.crossed_target(
        env,
        success_x=5.35,
        min_height=0.55,
        max_abs_y=0.55,
        target_y=-0.80,
        asset_cfg=None,
    )

    assert torch.equal(success_reward, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(crossed, torch.tensor([True, False, False, False]))


def test_lateral_out_and_corridor_are_centered_on_box_line():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards, terminations

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [0.0, -0.80, 0.80],
                [0.0, -0.20, 0.80],
                [0.0, 0.45, 0.80],
                [0.0, -2.10, 0.80],
            ]
        )
        root_lin_vel_w = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

    class _Env:
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    env = _Env()
    corridor = rewards.lateral_corridor_barrier(
        env,
        safe_abs_y=0.45,
        max_abs_y=1.25,
        target_y=-0.80,
        asset_cfg=None,
    )
    lateral_out = terminations.lateral_out_of_bounds(env, max_abs_y=1.25, target_y=-0.80, asset_cfg=None)
    lateral_penalty = rewards.lateral_deviation_l2(env, target_y=-0.80, asset_cfg=None)

    assert torch.allclose(corridor, torch.tensor([0.0, 0.03515625, 1.0, 1.0]))
    assert torch.equal(lateral_out, torch.tensor([False, False, False, True]))
    assert torch.allclose(lateral_penalty, torch.tensor([0.0, 0.36, 1.5625, 1.69]))


def test_lateral_corridor_and_velocity_penalties_activate_before_lateral_out():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [0.0, 0.00, 0.80],
                [0.0, 0.45, 0.80],
                [0.0, 0.85, 0.80],
                [0.0, 1.25, 0.80],
            ]
        )
        root_lin_vel_w = torch.tensor(
            [
                [0.0, 0.00, 0.0],
                [0.0, 0.20, 0.0],
                [0.0, -0.50, 0.0],
                [0.0, 1.00, 0.0],
            ]
        )

    class _Env:
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    env = _Env()
    corridor = rewards.lateral_corridor_barrier(env, safe_abs_y=0.45, max_abs_y=1.25)
    lateral_velocity = rewards.lateral_velocity_l2(env)

    assert torch.allclose(corridor, torch.tensor([0.0, 0.0, 0.25, 1.0]))
    assert torch.allclose(lateral_velocity, torch.tensor([0.0, 0.04, 0.25, 1.0]))


def test_box_support_region_rewards_centered_upright_robot_only():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [-0.85, 0.0, 0.50],
                [-0.85, 0.8, 0.50],
                [-1.50, 0.0, 0.80],
                [-0.85, 0.0, 0.20],
            ]
        )
        projected_gravity_b = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.8, 0.0, -0.4],
            ]
        )

    class _Env:
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    reward = rewards.box_support_region_reward(
        _Env(),
        box_x=-0.85,
        half_length=0.45,
        half_width=0.55,
        min_height=0.35,
    )

    assert torch.allclose(reward, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_box_support_region_can_center_on_nonzero_lateral_line():
    from atec_rl_lab.train.cross_pit_box.mdp import rewards

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [4.36, -0.80, 0.50],
                [4.36, 0.00, 0.50],
                [3.90, -0.80, 0.50],
                [4.36, -0.80, 0.10],
            ]
        )
        projected_gravity_b = torch.tensor(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.8, 0.0, -0.4],
            ]
        )

    class _Env:
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    reward = rewards.box_support_region_reward(
        _Env(),
        box_x=4.36,
        box_y=-0.80,
        half_length=0.45,
        half_width=0.55,
        min_height=0.35,
    )

    assert torch.allclose(reward, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_cross_pit_box_config_reduces_static_midpoint_rewards_and_adds_far_side_guidance():
    pytest.importorskip("pxr", reason="env config import requires Isaac/Omniverse runtime modules")

    from atec_rl_lab.train.cross_pit_box.env_cfg import CrossPitBoxG1EnvCfg

    cfg = CrossPitBoxG1EnvCfg()

    assert cfg.rewards.crossing_progress.weight <= 4.0
    assert cfg.rewards.incremental_crossing_progress.weight >= 10.0
    assert cfg.rewards.far_platform_region.weight >= cfg.rewards.box_support_region.weight
    assert cfg.rewards.lateral_deviation.weight <= -1.2
    assert cfg.rewards.crossing_success.params["success_x"] == 1.85
    assert cfg.curriculum.success_x.params["final_x"] == 2.55
    assert cfg.terminations.crossed.params["success_x_attr"] == "cross_pit_success_x"
    assert cfg.curriculum.success_x.params["start_iteration"] == 1000
    assert cfg.curriculum.success_x.params["end_iteration"] == 6000
    assert cfg.curriculum.success_x.params["steps_per_iteration"] == 32


def test_cross_pit_box_ppo_config_uses_second_version_stability_hyperparameters():
    ppo_cfg_path = Path(__file__).parent / "agents" / "rsl_rl_ppo_cfg.py"
    cfg_text = ppo_cfg_path.read_text()

    assert 'class_name="CrossPitBoxHeightmapActorCritic"' not in cfg_text
    assert "CrossPitBoxHeightmapActorCriticCfg" not in cfg_text
    assert "heightmap_shape" not in cfg_text
    assert "heightmap_latent_dim" not in cfg_text
    assert "RslRlPpoActorCriticCfg(" in cfg_text
    assert "init_noise_std=0.55" in cfg_text
    assert "entropy_coef=0.001" in cfg_text
    assert "learning_rate=2.5e-4" in cfg_text
    assert "desired_kl=0.004" in cfg_text


def test_cross_pit_box_scripts_do_not_register_custom_cnn_policy_with_rsl_rl_runner():
    train_text = (Path(__file__).parents[5] / "scripts" / "rsl_rl" / "train.py").read_text()
    play_text = (Path(__file__).parents[5] / "scripts" / "rsl_rl" / "play.py").read_text()

    assert "heightmap_actor_critic" not in train_text
    assert "CrossPitBoxHeightmapActorCritic" not in train_text
    assert "rsl_on_policy_runner" not in train_text
    assert "heightmap_actor_critic" not in play_text
    assert "CrossPitBoxHeightmapActorCritic" not in play_text
    assert "rsl_on_policy_runner" not in play_text


def test_cross_pit_box_heightmap_helpers_preserve_fixed_observation_shape():
    from atec_rl_lab.train.cross_pit_box.mdp.observations import ray_hits_to_heightmap

    ray_hits_w = torch.tensor(
        [
            [
                [0.0, -0.5, -0.2],
                [0.0, 0.5, 0.1],
                [1.0, -0.5, 0.3],
                [1.0, 0.5, float("inf")],
            ]
        ]
    )
    base_pos_w = torch.zeros(1, 3)
    base_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    heightmap = ray_hits_to_heightmap(ray_hits_w, base_pos_w, base_quat_w, grid_shape=(2, 2), default_height=-1.0)

    assert heightmap.shape == (1, 4)
    assert torch.allclose(heightmap, torch.tensor([[-0.2, 0.1, 0.3, -1.0]]))


def test_cross_pit_box_critic_privileged_observation_helpers_use_env_relative_state():
    from atec_rl_lab.train.cross_pit_box.mdp import observations

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [8.25, 0.20, 0.80],
                [-7.45, -0.35, 0.75],
            ]
        )

    class _Scene(dict):
        env_origins = torch.tensor(
            [
                [10.0, 0.0, 0.0],
                [-10.0, 0.0, 0.0],
            ]
        )

    class _Env:
        cross_pit_success_x = 2.55
        scene = _Scene({"robot": type("_Asset", (), {"data": _AssetData})()})

    env = _Env()

    assert torch.allclose(
        observations.base_pos_env(env),
        torch.tensor([[-1.75, 0.20, 0.80], [2.55, -0.35, 0.75]]),
    )
    assert torch.allclose(observations.box_center_lateral_error(env, box_y=0.0), torch.tensor([[0.20], [-0.35]]))
    assert torch.allclose(observations.abs_lateral_error(env, box_y=0.0), torch.tensor([[0.20], [0.35]]))
    assert torch.allclose(
        observations.target_x_error(env, success_x=1.85, success_x_attr="cross_pit_success_x"),
        torch.tensor([[4.30], [0.00]]),
        atol=1.0e-6,
    )


def test_cross_pit_box_critic_stage_privileged_observations_encode_gap_landmarks():
    from atec_rl_lab.train.cross_pit_box.mdp import observations

    class _AssetData:
        root_pos_w = torch.tensor(
            [
                [3.60, -0.80, 0.80],
                [4.10, -0.80, 0.35],
                [4.70, -0.80, 0.80],
            ]
        )

    class _Env:
        cross_pit_success_x = 5.80
        scene = {"robot": type("_Asset", (), {"data": _AssetData})()}

    stage = observations.crossing_stage_privileged(
        _Env(),
        near_lip_x=3.72,
        box_front_x=4.06,
        box_center_x=4.36,
        box_back_x=4.66,
        far_lip_x=4.68,
        far_platform_x=5.10,
        success_x=3.0,
        success_x_attr="cross_pit_success_x",
        margin=0.12,
    )
    distances = observations.box_gap_privileged(
        _Env(),
        near_lip_x=3.72,
        box_front_x=4.06,
        box_center_x=4.36,
        box_back_x=4.66,
        far_lip_x=4.68,
        far_platform_x=5.10,
        success_x=3.0,
        success_x_attr="cross_pit_success_x",
    )

    assert stage.shape == (3, 7)
    assert distances.shape == (3, 7)
    assert stage[0, 0] < stage[1, 0] < stage[2, 0]
    assert stage[0, 2] < 0.1
    assert stage[1, 1] > 0.5
    assert stage[2, 4] > 0.5
    assert distances[0, 0] < 0.0
    assert distances[2, 4] > 0.0
