from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _fake_official_obs(batch: int = 1) -> dict:
    proprio = torch.zeros(batch, 111, dtype=torch.float32)
    proprio[:, 6] = 0.65
    proprio[:, 11] = -1.0
    extero = torch.linspace(-1.0, 1.0, 5760, dtype=torch.float32).repeat(batch, 1)
    return {
        "proprio": proprio,
        "extero": extero,
        "image": {
            "head_depth": torch.ones(batch, 480, 640, 1, dtype=torch.float32),
            "ee_depth": torch.ones(batch, 480, 640, 1, dtype=torch.float32),
        },
    }


def test_cross_pit_policy_loads_model_and_builds_expected_actor_observation():
    from demo.cross_pit_box_policy import CrossPitBoxPolicy

    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(model_path=model_path, device="cpu")
    obs = _fake_official_obs()

    policy_obs = policy.build_policy_obs(obs)
    assert policy_obs.shape == (1, 492)

    action = policy.predict(obs)
    assert action.shape == (1, 33)
    assert torch.isfinite(action).all()


def test_cross_pit_blind_policy_builds_108d_actor_observation_without_heightmap():
    from demo.cross_pit_box_policy import BLIND_ACTOR_OBS_DIM, CrossPitBoxPolicy

    model_path = Path(__file__).resolve().parent / "cross_pit_box_v1_5_blind_model_19999.pt"
    policy = CrossPitBoxPolicy(model_path=model_path, device="cpu")
    obs = _fake_official_obs()

    policy_obs = policy.build_policy_obs(obs)
    action = policy.predict(obs)

    assert policy.actor_obs_dim == BLIND_ACTOR_OBS_DIM
    assert not policy.uses_heightmap
    assert policy_obs.shape == (1, BLIND_ACTOR_OBS_DIM)
    assert action.shape == (1, 33)
    assert policy.get_debug_snapshot()["heightmap_mean"] is None


def test_cross_pit_policy_default_checkpoint_is_verified_v1_5_blind(monkeypatch):
    from demo.cross_pit_box_policy import BLIND_ACTOR_OBS_DIM, CrossPitBoxPolicy

    monkeypatch.delenv("ATEC_CROSS_PIT_BOX_CHECKPOINT", raising=False)
    monkeypatch.delenv("ATEC_CROSS_PIT_STABILIZE_STEPS", raising=False)
    monkeypatch.delenv("ATEC_CROSS_PIT_ACTION_OFFSET_SCALE", raising=False)

    policy = CrossPitBoxPolicy(device="cpu")

    assert policy.model_path.name == "cross_pit_box_v1_5_blind_model_19999.pt"
    assert policy.actor_obs_dim == BLIND_ACTOR_OBS_DIM
    assert not policy.uses_heightmap
    assert policy.stabilize_steps == 2
    assert policy.action_offset_scale == 1.0


def test_cross_pit_policy_can_require_verified_deploy_defaults(monkeypatch):
    from demo.cross_pit_box_policy import CrossPitBoxPolicy

    monkeypatch.setenv("ATEC_CROSS_PIT_REQUIRE_DEPLOY_DEFAULTS", "1")
    monkeypatch.delenv("ATEC_CROSS_PIT_BOX_CHECKPOINT", raising=False)
    monkeypatch.delenv("ATEC_CROSS_PIT_STABILIZE_STEPS", raising=False)

    CrossPitBoxPolicy(device="cpu")

    monkeypatch.setenv("ATEC_CROSS_PIT_BOX_CHECKPOINT", "cross_pit_box_model_19999.pt")
    with pytest.raises(ValueError, match="deploy defaults"):
        CrossPitBoxPolicy(device="cpu")


def test_cross_pit_policy_can_require_verified_deploy_defaults_without_env(monkeypatch):
    from demo.cross_pit_box_policy import CrossPitBoxPolicy

    monkeypatch.delenv("ATEC_CROSS_PIT_REQUIRE_DEPLOY_DEFAULTS", raising=False)
    monkeypatch.setenv("ATEC_CROSS_PIT_BOX_CHECKPOINT", "cross_pit_box_model_19999.pt")

    with pytest.raises(ValueError, match="deploy defaults"):
        CrossPitBoxPolicy(device="cpu", require_deploy_defaults=True)


def test_cross_pit_policy_uses_zero_last_action_on_first_handoff_observation():
    from demo.cross_pit_box_policy import CrossPitBoxPolicy

    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        action_clip=100.0,
        stabilize_steps=0,
    )
    obs = _fake_official_obs()
    obs["proprio"][:, 78:111] = 0.25

    policy_obs = policy.build_policy_obs(obs)

    assert torch.allclose(policy_obs[:, 75:108], torch.zeros(1, 33))


def test_cross_pit_policy_can_seed_first_last_action_from_taskd_env_action(monkeypatch):
    from demo.cross_pit_box_policy import CROSS_PIT_RESET_ACTION, CrossPitBoxPolicy

    monkeypatch.setenv("ATEC_CROSS_PIT_INITIAL_LAST_ACTION", "env")
    model_path = Path(__file__).resolve().parent / "cross_pit_box_v1_5_blind_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        action_clip=100.0,
        stabilize_steps=0,
    )
    obs = _fake_official_obs()
    env_action = torch.linspace(-0.5, 0.5, 33).view(1, 33)
    obs["proprio"][:, 78:111] = env_action

    policy_obs = policy.build_policy_obs(obs)

    expected_training_action = env_action - CROSS_PIT_RESET_ACTION.view(1, -1)
    assert torch.allclose(policy_obs[:, 75:108], expected_training_action, atol=1.0e-6)


def test_cross_pit_policy_warms_velocity_observations_from_training_reset(monkeypatch):
    from demo.cross_pit_box_policy import CrossPitBoxPolicy

    monkeypatch.setenv("ATEC_CROSS_PIT_VELOCITY_OBS_WARMUP_STEPS", "4")
    monkeypatch.setenv("ATEC_CROSS_PIT_BASE_ANG_VEL_OBS_CLIP", "2.0")
    monkeypatch.setenv("ATEC_CROSS_PIT_JOINT_VEL_OBS_CLIP", "2.0")
    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        action_clip=100.0,
        stabilize_steps=0,
    )
    obs = _fake_official_obs()
    obs["proprio"][:, 3:6] = torch.tensor([[5.0, -5.0, 1.0]])
    obs["proprio"][:, 45:78] = 30.0

    first_obs = policy.build_policy_obs(obs)
    policy._policy_step_count = 2
    half_warm_obs = policy.build_policy_obs(obs)

    assert torch.allclose(first_obs[:, 0:3], torch.zeros(1, 3))
    assert torch.allclose(first_obs[:, 42:75], torch.zeros(1, 33))
    assert torch.allclose(half_warm_obs[:, 0:3], torch.tensor([[1.0, -1.0, 0.5]]))
    assert torch.allclose(half_warm_obs[:, 42:75], torch.ones(1, 33))


def test_cross_pit_policy_does_not_clip_velocity_observations_by_default(monkeypatch):
    from demo.cross_pit_box_policy import CrossPitBoxPolicy

    monkeypatch.delenv("ATEC_CROSS_PIT_VELOCITY_OBS_WARMUP_STEPS", raising=False)
    monkeypatch.delenv("ATEC_CROSS_PIT_BASE_ANG_VEL_OBS_CLIP", raising=False)
    monkeypatch.delenv("ATEC_CROSS_PIT_JOINT_VEL_OBS_CLIP", raising=False)
    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        action_clip=100.0,
        stabilize_steps=0,
    )
    obs = _fake_official_obs()
    obs["proprio"][:, 3:6] = torch.tensor([[5.0, -5.0, 1.0]])
    obs["proprio"][:, 45:78] = 60.0

    policy_obs = policy.build_policy_obs(obs)

    assert torch.allclose(policy_obs[:, 0:3], torch.tensor([[5.0, -5.0, 1.0]]))
    assert torch.allclose(policy_obs[:, 42:75], torch.full((1, 33), 6.0))


def test_cross_pit_policy_rebases_taskd_joint_position_rel_to_handoff_pose():
    from demo.cross_pit_box_policy import CROSS_PIT_OFFICIAL_JOINT_POS_OFFSET, CrossPitBoxPolicy

    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        action_clip=100.0,
        stabilize_steps=0,
    )
    obs = _fake_official_obs()
    obs["proprio"][:, 12:45] = CROSS_PIT_OFFICIAL_JOINT_POS_OFFSET.view(1, -1)

    policy_obs = policy.build_policy_obs(obs)

    assert torch.allclose(policy_obs[:, 9:42], torch.zeros(1, 33), atol=1.0e-6)


def test_cross_pit_policy_treats_taskd_handoff_joint_rel_offset_as_training_relative_zero():
    from demo.cross_pit_box_policy import CROSS_PIT_OFFICIAL_JOINT_POS_OFFSET, CrossPitBoxPolicy

    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        action_clip=100.0,
        stabilize_steps=0,
    )
    obs = _fake_official_obs()
    obs["proprio"][:, 12:45] = CROSS_PIT_OFFICIAL_JOINT_POS_OFFSET.view(1, -1)

    policy_obs = policy.build_policy_obs(obs)

    assert torch.allclose(policy_obs[:, 9:42], torch.zeros(1, 33), atol=1.0e-6)


def test_cross_pit_policy_reorders_taskd_joint_observation_to_training_order():
    from demo.cross_pit_box_policy import (
        CROSS_PIT_OFFICIAL_JOINT_POS_OFFSET,
        CROSS_PIT_OBSERVATION_JOINT_NAMES,
        CrossPitBoxPolicy,
        TASKD_JOINT_NAMES,
    )

    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        action_clip=100.0,
        stabilize_steps=0,
    )
    taskd_values_by_name = {name: float(index + 1) for index, name in enumerate(TASKD_JOINT_NAMES)}
    obs = _fake_official_obs()
    obs["proprio"][:, 12:45] = CROSS_PIT_OFFICIAL_JOINT_POS_OFFSET.view(1, -1) + torch.tensor(
        [taskd_values_by_name[name] for name in TASKD_JOINT_NAMES]
    )

    policy_obs = policy.build_policy_obs(obs)
    expected = torch.tensor([taskd_values_by_name[name] for name in CROSS_PIT_OBSERVATION_JOINT_NAMES])

    assert torch.allclose(policy_obs[:, 9:42], expected.view(1, -1), atol=1.0e-6)


def test_cross_pit_snapshot_joint_order_is_not_taskd_action_order():
    from demo.cross_pit_box_policy import CROSS_PIT_TRAINING_JOINT_NAMES

    snapshot_order = (
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
    )

    assert tuple(CROSS_PIT_TRAINING_JOINT_NAMES[:4]) != snapshot_order



def test_cross_pit_policy_reorders_taskd_joint_state_to_native_training_observation_order():
    from demo.cross_pit_box_policy import (
        CROSS_PIT_OFFICIAL_JOINT_POS_OFFSET,
        CROSS_PIT_OBSERVATION_JOINT_INDEX_IN_TASKD_ORDER,
        CrossPitBoxPolicy,
    )

    model_path = Path(__file__).resolve().parent / "cross_pit_box_v1_5_blind_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        action_clip=100.0,
        stabilize_steps=0,
    )
    obs = _fake_official_obs()
    taskd_joint_delta = torch.arange(1, 34, dtype=torch.float32).view(1, 33)
    taskd_joint_vel = torch.arange(101, 134, dtype=torch.float32).view(1, 33)
    obs["proprio"][:, 12:45] = CROSS_PIT_OFFICIAL_JOINT_POS_OFFSET.view(1, -1) + taskd_joint_delta
    obs["proprio"][:, 45:78] = taskd_joint_vel

    policy_obs = policy.build_policy_obs(obs)
    expected_index = torch.tensor(CROSS_PIT_OBSERVATION_JOINT_INDEX_IN_TASKD_ORDER, dtype=torch.long)

    assert torch.allclose(policy_obs[:, 9:42], taskd_joint_delta.index_select(1, expected_index))
    assert torch.allclose(policy_obs[:, 42:75], taskd_joint_vel.index_select(1, expected_index) * 0.1)

def test_cross_pit_policy_reorders_actor_action_back_to_taskd_order():
    from demo.cross_pit_box_policy import (
        ACTION_DIM,
        CROSS_PIT_RESET_ACTION,
        CROSS_PIT_TRAINING_JOINT_NAMES,
        CrossPitBoxPolicy,
        TASKD_JOINT_NAMES,
    )

    class _IndexActor(torch.nn.Module):
        def forward(self, obs):
            return torch.arange(1, ACTION_DIM + 1, dtype=obs.dtype, device=obs.device).view(1, -1)

    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        action_clip=100.0,
        stabilize_steps=0,
    )
    policy.actor = _IndexActor()

    action = policy.predict(_fake_official_obs())
    actor_values_by_name = {
        name: float(index + 1)
        for index, name in enumerate(CROSS_PIT_TRAINING_JOINT_NAMES)
    }
    expected = torch.tensor([actor_values_by_name[name] for name in TASKD_JOINT_NAMES]).view(1, -1)
    expected = expected + CROSS_PIT_RESET_ACTION.view(1, -1)
    assert torch.allclose(action, expected)


def test_cross_pit_policy_adds_handoff_action_offset_by_default(monkeypatch):
    from demo.cross_pit_box_policy import ACTION_DIM, CROSS_PIT_RESET_ACTION, CrossPitBoxPolicy

    class _ZeroActor(torch.nn.Module):
        def forward(self, obs):
            return torch.zeros(obs.shape[0], ACTION_DIM, dtype=obs.dtype, device=obs.device)

    monkeypatch.delenv("ATEC_CROSS_PIT_ACTION_OFFSET_SCALE", raising=False)
    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        action_clip=100.0,
        stabilize_steps=0,
    )
    policy.actor = _ZeroActor()

    action = policy.predict(_fake_official_obs())
    policy_obs_after_first_action = policy.build_policy_obs(_fake_official_obs())

    assert torch.allclose(action, CROSS_PIT_RESET_ACTION.view(1, ACTION_DIM), atol=1.0e-6)
    assert torch.allclose(policy_obs_after_first_action[:, 75:108], torch.zeros(1, ACTION_DIM), atol=1.0e-6)


def test_cross_pit_policy_can_disable_handoff_action_offset_for_diagnostics(monkeypatch):
    from demo.cross_pit_box_policy import ACTION_DIM, CrossPitBoxPolicy

    class _ZeroActor(torch.nn.Module):
        def forward(self, obs):
            return torch.zeros(obs.shape[0], ACTION_DIM, dtype=obs.dtype, device=obs.device)

    monkeypatch.setenv("ATEC_CROSS_PIT_ACTION_OFFSET_SCALE", "0.0")
    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        action_clip=100.0,
        stabilize_steps=0,
    )
    policy.actor = _ZeroActor()

    action = policy.predict(_fake_official_obs())
    policy_obs_after_first_action = policy.build_policy_obs(_fake_official_obs())

    assert torch.allclose(action, torch.zeros(1, ACTION_DIM), atol=1.0e-6)
    assert torch.allclose(policy_obs_after_first_action[:, 75:108], torch.zeros(1, ACTION_DIM), atol=1.0e-6)


def test_cross_pit_policy_feeds_raw_actor_action_back_as_last_action():
    from demo.cross_pit_box_policy import ACTION_DIM, CROSS_PIT_RESET_ACTION, CrossPitBoxPolicy

    class _ConstantActor(torch.nn.Module):
        def forward(self, obs):
            return torch.full((obs.shape[0], ACTION_DIM), 0.75, dtype=obs.dtype, device=obs.device)

    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        action_clip=100.0,
        stabilize_steps=0,
    )
    policy.actor = _ConstantActor()

    action = policy.predict(_fake_official_obs())
    policy_obs_after_first_action = policy.build_policy_obs(_fake_official_obs())

    expected_taskd_action = torch.full((1, ACTION_DIM), 0.75) + CROSS_PIT_RESET_ACTION.view(1, -1)
    assert torch.allclose(action, expected_taskd_action)
    assert torch.allclose(policy_obs_after_first_action[:, 75:108], torch.full((1, ACTION_DIM), 0.75))


def test_cross_pit_policy_adds_reset_pose_offset_to_deploy_action(monkeypatch):
    from demo.cross_pit_box_policy import ACTION_DIM, CROSS_PIT_RESET_ACTION, CrossPitBoxPolicy

    class _ConstantActor(torch.nn.Module):
        def forward(self, obs):
            return torch.full((obs.shape[0], ACTION_DIM), 0.5, dtype=obs.dtype, device=obs.device)

    monkeypatch.delenv("ATEC_CROSS_PIT_ACTION_OFFSET_SCALE", raising=False)
    model_path = Path(__file__).resolve().parent / "cross_pit_box_v1_5_blind_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        require_deploy_defaults=True,
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        stabilize_steps=2,
    )
    policy.actor = _ConstantActor()
    obs = _fake_official_obs()
    obs["proprio"][:, 12:45] = 0.10
    obs["proprio"][:, 45:78] = 0.0

    policy.predict(obs)
    policy.predict(obs)
    actor_action = policy.predict(obs)
    policy_obs_after_actor = policy.build_policy_obs(obs)

    expected_taskd_action = torch.full((1, ACTION_DIM), 0.5) + CROSS_PIT_RESET_ACTION.view(1, -1)
    assert torch.allclose(actor_action, expected_taskd_action)
    assert torch.allclose(policy_obs_after_actor[:, 75:108], torch.full((1, ACTION_DIM), 0.5))


def test_cross_pit_policy_adds_heading_correction_to_velocity_command(monkeypatch):
    from demo.cross_pit_box_policy import CrossPitBoxPolicy

    monkeypatch.setenv("ATEC_CROSS_PIT_HEADING_KP", "0.5")
    model_path = Path(__file__).resolve().parent / "cross_pit_box_v1_5_blind_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        action_clip=100.0,
        stabilize_steps=0,
    )
    obs = _fake_official_obs()
    obs["_debug_root_quat_w"] = torch.tensor([[0.9950042, 0.0, 0.0, 0.0998334]], dtype=torch.float32)

    policy_obs = policy.build_policy_obs(obs)

    assert policy_obs[0, 6].item() == pytest.approx(0.6644027)
    assert policy_obs[0, 7].item() == pytest.approx(0.0)
    assert policy_obs[0, 8].item() < 0.0


def test_cross_pit_lidar_heightmap_preserves_lateral_locality():
    from demo.cross_pit_box_policy import compress_official_height_scan

    scan = torch.full((1, 5760), float("-inf"), dtype=torch.float32)
    scan = scan.view(1, 16, 360)
    # A finite cluster at one forward-right azimuth should not be broadcast across
    # the full lateral width of the 24x16 CrossPitBox heightmap.
    scan[:, :, 190] = 0.4

    heightmap = compress_official_height_scan(scan.view(1, -1)).view(1, 24, 16)
    valid_per_forward_bin = (heightmap[0] > -0.9).sum(dim=1)

    assert int(valid_per_forward_bin.max().item()) < 16


def test_cross_pit_lidar_heightmap_uses_projected_gravity_for_base_frame_heights():
    from demo.cross_pit_box_policy import (
        HEIGHTMAP_DIM,
        HEIGHTMAP_X_RANGE,
        HEIGHTMAP_Y_RANGE,
        _taskd_lidar_directions,
        compress_official_height_scan,
    )

    directions = _taskd_lidar_directions(device=torch.device("cpu"), dtype=torch.float32)
    valid = (
        (directions[:, 2] < -0.2)
        & (directions[:, 0] > 0.4)
        & (directions[:, 0] * 2.0 >= HEIGHTMAP_X_RANGE[0])
        & (directions[:, 0] * 2.0 <= HEIGHTMAP_X_RANGE[1])
        & (directions[:, 1] * 2.0 >= HEIGHTMAP_Y_RANGE[0])
        & (directions[:, 1] * 2.0 <= HEIGHTMAP_Y_RANGE[1])
    )
    ray_index = int(torch.nonzero(valid, as_tuple=False)[0].item())
    distance = 2.0
    expected_base_z = float(distance * directions[ray_index, 2].item())
    scan = torch.full((1, 5760), float("-inf"), dtype=torch.float32)
    # IsaacLab height_scan is sensor_z - hit_z - 0.5. With flat gravity this is
    # equivalent to the world vertical drop minus the same offset.
    scan[0, ray_index] = -expected_base_z - 0.5

    heightmap = compress_official_height_scan(scan, projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]))

    assert heightmap.shape == (1, HEIGHTMAP_DIM)
    assert heightmap.max().item() == pytest.approx(expected_base_z, abs=1.0e-5)


def test_cross_pit_policy_can_use_debug_head_depth_heightmap(monkeypatch):
    from demo.cross_pit_box_policy import CrossPitBoxPolicy, HEIGHTMAP_DIM

    monkeypatch.setenv("ATEC_CROSS_PIT_HEIGHTMAP_SOURCE", "head_depth")
    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        stabilize_steps=0,
    )
    obs = _fake_official_obs()
    obs["extero"][:] = float("-inf")
    obs["image"]["head_depth"] = torch.full((1, 8, 8, 1), 2.0, dtype=torch.float32)
    obs["_debug_head_camera_intrinsics"] = torch.tensor(
        [[[4.0, 0.0, 3.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    obs["_debug_head_camera_pos_w"] = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    obs["_debug_head_camera_quat_w"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    obs["_debug_root_pos_w"] = torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32)
    obs["_debug_root_quat_w"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    policy_obs = policy.build_policy_obs(obs)
    heightmap = policy_obs[:, -HEIGHTMAP_DIM:]

    assert heightmap.shape == (1, HEIGHTMAP_DIM)
    assert torch.isfinite(heightmap).all()
    assert torch.any(heightmap > -0.999)
    assert policy.get_debug_snapshot()["heightmap_source"] == "head_depth"


def test_cross_pit_policy_reports_actual_debug_ray_hits_heightmap_source(monkeypatch):
    from demo.cross_pit_box_policy import HEIGHTMAP_DIM, CrossPitBoxPolicy

    monkeypatch.delenv("ATEC_CROSS_PIT_HEIGHTMAP_SOURCE", raising=False)
    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        action_clip=100.0,
        stabilize_steps=0,
    )
    obs = _fake_official_obs()
    obs["_debug_root_pos_w"] = torch.tensor([[0.0, 0.0, 0.7]], dtype=torch.float32)
    obs["_debug_root_quat_w"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    ray_hits = torch.zeros(1, HEIGHTMAP_DIM, 3, dtype=torch.float32)
    ray_hits[..., 0] = 0.4
    ray_hits[..., 2] = 0.2
    obs["_debug_depth_ray_hits_w"] = ray_hits

    policy.build_policy_obs(obs)

    assert policy.get_debug_snapshot()["heightmap_source_actual"] == "debug_ray_hits"


def test_cross_pit_terrain_map_heightmap_models_ground_box_and_pit():
    from demo.cross_pit_box_policy import (
        CROSS_PIT_HANDOFF_LOCAL_X,
        CROSS_PIT_HANDOFF_LOCAL_Y,
        CROSS_PIT_HANDOFF_ROOT_Z,
        CROSS_PIT_HANDOFF_YAW,
        HEIGHTMAP_DIM,
        terrain_map_heightmap,
    )

    heightmap = terrain_map_heightmap(
        torch.tensor([CROSS_PIT_HANDOFF_LOCAL_X]),
        torch.tensor([CROSS_PIT_HANDOFF_LOCAL_Y]),
        torch.tensor([CROSS_PIT_HANDOFF_ROOT_Z]),
        torch.tensor([CROSS_PIT_HANDOFF_YAW]),
    )

    assert heightmap.shape == (1, HEIGHTMAP_DIM)
    assert torch.isclose(heightmap.max(), torch.tensor(-CROSS_PIT_HANDOFF_ROOT_Z), atol=1.0e-5)
    assert torch.any(heightmap <= -0.999)
    assert torch.any((-0.95 < heightmap) & (heightmap < -0.90))


def test_cross_pit_terrain_map_heightmap_models_side_platform():
    from demo.cross_pit_box_policy import HEIGHTMAP_DIM, terrain_map_heightmap

    heightmap = terrain_map_heightmap(
        torch.tensor([2.40]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
        d435_link_offset_b=torch.zeros((1, 3)),
    )

    assert heightmap.shape == (1, HEIGHTMAP_DIM)
    assert heightmap.max().item() == pytest.approx(0.927, abs=0.05)
    assert torch.any(heightmap > 0.85)


def test_cross_pit_terrain_map_heightmap_uses_projected_gravity_like_training_raycast():
    from demo.cross_pit_box_policy import (
        CROSS_PIT_ENV_ORIGIN_X,
        HEIGHTMAP_DIM,
        terrain_map_heightmap,
    )

    # First native CrossPitBox play frame from model_19999: the training raycast
    # reports mean ~= -0.899, max ~= -0.686, and 227 non-default cells.
    heightmap = terrain_map_heightmap(
        torch.tensor([-1.7389843463897705 - CROSS_PIT_ENV_ORIGIN_X]),
        torch.tensor([-0.834780752658844]),
        torch.tensor([0.733408510684967]),
        torch.tensor([0.18481599561158227]),
        projected_gravity=torch.tensor([[-0.11321479827165604, -0.05308475345373154, -0.9921514391899109]]),
    )

    assert heightmap.shape == (1, HEIGHTMAP_DIM)
    assert -0.92 < float(heightmap.mean()) < -0.88
    assert -0.72 < float(heightmap.max()) < -0.68
    assert 200 <= int((heightmap > -0.999).sum()) <= 240


def test_cross_pit_terrain_map_heightmap_uses_d435_link_offset():
    from demo.cross_pit_box_policy import (
        CROSS_PIT_HANDOFF_LOCAL_X,
        CROSS_PIT_HANDOFF_LOCAL_Y,
        CROSS_PIT_HANDOFF_ROOT_Z,
        CROSS_PIT_HANDOFF_YAW,
        terrain_map_heightmap,
    )

    base_kwargs = dict(
        local_x=torch.tensor([CROSS_PIT_HANDOFF_LOCAL_X]),
        local_y=torch.tensor([CROSS_PIT_HANDOFF_LOCAL_Y]),
        root_z=torch.tensor([CROSS_PIT_HANDOFF_ROOT_Z]),
        yaw=torch.tensor([CROSS_PIT_HANDOFF_YAW]),
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
    )
    root_origin_scan = terrain_map_heightmap(**base_kwargs, d435_link_offset_b=torch.tensor([[0.0, 0.0, 0.0]]))
    d435_scan = terrain_map_heightmap(**base_kwargs, d435_link_offset_b=torch.tensor([[0.19, 0.0, 0.43]]))

    assert not torch.allclose(root_origin_scan, d435_scan)
    assert torch.mean(torch.abs(d435_scan - root_origin_scan)) > 0.03
    assert d435_scan.max().item() == pytest.approx(root_origin_scan.max().item(), abs=1.0e-6)


def test_cross_pit_d435_offset_tracks_waist_pose_from_native_logs():
    from demo.cross_pit_box_policy import estimate_d435_link_offset_b_from_waist

    waist = torch.tensor(
        [
            [-0.3175305426120758, -0.135825514793396, 0.30181464552879333],
            [-0.023026779294013977, -0.27141499519348145, 0.25113746523857117],
        ],
        dtype=torch.float32,
    )

    offset = estimate_d435_link_offset_b_from_waist(waist)

    expected = torch.tensor(
        [
            [0.19194631278514862, 0.01125900074839592, 0.4313145577907562],
            [0.16150599718093872, 0.12099338322877884, 0.4266469180583954],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(offset, expected, atol=0.003)


def test_cross_pit_policy_terrain_map_uses_dynamic_d435_offset_from_waist(monkeypatch):
    from demo.cross_pit_box_policy import CROSS_PIT_RESET_ACTION, CrossPitBoxPolicy, HEIGHTMAP_DIM

    monkeypatch.setenv("ATEC_CROSS_PIT_HEIGHTMAP_SOURCE", "terrain_map")
    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        stabilize_steps=0,
    )
    obs = _fake_official_obs()
    obs["_debug_root_pos_w"] = torch.tensor([[-1.7432950735092163, -0.9189060926437378, 0.733408510684967]])
    obs["_debug_root_quat_w"] = torch.tensor([[0.9904746413230896, 0.033361710608005524, -0.053021665662527084, 0.12262003868818283]])
    obs["proprio"][:, 12:45] = 0.5 * CROSS_PIT_RESET_ACTION.view(1, -1)
    baseline = policy.build_policy_obs(obs)[:, -HEIGHTMAP_DIM:]

    obs["proprio"][:, 12 + 12] = 0.5 * CROSS_PIT_RESET_ACTION[12] + 0.35
    obs["proprio"][:, 12 + 13] = 0.5 * CROSS_PIT_RESET_ACTION[13] - 0.25
    shifted = policy.build_policy_obs(obs)[:, -HEIGHTMAP_DIM:]
    delta = torch.abs(shifted - baseline)

    assert not torch.allclose(baseline, shifted)
    assert torch.max(delta) > 0.20
    assert int((delta > 0.01).sum()) >= 20


def test_cross_pit_policy_d435_offset_uses_current_waist_joint_pose():
    from demo.cross_pit_box_policy import (
        CROSS_PIT_OBSERVATION_JOINT_NAMES,
        CROSS_PIT_OFFICIAL_JOINT_POS_OFFSET,
        CrossPitBoxPolicy,
        estimate_d435_link_offset_b_from_waist,
    )

    policy = object.__new__(CrossPitBoxPolicy)
    policy._current_joint_pos_rel = CROSS_PIT_OFFICIAL_JOINT_POS_OFFSET.view(1, -1).clone()

    joint_pos_observation_order = policy._taskd_to_observation(policy._current_joint_pos_rel)
    waist = torch.stack(
        (
            joint_pos_observation_order[:, CROSS_PIT_OBSERVATION_JOINT_NAMES.index("waist_yaw_joint")],
            joint_pos_observation_order[:, CROSS_PIT_OBSERVATION_JOINT_NAMES.index("waist_roll_joint")],
            joint_pos_observation_order[:, CROSS_PIT_OBSERVATION_JOINT_NAMES.index("waist_pitch_joint")],
        ),
        dim=-1,
    )

    expected = estimate_d435_link_offset_b_from_waist(waist)

    assert torch.allclose(policy._estimate_current_d435_link_offset_b(), expected, atol=1.0e-6)


def test_cross_pit_policy_prefers_terrain_map_heightmap_over_sparse_lidar(monkeypatch):
    from demo.cross_pit_box_policy import CrossPitBoxPolicy, HEIGHTMAP_DIM

    monkeypatch.setenv("ATEC_CROSS_PIT_HEIGHTMAP_SOURCE", "terrain_map")
    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(model_path=model_path, device="cpu", handoff_blend_steps=0)
    obs = _fake_official_obs()
    obs["extero"][:] = float("-inf")

    policy_obs = policy.build_policy_obs(obs)
    heightmap = policy_obs[:, -HEIGHTMAP_DIM:]

    assert torch.isfinite(heightmap).all()
    assert heightmap.max() > -0.9
    assert heightmap.min() <= -0.999


def test_cross_pit_terrain_map_heightmap_tracks_integrated_root_height(monkeypatch):
    from demo.cross_pit_box_policy import CrossPitBoxPolicy, HEIGHTMAP_DIM

    monkeypatch.setenv("ATEC_CROSS_PIT_HEIGHTMAP_SOURCE", "terrain_map")
    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        stabilize_steps=0,
    )
    obs = _fake_official_obs()
    obs["proprio"][:, 0:3] = torch.tensor([[0.0, 0.0, -1.0]])
    obs["proprio"][:, 9:12] = torch.tensor([[0.0, 0.0, -1.0]])

    first_heightmap = policy.build_policy_obs(obs)[:, -HEIGHTMAP_DIM:]
    second_heightmap = policy.build_policy_obs(obs)[:, -HEIGHTMAP_DIM:]
    unclipped = first_heightmap > -0.98

    assert torch.mean(second_heightmap[unclipped] - first_heightmap[unclipped]) > 0.015



def test_cross_pit_policy_uses_debug_root_pose_for_terrain_map_heightmap(monkeypatch):
    from demo.cross_pit_box_policy import CrossPitBoxPolicy, HEIGHTMAP_DIM

    monkeypatch.setenv("ATEC_CROSS_PIT_HEIGHTMAP_SOURCE", "terrain_map")
    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        stabilize_steps=0,
    )
    obs = _fake_official_obs()
    obs["proprio"][:, 0:3] = torch.tensor([[10.0, 0.0, 0.0]])
    obs["_debug_root_pos_w"] = torch.tensor([[-1.7432950735092163, -0.9189060926437378, 0.733408510684967]])
    obs["_debug_root_quat_w"] = torch.tensor([[0.9904746413230896, 0.033361710608005524, -0.053021665662527084, 0.12262003868818283]])

    first_heightmap = policy.build_policy_obs(obs)[:, -HEIGHTMAP_DIM:]
    second_heightmap = policy.build_policy_obs(obs)[:, -HEIGHTMAP_DIM:]

    assert torch.allclose(first_heightmap, second_heightmap)
    assert policy._local_xy_yaw is not None
    assert policy._local_xy_yaw[0, 0].item() == pytest.approx(2.4567047357559204, abs=1e-5)

def test_cross_pit_policy_blends_initial_actions_from_environment_action():
    from demo.cross_pit_box_policy import ACTOR_OBS_DIM, ACTION_DIM, CROSS_PIT_RESET_ACTION, CrossPitBoxPolicy

    class _ConstantActor(torch.nn.Module):
        def forward(self, obs):
            assert obs.shape[-1] == ACTOR_OBS_DIM
            return torch.ones(obs.shape[0], ACTION_DIM, dtype=obs.dtype, device=obs.device)

    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        prealign_steps=0,
        handoff_blend_steps=4,
        handoff_max_delta=10.0,
        stabilize_steps=0,
    )
    policy.actor = _ConstantActor()
    obs = _fake_official_obs()
    obs["proprio"][:, 78:111] = 0.2

    first_action = policy.predict(obs)
    second_action = policy.predict(obs)

    actor_taskd_action = torch.ones(1, ACTION_DIM) + CROSS_PIT_RESET_ACTION.view(1, -1)
    start_action = torch.full((1, ACTION_DIM), 0.2)
    assert torch.allclose(first_action, start_action + 0.25 * (actor_taskd_action - start_action))
    assert torch.allclose(second_action, start_action + 0.50 * (actor_taskd_action - start_action))


def test_cross_pit_policy_rate_limits_actor_takeover_from_environment_action():
    from demo.cross_pit_box_policy import ACTOR_OBS_DIM, ACTION_DIM, CrossPitBoxPolicy

    class _ConstantActor(torch.nn.Module):
        def forward(self, obs):
            assert obs.shape[-1] == ACTOR_OBS_DIM
            return torch.ones(obs.shape[0], ACTION_DIM, dtype=obs.dtype, device=obs.device) * 5.0

    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        prealign_steps=0,
        handoff_blend_steps=0,
        handoff_max_delta=0.2,
        stabilize_steps=0,
    )
    policy.actor = _ConstantActor()
    obs = _fake_official_obs()
    obs["proprio"][:, 78:111] = 0.0

    first_action = policy.predict(obs)
    second_action = policy.predict(obs)

    assert torch.allclose(first_action, torch.full((1, 33), 0.2))
    assert torch.allclose(second_action, torch.full((1, 33), 0.4))


def test_cross_pit_policy_prealigns_to_zero_raw_action_before_actor_takeover():
    from demo.cross_pit_box_policy import ACTION_DIM, CROSS_PIT_RESET_ACTION, CrossPitBoxPolicy

    class _ConstantActor(torch.nn.Module):
        def forward(self, obs):
            return torch.ones(obs.shape[0], ACTION_DIM, dtype=obs.dtype, device=obs.device) * 5.0

    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        prealign_steps=4,
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
        stabilize_steps=0,
    )
    policy.actor = _ConstantActor()
    obs = _fake_official_obs()
    obs["proprio"][:, 78:111] = 0.0

    first_action = policy.predict(obs)
    fourth_action = None
    for _ in range(3):
        fourth_action = policy.predict(obs)
    fifth_action = policy.predict(obs)

    reset_action = CROSS_PIT_RESET_ACTION.view(1, -1)
    assert torch.allclose(first_action, 0.25 * reset_action)
    assert torch.allclose(fourth_action, reset_action)
    assert torch.allclose(fifth_action, torch.ones(1, ACTION_DIM) * 5.0 + reset_action)


def test_cross_pit_policy_stabilizes_handoff_from_current_joint_pose(monkeypatch):
    from demo.cross_pit_box_policy import ACTION_DIM, CROSS_PIT_RESET_ACTION, CrossPitBoxPolicy

    class _ConstantActor(torch.nn.Module):
        def forward(self, obs):
            return torch.ones(obs.shape[0], ACTION_DIM, dtype=obs.dtype, device=obs.device) * 5.0

    monkeypatch.setenv("ATEC_CROSS_PIT_STABILIZE_STEPS", "2")
    monkeypatch.setenv("ATEC_CROSS_PIT_STABILIZE_VEL_GAIN", "0.04")
    model_path = Path(__file__).resolve().parent / "cross_pit_box_model_19999.pt"
    policy = CrossPitBoxPolicy(
        model_path=model_path,
        device="cpu",
        prealign_steps=0,
        handoff_blend_steps=0,
        handoff_max_delta=0.0,
    )
    policy.actor = _ConstantActor()
    obs = _fake_official_obs()
    obs["proprio"][:, 12:45] = 0.10
    obs["proprio"][:, 45:78] = 2.0

    first_action = policy.predict(obs)
    second_action = policy.predict(obs)
    actor_action = policy.predict(obs)

    expected_stabilize_action = torch.full((1, ACTION_DIM), 2.0 * (0.10 - 0.04 * 2.0))
    assert torch.allclose(first_action, expected_stabilize_action)
    assert torch.allclose(second_action, expected_stabilize_action)
    assert torch.allclose(actor_action, torch.ones(1, ACTION_DIM) * 5.0 + CROSS_PIT_RESET_ACTION.view(1, -1))


def test_boxpush_stage_defaults_to_v2_planner_checkpoint():
    planner_text = (Path(__file__).resolve().parent / "boxpush_planner_solution.py").read_text()

    assert 'default_name="boxpush_planner_v2.pt"' in planner_text


def test_boxpush_loader_falls_back_from_missing_legacy_checkpoint_name(monkeypatch):
    import demo.boxpush_planner_solution as box_module

    monkeypatch.setenv("ATEC_BOXPUSH_PLANNER_CHECKPOINT", "boxpush_planner_10hz.pt")
    original_exists = box_module.Path.exists

    def fake_exists(self):
        if self.name == "boxpush_planner_10hz.pt":
            return False
        return original_exists(self)

    monkeypatch.setattr(box_module.Path, "exists", fake_exists, raising=False)

    solution = box_module.BoxPushPlannerSolution(device="cpu")

    assert solution.planner_checkpoint.name == "boxpush_planner_v2.pt"


def test_alg_solution_passes_explicit_boxpush_assets_and_lazily_builds_cross_policy(monkeypatch):
    solution_module = importlib.import_module("demo.solution")
    calls: dict[str, dict] = {}

    class _FakeBoxPush:
        def __init__(self, **kwargs):
            calls["boxpush"] = kwargs

        def get_action_spec(self):
            return {}

        def reset(self, **kwargs):
            return None

        def predicts(self, obs, current_score):
            return {"action": [[0.0] * 33], "giveup": False}

    class _FakeCross:
        def __init__(self, **kwargs):
            calls["cross"] = kwargs

        def reset(self, **kwargs):
            return None

        def predicts(self, obs, current_score):
            return {"action": [[0.0] * 33], "giveup": False}

    monkeypatch.setattr(solution_module, "BoxPushPlannerSolution", _FakeBoxPush)
    monkeypatch.setattr(solution_module, "CrossPitBoxPolicy", _FakeCross)

    agent = solution_module.AlgSolution(handoff_time_s=0.0)

    assert calls["boxpush"]["planner_checkpoint"] == "boxpush_planner_v2.pt"
    assert calls["boxpush"]["walk_policy"] == "gr00t_walk.pt"
    assert calls["boxpush"]["balance_policy"] == "gr00t_balance.pt"
    assert "cross" not in calls

    agent.predicts(_fake_official_obs(), current_score=0.0)

    assert "model_path" not in calls["cross"]
    assert calls["cross"]["target_x"] == 5.8
    assert calls["cross"]["forward_command"] == pytest.approx(0.6644027)
    assert "stabilize_steps" not in calls["cross"]
    assert calls["cross"]["stabilize_vel_gain"] == pytest.approx(0.025)
    assert calls["cross"]["handoff_blend_steps"] == 0
    assert calls["cross"]["handoff_max_delta"] == 0.0
    assert calls["cross"]["require_deploy_defaults"] is True


def test_alg_solution_can_disable_cross_pit_deploy_guard_for_diagnostics(monkeypatch):
    solution_module = importlib.import_module("demo.solution")
    captured_kwargs = {}

    class _FakeBoxPush:
        def __init__(self, **kwargs):
            pass

        def get_action_spec(self):
            return {}

        def reset(self, **kwargs):
            pass

        def predicts(self, obs, current_score):
            return {"action": [[0.0] * 33], "giveup": False}

    class _FakeCross:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def reset(self, **kwargs):
            pass

        def predicts(self, obs, current_score):
            return {"action": [[1.0] * 33], "giveup": False}

    monkeypatch.setenv("ATEC_CROSS_PIT_REQUIRE_DEPLOY_DEFAULTS", "0")
    monkeypatch.setattr(solution_module, "BoxPushPlannerSolution", _FakeBoxPush)
    monkeypatch.setattr(solution_module, "CrossPitBoxPolicy", _FakeCross)

    agent = solution_module.AlgSolution(handoff_time_s=0.0, step_dt=0.02)
    agent.predicts(_fake_official_obs(), current_score=0.0)

    assert captured_kwargs["require_deploy_defaults"] is False


def test_alg_solution_leaves_v1_stabilizer_to_checkpoint_default(monkeypatch):
    solution_module = importlib.import_module("demo.solution")
    captured_kwargs = {}

    class _FakeBoxPush:
        def __init__(self, **kwargs):
            pass

        def get_action_spec(self):
            return {}

        def reset(self, **kwargs):
            pass

        def predicts(self, obs, current_score):
            return {"action": [[0.0] * 33], "giveup": False}

    class _FakeCross:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def reset(self, **kwargs):
            pass

        def predicts(self, obs, current_score):
            return {"action": [[1.0] * 33], "giveup": False}

    monkeypatch.setenv("ATEC_CROSS_PIT_BOX_CHECKPOINT", "cross_pit_box_model_19999.pt")
    monkeypatch.setattr(solution_module, "BoxPushPlannerSolution", _FakeBoxPush)
    monkeypatch.setattr(solution_module, "CrossPitBoxPolicy", _FakeCross)

    agent = solution_module.AlgSolution(handoff_time_s=0.0, step_dt=0.02)
    agent.predicts(_fake_official_obs(), current_score=0.0)

    assert "stabilize_steps" not in captured_kwargs
    assert captured_kwargs["stabilize_vel_gain"] == pytest.approx(0.025)


def test_alg_solution_switches_from_boxpush_to_cross_pit_after_handoff(monkeypatch):
    solution_module = importlib.import_module("demo.solution")

    calls: list[str] = []

    class _FakeBoxPush:
        def __init__(self, **kwargs):
            self.reset_calls = 0
            calls.append(f"box_init:{sorted(kwargs)}")

        def get_action_spec(self):
            return {}

        def reset(self, **kwargs):
            self.reset_calls += 1

        def predicts(self, obs, current_score):
            calls.append("box")
            return {"action": [[0.0] * 33], "giveup": False}

    class _FakeCross:
        def __init__(self, **kwargs):
            self.reset_calls = 0
            calls.append(f"cross_init:{sorted(kwargs)}")

        def reset(self, **kwargs):
            self.reset_calls += 1

        def predicts(self, obs, current_score):
            calls.append("cross")
            return {"action": [[1.0] * 33], "giveup": False}

    monkeypatch.setattr(solution_module, "BoxPushPlannerSolution", _FakeBoxPush)
    monkeypatch.setattr(solution_module, "CrossPitBoxPolicy", _FakeCross)

    agent = solution_module.AlgSolution(handoff_time_s=9.5, step_dt=0.02)
    obs = _fake_official_obs()

    first = agent.predicts(obs, current_score=0.0)
    for _ in range(474):
        agent.predicts(obs, current_score=0.0)
    switched = agent.predicts(obs, current_score=0.0)

    assert "box" in calls
    assert calls[-1] == "cross"
    assert first["action"][0][0] == 0.0
    assert switched["action"][0][0] == 1.0
    assert agent.get_debug_snapshot()["stage"] == "cross_pit_box"


def test_alg_solution_defaults_to_best_verified_handoff_time(monkeypatch):
    solution_module = importlib.import_module("demo.solution")

    class _FakeBoxPush:
        def __init__(self, **kwargs):
            pass

        def get_action_spec(self):
            return {}

        def reset(self, **kwargs):
            pass

        def predicts(self, obs, current_score):
            return {"action": [[0.0] * 33], "giveup": False}

    class _FakeCross:
        def __init__(self, **kwargs):
            pass

        def reset(self, **kwargs):
            pass

        def predicts(self, obs, current_score):
            return {"action": [[1.0] * 33], "giveup": False}

    monkeypatch.delenv("ATEC_CROSS_PIT_HANDOFF_TIME", raising=False)
    monkeypatch.setattr(solution_module, "BoxPushPlannerSolution", _FakeBoxPush)
    monkeypatch.setattr(solution_module, "CrossPitBoxPolicy", _FakeCross)

    agent = solution_module.AlgSolution(step_dt=0.02)

    assert agent.get_debug_snapshot()["handoff_time_s"] == 13.0


def test_debug_push_then_cross_obs_stats_include_image_depth_terms():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "debug_push_then_cross_headless.py"
    source = script_path.read_text()
    tree = __import__("ast").parse(source)
    wanted = {
        "_tensor_stats",
        "_image_stats",
        "_obs_stats",
    }
    module_ast = __import__("ast").Module(
        body=[node for node in tree.body if isinstance(node, __import__("ast").FunctionDef) and node.name in wanted],
        type_ignores=[],
    )
    __import__("ast").fix_missing_locations(module_ast)
    namespace = {"torch": torch, "Any": object}
    exec(compile(module_ast, str(script_path), "exec"), namespace)

    obs = _fake_official_obs()
    obs["image"]["head_depth"] = torch.linspace(0.1, 2.0, 8, dtype=torch.float32).view(1, 2, 4, 1)

    stats = namespace["_obs_stats"](obs)

    assert stats["image"]["head_depth"]["shape"] == [1, 2, 4, 1]
    assert stats["image"]["head_depth"]["min"] == pytest.approx(0.1)
    assert stats["image"]["head_depth"]["max"] == pytest.approx(2.0)


def test_alg_solution_switches_to_walkout_after_cross_pit_far_side(monkeypatch):
    solution_module = importlib.import_module("demo.solution")
    calls: list[str] = []

    class _FakeLowLevel:
        def predict(self, obs, command):
            calls.append(f"walkout_cmd:{command[0].tolist()}")
            return torch.full((1, 33), 3.0)

    class _FakeBoxPush:
        def __init__(self, **kwargs):
            self.low_level = _FakeLowLevel()

        def get_action_spec(self):
            return {}

        def reset(self, **kwargs):
            pass

        def predicts(self, obs, current_score):
            calls.append("box")
            return {"action": [[0.0] * 33], "giveup": False}

    class _FakeCross:
        def __init__(self, **kwargs):
            self.local_x = 0.0
            self.local_y = 0.0
            self.local_root_z = 0.7

        def reset(self, **kwargs):
            pass

        def predicts(self, obs, current_score):
            calls.append("cross")
            return {"action": [[1.0] * 33], "giveup": False}

        def get_debug_snapshot(self):
            return {
                "local_x": self.local_x,
                "local_y": self.local_y,
                "local_root_z": self.local_root_z,
            }

    monkeypatch.setattr(solution_module, "BoxPushPlannerSolution", _FakeBoxPush)
    monkeypatch.setattr(solution_module, "CrossPitBoxPolicy", _FakeCross)
    monkeypatch.setenv("ATEC_WALKOUT_ENABLE", "1")

    agent = solution_module.AlgSolution(handoff_time_s=0.0, step_dt=0.02)
    obs = _fake_official_obs()

    first = agent.predicts(obs, current_score=0.0)
    agent._cross_pit.local_x = 5.85
    agent._cross_pit.local_y = -1.0
    agent._cross_pit.local_root_z = 0.55
    walkout = agent.predicts(obs, current_score=0.0)

    assert first["action"][0][0] == 1.0
    assert walkout["action"][0][0] == 3.0
    assert calls[-1].startswith("walkout_cmd:")
    assert agent.get_debug_snapshot()["stage"] == "walkout"


def test_alg_solution_passes_forward_command_override_to_cross_pit(monkeypatch):
    solution_module = importlib.import_module("demo.solution")
    captured_kwargs = {}

    class _FakeBoxPush:
        def __init__(self, **kwargs):
            pass

        def get_action_spec(self):
            return {}

        def reset(self, **kwargs):
            pass

        def predicts(self, obs, current_score):
            return {"action": [[0.0] * 33], "giveup": False}

    class _FakeCross:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def reset(self, **kwargs):
            pass

        def predicts(self, obs, current_score):
            return {"action": [[1.0] * 33], "giveup": False}

    monkeypatch.setenv("ATEC_CROSS_PIT_FORWARD_COMMAND", "0.6644")
    monkeypatch.setattr(solution_module, "BoxPushPlannerSolution", _FakeBoxPush)
    monkeypatch.setattr(solution_module, "CrossPitBoxPolicy", _FakeCross)

    agent = solution_module.AlgSolution(handoff_time_s=0.0, step_dt=0.02)
    agent.predicts(_fake_official_obs(), current_score=0.0)

    assert captured_kwargs["forward_command"] == pytest.approx(0.6644)


def test_alg_solution_can_use_elapsed_time_from_reset_metadata(monkeypatch):
    solution_module = importlib.import_module("demo.solution")

    class _FakeBoxPush:
        def __init__(self, **kwargs):
            pass

        def get_action_spec(self):
            return {}

        def reset(self, **kwargs):
            pass

        def predicts(self, obs, current_score):
            return {"action": [[0.0] * 33], "giveup": False}

    class _FakeCross:
        def __init__(self, **kwargs):
            pass

        def reset(self, **kwargs):
            pass

        def predicts(self, obs, current_score):
            return {"action": [[2.0] * 33], "giveup": False}

    monkeypatch.setattr(solution_module, "BoxPushPlannerSolution", _FakeBoxPush)
    monkeypatch.setattr(solution_module, "CrossPitBoxPolicy", _FakeCross)

    agent = solution_module.AlgSolution(handoff_time_s=9.5, step_dt=0.02)
    agent.reset(elapsed_time=9.6)

    action = agent.predicts(_fake_official_obs(), current_score=0.0)

    assert action["action"][0][0] == 2.0


def test_demo_dockerfile_packages_push_then_cross_solution_assets():
    dockerfile = Path(__file__).resolve().parent / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")
    active_lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]

    assert "COPY --chown=admin:admin solution.py ./solution/solution.py" in active_lines
    assert "COPY --chown=admin:admin solution_zero.py ./solution/solution.py" not in active_lines

    required_solution_files = [
        "boxpush_planner_solution.py",
        "boxpush_low_level.py",
        "planner_inference_policy.py",
        "cross_pit_box_policy.py",
    ]
    required_assets = [
        "boxpush_planner_v2.pt",
        "boxpush_planner_10hz.pt",
        "gr00t_walk.pt",
        "gr00t_balance.pt",
        "cross_pit_box_v1_5_blind_model_19999.pt",
    ]
    for name in required_solution_files + required_assets:
        assert f"COPY --chown=admin:admin {name} ./solution/" in active_lines
    assert "COPY --chown=admin:admin solution_v1_debug.py ./solution/" not in active_lines
    assert "COPY --chown=admin:admin cross_pit_box_model_19999.pt ./solution/" not in active_lines


def test_demo_v1_debug_entrypoint_documents_local_v1_defaults():
    entrypoint = Path(__file__).resolve().parent / "solution_v1_debug.py"
    text = entrypoint.read_text(encoding="utf-8")

    assert "cross_pit_box_model_19999.pt" in text
    assert '"ATEC_CROSS_PIT_REQUIRE_DEPLOY_DEFAULTS", "0"' in text
    assert '"ATEC_CROSS_PIT_FORWARD_COMMAND", "0.755"' in text
    assert '"ATEC_CROSS_PIT_STABILIZE_STEPS", "2"' in text
    assert "ATEC_PLAY_ATTACH_CROSS_PIT_DEPTH_SCANNER" in text
    assert "class AlgSolution" in text
