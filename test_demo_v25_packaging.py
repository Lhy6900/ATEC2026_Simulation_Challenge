from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent
DEMO_V25_DIR = REPO_ROOT / "demoV2.5"


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


def _import_demo_v25_policy(monkeypatch):
    monkeypatch.syspath_prepend(str(DEMO_V25_DIR))
    sys.modules.pop("cross_pit_box_policy", None)
    return importlib.import_module("cross_pit_box_policy")


def test_demo_v25_folder_contains_submission_assets():
    expected_files = {
        "solution.py",
        "server.py",
        "run.sh",
        "requirements.txt",
        "boxpush_planner_solution.py",
        "boxpush_low_level.py",
        "planner_inference_policy.py",
        "cross_pit_box_policy.py",
        "boxpush_planner_v2.pt",
        "boxpush_planner_10hz.pt",
        "gr00t_walk.pt",
        "gr00t_balance.pt",
        "cross_pit_box_v2_5_blind_model_19999.pt",
        "Dockerfile",
    }

    missing = sorted(name for name in expected_files if not (DEMO_V25_DIR / name).exists())

    assert missing == []


def test_demo_v25_solution_defaults_to_9p5s_handoff_and_final_v25_checkpoint():
    solution_text = (DEMO_V25_DIR / "solution.py").read_text()

    assert "default_handoff_time = 9.5" in solution_text
    assert "cross_pit_box_v2_5_blind_model_19999.pt" in solution_text
    assert '"target_x": 5.8' in solution_text
    assert '"stabilize_vel_gain": 0.0' in solution_text
    assert '"stabilize_steps": 0' in solution_text
    assert '"initial_last_action": os.environ.get("ATEC_CROSS_PIT_INITIAL_LAST_ACTION", "env_raw")' in solution_text


def test_demo_v25_dockerfile_packages_real_v25_solution():
    dockerfile = (DEMO_V25_DIR / "Dockerfile").read_text()

    assert "COPY --chown=admin:admin solution.py ./solution/solution.py" in dockerfile
    assert "COPY --chown=admin:admin cross_pit_box_v2_5_blind_model_19999.pt ./solution/" in dockerfile
    assert "cross_pit_box_v1_5_blind_model_19999.pt" not in dockerfile


def test_demo_v25_policy_loads_v25_blind_checkpoint_with_9p5s_defaults(monkeypatch):
    policy_module = _import_demo_v25_policy(monkeypatch)

    for name in (
        "ATEC_CROSS_PIT_BOX_CHECKPOINT",
        "ATEC_CROSS_PIT_STABILIZE_STEPS",
        "ATEC_CROSS_PIT_INITIAL_LAST_ACTION",
        "ATEC_CROSS_PIT_REQUIRE_DEPLOY_DEFAULTS",
    ):
        monkeypatch.delenv(name, raising=False)

    policy = policy_module.CrossPitBoxPolicy(device="cpu")

    assert policy.model_path.name == "cross_pit_box_v2_5_blind_model_19999.pt"
    assert policy.actor_obs_dim == policy_module.BLIND_ACTOR_OBS_DIM
    assert not policy.uses_heightmap
    assert policy.target_x == pytest.approx(5.8)
    assert policy.stabilize_steps == 0
    assert policy.stabilize_vel_gain == pytest.approx(0.0)
    assert policy.initial_last_action == "env_raw"
    assert policy_module.CROSS_PIT_HANDOFF_LOCAL_X == pytest.approx(3.4137089252471924)
    assert policy_module.CROSS_PIT_HANDOFF_LOCAL_Y == pytest.approx(-1.5012480020523071)
    assert policy_module.CROSS_PIT_HANDOFF_YAW == pytest.approx(-0.13635359439321332)


def test_demo_v25_policy_rebases_joint_pos_to_counter950_handoff(monkeypatch):
    policy_module = _import_demo_v25_policy(monkeypatch)
    policy = policy_module.CrossPitBoxPolicy(device="cpu", require_deploy_defaults=True)
    obs = _fake_official_obs()
    obs["proprio"][:, 12:45] = policy_module.CROSS_PIT_OFFICIAL_JOINT_POS_OFFSET.view(1, -1)

    policy_obs = policy.build_policy_obs(obs)

    assert policy_obs.shape == (1, policy_module.BLIND_ACTOR_OBS_DIM)
    assert torch.allclose(policy_obs[:, 9:42], torch.zeros(1, 33), atol=1.0e-6)


def test_demo_v25_policy_can_seed_first_last_action_from_raw_taskd_env_action(monkeypatch):
    policy_module = _import_demo_v25_policy(monkeypatch)
    monkeypatch.setenv("ATEC_CROSS_PIT_INITIAL_LAST_ACTION", "env_raw")
    policy = policy_module.CrossPitBoxPolicy(device="cpu", require_deploy_defaults=True)
    obs = _fake_official_obs()
    env_action = torch.linspace(-0.5, 0.5, 33).view(1, 33)
    obs["proprio"][:, 78:111] = env_action

    policy_obs = policy.build_policy_obs(obs)

    assert torch.allclose(policy_obs[:, 75:108], env_action, atol=1.0e-6)


def test_play_atec_task_can_load_solution_from_file_for_local_v25_validation():
    script_text = (REPO_ROOT / "scripts" / "play_atec_task.py").read_text()

    assert "ATEC_SOLUTION_FILE" in script_text
    assert "importlib.util.spec_from_file_location" in script_text
