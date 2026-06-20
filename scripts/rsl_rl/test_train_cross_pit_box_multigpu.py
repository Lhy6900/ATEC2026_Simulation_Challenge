import os
import subprocess
from pathlib import Path


def test_cross_pit_box_dry_run_defaults_to_v1_and_torchrun_three_gpus():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "rsl_rl" / "train_cross_pit_box_multigpu.sh"

    assert not (repo_root / "scripts" / "rsl_rl" / "train_box_bridge_multigpu.sh").exists()
    assert script.exists()

    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "MAX_ITERATIONS": "1",
            "RUN_NAME": "dry_run_cross_pit_box",
        }
    )
    env.pop("GPUS", None)
    env.pop("NUM_ENVS_PER_GPU", None)

    result = subprocess.run([str(script)], cwd=repo_root, env=env, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "ATEC-Isaac-TaskD-G1-CrossPitBox-v1" in result.stdout
    assert "GPUs: 1,2,3" in result.stdout
    assert "Launch mode: torchrun" in result.stdout
    assert "Num envs per GPU/rank: 4096" in result.stdout
    assert "BoxBridge" not in result.stdout
    assert "--enable_cameras" not in result.stdout
    assert "--fast_exit" in result.stdout
    assert "Cleanup existing: 1" in result.stdout
    assert "PyTorch CUDA alloc conf:" in result.stdout
    assert "Per-rank torchrun Isaac state: 1" in result.stdout
    assert "--standalone" in result.stdout
    assert "torchrun will map LOCAL_RANK 0..2 onto CUDA_VISIBLE_DEVICES=1,2,3." in result.stdout
    assert "Rank 0 -> logical cuda:0 -> physical GPU 1" in result.stdout
    assert "Rank 1 -> logical cuda:1 -> physical GPU 2" in result.stdout
    assert "Rank 2 -> logical cuda:2 -> physical GPU 3" in result.stdout
    assert "CUDA_VISIBLE_DEVICES=4" not in result.stdout


def test_cross_pit_box_dry_run_can_override_to_per_rank_four_gpus_and_large_env_count():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "rsl_rl" / "train_cross_pit_box_multigpu.sh"

    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "GPUS": "1,2,3,4",
            "LAUNCH_MODE": "per_rank",
            "NUM_ENVS_PER_GPU": "8192",
            "MAX_ITERATIONS": "1",
            "RUN_NAME": "dry_run_cross_pit_box_large",
        }
    )

    result = subprocess.run([str(script)], cwd=repo_root, env=env, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "GPUs: 1,2,3,4" in result.stdout
    assert "Launch mode: per_rank" in result.stdout
    assert "Num envs per GPU/rank: 8192" in result.stdout
    assert "Rank 3 -> CUDA_VISIBLE_DEVICES=4 RANK=3 LOCAL_RANK=0 WORLD_SIZE=4" in result.stdout


def test_cross_pit_box_dry_run_defaults_to_long_training_budget():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "rsl_rl" / "train_cross_pit_box_multigpu.sh"

    env = os.environ.copy()
    env.update({"DRY_RUN": "1"})
    env.pop("MAX_ITERATIONS", None)

    result = subprocess.run([str(script)], cwd=repo_root, env=env, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "--max_iterations 20000" in result.stdout


def test_cross_pit_box_dry_run_can_select_v1_task():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "rsl_rl" / "train_cross_pit_box_multigpu.sh"

    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "TASK_NAME": "ATEC-Isaac-TaskD-G1-CrossPitBox-v1",
            "GPUS": "1",
            "NUM_ENVS_PER_GPU": "128",
            "MAX_ITERATIONS": "1",
            "RUN_NAME": "dry_run_cross_pit_box_v1",
        }
    )

    result = subprocess.run([str(script)], cwd=repo_root, env=env, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "Task: ATEC-Isaac-TaskD-G1-CrossPitBox-v1" in result.stdout
    assert "--task ATEC-Isaac-TaskD-G1-CrossPitBox-v1" in result.stdout


def test_cross_pit_box_9p5s_wrapper_defaults_to_v2_task_and_run_name():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "rsl_rl" / "train_cross_pit_box_9p5s_multigpu.sh"

    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "MAX_ITERATIONS": "1",
        }
    )
    env.pop("TASK_NAME", None)
    env.pop("RUN_NAME", None)
    env.pop("GPUS", None)
    env.pop("NUM_ENVS_PER_GPU", None)

    result = subprocess.run([str(script)], cwd=repo_root, env=env, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "Task: ATEC-Isaac-TaskD-G1-CrossPitBox-v2" in result.stdout
    assert "Launch mode: per_rank" in result.stdout
    assert "Run name: cross_pit_box_v2_9p5s_mlp_pbrs_x_gpus_123_env4096" in result.stdout
    assert "--task ATEC-Isaac-TaskD-G1-CrossPitBox-v2" in result.stdout


def test_cross_pit_box_blind_v1_5_wrapper_defaults_to_v1_5_task_and_run_name():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "rsl_rl" / "train_cross_pit_box_v1_5_blind_multigpu.sh"

    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "MAX_ITERATIONS": "1",
        }
    )
    env.pop("TASK_NAME", None)
    env.pop("RUN_NAME", None)
    env.pop("GPUS", None)
    env.pop("NUM_ENVS_PER_GPU", None)

    result = subprocess.run([str(script)], cwd=repo_root, env=env, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "Task: ATEC-Isaac-TaskD-G1-CrossPitBox-v1.5" in result.stdout
    assert "Launch mode: per_rank" in result.stdout
    assert "Run name: cross_pit_box_v1_5_blind_mlp_pbrs_x_gpus_123_env4096" in result.stdout
    assert "--task ATEC-Isaac-TaskD-G1-CrossPitBox-v1.5" in result.stdout


def test_cross_pit_box_blind_v2_5_wrapper_defaults_to_v2_5_task_and_run_name():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "rsl_rl" / "train_cross_pit_box_v2_5_blind_multigpu.sh"

    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "MAX_ITERATIONS": "1",
        }
    )
    env.pop("TASK_NAME", None)
    env.pop("RUN_NAME", None)
    env.pop("GPUS", None)
    env.pop("NUM_ENVS_PER_GPU", None)

    result = subprocess.run([str(script)], cwd=repo_root, env=env, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "Task: ATEC-Isaac-TaskD-G1-CrossPitBox-v2.5" in result.stdout
    assert "GPUs: 5,6,7" in result.stdout
    assert "Launch mode: torchrun" in result.stdout
    assert "Run name: cross_pit_box_v2_5_blind_mlp_pbrs_x_gpus_567_env4096" in result.stdout
    assert "Cleanup existing: 0" in result.stdout
    assert "Master: 127.0.0.1:29592" in result.stdout
    assert "Per-rank torchrun Isaac state: 1" in result.stdout
    assert "--task ATEC-Isaac-TaskD-G1-CrossPitBox-v2.5" in result.stdout
    assert "torchrun will map LOCAL_RANK 0..2 onto CUDA_VISIBLE_DEVICES=5,6,7." in result.stdout
    assert "Rank 0 -> logical cuda:0 -> physical GPU 5" in result.stdout
    assert "Rank 1 -> logical cuda:1 -> physical GPU 6" in result.stdout
    assert "Rank 2 -> logical cuda:2 -> physical GPU 7" in result.stdout


def test_cross_pit_box_dry_run_can_enable_rendering_cameras_explicitly():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "rsl_rl" / "train_cross_pit_box_multigpu.sh"

    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "ENABLE_CAMERAS": "1",
            "GPUS": "1",
            "NUM_ENVS_PER_GPU": "128",
            "MAX_ITERATIONS": "1",
            "RUN_NAME": "dry_run_cross_pit_box_enable_cameras",
        }
    )

    result = subprocess.run([str(script)], cwd=repo_root, env=env, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "Enable cameras: 1" in result.stdout
    assert "--enable_cameras" in result.stdout


def test_cross_pit_box_dry_run_can_disable_fast_exit_for_debugging():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "rsl_rl" / "train_cross_pit_box_multigpu.sh"

    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "FAST_EXIT_AFTER_TRAIN": "0",
            "GPUS": "1",
            "NUM_ENVS_PER_GPU": "128",
            "MAX_ITERATIONS": "1",
            "RUN_NAME": "dry_run_cross_pit_box_disable_fast_exit",
        }
    )

    result = subprocess.run([str(script)], cwd=repo_root, env=env, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "Fast exit after train: 0" in result.stdout
    assert "--fast_exit" not in result.stdout


def test_cross_pit_box_dry_run_can_disable_existing_process_cleanup():
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "rsl_rl" / "train_cross_pit_box_multigpu.sh"

    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "CLEANUP_EXISTING": "0",
            "GPUS": "1",
            "NUM_ENVS_PER_GPU": "128",
            "MAX_ITERATIONS": "1",
            "RUN_NAME": "dry_run_cross_pit_box_no_cleanup",
        }
    )

    result = subprocess.run([str(script)], cwd=repo_root, env=env, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    assert "Cleanup existing: 0" in result.stdout


def test_cross_pit_box_train_script_has_defensive_process_cleanup():
    repo_root = Path(__file__).resolve().parents[2]
    script_text = (repo_root / "scripts" / "rsl_rl" / "train_cross_pit_box_multigpu.sh").read_text()

    assert "cleanup_existing_training" in script_text
    assert "terminate_pids" in script_text
    assert "trap cleanup INT TERM EXIT" in script_text
    assert "wait -n" in script_text
    assert 'index($0, "train_cross_pit_box_multigpu.sh")' not in script_text


def test_cross_pit_box_train_script_logs_rank_outputs_for_debugging():
    repo_root = Path(__file__).resolve().parents[2]
    script_text = (repo_root / "scripts" / "rsl_rl" / "train_cross_pit_box_multigpu.sh").read_text()

    assert "LOG_RANK_OUTPUTS" in script_text
    assert "RANK_LOG_DIR" in script_text
    assert "rank${rank}.log" in script_text
    assert "tee" in script_text


def test_cross_pit_box_existing_cleanup_does_not_match_launcher_script_name():
    repo_root = Path(__file__).resolve().parents[2]
    script_text = (repo_root / "scripts" / "rsl_rl" / "train_cross_pit_box_multigpu.sh").read_text()

    assert "train_cross_pit_box_multigpu.sh" not in script_text.split("find_existing_training_pids()", 1)[1].split("}", 1)[0]


def test_cross_pit_box_existing_cleanup_uses_proc_cmdline_not_ps_pipeline_text():
    repo_root = Path(__file__).resolve().parents[2]
    script_text = (repo_root / "scripts" / "rsl_rl" / "train_cross_pit_box_multigpu.sh").read_text()
    cleanup_body = script_text.split("find_existing_training_pids()", 1)[1].split("cleanup_existing_training()", 1)[0]

    assert "/proc/${pid}/cmdline" in cleanup_body
    assert "ps -eo" not in cleanup_body
    assert "awk" not in cleanup_body


def test_cross_pit_box_existing_cleanup_matches_all_cross_pit_box_versions():
    repo_root = Path(__file__).resolve().parents[2]
    script_text = (repo_root / "scripts" / "rsl_rl" / "train_cross_pit_box_multigpu.sh").read_text()
    cleanup_body = script_text.split("find_existing_training_pids()", 1)[1].split("cleanup_existing_training()", 1)[0]

    assert "ATEC-Isaac-TaskD-G1-CrossPitBox" in cleanup_body
    assert "--task ${TASK_NAME}" not in cleanup_body


def test_cross_pit_box_train_script_syncs_resume_state_after_checkpoint_load():
    repo_root = Path(__file__).resolve().parents[2]
    train_script = (repo_root / "scripts" / "rsl_rl" / "train.py").read_text()

    assert "sync_resume_training_state(env, runner)" in train_script
    assert train_script.index("runner.load(resume_path)") < train_script.index("sync_resume_training_state(env, runner)")


def test_cross_pit_box_train_script_waits_for_rank_zero_setup_before_master_store():
    repo_root = Path(__file__).resolve().parents[2]
    script_text = (repo_root / "scripts" / "rsl_rl" / "train_cross_pit_box_multigpu.sh").read_text()

    assert "WAIT_FOR_MASTER" in script_text
    assert "MASTER_READY_TIMEOUT_SECONDS" in script_text
    assert "wait_for_master_store" in script_text
    assert "master_store_listening" in script_text
    assert "Waiting for rank 0 TCPStore" in script_text
    assert 'launch_rank 0\nPIDS+=("$!")' in script_text

    main_flow = script_text.split('launch_rank 0\nPIDS+=("$!")', 1)[1]
    assert main_flow.index('wait_for_rank_setup "0"') < main_flow.index("wait_for_master_store")
    assert main_flow.index("wait_for_master_store") < main_flow.index("for ((rank = 1; rank < NPROC_PER_NODE; rank++))")


def test_cross_pit_box_train_script_stages_rank_launches_until_environment_setup_completes():
    repo_root = Path(__file__).resolve().parents[2]
    script_text = (repo_root / "scripts" / "rsl_rl" / "train_cross_pit_box_multigpu.sh").read_text()

    assert "WAIT_FOR_RANK_SETUP" in script_text
    assert "RANK_SETUP_TIMEOUT_SECONDS" in script_text
    assert "wait_for_rank_setup" in script_text
    assert "rank_setup_completed" in script_text
    assert "[INFO]: Completed setting up the environment" in script_text
    assert 'wait_for_rank_setup "${rank}"' in script_text


def test_cross_pit_box_train_script_isolates_isaac_user_state_per_rank():
    repo_root = Path(__file__).resolve().parents[2]
    script_text = (repo_root / "scripts" / "rsl_rl" / "train_cross_pit_box_multigpu.sh").read_text()

    assert "LAUNCH_ID" in script_text
    assert "RANK_STATE_ROOT" in script_text
    assert "rank_state_dir" in script_text
    assert "XDG_CACHE_HOME" in script_text
    assert "XDG_CONFIG_HOME" in script_text
    assert "XDG_DATA_HOME" in script_text
    assert "OMNI_USER_CACHE_DIR" in script_text
    assert "OMNI_USER_DATA_DIR" in script_text
    assert "OMNI_USER_LOG_DIR" in script_text
    assert 'TMPDIR="${rank_state_dir}/tmp"' in script_text
    assert 'TMPDIR="${RANK_STATE_ROOT}/torchrun/tmp"' in script_text
    assert "PER_RANK_TORCHRUN_STATE" in script_text
    assert 'ATEC_PER_RANK_OMNI_STATE="${PER_RANK_TORCHRUN_STATE}"' in script_text
    assert 'ATEC_RANK_STATE_ROOT="${RANK_STATE_ROOT}/torchrun_ranks"' in script_text


def test_train_py_configures_rank_local_isaac_state_before_app_launcher():
    repo_root = Path(__file__).resolve().parents[2]
    train_text = (repo_root / "scripts" / "rsl_rl" / "train.py").read_text()

    assert "_configure_rank_local_isaac_state()" in train_text
    assert "ATEC_PER_RANK_OMNI_STATE" in train_text
    assert "ATEC_RANK_STATE_ROOT" in train_text
    assert "OMNI_USER_CACHE_DIR" in train_text
    assert "OMNI_USER_DATA_DIR" in train_text
    assert "OMNI_USER_LOG_DIR" in train_text
    assert train_text.index("_configure_rank_local_isaac_state()") < train_text.index("AppLauncher.add_app_launcher_args")
    assert train_text.index("_configure_rank_local_isaac_state()") < train_text.index("app_launcher = AppLauncher")


def test_cross_pit_box_train_script_serializes_isaac_startup_across_jobs():
    repo_root = Path(__file__).resolve().parents[2]
    script_text = (repo_root / "scripts" / "rsl_rl" / "train_cross_pit_box_multigpu.sh").read_text()

    assert "SERIALIZE_ISAAC_STARTUP" in script_text
    assert "ISAAC_STARTUP_LOCK_FILE" in script_text
    assert "TORCHRUN_STARTUP_LOCK_SECONDS" in script_text
    assert "acquire_isaac_startup_lock" in script_text
    assert "release_isaac_startup_lock" in script_text
    assert "flock" in script_text
    assert 'torchrun_pid="$!"' in script_text

    startup_flow = script_text.split("\nacquire_isaac_startup_lock\n\nlaunch_rank 0", 1)[1]
    assert startup_flow.index('wait_for_rank_setup "${rank}"') < startup_flow.index("\nrelease_isaac_startup_lock\n")
