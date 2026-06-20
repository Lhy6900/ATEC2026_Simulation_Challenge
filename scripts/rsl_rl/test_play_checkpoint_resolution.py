from pathlib import Path


def test_play_resolves_checkpoint_filename_with_load_run():
    repo_root = Path(__file__).resolve().parents[2]
    play_script = (repo_root / "scripts" / "rsl_rl" / "play.py").read_text()

    assert "get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)" in play_script
    assert "retrieve_file_path(args_cli.checkpoint)" not in play_script


def test_play_sets_simulation_log_dir_to_checkpoint_run_dir():
    repo_root = Path(__file__).resolve().parents[2]
    play_script = (repo_root / "scripts" / "rsl_rl" / "play.py").read_text()

    assert "env_cfg.log_dir = log_dir" in play_script
    assert "env_cfg.sim.log_dir = log_dir" in play_script


def test_play_preserves_cross_pit_box_fixed_single_tile_terrain():
    repo_root = Path(__file__).resolve().parents[2]
    play_script = (repo_root / "scripts" / "rsl_rl" / "play.py").read_text()

    assert 'task_name.endswith("CrossPitBox-v0")' in play_script
    assert 'task_name.endswith("CrossPitBox-v1")' in play_script
    assert 'task_name.endswith("CrossPitBox-v1.5")' in play_script
    assert 'task_name.endswith("CrossPitBox-v2")' in play_script
    assert 'task_name.endswith("CrossPitBox-v2.5")' in play_script
    assert "if not is_cross_pit_box_task:" in play_script


def test_play_can_fix_cross_pit_box_success_target_for_checkpoint_evaluation():
    repo_root = Path(__file__).resolve().parents[2]
    play_script = (repo_root / "scripts" / "rsl_rl" / "play.py").read_text()

    assert "--cross_pit_success_x" in play_script
    assert "env_cfg.curriculum.success_x = None" in play_script
    assert "env.unwrapped.cross_pit_success_x = float(args_cli.cross_pit_success_x)" in play_script
