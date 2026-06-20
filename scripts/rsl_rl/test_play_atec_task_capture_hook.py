from pathlib import Path


def _play_script_text() -> str:
    return (Path(__file__).resolve().parents[1] / "play_atec_task.py").read_text()


def test_play_capture_hook_is_env_var_gated():
    script_text = _play_script_text()

    assert "ATEC_PLAY_CAPTURE_TIME" in script_text
    assert "ATEC_PLAY_CAPTURE_JSON" in script_text
    assert "ATEC_PLAY_CAPTURE_PNG" in script_text
    assert "os.environ.get(\"ATEC_PLAY_CAPTURE_TIME\")" in script_text
    assert "os.environ.get(\"ATEC_PLAY_CAPTURE_JSON\")" in script_text
    assert "os.environ.get(\"ATEC_PLAY_CAPTURE_PNG\")" in script_text


def test_play_capture_hook_runs_before_done_break():
    script_text = _play_script_text()

    capture_call = script_text.index("_maybe_write_play_capture(")
    done_check = script_text.index("done = (terminated.item() or truncated.item())")

    assert capture_call < done_check


def test_play_capture_hook_records_full_input_action():
    script_text = _play_script_text()

    assert '"input_action": _json_ready(action)' in script_text
    assert "action=actions" in script_text


def test_play_capture_png_sidecar_does_not_reuse_state_json_suffix():
    script_text = _play_script_text()

    assert 'path.with_suffix(path.suffix + ".json")' in script_text
    assert 'path.with_suffix(".json")' not in script_text


def test_play_final_capture_hook_is_env_var_gated_and_runs_on_done():
    script_text = _play_script_text()

    assert "ATEC_PLAY_FINAL_CAPTURE_JSON" in script_text
    assert "final_capture_json = os.environ.get(\"ATEC_PLAY_FINAL_CAPTURE_JSON\")" in script_text
    assert 'capture_reason="done"' in script_text

    done_check = script_text.index("done = (terminated.item() or truncated.item())")
    final_capture_call = script_text.index('capture_reason="done"')
    done_break = script_text.index("if done:", done_check)

    assert done_check < final_capture_call < done_break


def test_play_trace_and_capture_record_named_done_terms():
    script_text = _play_script_text()

    assert "def _done_terms(" in script_text
    assert '"done_terms": _done_terms(env)' in script_text


def test_play_cross_pit_depth_scanner_debug_hook_is_env_var_gated():
    script_text = _play_script_text()

    assert "ATEC_PLAY_ATTACH_CROSS_PIT_DEPTH_SCANNER" in script_text
    assert "attach_cross_pit_depth_scanner = _env_flag(\"ATEC_PLAY_ATTACH_CROSS_PIT_DEPTH_SCANNER\")" in script_text
    assert "if attach_cross_pit_depth_scanner:" in script_text
    assert "env_cfg.scene.depth_scanner = _make_cross_pit_depth_scanner_cfg()" in script_text


def test_play_cross_pit_depth_scanner_matches_training_raycaster_shape():
    script_text = _play_script_text()

    assert "from isaaclab.sensors import MultiMeshRayCasterCfg, patterns" in script_text
    assert 'prim_path="{ENV_REGEX_NS}/Robot/d435_link"' in script_text
    assert "patterns.GridPatternCfg(" in script_text
    assert "resolution=0.1" in script_text
    assert "size=(2.3000000000000003, 1.5)" in script_text
    assert 'ray_alignment="yaw"' in script_text
    assert "MultiMeshRayCasterCfg.OffsetCfg(pos=(1.3, 0.0, 2.5))" in script_text
    assert 'prim_expr="{ENV_REGEX_NS}/Box"' in script_text


def test_play_cross_pit_depth_scanner_passes_debug_ray_hits_to_solution():
    script_text = _play_script_text()

    assert "def _attach_cross_pit_debug_obs(" in script_text
    assert '"_debug_root_pos_w"' in script_text
    assert '"_debug_root_quat_w"' in script_text
    assert '"_debug_depth_ray_hits_w"' in script_text
    assert "obs_for_policy = _attach_cross_pit_debug_obs(env, obs)" in script_text
    assert "resp = solution.predicts(obs_for_policy, total_episode_reward)" in script_text
