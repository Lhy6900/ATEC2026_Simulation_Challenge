from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
CROSS_PIT_DIR = REPO_ROOT / "source" / "atec_rl_lab" / "atec_rl_lab" / "train" / "cross_pit_box"


def _literal_constant(path: Path, name: str) -> float:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return float(ast.literal_eval(node.value))
    raise AssertionError(f"{name} not found in {path}")


def test_cross_pit_v1_and_v2_curriculum_final_target_matches_taskd_x_reached_frame():
    assert _literal_constant(CROSS_PIT_DIR / "env_cfg_v1.py", "CROSS_PIT_SUCCESS_FINAL_X") == 7.8
    assert _literal_constant(CROSS_PIT_DIR / "env_cfg_v2.py", "CROSS_PIT_SUCCESS_FINAL_X") == 7.8


def test_blind_cross_pit_variants_inherit_the_updated_base_targets():
    v1_5_text = (CROSS_PIT_DIR / "env_cfg_v1_5.py").read_text()
    v2_5_text = (CROSS_PIT_DIR / "env_cfg_v2_5.py").read_text()

    assert "from .env_cfg_v1 import CrossPitBoxG1V1EnvCfg" in v1_5_text
    assert "class CrossPitBoxG1V15EnvCfg(CrossPitBoxG1V1EnvCfg)" in v1_5_text
    assert "from .env_cfg_v2 import CrossPitBoxG1V2EnvCfg" in v2_5_text
    assert "class CrossPitBoxG1V25EnvCfg(CrossPitBoxG1V2EnvCfg)" in v2_5_text
