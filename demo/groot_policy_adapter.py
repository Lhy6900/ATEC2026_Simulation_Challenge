import math
import os
from collections import deque
from typing import Any

import numpy as np


# ATEC G1 joint order (33 DoF):
# 29 body joints + 4 Dex1 finger joints.
ATEC_G1_JOINT_NAMES_33 = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hand_Joint1_1",
    "left_hand_Joint2_1",
    "right_hand_Joint1_1",
    "right_hand_Joint2_1",
]

# GR00T body model order (29 DoF) matches the ATEC body subset.
GR00T_BODY_JOINT_NAMES_29 = ATEC_G1_JOINT_NAMES_33[:29]

# ATEC environment consumes relative actions with its own scale.
ATEC_ENV_ACTION_SCALE = 0.5
ATEC_BODY_DIM = 29
ATEC_ACTION_DIM = 33
ATEC_FINGER_DIM = 4

DEFAULT_BASE_HEIGHT_CMD = 0.74
DEFAULT_NAV_CMD = [0.0, 0.0, 0.0]

# The first 29 ATEC joints are exactly the GR00T body joints.
ATEC_TO_GR00T_BODY_INDICES = list(range(ATEC_BODY_DIM))

# Default joint pose from the ATEC asset.
ATEC_G1_DEFAULT_33 = [
    -0.10,
    0.0,
    0.0,
    0.30,
    -0.20,
    0.0,
    -0.10,
    0.0,
    0.0,
    0.30,
    -0.20,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.3,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.3,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]

# Conservative upper-body home pose borrowed from the ATEC default pose.
DEFAULT_UPPER_BODY_TARGET_29 = ATEC_G1_DEFAULT_33[:29]

GR00T_LOWER_BODY_DEFAULT_15 = [
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0,
]
GR00T_CMD_SCALE = [2.0, 2.0, 0.5]
GR00T_ANG_VEL_SCALE = 0.5
GR00T_DOF_POS_SCALE = 1.0
GR00T_DOF_VEL_SCALE = 0.05
GR00T_ACTION_SCALE = 0.25
GR00T_HISTORY_LEN = 6
GR00T_NUM_ACTIONS = 15
GR00T_SINGLE_OBS_DIM = 86
GR00T_NUM_OBS = GR00T_SINGLE_OBS_DIM * GR00T_HISTORY_LEN

LOWER_BODY_ATEC_INDICES = list(range(15))

# Canonical indices for waist joints in the 29-DoF body vector.
WAIST_CANONICAL_INDICES = [12, 13, 14]  # waist_yaw, waist_roll, waist_pitch


def compute_torso_rpy_from_upper_body(
    waist_q: list[float] | np.ndarray,
) -> list[float]:
    """Compute torso RPY relative to pelvis via analytical waist FK.

    G1 waist chain: pelvis -> yaw(z) -> roll(x) -> pitch(y) -> torso_link.
    This matches the FK computation in GR00T's g1_decoupled_whole_body_policy.py
    and compare_sbt's DecoupledAdapterV2._compute_torso_rpy_from_upper_semantic_pose.
    In both references the FK is evaluated with base orientation = identity, so
    pelvis yaw removal is a no-op and torso_rpy equals the RPY of the waist rotation.
    """
    waist = np.asarray(waist_q, dtype=np.float64)
    if waist.shape != (3,):
        raise ValueError(f"Expected 3 waist values, got {waist.shape}")

    yaw, roll, pitch = float(waist[0]), float(waist[1]), float(waist[2])

    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)

    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R = Rz @ Rx @ Ry

    # Extract RPY (xyz convention)
    p = math.asin(np.clip(-R[2, 0], -1.0, 1.0))
    cp2 = math.cos(p)
    if abs(cp2) > 1e-8:
        r = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(R[1, 0], R[0, 0])
    else:
        r = math.atan2(-R[1, 2], R[1, 1])
        y = 0.0

    return [r, p, y]

# Lower-body actuator gains aligned to the GR00T lower-body policy config.
ALIGNED_LOWER_BODY_KP = [
    150.0,
    150.0,
    150.0,
    200.0,
    40.0,
    40.0,
    150.0,
    150.0,
    150.0,
    200.0,
    40.0,
    40.0,
    250.0,
    250.0,
    250.0,
]
ALIGNED_LOWER_BODY_KD = [
    2.0,
    2.0,
    2.0,
    4.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    4.0,
    2.0,
    2.0,
    5.0,
    5.0,
    5.0,
]


def _to_flat_list(value: Any) -> list[float]:
    """Convert nested tensor/array/list-like values into a flat Python list."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, (int, float)):
        return [float(value)]

    flat: list[float] = []

    def _walk(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for sub_item in item:
                _walk(sub_item)
        else:
            flat.append(float(item))

    _walk(value)
    return flat


class GrootPolicyAdapter:
    """Minimal adapter that converts ATEC obs into a GR00T-like body action."""

    def __init__(
        self,
        load_runtime: bool = True,
        fixed_nav_cmd: list[float] | None = None,
        base_height_cmd: float = DEFAULT_BASE_HEIGHT_CMD,
        nav_scale: float = 0.0,
        lower_body_blend: float = 1.0,
    ) -> None:
        self.fixed_nav_cmd = list(fixed_nav_cmd or DEFAULT_NAV_CMD)
        self.base_height_cmd = float(base_height_cmd)
        self.nav_scale = float(nav_scale)
        self.lower_body_blend = float(lower_body_blend)
        self._runtime = None
        self._runtime_error = None
        self._obs_history: deque[list[float]] = deque(maxlen=GR00T_HISTORY_LEN)
        self._last_lower_body_action = [0.0] * GR00T_NUM_ACTIONS
        self._debug_snapshot = {
            "runtime_loaded": False,
            "nav_scale": self.nav_scale,
            "nav_cmd": [0.0, 0.0, 0.0],
            "base_height_cmd": self.base_height_cmd,
        }
        if load_runtime:
            self._try_load_runtime()

    def reset(self) -> None:
        """Reset episodic state. Kept minimal for the first submission version."""
        self._obs_history.clear()
        self._last_lower_body_action = [0.0] * GR00T_NUM_ACTIONS
        return None

    def _try_load_runtime(self) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on runtime image
            self._runtime_error = exc
            self._runtime = None
            return

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_paths = {
            "balance": os.path.join(base_dir, "GR00T-WholeBodyControl-Balance.onnx"),
            "walk": os.path.join(base_dir, "GR00T-WholeBodyControl-Walk.onnx"),
        }
        if not all(os.path.exists(path) for path in model_paths.values()):
            self._runtime = None
            return

        self._runtime = {
            "ort": ort,
            "balance": ort.InferenceSession(model_paths["balance"]),
            "walk": ort.InferenceSession(model_paths["walk"]),
        }
        self._debug_snapshot["runtime_loaded"] = True

    def get_navigation_command(self) -> list[float]:
        nav_cmd = [self.nav_scale * value for value in self.fixed_nav_cmd]
        self._debug_snapshot["nav_scale"] = self.nav_scale
        self._debug_snapshot["nav_cmd"] = list(nav_cmd)
        return nav_cmd

    def set_navigation_scale(self, scale: float) -> None:
        self.nav_scale = max(0.0, min(1.0, float(scale)))

    def set_navigation_command(self, cmd: list[float]) -> None:
        self.fixed_nav_cmd = list(cmd)

    def set_base_height_cmd(self, height: float) -> None:
        self.base_height_cmd = float(height)

    def _select_policy_session(self, nav_cmd: list[float]) -> str:
        nav_norm = math.sqrt(sum(value * value for value in nav_cmd))
        return "walk" if nav_norm >= 0.05 else "balance"

    def extract_joint_state(self, obs: dict[str, Any]) -> dict[str, list[float]]:
        proprio = _to_flat_list(obs["proprio"])
        expected = 12 + ATEC_ACTION_DIM * 3
        if len(proprio) < expected:
            raise ValueError(
                f"Expected proprio with at least {expected} values, got {len(proprio)}"
            )

        idx = 0
        base_lin_vel = proprio[idx : idx + 3]
        idx += 3
        base_ang_vel = proprio[idx : idx + 3]
        idx += 3
        velocity_commands = proprio[idx : idx + 3]
        idx += 3
        projected_gravity = proprio[idx : idx + 3]
        idx += 3

        q_rel_33 = proprio[idx : idx + ATEC_ACTION_DIM]
        idx += ATEC_ACTION_DIM
        dq_rel_33 = proprio[idx : idx + ATEC_ACTION_DIM]

        q_atec_33 = [
            default_q + relative_q for default_q, relative_q in zip(ATEC_G1_DEFAULT_33, q_rel_33)
        ]
        dq_atec_33 = list(dq_rel_33)
        q_gr00t_29 = [q_atec_33[idx] for idx in ATEC_TO_GR00T_BODY_INDICES]
        dq_gr00t_29 = [dq_atec_33[idx] for idx in ATEC_TO_GR00T_BODY_INDICES]

        return {
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "velocity_commands": velocity_commands,
            "projected_gravity": projected_gravity,
            "q_atec_33": q_atec_33,
            "dq_atec_33": dq_atec_33,
            "q_gr00t_29": q_gr00t_29,
            "dq_gr00t_29": dq_gr00t_29,
        }

    def _build_gr00t_observation(self, joint_state: dict[str, list[float]]) -> list[float]:
        q_body = joint_state["q_gr00t_29"]
        dq_body = joint_state["dq_gr00t_29"]
        q_lower = q_body[:ATEC_BODY_DIM]
        dq_lower = dq_body[:ATEC_BODY_DIM]

        q_scaled = []
        dq_scaled = []
        for idx in range(ATEC_BODY_DIM):
            default_q = (
                GR00T_LOWER_BODY_DEFAULT_15[idx]
                if idx < len(GR00T_LOWER_BODY_DEFAULT_15)
                else 0.0
            )
            q_scaled.append((q_lower[idx] - default_q) * GR00T_DOF_POS_SCALE)
            dq_scaled.append(dq_lower[idx] * GR00T_DOF_VEL_SCALE)

        cmd = self.get_navigation_command()
        omega_scaled = [value * GR00T_ANG_VEL_SCALE for value in joint_state["base_ang_vel"]]
        gravity = list(joint_state["projected_gravity"])
        waist_q = [q_body[i] for i in WAIST_CANONICAL_INDICES]
        torso_rpy = compute_torso_rpy_from_upper_body(waist_q)

        single_obs = [0.0] * GR00T_SINGLE_OBS_DIM
        single_obs[0:3] = [cmd[i] * GR00T_CMD_SCALE[i] for i in range(3)]
        single_obs[3] = self.base_height_cmd
        single_obs[4:7] = torso_rpy
        single_obs[7:10] = omega_scaled
        single_obs[10:13] = gravity
        single_obs[13 : 13 + ATEC_BODY_DIM] = q_scaled
        single_obs[13 + ATEC_BODY_DIM : 13 + 2 * ATEC_BODY_DIM] = dq_scaled
        single_obs[13 + 2 * ATEC_BODY_DIM : 13 + 2 * ATEC_BODY_DIM + GR00T_NUM_ACTIONS] = (
            self._last_lower_body_action
        )
        return single_obs

    def _stack_history(self, single_obs: list[float]) -> list[float]:
        self._obs_history.append(single_obs)
        while len(self._obs_history) < GR00T_HISTORY_LEN:
            self._obs_history.appendleft([0.0] * GR00T_SINGLE_OBS_DIM)

        full_obs: list[float] = []
        for hist_obs in self._obs_history:
            full_obs.extend(hist_obs)
        if len(full_obs) != GR00T_NUM_OBS:
            raise ValueError(f"Expected {GR00T_NUM_OBS} obs values, got {len(full_obs)}")
        return full_obs

    def _run_lower_body_policy(self, joint_state: dict[str, list[float]]) -> list[float]:
        if self._runtime is None:
            self._debug_snapshot["policy_session"] = "fallback_default_pose"
            return list(GR00T_LOWER_BODY_DEFAULT_15)

        ort = self._runtime["ort"]
        single_obs = self._build_gr00t_observation(joint_state)
        full_obs = self._stack_history(single_obs)
        nav_cmd = self._debug_snapshot.get("nav_cmd", [0.0, 0.0, 0.0])
        session_key = self._select_policy_session(nav_cmd)
        self._debug_snapshot["policy_session"] = session_key
        self._debug_snapshot["obs_head"] = [round(value, 4) for value in full_obs[:12]]
        session = self._runtime[session_key]
        input_name = session.get_inputs()[0].name
        obs_array = ort.OrtValue.ortvalue_from_numpy(
            np.asarray([full_obs], dtype="float32")
        )
        output = session.run(None, {input_name: obs_array.numpy()})[0][0]
        lower_action = [float(value) for value in output.tolist()]
        if len(lower_action) != GR00T_NUM_ACTIONS:
            raise ValueError(
                f"Expected {GR00T_NUM_ACTIONS} lower-body actions, got {len(lower_action)}"
            )
        self._last_lower_body_action = lower_action
        self._debug_snapshot["lower_action_head"] = [round(value, 4) for value in lower_action[:6]]
        return [
            lower_action[idx] * GR00T_ACTION_SCALE + GR00T_LOWER_BODY_DEFAULT_15[idx]
            for idx in range(GR00T_NUM_ACTIONS)
        ]

    def compute_body_target(self, obs: dict[str, Any]) -> list[float]:
        """Combine lower-body ONNX output with a conservative upper-body hold target."""
        joint_state = self.extract_joint_state(obs)
        body_target = list(DEFAULT_UPPER_BODY_TARGET_29)
        lower_body_target = self._run_lower_body_policy(joint_state)
        for local_idx, atec_idx in enumerate(LOWER_BODY_ATEC_INDICES):
            current_q = joint_state["q_gr00t_29"][atec_idx]
            target_q = lower_body_target[local_idx]
            body_target[atec_idx] = current_q + self.lower_body_blend * (target_q - current_q)
        self._debug_snapshot["q_target_gr00t_head"] = [
            round(value, 4) for value in body_target[:6]
        ]
        return body_target

    def map_body_target_to_atec_action(self, body_target_29: list[float]) -> list[float]:
        if len(body_target_29) != ATEC_BODY_DIM:
            raise ValueError(
                f"Expected {ATEC_BODY_DIM} body target values, got {len(body_target_29)}"
            )

        action_33 = [0.0] * ATEC_ACTION_DIM
        for gr00t_idx, atec_idx in enumerate(ATEC_TO_GR00T_BODY_INDICES):
            delta = body_target_29[gr00t_idx] - ATEC_G1_DEFAULT_33[atec_idx]
            action_33[atec_idx] = delta / ATEC_ENV_ACTION_SCALE

        # Fingers stay at the default offset for the minimal submission version.
        for idx in range(ATEC_BODY_DIM, ATEC_ACTION_DIM):
            action_33[idx] = 0.0
        return action_33

    def policy_inference(self, obs: dict[str, Any]) -> list[list[float]]:
        body_target_29 = self.compute_body_target(obs)
        action_33 = self.map_body_target_to_atec_action(body_target_29)
        self._debug_snapshot["action_atec_head"] = [round(value, 4) for value in action_33[:8]]
        return [action_33]

    def get_debug_snapshot(self) -> dict[str, Any]:
        return dict(self._debug_snapshot)
