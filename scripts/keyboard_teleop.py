"""Keyboard teleop for ATEC G1 navigation and base-height control.

Key bindings:
    W / S   Forward / Backward   (vx)
    A / D   Left / Right         (vy)
    Q / E   Yaw left / right     (vyaw)
    Z / X   Height up / down     (step ±0.02 per press)
    R       Reset all commands

Movement keys are read from both Kit keyboard events and the per-frame
keyboard state. This keeps teleop responsive when event delivery is spotty.
Velocity commands track fixed targets while held and decay to 0 on release.
Height persists.
"""

from __future__ import annotations

import weakref
import time

import numpy as np

try:
    import carb
    import omni
except ImportError:
    carb = None
    omni = None


# Velocity limits. Policies may clamp further internally.
VX_MIN, VX_MAX = -0.6, 0.8
VY_MIN, VY_MAX = -0.5, 0.5
VYAW_MIN, VYAW_MAX = -1.0, 1.0
HEIGHT_MIN, HEIGHT_MAX = 0.60, 0.85
HEIGHT_DEFAULT = 0.74
HEIGHT_STEP = 0.02

# Fixed command targets while keys are held.
VX_FORWARD_CMD = 0.75
VX_BACKWARD_CMD = 0.35
VY_CMD = 0.35
VYAW_CMD = 0.75

# Per-step smoothing / decay. These are intentionally quick enough to make
# teleop visible in Isaac Sim while avoiding hard command discontinuities.
RESPONSE_RATE = 0.35
DECAY_RATE = 0.75
ZERO_EPS = 1e-3

# Movement keys that use press counting
_MOVEMENT_KEYS = {"W", "S", "A", "D", "Q", "E"}


class KeyboardTeleop:
    def __init__(self) -> None:
        if carb is None or omni is None:
            raise RuntimeError("KeyboardTeleop requires carb and omni (Isaac Sim runtime)")

        self._nav = np.zeros(3, dtype=np.float64)
        self._height = HEIGHT_DEFAULT
        self._pending_switch: str | None = None
        self._last_report = (0.0, 0.0, 0.0, HEIGHT_DEFAULT)
        self._last_report_time = 0.0

        # Held state per key. Kit events maintain this state, while per-frame
        # polling keeps it correct if an event edge is missed.
        self._press_count: dict[str, int] = {k: 0 for k in _MOVEMENT_KEYS}
        self._polled_down: dict[str, bool] = {k: False for k in _MOVEMENT_KEYS}

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_inputs = self._resolve_keyboard_inputs()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        print("[KeyboardTeleop] Ready. WASD=move, QE=yaw, ZX=height, R=reset, B=blind, H=homie")

    def close(self) -> None:
        if self._keyboard_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def _is_held(self, key: str) -> bool:
        return self._press_count.get(key, 0) > 0

    def _axis_held(self, axis: int, positive_key: str, negative_key: str) -> bool:
        return self._is_held(positive_key) or self._is_held(negative_key)

    def _normalize_key_name(self, raw_key) -> str:
        key_name = getattr(raw_key, "name", raw_key)
        key_name = str(key_name).upper()
        if "." in key_name:
            key_name = key_name.rsplit(".", 1)[-1]
        if key_name.startswith("KEY_"):
            key_name = key_name[4:]
        return key_name

    def _normalize_event_type(self, raw_type) -> str:
        type_name = getattr(raw_type, "name", raw_type)
        type_name = str(type_name).upper()
        if "." in type_name:
            type_name = type_name.rsplit(".", 1)[-1]
        return type_name

    def _resolve_keyboard_inputs(self) -> dict[str, object | None]:
        keyboard_input = getattr(carb.input, "KeyboardInput", None)
        if keyboard_input is None:
            return {key: None for key in _MOVEMENT_KEYS}

        resolved: dict[str, object | None] = {}
        for key in _MOVEMENT_KEYS:
            resolved[key] = getattr(keyboard_input, key, None)
            if resolved[key] is None:
                resolved[key] = getattr(keyboard_input, f"KEY_{key}", None)
        return resolved

    def _read_key_down(self, key: str) -> bool | None:
        key_input = self._keyboard_inputs.get(key)
        if key_input is None:
            return None

        get_keyboard_value = getattr(self._input, "get_keyboard_value", None)
        if get_keyboard_value is not None:
            try:
                return bool(get_keyboard_value(self._keyboard, key_input))
            except TypeError:
                try:
                    return bool(get_keyboard_value(None, key_input))
                except Exception:
                    pass
            except Exception:
                pass

        get_keyboard_button_flags = getattr(self._input, "get_keyboard_button_flags", None)
        button_down = getattr(carb.input, "BUTTON_FLAG_DOWN", None)
        if get_keyboard_button_flags is None or button_down is None:
            return None
        for keyboard in (self._keyboard, None):
            try:
                return bool(get_keyboard_button_flags(keyboard, key_input) & button_down)
            except Exception:
                continue
        return None

    def _poll_keyboard_state(self) -> None:
        for key in _MOVEMENT_KEYS:
            is_down = self._read_key_down(key)
            if is_down is None:
                continue
            if is_down:
                self._polled_down[key] = True
                self._press_count[key] = 1
            elif self._polled_down.get(key, False):
                self._polled_down[key] = False
                self._press_count[key] = 0

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        key_name = self._normalize_key_name(event.input)
        event_type = self._normalize_event_type(event.type)

        if event_type in {"KEY_PRESS", "KEY_REPEAT"}:
            if key_name in _MOVEMENT_KEYS:
                should_report = not self._is_held(key_name)
                self._press_count[key_name] = 1
                if should_report:
                    print(f"[KeyboardTeleop] press {key_name}")
            elif key_name == "Z":
                self._height = min(HEIGHT_MAX, self._height + HEIGHT_STEP)
                print(f"[KeyboardTeleop] height={self._height:.2f}")
            elif key_name == "X":
                self._height = max(HEIGHT_MIN, self._height - HEIGHT_STEP)
                print(f"[KeyboardTeleop] height={self._height:.2f}")
            elif key_name == "R":
                self._nav.fill(0.0)
                self._height = HEIGHT_DEFAULT
                for k in _MOVEMENT_KEYS:
                    self._press_count[k] = 0
                print("[KeyboardTeleop] reset")
            elif key_name == "B":
                self._pending_switch = "blind"
                print("[KeyboardTeleop] switch=blind")
            elif key_name == "H":
                self._pending_switch = "homie"
                print("[KeyboardTeleop] switch=homie")

        elif event_type == "KEY_RELEASE":
            if key_name in _MOVEMENT_KEYS:
                self._press_count[key_name] = 0
                self._polled_down[key_name] = False
                print(f"[KeyboardTeleop] release {key_name}")

        return True

    def advance(self) -> tuple[list[float], float, str | None]:
        self._poll_keyboard_state()

        target = np.zeros(3, dtype=np.float64)
        w_held = self._is_held("W")
        s_held = self._is_held("S")
        a_held = self._is_held("A")
        d_held = self._is_held("D")
        q_held = self._is_held("Q")
        e_held = self._is_held("E")

        if w_held and not s_held:
            target[0] += VX_FORWARD_CMD
        elif s_held and not w_held:
            target[0] -= VX_BACKWARD_CMD
        if a_held and not d_held:
            target[1] += VY_CMD
        elif d_held and not a_held:
            target[1] -= VY_CMD
        if q_held and not e_held:
            target[2] += VYAW_CMD
        elif e_held and not q_held:
            target[2] -= VYAW_CMD

        target[0] = np.clip(target[0], VX_MIN, VX_MAX)
        target[1] = np.clip(target[1], VY_MIN, VY_MAX)
        target[2] = np.clip(target[2], VYAW_MIN, VYAW_MAX)

        axis_keys = (("W", "S"), ("A", "D"), ("Q", "E"))
        for axis, (positive_key, negative_key) in enumerate(axis_keys):
            if self._axis_held(axis, positive_key, negative_key):
                self._nav[axis] += (target[axis] - self._nav[axis]) * RESPONSE_RATE
            else:
                self._nav[axis] *= DECAY_RATE
                if abs(self._nav[axis]) < ZERO_EPS:
                    self._nav[axis] = 0.0

        self._nav[0] = np.clip(self._nav[0], VX_MIN, VX_MAX)
        self._nav[1] = np.clip(self._nav[1], VY_MIN, VY_MAX)
        self._nav[2] = np.clip(self._nav[2], VYAW_MIN, VYAW_MAX)

        switch = self._pending_switch
        self._pending_switch = None

        nav_cmd = [float(self._nav[0]), float(self._nav[1]), float(self._nav[2])]
        self._report_if_changed(nav_cmd, float(self._height))

        return nav_cmd, float(self._height), switch

    def _report_if_changed(self, nav_cmd: list[float], height: float) -> None:
        now = time.time()
        current = (
            round(nav_cmd[0], 3),
            round(nav_cmd[1], 3),
            round(nav_cmd[2], 3),
            round(height, 3),
        )
        if current == self._last_report or now - self._last_report_time < 0.25:
            return
        self._last_report = current
        self._last_report_time = now
        print(
            "[KeyboardTeleop] "
            f"nav=[{nav_cmd[0]:+.3f}, {nav_cmd[1]:+.3f}, {nav_cmd[2]:+.3f}], "
            f"height={height:.2f}"
        )
