"""Keyboard teleop for ATEC G1 navigation and base-height control.

Key bindings:
    W / S   Forward / Backward   (vx, hold to accelerate)
    A / D   Left / Right         (vy, hold to accelerate)
    Q / E   Yaw left / right     (vyaw, hold to accelerate)
    Z / X   Height up / down     (step ±0.02 per press)
    R       Reset all commands

Velocities decay to 0 on key release. Height persists.
"""

from __future__ import annotations

import weakref

import numpy as np

try:
    import carb
    import omni
except ImportError:
    carb = None
    omni = None


# Velocity limits (generous; each policy clamps further via its own limits)
VX_MIN, VX_MAX = -1.5, 1.5
VY_MIN, VY_MAX = -1.0, 1.0
VYAW_MIN, VYAW_MAX = -1.0, 1.0
HEIGHT_MIN, HEIGHT_MAX = 0.60, 0.85
HEIGHT_DEFAULT = 0.74
HEIGHT_STEP = 0.02

# Per-step acceleration / decay
ACCEL_RATE = 0.01
DECAY_RATE = 0.92

# Movement keys that use press counting
_MOVEMENT_KEYS = {"W", "S", "A", "D", "Q", "E"}

# Map key to (axis_index, sign) for acceleration
_KEY_AXIS: dict[str, tuple[int, float]] = {
    "W": (0, 1.0),
    "S": (0, -1.0),
    "A": (1, 1.0),
    "D": (1, -1.0),
    "Q": (2, 1.0),
    "E": (2, -1.0),
}


class KeyboardTeleop:
    def __init__(self) -> None:
        if carb is None or omni is None:
            raise RuntimeError("KeyboardTeleop requires carb and omni (Isaac Sim runtime)")

        self._nav = np.zeros(3, dtype=np.float64)
        self._height = HEIGHT_DEFAULT
        self._pending_switch: str | None = None

        # Net press count per key: increment on PRESS, decrement on RELEASE.
        # While physically held, count stays >= 1 despite key-repeat artifacts.
        self._press_count: dict[str, int] = {k: 0 for k in _MOVEMENT_KEYS}

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
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

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        key_name = getattr(event.input, "name", event.input)

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key_name in _MOVEMENT_KEYS:
                self._press_count[key_name] = self._press_count.get(key_name, 0) + 1
            elif key_name == "Z":
                self._height = min(HEIGHT_MAX, self._height + HEIGHT_STEP)
            elif key_name == "X":
                self._height = max(HEIGHT_MIN, self._height - HEIGHT_STEP)
            elif key_name == "R":
                self._nav.fill(0.0)
                self._height = HEIGHT_DEFAULT
                for k in _MOVEMENT_KEYS:
                    self._press_count[k] = 0
            elif key_name == "B":
                self._pending_switch = "blind"
            elif key_name == "H":
                self._pending_switch = "homie"

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if key_name in _MOVEMENT_KEYS:
                self._press_count[key_name] = 0

        return True

    def advance(self) -> tuple[list[float], float]:
        vx, vy, vyaw = self._nav

        # Accelerate while held
        if self._is_held("W"):
            vx += ACCEL_RATE
        if self._is_held("S"):
            vx -= ACCEL_RATE
        if self._is_held("A"):
            vy += ACCEL_RATE
        if self._is_held("D"):
            vy -= ACCEL_RATE
        if self._is_held("Q"):
            vyaw += ACCEL_RATE
        if self._is_held("E"):
            vyaw -= ACCEL_RATE

        # Clamp
        vx = np.clip(vx, VX_MIN, VX_MAX)
        vy = np.clip(vy, VY_MIN, VY_MAX)
        vyaw = np.clip(vyaw, VYAW_MIN, VYAW_MAX)

        # Decay only axes where no key is held
        if not self._axis_held(0, "W", "S"):
            vx *= DECAY_RATE
            if abs(vx) < 1e-4:
                vx = 0.0
        if not self._axis_held(1, "A", "D"):
            vy *= DECAY_RATE
            if abs(vy) < 1e-4:
                vy = 0.0
        if not self._axis_held(2, "Q", "E"):
            vyaw *= DECAY_RATE
            if abs(vyaw) < 1e-4:
                vyaw = 0.0

        self._nav[0] = vx
        self._nav[1] = vy
        self._nav[2] = vyaw

        switch = self._pending_switch
        self._pending_switch = None

        return [float(vx), float(vy), float(vyaw)], float(self._height), switch
