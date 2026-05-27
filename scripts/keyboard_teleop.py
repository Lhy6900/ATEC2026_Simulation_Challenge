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


# Velocity limits
VX_MIN, VX_MAX = -0.2, 0.8
VY_MIN, VY_MAX = -0.5, 0.5
VYAW_MIN, VYAW_MAX = -0.5, 0.5
HEIGHT_MIN, HEIGHT_MAX = 0.60, 0.85
HEIGHT_DEFAULT = 0.74
HEIGHT_STEP = 0.02

# Per-step acceleration / decay
ACCEL_RATE = 0.01
DECAY_RATE = 0.92


class KeyboardTeleop:
    def __init__(self) -> None:
        if carb is None or omni is None:
            raise RuntimeError("KeyboardTeleop requires carb and omni (Isaac Sim runtime)")

        self._nav = np.zeros(3, dtype=np.float64)
        self._height = HEIGHT_DEFAULT

        self._held_keys: set[str] = set()

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        print("[KeyboardTeleop] Ready. WASD=move, QE=yaw, ZX=height, R=reset")

    def close(self) -> None:
        if self._keyboard_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        key_name = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._held_keys.add(key_name)
            if key_name == "Z":
                self._height = min(HEIGHT_MAX, self._height + HEIGHT_STEP)
            elif key_name == "X":
                self._height = max(HEIGHT_MIN, self._height - HEIGHT_STEP)
            elif key_name == "R":
                self._nav.fill(0.0)
                self._height = HEIGHT_DEFAULT
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._held_keys.discard(key_name)
        return True

    def advance(self) -> tuple[list[float], float]:
        vx, vy, vyaw = self._nav

        # Accelerate while held
        if "W" in self._held_keys:
            vx += ACCEL_RATE
        if "S" in self._held_keys:
            vx -= ACCEL_RATE
        if "D" in self._held_keys:
            vy -= ACCEL_RATE
        if "A" in self._held_keys:
            vy += ACCEL_RATE
        if "E" in self._held_keys:
            vyaw -= ACCEL_RATE
        if "Q" in self._held_keys:
            vyaw += ACCEL_RATE

        # Clamp
        vx = np.clip(vx, VX_MIN, VX_MAX)
        vy = np.clip(vy, VY_MIN, VY_MAX)
        vyaw = np.clip(vyaw, VYAW_MIN, VYAW_MAX)

        # Decay unheld axes toward 0
        if "W" not in self._held_keys and "S" not in self._held_keys:
            vx *= DECAY_RATE
            if abs(vx) < 1e-4:
                vx = 0.0
        if "A" not in self._held_keys and "D" not in self._held_keys:
            vy *= DECAY_RATE
            if abs(vy) < 1e-4:
                vy = 0.0
        if "Q" not in self._held_keys and "E" not in self._held_keys:
            vyaw *= DECAY_RATE
            if abs(vyaw) < 1e-4:
                vyaw = 0.0

        self._nav[0] = vx
        self._nav[1] = vy
        self._nav[2] = vyaw

        return [float(vx), float(vy), float(vyaw)], float(self._height)
