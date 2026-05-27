# Keyboard Teleop Mode

## Launch

```bash
PYTHONPATH=. python scripts/play_atec_task.py --task ATEC-TaskD-G1 --enable_cameras --keyboard --debug
```

`--keyboard` enables real-time keyboard control. Without it, the robot uses the default fixed navigation command.

## Key Bindings

| Key | Action | Range | Behavior |
|-----|--------|-------|----------|
| W | Forward (vx +) | [-0.2, 0.5] | Hold to accelerate, release to decay |
| S | Backward (vx -) | [-0.2, 0.5] | Hold to accelerate, release to decay |
| A | Left (vy -) | [-0.3, 0.3] | Hold to accelerate, release to decay |
| D | Right (vy +) | [-0.3, 0.3] | Hold to accelerate, release to decay |
| Q | Yaw left (vyaw -) | [-0.5, 0.5] | Hold to accelerate, release to decay |
| E | Yaw right (vyaw +) | [-0.5, 0.5] | Hold to accelerate, release to decay |
| Z | Height up | [0.60, 0.85] | Step +0.02 per press, persists |
| X | Height down | [0.60, 0.85] | Step -0.02 per press, persists |
| R | Reset all | vx=vy=vyaw=0, height=0.74 | Instant |

## Behavior Details

- **Velocity axes** (WASD / QE): holding a key continuously increases the command at 0.01/step until the limit is reached; releasing the key decays the command exponentially toward 0 (factor 0.92/step).
- **Height** (Z / X): each press adjusts by 0.02. Height does not decay — it stays where you set it.
- **Reset** (R): instantly zeros all velocity commands and restores height to 0.74.
- When keyboard mode is active, the automatic navigation ramp-up is disabled — you have full control.
