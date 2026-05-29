# Keyboard Teleop Mode

## Launch

```bash
PYTHONPATH=. python scripts/play_atec_task.py --task ATEC-TaskD-G1 --enable_cameras --keyboard --debug
```

`--keyboard` enables real-time keyboard control. Without it, the robot uses the default fixed navigation command.

## Key Bindings

| Key | Action | Range | Behavior |
|-----|--------|-------|----------|
| W | Forward (vx +) | target +0.60 | Hold for target command, release to decay |
| S | Backward (vx -) | target -0.35 | Hold for target command, release to decay |
| A | Left (vy +) | target +0.35 | Hold for target command, release to decay |
| D | Right (vy -) | target -0.35 | Hold for target command, release to decay |
| Q | Yaw left (vyaw +) | target +0.75 | Hold for target command, release to decay |
| E | Yaw right (vyaw -) | target -0.75 | Hold for target command, release to decay |
| Z | Height up | [0.60, 0.85] | Step +0.02 per press, persists |
| X | Height down | [0.60, 0.85] | Step -0.02 per press, persists |
| R | Reset all | vx=vy=vyaw=0, height=0.74 | Instant |

## Behavior Details

- **Velocity axes** (WASD / QE): movement keys are checked every simulation step with Kit keyboard polling plus event fallback. Holding a key tracks the target command smoothly; releasing the key decays the command toward 0.
- **Height** (Z / X): each press adjusts by 0.02. Height does not decay — it stays where you set it.
- **Reset** (R): instantly zeros all velocity commands and restores height to 0.74.
- When keyboard mode is active, the automatic navigation ramp-up is disabled — you have full control.
