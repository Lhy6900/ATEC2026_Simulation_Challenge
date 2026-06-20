# CrossPit v1 Handoff Debug Notes

## Context

The TaskD integrated demo performs a box-push stage first, then hard-switches at
about 13.0 s into a CrossPitBox locomotion policy. The standalone CrossPitBox-v1
play command with `model_19999.pt` works well, but the integrated hard switch
fell shortly after the switch.

The original push-only command was treated as the baseline and was not blamed:

```bash
CUDA_VISIBLE_DEVICES=0 \
  TMPDIR=/data/home/zhukq/Workspace/Atec/tmp_isaaclab \
  ATEC_BOXPUSH_PLANNER_CHECKPOINT=boxpush_planner_v2.pt \
  PYTHONPATH=. \
  python scripts/play_atec_task.py --task ATEC-TaskD-G1 --enable_cameras --debug
```

The 13 s handoff state was stable in that baseline. The CrossPit policy was
trained from that state neighborhood.

## 2026-06-19 Superseding Correction

The earlier v1.5 hard-switch conclusion below is now superseded by a later
native-vs-demo observation/action comparison. The core issue was not only the
heightmap or handoff velocity distribution. The demo wrapper was feeding the
blind actor a joint observation vector in the wrong joint order, and it was also
using the wrong TaskD action offset convention.

Final corrected v1.5 observation/action conventions:

```text
checkpoint = cross_pit_box_v1_5_blind_model_19999.pt
target_x = 5.8
forward_command = 0.6644027
action_offset_scale = 1.0
initial_last_action = zero
heading_kp = 0.0
handoff_blend_steps = 0
handoff_max_delta = 0.0
```

Important details:

- TaskD public proprio/action vectors use `UNITREE_G1_29DOF_DEX1_CFG.joint_names`
  order.
- Native CrossPitBox observations resolve the same joint regex into PhysX
  articulation order. Therefore only `joint_pos` and `joint_vel` in the actor
  observation must be reordered before inference.
- Actor actions remain in TaskD public action order, then the CrossPit reset
  action offset is added before sending them to TaskD.
- The native RSL policy's first command observation for this checkpoint was
  `0.6644027`; using that value keeps the integrated wrapper aligned with the
  standalone play path.
- `CrossPitBoxPolicy` uses a checkpoint-aware stabilizer default. The 108-D
  blind v1.5 checkpoint defaults to `stabilize_steps=2`; the 492-D v1 heightmap
  checkpoint defaults to `stabilize_steps=0`. `demo/solution.py` intentionally
  leaves `stabilize_steps` unset so the checkpoint controls this behavior.

With those corrections, the captured TaskD 13 s snapshot successfully reaches
the TaskD `x_reached` condition under the v1.5 blind policy:

```text
handoff snapshot: logs/taskd_capture_play_path_boxpush_13s_lazy.json
verification log: logs/inject13_v15_current_defaults_verify.jsonl
first local x >= 5.8: t ~= 15.24 s
done condition: x_reached=True at t ~= 16.40 s
final local x ~= 7.70
```

One caveat remains: a later full headless reset-to-finish validation of
`play_atec_task.py` fell during the box-push stage at about 9.32 s, before the
CrossPit handoff. This conflicts with the user's GUI baseline and with the
stored 13 s snapshot. Treat that as a separate box-push/headless nondeterminism
or environment validation issue. It is not evidence that the corrected v1.5
CrossPit handoff wrapper is still exploding at 13 s.

## 2026-06-20 Integrated GUI-Like v1.5 Results

The current GUI-like validation path is:

```bash
CUDA_VISIBLE_DEVICES=0 \
  TMPDIR=/data/home/zhukq/Workspace/Atec/tmp_isaaclab \
  ATEC_BOXPUSH_PLANNER_CHECKPOINT=boxpush_planner_v2.pt \
  ATEC_BOXPUSH_WALK_POLICY=gr00t_walk.pt \
  ATEC_BOXPUSH_BALANCE_POLICY=gr00t_balance.pt \
  ATEC_CROSS_PIT_BOX_CHECKPOINT=cross_pit_box_v1_5_blind_model_19999.pt \
  ATEC_CROSS_PIT_HANDOFF_TIME=13.00 \
  ATEC_DEMO_STEP_DT=0.02 \
  PYTHONPATH=. \
  python scripts/play_atec_task.py --task ATEC-TaskD-G1 --enable_cameras --debug
```

Rechecking the box-push-only path with `--enable_cameras` captured the same
stable 13.0 s state as the original baseline:

```text
robot local pose ~= (x=2.4546, y=-0.9182, z=0.7319, yaw=0.2457)
box local pose   ~= (x=4.3634, y=-0.7997, z=-0.6000)
terminated=False
```

This confirms the earlier 9.32 s headless fall was a headless/rendering-path
validation artifact, not a box-push policy regression.

Additional integrated CrossPit findings:

- `handoff_time_s=13.00` is better than `12.96`. With no stabilization it reaches
  about `local_x=6.76` before falling; `12.96` falls earlier around
  `local_x=4.89`.
- A 2-step handoff damping window improves the trajectory substantially:
  `ATEC_CROSS_PIT_STABILIZE_STEPS=2`,
  `ATEC_CROSS_PIT_STABILIZE_VEL_GAIN=0.025`. The best GUI-like run reached about
  `local_x=7.73`, `local_y=-0.43`, then terminated after crossing the official
  TaskD `x_reached` line.
- A 3-step damping window is too strong and stalls near the box/far-lip region.
- A post-crossing `walkout` phase using the GR00T low-level walker did trigger,
  but did not solve the final shortfall. It reached about `local_x=7.59` before
  falling, still short of the official TaskD success line.

Important target-line clarification:

TaskD's official `x_reached` termination uses world `robot_x > 3.5`. In the
CrossPit local frame with `env_origin_x ~= -4.2`, that is approximately
`local_x > 7.7`. The CrossPit training `target_x=5.8` therefore means
"success in the CrossPit training curriculum", not full official TaskD success.

Current deploy defaults are set to the strongest validated integrated baseline,
not a guaranteed full-task success:

```text
handoff_time_s = 13.00
forward_command = 0.6644027
action_offset_scale = 1.0
checkpoint = cross_pit_box_v1_5_blind_model_19999.pt
stabilize_steps = 2  # selected by CrossPitBoxPolicy for the blind 108-D actor
stabilize_vel_gain = 0.025
walkout_enable = 0
```

This is the first integrated GUI-like run in this debug pass that crossed the
official TaskD success line. It still terminates with a low body height right
after crossing, so the margin is not large, but it satisfies the measured
`x_reached` condition.

Exact validation artifact:

```text
trace: logs/current_gui_like_cross_stab2_gain025_trace.jsonl
final post-step frame:
  elapsed_time_s ~= 14.92
  world robot x ~= 3.5257
  local robot x ~= 7.7257
  robot z ~= 0.3072
  terminated=True
  truncated=False
TaskD x_reached threshold: world robot x > 3.5
TaskD fall threshold: robot z < 0.25
```

Because the final frame is above both the official x threshold and the fall
height threshold, this run is classified as `x_reached`, not a fall.

Fresh recheck with named termination terms after the play trace/capture hook was
extended:

```text
trace: logs/play_gui_v15_trace_done_terms.jsonl
final capture: logs/play_gui_v15_final_done_terms.json
elapsed_time_s = 14.920000076293945
stage = cross_pit_box
actor_obs_dim = 108
uses_heightmap = False
stabilize_steps = 2
final local robot pose ~= (x=7.7257, y=-0.4273, z=0.3072)
done_terms = {"time_out": false, "fall": false, "x_reached": true}
```

This resolves the immediate ambiguity from a plain `terminated=True`: the tested
GUI entrypoint run terminated because TaskD `x_reached` fired, not because the
robot fell. The body is low at the crossing line, so visually it can look close
to failure, but the measured termination term is success.

The same v1.5 deploy setting was also validated from a direct 13 s TaskD
snapshot injection, which removes box-push rollout nondeterminism from the test:

```text
checkpoint = cross_pit_box_v1_5_blind_model_19999.pt
handoff snapshot = logs/taskd_handoff_capture/counter650_gui_equivalent.json
trace = logs/current_debug_v15_injected_counter650_offset1_stab2_truepose.jsonl
action_offset_scale = 1.0
stabilize_steps = 2
stabilize_vel_gain = 0.025
final elapsed_time_s ~= 14.92
final local robot pose ~= (x=7.716, y=-0.745, z=0.630)
done terms: x_reached=True, fall=False
```

This direct-snapshot test and the full GUI-like rollout both support the same
deploy conclusion: the v1.5 blind policy's observation/action handoff is aligned
well enough to cross the official TaskD x threshold under the current wrapper.

Minimum strategy assets for this path:

```text
demo/solution.py
demo/boxpush_planner_solution.py
demo/boxpush_low_level.py
demo/planner_inference_policy.py
demo/cross_pit_box_policy.py
demo/boxpush_planner_v2.pt
demo/boxpush_planner_10hz.pt
demo/gr00t_walk.pt
demo/gr00t_balance.pt
demo/cross_pit_box_v1_5_blind_model_19999.pt
```

`boxpush_planner_10hz.pt` is included as a compatibility fallback for older
server-side asset names; the current code explicitly requests
`boxpush_planner_v2.pt`.

For a complete cloud submission package, include the server entry files as
well:

```text
demo/server.py
demo/run.sh
demo/requirements.txt
```

Read-only v1.5 audit summary:

- The deployed blind checkpoint is a 108-D actor:
  `base_ang_vel(3) + projected_gravity(3) + velocity_commands(3) +
  joint_pos(33) + joint_vel(33) + last_action(33)`.
- This 108-D layout matches the v1/v2 CrossPit observation prefix with the
  terminal heightmap removed. It should not be confused with the current
  experimental `env_cfg.py` layout, which contains extra fields for newer
  training experiments.
- The TaskD handoff wrapper reorders joint position and velocity into native
  CrossPit observation order, scales joint velocity by `0.1`, subtracts the
  CrossPit reset joint position from the observed joint position, initializes
  the first actor-space `last_action` to zero, and converts actor raw output
  back to TaskD action space by adding the CrossPit reset action offset.
- The 2-step stabilizer is an empirical handoff damping layer. It is supported
  by the integrated GUI-like trace, but it also means the first two CrossPit
  actions are not direct actor outputs.
- Remaining deploy risk is margin, not alignment: the run crosses the official
  x line, but the body height is low and the policy is not yet a clean stable
  stand-up-after-crossing behavior.

## Main Evidence

The important separation test was done in the native CrossPitBox-v1 environment
from the captured TaskD 13 s snapshot:

- Native RSL-RL policy with the v1 `model_19999.pt` crossed successfully.
- The demo `CrossPitBoxPolicy` wrapper also crossed successfully when fed the
  native training `depth_scanner.data.ray_hits_w`.
- The same demo wrapper failed when it had to rebuild the 384-D heightmap from
  its own `terrain_map_heightmap()` approximation.

This narrowed the issue to observation reconstruction, not:

- box-push stability,
- the 13 s robot state,
- checkpoint quality,
- action dimension/order,
- or raw action offset semantics.

## Root Cause Found So Far

The v1 actor uses a 384-D depth heightmap. In training, that heightmap is
generated by:

- `MultiMeshRayCasterCfg(prim_path="{ENV_REGEX_NS}/Robot/d435_link")`
- `ray_alignment="yaw"`
- `offset=(1.3, 0.0, 2.5)`
- then `ray_hits_to_heightmap(..., base_pos_w, base_quat_w)`.

The demo approximation treated the scan as if it were emitted from the base/root
pose. That is not equivalent. The training scanner follows `d435_link`, and
`d435_link` moves relative to the articulation root as the waist/torso moves.

Measured examples from native rollout:

- at the handoff frame, `d435_link` relative to root was approximately
  `(0.1885, 0.0052, 0.4331)`.
- one policy step later it was approximately `(0.1614, 0.1066, 0.4309)`.
- a few steps later it moved toward `(0.0904, 0.1650, 0.4337)`.

So a base-root scan or a single fixed offset causes terrain and box edges to be
seen at the wrong position by the actor. The standalone training/play path never
has this mismatch because it uses the real native raycaster.

## Debug Changes Added

`scripts/rsl_rl/debug_cross_pit_native_headless.py` was extended with:

- `--policy_source demo` to run the demo wrapper inside the native CrossPit env.
- `--demo_checkpoint` to select the checkpoint for that wrapper.
- `--demo_true_raycast` to explicitly feed native ray hits to the wrapper.
- `--log_pre_step` to log pre-step policy observations/actions.
- sensor/body logging for `depth_scanner`, `pelvis`, `torso_link`, and
  `d435_link`.

`demo/cross_pit_box_policy.py` also has debug support for
`_debug_depth_ray_hits_w`, so native ray hits can be converted by the demo code
for controlled comparison.

## Current Decision

The v1 heightmap-alignment work is parked for now. The likely fix would be to
rebuild the demo heightmap from a closer approximation of the training
`d435_link` raycaster, or from real camera/depth data if the official interface
provides enough camera pose/intrinsic information.

The next test path is CrossPitBox-v1.5 blind walking. Since v1.5 removes
`depth_heightmap` from the actor observation, it should avoid the heightmap
coordinate mismatch at hard switch time.

Read-only v1 audit summary:

- The v1 492-D actor observation ordering is correct in the demo wrapper:
  `base_ang_vel(3) + projected_gravity(3) + velocity_commands(3) +
  joint_pos_rel(33) + joint_vel*0.1(33) + last_action(33) +
  depth_heightmap(384)`.
- Joint position/velocity reorder and the TaskD action-origin conversion are
  implemented in the wrapper. Full TaskD deployment should keep
  `action_offset_scale=1.0`; only native debug comparisons should use `0.0`.
- The unsolved mismatch is specifically the last 384 heightmap dimensions:
  standalone/native v1 uses the real `d435_link` raycaster, while the submitted
  demo can only rebuild an analytical terrain map with an approximate fixed
  sensor offset.
- Therefore v1 should not be the deploy path unless TaskD exposes an equivalent
  real raycast/depth reconstruction path.

Two additional v1 observation-reconstruction hypotheses were tested and rejected:

- `ATEC_CROSS_PIT_HEIGHTMAP_SOURCE=extero`, using the official TaskD 16x360
  torso LiDAR compressed into 24x16, terminated at about `13.72 s` with local
  pose `(x ~= 3.90, z ~= 0.242)`. This is a fall, not a crossing, and is worse
  than the v1.5 blind deploy path.
- A one-off diagnostic replaced the v1 heightmap with `head_depth` projected
  through the live `head_camera` pose/intrinsics from the scene. The G1
  `head_camera` is attached to `d435_link`, but the current projection produced
  an all-default heightmap (`mean=min=max=-1.0`) during handoff, so no valid
  points landed in the trained forward 24x16 grid. This path needs a separate
  camera-coordinate/depth-semantics fix before it can be considered deployable.

Because both official-observation alternatives failed, no v1 heightmap-source
change was promoted into the submission defaults.

## 2026-06-20 v1 Retest

The current wrapper defaults were rechecked against the v1 heightmap checkpoint:

```text
checkpoint = cross_pit_box_model_19999.pt
heightmap_source = terrain_map
target_x = 5.8
forward_command = 0.6644027
action_offset_scale = 1.0
handoff_time_s = 13.00
stabilize_steps = 2
stabilize_vel_gain = 0.025
```

The full GUI-like TaskD hard-switch rollout still failed:

```text
trace: logs/current_gui_like_v1_terrain_stab2_offset1_trace.jsonl
final: logs/current_gui_like_v1_terrain_stab2_offset1_final.json
final elapsed_time_s ~= 13.96
final local robot pose ~= (x=4.266, y=-0.008, z=0.241, yaw=1.935)
final box pose ~= (x=4.309, y=-0.793, z=-0.556)
heightmap mean/min/max ~= (-0.947, -1.000, -0.451)
base_ang_vel_actor_norm ~= 23.05
```

This is a fall/instability shortly after the hard switch, not an official
TaskD `x_reached` crossing.

A second isolated test injected the captured `counter650_gui_equivalent` 13 s
handoff snapshot directly into TaskD, attached the live scene root pose to the
wrapper so the analytical heightmap did not depend on velocity integration, and
then ran the same v1 policy:

```text
trace: logs/current_debug_v1_injected_counter650_truepose.jsonl
final elapsed_time_s ~= 13.82
final local robot x ~= 3.70
final robot z ~= 0.246
done terms: fall=True, x_reached=False
heightmap mean range ~= [-0.925, -0.395]
heightmap max range ~= [-0.726, 0.305]
```

Therefore the latest v1 failure is not explained only by push-stage
nondeterminism or by the wrapper's dead-reckoned local pose. Even when the
handoff snapshot and live root pose are controlled, v1 still falls before the
near lip.

Comparing the first few native CrossPit frames from
`logs/native_exact_counter650_obs_actions.jsonl` against the TaskD injected run
shows a narrower failure mode:

- The first v1 actor action is close to the native action, so the checkpoint
  load, 492-D observation prefix, joint order, and action offset convention are
  not the first-order error.
- Within two to four policy steps the TaskD rollout has a much weaker/different
  physical response than native CrossPit; root velocity, angular velocity,
  heightmap range, and subsequent actor outputs diverge.
- The v1 heightmap actor is very sensitive to that early divergence. Once the
  terrain-map observation differs near the near lip, it emits large corrective
  actions and the robot falls.

This makes v1 a poor deploy candidate in the current hard-switch wrapper unless
we either reconstruct the training raycast nearly exactly from TaskD data or
retrain/fine-tune v1 directly in the TaskD handoff dynamics.

An action-offset ablation was also tested because native RSL play forwards actor
outputs directly to the action term:

```text
offset=1.0, stabilizer=2:
  trace: logs/current_debug_v1_injected_counter650_truepose.jsonl
  final t ~= 13.82, local x ~= 3.74, z ~= 0.246, fall=True

offset=0.0, stabilizer=0:
  trace: logs/current_debug_v1_injected_counter650_offset0_truepose.jsonl
  final t ~= 13.46, local x ~= 3.05, z ~= 0.238, fall=True

offset=0.0, stabilizer=2:
  trace: logs/current_debug_v1_injected_counter650_offset0_stab2_truepose.jsonl
  final t ~= 13.92, local x ~= 3.65, z ~= 0.223, fall=True
```

Disabling the reset-action offset does not solve v1. The best of these v1
hard-switch variants is still the current `offset=1.0, stabilizer=2` setting,
and it still fails before the gap crossing.

## CrossPitBox-v1.5 Blind Handoff Tests

The v1.5 blind checkpoint tested here is:

```text
demo/cross_pit_box_v1_5_blind_model_19999.pt
```

The standalone training result was much weaker than the v1 heightmap policy but
still usable in its own environment: final `crossed_rate` was about 0.70 at
`target_x ~= 5.11`.

Native CrossPit replay from the captured TaskD 13 s snapshot succeeded when the
demo wrapper was run inside the native CrossPitBox environment. In that native
test the policy advanced through the near lip and eventually reached about
`x = 5.60`, so the checkpoint and wrapper can work when the downstream
environment matches the training task.

Integrated TaskD hard-switch tests did not yet cross:

- v1.5 blind, raw policy action, no heading correction:
  - final local pose was about `(x=3.85, y=-1.17, z=0.24)`;
  - the robot fell near the near lip / pit entry.
- v1.5 blind, raw policy action, no heading correction, actor velocity
  observation warmup plus clipping:
  - final local pose was about `(x=4.12, y=-1.47, z=0.19)`;
  - the robot moved farther forward but side drift became worse and it fell.

This means v1.5 did remove the heightmap reconstruction failure mode, but the
hard switch is still not equivalent to native CrossPit training/play. The most
visible remaining mismatch is the post-handoff dynamics: at 13 s the push stage
hands over a robot with large joint and angular velocities. The blind policy was
trained around the same pose neighborhood, but not around this exact transient
velocity distribution inside the TaskD environment. After switching, TaskD
diverges from native CrossPit within the first few policy steps, with increasing
lateral drift and lower base height before the robot reaches the box.

Additional observation-action experiments from this earlier pass:

- Adding a TaskD reset-action offset to the policy output looked worse in this
  early integrated test, but this was later shown to be confounded by the joint
  observation order mismatch. The corrected wrapper uses
  `ATEC_CROSS_PIT_ACTION_OFFSET_SCALE=1.0`.
- Heading correction also did not fix the failure. The better empirical setting
  so far is no explicit heading correction
  (`ATEC_CROSS_PIT_HEADING_KP=0.0`).
- Clipping/warming up actor velocity observations reduced the first-step
  observation shock, but it did not solve the downstream lateral drift.

Current root-cause assessment:

The original push-box planner is still considered healthy. The remaining issue
is the handoff distribution mismatch between the TaskD integrated environment
and the CrossPit training environment. A robust solution probably needs either:

- a continuation policy trained directly from recorded TaskD handoff states,
  including nonzero base/joint velocities and the real TaskD box/ground setup;
  or
- a more deliberate handoff controller that first damps the high joint/base
  velocities without losing the crossing setup, then enters the CrossPit policy.

## 2026-06-19 Follow-Up Results

Two explorer agents checked the two policy families independently:

- v1 heightmap policy: the dominant issue remains heightmap reconstruction.
  The native policy succeeds, and the demo wrapper succeeds in native CrossPit
  when fed true native ray hits. The TaskD wrapper's analytical terrain map is
  still only an approximation of the `d435_link` raycaster.
- v1.5 blind policy: the 108-D actor observation order is correct. Empirically
  the best action semantics are still raw actor actions
  (`action_offset_scale=0.0`) with joint-position rebase enabled and no heading
  correction.

Important negative tests:

- Removing the joint-position rebase was worse. The actor then saw
  `joint_pos_actor_norm ~= 1.0` at handoff and the integrated rollout drifted to
  about `(x=3.94, y=-2.77, z=0.23)` before falling.
- A short stabilizing window plus blend/rate limiting reduced the immediate
  motion, but it made the policy miss the crossing rhythm and fall earlier
  around `x=3.22`.
- Heading correction helped keep lateral position closer to the centerline, but
  it strongly reduced forward progress. `heading_kp=0.5` fell around `x=3.08`;
  `heading_kp=0.15` fell around `x=3.47`.
- Moving handoff earlier to `12.92 s` was worse than `12.96 s`.

Best integrated v1.5 blind setting seen during this earlier, now-superseded
pass:

```text
handoff_time_s = 12.96
action_offset_scale = 0.0
heading_kp = 0.0
stabilize_steps = 0
handoff_blend_steps = 0
handoff_max_delta = 0.0
```

This is no longer the recommended deploy setting. It was the best setting before
the observation-order mismatch was identified.

v1 heightmap wrapper work:

- `terrain_map_heightmap()` now accepts a `d435_link_offset_b` parameter and the
  demo wrapper uses an approximate default d435 offset. This makes the analytical
  heightmap more faithful to the training sensor origin.
- The integrated v1 heightmap run still failed early, around `x=3.05`, because
  the analytical heightmap near the gap still diverges from true native raycast
  behavior. This path should not be treated as solved without a real TaskD
  raycast/depth reconstruction.

Additional single-variable tests:

- Matching the native v1.5 velocity command exactly
  (`forward_command=0.6644`) was not better in the integrated TaskD handoff.
  It reduced lateral drift somewhat but only reached about
  `(x=3.92, y=-1.42, z=0.24)`. The stronger integrated baseline remains
  `forward_command=0.73`, which reached about
  `(x=4.49, y=-2.03, z=0.22)`.
- A small heading correction (`heading_kp=0.05`, clipped at `0.2`) also did not
  cross. It reached about `(x=4.45, y=-1.82, z=0.24)`, slightly behind the
  no-heading baseline. Larger heading corrections were already worse.
- A pure box-push trace from `9.0 s` to `14.5 s` showed that `12.96-13.00 s`
  is a high-velocity phase in TaskD. Around `13.00 s`, the observed
  `joint_vel_norm` was about `68`, while around `12.50 s` it was about `2.4`.
- Switching earlier at `12.50 s` did not fix the problem; it fell much earlier,
  around `(x=2.99, y=-1.68, z=0.24)`. The lower-velocity phase has worse pose
  and phase alignment for the trained v1.5 policy.

Earlier conclusion, now superseded:

At that time the hard-switch failure did not look like a simple default-parameter
issue. The best deploy default then was:

```text
handoff_time_s = 12.96
forward_command = 0.73
action_offset_scale = 0.0
heading_kp = 0.0
stabilize_steps = 0
handoff_blend_steps = 0
handoff_max_delta = 0.0
```

The later corrected wrapper invalidates this as the primary conclusion. Training
from richer TaskD handoff distributions may still improve robustness, but the
first-order integration bug was the actor observation/action convention mismatch.

## 2026-06-20 v1 Heightmap Follow-Up

Two additional read-only audits split the remaining v1 problem into heightmap
and action-space paths:

- The action-space audit found no remaining evidence for a joint order, action
  scale, reset-offset, or last-action convention bug. v1 and v1.5 share the same
  action wrapper. v1.5 succeeds under that wrapper, and v1 first-step raw actor
  actions are close to native v1 actions.
- The heightmap audit found the remaining v1-specific risk in the 384-D
  heightmap. The training policy consumes true `d435_link` `MultiMeshRayCaster`
  hit points over `/World/ground` and `BoxSupport`; the demo wrapper can only
  approximate that from fixed terrain constants or from the official 360-degree
  lidar/depth observations.

New single-variable v1 injection tests from the same counter-650 13 s TaskD
snapshot:

```text
checkpoint = cross_pit_box_model_19999.pt
action_offset_scale = 1.0
heightmap_source = terrain_map
handoff snapshot = logs/taskd_handoff_capture/counter650_gui_equivalent.json
```

Results:

```text
stabilize_steps=2, forward_command=0.6644:
  final local x ~= 3.70, fall=True

stabilize_steps=0, forward_command=0.6644:
  final local x ~= 4.04, fall=True
  trace = logs/current_debug_v1_injected_counter650_offset1_nostab_truepose.jsonl

stabilize_steps=0, forward_command=0.7209:
  final local x ~= 4.53, fall=True
  lateral drift reached |y| ~= 1.30
  trace = logs/current_debug_v1_injected_counter650_offset1_nostab_cmd072_truepose.jsonl
```

Interpretation:

- The v1 policy should not inherit the v1.5 two-step handoff stabilizer by
  default. Removing it improves v1 progress, so `demo/solution.py` now leaves
  `stabilize_steps` unset and `CrossPitBoxPolicy` chooses a checkpoint-aware
  default: 108-D blind actors use 2 steps, while 492-D heightmap actors use 0.
- Increasing the v1 forward command moves farther but causes lateral divergence.
  That is not a clean fix.
- v1 remains unsolved for submission. The best current deploy path is still the
  v1.5 blind checkpoint with the corrected 108-D observation/action wrapper.

Additional 2026-06-20 heightmap finding:

- The apparent `pit_depth` mismatch was not the main source of v1 divergence:
  the actor input clamps heightmap values to `[-1, 1]`, so `-1.2` and `-1.0`
  mostly collapse to the same value.
- A real missing geometry term was the side platform from
  `pit_and_platform_terrain`. Native v1 raycast logs show rays hitting a
  `z ~= 0.927` platform on the positive-y side of the pit. The demo analytical
  `terrain_map_heightmap()` previously modeled only flat ground, the pit bottom,
  and the box support, so it under-reported those cells by roughly 0.7 m in
  later v1 rollout frames.
- `demo/cross_pit_box_policy.py` now includes this side-platform region in the
  analytical terrain map and resets the pit-bottom constant to the training
  value `-1.0`. A regression test was added in
  `demo/test_push_then_cross_solution.py`.
- This reduces one source of v1 observation error, but it still does not make
  v1 a solved submission path. The native raycaster uses live `d435_link`
  sensor yaw and offset; the cloud/demo solution does not receive those internal
  simulator fields, so a 492-D heightmap actor can still receive a shifted
  heightmap after hard switch. The v1.5 blind checkpoint remains the recommended
  deploy policy.

Follow-up GPU0 TaskD injection checks after adding the side platform:

```text
handoff snapshot = logs/taskd_handoff_capture/counter650_gui_equivalent.json
handoff elapsed = 13.0 s
v1 checkpoint = cross_pit_box_model_19999.pt
```

Results:

```text
v1, terrain_map, official observations:
  final t ~= 13.70, fall=True, x_reached=False
  root ~= (0.05, -0.30, 0.22), local_x ~= 4.26
  trace = logs/current_v1_handoff_after_platform_official_obs.jsonl

v1, terrain_map, attached true root pose:
  final t ~= 13.66, fall=True, x_reached=False
  root ~= (-0.66, -0.05, 0.22), local_x ~= 3.50
  trace = logs/current_v1_handoff_after_platform_truepose.jsonl

v1, official extero-compressed heightmap:
  final t ~= 13.82, fall=True, x_reached=False
  root ~= (-0.14, -0.36, 0.21), local_x ~= 4.02
  trace = logs/current_v1_handoff_extero.jsonl

v1, terrain_map, frozen box:
  final t ~= 13.70, fall=True, x_reached=False
  root ~= (0.05, -0.30, 0.22), local_x ~= 4.26
  trace = logs/current_v1_handoff_freezebox.jsonl

v1, terrain_map, two-step stabilizer:
  final t ~= 13.74, fall=True, x_reached=False
  root ~= (-0.25, -0.10, 0.22), local_x ~= 3.96
  trace = logs/current_v1_handoff_platform_stab2.jsonl

v1, terrain_map, velocity-observation warmup over four policy steps:
  final t ~= 13.58, fall=True, x_reached=False
  root ~= (-0.31, -0.08, 0.24), local_x ~= 3.89
  trace = logs/current_v1_handoff_platform_velwarm4.jsonl

v1.5 blind, default deploy path:
  final t ~= 14.92, fall=False, x_reached=True
  root ~= (3.52, -0.75, 0.63)
  actor_obs_dim = 108, stabilize_steps = 2
  trace = logs/current_v15_handoff_recheck.jsonl
```

The key difference at handoff is now clear: native v1 reset begins almost at
rest (`base_ang_vel ~= 0`, `joint_vel ~= 0`) and with a true native raycast
heightmap, while the TaskD 13 s state still contains nonzero angular/joint
velocity and only approximate or official-lidar-derived heightmap information.
The blind v1.5 policy handles this mismatch and crosses; v1 does not. For the
submission path, keep `cross_pit_box_v1_5_blind_model_19999.pt` as the default.

## 2026-06-20 Reverification and v1 Parking Decision

Two independent read-only audits split the remaining v1 issue into observation
and transition paths:

- The 492-D v1 actor observation layout is correct:
  `base_ang_vel(3) + projected_gravity(3) + velocity_commands(3) +
  joint_pos(33) + joint_vel(33) + last_action(33) + heightmap(384)`.
- Joint position/velocity reorder, action offset, action scale, and first
  `last_action=zero` are consistent with native CrossPit semantics.
- The action offset is required: TaskD and CrossPit use the same
  `JointPositionAction(scale=0.5, use_default_offset=True)` form, but their
  default joint poses differ. The demo wrapper therefore maps actor output with
  `a_taskd = a_train + CROSS_PIT_RESET_ACTION`.
- The most visible v1 mismatch is the hard-switch observation distribution:
  native v1 starts almost at rest, while TaskD 13 s handoff has nonzero base
  angular velocity and joint velocity.
- The v1 384-D heightmap is still not equivalent to the native training raycast.
  `terrain_map` is only an approximation; `extero` compression is clearly OOD
  for this actor.

A corrected native-v1 done-term parser showed that standalone native v1 does
reach the native `crossed=True` condition:

```text
trace = logs/current_native_v1_correct_done_recheck.jsonl
pre-step 48 root ~= (x=1.49, y=-0.80, z=0.96)
post-step 49 done terms: crossed=True
```

This does not rescue the TaskD hard-switch path. The latest TaskD v1 injection
tests still fall before the official `x_reached` line, while the v1.5 blind
path succeeds from the same 13 s snapshot.

Fresh default-path verification:

```text
command:
  CUDA_VISIBLE_DEVICES=0 TMPDIR=tmp_isaaclab
  ATEC_CROSS_PIT_BOX_CHECKPOINT=cross_pit_box_v1_5_blind_model_19999.pt
  python scripts/debug_push_then_cross_headless.py
    --task ATEC-TaskD-G1
    --disable_cameras
    --num_envs 1
    --max_steps 260
    --inject_handoff_snapshot logs/taskd_handoff_capture/counter650_gui_equivalent.json
    --inject_elapsed_s 13.0
    --log_path logs/current_v15_default_path_reverify.jsonl

result:
  checkpoint = demo/cross_pit_box_v1_5_blind_model_19999.pt
  actor_obs_dim = 108
  stabilize_steps = 2
  terminal step = 95
  done terms = {time_out=False, fall=False, x_reached=True}
  final root ~= (x=3.516, y=-0.745, z=0.630)
  final local pose ~= (x=7.416, y=-0.014, z=0.697)
```

Current decision:

- Keep the deploy/default submission path on v1.5 blind:
  `cross_pit_box_v1_5_blind_model_19999.pt`.
- Do not promote v1 heightmap changes into the default path.
- If v1 is revisited, test these single-variable changes before changing
  defaults:
  `ATEC_CROSS_PIT_VELOCITY_OBS_WARMUP_STEPS=1`, and a v1-only transition of
  `stabilize_steps=2` plus `handoff_blend_steps=8`.

## 2026-06-20 Native Raycast Isolation

The v1 action/proprio wrapper was isolated against the native CrossPitBox env:

```text
command:
  CUDA_VISIBLE_DEVICES=0 TMPDIR=tmp_isaaclab
  ATEC_CROSS_PIT_ACTION_OFFSET_SCALE=0
  python scripts/rsl_rl/debug_cross_pit_native_headless.py
    --task ATEC-Isaac-TaskD-G1-CrossPitBox-v1
    --num_envs 1
    --max_steps 260
    --policy_source demo
    --demo_checkpoint demo/cross_pit_box_model_19999.pt
    --demo_true_raycast
    --demo_native_order_proprio
    --inject_taskd_snapshot logs/taskd_handoff_capture/counter650_gui_equivalent.json
    --cross_pit_success_x 5.8
    --log_path logs/current_native_v1_demo_wrapper_true_raycast_nativeorder_offset0_fixed.jsonl

result:
  crossed=True
  terminal step ~= 51
```

This proves the demo wrapper's action transform, joint observation reorder, and
TaskD handoff joint-position rebasing are correct when the policy receives the
native training raycast heightmap.

The failing TaskD-v1 hard switch therefore sits at the heightmap boundary. A
camera-enabled TaskD probe confirmed:

- Official TaskD observations contain `image/head_depth` with valid numeric
  data, and Isaac's internal `head_camera.data` exposes `pos_w`,
  `quat_w_world`, and `intrinsic_matrices` while running locally.
- The submitted `solution.py` API receives image tensors, not the camera pose or
  intrinsics. Reconstructing the exact native raycast distribution from the
  official payload is therefore not available in a robust deploy form.
- A local projection probe from `head_depth` gives a plausible but different
  24x16 map, and the injection-style debug run can retain stale camera frames
  immediately after writing a snapshot. It is not strong enough evidence to
  replace the verified v1.5 path.
- TaskD geometry at handoff matches the v1 training constants: box local
  x/y/z ~= `(4.3634, -0.7997, -0.6000)`, box top ~= `-0.20`, near/far pit lips
  are still modeled as `3.72/4.68`.

Final current conclusion: v1 is a valid native CrossPit policy but is not a
safe official TaskD hard-switch policy without a reliable deploy-time equivalent
of the training raycaster. Keep v1.5 blind as the default submission path.

## 2026-06-20 Head-Depth v1 Diagnostic

A diagnostic-only `ATEC_CROSS_PIT_HEIGHTMAP_SOURCE=head_depth` path was added
for local experiments. It projects TaskD `image/head_depth` into the same 24x16
heightmap shape when a debug runner also attaches:

```text
_debug_head_camera_intrinsics
_debug_head_camera_pos_w
_debug_head_camera_quat_w
_debug_root_pos_w
_debug_root_quat_w
```

This path is not a cloud deploy default because the official `solution.py`
payload does not include camera pose/intrinsics.

Single-variable GPU0 test from the captured 13 s handoff snapshot:

```text
log = logs/current_v1_head_depth_debug_source.jsonl
checkpoint = demo/cross_pit_box_model_19999.pt
heightmap_source = head_depth
final elapsed ~= 13.60 s
done terms = {time_out=False, fall=True, x_reached=False}
final local x ~= 3.27
```

The diagnostic proved the wrapper can consume a TaskD head-depth projection, but
it did not rescue v1. The v1 policy still falls before reaching the official
TaskD success line, while the v1.5 blind default has direct `x_reached=True`
evidence. This further supports keeping v1.5 as the deploy path.

To prevent accidental cloud/config drift back to v1, the policy wrapper now
supports:

```text
ATEC_CROSS_PIT_REQUIRE_DEPLOY_DEFAULTS=1
```

When enabled, it fail-fast checks the verified deploy assumptions:

```text
checkpoint = cross_pit_box_v1_5_blind_model_19999.pt
actor_obs_dim = 108
uses_heightmap = False
stabilize_steps = 2
action_offset_scale = 1.0
initial_last_action = zero
handoff_blend_steps = 0
handoff_max_delta = 0.0
```

## 2026-06-20 Dynamic d435 Offset Retest

The fixed analytical `d435_link` offset was too weak as a model of the native
raycaster frame. Native logs show the camera frame moves relative to the root as
the waist changes pose. A lightweight kinematic estimate was added to
`demo/cross_pit_box_policy.py`:

```text
d435_link_offset_b = f(waist_yaw, waist_roll, waist_pitch)
```

The fit was derived from native v1 logs and is intentionally small in scope: it
uses only the waist joints already available in the actor proprioception. Unit
tests lock two measured native poses and also verify that the terrain-map
heightmap changes when the waist pose changes.

Validation:

```text
PYTHONPATH=. pytest -q demo/test_push_then_cross_solution.py
  44 passed

PYTHONPATH=. pytest -q \
  demo/test_push_then_cross_solution.py \
  demo/test_solution_counter.py \
  scripts/rsl_rl/test_capture_taskd_handoff_state.py \
  scripts/rsl_rl/test_play_atec_task_capture_hook.py \
  scripts/rsl_rl/test_play_checkpoint_resolution.py
  62 passed
```

TaskD v1 hard-switch retest after the dynamic offset:

```text
trace = logs/current_v1_dynamic_d435_offset.jsonl
checkpoint = cross_pit_box_model_19999.pt
heightmap_source = terrain_map
actor_obs_dim = 492
stabilize_steps = 0
final elapsed ~= 13.70 s
done terms = {time_out=False, fall=True, x_reached=False}
final local pose ~= (x=3.92, y=-0.20, z=0.25)
heightmap mean/min/max ~= (-0.40, -1.00, 0.23)
```

This is a measurable improvement over the earlier attached-true-pose v1 test,
which fell around `local_x ~= 3.50`, but it is still not a solved TaskD
handoff. The robot reaches the near-lip/gap-entry region and then falls before
the official TaskD `x_reached` condition.

Current interpretation:

- The dynamic camera offset removes one real source of v1 heightmap error.
- The remaining v1 failure is still coupled to TaskD hard-switch dynamics and
  the fact that the deploy wrapper cannot reproduce the native
  `MultiMeshRayCaster` observation exactly.
- v1 remains diagnostic/research only for the integrated TaskD pipeline.
- The verified deploy/default path remains the v1.5 blind checkpoint:
  `cross_pit_box_v1_5_blind_model_19999.pt`.

## 2026-06-20 Deploy Guard Hardening

The immediate integrated-demo risk is no longer that v1.5 blind is misaligned in
the current default path. The stronger risk is accidentally running an old
experiment configuration, for example:

```text
ATEC_CROSS_PIT_BOX_CHECKPOINT=cross_pit_box_model_19999.pt
ATEC_CROSS_PIT_STABILIZE_STEPS=0
ATEC_CROSS_PIT_ACTION_OFFSET_SCALE=0
```

Those values are valid for diagnostics, but they are not the verified deploy
handoff. To prevent silent drift, `demo/solution.py` now passes
`require_deploy_defaults=True` by default when it constructs
`CrossPitBoxPolicy`.

With this guard active, the wrapper fail-fast checks:

```text
checkpoint = cross_pit_box_v1_5_blind_model_19999.pt
actor_obs_dim = 108
uses_heightmap = False
stabilize_steps = 2
action_offset_scale = 1.0
initial_last_action = zero
handoff_blend_steps = 0
handoff_max_delta = 0.0
```

For local v1/v1.5 ablations only, disable the guard explicitly:

```bash
ATEC_CROSS_PIT_REQUIRE_DEPLOY_DEFAULTS=0
```

Fresh validation after enabling the guard:

```text
PYTHONPATH=. pytest -q \
  demo/test_push_then_cross_solution.py \
  demo/test_solution_counter.py \
  scripts/rsl_rl/test_capture_taskd_handoff_state.py \
  scripts/rsl_rl/test_play_atec_task_capture_hook.py \
  scripts/rsl_rl/test_play_checkpoint_resolution.py
  64 passed

trace = logs/current_v15_deploy_guard_gpu0_verify.jsonl
checkpoint = cross_pit_box_v1_5_blind_model_19999.pt
actor_obs_dim = 108
uses_heightmap = False
stabilize_steps = 2
action_offset_scale = 1.0
require_deploy_defaults = True
final elapsed ~= 14.92 s
done terms = {time_out=False, fall=False, x_reached=True}
final root ~= (x=3.516, y=-0.745, z=0.630)
```

The same deploy path was also verified through the real `scripts/play_atec_task.py`
entrypoint, with cameras enabled and without injecting a saved state:

```text
trace = logs/current_v15_play_atec_task_gui_trace.jsonl
final = logs/current_v15_play_atec_task_gui_final.json
first cross-pit frame ~= step 650, elapsed 12.98 s
checkpoint = cross_pit_box_v1_5_blind_model_19999.pt
actor_obs_dim = 108
uses_heightmap = False
stabilize_steps = 2
require_deploy_defaults = True
final elapsed ~= 14.92 s
done terms = {time_out=False, fall=False, x_reached=True}
final root ~= (x=3.526, y=-0.427, z=0.307)
```

Note: `scripts/debug_push_then_cross_headless.py` is not a faithful replacement
for the camera-enabled push stage. A full no-camera/headless run failed in the
box-push stage at about 9.32 s. Use it for injected handoff diagnostics, but use
`scripts/play_atec_task.py --enable_cameras` when validating the integrated
submission behavior.

The v1 investigation still points to the heightmap boundary: v1 native and v1
with true native ray hits are healthy, while TaskD hard-switch v1 using
analytical terrain-map heightmaps is not. If v1 is revisited, the next
meaningful test is a training-equivalent `d435_link` raycaster in the TaskD
debug path. Continuing to tune action offsets or small stabilizer windows before
that observation gap is closed is unlikely to produce a robust fix.

## 2026-06-20 TaskD d435 Raycaster Diagnostic

The debug runner can now attach a local, diagnostic-only raycaster to
`{ENV_REGEX_NS}/Robot/d435_link` and pass its ray hits into the v1 policy as
`_debug_depth_ray_hits_w`. This approximates the CrossPitBox-v1 training
heightmap more closely than the analytical `terrain_map` bridge.

Two single-variable TaskD handoff tests were run from the same counter-650,
13.0 s snapshot:

```text
checkpoint = cross_pit_box_model_19999.pt
heightmap = attached TaskD d435 raycaster
stabilize_steps = 2
target_x = 5.8
```

Default v1 forward command:

```text
trace = logs/current_v1_taskd_debug_depth_scanner_stab2_gpu0.jsonl
forward_command = 0.6644027
final elapsed ~= 14.28 s
done terms = {time_out=False, fall=True, x_reached=False}
final root ~= (x=2.917, y=-0.934, z=0.215)
```

Higher forward command:

```text
trace = logs/current_v1_taskd_debug_depth_scanner_stab2_cmd073_gpu0.jsonl
forward_command = 0.73
final elapsed ~= 14.58 s
done terms = {time_out=False, fall=True, x_reached=False}
final root ~= (x=3.439, y=-0.775, z=0.222)
```

The raycaster and higher command both improve forward progress, but they still
do not solve v1. The robot reaches the late near-gap region and then falls
before the environment's official `x_reached` termination. This test also
exposed a diagnostic-script pitfall: using a bare `root_x > 3.5` threshold can
misclassify an unstable near-crossing as success. The debug runner now returns
success only when the environment reports `x_reached=True`.

Current deploy conclusion remains unchanged:

- Keep `cross_pit_box_v1_5_blind_model_19999.pt` as the default demo policy.
- Keep the v1 heightmap policy behind
  `ATEC_CROSS_PIT_REQUIRE_DEPLOY_DEFAULTS=0` for diagnostics only.
- Do not promote the TaskD debug raycaster into the submission path unless the
  official evaluation interface can also create the same sensor.

## 2026-06-20 V1 Native-vs-TaskD Handoff Isolation

The native CrossPitBox diagnostic logger was corrected in two ways:

- it pins `cross_pit_success_x` again after `RslRlVecEnvWrapper` construction,
  because the wrapper resets the env during initialization;
- it records both world root position and env-local root position. IsaacLab
  resets terminated envs before returning observations, so `done=True` records
  are explicitly marked as post-reset and include the last pre-step record.

With these fixes, v1 is not a fake native success. Injecting the same
counter-650 13s snapshot into the native `ATEC-Isaac-TaskD-G1-CrossPitBox-v1`
environment gives:

```text
trace = logs/current_native_v1_inject_taskd_snapshot_envlocal_gpu0.jsonl
checkpoint = model_19999.pt
success_x = 5.8
terminal pre-step env-local root ~= (x=5.739, y=-0.788, z=1.026)
post-step terms = {crossed=True, fallen=False}
```

The same native environment also crosses when driven through the demo wrapper
using native ray hits:

```text
trace = logs/current_native_v1_demo_wrapper_true_raycast_actionoffset0_gpu0.jsonl
policy_source = demo wrapper
terminal pre-step env-local root ~= (x=5.733, y=-0.817, z=1.102)
post-step terms = {crossed=True, fallen=False}
```

That rules out the checkpoint itself and the broad demo wrapper observation /
action ordering as the primary failure.

TaskD hard-switch diagnostics from the same injected snapshot still fail:

```text
trace = logs/current_v1_taskd_dynamic_envlocal_gpu0.jsonl
TaskD box = dynamic
max env-local root ~= (x=4.736, y=-1.338, z=0.234)
terms = {fall=True, x_reached=False}
```

Freezing the TaskD box at the injected pose does not fix it:

```text
trace = logs/current_v1_taskd_freezebox_envlocal_gpu0.jsonl
TaskD box = frozen diagnostic
max env-local root ~= (x=4.777, y=-1.095, z=0.225)
terms = {fall=True, x_reached=False}
```

Disabling the v1 action offset compensation is worse:

```text
trace = logs/current_v1_taskd_actionoffset0_envlocal_gpu0.jsonl
ATEC_CROSS_PIT_ACTION_OFFSET_SCALE=0
max env-local root ~= (x=3.964, y=-0.454, z=0.236)
terms = {fall=True, x_reached=False}
```

A yaw heading command also does not solve the failure:

```text
trace = logs/current_v1_taskd_headingkp1_envlocal_gpu0.jsonl
ATEC_CROSS_PIT_HEADING_KP=1.0
max env-local root ~= (x=5.034, y=-1.360, z=0.243)
terms = {fall=True, x_reached=False}
```

Working interpretation:

- The v1 policy reaches the hard section in TaskD, but the root height collapses
  near `x=4.7-5.0` instead of rising to the native trajectory's `z~=1.0`.
- This is not just dynamic box motion, since freezing the box still falls.
- The most likely remaining gap is training-vs-TaskD terrain/contact semantics:
  native v1 trains with a kinematic `BoxSupport` and a modified TaskD terrain
  height range, while official TaskD uses its own generated terrain and dynamic
  box object. The policy is highly sensitive to that support geometry.

Deploy implication:

- Keep using the v1.5 blind policy as the integrated default; it has already
  been verified through `scripts/play_atec_task.py --enable_cameras`.
- Treat v1 heightmap as a diagnostic/retraining branch, not a submission
  default. To make v1 deployable, retrain/fine-tune directly in the TaskD
  support geometry rather than relying on the kinematic CrossPitBox proxy.

## 2026-06-20 Parallel Handoff-Alignment Review

The switch problem was split into two independent policy checks.

v1 heightmap:

- A concrete observation-space bug was found in
  `CrossPitBoxPolicy._estimate_current_d435_link_offset_b()`.
- TaskD `proprio[:, 12:45]` is already the current relative joint pose around
  the official default pose. The wrapper was adding `0.5 * CROSS_PIT_RESET_ACTION`
  again before estimating the waist-dependent `d435_link` offset. That
  double-counted the reset waist pose and shifted the synthetic terrain-map
  heightmap.
- The wrapper now estimates d435 offset directly from the current TaskD relative
  joint pose, after reordering into the native observation order.

v1.5 blind:

- The deploy action mapping remains
  `a_taskd = a_actor + CROSS_PIT_RESET_ACTION`.
- Both the CrossPit training env and TaskD use
  `JointPositionAction(scale=0.5, use_default_offset=True)`, but their default
  joint offsets differ. Removing this mapping applies the actor output around
  the wrong TaskD default pose.
- The v1.5 wrapper should keep `action_offset_scale=1.0`,
  `initial_last_action=zero`, `stabilize_steps=2`, and no handoff blending.

Fresh GPU0 injected-handoff checks:

```text
v1.5 blind, immediate 13.0 s handoff:
log = logs/current_v15_offset1_immediate_switch_gpu0.jsonl
action_offset_scale = 1.0
done terms = {time_out=False, fall=False, x_reached=True}
terminal env-local root ~= (x=7.716, y=-0.745, z=0.630)
```

```text
v1.5 blind, same injected state:
log = logs/current_v15_offset0_switch_recheck_gpu0.jsonl
action_offset_scale = 0.0
done terms = {time_out=False, fall=True, x_reached=False}
terminal env-local root ~= (x=6.469, y=-0.798, z=0.231)
```

```text
v1 heightmap, d435-offset fix, immediate 13.0 s handoff:
log = logs/current_v1_d435fix_immediate_switch_gpu0.jsonl
action_offset_scale = 1.0
done terms = {time_out=False, fall=True, x_reached=False}
terminal env-local root ~= (x=3.580, y=-0.351, z=0.241)
```

Interpretation:

- v1.5 no longer shows a switch-instant explosion under the deploy settings;
  the injected state crosses the TaskD `x_reached` condition.
- `action_offset_scale=0.0` is not a fix for v1.5.
- The v1 heightmap d435 correction fixes a real wrapper bug, but v1 still does
  not cross in TaskD. Keep v1 diagnostic-only unless it is retrained or further
  tuned against the real TaskD support geometry.

## 2026-06-20 v1 D435 Raycast Command Window

The remaining v1 question was whether the hard switch itself is impossible, or
whether the deployed heightmap proxy is too different from the v1 training
raycaster.

A local TaskD diagnostic run was added that can attach a CrossPitBox-v1
`d435_link` raycaster to the TaskD scene and pass the resulting ray hits to the
demo wrapper as `_debug_depth_ray_hits_w`. This is explicitly gated behind
`ATEC_PLAY_ATTACH_CROSS_PIT_DEPTH_SCANNER=1` in `scripts/play_atec_task.py` and
behind `--attach_cross_pit_depth_scanner` in
`scripts/debug_push_then_cross_headless.py`.

With the captured 13.0 s TaskD handoff snapshot, v1 succeeds when it receives
the training-equivalent raycast and a slightly higher forward command:

```text
trace = logs/current_v1_taskd_debug_depth_scanner_stab2_cmd0755_gpu0.jsonl
checkpoint = cross_pit_box_model_19999.pt
ATEC_CROSS_PIT_FORWARD_COMMAND = 0.755
ATEC_CROSS_PIT_STABILIZE_STEPS = 2
raycast source = debug d435 depth_scanner ray_hits_w
done terms = {time_out=False, fall=False, x_reached=True}
terminal world root ~= (x=3.511, y=-0.973, z=0.854)
terminal env-local root ~= (x=7.711, y=-0.973, z=0.854)
```

The same command value with the deploy terrain-map proxy still fails:

```text
trace = logs/current_v1_taskd_terrain_stab2_cmd0755_gpu0.jsonl
checkpoint = cross_pit_box_model_19999.pt
ATEC_CROSS_PIT_FORWARD_COMMAND = 0.755
heightmap source = terrain_map
done terms = {time_out=False, fall=True, x_reached=False}
terminal env-local root ~= (x=3.561, z=0.245)
```

Command sweep with the debug raycaster:

```text
0.735 -> fall before crossing
0.745 -> fall before crossing
0.755 -> x_reached=True, fall=False
0.800 -> fall after overspeed/poor contact
```

Interpretation:

- v1's actor/checkpoint/action mapping can complete the TaskD hard-switch when
  the observation matches its training raycaster closely enough.
- The default v1 `terrain_map` heightmap proxy is not accurate enough for this
  policy; it causes a qualitatively different contact decision near the pit.
- The cloud/demo default should remain v1.5 blind, because the official
  `AlgSolution.predicts()` interface does not create or receive this diagnostic
  raycaster data.
- For local visualization only, v1 can be tested with
  `ATEC_PLAY_ATTACH_CROSS_PIT_DEPTH_SCANNER=1`,
  `ATEC_CROSS_PIT_REQUIRE_DEPLOY_DEFAULTS=0`,
  `ATEC_CROSS_PIT_BOX_CHECKPOINT=cross_pit_box_model_19999.pt`, and
  `ATEC_CROSS_PIT_FORWARD_COMMAND=0.755`.

## 2026-06-20 Full GUI Play Verification

The headless `play_atec_task.py --headless --enable_cameras` path is not a
valid hard-switch verifier for this task: both v1.5 and v1 fell during the
box-push phase at about 9.32 s in that mode, before the configured 13.0 s
handoff. The user-facing GUI/non-headless path is the representative path.

Fresh GUI/non-headless runs on GPU0:

```text
v1.5 blind deploy default:
log = logs/current_baseline_v15_gui_full_final.json
checkpoint = cross_pit_box_v1_5_blind_model_19999.pt
heightmap = disabled, actor_obs_dim = 108
done terms = {fall=False, time_out=False, x_reached=True}
elapsed = 14.92 s
terminal local root ~= (x=7.564, y=-0.298, z=0.362)
```

```text
v1 heightmap with local debug d435 raycaster:
log = logs/current_v1_gui_debug_depth_cmd0755_actualsource_final.json
checkpoint = cross_pit_box_model_19999.pt
ATEC_CROSS_PIT_FORWARD_COMMAND = 0.755
ATEC_PLAY_ATTACH_CROSS_PIT_DEPTH_SCANNER = 1
heightmap_source_actual = debug_ray_hits
done terms = {fall=False, time_out=False, x_reached=True}
elapsed = 14.38 s
terminal local root ~= (x=7.549, y=-1.043, z=0.635)
```

```text
v1 heightmap with terrain-map proxy only:
log = logs/current_v1_gui_terrain_cmd0755_full_final.json
checkpoint = cross_pit_box_model_19999.pt
ATEC_CROSS_PIT_FORWARD_COMMAND = 0.755
heightmap_source_actual = terrain_map
done terms = {fall=True, time_out=False, x_reached=False}
elapsed = 13.84 s
terminal local root ~= (x=4.418, y=-0.382, z=0.272)
```

This closes the state-switch diagnosis:

- v1.5 is deployable through the normal demo interface and remains the default
  submission route.
- v1 can also hard-switch successfully, but only when the actor receives the
  training-equivalent d435 raycast heightmap. This is available in local
  `play_atec_task.py` through a diagnostic env_cfg hook, not through the cloud
  `AlgSolution.predicts()` API.
- The remaining v1 deploy gap is not action-space alignment; it is observation
  fidelity for the 384-D heightmap slot.
