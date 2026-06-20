# multi-agent9h 调试总结

## 目标

这次工作的目标是把 TaskD 拆成两个阶段：

1. 前 13 秒沿用原来的推箱子 planner，保持 `ATEC-TaskD-G1` 的推箱子行为不变。
2. 在 13.0 s 左右硬切换到已经训练好的跨沟壑 policy，让机器人从 13 秒附近的稳定状态继续越过箱子和沟壑。

最终保留了两个可运行版本：

- **云端默认版本：v1.5 blind**，入口是 `demo/solution.py`，不依赖额外高程图传感器，适合提交测评。
- **本机调试版本：v1 heightmap**，入口是 `demo/solution_v1_debug.py`，需要本机 `play_atec_task.py` 额外挂训练同款 D435 raycaster 才能复现实验效果。

## 主要问题

### 1. 训练通过率和 play 结果一度不一致

早期 v0/v1 训练里 `crossed` 比例看起来很高，但在 Isaac Sim 里 play 时机器人没有真正完成 TaskD。根因有两个：

- 训练任务里的 `target_x/crossed` 一开始更像 CrossPit 局部目标，不等价于 TaskD 官方 `x_reached`。
- TaskD 官方终止线是世界坐标 `robot_x > 3.5`，换成 CrossPit 局部坐标大约是 `local_x > 7.7`，而训练里 `target_x=5.8` 只是越障 curriculum 目标。

后续修正了目标线理解，并在 debug 输出里同时记录 local/world pose 和 `done_terms`，避免只看到 `terminated=True` 却不知道是成功还是摔倒。

### 2. 硬切换后暴走或直接摔倒

单独运行 CrossPit policy 正常，但从推箱子阶段 13 秒硬切换后暴走/摔倒。排查后发现不是推箱子 planner 的问题：原始推箱子命令在 GUI 下 13 秒状态稳定，箱子位置和机器人姿态也基本一致。

真正的问题集中在 handoff wrapper：

- TaskD public obs/action 的关节顺序和 CrossPit 训练环境内部关节顺序不同。
- action offset 约定理解错了，CrossPit actor 输出需要加回训练 reset action offset。
- 首帧 `last_action` 不能直接用 TaskD 上一阶段动作，默认用 zero 更贴近训练 reset。
- 切换瞬间速度观测和姿态略有分布差，v1.5 blind 需要 2 step 的轻量稳定窗口。

最终 `CrossPitBoxPolicy` 里统一处理了关节重排、action offset、首帧 last action、checkpoint-aware 默认参数。

### 3. v1 heightmap 的观测来源不匹配

v1 是 492 维 actor obs，其中 384 维是训练环境里的 D435 heightmap。云端/TaskD demo 默认 obs 只有官方传入的 proprio、extero、image，不能自动获得训练环境里的 raycaster heightmap。

因此出现了一个关键现象：

- v1 在单独 `scripts/rsl_rl/play.py` 里很好。
- v1 在集成 TaskD 时如果只用 `terrain_map/head_depth/extero` 近似 heightmap，会摔倒。
- v1 在本机 `play_atec_task.py` 里额外挂 D435 raycaster，并把 `_debug_depth_ray_hits_w` 传给 solution 后，可以成功。

这说明 v1 policy 本身有效，但它依赖一个云端官方接口未必提供的观测通道。基于这个风险，云端默认不使用 v1 heightmap，而使用 v1.5 blind。

### 4. headless 验证给过假阴性

曾经用 `--headless --enable_cameras` 验证完整流程，结果推箱子阶段 9.32 s 左右就摔了。这和用户 GUI baseline、13 秒 capture 都矛盾。

后续确认：这是 headless/rendering path 的验证伪差，不能作为推箱子阶段失败的证据。正式判断以 GUI/非 headless + `--enable_cameras` 的完整 rollout 和 13 秒 snapshot injection 为准。

### 5. 云端打包曾经缺资产

云端报过：

```text
FileNotFoundError: Missing required policy asset: boxpush_planner_10hz.pt
```

说明 demo 文件不能只提交 `solution.py` 和少量源码，所有 checkpoint fallback 都要一起提交。现在 `demo/Dockerfile` 也已经从全 0 demo 改成真实 push-then-cross 入口，避免 Docker 方式提交时误跑 `solution_zero.py`。

## 当前解决方案

### 默认提交方案：v1.5 blind

入口：`demo/solution.py`

核心配置：

```text
handoff_time_s = 13.00
checkpoint = cross_pit_box_v1_5_blind_model_19999.pt
actor_obs_dim = 108
uses_heightmap = False
target_x = 5.8
forward_command = 0.6644027
action_offset_scale = 1.0
initial_last_action = zero
stabilize_steps = 2
stabilize_vel_gain = 0.025
handoff_blend_steps = 0
handoff_max_delta = 0.0
```

这个版本的优势是只依赖云端官方 obs 可获得的信息，不需要额外改环境传感器。

### 本机调试方案：v1 heightmap

入口：`demo/solution_v1_debug.py`

核心配置：

```text
checkpoint = cross_pit_box_model_19999.pt
actor_obs_dim = 492
uses_heightmap = True
target_x = 5.8
forward_command = 0.755
stabilize_steps = 2
ATEC_PLAY_ATTACH_CROSS_PIT_DEPTH_SCANNER = 1
```

这个版本必须在本机 `scripts/play_atec_task.py` 中开启 D435 raycaster hook，让 policy 的 `heightmap_source_actual=debug_ray_hits`。如果没有这个 hook，它会退化到近似 terrain map/head depth，效果不可靠。

## 最终验证结果

### v1.5 blind 完整 GUI 流程

日志：`logs/current_baseline_v15_gui_full_final.json`

```text
elapsed_time_s = 14.920000076293945
stage = cross_pit_box
actor_obs_dim = 108
uses_heightmap = False
local_x = 7.563549518585205
local_y = -0.2978816032409668
local_z = 0.36197352409362793
done_terms = {"fall": false, "time_out": false, "x_reached": true}
```

实测完整运行墙钟时间约 **101 秒**。

### v1 heightmap + 本机 D435 debug raycaster

日志：`logs/current_v1_gui_debug_depth_cmd0755_actualsource_final.json`

```text
elapsed_time_s = 14.380000114440918
stage = cross_pit_box
actor_obs_dim = 492
uses_heightmap = True
heightmap_source_actual = debug_ray_hits
local_x = 7.549161434173584
local_y = -1.0433639287948608
local_z = 0.6346077919006348
done_terms = {"fall": false, "time_out": false, "x_reached": true}
```

实测完整运行墙钟时间约 **106 秒**。

### v1 heightmap 但不挂 D435 debug raycaster

日志：`logs/current_v1_gui_terrain_cmd0755_full_final.json`

```text
elapsed_time_s = 13.84000015258789
actor_obs_dim = 492
uses_heightmap = True
heightmap_source = terrain_map
local_x = 4.41808557510376
local_z = 0.27237364649772644
done_terms = {"fall": true, "time_out": false, "x_reached": false}
```

这个对照实验说明：v1 不是不能用，而是必须匹配训练时的真实 heightmap 观测。

## 本机可视化命令

### v1.5 blind，推荐云端默认方案

```bash
CUDA_VISIBLE_DEVICES=0 \
TMPDIR=/data/home/zhukq/Workspace/Atec/tmp_isaaclab \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
__NV_PRIME_RENDER_OFFLOAD=0 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
ATEC_BOXPUSH_PLANNER_CHECKPOINT=boxpush_planner_v2.pt \
ATEC_BOXPUSH_WALK_POLICY=gr00t_walk.pt \
ATEC_BOXPUSH_BALANCE_POLICY=gr00t_balance.pt \
ATEC_CROSS_PIT_BOX_CHECKPOINT=cross_pit_box_v1_5_blind_model_19999.pt \
ATEC_CROSS_PIT_REQUIRE_DEPLOY_DEFAULTS=1 \
ATEC_CROSS_PIT_HANDOFF_TIME=13.0 \
ATEC_DEMO_STEP_DT=0.02 \
PYTHONPATH=. \
python scripts/play_atec_task.py --task ATEC-TaskD-G1 --enable_cameras --device cuda:0
```

### v1 heightmap，本机调试方案

```bash
CUDA_VISIBLE_DEVICES=0 \
TMPDIR=/data/home/zhukq/Workspace/Atec/tmp_isaaclab \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
__NV_PRIME_RENDER_OFFLOAD=0 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
ATEC_BOXPUSH_PLANNER_CHECKPOINT=boxpush_planner_v2.pt \
ATEC_BOXPUSH_WALK_POLICY=gr00t_walk.pt \
ATEC_BOXPUSH_BALANCE_POLICY=gr00t_balance.pt \
ATEC_CROSS_PIT_BOX_CHECKPOINT=cross_pit_box_model_19999.pt \
ATEC_CROSS_PIT_REQUIRE_DEPLOY_DEFAULTS=0 \
ATEC_CROSS_PIT_FORWARD_COMMAND=0.755 \
ATEC_CROSS_PIT_STABILIZE_STEPS=2 \
ATEC_PLAY_ATTACH_CROSS_PIT_DEPTH_SCANNER=1 \
ATEC_CROSS_PIT_HANDOFF_TIME=13.0 \
ATEC_DEMO_STEP_DT=0.02 \
PYTHONPATH=. \
python scripts/play_atec_task.py --task ATEC-TaskD-G1 --enable_cameras --device cuda:0
```

## 云端提交文件建议

推荐提交 v1.5 blind 默认方案。需要提交 `demo/` 里的这些文件：

```text
solution.py
server.py
run.sh
requirements.txt
boxpush_planner_solution.py
boxpush_low_level.py
planner_inference_policy.py
cross_pit_box_policy.py
boxpush_planner_v2.pt
boxpush_planner_10hz.pt
gr00t_walk.pt
gr00t_balance.pt
cross_pit_box_v1_5_blind_model_19999.pt
```

如果云端要求 Dockerfile 构建，也一起提交：

```text
Dockerfile
```

不建议默认提交为 v1 heightmap。如果只是想把 v1 本机调试版本也备份到云端文件包，可额外带上：

```text
solution_v1_debug.py
cross_pit_box_model_19999.pt
```

但除非测评环境允许像本机一样给 `solution.py` 注入 `_debug_depth_ray_hits_w` 或允许修改环境加 D435 raycaster，否则不要把 `solution_v1_debug.py` 改名成云端默认 `solution.py`。

## 经验

1. RL 训练里的 `crossed` 必须和最终任务判定线对齐。否则 tensorboard 看起来很好，实际任务未必完成。
2. sim-to-wrapper 的 obs/action 顺序比 policy 本身更容易出错，尤其是多环境 articulation order 和官方接口 order 不一致时。
3. 高程图 policy 的部署风险不在网络结构，而在观测源一致性。没有训练同款 raycaster，CNN/MLP 都可能吃到分布外输入。
4. 完整任务验证必须看 named termination terms，不能只看 `terminated=True`。
5. headless 验证不一定等价于 GUI 验证，特别是这里推箱子阶段依赖 camera/render path。
6. demo 提交要把所有 fallback checkpoint 一起带上，云端缺一个 `.pt` 就会在初始化阶段直接失败。

