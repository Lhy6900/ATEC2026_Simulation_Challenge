# Keyboard Loco - GR00T & Blind 策略键盘控制使用说明

## 概述

本项目集成了两种 G1 运控策略，并提供键盘实时控制模式，方便调试和交互式测试：

- **GR00T (homie)**：GR00T Decoupled Whole Body Control 策略，支持速度和身高控制
- **Blind**：blind locomotion 策略 (policy.pt)，只支持速度控制

## 启动命令

```bash
# 默认使用 GR00T 策略
PYTHONPATH=. python scripts/play_atec_task.py --task ATEC-TaskD-G1 --enable_cameras --keyboard --debug

# 指定 blind 策略启动
PYTHONPATH=. python scripts/play_atec_task.py --task ATEC-TaskD-G1 --enable_cameras --keyboard --debug --policy blind
```

- `--keyboard`：启用键盘控制模式，不添加则使用默认零速度命令（机器人站立）
- `--policy gr00t|blind`：选择策略，默认 gr00t
- `--debug`：打印每步 reward 和时间信息
- `--enable_cameras`：启用渲染

## 键盘绑定

| 按键 | 功能 | 行为 |
|------|------|------|
| W | 前进 (vx+) | 按住加速，松开衰减 |
| S | 后退 (vx-) | 按住加速，松开衰减 |
| A | 右移 (vy+) | 按住加速，松开衰减 |
| D | 左移 (vy-) | 按住加速，松开衰减 |
| Q | 右转 (vyaw+) | 按住加速，松开衰减 |
| E | 左转 (vyaw-) | 按住加速，松开衰减 |
| Z | 身高升高 | 每按一次 +0.02，保持不衰减 |
| X | 身高降低 | 每按一次 -0.02，保持不衰减 |
| R | 重置 | 速度归零，身高恢复默认 0.74 |
| B | 切换到 blind 策略 | 即时切换 |
| H | 切换到 homie (GR00T) 策略 | 即时切换 |

## 行为说明

- **速度轴**（WASD / QE）：按住持续加速（0.01/步），到达上限后不再增加；松开后以指数衰减回归 0（每步乘 0.92）
- **身高**（Z / X）：每次按键调整 0.02，不衰减，停留在设定值（仅 GR00T 策略生效）
- **重置**（R）：速度归零，身高恢复默认 0.74
- **策略切换**（B / H）：即时切换，切换后速度从零开始
- 键盘模式激活时，自动导航 ramp 被禁用，完全由键盘控制

## 速度上限

keyboard_teleop 提供宽松的上限，各策略内部进一步 clamp：

**GR00T 策略上限**（`demo/groot_policy_adapter.py` + `scripts/keyboard_teleop.py`）：

| 参数 | 键盘上限 | 策略内部上限 |
|------|----------|-------------|
| vx | [-1.5, 1.5] | 由 nav_scale × fixed_nav_cmd 控制 |
| vy | [-1.0, 1.0] | 同上 |
| vyaw | [-1.0, 1.0] | 同上 |
| height | [0.60, 0.85] | [0.60, 0.85] |

**Blind 策略上限**（`demo/solution_blind.py`）：

| 参数 | 值 |
|------|-----|
| VX_MIN, VX_MAX | -0.5, 1.5 |
| VY_MIN, VY_MAX | -0.75, 0.75 |
| VYAW_MIN, VYAW_MAX | -1.0, 1.0 |

## 修改速度上限和手感

**键盘通用参数** — 编辑 `scripts/keyboard_teleop.py` 顶部：

```python
VX_MIN, VX_MAX = -1.5, 1.5    # 键盘层宽松上限
VY_MIN, VY_MAX = -1.0, 1.0
VYAW_MIN, VYAW_MAX = -1.0, 1.0
HEIGHT_MIN, HEIGHT_MAX = 0.60, 0.85
HEIGHT_DEFAULT = 0.74
HEIGHT_STEP = 0.02

ACCEL_RATE = 0.01   # 每步加速量
DECAY_RATE = 0.92   # 松开后每步衰减因子
```

**Blind 策略专用上限** — 编辑 `demo/solution_blind.py`：

```python
VX_MIN, VX_MAX = -0.5, 1.5
VY_MIN, VY_MAX = -0.75, 0.75
VYAW_MIN, VYAW_MAX = -1.0, 1.0
```

---

## 相比原版的改动汇总

### 修改的原版文件

**`demo/requirements.txt`**
- 添加 `onnxruntime` 依赖

**`demo/solution.py`**
- 原版为空壳（返回零动作），改为策略分发器
- 支持运行时切换 gr00t / blind 策略
- `set_policy()` / `get_policy_name()` 全局接口
- 转发 `reset()` / `predicts()` / `set_keyboard_command()` / `get_debug_snapshot()`

**`scripts/play_atec_task.py`**
- 添加 `--keyboard` 和 `--policy` CLI 参数
- 集成 `KeyboardTeleop`：每步获取键盘命令、检测策略切换请求
- 注释掉 `camera_follow(env)`（解决视角被强制锁定的问题）
- 退出时清理键盘订阅

**`source/atec_rl_lab/atec_rl_lab/tasks/task_d/env_cfg.py`**
- 为 G1 的 legs/feet actuator 设置与 GR00T 策略对齐的 PD 增益，替代原版默认值
  - legs: hip 150/2, knee 200/4, waist 250/5
  - feet: stiffness 40, damping 2
- 初始关节位姿保持原版不变

### 新增文件

| 文件 | 说明 |
|------|------|
| `demo/groot_policy_adapter.py` | GR00T 策略适配器核心：ATEC obs → GR00T obs 构建（含腰部 FK 计算 torso_rpy）、ONNX 推理（balance/walk 切换）、上体固定默认姿态 + 下体策略输出、运行时命令接口 |
| `demo/solution_homie.py` | GR00T 方案包装器：状态机（startup_zero → policy）、键盘控制支持 |
| `demo/solution_blind.py` | Blind 方案：policy.pt 推理，velocity_commands 完全由键盘驱动，自有速度上限 clamp |
| `demo/policy.pt` | Blind locomotion 策略模型 |
| `demo/GR00T-WholeBodyControl-Balance.onnx` | GR00T balance 策略模型 |
| `demo/GR00T-WholeBodyControl-Walk.onnx` | GR00T walk 策略模型 |
| `scripts/keyboard_teleop.py` | 键盘控制器：carb.input 事件订阅、按键计数器追踪、按住加速/松开衰减/身高步进/策略切换 |
| `scripts/README_keyboard.md` | 键盘控制简要说明 |
| `keyboard_loco_user.md` | 本文档 |

### 策略适配关键参数（`demo/groot_policy_adapter.py`）

| 参数 | 值 | 说明 |
|------|-----|------|
| `GR00T_CMD_SCALE` | [2.0, 2.0, 0.5] | 导航命令缩放 |
| `GR00T_ANG_VEL_SCALE` | 0.5 | 角速度缩放 |
| `GR00T_DOF_POS_SCALE` | 1.0 | 关节位置缩放 |
| `GR00T_DOF_VEL_SCALE` | 0.05 | 关节速度缩放 |
| `GR00T_ACTION_SCALE` | 0.25 | 策略输出缩放 |
| `GR00T_HISTORY_LEN` | 6 | 观测历史长度 |
| `GR00T_SINGLE_OBS_DIM` | 86 | 单帧观测维度 |
| `GR00T_NUM_ACTIONS` | 15 | 下体动作维度 |
| `ATEC_ENV_ACTION_SCALE` | 0.5 | ATEC 环境动作缩放 |
| `DEFAULT_BASE_HEIGHT_CMD` | 0.74 | 默认身高命令 |
