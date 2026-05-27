# Keyboard Loco - GR00T 策略迁移 & 键盘控制使用说明

## 概述

本项目将 GR00T Decoupled Whole Body Control 策略迁移到 ATEC G1-TaskD 环境中，并提供键盘实时控制模式，方便调试和交互式测试。

## 启动命令

```bash
PYTHONPATH=. python scripts/play_atec_task.py --task ATEC-TaskD-G1 --enable_cameras --keyboard --debug
```

- `--keyboard`：启用键盘控制模式，不添加则使用默认零速度命令（机器人站立）
- `--debug`：打印每步 reward 和时间信息
- `--enable_cameras`：启用渲染

## 键盘绑定

| 按键 | 功能 | 范围 | 行为 |
|------|------|------|------|
| W | 前进 (vx+) | [-0.2, 0.8] | 按住加速，松开衰减 |
| S | 后退 (vx-) | [-0.2, 0.8] | 按住加速，松开衰减 |
| A | 右移 (vy+) | [-0.5, 0.5] | 按住加速，松开衰减 |
| D | 左移 (vy-) | [-0.5, 0.5] | 按住加速，松开衰减 |
| Q | 右转 (vyaw+) | [-0.5, 0.5] | 按住加速，松开衰减 |
| E | 左转 (vyaw-) | [-0.5, 0.5] | 按住加速，松开衰减 |
| Z | 身高升高 | [0.60, 0.85] | 每按一次 +0.02，保持不衰减 |
| X | 身高降低 | [0.60, 0.85] | 每按一次 -0.02，保持不衰减 |
| R | 重置 | vx=vy=vyaw=0, height=0.74 | 即时生效 |

## 行为说明

- **速度轴**（WASD / QE）：按住持续加速（0.01/步），到达上限后不再增加；松开后以指数衰减回归 0（每步乘 0.92）
- **身高**（Z / X）：每次按键调整 0.02，不衰减，停留在设定值
- **重置**（R）：速度归零，身高恢复默认 0.74
- 键盘模式激活时，自动导航 ramp 被禁用，完全由键盘控制

## 修改速度上限和手感

编辑 `scripts/keyboard_teleop.py` 顶部常量：

```python
# 速度上下限
VX_MIN, VX_MAX = -0.2, 0.8
VY_MIN, VY_MAX = -0.5, 0.5
VYAW_MIN, VYAW_MAX = -0.5, 0.5
HEIGHT_MIN, HEIGHT_MAX = 0.60, 0.85
HEIGHT_DEFAULT = 0.74
HEIGHT_STEP = 0.02

# 加速和衰减
ACCEL_RATE = 0.01   # 每步加速量，越大手感越灵敏
DECAY_RATE = 0.92   # 松开后每步衰减因子，越小减速越快
```

---

## 相比原版的改动汇总

### 修改的原版文件

**`demo/requirements.txt`**
- 添加 `onnxruntime` 依赖

**`demo/solution.py`**
- 原版为空壳（返回零动作），改为委托给 `HomieSolution`
- 增加 `reset()`、`set_keyboard_command()`、`get_debug_snapshot()` 转发方法

**`scripts/play_atec_task.py`**
- 添加 `--keyboard` CLI 参数
- 集成 `KeyboardTeleop`：每步获取键盘命令并传给 solution
- 添加 `get_debug_snapshot()` 调试输出
- 注释掉 `camera_follow(env)`（解决视角被强制锁定的问题）
- 退出时清理键盘订阅

**`source/atec_rl_lab/atec_rl_lab/tasks/task_d/env_cfg.py`**
- 为 G1 的 legs/feet actuator 设置与 GR00T 策略对齐的 PD 增益（stiffness/damping），替代原版默认值
  - legs: hip 150/2, knee 200/4, waist 250/5
  - feet: stiffness 40, damping 2
- 初始关节位姿保持原版不变

### 新增文件

| 文件 | 说明 |
|------|------|
| `demo/groot_policy_adapter.py` | GR00T 策略适配器核心：ATEC obs → GR00T obs 构建（含腰部 FK 计算 torso_rpy）、ONNX 推理（balance/walk 切换）、上体固定默认姿态 + 下体策略输出、运行时命令接口 |
| `demo/solution_homie.py` | 解决方案包装器：状态机（startup_zero → policy）、键盘控制支持 |
| `scripts/keyboard_teleop.py` | 键盘控制器：carb.input 事件订阅、按住加速/松开衰减/身高步进/重置 |
| `scripts/README_keyboard.md` | 键盘控制简要说明 |
| `demo/GR00T-WholeBodyControl-Balance.onnx` | GR00T balance 策略模型 |
| `demo/GR00T-WholeBodyControl-Walk.onnx` | GR00T walk 策略模型 |

### 策略适配关键参数（位于 `demo/groot_policy_adapter.py`）

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

策略选择逻辑：导航命令范数 >= 0.05 使用 walk 模型，否则使用 balance 模型。
