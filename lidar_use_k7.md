# TaskD LiDAR 使用说明

## 运行

GUI + 键盘 + 点云可视化：

```bash
PYTHONPATH=. python scripts/play_atec_task.py --task ATEC-TaskD-G1 --enable_cameras --keyboard --debug --sensor_vis
```

实时查看 CSV：

```bash
tail -f logs/lidar_perception/taskd_lidar_pose.csv
```

CSV 默认每 2s wall time 写一次。

开发验证时可额外写估计/真值对比 CSV：

```bash
PYTHONPATH=. python scripts/play_atec_task.py --task ATEC-TaskD-G1 --enable_cameras --debug --sensor_vis --lidar_pose_gt_log logs/lidar_perception/taskd_lidar_pose_gt.csv
```

## CSV 路径

默认输出：

```text
logs/lidar_perception/taskd_lidar_pose.csv
```

关键字段：

```text
box_x_m, box_y_m, box_yaw_deg
ditch_x_m, ditch_y_m, ditch_yaw_deg
box_conf, ditch_conf
```

## 坐标系

LiDAR 坐标系近似等同 G1 torso/base 坐标系：

```text
+x: 机器人正前方
+y: 机器人正左方
+z: 竖直向上
```

输出的箱子和沟壑 `x/y/yaw` 都是相对于 LiDAR 坐标系。

箱子：

```text
x 轴: 箱子长边方向
y 轴: 箱子短边方向
yaw: 箱子长边相对 LiDAR +x 的角度
```

沟壑：

```text
x 轴: 沟壑长度方向
y 轴: 沟壑宽度方向
yaw: 沟壑长度方向相对 LiDAR +x 的角度
```

## 识别方法

只使用 LiDAR `ray_hits_w`：

1. 将世界点云转到 LiDAR 坐标系。
2. 鲁棒拟合局部地面平面。
3. 箱子：提取高于地面的点云簇，用箱子尺寸先验 `1.0 x 0.8 x 0.6m` 拟合矩形中心和长边 yaw，并用初始世界系主方向避免长短边互换。
4. 沟壑：提取低于地面的点云簇；底部点不足时检测地面高度突变边缘，区分前沿/后沿，再用沟壑宽度先验推出中心。

## 滤波与保持

输出不是纯单帧检测，也不是简单保持上一帧，而是：

```text
单帧几何检测 + 世界系目标轨迹 + LiDAR 系连续输出限速
```

主要规则：

- 箱子 yaw 用初始多帧结果和世界系长边方向做锚定。
- 沟壑初始化先做主簇筛选，避免启动阶段 y 离群点污染轨迹。
- 单帧 x/y/yaw 离群时，不直接采用坏测量，而是沿世界系轨迹预测后再投影回当前 LiDAR 坐标系。
- 最后在 LiDAR 坐标系做小步连续限速，避免相邻日志出现突跳。
- 即使当前帧检测无效，输出也会随 LiDAR 位姿投影连续变化，不会 1s 内完全卡死在上一帧数值。
- `+90deg` 和 `-90deg` 属于同一条无向轴，但日志会按历史连续分支输出，避免 180deg 假跳变。

`conf` 是几何置信度，不是概率；越高表示尺寸、点数和形状越匹配当前先验。

6月2日补充：

  验证结果，TaskD-G1 seed 0/1/2/3 短程 GT 全部达标，没有超阈值行：

  seed0: box max_x 0.0472 max_y 0.0480 max_yaw 0.1198 | ditch max_x 0.0808 max_y 0.0907 max_yaw 0.0424
  seed1: box max_x 0.0588 max_y 0.0531 max_yaw 0.1508 | ditch max_x 0.0302 max_y 0.0761 max_yaw 0.0517
  seed2: box max_x 0.0491 max_y 0.0577 max_yaw 0.1327 | ditch max_x 0.0476 max_y 0.0692 max_yaw 0.0342
  seed3: box max_x 0.0267 max_y 0.0484 max_yaw 0.1005 | ditch max_x 0.0497 max_y 0.0750 max_yaw 0.0342

  单元测试也重新跑过：40 tests OK。

  正常 GUI 测试命令还是：

  PYTHONPATH=. python scripts/play_atec_task.py --task ATEC-TaskD-G1 --enable_cameras --keyboard --debug --sensor_vis

6月3日补充：

  近距离箱子验证可以用调试参数把箱子移动到 LiDAR 坐标系下的指定位置：

  PYTHONPATH=. python scripts/play_atec_task.py --task ATEC-TaskD-G1 --headless --device cuda:0 --debug_lidar_gpu_probe --debug_zero_actions --disable_image_obs --debug_box_pose_lidar 0.70 0.25 30 --max_steps 70

  箱子近距离识别现在会过滤退化边、重复帧和明显错轴 yaw；启动期用多帧候选聚类，初始化后用 yaw 锚点和 20deg 门控拒绝跳变。实测把箱子放在 LiDAR 坐标 `(0.70, 0.25, 30deg)` 时，稳定输出约为 `(0.77, 0.17, 26.8deg)`，首个有效输出误差约 `0.106m / 3.2deg`。

  GPU 并行正式入口在 `demo.solution.get_lidar_perception(num_envs, device="cuda")`，核心实现在 `demo/lidar_perception.py` 和 `demo/lidar_perception_torch.py`。点云变换、地面拟合、box/ditch 估计和时间稳定器都保持 batched torch tensor，正式 GPU 路径不使用 `.cpu()` / `.numpy()` / `.item()`，输出形状为 `(num_envs, 3)`。

  已在本机 `cuda:0` 上用 `num_envs=2` probe 验证并行输出 `shape=(2, 3)`；近距离箱子 raw 结果即使出现 `+60deg/+75deg/+-80deg` 等错轴解释，稳定输出也会保持在正确 yaw 附近，不再出现 2s 日志内的大幅跳变。
