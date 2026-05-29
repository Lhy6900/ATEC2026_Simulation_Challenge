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
3. 箱子：提取高于地面的点云簇，用箱子尺寸先验 `1.0 x 0.8 x 0.6m` 拟合矩形中心和长边 yaw。
4. 沟壑：优先提取低于地面的点云簇；如果底部点不足，则检测地面高度突变边缘，再用沟壑宽度先验推出中心。

## 滤波与保持

输出不是纯单帧检测，而是：

```text
单帧几何检测 + 时间连续性滤波 + 上一帧保持
```

主要规则：

- 箱子 yaw 用初始多帧结果做锚定。
- 如果 yaw 突然跳变过大，拒绝这次 yaw 更新。
- 如果 x/y 单次跳变过大，限制变化幅度。
- 如果当前帧目标无效，沿用上一帧有效结果。
- `+90deg` 和 `-90deg` 属于同一条无向轴，但日志会按历史连续分支输出，避免 180deg 假跳变。

`conf` 是几何置信度，不是概率；越高表示尺寸、点数和形状越匹配当前先验。
