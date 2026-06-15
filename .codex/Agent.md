# 项目要求

本项目为Robotics仿真竞赛项目，需要按照竞赛要求修改代码，完成taskD。

* 你撰写的文档、注释、给用户的输出需采用简体中文
* 提交时仅允许修改demo文件夹中内容，并且保证solution.py和其中的AlgSolution结构不变，符合README.md的规范。（原项目完全符合，不用你再次检查。如果你没有修改相关文件则不用再次检查）
* 使用conda环境isaaclab运行程序，默认测试指令为：ATEC_BOXPUSH_PLANNER_CHECKPOINT=boxpush_planner_v2.pt PYTHONPATH=. python scripts/play_atec_task.py --task ATEC-TaskD-G1 --enable_cameras --debug
* 第一步推箱子进入坑已经实现了，现在的目标是训练一个策略能跨过坑，到达终点。