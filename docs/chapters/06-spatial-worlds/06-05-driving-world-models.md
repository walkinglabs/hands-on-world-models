# 6.5　自动驾驶为什么预测未来占用

驾驶系统不只需要生成一段清楚道路视频。规划器更关心未来哪里被车辆和行人占用，自车计划是否会进入这些区域。

## 输入与输出

```text
过去多相机图像 + 标定 + ego motion / ego plan
→ future BEV / occupancy / agent trajectories
```

多相机首先通过几何或学习方法进入共同空间，时序模型再预测未来。

## 自车动作为何重要

自车左转和直行时，未来相机视角、可见区域和碰撞关系不同。若数据只包含过去图像和未来标签，没有候选 ego plan，模型只能做 open-loop future prediction。

这种预测仍有价值，但不能声称已经比较不同驾驶动作。

## 评价

Future Occupancy 使用不同 horizon 的 IoU、动态类别结果和时间一致性。Agent trajectory 使用 ADE/FDE。碰撞率和 off-road 需要真实闭环模拟器与规划器，离线标签无法单独提供。

远距离小目标、遮挡后重现、转弯和罕见交互应单独报告，不让平均指标掩盖安全相关失败。

## 数据台阶

先用项目内标定立方体检查内外参和 BEV，再用 nuScenes-mini 做单帧 BEV 与未来 Occupancy。若数据不含明确控制动作，课程会标记为 open-loop，而非闭环驾驶世界模型。

## PA1-E

学生完成共同 E1 后，在 3D/4D 与驾驶分支中选择一个。设计目标为单张 24GB 配方；只有提交完整日志、峰值显存、曲线和 checkpoint 后才标记为真机验证。

## 小结

- [ ] 驾驶世界模型主要预测 future BEV、Occupancy 或轨迹，而非只追求漂亮视频。
- [ ] 候选 ego plan 是比较驾驶动作后果的必要条件。
- [ ] 离线 future prediction 与闭环规划使用不同证据。
