# 2.2　Replay Buffer 怎样取出可学习的数据

一段 episode 只记录一次经历。Replay Buffer 保存许多段经历，让模型能够反复取样训练。

## 不要跨过 episode 边界

episode A 结束以后，episode B 会从新的随机状态开始。若先把数组全部拼起来再截取序列，模型可能学到：A 的终点经过一个不存在的动作，突然来到 B 的起点。

正确顺序是先选择一段足够长的 episode，再从其中截取连续片段。

## 不要随机拆散相邻帧

相邻视频帧几乎相同。若随机按帧切分，训练帧的近邻很容易进入测试集，测试成绩会显得异常漂亮。

课程按 episode、scene、seed 或 task template 分组切分。动作条件任务还要检查测试集中是否包含训练从未覆盖的动作区域。

## 小结

- [ ] Replay Buffer 只在同一 episode 内采样连续片段。
- [ ] train、val、test 按 episode 或场景切分。
- [ ] 测试集不能只是训练帧旁边的几张近邻图片。
