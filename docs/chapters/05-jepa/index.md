# 第 5 章　JEPA：怎样只预测有用的未来

一片树叶下一秒向左抖还是向右抖，通常不影响机器人绕开桌子。JEPA 因此不要求还原所有像素，而是预测较稳定的特征。

## 本章文章

1. [为什么改成预测特征](./05-01-feature-prediction.md)
2. [Mask、EMA 与表示坍缩](./05-02-mask-ema-collapse.md)
3. [从图片 JEPA 到视频 JEPA](./05-03-video-jepa.md)
4. [Action-JEPA：让特征未来听从动作](./05-04-action-jepa.md)

## 本章实验与作业

- [C1–C2：JEPA 路线实验](/labs/route-bc)
- [PA1-C：Tiny Video-JEPA](/assignments/pa1-c)

被动视频可以检查表示质量，不能单独证明模型理解控制。只有加入时间对齐的动作以后，才检查反事实与规划。
