# 第 4 章　JEPA：在特征空间预测

一片树叶下一秒向左抖还是向右抖，通常不影响机器人绕开桌子。JEPA 因此不要求还原所有像素，而是在特征空间里预测较稳定的未来。

本路线把预测目标从像素换成特征，用 mask、EMA 与 stop-gradient 避免表示坍缩，再在视频上引入动作，让特征未来听从控制信号。

## 本章文章

1. [4.1 预测特征而非像素](./04-01-feature-prediction.md)
2. [4.2 掩码、EMA 与表示坍缩](./04-02-mask-ema-collapse.md)
3. [4.3 视频 JEPA](./04-03-video-jepa.md)
4. [4.4 动作条件 JEPA（Action-JEPA）](./04-04-action-jepa.md)

## 本章动手 notebook

- [C1 动手：视频特征预测（JEPA）](/labs/route-bc)
- [C2 动手：动作条件特征预测（Action-JEPA）](/labs/route-bc)

被动视频可以检查表示质量，不能单独证明模型理解控制。只有加入时间对齐的动作以后，才检查反事实与规划。
