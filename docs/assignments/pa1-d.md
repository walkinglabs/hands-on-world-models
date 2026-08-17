# PA1-D · 动手：Tiny VLA 与 World-Model Checker

## 必做核心：直接 VLA

1. 生成 Tabletop 数据，按场景 seed 切分；
2. 完成 state-only BC 基线；
3. 训练 image + instruction + proprioception 的 action chunk 模型；
4. 做换语言、换颜色、换障碍位置的反事实与 OOD；
5. 在环境中报告成功率、碰撞率与延迟，不只报动作 MSE。

## 世界模型扩展

训练 `state + candidate action → next state + collision`，让 VLA 产生多个候选并重排。比较直接 VLA 与 lookahead 的真实闭环成功率、碰撞率和延迟。

若 checker 没有提高结果，也应提交：它是因为候选里没有好动作、后果预测错误，还是碰撞权重不合适？

## T2 迁移

PushT 为推荐小数据；LIBERO 只作进阶。数据必须包含原始 instruction、时间索引、控制频率、proprioception 与下一观察。

## 24GB 目标

小视觉 Encoder、小语言 Encoder、chunk 8–16、单卡 reserved 目标不超过 22GB。当前未完整实测。
