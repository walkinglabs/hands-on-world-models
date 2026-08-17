# PA1-C · 动手：训练并审问一个 Tiny Video-JEPA

目标不是把 feature loss 降到最低，而是弄清表示保留了什么、是否受动作控制、能否帮助一个小任务。

## 必做

1. 在 PixelWorld 训练被动 Tiny Video-JEPA；
2. 报告 feature loss、方差、协方差或最近邻，排查坍缩；
3. 冻结 Encoder，用 linear probe 读出位置与速度；
4. 加入真实动作，训练 Action-JEPA；
5. 固定历史替换动作，检查预测 feature 与 probe 结果；
6. 完成一步或短 horizon MPC，并在真实环境检查成功率；
7. 与像素预测或无动作 predictor 做公平对照；
8. 提交显存、时间、曲线与 checkpoint 哈希。

## 选做迁移

在 UCF101-mini 上做被动视频表示迁移。它可以报告 action recognition 或 probe，不能用于声称机器人控制。

## 24GB 目标

`8–16` 帧、`112×112` 或更小分辨率、Tiny/Small ViT、混合精度可选，峰值 reserved 目标不超过 22GB。当前未完整实测。

## 最后回答

模型丢掉了什么？在当前任务里这是否有益？换一个任务以后，原来被丢掉的信息会不会重新变得重要？
