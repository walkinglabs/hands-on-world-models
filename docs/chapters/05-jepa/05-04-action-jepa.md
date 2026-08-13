# 5.4　Action-JEPA：让特征未来听从动作

同一个机器人状态下，向左推和向右推会得到不同未来。被动 JEPA 只看视频，无法知道动作是变化的原因还是伴随信号。

Action-JEPA 把候选动作加入 Predictor：

```text
历史 features + action → predicted future feature
```

## 数据要求

一条样本至少需要 observation、action、next observation 和时间戳。机器人任务还要记录 proprioception、控制频率和执行延迟。

UCF101-mini 没有动作标签，只能用于被动预训练。PixelWorld 或机器人轨迹才可用于动作条件阶段。

## 反事实检查

固定同一历史和环境随机源，只替换动作。预测 feature 应朝不同未来移动。

仅比较 feature MSE 不够，因为模型可能对动作变化不敏感。可以训练位置 probe，把预测 feature 映射成可解释位置，再检查动作方向。

## 最小规划器

为每个候选动作预测一步 feature，用 reward head 或目标距离评价，选择得分最高的动作。

若加入 Action-JEPA 后规划没有优于“保持原动作”基线，表示可能没有保留任务需要的可控信息。

课程的 C2 会按 episode seed 分开 probe 的训练与测试。若线性头只在训练 feature 上拟合得很好，我们只能说表示能够记住样本，不能说它保留了可迁移的位置。短期动作选择也使用同一个 held-out probe，把候选 feature 映射成位置以后再比较目标距离。

## 与 Dreamer 的边界

Action-JEPA 和 Dreamer 都可以在 latent 中预测未来。JEPA 的重点是非生成特征目标与表示质量；Dreamer 进一步训练 reward、continue、Actor 和 Critic，目标是提高真实回报。

## PA1-C

C1 完成 tubelet、mask、EMA 和 Tiny Video-JEPA；C2 完成坍缩检查、linear probe、动作条件和一步规划。主指标分别是表示统计、probe、动作敏感性和规划增益。

## 小结

- [ ] Action-JEPA 把动作作为未来特征预测的条件。
- [ ] 被动预训练与动作条件训练需要不同数据。
- [ ] 只有动作反事实和下游选择，才能说明特征支持控制。
