# 3.5　MuZero：只保留搜索需要的信息

Dreamer 通常用 Decoder 或观察预测帮助训练状态。棋类规划却不需要重画棋盘纹理，只需要知道动作之后的 reward、policy 和 value。

MuZero 因此学习一种 **value-equivalent** 表示：状态不必还原完整观察，但要保留搜索所需的信息。

## 三个网络

```text
Representation：观察历史 → 初始隐状态 s_0
Dynamics：s_t + action → s_{t+1} + reward
Prediction：s_t → policy + value
```

动作成为搜索树的边。Dynamics 在隐空间展开下一状态，Prediction 为搜索提供先验概率和值。

## MCTS 怎样使用模型

MCTS 反复执行选择、扩展、评估和回传。访问次数形成比原策略更集中的搜索策略，作为新的训练目标。

训练目标来自真实 reward、搜索 policy 和后续 value。模型从自我对弈或环境轨迹中不断更新。

## 与 Dreamer 的不同

Dreamer 使用可微想象训练 Actor，适合连续控制；MuZero 使用树搜索改进 policy，经典任务多为离散动作。

Dreamer 的状态常受观察重建约束；MuZero 主动舍弃与 reward、policy、value 无关的细节。代价是隐状态更难用人眼解释，也可能漏掉新任务后来需要的信息。

## Mini-MuZero 的课程范围

选修实验使用四子棋。棋盘规则提供准确环境，学生实现 representation、dynamics、prediction 和小型 MCTS，比较“直接 policy”与“加入搜索”的胜率。

这不等于复现工业规模 MuZero。课程目标是看清：改变预测目标以后，模型保留的信息和评价方式也随之改变。

## 小结

- [ ] MuZero 不要求重建观察，只预测 reward、policy 和 value。
- [ ] MCTS 在隐空间展开动作树，用访问次数改进 policy。
- [ ] Value-equivalent 表示适合当前规划目标，不保证保留所有世界信息。
