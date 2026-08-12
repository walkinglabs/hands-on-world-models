# 1.6　Value、Policy 与 Planner

世界模型可以预测多条未来，但它不知道哪条未来符合任务。机器人要到达终点、避免碰撞或节省时间，还需要把任务目标放进比较过程。

## Reward、Return 与 Value

奖励（reward）描述一步结果。回报（return）是整条未来的折扣累计奖励：

```text
G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...
```

价值（value）估计从当前状态继续行动，大约还能得到多少回报。它让有限 horizon 的规划器不必一直预测到任务结束。

TD-Learning 用下一状态的价值估计当前目标。TD-λ 把不同长度的回报估计混合起来，在偏差和方差之间取舍。

## Policy、Actor 与 Critic

Policy 根据当前信息输出动作或动作分布。Actor 是负责行动的 Policy；Critic 评价 Actor 当前选择的未来价值。

Actor-Critic 可以完全不使用世界模型。Dreamer 的特点是：Actor 和 Critic 在世界模型生成的想象轨迹上训练。

## Planner 与 MPC

Planner 不直接记住一个动作答案，而是调用世界模型比较候选动作序列。

MPC 每次预测几步，只执行第一步，再根据真实观察重新规划。模型不完全准确时，这个循环提供持续修正。

## CEM 与 MCTS

连续动作无法全部枚举。CEM 先随机采样许多动作序列，保留较好的部分，再围绕它们重新采样。PlaNet 使用这种方法。

MCTS 把动作看成树的边，把更多搜索预算给有希望的分支。MuZero 学习 reward、policy 和 value，再在隐空间运行 MCTS。

## 怎样选择

动作空间较小、规则明确时，枚举或树搜索很自然。连续控制适合 CEM。需要快速反应时，可以用规划数据或模型想象训练 Actor。

是否需要 Planner，取决于目标变化、实时预算和模型可靠性。简单稳定任务只使用 Policy 也可能更合适。

## 小结

- [ ] Reward 描述一步，return 描述一条未来，value 估计剩余回报。
- [ ] Policy 直接输出动作，Planner 调用模型比较候选未来。
- [ ] CEM 适合连续序列采样，MCTS 适合树形搜索。
