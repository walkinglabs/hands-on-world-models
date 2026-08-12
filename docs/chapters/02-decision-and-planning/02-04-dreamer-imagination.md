# 2.4　Dreamer：在想象中训练 Actor

PlaNet 每做一个真实动作都要重新运行 CEM。若环境要求高频控制，这个开销可能太大。

Dreamer 把世界模型中的好动作练成 Actor，使部署时只需一次前向计算。

## 从真实状态开始想象

我们先从 Replay Buffer 采样真实序列，用 posterior 得到可靠的起始状态。随后冻结世界模型，使用 prior 展开想象：

```text
a_t ~ Actor(s_t)
s_{t+1} ~ WorldModel(s_t, a_t)
r_t = RewardHead(s_t)
```

想象 horizon 常比任务总长度短。过长会积累模型误差，过短则看不到延迟后果。

## Critic 与 TD-λ

Critic 估计想象状态的 value。TD-λ 混合不同长度的回报目标，让训练不只依赖一步 bootstrap，也不必等待完整任务结束。

Actor 的目标是提高想象回报。连续动作常使用经过 `tanh` 变换的正态分布，使动作保持在合法范围。

## 一轮训练循环

```text
真实环境收集 episode
→ Replay Buffer 采样序列
→ 训练 Encoder、RSSM 与预测 heads
→ 从 posterior 状态开始想象
→ 训练 Critic
→ 训练 Actor
→ 回到真实环境收集新数据
```

世界模型、Actor 和 Critic 各有优化器与目标。日志需要分开记录，避免策略退化被总 loss 遮住。

## DreamerV3 的稳定工具

Symlog 压缩跨度很大的 reward 与 value；Twohot 把标量目标放到相邻 bin；Unimix 防止离散概率过早变成绝对值；Free Bits 保护随机状态。

这些技巧解决数值尺度、分类饱和与 KL 使用等问题。错误的动作对齐或模型漏洞不能靠它们修复。

## PA1-A 检查什么

Dreamer-lite 先在 PixelWorld 训练，再使用 DMC Cartpole 小配置。提交真实 return、环境交互步数、多步漂移、峰值显存和一组固定 seed 的模型漏洞。

## 小结

- [ ] Dreamer 从真实 posterior 状态出发，在 prior 中生成想象轨迹。
- [ ] Actor 选择动作，Critic 用 TD-λ 评价想象未来。
- [ ] 最终结果由真实环境回报和样本效率检查。
