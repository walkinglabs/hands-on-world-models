# PA1-A · 做出一台 Dreamer-lite

PA1-A 是路线 A 的小整机。目标不是复现 DreamerV3 的排行榜，而是完成一条有证据的真实循环：从环境收集数据，在 RSSM 中想象，用 Actor-Critic 学习，再回到环境检查。

## 必做阶段

### 阶段一：PixelWorld

复用项目内 PixelWorld 生成器，增加可学习策略与明确目标。完成：

- Replay Buffer 连续序列采样；
- CNN Encoder 与 RSSM；
- observation、reward、continue 与 KL losses；
- 从 posterior state 开始的 imagination；
- Actor、Critic 与 TD-λ；
- 真实 episode 收集与再次训练。

### 阶段二：DMC Cartpole 小迁移

使用 DeepMind Control Suite Cartpole 的固定小配置。若本地不方便安装 MuJoCo，可以只提交 PixelWorld 必做结果，并把 DMC 标为未运行，不能使用他人曲线代替。

## 必交证据

1. 数据卡与环境 wrapper；
2. 世界模型各项 loss 曲线；
3. one-step 与 5/15/30-step 预测或 latent 指标；
4. 随机策略、PlaNet/CEM 或无模型策略中的至少一个基线；
5. 真实环境 return 对真实交互步数；
6. 同一起点替换动作的反事实；
7. 一组 Actor 或 Planner 利用模型漏洞的失败；
8. 资源清单与 checkpoint 哈希。

## 24GB 目标配方

| 项目                |          目标值 |
| ------------------- | --------------: |
| 观察                |     `64×64 RGB` |
| batch               |              16 |
| sequence length     |              32 |
| deterministic state |             256 |
| stochastic state    |              32 |
| imagination horizon |              15 |
| mixed precision     |            可选 |
| peak reserved       | 目标不超过 22GB |

这张表是设计目标，不是实测结果。只有完整训练结束并提交 GPU/CUDA、peak allocated、peak reserved、总时间、曲线和 checkpoint 哈希以后，才可以在状态页改成“24GB 已验证”。

## 替代选题：Mini-MuZero

若更喜欢棋类搜索，可以使用四子棋完成 Mini-MuZero，替代 Dreamer-lite。它必须包含 Representation、Dynamics、Prediction、MCTS 和搜索改进前后的胜率。

不能同时把两个项目各做一半。选择一种，完整提交数据—模型—规划—真实检查。

## 研究问题

最后回答：模型最稳定的失败来自哪一项？

```text
状态没有保存重要信息
动态在长 horizon 漂移
reward/value 误导 Actor
Planner 进入 OOD 区域
真实数据没有覆盖策略访问状态
```

至少提出两种解释，再选一项最小改动带到第 7 章。
