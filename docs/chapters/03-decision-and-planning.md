# 第 3 章　决策与规划：怎样少在现实中试错

PA0 的模型在一条短直线上工作。状态只有一个整数，因此可以直接统计每个动作的结果。

现在把观察换成图片。小车的速度藏在历史里，地面偶尔打滑，动作又是连续的。状态表迅速变大，穷举动作也变得困难。

这条路线只关心一个主要问题：真实环境中的试错很贵，能否先从经历中学出一个较短的内部世界，再在其中练习动作？

## 3.1 这条路线交出什么

模型主要预测：

```text
下一 latent 状态
下一步 reward
任务是否继续
```

Decoder 可以帮助训练 latent，却不是最终目的。真正的结果要由真实环境 return、成功率和样本效率检查。

动作在这里有两个角色。它首先是 dynamics 的输入：模型根据动作推测未来。它也由 CEM 或 Actor 提出：前者现场搜索，后者直接输出。

这条路线暂时不追求高清画面。若目标是让人看见可交互视频，第 4 章会使用另一种输出与评价。

## 3.2 从图片到 latent

PixelWorld 每张图片有 `16×16×3=768` 个数。我们先用两层 CNN 把它编码成 64 维向量。

```text
observation [B,16,16,3]
→ CNN Encoder
→ embedding [B,64]
```

Embedding 还不是动态状态。单张图片只说明方块在哪里，没有说明它从哪边来。RSSM 会把当前 embedding 与历史、动作结合。

为什么不直接在像素上做所有预测？一个原因是计算量。另一个原因是任务可能只关心位置、速度、目标和碰撞，不需要每个背景像素。

但 latent 也可能压掉重要信息。我们不能只看重建图好不好，还要检查 reward、动作敏感性和真实任务。

## 3.3 RSSM 怎样从记忆走到预测

F1 用相邻位置差手工留下速度。RSSM 要从数据中学习一份可以长期更新的状态。

它把内部状态分成两部分：

```text
h_t：确定状态，由 GRU 更新
z_t：随机状态，描述当前仍不确定的部分
```

每一步先把上一随机状态与动作交给 GRU：

```text
h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
```

只有历史和动作时，模型给出 prior：

```text
p(z_t | h_t)
```

训练时还能看到当前图片。Encoder 的 embedding 与 `h_t` 一起给出 posterior：

```text
q(z_t | h_t, embed(o_t))
```

Posterior 可以利用真实观察修正当前状态。Prior 则必须在没有未来观察时独立想象。

若二者完全无关，模型在训练时知道真实图片，进入想象以后却会立刻迷路。因此要用 KL 散度让 prior 学着接近 posterior。

### 为什么不能只用确定 GRU

同一辆小车在湿滑地面向右加速，可能正常前进，也可能停留或打滑。确定网络若用均方误差预测一个结果，容易给出多个未来的平均。

随机状态允许模型表示多种可能。它不能自动保证概率校准，却至少让“未来不唯一”进入模型接口。

## 3.4 世界模型怎样训练

教学版模型从 Replay Buffer 采样 `[B,T]` 连续片段。动作 `a_t` 用于预测 `o_{t+1}` 对应的状态。

模型有四组训练信号。

### 观察重建

Decoder 从 posterior feature 重建观察：

```text
[h_t, z_t] → reconstructed observation
```

重建迫使 latent 保留视觉信息。它也可能浪费容量在背景纹理，因此不是所有世界模型都需要 Decoder。

### Reward 预测

Reward head 让 latent 保存任务相关变化。若小物体决定奖励，重建损失却更关心大背景，reward head 可以提供另一种压力。

### Continue 预测

模型预测任务是否会继续。想象 rollout 中，continue 可以让终止后的奖励不再传播。

### Prior 与 posterior 的 KL

KL 让只看历史和动作的 prior 接近看见真实观察的 posterior。

教学版总损失是：

```text
reconstruction + reward + continue + 0.1 × KL
```

这个系数只是小实验设置，不是 DreamerV3 官方配方。

### Free Bits 为什么出现

若 Decoder 只靠确定状态就能重建，模型可能让 posterior 与 prior 完全相同，不再使用随机状态。KL 很低，却失去了不确定性表示。

Free Bits 让一小段 KL 不被继续惩罚：

```text
KL loss = max(free_nats, KL)
```

它不是让 KL 越大越好，而是防止优化器为了节省一点 KL，立即关掉随机通道。

## 3.5 一步训练以后，先做多步检查

训练 loss 下降，只说明模型在采样片段上更好地完成训练目标。

我们还要从一个真实 posterior 出发，不再给后续观察，只使用 prior 连续预测：

```text
真实历史 → posterior state
→ action_1 → prior state_1
→ action_2 → prior state_2
→ ...
```

检查至少包括：

- horizon 为 1、5、15 时的重建或状态误差；
- 同一起点替换动作，latent 与 reward 是否改变；
- 陌生速度或颜色下，误差是否更大；
- prior 的不确定性是否与实际错误大致对应。

若模型从第 6 步开始明显漂移，Planner 就不应在没有惩罚和重规划的情况下相信 50 步未来。

## 3.6 PlaNet：每次现场搜索

现在假设 RSSM 已经能够短期推演。我们还没有动作。

PlaNet 使用 CEM 搜索动作序列：

1. 从较宽分布采样许多动作序列；
2. 在 RSSM 中 rollout；
3. 累加 reward 与终止信息；
4. 保留较好的若干序列；
5. 用它们更新采样分布；
6. 重复几轮，只执行最佳序列第一步。

这仍然是 MPC。执行后读入真实观察，posterior 修正状态，再重新搜索。

PlaNet 的优点是目标改变时可以直接换 reward 再规划。代价是每次行动都要搜索，实时性受 population、horizon 和迭代轮数影响。

## 3.7 Dreamer：把好动作练成 Actor

若环境较稳定，每次从头 CEM 会重复大量工作。

Dreamer 从真实数据得到 posterior state，冻结或固定世界模型，然后让 Actor 在 latent 中选择动作：

```text
state_t → Actor → action_t
state_t + action_t → RSSM prior → state_{t+1}
state_{t+1} → reward / continue
```

重复 `H` 步，得到一段 imagination trajectory。

Actor 的目标是提高想象回报。Critic 学习估计每个 imagined state 的未来价值。这样，搜索得到的经验被压进一个可以快速输出动作的网络。

### TD-λ 为什么出现

只用一步 TD，依赖 Critic 的 bootstrap，偏差可能较大。一直累加到 horizon 末端，方差和模型误差又会增大。

TD-λ 在短期真实 reward 与后续 value 之间做加权组合。`λ` 越接近 1，越依赖较长回报；越接近 0，越依赖一步 bootstrap。

教学版 A2 会从后往前计算 TD-λ target，并检查 shape 与有限数值。完整 Dreamer 还会处理梯度如何穿过 dynamics、Actor 分布和 value normalization。

## 3.8 一轮 Dreamer 训练怎样循环

将部件接起来以后，一轮训练包含七步。

### 第一步：收集真实数据

用当前 Actor 或随机策略在环境中运行 episode，保存 observation、action、reward 与 done 到 Replay Buffer。

Dreamer 是 off-policy，可以重复使用历史数据。数据太旧时仍可能与当前策略访问区域不同，因此 Buffer 组成也要监控。

### 第二步：训练世界模型

从 Buffer 采样连续片段，更新 Encoder、RSSM、Decoder、reward 与 continue heads。

### 第三步：选择真实起点

从 posterior sequence 中取一些状态，作为想象起点。这样，imagination 从数据附近开始，而不是任意 latent。

### 第四步：在梦境中 rollout

Actor 选择动作，RSSM prior 预测下一状态，heads 预测 reward 与 continue。教学 smoke 使用 5 步，完整配方常使用更长但仍有限的 horizon。

### 第五步：训练 Critic

用 TD-λ target 拟合 imagined state 的价值。

### 第六步：训练 Actor

提高想象中的预期回报。离散动作可用分布梯度或直通方式，连续动作常使用 tanh 变换的正态分布。

### 第七步：回到真实环境

Actor 执行动作，收集新数据，再回到第一步。

这是一条持续循环，不是先把 world model 训练到“完全正确”，再永远不变地交给策略。

## 3.9 DreamerV3 的工程细节放在哪里

DreamerV3 希望同一组超参数适应不同奖励尺度和任务。几项常见技巧各自对应具体失败。

Symlog 把跨度很大的 reward 与 value 压到较温和的尺度。

Twohot 不把标量只回归成一个数，而是在相邻两个 bin 上分配概率，提供更平滑的分类式训练信号。

Free Bits 防止随机 latent 被 KL 压到不用。

Unimix 把大部分模型概率与少量均匀分布混合，防止离散分布过早变成绝对 0 或 1。

这些技巧会放在 A2 末尾做小消融或数值检查，不各开一份 Notebook。课程首先要知道它们修复哪种失败，再阅读官方实现。

## 3.10 MuZero 为什么放在同一章作为选修

Dreamer 通常让 latent 支持 observation、reward 和 continue 预测。MuZero 做出另一种选择：不要求重建原始观察，只学习树搜索需要的信息。

它有三个网络：

```text
Representation：观察 → 初始隐状态
Dynamics：隐状态 + 动作 → 下一隐状态 + reward
Prediction：隐状态 → policy + value
```

MCTS 把动作当作树的边，在隐空间展开未来。搜索得到的改进 policy 与 value 反过来训练网络。

这是一种 value-equivalent 思路：内部状态不必画回每个像素，只要保留规划所需的信息。

代价也很清楚。状态可能只适合当前 reward、policy 和搜索目标，换一个任务时不一定保留足够信息；MCTS 也更适合离散动作。

课程将 Mini-MuZero 四子棋作为 A2 论文拔高和 PA1 替代选题，不与 Dreamer-lite 一起设为必做。

## 3.11 两份 Notebook

### A1：学习一个 latent world

`A1-learn-a-latent-world.ipynb` 使用项目内 PixelWorld：

```text
CNN Encoder
→ RSSM prior/posterior
→ reconstruction/reward/continue/KL
→ 训练 smoke
→ 多步 prior rollout
```

CPU smoke 使用 4 个 episode、8 步序列和小网络。它只验证接口、梯度和 loss 下降。

### A2：在想象中行动

`A2-act-in-imagination.ipynb` 从真实 posterior state 出发：

```text
Actor 采样动作
→ RSSM 想象 5 步
→ reward/continue
→ Critic value
→ TD-λ target
→ Actor 与 Critic 更新
```

A2 会同时回顾 PlaNet/CEM 与 Dreamer/Actor 的区别，但不额外增加第三份 Notebook。

## 3.12 PA1-A 与资源边界

PA1-A 要把 Replay Buffer、RSSM、imagination 和 Actor-Critic 接成 Dreamer-lite。

数据台阶是：

```text
PixelWorld CPU smoke
→ PixelWorld GPU 小训练
→ DMC Cartpole 小配置迁移
```

主要指标：

- 真实环境 return 与成功率；
- 达到指定 return 使用的真实交互量；
- horizon 增长时的漂移；
- Planner 或 Actor 是否利用模型漏洞；
- 推理延迟与峰值显存。

24GB 目标配方只是一份待实测设计：`64×64` 观察、batch 16、sequence 32、deter 256、stoch 32、imagination horizon 15。发布时仍需提交真实 GPU、peak reserved、时间、曲线和 checkpoint 哈希。

## 3.13 这条路线仍然会失败

latent 可能重建得很好，却没有保留一个决定奖励的小物体。

短 horizon 内 reward 准确，长 horizon 却漂移。Actor 可能专门走向模型过于乐观的区域。

世界模型与 Actor 同时更新时，Actor 面对的梦境分布也不断变化。

换目标以后，Dynamics 理论上可以复用，但若 latent 训练时只保留旧 reward 需要的信息，新任务仍可能失败。

这些失败不应该被最好的一条 return 曲线遮住。PA1-A 至少提交一组固定 seed 的模型漏洞，并在第 8 章判断应该补数据、改状态、改目标还是限制规划。

## 小结

- [ ] RSSM 的 prior 只看历史和动作，posterior 还看当前观察。
- [ ] reconstruction、reward、continue 与 KL 分别约束 latent 的不同部分。
- [ ] PlaNet 用 CEM 现场搜索，Dreamer 在 imagination 中训练 Actor。
- [ ] Critic 与 TD-λ 让短 imagined horizon 仍能估计较远价值。
- [ ] MuZero 不重建像素，只保留 reward、policy 与 value 所需信息。
- [ ] CPU smoke 证明接口可运行，不证明 DreamerV3 规模结果或 24GB 完整训练。

若更关心“未来画面是否听从按键”，下一章会直接预测可观看的世界，而不是只在 latent 中规划。
