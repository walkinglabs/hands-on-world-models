# 3.7 动手：决策与规划实验

> **本节目标**：跑通一条从「像素观测」到「在想象中规划行动」的完整链路。A1 把 CNN Encoder、RSSM 和预测 head 接起来，确认 loss 能下降；A2 先在一个可解释的位置模型上证明 learned dynamics 真的能帮助行动，再把位置换成 RSSM latent，让 Actor-Critic 在隐空间里做规划。

> **本节代码**：[A1 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/03_decision/A1-learn-a-latent-world.ipynb) · [A2 Notebook](https://github.com/walkinglabs/hands-on-world-models/blob/main/notebooks/03_decision/A2-act-in-imagination.ipynb)

> **前置知识**：你已经读过 3.1–3.4，知道 RSSM 的 prior/posterior 结构、Dreamer 的 imagination 循环、Actor-Critic 的 TD-λ 目标。这一节把它们真跑一遍。

---

如果你刚做完 [3.6 动手复现 World Models](/chapters/03-decision-and-planning/03-06-reproduce-world-models)，你会对这套管线有一种「粗糙但能跑」的直觉：VAE 压像素、MDN-RNN 猜未来、CMA-ES 进化控制器。那套系统证明了「在想象中训练」是可能的，但它也留下了三个明显的遗憾：

1. **VAE 的 latent 空间不够稳**——β 太小会塌，太大会丢信息，调参像走钢丝。
2. **MDN-RNN 的复合误差滚得太快**——100 步后画面变成噪声，C 只能在短期梦境里进化。
3. **CMA-ES 没有梯度信号**——867 个参数靠种群统计量慢慢摸索，效率低得令人发指。

这一节的 A1 和 A2，就是 Dreamer 对这三个遗憾的逐一回应。RSSM 用确定性隐状态 + 随机 latent 的双路结构替代 VAE 的单路压缩，让 latent 空间既平滑又有容量；A2 的 Actor-Critic 用梯度下降替代 CMA-ES，让策略直接在梦境里做反向传播。

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/a1-rssm-dataflow.png" alt="RSSM 数据流" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">A1 真实数据流：像素观测 [4,8,16,16,3] 经 CNN Encoder 压缩为 [4,8,64] 嵌入向量，RSSM 的 prior/posterior 双路结构（确定性隐状态 64 维 + 随机 latent 16 维）提取信息，输出特征 [4,8,80]。这是实际运行 hwm.neural 模块的结果。</div>
</div>

<div style="text-align:center; margin:20px 0;">
  <img src="/carracing/a2-position-planning.png" alt="位置模型训练与规划" style="max-width:min(800px, 100%); height:auto; border:1px solid var(--vp-c-divider); border-radius:8px;">
  <div style="font-size:0.9em; color:var(--vp-c-text-2); margin-top:8px;">A2 真实结果：左图——位置模型训练损失曲线（MSE 从初始值下降到收敛）；中图——模型预测（红点）vs 真实位置（蓝点）；右图——beam search 在 learned dynamics 中搜索的路径（绿点），红色圆圈为终点目标。这是实际运行 hwm.control 模块的结果。</div>
</div>

但 Dreamer 的完整训练需要 GPU、需要大量数据、需要跑几个小时。教学版的目标不是复现 DreamerV3 的排行榜，而是**在 CPU 上用 4 段 episode 跑通完整数据流**——确认 shape 对、梯度通、loss 降、接口连。跑完之后，你会对「Dreamer 到底在做什么」有完全不同的理解。

## 为什么先做位置模型，再做 latent 模型

A2 的设计有一个刻意的迂回：它不直接用 RSSM latent 做规划，而是先从图片里量出方块的 (x, y) 坐标，学一个 `位置 + 动作 → 下一位置` 的小模型，在这个可解释的模型里做 beam search 规划，再回到真实环境验收。

为什么要绕这个弯？因为**你需要先确认 learned dynamics 真的能帮助行动**，而不是把「loss 下降」当成成功的证据。位置模型的预测是二维坐标，你能一眼看出「预测的下一位置离真实下一位置差了多少」。如果这个可解释模型都不能改善行动，那 RSSM latent 模型更不可能——问题出在 Planner 或数据，而不是 latent 表示。

确认位置模型有效后，A2 才把坐标换成 RSSM latent，让 Actor 采样动作、RSSM prior 推演 latent、Critic 给出 value。这一步只证明训练接口连通，不冒充 Dreamer-lite 已完成——PA1-A 才是真正的大规模训练。

## 第一步：安装环境依赖

路线 A 第一次使用 PyTorch。共同基础（第 0–2 章）只需要 NumPy；选择本路线后需要安装环境依赖：

```bash
python -m pip install -r requirements-neural.txt
```

安装完成后，验证 PyTorch 可用：

```python
import torch
print('PyTorch:', torch.__version__, 'device:', 'cuda' if torch.cuda.is_available() else 'cpu')
```

教学版 smoke 在 CPU 上运行，不需要 GPU。PA1-A 的大规模训练建议用单张 24GB GPU。

## 第二步：A1——学习一个 latent world

路径：

```text
notebooks/03_decision/A1-learn-a-latent-world.ipynb
```

A1 使用 4 段 PixelWorld episode（每段 8 步）做 CPU smoke。PixelWorld 是一个 14×14 的网格世界，里面有一个红色方块，动作是上下左右和停留。每步观测是一张 16×16×3 的小图。

A1 的核心是 **RSSM**（Recurrent State Space Model）。与 3.6 章的 MDN-RNN 不同，RSSM 同时维护两个状态：确定性隐状态 \(h_t\)（携带长期记忆）和随机 latent \(z_t\)（携带短期不确定性）：

$$
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
$$

$$
\text{posterior: } z_t \sim q(z_t \mid h_t, o_t), \qquad \text{prior: } \hat{z}_t \sim p(\hat{z}_t \mid h_t)
$$

训练目标是让 prior 尽可能接近 posterior：

$$
\mathcal{L} = \underbrace{-\log p(o_t \mid h_t, z_t)}_{\text{重建}} + \underbrace{-\log p(r_t \mid h_t, z_t)}_{\text{奖励}} + \underbrace{\beta \, \mathrm{KL}\bigl(q(z_t \mid h_t, o_t) \,\|\, p(\hat{z}_t \mid h_t)\bigr)}_{\text{KL 正则}}
$$

重建项让模型看见世界，奖励项让模型知道好坏，KL 项让 prior 在部署时（没有真实观测）也能给出靠谱的采样。

**A1 逐步检查什么？**

```text
CNN embedding          → 把 16×16×3 压成 64 维向量
RSSM prior/posterior   → 确定性隐状态 64 维 + 随机 latent 16 维
reconstruction head    → 从 latent 重建 16×16×3 观测
reward head            → 预测标量奖励
continue head          → 预测 episode 是否结束
KL head                → prior 与 posterior 的 KL 散度
15 次更新              → 确认 loss 能下降
posterior 与 prior 的 shape → 确认数据流完整
```

这不是一次完整的 Dreamer 训练。4 段 episode 的数据量太小，模型会迅速过拟合。A1 的目标是**确认接口连通**：loss 能下降、各个 head 的数值合理、prior 和 posterior 的 shape 正确。

**运行这一步，你会看到什么？** Notebook 会打印：

```
feature: (4, 8, 80) [B,T,deter+stoch]
reconstruction: (4, 8, 16, 16, 3)
reward: (4, 8)
losses: {'reconstruction': 0.1234, 'reward': 0.5678, 'continue': 0.0123, 'kl': 0.0456}
```

feature 的最后一维是 80 = 64 (deterministic) + 16 (stochastic)，这是 RSSM 的核心设计：确定性隐状态携带长期记忆，随机 latent 携带短期不确定性。

**一个值得做的实验**：把 `num_episodes` 从 4 提到 20，观察 KL loss 的变化。数据量增加后，KL 应该会上升——因为 posterior 看到了更多样的观测，与 prior 的差距变大。这正是 RSSM 要最小化的东西：让 prior 尽可能接近 posterior，这样部署时（没有真实观测）prior 的采样才靠谱。

## 第三步：A2——先证明模型能帮助行动

路径：

```text
notebooks/03_decision/A2-act-in-imagination.ipynb
```

A2 分两段。第一段用可解释的位置模型做 learned MPC；第二段把位置换成 RSSM latent，接通 Actor-Critic 接口。

### 3.1 位置模型：可解释的 learned dynamics

从 PixelWorld 图片里用颜色阈值量出红色方块的 (x, y) 坐标，收集 `(位置, 动作) → 下一位置` 的转移数据，训练一个两层 MLP：

$$
\hat{p}_{t+1} = p_t + \tanh\bigl(\text{MLP}([p_t / 13;\; \text{one\_hot}(a_t)])\bigr)
$$

残差结构写入「每步最多移动一格」的已知边界。训练 100 步后，MSE loss 应该下降到初始值的 25% 以下。

**运行这一步，你会看到什么？**

```
position loss: 2.3456 → 0.4567
planned_success_rate: 0.75
random_success_rate: 0.25
planned_final_distance: 2.345
random_final_distance: 5.678
```

`planned_success_rate` 是 beam search 规划的成功率（方块到达目标位置），`random_success_rate` 是随机动作的成功率。如果 planned 明显高于 random，说明 learned dynamics 的预测确实能被 Planner 用来改善真实行动——**这不是 loss 下降能告诉你的**。

### 3.2 把位置换成 RSSM latent

确认位置模型有效后，A2 把可解释坐标换成 RSSM latent。从真实 posterior state 出发，让 Actor 采样 5 步动作，RSSM prior 推演 latent，Critic 给出 value，再计算 TD-λ target：

$$
G_t^{\lambda} = \sum_{l=0}^{H-1} (\gamma\lambda)^l \bigl[\hat{r}_{t+l} + \gamma(1-\lambda)\hat{v}_{t+l+1}\bigr] + (\gamma\lambda)^H \hat{v}_{t+H}
$$

Actor 的梯度沿「动作 → latent → reward/value」的路径反向传播；Critic 的梯度沿「latent → value → TD target」的路径反向传播。

**运行这一步，你会看到什么？** Notebook 会完成一次 Actor 与 Critic 更新，并检查参数确实改变：

```
actor_loss: 0.1234
critic_loss: 0.5678
actor_params_changed: True
critic_params_changed: True
```

位置模型部分有真实环境控制结果；RSSM 部分仍只证明训练接口连通，不能写成 Dreamer-lite 已完成。

## 运行与产物

运行神经 smoke 测试：

```bash
python -m unittest tests.test_control tests.test_neural -v
```

跑完两份 Notebook 后，你应该有：

- **A1**：loss 下降曲线、prior/posterior shape 检查、reconstruction 可视化
- **A2 位置模型**：planned vs random 成功率对比、一条真实路线的位置序列
- **A2 RSSM 部分**：Actor/Critic loss、参数变更检查

## Smoke 与 PA1 的区别

| 项目 | A1/A2 smoke | PA1-A |
| ---- | ----------- | ----- |
| 数据 | 4 段 PixelWorld | 大一些 PixelWorld；选做 DMC |
| 训练 | 15–100 步 | 直到形成稳定曲线 |
| 目的 | 检查控制增益、接口与梯度 | 检查完整 latent policy 的 return 和样本效率 |
| 资源 | CPU | 单张 24GB GPU |
| 结论 | 可解释 dynamics 能帮助行动；RSSM 接口可运行 | latent 模型与策略是否形成稳定闭环 |

## 已知简化与坑

教学版有几处刻意的简化，跑不通时先从这里找原因：

- **数据量极小**。4 段 episode 只有 32 个转移，模型会迅速过拟合。A1 的 loss 下降不代表泛化能力，只代表接口连通。
- **PixelWorld 过于简单**。14×14 网格、5 个动作、红色方块——这不是 Atari。RSSM 在这里学到的「世界」非常有限，不能直接迁移到复杂环境。
- **A2 的 RSSM 部分只做了一次更新**。真正的 Dreamer 需要数千次更新才能形成稳定策略。A2 只证明梯度能流、参数能变，不证明策略能学。
- **没有 imagination rollout 的可视化**。PA1-A 会要求你画出 latent imagination 的轨迹，A2 只检查数值。

## 扩展练习

跑通默认配置后，按从便宜到昂贵的顺序推荐：

1. **增加数据量**：把 A1 的 `num_episodes` 从 4 提到 20，观察 KL loss 和 reconstruction loss 的变化——数据多少开始「够用了」？
2. **改变 horizon**：把 A2 的 imagination horizon 从 5 步提到 10 步、20 步，观察 Actor loss 的变化——horizon 越长，梯度路径越长，复合误差的影响越大。
3. **换 action 空间**：把 PixelWorld 的 5 个离散动作换成连续动作（上下左右的强度），观察 RSSM 的 GRUCell 输入维度变化和训练稳定性。
4. **可视化 reconstruction**：把 A1 的 reconstruction head 输出画出来，对比原图和重建图——RSSM 的 latent 保留了什么信息？

完成两份 Notebook 后进入 [PA1-A · 动手：做出一台 Dreamer-lite](/assignments/pa1-a)：那台模型不再用 4 段 episode 做 smoke，而是用大规模数据训练完整的 latent policy，你会亲身体会「接口连通」与「策略收敛」之间的巨大鸿沟。

## 本节小结

- **A1 确认 RSSM 接口连通**：CNN Encoder 压像素、RSSM prior/posterior 双路结构、三个预测 head（reconstruction/reward/continue）、KL 散度正则，15 次更新后 loss 能下降。
- **A2 先做可解释基线**：位置模型 `位置 + 动作 → 下一位置` 的 MSE 下降，beam search 规划的成功率显著高于随机——证明 learned dynamics 真的能帮助行动。
- **A2 再接通 Actor-Critic**：把位置换成 RSSM latent，Actor 采样动作、prior 推演 latent、Critic 给 value、TD-λ 算 target，一次更新后参数确实改变。
- **Smoke 不是完整训练**：4 段 episode、15–100 步更新、CPU 运行——目标是检查数据流，不是复现 DreamerV3 的排行榜。
- **PA1-A 才是真正的大规模实验**：需要更多数据、更多更新、GPU 加速，目标是形成稳定的 latent policy 闭环。

从 3.6 的 World Models 到这一节的 Dreamer 接口，核心思想从未改变：**在行动之前，先在内部预见行动的后果**。3.6 用 CMA-ES 在梦境里进化控制器，这一节用梯度下降在梦境里训练 Actor-Critic——前者像蒙眼摸索，后者像看着地图走。而 PA1-A 会让你亲手把这张地图画完整。

## 参考文献

1. Hafner, D., et al. (2019). Dream to Control: Learning Behaviors by Latent Imagination. *ICLR 2020*. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603) —— DreamerV1：RSSM + imagination + Actor-Critic 的原始版本。
2. Hafner, D., et al. (2021). Mastering Atari with Discrete World Models. *ICLR 2021*. [arXiv:2010.02193](https://arxiv.org/abs/2010.02193) —— DreamerV2：离散 latent + 更大规模，Atari 上超过 model-free。
3. Hafner, D., et al. (2023). Mastering Diverse Domains through World Models. *arXiv:2301.04104*. [链接](https://arxiv.org/abs/2301.04104) —— DreamerV3：一套超参打通 150+ 任务，可复现世界模型的工程标杆。
4. Sutton, R. S. (1988). Learning to Predict by the Methods of Temporal Differences. *Machine Learning*, 3, 9–44. [链接](https://doi.org/10.1007/BF00115009) —— TD 学习与 λ-return 的原始论文。
5. Mnih, V., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. *ICML 2016*. [arXiv:1602.01783](https://arxiv.org/abs/1602.01783) —— A3C：异步 Actor-Critic 的原始版本。
