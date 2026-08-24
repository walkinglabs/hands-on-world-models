# 4.1　潜在状态世界模型

> **第 4 章 · 决策与规划**
>
> 这条路线研究一个具体问题：真实试错很贵时，能否先学出一个更短、更可预测的内部世界，再在里面尝试动作？
>
> 我们从一张图片出发：把它压成少量信息（潜在状态），用 RSSM 同时记住历史和保留不确定性，再分别用三种方式利用这个世界——PlaNet 在里面实时搜索动作（CEM），Dreamer 把好动作练进 actor-critic（想象 + λ 回报），MuZero 则干脆不重建画面，只学搜索需要的 reward、policy 和 value（MCTS）。
>
> 主指标是真实环境回报、样本效率和多步漂移。重建图像只是训练手段，不是本路线的最终成绩。
>
> 👉 本章实验：[动手：学出一个潜在世界（RSSM）](/chapters/04-decision-and-planning/07-decision-and-planning)、[在想象中行动（PlaNet、Dreamer、MuZero）](/chapters/04-decision-and-planning/07-decision-and-planning)

CRAFTER 里agent要砍树、挖矿、喝水。一帧画面有上万个像素，但真正决定下一步结果的，只有「手里有没有斧头」「脚下还有几格木头」「面前有没有怪物」这类少量信息。

本章要解决的问题是：当真实试错很贵时，能否先学一个更短、更可预测的内部世界，再在里面尝试动作。第一步，就是把图片压成这种「少量信息」。

## 从像素到 embedding

所谓 encoder，就是一个把高维观察压成低维向量的网络。PixelWorld 的每帧是 `16×16×3` 图片，共 $16\times16\times3=768$ 个数值。CNN encoder 把它压成 embedding $e_t\in\mathbb{R}^{D}$：

$$
e_t = f_\theta(o_t),\qquad o_t\in\mathbb{R}^{768},\ e_t\in\mathbb{R}^{D}
$$

`D` 远小于 768，比如 32 维。embedding 只描述当前这一帧，不会自动记住历史。两段视频末帧相同、运动方向相反，它们的 embedding 也几乎相同。

## 动态状态要带上历史和动作

规划器真正需要的不是「这一帧长什么样」，而是「给定动作，下一步会怎样」。我们把 embedding、上一状态、上一动作一起交给状态模型：

$$
s_t = g_\phi(s_{t-1},\, e_t,\, a_{t-1})
$$

这个 $s_t$ 才是 reward head、value head、decoder 和 actor 共用的动态状态。它把「现在看到的」和「过去记得的、上一步做了什么」合在一起。

至于状态 $s_t$ 内部长什么样，是下一篇 RSSM 的主题。这里先记住一句话：**动态状态 ≠ 图像特征**。

## 观察模型：用 decoder 提供密集训练信号

只压不还原，embedding 可能丢掉很多有用信息。decoder 把 latent 还原回观察，提供像素级别的训练信号：

$$
\hat o_t = d_\psi(s_t),\qquad \mathcal{L}_{\text{obs}} = \lVert \hat o_t - o_t\rVert_2^2
$$

decoder 不是本路线的最终产品。即使重建画面很好，latent 仍可能漏掉奖励、终止或动作造成的细微变化。

## 预测 reward 与 continue

状态对控制有没有用，要看它能否解释动作的结果。我们在状态上接两个小 head：

$$
\hat r_t = R_\rho(s_t),\qquad \hat c_t = C_\kappa(s_t)
$$

$\hat r_t$ 预测这一步的 reward，$\hat c_t$ 预测 episode 是否继续（continue 概率）。它们和 decoder 一起，从不同角度约束同一个 $s_t$。

一个状态同时能重建画面、预测 reward、预测是否结束，才说明它把决策需要的信息都留下了。

## 一步预测的最朴素基线

最简单的 latent 预测基线是「直接复制当前 latent」，或者用线性层预测 $\hat s_{t+1}=A s_t$。PixelWorld 大多时刻方块静止，这类基线可能得到不低的分数。

真正的检查是**固定历史、替换动作，再连续 rollout**。模型若只学到惯性（不管做什么动作方块都不动），就不会随动作产生正确的分叉，规划器也无从利用。

## 本路线暂时不追求什么

我们不要求 latent 解码出高清可观看的视频。如果最终使用者是人或游戏玩家，第 4 章的互动视频路线更合适。本路线的 latent 是给规划器和 actor-critic 用的，可解释性和画面保真度都不是首要指标。

## 小结

- [ ] 图像 embedding 只表示当前一帧，动态状态还要结合历史与动作。
- [ ] decoder、reward head、continue head 从不同角度约束同一个状态。
- [ ] latent 是否有用，由 reward、continue、多步预测和真实控制共同检查。

下一篇我们拆开 $s_t$，看 RSSM 怎样把「长期记忆」和「当前不确定性」分开存放。动手实现见 [4.7 动手：Dreamer 的简化实现](/chapters/04-decision-and-planning/07-decision-and-planning) 的第一份 Notebook「学出一个潜在世界」。

---

## 参考资料

### 实践博客

1. [worldmodels.github.io (Ha & Schmidhuber)](https://worldmodels.github.io/) —— V-M-C 三件套的交互讲解，是本路线结构的最简原型，配 4.6 的复现实验。
2. [PlaNet 项目主页 (Danijar Hafner)](https://danijar.com/project/planet/) —— 作者本人的页面：RSSM 结构图、CEM 规划可视化与全部代码。
3. [Dreamer 项目主页 (Danijar Hafner)](https://danijar.com/project/dreamer/) —— 想象训练的图示与实验说明，配 4.4。
4. [DreamerV3 项目主页 (Danijar Hafner)](https://danijar.com/project/dreamerv3/) —— 一套超参打通 150+ 任务的可复现配方，含实验细节。
5. [MuZero: Mastering Go, chess, shogi and Atari (DeepMind, 2020)](https://deepmind.google/discover/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules/) —— 官方博客，把隐式模型 + MCTS 讲给非专业读者，配 4.5。

### 经典文献

1. [Learning Latent Dynamics for Planning from Pixels: PlaNet (Hafner et al., 2019)](https://arxiv.org/abs/1811.04551) —— RSSM 与 CEM 规划的原始论文，本章 4.2、4.3 的直接来源。
2. [Dream to Control: Learning Behaviors by Latent Imagination (Hafner et al., 2020)](https://arxiv.org/abs/1912.01603) —— Dreamer 原始论文：在想象中反向传播训练 Actor-Critic。
3. [Mastering Diverse Domains through World Models: DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104) —— 一套超参打通 150+ 任务的工程报告，是“可复现世界模型”的标杆。
4. [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (Schrittwieser et al., 2020)](https://arxiv.org/abs/1911.08265) —— MuZero 原始论文：只靠学到的隐式模型规划，不需要环境规则。
5. [Model-Based Reinforcement Learning in Atari without Dreaming (Kaiser et al., 2020)](https://arxiv.org/abs/1903.00374) —— SimPLe：用 10 万帧真实数据训练视频世界模型再训练策略，展示了小预算下的完整评测协议。
