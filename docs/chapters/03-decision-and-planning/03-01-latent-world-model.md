# 2.1　潜在状态世界模型

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

我们不要求 latent 解码出高清可观看的视频。如果最终使用者是人或游戏玩家，第 3 章的互动视频路线更合适。本路线的 latent 是给规划器和 actor-critic 用的，可解释性和画面保真度都不是首要指标。

## 小结

- [ ] 图像 embedding 只表示当前一帧，动态状态还要结合历史与动作。
- [ ] decoder、reward head、continue head 从不同角度约束同一个状态。
- [ ] latent 是否有用，由 reward、continue、多步预测和真实控制共同检查。

下一篇我们拆开 $s_t$，看 RSSM 怎样把「长期记忆」和「当前不确定性」分开存放。动手实现见 [A1：潜在状态世界模型（RSSM）](/labs/route-a)。
