# 第 3 章　决策与规划

这条路线研究一个具体问题：真实试错很贵时，能否先学出一个更短、更可预测的内部世界，再在里面尝试动作？

我们从一张图片出发：把它压成少量信息（潜在状态），用 RSSM 同时记住历史和保留不确定性，再分别用两种方式利用这个世界——PlaNet 在里面实时搜索动作（CEM），Dreamer 把好动作练进 actor-critic（想象 + λ 回报），MuZero 则干脆不重建画面，只学搜索需要的 reward、policy 和 value（MCTS）。

公式在本章是主角：RSSM 的 posterior/prior/KL、CEM 的采样目标、Dreamer 的 λ 回报和 actor-critic 损失、MuZero 的三个网络和 PUCT，都会边讲边列。

## 本章文章

1. [3.1　潜在状态世界模型](./03-01-latent-world-model.md)
2. [3.2　RSSM：记忆与不确定性](./03-02-rssm-training.md)
3. [3.3　PlaNet 与交叉熵方法（CEM）](./03-03-planet-and-cem.md)
4. [3.4　Dreamer：在想象中训练](./03-04-dreamer-imagination.md)
5. [3.5　MuZero 与蒙特卡洛树搜索](./03-05-muzero.md)
6. [3.6　动手复现 World Models](./03-06-reproduce-world-models.md)

## 本章动手实验

- [动手：潜在状态世界模型（RSSM）](/labs/route-a) — A1
- [动手：在想象中规划与行动（PlaNet、Dreamer、MuZero）](/labs/route-a) — A2
- [动手：复现 World Models 的 V-M-C 管线](/chapters/03-decision-and-planning/03-06-reproduce-world-models) — 3.6，含 CarRacing 训练脚本

主指标是真实环境回报、样本效率和多步漂移。重建图像只是训练手段，不是本路线的最终成绩。

## 参考资料

### 实践博客（5 篇）

1. [worldmodels.github.io (Ha & Schmidhuber)](https://worldmodels.github.io/) —— V-M-C 三件套的交互讲解，是本路线结构的最简原型，配 3.6 的复现实验。
2. [PlaNet 项目主页 (Danijar Hafner)](https://danijar.com/project/planet/) —— 作者本人的页面：RSSM 结构图、CEM 规划可视化与全部代码。
3. [Dreamer 项目主页 (Danijar Hafner)](https://danijar.com/project/dreamer/) —— 想象训练的图示与实验说明，配 3.4。
4. [DreamerV3 项目主页 (Danijar Hafner)](https://danijar.com/project/dreamerv3/) —— 一套超参打通 150+ 任务的可复现配方，含实验细节。
5. [MuZero: Mastering Go, chess, shogi and Atari (DeepMind, 2020)](https://deepmind.google/discover/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules/) —— 官方博客，把隐式模型 + MCTS 讲给非专业读者，配 3.5。

### 原始论文（5 篇）

1. [Learning Latent Dynamics for Planning from Pixels: PlaNet (Hafner et al., 2019)](https://arxiv.org/abs/1811.04551) —— RSSM 与 CEM 规划的原始论文，本章 3.2、3.3 的直接来源。
2. [Dream to Control: Learning Behaviors by Latent Imagination (Hafner et al., 2020)](https://arxiv.org/abs/1912.01603) —— Dreamer 原始论文：在想象中反向传播训练 Actor-Critic。
3. [Mastering Diverse Domains through World Models: DreamerV3 (Hafner et al., 2023)](https://arxiv.org/abs/2301.04104) —— 一套超参打通 150+ 任务的工程报告，是“可复现世界模型”的标杆。
4. [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (Schrittwieser et al., 2020)](https://arxiv.org/abs/1911.08265) —— MuZero 原始论文：只靠学到的隐式模型规划，不需要环境规则。
5. [Model-Based Reinforcement Learning in Atari without Dreaming (Kaiser et al., 2020)](https://arxiv.org/abs/1903.00374) —— SimPLe：用 10 万帧真实数据训练视频世界模型再训练策略，展示了小预算下的完整评测协议。
