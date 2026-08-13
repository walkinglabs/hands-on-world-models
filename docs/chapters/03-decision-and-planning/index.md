# 第 2 章　决策与规划

这条路线研究一个具体问题：真实试错很贵时，能否先学出一个更短、更可预测的内部世界，再在里面尝试动作？

我们从一张图片出发：把它压成少量信息（潜在状态），用 RSSM 同时记住历史和保留不确定性，再分别用两种方式利用这个世界——PlaNet 在里面实时搜索动作（CEM），Dreamer 把好动作练进 actor-critic（想象 + λ 回报），MuZero 则干脆不重建画面，只学搜索需要的 reward、policy 和 value（MCTS）。

公式在本章是主角：RSSM 的 posterior/prior/KL、CEM 的采样目标、Dreamer 的 λ 回报和 actor-critic 损失、MuZero 的三个网络和 PUCT，都会边讲边列。

## 本章文章

1. [2.1　潜在状态世界模型](./02-01-latent-world-model.md)
2. [2.2　RSSM：记忆与不确定性](./02-02-rssm-training.md)
3. [2.3　PlaNet 与交叉熵方法（CEM）](./02-03-planet-and-cem.md)
4. [2.4　Dreamer：在想象中训练](./02-04-dreamer-imagination.md)
5. [2.5　MuZero 与蒙特卡洛树搜索](./02-05-muzero.md)

## 本章动手实验

- [动手：潜在状态世界模型（RSSM）](/labs/route-a) — A1
- [动手：在想象中规划与行动（PlaNet、Dreamer、MuZero）](/labs/route-a) — A2

主指标是真实环境回报、样本效率和多步漂移。重建图像只是训练手段，不是本路线的最终成绩。
