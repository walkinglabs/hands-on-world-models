# 第 1 章　世界模型的基本问题

这一章暂时不用神经网络。我们先问一张照片遗漏了什么，再用连续观察形成内部状态，学习世界怎样变化。只有任务需要行动时，动作、规划器和策略才继续加入。

六篇小章按顺序阅读。前五篇每篇只增加一个困难，最后一篇动手把它们接起来。公式夹在讲解里，遇到一处看一处，不必先把符号全记下来。

## 本章文章

1. [1.1 观察、状态与历史](./01-current-observation.md)：区分观察与状态，说明历史为何出现。
2. [1.2 动作条件预测](./02-action-conditioned-future.md)：从普通续写走到动作条件预测。
3. [1.3 多步预测与规划](./03-rollout-planning-policy.md)：在网格世界中区分世界模型、规划器和策略。
4. [1.4 从经历学习动态](./04-learned-dynamics.md)：从连续经历学习概率转移，并检查模型错误。
5. [1.5 经典世界模型（World Models、PlaNet、Dreamer）](./05-classic-world-models.md)：把我们得到的部件与 V–M–C、PlaNet 和 Dreamer 对照。
6. [1.6 动手：从零重新发明世界模型](./06-invent-a-world-model.md)：把五篇文章放进同一份 Notebook。

## 学完以后

我们应当能画出下面三个接口，并说明一个简单任务为什么可能只需要其中一两个：

```text
世界模型 f：  当前状态 + 候选动作 → 可能的未来      s_{t+1} = f(s_t, a_t)
规划器：      调用 f 比较几种未来 → 选择动作          a_t = Planner(s_t, f, 目标)
策略 π：      当前信息 → 直接输出动作                  a_t = π(s_t)
```

下一章会把网格世界换成图片、视频和连续经历。

> 👉 动手实验：[动手：从零重新发明世界模型](/chapters/01-why-world-models/06-invent-a-world-model)

## 参考资料

### 实践博客（5 篇）

1. [worldmodels.github.io (Ha & Schmidhuber)](https://worldmodels.github.io/) —— 论文配套交互网站，图示与可玩实验丰富，是公认的最佳入门读物。
2. [Deep Reinforcement Learning: Pong and Pixels (Karpathy, 2016)](https://karpathy.github.io/2016/05/31/rl/) —— 用两个环境把策略梯度、价值这些概念讲透的名博客，适合补本章的决策直觉。
3. [MuZero: Mastering Go, chess, shogi and Atari (DeepMind, 2020)](https://deepmind.google/discover/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules/) —— 官方博客：不给规则、只靠学到的模型规划，配本章“学到动态”一节。
4. [Genie 3: A new frontier for world models (Google DeepMind, 2025)](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) —— 实时可交互世界模型的官方博客，直观展示“动作条件未来”走到哪一步。
5. [Genie 2: A large-scale foundation world model (Google DeepMind, 2024)](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/) —— 单张图生成可玩 3D 世界的官方博客，适合建立对世界模型产品的第一印象。

### 原始论文（5 篇）

1. [World Models (Ha & Schmidhuber, 2018)](https://arxiv.org/abs/1803.10122) —— 现代世界模型的起点：VAE + RNN 搭出小模型，再在模型内部学习控制。
2. [Recurrent World Models Facilitate Policy Evolution (Ha & Schmidhuber, 2018)](https://proceedings.neurips.cc/paper/2018/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html) —— 同工作的 NeurIPS 正式版，含梦境训练与温度参数的完整细节。
3. [Making the World Differentiable (Schmidhuber, 1990)](https://people.idsia.ch/~juergen/world-models-planning-curiosity-fki-1990.pdf) —— “控制器 + 世界模型”框架的第一份技术报告，概念源头。
4. [A Path Towards Autonomous Machine Intelligence (LeCun, 2022)](https://openreview.net/forum?id=BZ5a1r-kVsf) —— 提出 JEPA 与世界模型架构蓝图的立场论文，第 6 章的理论源头。
5. [Understanding World or Predicting Future? A Comprehensive Survey of World Models (Ding et al., 2024)](https://arxiv.org/abs/2411.14499) —— 覆盖两条路线分野的综述，适合读完本章后建立全局地图。
