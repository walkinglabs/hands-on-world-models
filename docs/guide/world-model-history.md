# 世界模型简史

如果我们在 2015 年询问一位 AI 研究员"什么是世界模型"，他大概会耸耸肩——这个词还躺在几篇冷门论文里，指的是机器人控制中一个不起眼的动态模型。但如果我们把时间指针拨回八十年前，或者快进到今天视频生成与具身智能的浪潮，你会发现"世界模型"经历了一场跨越学科的漫长演变——它从心理学家的假说出发，经过控制论与强化学习的两次锻造，最终在深度学习的熔炉中分裂成两条路线：**一条为决策服务，一条为生成服务**。

在开始动手写代码之前，不妨先花几分钟走完这八十年。看懂这段历史，你就能理解本课程为什么要同时讲 Dreamer 和 Genie。

![左侧是 1940 年代心理学实验室里走迷宫的老鼠，右侧渐变为现代机器人实验室](/guide/wm-evolution.webp)

<div style="text-align:center; font-size:0.9em; color:var(--vp-c-text-2); margin-top:-10px; margin-bottom:20px;">
  <em>图 1：从老鼠的认知地图到机器的世界模型，同一个问题走了八十年。</em>
</div>

## 1. 心理学源头：大脑里的小模型（1940s）

世界模型的思想源头不在计算机科学，而在**心理学**。

1943 年，英国心理学家肯尼斯·克雷克（Kenneth Craik）在《The Nature of Explanation》中提出了一个惊人的假说：大脑会在内部构建一个外部世界的"**小尺度模型**"（small-scale model），用它推演各种行动方案的后果，再选出最好的那个。你听出来了吗？这几乎是今天世界模型定义的原文——只是载体从神经网络换成了大脑。

1948 年，爱德华·托尔曼（Edward Tolman）通过老鼠走迷宫实验提出了"**认知地图**"（cognitive map）假说：动物学到的不是"看到墙就右转"的刺激-反应链，而是环境的内部表征。半个多世纪后，这一假说随位置细胞与网格细胞的发现获得了 2014 年诺贝尔生理学或医学奖——生物大脑确实在绘制地图。

## 2. 数学化与模型法的确立（1957–1990s）

心理学的假说要变成算法，需要一次数学锻造。

1957 年，理查德·贝尔曼（Richard Bellman）提出**马尔可夫决策过程（MDP）**，把"在环境中做序列决策"变成可计算的数学对象。但贝尔曼的动态规划有个苛刻的前提：环境的转移概率必须**已知**。现实中这几乎从不成立——机器人不知道推开一扇门后走廊有多宽，AI 也不知道对手下一步会走哪步棋。于是两条路出现了：无模型路线（TD 学习、Q-Learning）干脆不学模型，直接估计价值；模型法路线则坚持**先学环境如何演化，再在模型内部推演**。

模型法的两个奠基工作都出现在 1990 年前后，像一对双胞胎：

- **1990–1991 年，理查德·萨顿提出 Dyna 架构**：用真实经验训练环境模型，再用模型"想象"出额外的训练经验。这是"**在想象中训练**"的最早形态——Dreamer 在 30 年后做的，本质上是同一件事的深度化版本。
- **1990 年，于尔根·施密德胡伯（Jürgen Schmidhuber）发表技术报告**，首次把“**控制器 + 世界模型**”作为一对系统提出：世界模型学习动作的后果，控制器利用它在内部做规划。“世界模型”一词在机器学习中的正式登场，由此算起。

![1990 年的手绘示意图：控制器与环境交互，RNN 世界模型在内部学习并预测环境的反应](/guide/world_models_1990.jpeg)

<div style="text-align:center; font-size:0.9em; color:var(--vp-c-text-2); margin-top:-10px; margin-bottom:20px;">
  <em>图 2：施密德胡伯 1990 年手绘的“控制器 + 世界模型”示意图——比深度学习时代早了二十八年。来源：<a href="https://worldmodels.github.io/" target="_blank" rel="noopener noreferrer">worldmodels.github.io</a>（CC-BY 4.0）</em>
</div>

又过了二十年，模型法才在真实硬件上证明自己。**2011 年**，Deisenroth 与 Rasmussen 的 **PILCO** 用高斯过程学动态模型，让真实机器人在区区几次尝试后就学会倒立——模型法样本效率的第一次有力展示。

## 3. 深度世界模型成型（2018–2023）

深度学习的加入，让"高维观测 + 学到的动态"第一次变得可行。这五年是概念密集涌现的五年：

- **2018 年**，David Ha 与 Schmidhuber 发表 **World Models**：VAE 压缩画面、MDN-RNN 预测未来、一个 867 个参数的小控制器完全在"梦境"中训练，再迁回真实环境。V-M-C 框架与交互式的展示方式让它成为公认的现代起点——本课程第 1 章的原型正是这篇论文，[4.6 动手：复现 World Models](/chapters/04-decision-and-planning/06-reproduce-world-models) 用 CarRacing 把它完整跑通。
- **2019 年**，Hafner 等的 **PlaNet** 提出 RSSM（递归状态空间模型），把确定性记忆与随机不确定性分开建模，配合 CEM 在隐空间规划。同年 DeepMind 的 **SimPLe** 证明：只用 10 万帧真实数据，加上视频世界模型生成的经验，就能训练出像样的 Atari 策略。
- **2020 年**，两条分岔同时出现：**Dreamer** 不再做昂贵的在线规划，而是直接在学习到的世界模型里反向传播训练 Actor-Critic；**MuZero** 则走向另一个极端——只学"对决策有用的隐式模型"，不重建任何观测，配合 MCTS 打通围棋与 Atari。
- **2021–2023 年**，**DreamerV2/V3** 引入离散 latent 与大规模工程打磨，一套超参数打通 150 多个任务；**TD-MPC2** 展示了模型预测控制路线的可扩展性。为决策服务的世界模型，至此成熟。

## 4. 路线分裂：生成式世界模型（2023 至今）

就在决策路线趋于稳定时，视频生成技术让世界模型裂出了第二条路线——目标不再是帮策略决策，而是**生成可控的未来本身**：

- **2023 年**，Wayve 发布 **GAIA-1**：视频 token + 自回归 Transformer + 扩散解码器，在驾驶场景里用文字和动作控制未来（第 8 章）。
- **2024 年**，三篇工作把这条路线推向高潮：DeepMind 的 **Genie** 从无动作标注的视频里学出潜在动作模型，单张图生成可玩环境；**GameNGen** 用扩散模型实时模拟 DOOM；**DIAMOND** 则把扩散世界模型接回强化学习，直接在其中训练策略——两条路线在这里交汇（第 5 章）。
- **2025 年**，路线继续分化：**Genie 2/3** 走向实时可交互的基础世界模型；**GAIA-2** 用可控反事实场景服务驾驶评测（第 8.5 节）；Meta 的 **V-JEPA 2-AC** 代表第三条中间路线——不生成像素，只预测抽象特征，再用 62 小时机器人数据实现零样本规划（第 6 章）。
- 同期，评测开始跟上：**WorldScore**、**WorldModelBench** 等基准把"画面像不像"与"演化对不对"拆开打分（第 9 章）。

## 5. 两条路线的分野与合流

回看整段历史，世界模型始终存在一个核心张力：

- **为决策服务的路线**（Dyna → PlaNet → Dreamer → MuZero）：输出抽象、预测快、可直接规划，但人看不见模型在想什么；
- **为生成服务的路线**（GAIA → Genie → GameNGen）：输出可见、可审查、能造数据，但像素预测昂贵，物理正确性存疑。

JEPA 路线试图两头兼顾：只预测抽象特征，不生成像素，同时保留动作与规划能力。这场争论仍在进行——也正是本课程第 6 章与第 9 章的中心议题。

## 小结

从克雷克的小尺度模型，到托尔曼的认知地图；从 Schmidhuber 的"控制器 + 世界模型"，到 Dreamer 的想象训练与 Genie 的可玩世界。世界模型的历史，就是一部 **"预测世界"与"利用预测"不断互相改造**的史诗——它的内核从未改变：**在行动之前，先在内部预见行动的后果**。

今天，世界模型已经不再是强化学习角落里的小众方向，它是具身智能、自动驾驶与可交互内容生成的共同底座。在接下来的章节中，我们将沿着这段历史的脉络，从第一行代码开始，亲手把这些伟大的想法实现出来。

## 参考文献

1. Craik, K. J. W. (1943). _The Nature of Explanation_. Cambridge University Press.
2. Tolman, E. C. (1948). Cognitive maps in rats and men. _Psychological Review_, 55(4), 189–208.
3. Sutton, R. S. (1991). Dyna, an Integrated Architecture for Learning, Planning, and Reacting. _ACM SIGART Bulletin_, 2(4), 160–163.
4. Schmidhuber, J. (1990). Making the World Differentiable: On Using Self-Supervised Fully Recurrent Neural Networks for Dynamic Reinforcement Learning and Planning in Non-Stationary Environments. _Technical Report FKI-126-90, TUM_.
5. Deisenroth, M. P., & Rasmussen, C. E. (2011). PILCO: A Model-Based and Data-Efficient Approach to Policy Search. _ICML_.
6. Ha, D., & Schmidhuber, J. (2018). Recurrent World Models Facilitate Policy Evolution. _NeurIPS_. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)，配套交互式文章：[worldmodels.github.io](https://worldmodels.github.io/)（CC-BY 4.0）。
7. Hafner, D., et al. (2019). Learning Latent Dynamics for Planning from Pixels. _ICML_. [arXiv:1811.04551](https://arxiv.org/abs/1811.04551)
8. Kaiser, Ł., et al. (2020). Model-Based Reinforcement Learning in Atari without Dreaming. _ICLR_. [arXiv:1903.00374](https://arxiv.org/abs/1903.00374)
9. Hafner, D., et al. (2020). Dream to Control: Learning Behaviors by Latent Imagination. _ICLR_. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603)
10. Schrittwieser, J., et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. _Nature_, 588, 604–609. [arXiv:1911.08265](https://arxiv.org/abs/1911.08265)
11. Hafner, D., et al. (2023). Mastering Diverse Domains through World Models. [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
12. Hu, A., et al. (2023). GAIA-1: A Generative World Model for Autonomous Driving. [arXiv:2309.17080](https://arxiv.org/abs/2309.17080)
13. Bruce, J., et al. (2024). Genie: Generative Interactive Environments. _ICML_. [arXiv:2402.15391](https://arxiv.org/abs/2402.15391)
14. Valevski, D., et al. (2024). Diffusion Models Are Real-Time Game Engines. [arXiv:2408.14837](https://arxiv.org/abs/2408.14837)
15. Bardes, A., et al. (2025). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
16. Duan, H., et al. (2025). WorldScore: A Unified Evaluation Benchmark for World Generation. [arXiv:2504.00983](https://arxiv.org/abs/2504.00983)
17. Ding, J., et al. (2024). Understanding World or Predicting Future? A Comprehensive Survey of World Models. [arXiv:2411.14499](https://arxiv.org/abs/2411.14499)
