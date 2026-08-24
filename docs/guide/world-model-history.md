# 世界模型八十年：从认知地图到可生成的世界

如果我们在 2015 年询问一位 AI 研究员"什么是世界模型"，他大概会耸耸肩——这个词还躺在几篇冷门论文里，指的是机器人控制中一个不起眼的动态模型。但如果我们把时间指针拨回八十年前，或者快进到今天视频生成与具身智能的浪潮，你会发现"世界模型"经历了一场跨越学科的漫长演变——它从心理学家的假说出发，经过控制论与强化学习的两次锻造，最终在深度学习的熔炉中分裂成两条路线：**一条为决策服务，一条为生成服务**。

在开始动手写代码之前，不妨先花十几分钟走完这八十年。看懂这段历史，你就能理解本课程为什么要同时讲 Dreamer 和 Genie。

![左侧是 1940 年代心理学实验室里走迷宫的老鼠，右侧渐变为现代机器人实验室](/guide/wm-evolution.webp)

<div style="text-align:center; font-size:0.9em; color:var(--vp-c-text-2); margin-top:-10px; margin-bottom:20px;">
  <em>图 1：从老鼠的认知地图到机器的世界模型，同一个问题走了八十年。</em>
</div>

![世界模型八十年时间线：从心理学假说到生成式基础模型的八个阶段](/guide/wm-timeline.svg)

<div style="text-align:center; font-size:0.9em; color:var(--vp-c-text-2); margin-top:-10px; margin-bottom:20px;">
  <em>图 2：整段历史的鸟瞰，八个阶段与两条路线。蓝线为决策服务，紫线为生成服务，它们至今仍在竞争与合流。</em>
</div>

## 1. 心理学源头：大脑里的小模型（1930s–1940s）

世界模型的思想源头不在计算机科学，而在**心理学**。

1943 年，英国心理学家肯尼斯·克雷克（Kenneth Craik）在《The Nature of Explanation》中提出了一个惊人的假说：大脑会在内部构建一个外部世界的"**小尺度模型**"（small-scale model），用它推演各种行动方案的后果，再选出最好的那个。用他自己的话说，有机体携带的是一个"与外部世界同构、且能以相同关系运转"的模型。你听出来了吗？这几乎是今天世界模型定义的原文——只是载体从神经网络换成了大脑。克雷克本人是参与雷达研制的战时科学家，遗憾的是他在 1945 年死于火车事故，没能看到假说被机器验证的那天。

值得注意的是，克雷克并非孤例。几乎同一时期：

- **1931 年**，心理学家托尔曼（Edward Tolman）就在实验中观察到：老鼠即使没有奖励也会"顺便"学会迷宫的布局——他后来称之为**潜伏学习**（latent learning）。这暗示学习的内容不是动作本身，而是环境结构。
- **1948 年**，托尔曼正式提出"**认知地图**"（cognitive map）假说：动物学到的不是"看到墙就右转"的刺激-反应链，而是环境的内部表征——一份地图。行为主义心理学家当时对此嗤之以鼻，因为"头脑里的地图"无法直接观测。
- 这场争论沉寂了半个多世纪，直到 1971 年奥基夫（O'Keefe）发现**位置细胞**、2005 年莫泽夫妇（Moser 夫妇）发现**网格细胞**，"大脑确实在画地图"才成为定论——三人分享了 2014 年诺贝尔生理学或医学奖。托尔曼的假说等了 66 年才等到证据。

心理学史给我们的启示至今有效：**世界模型研究的核心争议——模型该重建什么、表征该多具体——从第一天起就存在**。托尔曼的"地图派"（存抽象结构）与行为主义的"反应派"（只管输入输出）之争，正是今天 Dreamer（生成式隐变量）与 MuZero（纯价值导向隐式模型）之争的前身。

## 2. 控制论与动态规划：预见后果变成数学（1948–1957）

战后十年，"内部模型"从心理学假说变成了工程词汇。

- **1948 年**，维纳（Norbert Wiener）出版《控制论》，把"反馈"确立为机器与生物共通的调节原理——恒温器、炮火瞄准器与生物体都在用误差修正行为。控制论意义上的"模型"还是外部写死的传递函数，但"**用系统对外界的预测来纠正行动**"这个闭环思想已经就位。
- **1956 年**，达特茅斯会议创立"人工智能"一词；同一年，**贝尔曼（Richard Bellman）**在兰德公司提出**动态规划**，次年正式提出**马尔可夫决策过程（MDP）**：状态、动作、转移、奖励，把"在环境中做序列决策"变成可计算的数学对象，外加著名的贝尔曼方程。

贝尔曼的动态规划有个苛刻的前提：环境的转移概率必须**已知**。现实中这几乎从不成立——机器人不知道推开一扇门后走廊有多宽，AI 也不知道对手下一步会走哪步棋。"模型已知"这道裂缝，正是后来一切故事的开端。

## 3. 无模型与模型法的分野（1980s）

要跨过"模型未知"这道坎，社区在 1980 年代分成了两派。

- **无模型路线**（model-free）：干脆不学环境模型，直接估计"每个动作有多好"。1988 年 Sutton 的 **TD 学习**用自举（bootstrapping）估计长期回报，1989 年 Watkins 的 **Q-Learning** 证明离策略收敛。优点是对模型误差免疫，缺点是样本效率极低——在这个时代，它们在真实机器人上几乎不可用。
- **模型法路线**（model-based）：坚持**先学环境如何演化，再在模型内部推演**。样本效率高，但模型学错了，规划就会"垃圾进垃圾出"。

这个取舍——**样本效率 vs 对模型误差的鲁棒性**——是贯穿后文全部历史的第二根主线。两条路线在此后四十年里轮流领先，谁也没能彻底淘汰对方。

## 4. 模型法奠基：Dyna 与"控制器 + 世界模型"（1990–1991）

模型法的两个奠基工作都出现在 1990 年前后，像一对双胞胎：

- **Dyna 架构（Sutton，1990–1991）**：用真实经验训练环境模型，再用模型"想象"出额外的训练经验，混合训练策略或价值函数。这是"**在想象中训练**"的最早形态——Dreamer 在 30 年后做的，本质上是同一件事的深度化版本。
- **"控制器 + 世界模型"（Schmidhuber，1990）**：于尔根·施密德胡伯（Jürgen Schmidhuber）发表技术报告《Making the World Differentiable》，首次把"**控制器 + 世界模型**"作为一对系统提出：世界模型（一个全循环 RNN）学习动作的后果，控制器利用它在内部做规划，两者共同训练。"世界模型"一词在机器学习中的正式登场，由此算起。他当时的口号颇具野心：让世界变得可微——对世界模型求梯度，就能直接对行动求梯度。

![1990 年的手绘示意图：控制器与环境交互，RNN 世界模型在内部学习并预测环境的反应](/guide/world_models_1990.jpeg)

<div style="text-align:center; font-size:0.9em; color:var(--vp-c-text-2); margin-top:-10px; margin-bottom:20px;">
  <em>图 3：施密德胡伯 1990 年手绘的“控制器 + 世界模型”示意图——比深度学习时代早了二十八年。来源：<a href="https://worldmodels.github.io/" target="_blank" rel="noopener noreferrer">worldmodels.github.io</a>（CC-BY 4.0）</em>
</div>

这两条思路的分工延续至今：Dyna 关心"模型如何帮**学习**"（生成训练数据），Schmidhuber 关心"模型如何帮**规划**"（搜索好动作）。现代工作几乎都是两者的混合。

## 5. 世纪之交：统计学习与真实机器人（1995–2011）

1990 年代的模型法受限于两件事：学的模型（线性动态、决策树）太弱，数据太少。转机来自两个方向：

- **概率模型路线**：用高斯过程、贝叶斯网络刻画动态的不确定性，模型第一次能说"我不确定"。代表作是 **2011 年 Deisenroth & Rasmussen 的 PILCO**：用高斯过程学动态模型，并在模型分布上做解析规划，让一个真实机械臂在**区区十几次尝试**（约 20 秒经验）后就学会倒立摆——对比无模型方法动辄数小时的真实交互，模型法的样本效率第一次震撼了机器人学界。
- **大规模规划路线**：iLQG、MPC 等最优控制方法在机器人上成熟，"模型预测控制"（MPC）成为工业界标准——从化工厂到波士顿动力的全身控制器，用的都是"每步重规划"的 MPC。今天很多"世界模型"工作的推理端，本质上仍是 MPC——只是模型从手写物理换成了神经网络。

回头看，1990–2010 这二十年的低谷并非没有意义：它把"模型误差会毁掉规划"这个教训刻进了社区记忆。后来 PlaNet 的 RSSM、Dreamer 的离散隐变量、MuZero 的"只学有用的事"，全都是对这个教训的回应。

## 6. 深度世界模型成型：World Models 到 PlaNet（2018–2019）

深度学习的加入，让"高维观测 + 学到的动态"第一次变得可行。奠基之作是 Ha 与 Schmidhuber 的 **World Models**（NeurIPS 2018，配一篇著名的交互式网页文章），由三件套组成：

- **V**（Vision）：一个变分自编码器 VAE 把 64×64 的画面压缩成 32 维隐向量——"看见"即"压缩"；
- **M**（Memory）：一个 MDN-RNN 预测下一帧隐向量的分布——"记住"即"预测"；
- **C**（Controller）：一个仅有 **867 个参数**的线性控制器，完全在这个"梦境"里用进化策略训练，再零样本迁回真实环境。

![VAE 编码器把画面压缩为隐向量、解码器重建画面](/guide/vae.svg)

<div style="text-align:center; font-size:0.9em; color:var(--vp-c-text-2); margin-top:-10px; margin-bottom:20px;">
  <em>图 4：V 组件——VAE。编码器把高维画面压成低维隐向量，世界模型只在隐空间里预测未来，这是所有后续工作的共同起点。</em>
</div>

![MDN-RNN 根据当前隐状态与动作预测下一帧隐状态的分布](/guide/mdn_rnn.svg)

<div style="text-align:center; font-size:0.9em; color:var(--vp-c-text-2); margin-top:-10px; margin-bottom:20px;">
  <em>图 5：M 组件——MDN-RNN。混合密度输出让模型能表达“未来有多种可能”，而不是 collapsing 到模糊的均值。</em>
</div>

这篇工作的意义不在性能——它跑的只是 CarRacing 和 VizDoom 两个玩具任务——而在**范式**：证明了"压缩 → 预测 → 在预测里训练"这条流水线可以端到端成立。本课程第 1 章的原型正是这篇论文，[4.6 动手：World Models 的复现](/chapters/04-decision-and-planning/06-reproduce-world-models) 用 CarRacing 把它完整跑通。

第二年，DeepMind 的 Danijar Hafner 等发表 **PlaNet**（2019），贡献有二：

- **RSSM（递归状态空间模型）**：把 RNN 的确定性记忆（想起上一步）与 VAE 的随机隐变量（世界有偶然性）**分开建模、结合使用**——确定性路径保证长程预测不发散，随机路径防止模型对不确定的未来过度自信。这个结构成为后续 Dreamer 全系列的骨架，也是本课程 [4.2 RSSM](/chapters/04-decision-and-planning/02-rssm-training) 的主角。
- **隐空间 CEM 规划**：不再生成像素，直接在隐空间里用交叉熵方法搜索动作序列，一步规划只需毫秒级——模型法第一次在像素输入上实时跑起来。

同年，DeepMind 的 **SimPLe** 证明：只用 10 万帧真实 Atari 数据（约 2 小时游戏），加上视频世界模型生成的额外经验，就能训练出像样的策略——而无模型基线需要 5 亿帧。"模型省数据"从格言变成了数字。

## 7. 想象训练与隐式模型：决策路线成熟（2020–2023）

2020 年，两条相反的哲学同时开花结果：

- **Dreamer**（Hafner 等，ICLR 2020）：不再做昂贵的在线规划，而是把世界模型当作可微"模拟器"，在想象的隐状态轨迹上**反向传播**，直接训练 Actor-Critic。规划从"推理时搜索"变成了"训练时想象"——Dyna 思想的深度化完整体。DreamerV2（2021）把隐变量改成离散类别并赢得 Atari 100k 挑战；**DreamerV3**（2023）用一套固定超参数打通 150 多个任务，成为第一个在 Minecraft 里从零采到钻石的模型。
- **MuZero**（Schrittwieser 等，*Nature* 2020）：走向另一个极端——**不重建任何观测**，只学"对预测价值与奖励有用的隐式模型"，配合蒙特卡洛树搜索同时打通围棋、国际象棋、将棋与 Atari。AlphaGo 需要人类规则知识，AlphaZero 需要完美模拟器，MuZero 两者都不要。它证明了：世界模型不必"像世界"，只需"够用"。

Dreamer 与 MuZero 的对照，本质是"**模型该重建世界，还是只服务于决策**"这场古老争论（回想托尔曼与行为主义）在现代的重演。本课程 [4.5 MuZero](/chapters/04-decision-and-planning/05-muzero) 与 [4.4 Dreamer](/chapters/04-decision-and-planning/04-dreamer-imagination) 会分别动手实现两者的最小版本。

与 DreamerV3 同期，**TD-MPC2**（2024）沿 MPC 路线把模型预测控制扩展到 100 多个连续控制任务并支持多任务联合训练，说明"在线规划"这条路并未被"想象训练"取代。为决策服务的世界模型，至此成熟。

## 8. 路线分裂：生成式世界模型（2023–2024）

就在决策路线趋于稳定时，视频生成技术让世界模型裂出了第二条路线——目标不再是帮策略决策，而是**生成可控的未来本身**。技术上的分水岭是 **VQ token 化 + 自回归 Transformer + 扩散解码器**这套组合拳（本课程第 5 章）：它把"预测下一帧"从像素回归问题变成了序列生成问题。

- **2023 年**，Wayve 发布 **GAIA-1**：9B 参数视频世界模型，视频 token + 自回归 Transformer + 扩散解码器，在驾驶场景里用文字和动作控制未来——输入"迎面驶来一辆卡车"，画面里就真的出现卡车。世界模型第一次被当成"可控数据工厂"而非"策略教练"（第 8 章）。
- **2024 年**，三篇工作把这条路线推向高潮：
  - DeepMind 的 **Genie** 从**无动作标注**的互联网视频里学出潜在动作模型，单张图片生成可交互环境——"看游戏视频学会做游戏"；
  - **GameNGen**（Google Research）用扩散模型实时模拟 DOOM，每秒 20 帧，玩家难以区分这是模拟还是游戏引擎；
  - **DIAMOND** 则把扩散世界模型接回强化学习，直接在其中训练策略——生成路线与决策路线在这里第一次交汇（第 5 章）。

![World Models 论文的交互式演示：策略在学到的梦境环境中驾驶](/guide/wm-sandbox.webp)

<div style="text-align:center; font-size:0.9em; color:var(--vp-c-text-2); margin-top:-10px; margin-bottom:20px;">
  <em>图 6：在“梦境”里驾驶。策略完全在世界模型想象出的环境中训练与行动——这条流水线从 2018 年的玩具，长成了 2025 年的基础设施。来源：<a href="https://worldmodels.github.io/" target="_blank" rel="noopener noreferrer">worldmodels.github.io</a>（CC-BY 4.0）</em>
</div>

## 9. 基础模型化、第三路线与评测（2025 至今）

2025 年之后，格局进一步分化成三极：

- **生成式基础模型**：**Genie 2/3** 走向实时可交互的基础世界模型，任何人给出一张图就能得到一个可玩世界，并开始探索作为机器人策略训练环境的可能；**GAIA-2** 用可控反事实场景服务驾驶安全评测（第 8.5 节）。
- **特征预测的第三路线**：Meta 的 **V-JEPA 2-AC** 不生成像素，只预测抽象特征，再用 62 小时机器人数据实现零样本规划（第 6 章）。LeCun 倡导的 JEPA 路线主张用"丢弃像素"换取效率与鲁棒。
- **评测体系成形**：**WorldScore**、**WorldModelBench** 等基准把"画面像不像"（视觉保真度）与"演化对不对"（动态正确性、动作响应、反事实一致性）拆开打分（第 9 章）。没有评测的繁荣都是假繁荣——这是每条技术路线成熟的必经一步。

合流的迹象也已经出现：DIAMOND 用生成式世界模型训练策略；DreamerV3 的隐空间正被用作轻量"语义仿真器"。八十年前克雷克说大脑用小模型推演行动——他没有说模型必须画得多逼真。**逼真与否是手段，预见后果才是目的**。

## 10. 两条路线的分野与合流

回看整段历史，世界模型始终存在一个核心张力：

| | 为决策服务 | 为生成服务 |
|---|---|---|
| 代表 | Dyna → PlaNet → Dreamer → MuZero | GAIA → Genie → GameNGen |
| 输出 | 抽象隐状态 | 像素/视频 |
| 优点 | 预测快、可直接规划与训练 | 可见、可审查、能造数据 |
| 缺点 | 人看不见模型在想什么 | 像素预测昂贵，物理正确性存疑 |
| 典型用途 | 机器人、游戏 AI | 内容生成、驾驶仿真、数据增强 |

JEPA 路线试图两头兼顾：只预测抽象特征，不生成像素，同时保留动作条件与规划能力。这场争论仍在进行——也正是本课程第 6 章与第 9 章的中心议题。

## 小结

从克雷克的小尺度模型，到托尔曼的认知地图；从维纳的反馈与贝尔曼的 MDP，到 Sutton 的 Dyna、Schmidhuber 的"控制器 + 世界模型"；从 PILCO 上的真实倒立摆，到 World Models 的三件套、PlaNet 的 RSSM、Dreamer 的想象训练、MuZero 的隐式模型；再到 Genie 的可玩世界与 V-JEPA 的特征预测。世界模型的历史，就是一部 **"预测世界"与"利用预测"不断互相改造**的史诗——它的内核从未改变：**在行动之前，先在内部预见行动的后果**。

今天，世界模型已经不再是强化学习角落里的小众方向，它是具身智能、自动驾驶与可交互内容生成的共同底座。在接下来的章节中，我们将沿着这段历史的脉络，从第一行代码开始，亲手把这些伟大的想法实现出来。

## 参考文献

1. Craik, K. J. W. (1943). _The Nature of Explanation_. Cambridge University Press.
2. Tolman, E. C. (1948). Cognitive maps in rats and men. _Psychological Review_, 55(4), 189–208.
3. Wiener, N. (1948). _Cybernetics: Or Control and Communication in the Animal and the Machine_. MIT Press.
4. Bellman, R. (1957). _Dynamic Programming_. Princeton University Press.
5. Sutton, R. S. (1988). Learning to Predict by the Methods of Temporal Differences. _Machine Learning_, 3, 9–44.
6. Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. _Machine Learning_, 8, 279–292.
7. Sutton, R. S. (1991). Dyna, an Integrated Architecture for Learning, Planning, and Reacting. _ACM SIGART Bulletin_, 2(4), 160–163.
8. Schmidhuber, J. (1990). Making the World Differentiable: On Using Self-Supervised Fully Recurrent Neural Networks for Dynamic Reinforcement Learning and Planning in Non-Stationary Environments. _Technical Report FKI-126-90, TUM_.
9. Deisenroth, M. P., & Rasmussen, C. E. (2011). PILCO: A Model-Based and Data-Efficient Approach to Policy Search. _ICML_.
10. Ha, D., & Schmidhuber, J. (2018). Recurrent World Models Facilitate Policy Evolution. _NeurIPS_. [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)，配套交互式文章：[worldmodels.github.io](https://worldmodels.github.io/)（CC-BY 4.0）。
11. Hafner, D., et al. (2019). Learning Latent Dynamics for Planning from Pixels. _ICML_. [arXiv:1811.04551](https://arxiv.org/abs/1811.04551)
12. Kaiser, Ł., et al. (2020). Model-Based Reinforcement Learning in Atari without Dreaming (SimPLe). _ICLR_. [arXiv:1903.00374](https://arxiv.org/abs/1903.00374)
13. Hafner, D., et al. (2020). Dream to Control: Learning Behaviors by Latent Imagination. _ICLR_. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603)
14. Schrittwieser, J., et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. _Nature_, 588, 604–609. [arXiv:1911.08265](https://arxiv.org/abs/1911.08265)
15. Hafner, D., et al. (2023). Mastering Diverse Domains through World Models. [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
16. Hansen, N., et al. (2024). TD-MPC2: Scalable, Robust World Models for Continuous Control. _ICLR_. [arXiv:2310.16828](https://arxiv.org/abs/2310.16828)
17. Hu, A., et al. (2023). GAIA-1: A Generative World Model for Autonomous Driving. [arXiv:2309.17080](https://arxiv.org/abs/2309.17080)
18. Bruce, J., et al. (2024). Genie: Generative Interactive Environments. _ICML_. [arXiv:2402.15391](https://arxiv.org/abs/2402.15391)
19. Valevski, D., et al. (2024). Diffusion Models Are Real-Time Game Engines. [arXiv:2408.14837](https://arxiv.org/abs/2408.14837)
20. Alonso, E., et al. (2024). Diffusion for World Modeling: Visual Details Matter in Atari. _NeurIPS_. [arXiv:2405.12399](https://arxiv.org/abs/2405.12399)
21. Bardes, A., et al. (2025). V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning. [arXiv:2506.09985](https://arxiv.org/abs/2506.09985)
22. Duan, H., et al. (2025). WorldScore: A Unified Evaluation Benchmark for World Generation. [arXiv:2504.00983](https://arxiv.org/abs/2504.00983)

### 综述与立场文章

23. Ding, J., et al. (2024). Understanding World or Predicting Future? A Comprehensive Survey of World Models. [arXiv:2411.14499](https://arxiv.org/abs/2411.14499)
24. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence (Version 0.9.2). _OpenReview_. [PDF](https://openreview.net/pdf?id=BZ5a1r-kVsf)——JEPA 路线与"世界模型是智能体必备组件"这一立场的源头文本，本页第 9 节与第 6 章的许多论点出自于此。
25. Ha, D. (2018). World Models — 博客版交互式文章. [worldmodels.github.io](https://worldmodels.github.io/)（CC-BY 4.0）——除论文外还包含可玩的浏览器演示，是理解"梦境训练"最直观的材料。
26. Gu, C., et al. (2025). World Models for Autonomous Driving: A Comprehensive Survey. [arXiv:2501.11260](https://arxiv.org/abs/2501.11260)——从驾驶视角整理生成式与决策式两条路线，与本书第 8 章互补。
