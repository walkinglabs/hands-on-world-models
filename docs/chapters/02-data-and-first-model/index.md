# 第 2 章　数据与第一个模型

前一章认识了常用组件。这一章先不急着换上神经网络，而是解决一个更基本的问题：机器的一段经历究竟怎样保存，模型又怎样从中学会“做了什么，后来发生了什么”。

## 本章文章

1. [一段经历怎样保存](./02-01-episodes-and-transitions.md)：把观察、动作、奖励和下一次观察按时间接好。
2. [Replay Buffer 怎样取出可学习的数据](./02-02-replay-buffer-and-splits.md)：连续采样，并避免把未来信息泄漏到测试集。
3. [从经历学出第一台模型](./02-03-first-learned-world.md)：用计数学习概率转移，再把预测放回行动循环。

## 本章实验与作业

- [F3：学习一个表格小世界](/labs/foundations)
- [PA0：第一台可学习世界](/assignments/pa0)

本章只有一份 Notebook。前三篇短文围绕同一份数据和同一个小环境展开，读完以后再在 F3 中一次接起来。

## 学完以后怎样选路

完成第 0–2 章以后，就可以从第三部分选择一条路线。五条路线都复用本章的数据语言，但彼此不是先修关系。

## 参考资料

### 实践博客（5 篇）

1. [OpenAI Spinning Up: Key Concepts of RL](https://spinningup.openai.com/en/latest/) —— OpenAI 官方入门教程，把 episode、轨迹、价值这些术语一次讲清。
2. [A (Long) Peek into Reinforcement Learning (Lilian Weng, 2018)](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) —— 从 MDP 到模型法 RL 的全景综述博客，适合给本章术语找上下文。
3. [Grokking Deep Reinforcement Learning (Kaixhin)](https://kaixhin.com/grokking-deep-reinforcement-learning/) —— 长文系列，逐层拆解深度 RL 的实践细节与常见坑。
4. [Human-level control through deep reinforcement learning (DeepMind, 2015)](https://deepmind.google/discover/blog/human-level-control-through-deep-reinforcement-learning/) —— DQN 发布时的官方博客，配本章经验回放的历史起点。
5. [Reinforcement Learning: An Introduction (Sutton & Barto, 2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html) —— 免费在线教材，本章用到的 MDP 与表格学习均可在此查证。

### 原始论文（5 篇）

1. [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://www.nature.com/articles/nature14236) —— DQN 的 Nature 论文，经验回放进深度学习的标志性工作。
2. [Prioritized Experience Replay (Schaul et al., 2016)](https://arxiv.org/abs/1511.05952) —— 按重要性取样回放，是本章“怎样取样”问题的直接回答。
3. [Rainbow: Combining Improvements in Deep RL (Hessel et al., 2018)](https://arxiv.org/abs/1710.02298) —— 把回放、分布、双 Q 等改进逐项消融，示范了数据设计怎样影响成绩。
4. [Revisiting Fundamentals of Experience Replay (Fedus et al., 2020)](https://arxiv.org/abs/2007.06700) —— 系统研究回放的容量、采样与保留策略，直接服务 2.2。
5. [Deep Recurrent Q-Learning for Partially Observable MDPs (Hausknecht & Stone, 2015)](https://arxiv.org/abs/1507.06527) —— 用循环网络补状态信息，呼应本章 episode 与隐状态的关系。
