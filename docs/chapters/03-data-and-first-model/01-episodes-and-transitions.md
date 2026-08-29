# 3.1　经历与状态转移

> **第 3 章 · 数据与第一个世界模型**
>
> 前一章认识了常用组件。这一章先不急着换上神经网络，而是解决一个更基本的问题：机器的一段经历究竟怎样保存，模型又怎样从中学会“做了什么，后来发生了什么”。
>
> 本章前三篇短文围绕同一份数据和同一个小环境展开，最终在 [3.5 动手：表格世界模型的从零开始实现](/chapters/03-data-and-first-model/05-rl-foundation-scratch) 中完成全流程串联。完成第 1–3 章后即可掌握世界模型共用的数据语言，进入后续五条具体路线。

世界模型不能只拿一堆互不相关的图片训练。它需要知道第几步看到了什么、做了什么，以及动作以后发生了什么。

课程使用下面的最小容器：

```python
Episode(
    observations=[T + 1, ...],
    actions=[T, ...],
    rewards=[T],
    dones=[T],
)
```

这里有 `T` 个动作，却有 `T + 1` 次观察。原因很简单：执行第一个动作以前要先看一次，执行最后一个动作以后还要再看一次。

一条可学习的 transition 可以写成：

```text
当前观察 o_t
→ 执行动作 a_t
→ 得到奖励 r_t
→ 看见下一观察 o_{t+1}
→ 记录任务是否结束 d_t
```

验证函数要检查长度、终止位置和时间戳。机器人数据还要保存语言指令、关节状态、控制频率和延迟，否则画面与动作可能错开。

## 小结

- [ ] 能解释为什么观察数量比动作数量多一。
- [ ] 能从 episode 中指出一条 transition。
- [ ] 能检查动作是否放在正确的两帧之间。

---

## 参考资料

### 实践博客

1. [OpenAI Spinning Up: Key Concepts of RL](https://spinningup.openai.com/en/latest/) —— OpenAI 官方入门教程，把 episode、轨迹、价值这些术语一次讲清。
2. [A (Long) Peek into Reinforcement Learning (Lilian Weng, 2018)](https://lilianweng.github.io/posts/2018-02-19-rl-overview/) —— 从 MDP 到模型法 RL 的全景综述博客，适合给本章术语找上下文。
3. [Grokking Deep Reinforcement Learning (Kaixhin)](https://kaixhin.com/grokking-deep-reinforcement-learning/) —— 长文系列，逐层拆解深度 RL 的实践细节与常见坑。
4. [Human-level control through deep reinforcement learning (DeepMind, 2015)](https://deepmind.google/discover/blog/human-level-control-through-deep-reinforcement-learning/) —— DQN 发布时的官方博客，配本章经验回放的历史起点。
5. [Reinforcement Learning: An Introduction (Sutton & Barto, 2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html) —— 免费在线教材，本章用到的 MDP 与表格学习均可在此查证。

### 经典文献

1. [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://www.nature.com/articles/nature14236) —— DQN 的 Nature 论文，经验回放进深度学习的标志性工作。
2. [Prioritized Experience Replay (Schaul et al., 2016)](https://arxiv.org/abs/1511.05952) —— 按重要性取样回放，是本章“怎样取样”问题的直接回答。
3. [Rainbow: Combining Improvements in Deep RL (Hessel et al., 2018)](https://arxiv.org/abs/1710.02298) —— 把回放、分布、双 Q 等改进逐项消融，示范了数据设计怎样影响成绩。
4. [Revisiting Fundamentals of Experience Replay (Fedus et al., 2020)](https://arxiv.org/abs/2007.06700) —— 系统研究回放的容量、采样与保留策略，直接服务 3.2。
5. [Deep Recurrent Q-Learning for Partially Observable MDPs (Hausknecht & Stone, 2015)](https://arxiv.org/abs/1507.06527) —— 用循环网络补状态信息，呼应本章 episode 与隐状态的关系。
