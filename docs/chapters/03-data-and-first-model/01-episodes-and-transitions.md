# 3.1 强化学习数据结构：回合（Episodes）与状态转移（Transitions）

> **本章导读**
>
> **讲什么：** 本章把上一章的神经网络积木接到“行动会改变未来”的数据上。我们会学习怎样记录一次交互、怎样切分和重放序列、怎样评价动作的长期结果，以及怎样借助模型预测控制在多个候选动作中作出选择。
>
> **为什么不能继续使用普通监督学习的做法：** 图像分类中的样本通常互不影响；交互数据却按时间相连，今天选择的动作还会改变明天能看到的数据。若忽略回合边界、时间相关性和动作带来的分布变化，即使训练损失很低，模型也可能学到泄漏的未来信息，或在闭环中迅速失效。
>
> **故事线：** `把经历写成转移与回合 → 正确存储、采样和切分 → 用价值与策略衡量长期结果 → 用 MPC 与 CEM 搜索动作 → 组装第一个数据—学习—决策闭环`

图像分类的数据在训练开始前通常已经固定；强化学习（Reinforcement Learning, RL）的数据却由智能体（Agent）与环境（Environment）边行动边产生。当前动作不仅贡献一条训练样本，还会改变下一时刻的状态以及以后能够收集到的数据。因此，交互记录必须保留时间顺序和回合边界。

<div align="center">
  <img src="/figures/03-data-and-first-model/source/01-episodes-and-transitions/dqn-fig1.png" alt="五个 Atari 画面把抽象的状态、动作与回合落到智能体实际接收的逐帧观测上。" width="86%">

_图 3.1-1：五个 Atari 画面把抽象的状态、动作与回合落到智能体实际接收的逐帧观测上。 出处：Volodymyr Mnih et al.，[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)（2013），Figure 1。_

</div>

强化学习通常用马尔可夫决策过程（Markov Decision Process, MDP）描述状态、动作、奖励与转移 [[Sutton & Barto, 1998]](http://incompleteideas.net/book/first/the-book.html)。Lin 较早系统研究了把过去经验保存并在后续学习中重放的机制 [[Lin, 1992]](https://doi.org/10.1007/BF00992699)；DQN 又把随机经验回放用于深度 Q 学习 [[Mnih et al., 2013]](https://arxiv.org/abs/1312.5602)。从缓冲区随机抽取转移可以减弱相邻样本的时间相关性，并支持小批量训练，但不会使样本在统计意义上自动独立。

本节先把一次交互写成**状态转移**，再把连续转移组织成**回合**（Episode），最后对应到批量张量的形状。

## 3.1.1 序列交互与变量定义

把交互看成按时间排列的序列，并用非负整数 $t=0,1,2,\dots$ 标记离散时间步。

在每一个时间步 $t$，环境会呈现出一个特定的**状态**（State），我们记为 $s_t$。状态是对当前环境物理属性的完整或部分描述。例如，在一个一维物理世界中，状态可能仅仅是物体的位置坐标；在更复杂的环境中，状态可以表示为一个包含多个特征的向量 $s_t \in \mathbb{R}^n$，其中 $n$ 是状态的维度。

<div align="center">
  <img src="/figures/03-data-and-first-model/source/01-episodes-and-transitions/lin-fig7.png" alt="网格环境中的智能体、敌人、食物与障碍说明一次动作会改变下一状态与可获奖励。" width="86%">

_图 3.1-2：网格环境中的智能体、敌人、食物与障碍说明一次动作会改变下一状态与可获奖励。 出处：Long-Ji Lin，[Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching](https://doi.org/10.1007/BF00992699)（1992），Figure 7。_

</div>

智能体在观察到状态 $s_t$ 后，需要做出一个**动作**（Action），记为 $a_t$。动作同样可以是一个标量或者一个高维向量 $a_t \in \mathbb{R}^m$。一旦智能体执行了动作 $a_t$，环境会根据其内部的物理规律发生改变，转移到一个新的状态 $s_{t+1}$。

同时，环境会反馈给智能体一个标量信号，称为**奖励**（Reward），记为 $r_t \in \mathbb{R}$。奖励用于量化智能体在时间步 $t$ 所执行动作的好坏。为了清晰起见，我们将单步的因果关系用以下函数形式表示：

$$
s_{t+1}, r_t = f_{\text{env}}(s_t, a_t)
$$

本节把动作 $a_t$ 后返回的奖励记为 $r_t$，便于与代码字段 `reward` 对齐。部分教材把同一个量记为 $r_{t+1}$；两种约定都可以，但同一条推导中不能混用。

这个函数式描述的是确定性环境。随机环境则用转移分布 $P(s_{t+1}\mid s_t,a_t)$ 与奖励分布或期望奖励 $R(s_t,a_t)$ 表示。无论环境是哪一种，缓冲区保存的都是一次实际交互得到的观测值。

此外，交互过程可能并非无限进行。当满足某些特定条件（例如任务成功、智能体发生碰撞或达到最大时间限制）时，交互会终止。为了标记这种终止状态，我们引入一个布尔（Boolean）变量，称为**终止标志**（Done flag），记为 $d_t \in \{0, 1\}$。当 $d_t = 1$ 时，意味着在获取状态 $s_{t+1}$ 后，整个交互序列宣告结束。

## 3.1.2 状态转移（Transitions）：经验的原子单元

“观察状态 $\rightarrow$ 执行动作 $\rightarrow$ 获得奖励与下一状态”构成一个单步交互。记录这条因果链的数据结构称为**状态转移**（Transition）。

对于任意时间步 $t$，一个状态转移 $e_t$ 被严格定义为一个包含五个元素的多元组（Tuple）：

$$
e_t = (s_t, a_t, r_t, s_{t+1}, d_t)
$$

<div align="center">
  <img src="/figures/03-data-and-first-model/latex/01-episodes-and-transitions/transition-shared-boundary.png" alt="相邻转移共享同一个下一状态，而终止标志把下一条转移划入新回合" width="86%">

_图 3.1-3：同一个 s_{t+1} 既是 e_t 的结果，也是下一转移的起点；d_t=1 时，这种状态索引关系不再表示同一回合的连续动力学。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

在这个五元组中，各个变量的物理含义和数据类型如下：

- $s_t$：当前状态向量。
- $a_t$：执行的动作向量。
- $r_t$：标量奖励信号。
- $s_{t+1}$：执行动作后到达的下一个状态向量。
- $d_t$：标量终止标志，通常在实现中用浮点数 `0.0` 或 `1.0` 表示。

这个五元组同时给出当前条件 $(s_t,a_t)$、即时反馈 $(r_t,d_t)$ 和下一步起点 $s_{t+1}$，足以构造常见的单步价值目标或动力学训练样本。多步方法还会把相邻转移按时间重新连接起来。

在实际工程实现中，为了充分利用现代硬件（如 GPU）的并行计算能力，我们极少针对单个状态转移进行计算。相反，我们会从经验回放缓冲区中随机采样出多个状态转移，将它们拼接（Stack）成一个批次（Batch）。假设我们采样了 $B$ 个状态转移，我们将原本一维或零维的独立变量沿着一个新的维度（称为批量维度，Batch dimension）进行拼接。此时，我们得到的变量张量维度将发生如下变化：

- 状态批量张量 $\mathbf{S}_t \in \mathbb{R}^{B \times n}$

<div align="center">
  <img src="/figures/03-data-and-first-model/latex/01-episodes-and-transitions/batch-stack-shapes.png" alt="B 条转移按字段列拆分并沿批量维堆叠成五类张量" width="86%">

_图 3.1-4：每条转移占一行，同名字段沿样本索引 i 堆叠；状态和动作保留特征维，奖励与终止标志保留单列。本文根据上式及其张量形状说明绘制；TikZ/LaTeX 编译。_

</div>

- 动作批量张量 $\mathbf{A}_t \in \mathbb{R}^{B \times m}$
- 奖励批量张量 $\mathbf{R}_t \in \mathbb{R}^{B \times 1}$
- 下一状态批量张量 $\mathbf{S}_{t+1} \in \mathbb{R}^{B \times n}$
- 终止标志批量张量 $\mathbf{D}_t \in \mathbb{R}^{B \times 1}$

这种严密的张量对齐，是确保我们在后续构建复杂损失函数（Loss functions）时避免维度广播（Broadcasting）错误的基础。

## 3.1.3 回合（Episodes）：时间的宏观轨迹

如果我们把状态转移视为离散的点，那么将这些点按照时间先后顺序连结起来，就构成了一条连续的线。在强化学习中，这条线被称为一条**轨迹**（Trajectory）或一个**回合**（Episode）。

<div align="center">
  <img src="/figures/03-data-and-first-model/source/01-episodes-and-transitions/drqn-fig1.png" alt="不同 Atari 游戏中的移动对象说明单帧观测为何必须按时间连接成回合或历史。" width="86%">

_图 3.1-5：不同 Atari 游戏中的移动对象说明单帧观测为何必须按时间连接成回合或历史。 出处：Matthew Hausknecht; Peter Stone，[Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527)（2015），Figure 1。_

</div>

设一个回合包含 $T$ 次动作，从 $s_0$ 开始，在第 $T-1$ 次转移后结束。它是有序序列而不是无序集合：

$$
\tau = (e_0, e_1, e_2, \dots, e_{T-1})
$$

或者将其展开为变量序列：

$$
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_{T-1}, a_{T-1}, r_{T-1}, s_T)
$$

这里 $T$ 是转移数。不同回合可能在不同时间满足终止条件，因此 $T$ 通常可变。

::: info 说明
永续任务在理论上没有自然终点，采样器仍会按固定长度切出片段。现代环境接口通常把**终止**（terminated）与**截断**（truncated）分开：前者表示任务进入终止状态，后者只是采样因时间上限等外部条件停止。后续计算 bootstrap 时，两者不能一概而论。下面的基础数据结构暂用一个 `done` 字段，生产实现应保留这一区别。
:::

回合级别的数据结构通常用于策略梯度（Policy Gradient）类算法。这类算法（如 REINFORCE 或 PPO）往往需要计算整个回合的累积回报（Cumulative Return）来评估当前策略的好坏。而在基于价值（Value-based）的方法（如 DQN）中，回合的边界意义仅仅在于截断时序差分（Temporal Difference）目标的计算传播。

## 3.1.4 数据结构的张量实现

代码中可以用字典或数据类维护预分配张量，避免每次写入都重新申请大块内存。

下面实现基础的状态转移存储结构 `TransitionBatch`。初始化时预分配连续张量，写满后用循环指针覆盖最旧位置。

```python
import torch

class TransitionBatch:
    """一个用于存储离散状态转移的经验批次类。"""
    def __init__(self, capacity, state_dim, action_dim, device="cpu"):
        self.capacity = capacity
        self.device = device

        # 预先分配固定大小的连续张量内存
        # 状态张量形状: (capacity, state_dim)
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)

        # 动作张量形状: (capacity, action_dim)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)

        # 标量信号往往只需要一维，形状: (capacity, 1)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

        # 维护一个指针用于指示当前插入的位置，以及一个计数器记录当前存储的数量
        self.pointer = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """将单个状态转移五元组写入缓冲区。"""
        # 将传入的数据转换为张量并送入指定设备
        # 假设传入的 state, action 为一维 numpy 数组或 list，reward, done 为标量
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor([done], dtype=torch.float32, device=self.device)

        # 将张量写入当前指针指向的内存行
        self.states[self.pointer] = state_tensor
        self.actions[self.pointer] = action_tensor
        self.rewards[self.pointer] = reward_tensor
        self.next_states[self.pointer] = next_state_tensor
        self.dones[self.pointer] = done_tensor

        # 移动指针（环形缓冲区逻辑：当到达容量上限时折返到起始位置 0）
        self.pointer = (self.pointer + 1) % self.capacity
        # 更新存储数量（取当前数量与容量之间的最小值）
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """随机采样指定数量的状态转移，返回一个批次（Batch）张量集合。"""
        # 利用 randint 在当前有效数据范围内生成随机索引
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        # 利用高级索引（Advanced indexing）并行抽取批次张量
        batch_states = self.states[indices]
        batch_actions = self.actions[indices]
        batch_rewards = self.rewards[indices]
        batch_next_states = self.next_states[indices]
        batch_dones = self.dones[indices]

        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
```

`torch.zeros` 一次性分配存储空间，`add` 只覆盖指针所在的行。这样可以减少动态扩容和反复拼接的开销。真实系统是否把缓冲区放在 GPU 上，要根据采集速度、显存和主机到设备的传输成本决定。

## 3.1.5 小结

- 强化学习与监督学习的本质区别在于其数据是随时间由智能体主动交互产生的。时间序列依赖是其核心挑战之一。
- **状态转移** $e_t = (s_t, a_t, r_t, s_{t+1}, d_t)$ 是打破强序列依赖、实施小批量模型训练的最基础数据构造单元。
- **回合** $\tau$ 构成了多步转移的时序轨迹，决定了状态评估的长远视界（Horizon）。
- 为了计算高效性，实现中通常通过张量在内存中预先连续分配以完成数据缓冲区的构建，从而使得经验的小批量抽取具有极低的系统延迟。

## 3.1.6 练习

1. 在实际训练中，如果环境给予的状态是一个形状为 `(3, 84, 84)` 的彩色图像矩阵，那么在上述的 `TransitionBatch` 中，状态张量 `self.states` 的最终张量形状应该是什么样的？（提示：考虑容量参数和批量拼接的维度增加）。
2. 我们在相关章节中将奖励张量的维度设定为 $\mathbb{R}^{B \times 1}$ 而不是一维张量 $\mathbb{R}^{B}$。这种做法在与计算模型预测的 $Q$ 值进行均方误差损失（MSE Loss）计算时有什么优势？（提示：回顾一下深度学习框架在执行四则运算时的“广播机制” Broadcasting 规则，以及如果维度不严格对齐可能会导致的严重逻辑错误）。
3. 假设你的 `TransitionBatch` 容量为 10，当前已经存放了 12 条独立的状态转移数据。此时调用 `sample(5)`，你能保证采样出的样本中不包含前 2 条数据吗？（提示：阅读代码中 `self.pointer` 所采取的模除 `%` 操作机制，分析这是一种什么样的数据结构策略）。
