# 3.1 强化学习数据结构：回合（Episodes）与状态转移（Transitions）

深度学习在计算机视觉和自然语言处理等领域取得了巨大成功，这在很大程度上归功于其对静态数据集（如图像或文本语料库）的有效利用。然而，强化学习（Reinforcement Learning, RL）面临着截然不同的数据环境。在强化学习中，智能体（Agent）并非被动地接收标注好的数据，而是通过与环境（Environment）进行主动交互来收集数据。这种交互产生的数据天然具有时间维度的序列依赖性。

早期的强化学习研究（如 [[Sutton & Barto, 1998]](http://incompleteideas.net/book/first/the-book.html) ）为我们建立了一套基于马尔可夫决策过程（Markov Decision Process, MDP）的数学框架。而在深度强化学习的黎明时期，[[Mnih et al., 2013]](https://arxiv.org/abs/1312.5602) 在深度Q网络（Deep Q-Network, DQN）的开创性论文中，巧妙地引入了“经验回放”（Experience Replay）机制。这一机制打破了时间序列的强相关性，将连续的经验序列拆解为离散的“状态转移”（Transitions），并允许我们在训练深度神经网络时像处理传统监督学习那样进行小批量（Mini-batch）采样。

在本节中，我们将从最基础的序列与函数映射出发，严格定义强化学习中数据的两种核心结构：微观的**状态转移**与宏观的**回合**（Episodes），并探讨如何用张量（Tensor）严谨地表达与存储这些数据。

## 3.1.1 序列交互与变量定义

为了准确描述智能体与环境的交互，我们需要引入一组随时间变化的数学变量。在高中数学中，我们学习过数列的概念，即按一定顺序排列的数字。在强化学习中，时间并非连续流淌的实数，而是被离散化为一系列时间步（Time steps）。我们用离散的非负整数变量 $t = 0, 1, 2, \dots$ 来表示当前的时间步。

在每一个时间步 $t$，环境会呈现出一个特定的**状态**（State），我们记为 $s_t$。状态是对当前环境物理属性的完整或部分描述。例如，在一个一维物理世界中，状态可能仅仅是物体的位置坐标；在更复杂的环境中，状态可以表示为一个包含多个特征的向量 $s_t \in \mathbb{R}^n$，其中 $n$ 是状态的维度。

智能体在观察到状态 $s_t$ 后，需要做出一个**动作**（Action），记为 $a_t$。动作同样可以是一个标量或者一个高维向量 $a_t \in \mathbb{R}^m$。一旦智能体执行了动作 $a_t$，环境会根据其内部的物理规律发生改变，转移到一个新的状态 $s_{t+1}$。

同时，环境会反馈给智能体一个标量信号，称为**奖励**（Reward），记为 $r_t \in \mathbb{R}$。奖励用于量化智能体在时间步 $t$ 所执行动作的好坏。为了清晰起见，我们将单步的因果关系用以下函数形式表示：

$$
s_{t+1}, r_t = f_{\text{env}}(s_t, a_t)
$$

上述该公式描述了一个确定性环境。在更严谨的学术表达中，状态的转移和奖励的生成往往服从一定的概率分布，即状态转移概率 $P(s_{t+1} | s_t, a_t)$ 和奖励函数 $R(s_t, a_t)$。但无论底层逻辑是确定性还是随机性，从数据收集的视角来看，我们记录下的仅仅是这一交互过程的观测值。

此外，交互过程可能并非无限进行。当满足某些特定条件（例如任务成功、智能体发生碰撞或达到最大时间限制）时，交互会终止。为了标记这种终止状态，我们引入一个布尔（Boolean）变量，称为**终止标志**（Done flag），记为 $d_t \in \{0, 1\}$。当 $d_t = 1$ 时，意味着在获取状态 $s_{t+1}$ 后，整个交互序列宣告结束。

## 3.1.2 状态转移（Transitions）：经验的原子单元

当我们审视该公式所描述的物理过程时，可以提取出一个完整的、不可分割的“交互循环”。这个循环包含了“观察状态 $\rightarrow$ 执行动作 $\rightarrow$ 获得奖励与新状态”的完整因果链条。我们将记录这一单步循环的数据结构定义为**状态转移**（Transition）。

对于任意时间步 $t$，一个状态转移 $e_t$ 被严格定义为一个包含五个元素的多元组（Tuple）：

$$
e_t = (s_t, a_t, r_t, s_{t+1}, d_t)
$$

在这个五元组中，各个变量的物理含义和数据类型如下：
* $s_t$：当前状态向量。
* $a_t$：执行的动作向量。
* $r_t$：标量奖励信号。
* $s_{t+1}$：执行动作后到达的下一个状态向量。
* $d_t$：标量终止标志，通常在实现中用浮点数 `0.0` 或 `1.0` 表示。

状态转移是深度强化学习中最原子的数据单元。为什么需要如此定义？因为这五个元素恰好满足了更新模型参数所需的全部信息。当我们使用神经网络来近似强化学习中的核心数学量（如价值函数或策略分布）时，我们关注的是当前时刻的决定对未来的影响。五元组该公式在提供当前上下文（$s_t, a_t$）的同时，也提供了环境的即时反馈（$r_t, d_t$）和未来的起始点（$s_{t+1}$）。

在实际工程实现中，为了充分利用现代硬件（如 GPU）的并行计算能力，我们极少针对单个状态转移进行计算。相反，我们会从经验回放缓冲区中随机采样出多个状态转移，将它们拼接（Stack）成一个批次（Batch）。假设我们采样了 $B$ 个状态转移，我们将原本一维或零维的独立变量沿着一个新的维度（称为批量维度，Batch dimension）进行拼接。此时，我们得到的变量张量维度将发生如下变化：
* 状态批量张量 $\mathbf{S}_t \in \mathbb{R}^{B \times n}$
* 动作批量张量 $\mathbf{A}_t \in \mathbb{R}^{B \times m}$
* 奖励批量张量 $\mathbf{R}_t \in \mathbb{R}^{B \times 1}$
* 下一状态批量张量 $\mathbf{S}_{t+1} \in \mathbb{R}^{B \times n}$
* 终止标志批量张量 $\mathbf{D}_t \in \mathbb{R}^{B \times 1}$

这种严密的张量对齐，是确保我们在后续构建复杂损失函数（Loss functions）时避免维度广播（Broadcasting）错误的基础。

## 3.1.3 回合（Episodes）：时间的宏观轨迹

如果我们把状态转移视为离散的点，那么将这些点按照时间先后顺序连结起来，就构成了一条连续的线。在强化学习中，这条线被称为一条**轨迹**（Trajectory）或一个**回合**（Episode）。

一个回合 $\tau$ 是从初始状态 $s_0$ 开始，直到遇到终止状态（即某个 $d_t = 1$）为止，所产生的所有状态转移序列的完整集合。数学上，我们可以将其表示为：

$$
\tau = (e_0, e_1, e_2, \dots, e_T)
$$

或者将其展开为变量序列：

$$
\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T, a_T, r_T, s_{t+1})
$$

这里的 $T$ 表示该回合的最终时间步。需要注意的是，$T$ 并不是一个固定的常数。由于智能体采取不同的策略，环境可能会在不同的时间步触发终止条件，因此不同回合的长度通常是可变的。

> [!NOTE]
> 在某些特殊场景中（例如永续任务，Continuing Tasks），理论上的回合长度可能是无限的（$T \to \infty$）。但在实际实现中，受限于计算机内存与计算资源的限制，我们通常会人为设定一个最大时间步长（Max steps），即发生**截断**（Truncation）。为了严谨起见，在高级实现中，研究者常将“自然终止”和“超时截断”使用两个不同的标志位来区分，但在我们的基础实现中，我们暂且将两者合并为 $d_t$。

回合级别的数据结构通常用于策略梯度（Policy Gradient）类算法。这类算法（如 REINFORCE 或 PPO）往往需要计算整个回合的累积回报（Cumulative Return）来评估当前策略的好坏。而在基于价值（Value-based）的方法（如 DQN）中，回合的边界意义仅仅在于截断时序差分（Temporal Difference）目标的计算传播。

## 3.1.4 数据结构的张量实现

接下来，我们将从数学推导过渡到实际的代码实现。在深度学习框架中，最有效率的存储和处理方式是直接使用张量（Tensor）。考虑到数据批次（Batch）的高效读取，我们会设计一个字典或者自定义的数据类（Data Class），在内部维护大容量的张量内存空间。

(**下面我们用代码来实现一个基础的状态转移缓冲存储结构**)：`TransitionBatch`。为了降低系统开销，我们通常会在初始化时就预先分配好连续的张量内存，在交互过程中仅仅通过移动指针（Index pointer）来覆盖旧数据。这种预分配策略是现代强化学习库的标准做法。

```{.python .input}
#@tab pytorch
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

在这个实现中，通过 `torch.zeros` 预先在指定的硬件设备（如 GPU）上分配大块连续内存。当调用 `add` 方法时，我们仅仅是用新的观测张量覆盖对应行的切片（Slice）。这种方法有效避免了在 Python 循环中频繁申请内存或使用诸如 `list.append()` 导致的大量列表重新分配和拼接操作，从而极大提升了数据收集的吞吐量。

## 3.1.5 小结

* 强化学习与监督学习的本质区别在于其数据是随时间由智能体主动交互产生的。时间序列依赖是其核心挑战之一。
* **状态转移** $e_t = (s_t, a_t, r_t, s_{t+1}, d_t)$ 是打破强序列依赖、实施小批量模型训练的最基础数据构造单元。
* **回合** $\tau$ 构成了多步转移的时序轨迹，决定了状态评估的长远视界（Horizon）。
* 为了计算高效性，实现中通常通过张量在内存中预先连续分配以完成数据缓冲区的构建，从而使得经验的小批量抽取具有极低的系统延迟。

## 3.1.6 练习

1. 在实际训练中，如果环境给予的状态是一个形状为 `(3, 84, 84)` 的彩色图像矩阵，那么在上述的 `TransitionBatch` 中，状态张量 `self.states` 的最终张量形状应该是什么样的？（提示：考虑容量参数和批量拼接的维度增加）。
2. 我们在相关章节中将奖励张量的维度设定为 $\mathbb{R}^{B \times 1}$ 而不是一维张量 $\mathbb{R}^{B}$。这种做法在与计算模型预测的 $Q$ 值进行均方误差损失（MSE Loss）计算时有什么优势？（提示：回顾一下深度学习框架在执行四则运算时的“广播机制” Broadcasting 规则，以及如果维度不严格对齐可能会导致的严重逻辑错误）。
3. 假设你的 `TransitionBatch` 容量为 10，当前已经存放了 12 条独立的状态转移数据。此时调用 `sample(5)`，你能保证采样出的样本中不包含前 2 条数据吗？（提示：阅读代码中 `self.pointer` 所采取的模除 `%` 操作机制，分析这是一种什么样的数据结构策略）。
