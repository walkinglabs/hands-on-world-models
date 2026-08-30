# 经验回放缓冲区与数据切分
:label:sec_replay_buffer

在强化学习与世界模型的训练中，数据并非像传统的图像分类任务那样静态地以完整的数据集形式存在。相反，智能体（Agent）在与环境的持续交互中，不断生成如流水线般的时间序列数据。这种在线数据收集方式带来了两个致命的挑战：时间相关性（Temporal Correlation）与样本效率（Sample Efficiency）。为了克服这两个挑战，经验回放（Experience Replay）机制应运而生。

早在强化学习探索的早期，Lin (1992) `[Lin, 1992]` 就探讨了在连接主义模型中缓存过去经验的思想，但真正让其在深度学习时代声名大噪的，是 DeepMind 团队在处理雅达利（Atari）游戏时提出的深度Q网络（DQN）`[Mnih et al., 2013]`。通过引入经验回放缓冲区，DQN 成功地将深度神经网络与强化学习稳定地结合在一起。在本节中，我们将从最基础的统计假设出发，严谨推导经验回放缓冲区的数学机制，并将其物理实现映射到张量操作中。

## 打破时间相关性：从独立同分布假设起步

在高中统计学与基础机器学习中，我们总是假定数据满足“独立同分布”（Independent and Identically Distributed, i.i.d.）。当我们试图拟合一条直线或者训练一个简单的回归模型时，每一次随机抽取的数据点，都不应该受到上一次抽取结果的影响。

设想我们在优化一个损失函数 $\mathcal{L}(\theta)$，根据随机梯度下降（SGD），参数的更新规则为：

$$ \theta_{t+1} = \theta_t - \alpha \nabla_{\theta} \mathcal{L}(x_t, y_t; \theta) $$
:eqlabel:eq_sgd_update

为了保证梯度下降能够无偏地逼近整体期望梯度，即 $\mathbb{E}_{(x, y)} [\nabla \mathcal{L}] = \nabla \mathbb{E}[\mathcal{L}]$，每次采样的样本 $(x_t, y_t)$ 必须是从总体数据分布中独立抽取的。

然而，在强化学习中，智能体在时间步 $t$ 观测到状态 $s_t$，采取动作 $a_t$，获得奖励 $r_t$，并转移到下一个状态 $s_{t+1}$。这一系列变量构成了马尔可夫决策过程（MDP）中的一条轨迹（Trajectory）。由于物理规律的连续性，$s_{t+1}$ 极大概率在空间上与 $s_t$ 极度接近。如果我们直接按照时间顺序 $(s_1, s_2, \dots)$ 将这些高度相关的数据持续喂给神经网络，模型将会在一段局部时间内只看到高度相似的特征，导致梯度更新的方向产生严重偏差。这种连续的同质化梯度会使得网络极其迅速地“遗忘”之前学过的其他状态的知识，引发灾难性遗忘（Catastrophic Forgetting）。

> 引用块类比：
> 想象一个试图学习“驾驶”的系统。如果它连续一小时都在笔直的高速公路上行驶，它会不断调整权重以强化“保持直行”的指令。当它突然驶出高速进入连续弯道时，由于参数已经极度拟合直行，面对转弯它将不知所措。解决之道并非只学当前，而是将其在高速公路和弯道的记忆混合起来，每次随机抽取一段来复习。这是全篇唯一一次类比，旨在说明在线连续学习时为何会发生权重坍缩。

这就是经验回放缓冲区的核心动机：通过构建一个容量庞大的数据池，存储过去的历史经验，并在训练时从中进行均匀随机采样，从而人为地打破样本之间的时间相关性，使其重新近似满足独立同分布的假设。同时，同一条经验可以被反复抽取和学习，极大地提升了样本效率。

## 经验回放的数学公式与张量表示

我们将智能体在单个时间步与环境交互产生的“经验片段”（Transition）定义为一个元组：

$$ e_t = (s_t, a_t, r_t, s_{t+1}, d_t) $$
:eqlabel:eq_transition_tuple

其中 $d_t \in \{0, 1\}$ 表示当前状态是否为终止状态（Done）。

经验回放缓冲区 $\mathcal{D}$ 本质上是一个最大容量为 $N$ 的集合。随着时间的推移，我们不断将新的经验加入集合中：

$$ \mathcal{D} = \{ e_1, e_2, \dots, e_{|\mathcal{D}|} \} $$
:eqlabel:eq_buffer_set

在每一次模型更新时，我们从 $\mathcal{D}$ 中均匀随机地抽取一个批量大小（Batch Size）为 $B$ 的小批量（Mini-batch）数据 $\mathcal{B}$。对于批量中的每一个样本 $i \in \{1, 2, \dots, B\}$，从集合中抽取特定经验 $e_k$ 的概率为：

$$ P(e_i = e_k) = \frac{1}{|\mathcal{D}|} $$
:eqlabel:eq_uniform_sampling

在张量（Tensor）表示下，如果我们假设状态向量长度为 $D_s$ （即 $s \in \mathbb{R}^{D_s}$），动作向量长度为 $D_a$ （即 $a \in \mathbb{R}^{D_a}$），那么抽取出的一个小批量 $\mathcal{B}$ 就可以被整齐地堆叠为多个多维张量。具体维度如下：

- 状态张量（States）：$\mathbf{S} \in \mathbb{R}^{B \times D_s}$
- 动作张量（Actions）：$\mathbf{A} \in \mathbb{R}^{B \times D_a}$
- 奖励张量（Rewards）：$\mathbf{R} \in \mathbb{R}^{B \times 1}$
- 下一状态张量（Next States）：$\mathbf{S}' \in \mathbb{R}^{B \times D_s}$
- 终止标志张量（Dones）：$\mathbf{D} \in \mathbb{R}^{B \times 1}$

这种矢量的矩阵堆叠形式（Vectorization）正是利用 GPU 进行大规模并行矩阵乘法的物理基础。所有离散的标量运算都被整合为严密的代数运算。

## 环形缓冲区（Ring Buffer）的物理实现

在实际的计算机系统中，物理内存是有限的，缓冲区容量 $N$ 不可能无限大。为了保证内存不会溢出，并且模型能够逐步淘汰过于陈旧的低质量经验（比如智能体初期盲目探索的数据），经验回放缓冲区通常被实现为一个**环形缓冲区（Ring Buffer）**。

环形缓冲区的核心数学原理建立在基础代数中的**模运算（Modulo Arithmetic）**之上。我们预先分配一个长度为 $N$ 的连续内存数组，并维护一个写入指针 $p$ ，它代表当前最新数据应该写入的位置。初始时 $p=0$。

当新来一条经验 $e_t$ 时，我们执行以下逻辑：
1. 将 $e_t$ 写入数组的第 $p$ 个位置。
2. 更新指针至下一个位置：

$$ p \leftarrow (p + 1) \pmod N $$
:eqlabel:eq_ring_buffer_modulo

由于模运算的存在，当存入的数据量超过容量 $N$ 时，新数据会自动覆盖掉最古老的数据（即回到索引 $0$ 的位置开始重新覆盖）。这样我们在 $\mathcal{O}(1)$ 的恒定时间复杂度内实现了数据的先进先出（FIFO）淘汰机制，同时彻底避免了内存的动态反复分配。

## 序列世界模型中的高级挑战：序列采样与截断

在标准强化学习中，上述的独立同分布单步随机采样已经足够。然而，在构建**世界模型（World Models）**时，我们的目标不仅仅是拟合单步的价值，而是要通过循环神经网络（RNN）或 Transformer 来预测未来的连续演化轨迹。此时，我们必须从缓冲区中抽取长度为 $L$ 的连续序列块（Sequence Chunks）。

假设序列长度为 $L$，每次采样的不再是孤立的 $e_t$，而是一个完整的序列：

$$ \tau_{t:t+L} = (s_t, a_t, s_{t+1}, a_{t+1}, \dots, s_{t+L}) $$
:eqlabel:eq_sequence_chunk

这就引入了额外的物理约束：**环境截断（Episode Boundary）**。在真实世界或游戏中，一段交互随时可能因为失败或通关而终止（$d_k = 1$）。如果我们在采样一段长度为 $L$ 的序列时，恰好跨越了这个终止边界，那么序列的前半部分属于上一局游戏，后半部分属于新一局游戏。这种不连续的数据如果强行喂给世界模型的 RNN，会导致模型试图去寻找两局毫无关联的游戏之间的“物理连贯性”，从而彻底破坏内部的状态表征。

因此，在序列张量的处理中，当读取到 $d_k = 1$ 的终止标志时，我们需要在随后的计算步骤中使用“掩码（Mask）”或重置 RNN 的隐藏状态（Hidden States），确保梯度不会逆向流过截断边界。

同时，为了评估世界模型是否真正学到了环境的内在动态规律，而不是简单地死记硬背了缓冲区里的训练轨迹，我们必须引入严谨的**数据切分（Data Splits）**机制。时间序列数据的切分方式必须避免未来信息泄露：通常我们在交互初期将完整回合（Episodes）或一定时间块独立分配给专门的验证集缓冲区（Validation Buffer），该缓冲区仅用于评估测试损失，绝不参与梯度反向传播。

## 代码实现：构建高效环形缓冲区

现在，我们将这些严谨的代数关系映射为具体的 PyTorch 与 NumPy 代码。我们利用 NumPy 数组作为底层的连续内存，以支持极速的内存分配和索引。

```{.python .input}
#@tab pytorch
import torch
import numpy as np
from typing import Tuple

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: str = 'cpu'):
        """
        初始化经验回放缓冲区。
        分配固定大小的连续内存块以提升读取效率，避免动态申请内存造成的碎片化。
        """
        self.capacity = capacity
        self.device = device
        
        # 预分配连续内存数组，利用 Float32 保证浮点运算的数值精度
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        # p 为写入指针，size 记录当前实际有效存储的数据量
        self.p = 0
        self.size = 0

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool):
        """
        (**将单步经验存入环形缓冲区**)
        """
        self.states[self.p] = state
        self.actions[self.p] = action
        self.rewards[self.p] = reward
        self.next_states[self.p] = next_state
        self.dones[self.p] = float(done)
        
        # 严格遵守模运算公式更新指针位置
        self.p = (self.p + 1) % self.capacity
        # 记录真实数据量，直到达到最大容量
        self.size = min(self.size + 1, self.capacity)

    def sample_transitions(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        (**均匀随机采样单步经验小批量**)
        """
        # 利用 numpy 随机生成索引，采用无放回采样
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        # 将底层切片数据立即转换为 PyTorch 张量并发送至指定的计算设备
        s = torch.tensor(self.states[indices], device=self.device)
        a = torch.tensor(self.actions[indices], device=self.device)
        r = torch.tensor(self.rewards[indices], device=self.device)
        s_next = torch.tensor(self.next_states[indices], device=self.device)
        d = torch.tensor(self.dones[indices], device=self.device)
        
        return s, a, r, s_next, d
```

如上所述，上述实现解决了最基础的单步马尔可夫决策数据需求。接下来，我们拓展缓冲区，使其支持具有时间连贯性的序列采样（Sequence Sampling），这是训练序列世界模型必不可少的一步。

```{.python .input}
#@tab pytorch
    def sample_sequences(self, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, ...]:
        """
        (**采样具有时间连续性的经验序列**)
        """
        # 防止采样到尚未被填充数据的末端，或越界情况
        valid_start_size = self.size - seq_len
        if valid_start_size <= 0:
            raise ValueError("缓冲区当前数据量不足以采样指定长度的序列。")
            
        # 注意：在完全严谨的生产环境中，这里还需要过滤掉跨越指针 p 的错位序列
        start_indices = np.random.choice(valid_start_size, batch_size, replace=False)
        
        # 初始化列表，用于按时间步展开序列
        seq_s, seq_a, seq_r, seq_d = [], [], [], []
        
        for i in range(seq_len):
            idx = start_indices + i
            seq_s.append(self.states[idx])
            seq_a.append(self.actions[idx])
            seq_r.append(self.rewards[idx])
            seq_d.append(self.dones[idx])
            
        # 在时间轴（axis=1）上堆叠序列并转换为张量
        # 变换后的张量维度为：(Batch Size, Sequence Length, Feature Dimension)
        s = torch.tensor(np.stack(seq_s, axis=1), device=self.device)
        a = torch.tensor(np.stack(seq_a, axis=1), device=self.device)
        r = torch.tensor(np.stack(seq_r, axis=1), device=self.device)
        d = torch.tensor(np.stack(seq_d, axis=1), device=self.device)
        
        return s, a, r, d
```

## 小结

- 经验回放利用历史数据缓存并打乱提取顺序，人为打破了在线强化学习中序列数据极强的时间相关性，重新逼近独立同分布假设。
- 利用基础模运算（Modulo Arithmetic）构建的环形缓冲区，在保证极高运行效率的同时，将内存空间限制在了可控范围内。
- 面向世界模型的序列采样，引入了时间维度，但这也伴随着如何处理环境截断边界的严峻挑战，其核心在于正确维护隐藏状态的连续性。
- 分离训练和验证缓冲区的数据切分策略，是阻止网络单纯记忆并评估其泛化能力的必要屏障。

## 练习

1. 在 `sample_sequences` 模块中，如果起始索引 `start_indices` 对应的序列区间内发生了 `done=True`，这意味着环境状态被重置。如果不作处理，这会对世界模型的动力学前向预测造成什么潜在危害？
   - *提示*：考虑物理状态在重置前后的变化率规律是否连续。
2. 尝试从概率角度推导：如果我们不再使用“满后覆旧”的环形覆盖机制，而是当缓冲区满时直接随机抛弃一半历史数据。这种做法会对数据分布 $P(e_i \sim \mathcal{D})$ 造成什么样的影响？与 FIFO 相比，哪种更利于学习最新的环境动态？
3. (编程挑战) 在目前 `sample_sequences` 的实现中，我们没有处理一种极其特殊的边界情况：当容量存满并发生环形覆盖后，指针 `p` 刚刚覆盖了索引 100 的位置，如果我们刚好抽取到起始索引为 98 且长度为 5 的序列，序列前半部是陈旧数据，后半部分却被更新为了最新数据。请尝试重写代码逻辑，严密排除这种发生越界错位的非法切片。

:begin_tab:pytorch
[讨论](https://discuss.d2l.ai/t/1234)
:end_tab:
