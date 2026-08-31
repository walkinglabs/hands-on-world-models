# 经验回放缓冲区与数据切分

强化学习与世界模型的数据通常来自持续交互，而不是训练前固定好的样本集合。相邻状态相关，同一条交互经验又可能很昂贵。经验回放（Experience Replay）把已收集的转移保存起来，并在后续更新中重新采样，用来缓和时间相关性并提高数据复用率。

Lin 较早系统研究了缓存并重放过去经验的机制 [[Lin, 1992]](https://doi.org/10.1007/BF00992699)。DQN 随后把随机经验回放与目标网络结合，用卷积网络从 Atari 画面学习动作价值 [[Mnih et al., 2013]](https://arxiv.org/abs/1312.5602)。本节关注三个实现问题：单步转移怎样均匀采样、固定容量怎样循环覆盖、序列与数据切分怎样避免越过回合边界。

## 打破时间相关性：从独立同分布假设起步

在高中统计学与基础机器学习中，我们总是假定数据满足“独立同分布”（Independent and Identically Distributed, i.i.d.）。当我们试图拟合一条直线或者训练一个简单的回归模型时，每一次随机抽取的数据点，都不应该受到上一次抽取结果的影响。

设想我们在优化一个损失函数 $\mathcal{L}(\theta)$，根据随机梯度下降（SGD），参数的更新规则为：

$$ \theta_{t+1} = \theta_t - \alpha \nabla_{\theta} \mathcal{L}(x_t, y_t; \theta) $$

在常见正则条件下，从目标数据分布采样得到的小批量梯度可以估计总体期望梯度。独立同分布会让这一分析更直接，但神经网络训练并不要求每一次样本都严格独立；关键是理解相关采样会怎样改变梯度估计的方差与偏差。

在强化学习轨迹中，$s_{t+1}$ 由 $s_t$、$a_t$ 和环境随机性共同产生。连续样本往往来自相近区域，按时间顺序更新会让一个小批量覆盖的状态较窄，梯度也更相关。这样可能降低优化效率并加剧对近期经验的偏置，但后果取决于环境、算法和数据分布，不能简单等同于必然发生灾难性遗忘。

> 一段驾驶轨迹可能连续数百步都在直道上。若训练只使用最新片段，小批量几乎全是“保持直行”；把直道、弯道与纠偏片段混合采样，能让每次更新覆盖更宽的状态范围。

经验回放据此保存过去经验，并从较大的时间跨度中抽取训练批次。它能减弱相邻样本直接排在同一批中的相关性，并允许一条经验被多次使用；它不会让数据自动变成真正的独立同分布样本，也不会消除行为策略随训练变化造成的分布漂移。

<div align="center">
  <img src="/figures/03-data-and-first-model/source/02-replay-buffer-and-splits/per-fig1.png" alt="Blind Cliffwalk 链显示相同转移按不同顺序回放会显著改变价值传播速度。" width="86%">

_图 3.2-1：Blind Cliffwalk 链显示相同转移按不同顺序回放会显著改变价值传播速度。 出处：Tom Schaul et al.，[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)（2016），Figure 1。_

</div>

优先经验回放进一步指出，即使转移都保存在同一个缓冲区里，它们在当前训练阶段的学习价值也可能不同。把整块记忆中的 TD 误差画出来，可以看见少数高误差区域会随训练不断迁移；这正是后续设计非均匀采样概率的直觉来源。

<div align="center">
  <img src="/figures/03-data-and-first-model/source/02-replay-buffer-and-splits/per-fig10.png" alt="整块回放记忆中的 TD 误差热图显示不同转移在训练过程中具有不同学习价值。" width="86%">

_图 3.2-2：整块回放记忆中的 TD 误差热图显示不同转移在训练过程中具有不同学习价值。 出处：Tom Schaul et al.，[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)（2016），Figure 10。_

</div>

## 经验回放的数学公式与张量表示

我们将智能体在单个时间步与环境交互产生的“经验片段”（Transition）定义为一个元组：

$$ e_t = (s_t, a_t, r_t, s_{t+1}, d_t) $$

其中 $d_t \in \{0, 1\}$ 表示当前状态是否为终止状态（Done）。

经验回放缓冲区 $\mathcal{D}$ 本质上是一个最大容量为 $N$ 的集合。随着时间的推移，我们不断将新的经验加入集合中：

$$ \mathcal{D} = \{ e_1, e_2, \dots, e_{|\mathcal{D}|} \} $$

在每一次模型更新时，我们从 $\mathcal{D}$ 中均匀随机地抽取一个批量大小（Batch Size）为 $B$ 的小批量（Mini-batch）数据 $\mathcal{B}$。对于批量中的每一个样本 $i \in \{1, 2, \dots, B\}$，从集合中抽取特定经验 $e_k$ 的概率为：

<div align="center">
  <img src="/figures/03-data-and-first-model/source/02-replay-buffer-and-splits/rainbow-fig4.png" alt="Rainbow 的逐游戏消融图显示，把均匀回放改为优先经验回放后，各 Atari 任务性能变化并不相同。" width="86%">

_图 3.2-3：Rainbow 的逐游戏消融图显示，把均匀回放改为优先经验回放后，各 Atari 任务性能变化并不相同。 出处：Matteo Hessel et al.，[Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)（2018），Figure 4。_

</div>

$$ P(e_i = e_k) = \frac{1}{|\mathcal{D}|} $$

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

$$
p \leftarrow (p + 1) \pmod N
$$

<div align="center">
  <img src="/figures/03-data-and-first-model/latex/02-replay-buffer-and-splits/ring-buffer-modulo.png" alt="固定容量槽位上的写指针经模 N 运算从末端回绕到零号槽" width="86%">

_图 3.2-4：写入 p=N−1 后，(p+1) mod N 令下一指针回到 0，因此无需移动数组就能覆盖最旧槽位。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

由于模运算的存在，当存入的数据量超过容量 $N$ 时，新数据会自动覆盖掉最古老的数据（即回到索引 $0$ 的位置开始重新覆盖）。这样我们在 $\mathcal{O}(1)$ 的恒定时间复杂度内实现了数据的先进先出（FIFO）淘汰机制，同时彻底避免了内存的动态反复分配。

## 序列世界模型中的高级挑战：序列采样与截断

在标准强化学习中，上述的独立同分布单步随机采样已经足够。然而，在构建**世界模型（World Models）**时，我们的目标不仅仅是拟合单步的价值，而是要通过循环神经网络（RNN）或 Transformer 来预测未来的连续演化轨迹。此时，我们必须从缓冲区中抽取长度为 $L$ 的连续序列块（Sequence Chunks）。

<div align="center">
  <img src="/figures/03-data-and-first-model/source/02-replay-buffer-and-splits/drqn-fig2.png" alt="DRQN 以单帧卷积特征驱动 LSTM，直观说明序列块怎样为循环状态提供时间上下文。" width="86%">

_图 3.2-5：DRQN 以单帧卷积特征驱动 LSTM，直观说明序列块怎样为循环状态提供时间上下文。 出处：Matthew Hausknecht; Peter Stone，[Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527)（2015），Figure 2。_

</div>

假设序列长度为 $L$，每次采样的不再是孤立的 $e_t$，而是一个完整的序列：

$$
\tau_{t:t+L} = (s_t, a_t, s_{t+1}, a_{t+1}, \dots, s_{t+L})
$$

<div align="center">
  <img src="/figures/03-data-and-first-model/latex/02-replay-buffer-and-splits/sequence-boundary-mask.png" alt="连续序列在终止标志处重置隐藏状态并阻断跨回合梯度" width="86%">

_图 3.2-6：当序列内部出现 d_k=1，下一回合的隐藏状态必须重置，反向梯度也不能跨越这条边界。本文根据上式及边界说明绘制；TikZ/LaTeX 编译。_

</div>

这就引入了额外的物理约束：**环境截断（Episode Boundary）**。在真实世界或游戏中，一段交互随时可能因为失败或通关而终止（$d_k = 1$）。如果我们在采样一段长度为 $L$ 的序列时，恰好跨越了这个终止边界，那么序列的前半部分属于上一局游戏，后半部分属于新一局游戏。这种不连续的数据如果强行喂给世界模型的 RNN，会导致模型试图去寻找两局毫无关联的游戏之间的“物理连贯性”，从而彻底破坏内部的状态表征。

因此，在序列张量的处理中，当读取到 $d_k = 1$ 的终止标志时，我们需要在随后的计算步骤中使用“掩码（Mask）”或重置 RNN 的隐藏状态（Hidden States），确保梯度不会逆向流过截断边界。

验证集必须在序列边界上切分。优先把完整回合分配给训练集或验证集；连续任务则按互不重叠的时间块切分，并在边界留出必要间隔。先随机拆散单步转移再切分，会让同一轨迹中几乎相同的相邻状态同时进入训练与验证，从而高估泛化能力。验证缓冲区只用于评估，不参与参数更新或训练采样。

## 代码实现：构建高效环形缓冲区

下面用 NumPy 预分配底层数组，并在采样时转换为 PyTorch 张量。

```python
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
        (将单步经验存入环形缓冲区)
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
        (均匀随机采样单步经验小批量)
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

单步采样只需索引各行。序列采样还要先恢复环形数组中的时间顺序，再排除跨越回合终点的起点。

```python
    def sample_sequences(self, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, ...]:
        """
        (采样具有时间连续性的经验序列)
        """
        if self.size < seq_len:
            raise ValueError("缓冲区当前数据量不足以采样指定长度的序列。")

        # 未写满时 0 是最旧位置；写满后 p 指向下一次写入位置，也就是当前最旧数据。
        oldest = 0 if self.size < self.capacity else self.p
        chronological = (oldest + np.arange(self.size)) % self.capacity

        valid_sequences = []
        for start in range(self.size - seq_len + 1):
            idx = chronological[start:start + seq_len]
            # 最后一条转移可以终止，但不能在序列内部提前进入下一回合。
            if not self.dones[idx[:-1]].any():
                valid_sequences.append(idx)

        if len(valid_sequences) < batch_size:
            raise ValueError("满足回合边界约束的序列数量不足。")

        chosen = np.random.choice(len(valid_sequences), batch_size, replace=False)
        indices = np.stack([valid_sequences[i] for i in chosen], axis=0)

        # 在时间轴（axis=1）上堆叠序列并转换为张量
        # 变换后的张量维度为：(Batch Size, Sequence Length, Feature Dimension)
        s = torch.tensor(self.states[indices], device=self.device)
        a = torch.tensor(self.actions[indices], device=self.device)
        r = torch.tensor(self.rewards[indices], device=self.device)
        d = torch.tensor(self.dones[indices], device=self.device)

        return s, a, r, d
```

## 小结

- **经验回放**从更大的历史窗口随机抽样，减弱批内时间相关性并复用数据，但不保证样本严格独立同分布。
- 利用基础模运算（Modulo Arithmetic）构建的**环形缓冲区**，在保证极高运行效率的同时，将内存空间限制在了可控范围内。
- 面向世界模型的序列采样，引入了时间维度，但这也伴随着如何处理环境截断边界的严峻挑战，其核心在于正确维护隐藏状态的连续性。
- **分离训练和验证缓冲区**的数据切分策略，是阻止网络单纯记忆并评估其泛化能力的必要屏障。
