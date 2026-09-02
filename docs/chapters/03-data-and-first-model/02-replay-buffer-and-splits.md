# 3.2 经验回放池与数据集划分 (Replay Buffer & Splits)

在传统监督学习（如图像分类、目标检测）中，算法赖以稳定收敛的基石是**独立同分布假设（Independent and Identically Distributed, i.i.d.）**——每一个训练样本都是从全局静态数据集中随机独立抽取出来的。

然而，在智能体与物理环境实时交互的过程中，产生的数据流却具有极端严重的**时序强相关性（Temporal Correlation）**与**非平稳性（Non-Stationarity）**：智能体在某一时刻的状态 $\mathbf{s}_{t+1}$ 几乎就是上一时刻 $\mathbf{s}_t$ 的微小延续。如果直接将这种按时间顺序连续产生的数据直接喂给深度神经网络，网络会瞬间过度拟合当前局部的微小状态空间，并灾难性地遗忘数秒前学到的宝贵经验。

为了打破时序相关性并大幅提升宝贵交互数据的利用效率，DeepMind 在 DQN 中发扬光大了**经验回放池（Replay Buffer）**机制；随后，Schaul 等人进一步提出了**优先经验回放（Prioritized Experience Replay, PER）**，让智能体像人类攻克错题集一样，优先学习那些带来巨大意外与误差的高价值样本。

本节我们将从统计独立性与重要性采样出发，严密推导 PER 优先概率分布与无偏权重修正公式，并使用纯底层 PyTorch 从零手写一个工业级环形经验回放池。

<div align="center">

<img src="/figures/03-data-and-first-model/source/02-replay-buffer-and-splits/per-fig10.png" alt="DQN 深度强化学习框架：智能体在环境中交互并将转移元组存入经验回放池，随机抽样训练 Q 网络。" width="86%">

_图 3.2-1：DQN 深度强化学习框架：智能体在环境中交互并将转移元组存入经验回放池，随机抽样训练 Q 网络。 出处：[Human-level control through deep reinforcement learning，Volodymyr Mnih et al.，2015](https://www.nature.com/articles/nature14236)。_

</div>

---

## 3.2.1 统计与存储基石：打破时序相关性与环形缓存设计

要理解经验回放池的威力，我们首先需要从统计采样理论审视在线交互的隐患。

### 1. 局部自相关的“认知近视”
当机械臂在桌子左侧抓取物体时，连续数千步的数据全都是“机械臂在左侧”的局部样本。如果网络在这段时间内连续执行梯度下降，其权重会迅速偏向左侧控制；当机械臂随后移动到右侧时，网络又会把左侧学到的控制规律彻底冲刷破坏。

### 2. 均匀经验回放（Uniform Replay Buffer）
经验回放池在内存中维护一个固定容量上限为 $N$（如 $N = 10^6$）的**环形队列缓冲区（Circular Buffer）**：
- 智能体产生的新转移元组 $(\mathbf{s}_t, \mathbf{a}_t, r_t, \mathbf{s}_{t+1}, d_t)$ 依次写入队列；当缓存满时，最古老的陈旧数据被自动覆盖；
- 在策略更新时，系统从整个回放池中**均匀随机抽取**一个批次（Batch Size $B = 64$ 或 $256$）；
- 抽出的样本可能分别来自昨天、前天或一分钟前的完全不同任务场景，从而完美打破了时序相邻样本之间的高度相关性，重新构建起近似独立同分布的平稳优化环境！

<div align="center">

<img src="/figures/03-data-and-first-model/latex/02-replay-buffer-and-splits/sequence-boundary-mask.png" alt="环形经验回放池先进先出覆盖机制与均匀小批量随机抽样示意" width="86%">

_图 3.2-2：环形经验回放池先进先出覆盖机制与均匀小批量随机抽样示意。_

</div>

---

## 3.2.2 核心数学推导一：优先经验回放 (PER) 与重要性采样修正

在均匀随机采样中，所有样本被抽中的概率均等。然而在物理交互中，机器人 $90\%$ 的动作可能都是平凡的平稳滑行，只有 $10\%$ 的动作涉及关键的避障或高难度抓取。

<div align="center">

<img src="/figures/03-data-and-first-model/source/02-replay-buffer-and-splits/per-fig1.png" alt="优先经验回放 (PER) 在 Atari 游戏中展示通过重要性优先抽样获得的显著性能提升。" width="86%">

_图 3.2-3：优先经验回放 (PER) 在 Atari 游戏中展示通过重要性优先抽样获得的显著性能提升。 出处：[Prioritized Experience Replay，Tom Schaul et al.，2015](https://arxiv.org/abs/1511.05952)。_

</div>

PER 提出以**时间差分误差（TD Error）的大小作为样本优先度**：

$$p_i = |\delta_i| + \epsilon = |r_i + \gamma \max_{a'} Q(\mathbf{s}'_i, a') - Q(\mathbf{s}_i, a_i)| + \epsilon$$

其中 $\epsilon > 0$ 为防止概率为 0 的微小正常数。

### 1. 优先采样概率分布
第 $i$ 个样本被抽中的概率被定义为指数重加权形式：

$$P(i) = \frac{p_i^\alpha}{\sum_{k=1}^N p_k^\alpha}$$

其中 $\alpha \in [0, 1]$ 为优先级调节超参数（$\alpha = 0$ 退化为纯均匀采样，$\alpha = 1$ 为完全按误差比例采样）。

### 2. 重要性采样权重修正（Importance Sampling Weights）
频繁采样高 TD 误差的样本虽然加快了难点攻克，但却人为改变了状态访问的真实稳态分布，引入了系统性期望偏差。
为了抵消这种采样偏置，PER 引入了**重要性采样修正权重**：

$$w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^\beta$$

为保证梯度更新的尺度稳定性，权重除以当前批次的最大值进行归一化：

$$\tilde{w}_i = \frac{w_i}{\max_j w_j} \in (0, 1]$$

带重要性权重的加权梯度损失为：

$$\mathcal{L}_{\text{PER}} = \frac{1}{B} \sum_{i=1}^B \tilde{w}_i \cdot \delta_i^2$$

### 3. PER 权重手算数值算例
设回放池中总共有 $N = 2$ 个样本，优先级超参数 $\alpha = 1.0, \beta = 1.0$：
- **样本 1（极其意外的高难度避障）**：TD 误差 $p_1 = 4.0$；
- **样本 2（平淡无奇的直行）**：TD 误差 $p_2 = 1.0$。

我们来手动计算各样本的采样概率与重要性权重：
1. **计算采样概率**：
   $$\text{Sum} = 4.0^1 + 1.0^1 = 5.0$$
   $$P(1) = \frac{4.0}{5.0} = 0.80, \quad P(2) = \frac{1.0}{5.0} = 0.20$$
2. **计算未归一化重要性权重 $w_i = \frac{1}{2 \times P(i)}$**：
   $$w_1 = \frac{1}{2 \times 0.80} = \frac{1}{1.60} = 0.625$$
   $$w_2 = \frac{1}{2 \times 0.20} = \frac{1}{0.40} = 2.500$$
3. **归一化权重（除以最大值 $2.500$）**：
   $$\tilde{w}_1 = \frac{0.625}{2.500} = 0.25, \quad \tilde{w}_2 = \frac{2.500}{2.500} = 1.00$$

初等代数的几步推导极其直观：样本 1 虽然被抽中的概率高达 $80\%$（高频复习），但它在单次梯度更新时的权重被压低为 $0.25$；而样本 2 虽然抽中概率仅 $20\%$，但一旦被抽中其梯度权重为 $1.00$。
**二者乘积 $P(1) \times \tilde{w}_1 = 0.80 \times 0.25 = 0.20$ 与 $P(2) \times \tilde{w}_2 = 0.20 \times 1.00 = 0.20$ 完全恒等！** 这种双向平衡既享受了优先采样的训练加速，又在数学期望上保持了严格的无偏性！

<details>
<summary><b>深入推导：基于重要性采样的拉东-尼科迪姆导数（Radon-Nikodym Derivative）无偏梯度估计证明（点击展开查看完整推导）</b></summary>

设真实均匀期望为 $\mathbb{E}_{U}[\delta(\theta)] = \int \delta(\theta, x) \frac{1}{N} dx$。
当根据提议分布 $P(x)$ 采样时，根据概率测度变换公式：
$$\mathbb{E}_{P} [w(x) \delta(\theta, x)] = \int \left( \frac{1}{N \cdot P(x)} \right) \delta(\theta, x) P(x) dx = \int \delta(\theta, x) \frac{1}{N} dx = \mathbb{E}_{U}[\delta(\theta)]$$
比值 $\frac{d\mathbb{P}_U}{d\mathbb{P}_P}(x) = \frac{1}{N P(x)}$ 构成了严密的拉东-尼科迪姆导数。当退火指数 $\beta \to 1$ 时，加权梯度估计量的一阶矩严格渐近无偏。
</details>

---

## 3.2.3 核心数学推导二：时序数据集划分（Episode-Level Split）与数据泄露防御

在训练用于世界模型的状态预测器时，如何将数据集划分为训练集（Train）、验证集（Val）与测试集（Test）？

<div align="center">

<img src="/figures/03-data-and-first-model/latex/02-replay-buffer-and-splits/sequence-boundary-mask.png" alt="按完整回合独立切分数据集，杜绝相邻帧之间的数据泄露与虚高评估" width="86%">

_图 3.2-4：按完整回合独立切分数据集，杜绝相邻帧之间的数据泄露与虚高评估。_

</div>

### 1. 随机单帧切分的“致命数据泄露”
如果将所有转移元组打散后按 $8:2$ 随机分配给训练集和验证集，在同一个回合中，时刻 $t$ 的状态可能在训练集，而时刻 $t+1$（仅仅相隔 $0.02$ 秒）在验证集。
由于两帧图像几乎一模一样，网络只需死记硬背训练帧就能在验证集上获得虚假的 $99\%$ 极高预测准确率，而在面对全新的未知场景时却瞬间崩溃！

### 2. 回合级切分（Episode-Level Split）铁律
正确的时序数据划分准则必须是：**以完整的回合（Episode）为最小不可分割单元进行划分！**
- 训练集包含回合 $1 \sim 80$ 的全部时序数据；
- 验证集包含回合 $81 \sim 100$ 的全部时序数据。
测试时，模型必须从全新的初始状态开始向前推演，从而真正检验其对物理因果规律的跨场景泛化能力。

<details>
<summary><b>深入推导：非平稳时序分布下的泛化误差界与 Rademacher 复杂度证明（点击展开查看完整推导）</b></summary>

设假说空间为 $\mathcal{H}$。对于依赖度由系数 $\beta(k)$ 刻画的强混合（$\beta$-Mixing）时序序列，利用 Yu (1994) 的独立块对偶引理（Independent Block Technique），将长度为 $T$ 的序列切分为 $2\mu$ 个长度为 $a$ 的独立子块。
经验 Rademacher 复杂度满足上界：
$$\mathcal{R}_T(\mathcal{H}) \le \mathcal{R}_\mu(\mathcal{H}) + \mathcal{O}\left( \sqrt{\frac{a}{T}} \right) + \mathcal{O}(\mu \beta(a))$$
按完整独立回合划分使得各轨迹之间相互统计独立（$\beta \equiv 0$），将泛化误差界的收敛速度牢牢锁定在经典扎实的 $\mathcal{O}(1/\sqrt{M})$（$M$ 为独立回合总数）。
</details>

---

## 3.2.4 纯底层 PyTorch 代码实现：从零手写支持环形覆盖与 PER 的经验回放池

下面我们使用纯底层 PyTorch 算子实现一个结构完备、具备环形队列覆盖与优先经验回放（PER）的张量缓冲区。

```python
import torch
import numpy as np

class PrioritizedReplayBuffer:
    """
    纯底层优先经验回放池 (PER Buffer)
    支持固定容量环形覆盖、基于 TD 误差优先采样与重要性权重修正
    """
    def __init__(self, capacity: int = 1000, state_dim: int = 4, action_dim: int = 1, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.ptr = 0
        self.size = 0

        # 连续内存张量预分配
        self.states = torch.zeros(capacity, state_dim)
        self.actions = torch.zeros(capacity, action_dim)
        self.rewards = torch.zeros(capacity, 1)
        self.next_states = torch.zeros(capacity, state_dim)
        self.dones = torch.zeros(capacity, 1)

        # 样本优先级数组 (初始赋予最大值 1.0 保证新样本被优先访问)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def push(self, state: torch.Tensor, action: torch.Tensor, reward: float, next_state: torch.Tensor, done: bool):
        """
        压入单条转移元组 (先进先出环形覆盖)
        """
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        self.priorities[idx] = self.max_priority
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int = 32) -> tuple[dict, np.ndarray, torch.Tensor]:
        """
        按优先级概率抽样并计算重要性采样权重
        """
        # 1. 计算抽样概率 P(i)
        prios = self.priorities[:self.size] ** self.alpha
        probs = prios / prios.sum()

        # 2. 依据概率分布抽取索引
        indices = np.random.choice(self.size, size=batch_size, p=probs, replace=False)

        # 3. 计算重要性权重 w_i = (N * P(i))^(-beta)
        total = self.size
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / weights.max() # 归一化至 [0, 1]
        torch_weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

        batch = {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices]
        }
        return batch, indices, torch_weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        根据最新计算的 TD 误差更新样本优先级
        """
        for idx, td in zip(indices, td_errors):
            prio = abs(td) + 1e-5
            self.priorities[idx] = prio
            self.max_priority = max(self.max_priority, prio)

# ===================================================================
# 单元测试与重要性权重校验
# ===================================================================
if __name__ == "__main__":
    buffer = PrioritizedReplayBuffer(capacity=100, state_dim=4, action_dim=1, alpha=0.6, beta=0.4)

    # 1. 模拟写入 20 条转移数据
    for i in range(20):
        s = torch.randn(4)
        a = torch.randn(1)
        r = float(i)
        s_next = torch.randn(4)
        done = False
        buffer.push(s, a, r, s_next, done)

    print(f"[Buffer Test] 当前回放池有效样本数: {buffer.size}")

    # 2. 采样一个小批量 (Batch Size = 4)
    batch, indices, weights = buffer.sample(batch_size=4)
    print(f"[Buffer Test] 抽取的样本索引: {indices.tolist()}")
    print(f"[Buffer Test] 重要性采样权重: {[round(x, 4) for x in weights.squeeze().tolist()]}")

    # 3. 模拟更新 TD 误差
    simulated_td_errors = np.array([5.0, 0.1, 2.0, 0.05])
    buffer.update_priorities(indices, simulated_td_errors)
    print(f"[Buffer Test] 更新后全局最大优先级: {buffer.max_priority:.4f}")

    assert batch["states"].shape == (4, 4), "采样状态张量形状不符！"
    assert weights.max().item() <= 1.0001, "重要性权重未正确归一化！"
    print("✓ 优先经验回放池、重要性采样权重与优先级动态更新单测全部通过！")
```

---

## 3.2.5 本节小结

回顾本节内容，我们建立了强化学习与世界模型数据流管理的核心准则：
1. **打破时序自相关**：经验回放池将强相关的连续事件流转化为近似平稳的 i.i.d. 批次，保障了深度网络的稳定收敛；
2. **PER 重点攻坚与无偏性**：以 TD 误差为牵引优先复习高价值难点，同时利用重要性采样权重完美消除概率偏置；
3. **回合级切分铁律**：以完整独立轨迹为单位划分训练与验证集，从源头杜绝了相邻帧的时序数据泄露。
