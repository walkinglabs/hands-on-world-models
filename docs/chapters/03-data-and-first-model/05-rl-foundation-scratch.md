# 强化学习基础模块的从零开始实现

在前面的章节中，我们已经探讨了监督学习和世界模型中的基础预测机制。然而，当我们的模型不仅需要被动地预测未来，还需要在环境中主动做出决策以最大化某种长期收益时，我们就踏入了强化学习（Reinforcement Learning, RL）的领域。强化学习的理论基础可以追溯到理查德·贝尔曼（Richard Bellman）在动态规划（Dynamic Programming）上的开创性工作 [[Bellman, 1957]](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming)，以及随后由 Sutton 和 Barto 建立的现代时序差分学习框架 [[Sutton & Barto, 1998]](http://incompleteideas.net/book/first/the-book.html)。在深度学习时代，将强大的神经网络与强化学习结合，催生了诸如深度Q网络（DQN） [[Mnih et al., 2013]](https://arxiv.org/abs/1312.5602) 和近端策略优化（PPO） [[Schulman et al., 2017]](https://arxiv.org/abs/1707.06347) 等突破性算法。

本节把马尔可夫决策过程（MDP）、回报、价值函数和贝尔曼方程落实到两个基础模块：经验回放缓冲区与环境交互循环。推导从标量递推开始，再对应到批量张量。

## 马尔可夫决策过程的严格表述

要用数学语言描述“在环境中做决策”，我们需要一个标准化的框架，即马尔可夫决策过程（Markov Decision Process, MDP）。MDP 由一个五元组 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ 构成。

<div align="center">
  <img src="/figures/03-data-and-first-model/source/05-rl-foundation-scratch/lin-fig1.png" alt="Lin 的 AHCON 框架把策略网络、评价网络、环境传感与强化信号连接为早期 actor–critic 数据流。" width="86%">

_图 3.5-1：Lin 的 AHCON 框架把策略网络、评价网络、环境传感与强化信号连接为早期 actor–critic 数据流。 出处：Long-Ji Lin，[Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching](https://doi.org/10.1007/BF00992699)（1992），Figure 1。_

</div>

首先，我们定义状态集合 $\mathcal{S}$ 和动作集合 $\mathcal{A}$。在离散时间步 $t = 0, 1, 2, \dots$ 中，智能体在时间步 $t$ 观察到环境的状态 $s_t \in \mathcal{S}$，并基于某种规则选择一个动作 $a_t \in \mathcal{A}$。

随后，环境根据状态转移概率函数 $P$ 发生演进。这里我们给出一维情况下的最基础概率定义。假设当前状态为 $s$，智能体采取了动作 $a$，环境转移到下一个确定的状态 $s'$ 的概率记为：

$$P(s' \mid s, a) = \mathbb{P}(s_{t+1} = s' \mid s_t = s, a_t = a)$$

同时，环境会根据当前的转移反馈给智能体一个标量奖励 $r_t$。奖励函数 $R(s, a)$ 描述了在状态 $s$ 下执行动作 $a$ 所获得的期望奖励：

$$R(s, a) = \mathbb{E}[r_{t+1} \mid s_t = s, a_t = a]$$

马尔可夫性的核心在于：**未来的状态演进仅依赖于当前的状态和动作，而与过去的历史轨迹无关**。

## 回报与贝尔曼方程的降维推导

### 累积折扣回报

在强化学习中，智能体的目标并非仅仅最大化眼前的单步奖励 $r_{t+1}$，而是要最大化整个生命周期内的总奖励。我们将从时间步 $t$ 开始，到未来所有时间步的累积奖励之和称为回报（Return），记为 $G_t$。

如果我们将未来每一项奖励直接相加，当时间趋于无穷时，这个和可能会发散。为了保证数学上的收敛性，并引入“未来的奖励不如当前的奖励确切”这一时间偏好，我们引入折扣因子（Discount Factor） $\gamma \in [0, 1)$。于是，标量回报 $G_t$ 被严格定义为：

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

提取公因子 $\gamma$ 后，回报可以递归表示：

$$G_t = r_{t+1} + \gamma (r_{t+2} + \gamma r_{t+3} + \dots) = r_{t+1} + \gamma G_{t+1}$$

### 状态价值函数与动作价值函数

由于状态转移和奖励可能是随机的，实际获得的回报 $G_t$ 也是一个随机变量。为了评估在某个状态下“到底有多好”，我们需要计算在给定策略 $\pi$ 下，回报的数学期望。策略 $\pi(a \mid s)$ 定义为在状态 $s$ 下选择动作 $a$ 的概率分布。

我们定义状态价值函数（State-Value Function） $V^\pi(s)$ 为在状态 $s$ 下，遵循策略 $\pi$ 的期望回报：

$$V^\pi(s) = \mathbb{E}_\pi [G_t \mid s_t = s]$$

同理，我们定义动作价值函数（Action-Value Function） $Q^\pi(s, a)$ 为在状态 $s$ 下，先执行动作 $a$，随后遵循策略 $\pi$ 的期望回报：

$$Q^\pi(s, a) = \mathbb{E}_\pi [G_t \mid s_t = s, a_t = a]$$

### 贝尔曼期望方程

把回报递推代入价值函数定义，并对下一状态与动作取条件期望，可得贝尔曼期望方程。它把当前价值写成即时奖励与下一状态价值之和。

$$
\begin{aligned}
V^\pi(s) &= \mathbb{E}_\pi [r_{t+1} + \gamma G_{t+1} \mid s_t = s] \\
&= \sum_{a \in \mathcal{A}} \pi(a \mid s) \mathbb{E}_\pi [r_{t+1} + \gamma G_{t+1} \mid s_t = s, a_t = a] \\
&= \sum_{a \in \mathcal{A}} \pi(a \mid s) \left( R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \mathbb{E}_\pi [G_{t+1} \mid s_{t+1} = s'] \right) \\
&= \sum_{a \in \mathcal{A}} \pi(a \mid s) \left( R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi(s') \right)
\end{aligned}
$$

<div align="center">
  <img src="/figures/03-data-and-first-model/latex/05-rl-foundation-scratch/bellman-double-reduction.png" alt="贝尔曼价值先按转移概率汇聚下一状态，再按策略概率汇聚动作" width="86%">

_图 3.5-2：内层 Σ_s' 在固定动作下对环境后继状态取期望，外层 Σ_a 再按策略概率汇聚动作，两次归约的随机来源不同。_

</div>

同理，对于动作价值函数 $Q^\pi(s, a)$，其贝尔曼期望方程为：

$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \sum_{a' \in \mathcal{A}} \pi(a' \mid s') Q^\pi(s', a')$$

这些方程表明，价值函数可以通过自身的下一次迭代来递归地定义。在深度强化学习中，由于状态空间通常是连续且高维的，我们无法通过纯粹的矩阵求逆来求解上述线性方程组，而是必须借助神经网络，通过时序差分（TD）误差来不断逼近这一等式。

## 经验回放缓冲区的从零实现

在现代强化学习中（尤其是离策略算法如 Q-learning），智能体在环境中交互产生的数据表现出极强的时间相关性。如果我们将这些连续的样本直接送入神经网络进行梯度下降，极易导致训练发散。

Lin 系统研究了经验回放 [[Lin, 1992]](https://doi.org/10.1007/BF00992699)，DQN 又把随机经验回放用于深度 Q 学习 [[Mnih et al., 2013]](https://arxiv.org/abs/1312.5602)。常见实现把转移元组 $(s_t, a_t, r_{t+1}, s_{t+1}, \text{done})$ 存入固定容量的循环缓冲区并均匀采样；“覆盖最旧数据”是工程选择，并非经验回放定义中必须采用的 FIFO 规则。

为了保证计算效率，我们不能使用 Python 原生的列表（list）来存储百万级别的经验，而是必须在一开始就预分配一块连续的张量（Tensor）内存，通过指针循环覆盖旧数据。

下面从零实现支持批量张量采样的经验回放缓冲区。

```python
import torch
import numpy as np

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, state_dim, action_dim, capacity, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0     # 当前写入的游标
        self.size = 0    # 当前缓冲区中已有的数据量

        # 预先分配固定大小的连续张量内存
        # 状态通常是多维连续值
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        # 动作假设为连续或离散，这里以连续向量为例
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        # 奖励和结束标志为标量，我们增加一个维度使其形状为 (capacity, 1) 以便后续矩阵运算
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def add(self, state, action, reward, next_state, done):
        """向缓冲区添加一条经验"""
        # 将数据写入游标指向的位置
        self.states[self.ptr] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.next_states[self.ptr] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.tensor(done, dtype=torch.float32, device=self.device)

        # 游标循环移动
        self.ptr = (self.ptr + 1) % self.capacity
        # 记录当前真实数量，不超过最大容量
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """随机采样一个批量的经验"""
        # 生成随机索引
        ind = torch.randint(0, self.size, size=(batch_size,), device=self.device)

        # 通过高级索引机制一次性提取张量
        return (
            self.states[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_states[ind],
            self.dones[ind]
        )
```

## 智能体环境交互主循环与策略评估

在强化学习的训练框架中，最核心的骨架是“交互-收集-更新”循环。在这里，我们模拟一个极简的连续状态空间环境。

为了计算我们在前文推导出的动作价值，深度强化学习将参数化的神经网络 $Q_\theta(s, a)$ 引入。网络的目标是最小化时序差分误差（TD Error）。假设我们利用当前经验元组 $(s, a, r, s', d)$，其中 $d$ 为指示当前回合是否结束的二值变量（1为结束，0为未结束）。根据贝尔曼最优方程的单步近似，目标值 $y$ 被构造为：

<div align="center">
  <img src="/figures/03-data-and-first-model/source/05-rl-foundation-scratch/dqn-fig3.png" alt="Seaquest 的价值曲线与对应画面显示 DQN 的动作价值如何随危险和奖励事件变化。" width="86%">

_图 3.5-3：Seaquest 的价值曲线与对应画面显示 DQN 的动作价值如何随危险和奖励事件变化。 出处：Volodymyr Mnih et al.，[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)（2013），Figure 3。_

</div>

$$
y = r + \gamma (1 - d) \max_{a'} Q_{\theta^{-}}(s', a')
$$

<div align="center">
  <img src="/figures/03-data-and-first-model/latex/05-rl-foundation-scratch/terminal-mask-td-target.png" alt="终止标志经过一减 d 门控下一状态价值，使终止转移只保留即时奖励" width="86%">

_图 3.5-4：d=0 时 bootstrap 项原样进入目标；d=1 时门值变成 0，目标严格退化为即时奖励 r。_

</div>

其中 $Q_{\theta^{-}}$ 是缓慢更新的目标网络，用于稳定训练（我们将在后续具体算法章节详细讨论其梯度截断机制）。此时的损失函数即为均方误差：

$$L(\theta) = \frac{1}{N} \sum_{i=1}^N \left( Q_\theta(s_i, a_i) - y_i \right)^2$$

因子 $(1-d)$ 控制是否 bootstrap。若 $d=1$ 表示真正的终止状态，下一状态价值应置为 0；若只是时间上限造成的截断，则是否置零取决于环境接口和算法约定，不能把两类边界混在一起。

下面用一个简化环境循环展示收集转移和组装批量数据的顺序。

```python
# 我们模拟一个随机与环境交互的过程
state_dim = 4
action_dim = 2
capacity = 10000
batch_size = 32
device = torch.device("cpu")

buffer = ReplayBuffer(state_dim, action_dim, capacity, device)

# 模拟的交互超参数
num_episodes = 5
max_steps = 100

for episode in range(num_episodes):
    # 重置环境，获取初始状态
    state = np.random.randn(state_dim)
    episode_reward = 0

    for step in range(max_steps):
        # 智能体基于当前策略选择动作 (这里用随机动作模拟)
        action = np.random.randn(action_dim)

        # 环境执行动作，返回下一个状态、奖励和结束标志
        next_state = np.random.randn(state_dim)
        reward = np.random.rand()
        done = 1.0 if step == max_steps - 1 else 0.0

        # 1. 交互与收集：将转移存入缓冲区
        buffer.add(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        # 2. 当缓冲区数据量足够时，进行批量采样与学习
        if buffer.size >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            # 在这里通常会将这些批量张量传入神经网络进行损失计算和梯度反传
            # loss = compute_loss(states, actions, rewards, next_states, dones)
            # optimizer.step()

            # 为了演示，我们仅验证采样的张量维度是否符合预期矩阵运算的形状
            assert states.shape == (batch_size, state_dim)
            assert rewards.shape == (batch_size, 1)
            assert dones.shape == (batch_size, 1)

        if done:
            break

    print(f"Episode {episode + 1} finished with Total Reward: {episode_reward:.2f}")
```

## 小结

本节从折扣回报推出贝尔曼期望方程，并实现了预分配经验缓冲区和交互循环。随机回放减弱批内时间相关性，但不把交互数据变成严格 IID。这里的缓冲区直接服务于 DQN 一类 off-policy 方法；PPO 通常使用新近采集的 on-policy 轨迹缓冲区，数据流不同。

<div align="center">
  <img src="/figures/03-data-and-first-model/source/05-rl-foundation-scratch/ppo-fig3.png" alt="多种 MuJoCo 环境中的算法曲线展示 PPO 这类 on-policy 数据流与离策略回放方法的不同评测对象。" width="86%">

_图 3.5-5：多种 MuJoCo 环境中的算法曲线展示 PPO 这类 on-policy 数据流与离策略回放方法的不同评测对象。 出处：John Schulman et al.，[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)（2017），Figure 3。_

</div>
