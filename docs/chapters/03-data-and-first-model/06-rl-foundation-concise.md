# 强化学习基础模块的简洁实现

前面几节从零构建了强化学习的基本计算图。现代实现通常把环境交互、轨迹或转移数据收集、价值估计以及策略或价值函数更新拆成模块。DQN 使用经验回放训练动作价值函数 [[Mnih et al., 2013]](https://arxiv.org/abs/1312.5602)，而 PPO 使用新近收集的 on-policy 轨迹进行多轮小批量更新 [[Schulman et al., 2017]](https://arxiv.org/abs/1707.06347)；因此，经验回放不是所有强化学习算法共有的必要模块。

<div align="center">
  <img src="/figures/03-data-and-first-model/source/06-rl-foundation-concise/dqn-fig2.png" alt="Breakout 与 Seaquest 的回报曲线展示 DQN 用回放数据训练动作价值函数后的学习进展。" width="86%">

_图 3.6-1：Breakout 与 Seaquest 的回报曲线展示 DQN 用回放数据训练动作价值函数后的学习进展。 出处：Volodymyr Mnih et al.，[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)（2013），Figure 2。_

</div>

强化学习的理论基础可以追溯到理查德·贝尔曼 (Richard Bellman) 在20世纪50年代提出的动态规划理论 [[Bellman, 1957]](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming)，以及理查德·萨顿 (Richard Sutton) 等人发展的时序差分学习 (Temporal-Difference Learning) [[Sutton, 1988]](https://doi.org/10.1007/BF00115009)。在当时，受限于计算力和数据规模，这些方法多用于状态空间离散且有限的表格型 (Tabular) 场景。而在深度学习框架的加持下，我们得以利用神经网络的高维非线性拟合能力，将这些经典的数学迭代过程转化为可以用梯度下降优化的目标函数。

本节用深度学习框架的常用容器重写前一节的基础模块，重点保留数学对象与代码张量之间的对应关系。

## 马尔可夫决策过程与价值函数的数学映射

在深入代码之前，我们必须首先在数学上严格定义我们要实现的对象。在强化学习中，智能体 (Agent) 与环境 (Environment) 的交互被形式化为一个马尔可夫决策过程。我们可以将其描述为一个元组 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$。

假设在一个离散的时间步 $t$ 中，智能体观察到当前的状态 $s_t \in \mathcal{S}$，并根据某种策略 $\pi(a_t|s_t)$ 选择一个动作 $a_t \in \mathcal{A}$。随后，环境接收该动作，依据状态转移概率 $P(s_{t+1}|s_t, a_t)$ 将状态更新为 $s_{t+1}$，并反馈给智能体一个标量奖励 $r_t = R(s_t, a_t)$。

智能体的核心目标是最大化未来的累积奖励。为了防止无限时间步长下的累积奖励发散，并体现出“远期奖励不如近期奖励重要”的自然衰减属性，我们引入一个常数折扣因子 (Discount Factor) $\gamma \in [0, 1)$。我们将从时间步 $t$ 开始的折扣累积回报 (Return) 严格定义为一个无穷级数：

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$$

如果我们将上式中的公比 $\gamma$ 提取出来，就可以巧妙地将其转化为一个递归表达式。这在高中数学的等比数列求和中是一个常见的代数技巧：

$$G_t = r_t + \gamma (r_{t+1} + \gamma r_{t+2} + \cdots)$$

$$
G_t = r_t + \gamma G_{t+1}
$$

<div align="center">
  <img src="/figures/03-data-and-first-model/latex/06-rl-foundation-concise/return-recursive-tail.png" alt="折扣回报把即时奖励与从下一时刻开始的完整尾回报递归相加" width="86%">

_图 3.6-2：把首项 r_t 单独取出后，剩余级数正是 G_{t+1}，只是整体多乘一个 γ，因此得到递推式 G_t=r_t+γG_{t+1}。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

这一递归形式是整个强化学习大厦的基石。然而，$G_t$ 是一个依赖于未来随机状态转移和随机策略的随机变量。为了评估在特定状态下执行特定动作的“好坏”，我们需要对其取数学期望。我们将动作价值函数 (Action-Value Function) $Q^\pi(s, a)$ 定义为在状态 $s$ 执行动作 $a$ 后，遵循策略 $\pi$ 所能获得的期望回报：

$$Q^\pi(s, a) = \mathbb{E}_\pi [G_t \mid S_t=s, A_t=a]$$

把回报递推代入动作价值定义，可得贝尔曼期望方程：

$$Q^\pi(s, a) = \mathbb{E}_{s' \sim P, a' \sim \pi} [r_t + \gamma Q^\pi(s', a') \mid S_t=s, A_t=a]$$

状态空间较大时，可以用参数为 $\theta$ 的网络 $Q_\theta(s,a)$ 近似动作价值。代码一端收集 $(s_t,a_t,r_t,s_{t+1})$ 转移，另一端构造价值或策略网络；具体训练目标由所选算法决定。

先导入张量、网络与缓冲区所需模块。

```python
import torch
from torch import nn
from torch.nn import functional as F
import collections
import random
```

## 经验回放缓冲区：打破时序相关性

监督学习的常见分析假设样本近似独立同分布。强化学习数据却来自连续轨迹 $s_0,a_0,r_0,s_1,a_1,\dots$，其中 $s_{t+1}$ 由 $s_t$、$a_t$ 和环境共同产生。连续批次可能覆盖很窄的状态区域，使梯度高度相关并偏向近期经验。

Lin 系统研究了经验回放（Experience Replay）：保存过去的经验，并在后续学习中重新呈现它们 [[Lin, 1992]](https://doi.org/10.1007/BF00992699)；DQN 随后把这一机制用于深度强化学习。现代实现常采用固定容量的循环缓冲区，容量满后覆盖最旧数据，但“FIFO 队列”是常见工程实现，并不是经验回放概念本身的必要条件。训练时，我们通常从缓冲区随机抽取由 $(s_t, a_t, r_t, s_{t+1}, d_t)$ 组成的小批量转移。

均匀随机采样会减少相邻转移同时进入一个批次的机会，但不会消除数据本身的相关性或分布漂移。

下面用 Python 标准库的 `collections.deque` 维护固定容量队列，并在采样时把各字段堆叠为张量。

```python
class ReplayBuffer:
    """强化学习的经验回放缓冲区简洁实现"""
    def __init__(self, capacity):
        # 使用 deque 可以自动处理队列满时的先进先出逻辑
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """将一步 MDP 转移记录到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机无放回采样，并直接将其转换为多维张量供网络训练"""
        transitions = random.sample(self.buffer, batch_size)
        # 解包转移元组的列表，重组为各属性的元组
        state, action, reward, next_state, done = zip(*transitions)

        # 将数据统一转换为 PyTorch 张量，并指定严谨的数据类型
        return (torch.tensor(state, dtype=torch.float32),
                torch.tensor(action, dtype=torch.int64),
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(next_state, dtype=torch.float32),
                torch.tensor(done, dtype=torch.float32))

    def size(self):
        """查询当前缓冲区中积累的转移样本数量"""
        return len(self.buffer)
```

## 基于多层感知机的策略网络与价值网络

在拥有了稳定的数据来源（经验回放池）之后，我们接下来需要定义算法的“大脑”，即策略网络和价值网络。

动作价值函数 $Q(s,a)$ 对一个状态—动作对输出标量。离散动作数为 $K$ 时，网络可只接收状态 $s$，一次输出 $K$ 个动作值；第 $k$ 个分量就是 $Q(s,a=k)$。因此网络实现的映射为 $\mathcal{S}\to\mathbb{R}^K$。

<div align="center">
  <img src="/figures/03-data-and-first-model/source/06-rl-foundation-concise/a3c-fig1.png" alt="异步 actor–critic 与 DQN 在五个 Atari 游戏上的学习速度对比，体现策略与价值模块组合后的经验结果。" width="86%">

_图 3.6-3：异步 actor–critic 与 DQN 在五个 Atari 游戏上的学习速度对比，体现策略与价值模块组合后的经验结果。 出处：Volodymyr Mnih et al.，[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)（2016），Figure 1。_

</div>

而对于策略网络 $\pi(a|s)$，其数学本质是给定状态 $s$ 时的条件概率分布。同理，我们将网络设计为输出一个维度为 $K$ 的向量，但为了满足概率的公理（非负且和为1），这些输出通常被视为对数几率 (Logits)，并在后续的损失函数计算中通过 Softmax 激活函数转化为严格的概率分布。

用 `nn.Sequential` 可以紧凑地定义 Q 网络和策略网络。下面都采用两层 ReLU MLP；它适合演示接口与形状，不代表对所有任务都足够。

```python
class QNetwork(nn.Module):
    """基于多层感知机的动作价值函数近似"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        # 利用 nn.Sequential 极大地精简了前向传播的定义
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # 输出层不使用激活函数，因为Q值可以是任意实数
        )

    def forward(self, state):
        """输入维度为 (batch_size, state_dim)，输出维度为 (batch_size, action_dim)"""
        return self.net(state)

class PolicyNetwork(nn.Module):
    """基于多层感知机的随机策略近似"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
            # 输出 logits，后续在动作采样或计算对数概率时配合 Softmax/Categorical 分布使用
        )

    def forward(self, state):
        return self.net(state)
```

在强化学习的计算图中，Q网络和策略网络扮演着将高维感知信息降维提炼为价值标量和动作概率的核心角色。通过这种简洁的模块化封装，我们将其内部复杂的矩阵乘法与偏置加法完全隐藏在了框架的高级API之下，使我们能够将精力集中于宏观的算法逻辑与贝尔曼更新目标的计算上。

## 小结

- 强化学习的核心是寻找能够最大化累积折扣回报的策略。我们可以通过递归的**贝尔曼方程**严格地在数学上定义价值函数。
- **`ReplayBuffer` 模块**的引入，通过缓存历史轨迹和均匀随机采样，打破了强化学习样本间严重的时间序列相关性，从而允许我们利用传统的基于独立同分布假设的优化算法（如随机梯度下降）来训练网络。
- **动作价值函数 $Q(s,a)$ 与策略分布 $\pi(a\mid s)$** 都可以先用 MLP 实现；后续算法会在此基础上加入目标网络、优势估计或熵正则等机制。
