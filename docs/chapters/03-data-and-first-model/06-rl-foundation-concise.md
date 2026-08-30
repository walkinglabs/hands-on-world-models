# 强化学习基础模块的简洁实现

在前面几节中，我们已经探讨了如何从零开始构建强化学习的基本计算图，并深入理解了其中的梯度传播机制。然而，在实际的现代深度强化学习研究与工程实践中，我们很少会手动实现每一个底层的张量操作。随着诸如深度Q网络 (Deep Q-Network, DQN) [[Mnih et al., 2013]](https://arxiv.org/abs/1312.5602) 与近端策略优化 (Proximal Policy Optimization, PPO) [[Schulman et al., 2017]](https://arxiv.org/abs/1707.06347) 等算法的提出，强化学习的算法范式逐渐收敛为几个标准化的核心模块：环境交互、经验回放、状态价值评估以及策略近似。

强化学习的理论基础可以追溯到理查德·贝尔曼 (Richard Bellman) 在20世纪50年代提出的动态规划理论 [[Bellman, 1957]](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming)，以及理查德·萨顿 (Richard Sutton) 等人发展的时序差分学习 (Temporal-Difference Learning) [[Sutton, 1988]](https://doi.org/10.1007/BF00115009)。在当时，受限于计算力和数据规模，这些方法多用于状态空间离散且有限的表格型 (Tabular) 场景。而在深度学习框架的加持下，我们得以利用神经网络的高维非线性拟合能力，将这些经典的数学迭代过程转化为可以用梯度下降优化的目标函数。

本节旨在利用深度学习框架的高级API，提供强化学习基础模块的“简洁实现”。我们将不仅局限于代码层面的简化，更重要的是，我们将严格追溯这些模块背后的数学原点，演示如何将抽象的马尔可夫决策过程 (Markov Decision Process, MDP) 严谨地映射为现代张量计算框架中的类与函数。

## 马尔可夫决策过程与价值函数的数学映射

在深入代码之前，我们必须首先在数学上严格定义我们要实现的对象。在强化学习中，智能体 (Agent) 与环境 (Environment) 的交互被形式化为一个马尔可夫决策过程。我们可以将其描述为一个元组 $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$。

假设在一个离散的时间步 $t$ 中，智能体观察到当前的状态 $s_t \in \mathcal{S}$，并根据某种策略 $\pi(a_t|s_t)$ 选择一个动作 $a_t \in \mathcal{A}$。随后，环境接收该动作，依据状态转移概率 $P(s_{t+1}|s_t, a_t)$ 将状态更新为 $s_{t+1}$，并反馈给智能体一个标量奖励 $r_t = R(s_t, a_t)$。

智能体的核心目标是最大化未来的累积奖励。为了防止无限时间步长下的累积奖励发散，并体现出“远期奖励不如近期奖励重要”的自然衰减属性，我们引入一个常数折扣因子 (Discount Factor) $\gamma \in [0, 1)$。我们将从时间步 $t$ 开始的折扣累积回报 (Return) 严格定义为一个无穷级数：

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$$

如果我们将上式中的公比 $\gamma$ 提取出来，就可以巧妙地将其转化为一个递归表达式。这在高中数学的等比数列求和中是一个常见的代数技巧：

$$G_t = r_t + \gamma (r_{t+1} + \gamma r_{t+2} + \cdots)$$

$$G_t = r_t + \gamma G_{t+1}$$

这一递归形式是整个强化学习大厦的基石。然而，$G_t$ 是一个依赖于未来随机状态转移和随机策略的随机变量。为了评估在特定状态下执行特定动作的“好坏”，我们需要对其取数学期望。我们将动作价值函数 (Action-Value Function) $Q^\pi(s, a)$ 定义为在状态 $s$ 执行动作 $a$ 后，遵循策略 $\pi$ 所能获得的期望回报：

$$Q^\pi(s, a) = \mathbb{E}_\pi [G_t \mid S_t=s, A_t=a]$$

结合式该公式，我们可以推导出强化学习中最著名的贝尔曼期望方程 (Bellman Expectation Equation)：

$$Q^\pi(s, a) = \mathbb{E}_{s' \sim P, a' \sim \pi} [r_t + \gamma Q^\pi(s', a') \mid S_t=s, A_t=a]$$

在深度强化学习中，我们不再使用表格来记录每一个状态-动作对的精确 $Q$ 值，而是使用一组参数为 $\theta$ 的神经网络 $Q_\theta(s, a)$ 来进行函数近似。我们的代码实现，本质上就是构建数据结构来收集式该公式中的元组 $(s_t, a_t, r_t, s_{t+1})$，并构建神经网络来拟合这个数学期望。

为了完成这些模块的构建，(**我们首先导入深度学习框架的必备模块以及标准库**)。

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F
import collections
import random
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
from tensorflow import keras
import collections
import random
```

## 经验回放缓冲区：打破时序相关性

在经典的监督学习中，我们通常假设训练数据是独立同分布的 (Independent and Identically Distributed, i.i.d.)。然而，在强化学习中，智能体收集的数据是一条连续的轨迹 (Trajectory)：$s_0, a_0, r_0, s_1, a_1, \dots$。显然，状态 $s_{t+1}$ 极大地依赖于上一个状态 $s_t$ 和动作 $a_t$。这种强烈的时间序列相关性会导致神经网络在利用梯度下降优化时产生严重的震荡甚至灾难性遗忘。

Lin 系统研究了经验回放（Experience Replay）：保存过去的经验，并在后续学习中重新呈现它们 [[Lin, 1992]](https://doi.org/10.1007/BF00992699)；DQN 随后把这一机制用于深度强化学习。现代实现常采用固定容量的循环缓冲区，容量满后覆盖最旧数据，但“FIFO 队列”是常见工程实现，并不是经验回放概念本身的必要条件。训练时，我们通常从缓冲区随机抽取由 $(s_t, a_t, r_t, s_{t+1}, d_t)$ 组成的小批量转移。

通过这种均匀的随机采样，我们在数学期望的层面上打破了数据样本之间的时序相关性，迫使采样的批量数据近似满足独立同分布的假设，从而极大地稳定了梯度的方向。

(**下面我们通过高级API实现一个简洁而高效的经验回放缓冲区**)。为了实现上的简洁，我们直接利用Python标准库中的 `collections.deque` 来维护固定长度的队列，并在采样时一次性将数据堆叠为张量 (Tensor)。

```{.python .input}
#@tab pytorch
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

```{.python .input}
#@tab tensorflow
class ReplayBuffer:
    """强化学习的经验回放缓冲区简洁实现"""
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """将一步 MDP 转移记录到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机无放回采样，并直接将其转换为多维张量供网络训练"""
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        
        # 将数据统一转换为 TensorFlow 张量
        return (tf.convert_to_tensor(state, dtype=tf.float32),
                tf.convert_to_tensor(action, dtype=tf.int32),
                tf.convert_to_tensor(reward, dtype=tf.float32),
                tf.convert_to_tensor(next_state, dtype=tf.float32),
                tf.convert_to_tensor(done, dtype=tf.float32))

    def size(self):
        """查询当前缓冲区中积累的转移样本数量"""
        return len(self.buffer)
```

## 基于多层感知机的策略网络与价值网络

在拥有了稳定的数据来源（经验回放池）之后，我们接下来需要定义算法的“大脑”，即策略网络和价值网络。

正如我们在式该公式中所定义的，动作价值函数 $Q(s, a)$ 的输入是状态和动作，输出是一个标量期望值。在状态连续而动作空间离散（例如 $a \in \{0, 1, \dots, K-1\}$）的环境中，为了避免对每一个特定的动作单独进行前向传播，我们通常将 $Q$ 网络设计为：仅接收状态向量 $s$ 作为输入，而在输出层产生一个维度为 $K$ 的向量，其中第 $k$ 个元素对应于动作 $k$ 的 $Q$ 值。这种架构在数学上等价于建立一个从状态空间到整个动作空间的非线性映射：$\mathcal{S} \mapsto \mathbb{R}^K$。

而对于策略网络 $\pi(a|s)$，其数学本质是给定状态 $s$ 时的条件概率分布。同理，我们将网络设计为输出一个维度为 $K$ 的向量，但为了满足概率的公理（非负且和为1），这些输出通常被视为对数几率 (Logits)，并在后续的损失函数计算中通过 Softmax 激活函数转化为严格的概率分布。

得益于深度学习框架提供的顺序容器（如 `nn.Sequential`），我们可以极其紧凑地实现这些高维非线性映射。(**以下展示了如何利用多层感知机 (MLP) 构建简洁的Q网络与策略网络**)。我们采用带有ReLU激活函数的双隐层结构，这在大多数基础强化学习任务中被证明是具备足够表达能力的。

```{.python .input}
#@tab pytorch
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

```{.python .input}
#@tab tensorflow
class QNetwork(keras.Model):
    """基于多层感知机的动作价值函数近似"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = keras.Sequential([
            keras.layers.Dense(hidden_dim, activation='relu'),
            keras.layers.Dense(hidden_dim, activation='relu'),
            # 输出层不使用激活函数，因为Q值可以是任意实数
            keras.layers.Dense(action_dim)
        ])

    def call(self, state):
        """输入维度为 (batch_size, state_dim)，输出维度为 (batch_size, action_dim)"""
        return self.net(state)

class PolicyNetwork(keras.Model):
    """基于多层感知机的随机策略近似"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = keras.Sequential([
            keras.layers.Dense(hidden_dim, activation='relu'),
            keras.layers.Dense(hidden_dim, activation='relu'),
            # 输出 logits
            keras.layers.Dense(action_dim)
        ])

    def call(self, state):
        return self.net(state)
```

在强化学习的计算图中，Q网络和策略网络扮演着将高维感知信息降维提炼为价值标量和动作概率的核心角色。通过这种简洁的模块化封装，我们将其内部复杂的矩阵乘法与偏置加法完全隐藏在了框架的高级API之下，使我们能够将精力集中于宏观的算法逻辑与贝尔曼更新目标的计算上。

## 小结

* 强化学习的核心是寻找能够最大化累积折扣回报的策略。我们可以通过递归的贝尔曼方程严格地在数学上定义价值函数。
* `ReplayBuffer` 模块的引入，通过缓存历史轨迹和均匀随机采样，打破了强化学习样本间严重的时间序列相关性，从而允许我们利用传统的基于独立同分布假设的优化算法（如随机梯度下降）来训练网络。
* 借助于深度学习框架的高级抽象容器，诸如动作价值函数 $Q(s, a)$ 和策略分布 $\pi(a|s)$ 可以被极其紧凑地建模为多层感知机。这为我们在后续章节中快速组装和实现复杂的深度强化学习算法奠定了极其稳固且简洁的基础。
