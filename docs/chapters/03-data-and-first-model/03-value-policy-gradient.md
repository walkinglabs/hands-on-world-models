# 策略梯度与价值函数基础

## 强化学习的优化范式追溯

监督学习需要输入对应的目标标签；序贯决策中，逐步给出“正确动作”往往比给出成功、失败或任务得分更困难。强化学习（Reinforcement Learning, RL）使用交互产生的奖励信号评价整段行为，再据此调整策略。

<div align="center">
  <img src="/figures/03-data-and-first-model/source/03-value-policy-gradient/ppo-fig5.png" alt="PPO 学到的人形策略连续转向并奔向目标，展示策略梯度最终优化的是可观察行为。" width="86%">

_图 3.3-1：PPO 学到的人形策略连续转向并奔向目标，展示策略梯度最终优化的是可观察行为。 出处：John Schulman et al.，[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)（2017），Figure 5。_

</div>

Williams 给出了用采样回报更新随机策略参数的 REINFORCE 算法 [[Williams, 1992]](https://doi.org/10.1007/BF00992696)。Sutton 等人随后给出策略梯度定理，并讨论结合函数近似价值函数的 actor–critic 形式 [[Sutton et al., 1999]](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)。本节从概率论推导策略梯度，并说明价值基线如何降低梯度估计方差；这两篇引用分别对应算法估计器与定理。

<div align="center">
  <img src="/figures/03-data-and-first-model/source/03-value-policy-gradient/ddpg-fig1.png" alt="DDPG 在摆杆、机械臂、跑步与驾驶任务上的画面展示确定性策略梯度的连续控制范围。" width="86%">

_图 3.3-2：DDPG 在摆杆、机械臂、跑步与驾驶任务上的画面展示确定性策略梯度的连续控制范围。 出处：Timothy P. Lillicrap et al.，[Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)（2016），Figure 1。_

</div>

## 策略与轨迹的数学描述

先在有限时域马尔可夫决策过程（MDP）中定义策略、轨迹和回报。时间步为 $t=0,1,2,\dots,T-1$。

### 策略函数的定义

策略（Policy）是在给定状态 $s$ 时选择动作 $a$ 的条件分布。参数化策略记为：

$$
\pi_\theta(a|s) = P(A_t=a | S_t=s; \theta)
$$

这里的 $\theta$ 是我们要优化的核心物理量。在最简单的一维离散动作空间中，$\pi_\theta(a|s)$ 可能仅仅是一个 softmax 函数的输出；而在连续动作空间中，它通常是一个高斯分布的概率密度函数，其中均值和方差由神经网络预测。

### 轨迹与期望回报

智能体在环境中从初始状态 $s_0$ 出发，依据策略 $\pi_\theta$ 选择动作 $a_0$，环境根据其转移概率分布（Transition Dynamics）$P(s_{1}|s_0, a_0)$ 给出下一个状态 $s_1$ 和奖励 $r_1$。这一过程循环往复，形成一条轨迹（Trajectory，或称为回合 Episode），记为 $\tau$：

$$
\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_{T-1}, a_{T-1}, r_T)
$$

轨迹发生的联合概率由策略和环境共同决定。利用概率论中的链式法则（乘法公式），我们可以将轨迹 $\tau$ 的发生概率写为：

$$
P(\tau | \theta) = P(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) P(s_{t+1}|s_t, a_t)
$$

其中 $P(s_0)$ 是初始状态分布。请注意，这个长串的连乘公式中，只有 $\pi_\theta(a_t|s_t)$ 包含了我们的模型参数 $\theta$。

一条轨迹的总回报（Return）是沿着该轨迹收集到的所有奖励之和。为了保证序列收敛（特别是在无限期任务中），我们引入折扣因子（Discount Factor） $\gamma \in [0, 1]$：

$$
R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_{t+1}
$$

## 目标函数与对数导数技巧

目标是寻找使期望回报尽可能大的参数 $\theta$，记目标函数为 $J(\theta)$。

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] = \int P(\tau | \theta) R(\tau) d\tau
$$

如果是离散动作空间，这里的积分号就替换为对所有可能轨迹的求和。
为了最大化 $J(\theta)$，在微积分和深度学习的体系中，我们最直接的方法就是使用梯度上升（Gradient Ascent）：

$$
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)
$$

其中 $\alpha$ 是学习率。现在的核心难题是：如何求解目标函数的梯度 $\nabla_\theta J(\theta)$？

### 对数导数技巧（Log-Derivative Trick）

在高中数学中，我们知道复合函数求导的链式法则。特别地，对于自然对数函数 $f(x) = \log x$，其导数为 $f'(x) = 1/x$。由此我们可以得到一个极为重要的恒等式，它被称为对数导数技巧：

$$
\nabla_\theta \log P(\tau|\theta) = \frac{\nabla_\theta P(\tau|\theta)}{P(\tau|\theta)}
$$

将其稍微移项，我们得到：

$$
\nabla_\theta P(\tau|\theta) = P(\tau|\theta) \nabla_\theta \log P(\tau|\theta)
$$

把这个恒等式代回目标函数梯度：

$$
\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \int P(\tau | \theta) R(\tau) d\tau \\
&= \int \nabla_\theta P(\tau | \theta) R(\tau) d\tau \\
&= \int P(\tau|\theta) \nabla_\theta \log P(\tau|\theta) R(\tau) d\tau \\
&= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau|\theta) R(\tau) \right]
\end{aligned}
$$

这样便把对轨迹概率的求导改写为一个可采样的期望。实际训练不必枚举所有轨迹，而是运行策略收集轨迹，用蒙特卡洛平均估计梯度。

### 拆解轨迹概率的对数梯度

接着，我们需要计算 $\nabla_\theta \log P(\tau|\theta)$。对该公式两边取对数，乘积将变成求和：

$$
\log P(\tau|\theta) = \log P(s_0) + \sum_{t=0}^{T-1} \left( \log \pi_\theta(a_t|s_t) + \log P(s_{t+1}|s_t, a_t) \right)
$$

若环境动力学和初始状态分布不依赖策略参数 $\theta$，它们的对数梯度为零，只剩策略项：

$$
\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

<div align="center">
  <img src="/figures/03-data-and-first-model/latex/03-value-policy-gradient/trajectory-score-cancellation.png" alt="轨迹概率中的策略因子依赖参数 theta，而初始分布和环境转移项梯度为零" width="86%">

_图 3.3-3：轨迹连乘取对数后变成逐时刻求和；只有策略因子含 θ，因此初始分布和环境转移项在求导时消失。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

代回目标梯度，可得经典的 REINFORCE 估计式：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau) \right]
$$

::: warning 注意
这个估计式不要求写出或求导环境转移概率，因此可用于 model-free 场景。$\nabla_\theta\log\pi_\theta$ 常称为 score function；REINFORCE 是 Williams 给出的采样更新算法名称。
:::

## 因果性与奖励截断

这个估计式的方差通常较高。项 $\nabla_\theta\log\pi_\theta(a_t\mid s_t)R(\tau)$ 用整条轨迹回报评价时刻 $t$ 的动作，其中还包含动作发生前已经得到的奖励。

从基本的逻辑因果律（Causality）出发：在时刻 $t$ 采取的动作 $a_t$，绝对不可能影响时刻 $t$ 之前的奖励 $r_{1}, r_{2}, \dots, r_t$。动作只能影响未来的奖励。因此，用整条轨迹的总回报来评估某个中间动作的好坏，引入了大量无意义的噪声。

在标准因果假设下，动作发生前的奖励与当前动作的 score 项期望为零。因此可把整条轨迹回报替换为从时刻 $t$ 开始的**未来累积回报**（return-to-go）$G_t$：

$$
G_t = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'+1}
$$

<div align="center">
  <img src="/figures/03-data-and-first-model/latex/03-value-policy-gradient/reward-to-go-causal-triangle.png" alt="每个动作只连接其后续奖励，过去奖励对应的梯度期望为零" width="86%">

_图 3.3-4：因果连线形成下三角支持区域；a_t 只能影响未来奖励，所以策略梯度用 G_t 而不是整条轨迹的总回报。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

由于 $R(\tau)$ 中从时刻 $t$ 开始的奖励权重是 $\gamma^{t'}$，而 $G_t$ 把首项重新缩放为 1，所以还要保留外层因子 $\gamma^t$：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \gamma^t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]
$$

在梯度上升中，正的 $G_t$ 会沿提高已采样动作对数概率的方向更新，负的 $G_t$ 则相反。回报的绝对零点会影响这种解释，因此实践中常减去基线，改为比较“比当前状态的通常结果好多少”。

## 价值函数的引入与基线技巧

尽管因果性截断降低了一部分方差，但蒙特卡洛采样的本质决定了 $G_t$ 的波动仍然极大。为了进一步稳定训练，我们需要引入**价值函数（Value Function）**的概念。

<div align="center">
  <img src="/figures/03-data-and-first-model/source/03-value-policy-gradient/a3c-fig2.png" alt="A3C 在多款 Atari 游戏上的分数散点把 actor–critic 的策略更新与经验性能联系起来。" width="86%">

_图 3.3-5：A3C 在多款 Atari 游戏上的分数散点把 actor–critic 的策略更新与经验性能联系起来。 出处：Volodymyr Mnih et al.，[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)（2016），Figure 2。_

</div>

### 状态价值与动作价值

状态价值函数 $V^\pi(s)$ 是从状态 $s$ 出发并遵循策略 $\pi$ 时的期望累积回报：

$$
V^\pi(s) = \mathbb{E}_{\tau \sim \pi} [G_t | S_t = s]
$$

动作价值函数 $Q^\pi(s,a)$ 是在状态 $s$ 先采取动作 $a$、随后遵循策略 $\pi$ 时的期望累积回报：

$$
Q^\pi(s, a) = \mathbb{E}_{\tau \sim \pi} [G_t | S_t = s, A_t = a]
$$

两者的关系非常直接：处于状态 $s$ 的期望价值，就是选择各种动作能够带来的期望价值，按照策略给出的概率分布进行的加权平均。这也是全概率公式的一种体现。

$$
V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a)
$$

### 引入基线（Baseline）

在该公式中，我们用 $G_t$ 作为评估动作好坏的权重。然而，如果环境的所有奖励都是非常大的正数（例如总是给 +100 到 +200 之间的奖励），那么所有的 $G_t$ 都会很大，策略梯度会不加区分地尝试提高所有动作的概率。

> 若某个状态下所有动作通常都能得到约 100 分，那么一次 101 分的结果只是略好于常态。减去该状态的平均水平后，更新权重由 101 变为约 1，更直接地表示这次动作的相对表现。

在数学上，我们可以证明，在策略梯度的计算式中减去任意一个不依赖于具体动作 $a_t$ 的基线函数 $b(s_t)$，都不会改变梯度的无偏期望值。

<div align="center">
  <img src="/figures/03-data-and-first-model/latex/03-value-policy-gradient/baseline-zero-expectation.png" alt="同一状态基线乘各动作 score 后按动作概率求和为零" width="86%">

_图 3.3-6：同一个 b(s) 被所有动作分支共享，可移出动作求和；剩余 score 的期望等于归一化概率总和 1 的梯度，因此为零。本文根据上述基线性质绘制；TikZ/LaTeX 编译。_

</div>

最自然、最合理的基线选择，正是状态价值函数 $V^\pi(s_t)$。

<div align="center">
  <img src="/figures/03-data-and-first-model/source/03-value-policy-gradient/ppo-fig4.png" alt="三种 3D 人形控制任务的学习曲线显示带价值估计的策略优化如何随训练提升。" width="86%">

_图 3.3-7：三种 3D 人形控制任务的学习曲线显示带价值估计的策略优化如何随训练提升。 出处：John Schulman et al.，[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)（2017），Figure 4。_

</div>

我们定义**优势函数（Advantage Function）** 为动作价值与状态价值之差：

$$
A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)
$$

若使用真实优势函数，策略梯度可写为：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \gamma^t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^\pi(s_t, a_t) \right]
$$

在实际代码实现中，如果我们只使用纯粹的 REINFORCE 算法，往往只会减去一个基于蒙特卡洛采样的基线。下面，我们将通过具体的代码来实现带有基线的 REINFORCE 算法。

## 代码实现：带基线的 REINFORCE

以下代码展示了如何使用深度学习框架实现带基线的策略梯度优化。我们假设环境提供了一维离散动作空间，并使用多层感知机（MLP）作为策略网络。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        # 经过线性层和ReLU激活
        x = F.relu(self.fc1(x))
        # 输出动作的对数概率前馈，使用softmax转换为离散概率分布
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs

def compute_returns(rewards, gamma=0.99):
    # 计算从后向前的未来累积回报 G_t
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    # 批内标准化回报：常见的数值稳定技巧，不等同于学习到的状态价值基线
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def reinforce_update(policy_net, optimizer, saved_log_probs, rewards, gamma=0.99):
    # 计算并标准化 G_t
    returns = compute_returns(rewards, gamma)

    policy_loss = []
    # 遍历轨迹中的每一步，计算对数梯度与回报的乘积
    for t, (log_prob, G) in enumerate(zip(saved_log_probs, returns)):
        # 策略梯度的实现：我们希望最大化目标函数 J，即最小化 -J
        # 由于 PyTorch 的优化器执行的是梯度下降（Gradient Descent）
        # 因此我们在前向附加一个负号
        policy_loss.append(-(gamma ** t) * log_prob * G)

    # 对时间步求和，反向传播
    policy_loss = torch.stack(policy_loss).sum()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
```

代码保留了与起点折扣目标一致的外层 $\gamma^t$。一些实现会省略这个因子，相当于采用不同的时间步加权约定，阅读代码时应先核对目标定义。这里没有训练 $V^\pi(s)$，而是对一条采样轨迹中的 returns 做批内标准化。减均值与除标准差常能改善数值尺度，但它不是状态条件基线，有限批次下也不能直接套用“任意动作无关基线保持严格无偏”的结论。Actor–Critic 会另行训练价值网络，并用估计优势更新策略。

## 小结

在本节中，我们详细拆解了强化学习中的策略梯度定理：

1. 从马尔可夫决策过程的轨迹分布出发，我们定义了由策略网络参数化的目标期望函数。
2. 通过引入复合函数求导中经典的**对数导数技巧**，我们将目标函数的梯度转化为可以从环境中采样的期望形式，克服了无法获知环境状态转移概率分布的难题。
3. 遵循严格的因果时间序律，我们使用未来累积回报取代了轨迹总回报。
4. 为了抑制蒙特卡洛采样带来的巨大方差，我们引入了状态价值、动作价值的严格定义，并通过推导优势函数，引入了用于修正权重的**基线（Baseline）**概念。

轨迹概率、score-function 估计、return-to-go 和基线是理解 PPO、TRPO 等策略优化方法的共同基础。SAC 还引入熵正则和 off-policy 价值学习，不能只由本节公式直接得到。
