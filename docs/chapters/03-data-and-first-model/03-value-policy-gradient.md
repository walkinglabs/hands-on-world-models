# 策略梯度与价值函数基础

## 强化学习的优化范式追溯

在深度学习的早期，监督学习依赖于明确的标签（即专家的示范）来更新模型。然而，在诸如游戏、机器人控制或自动驾驶等开放环境中，获取逐帧的最优决策标签是极其昂贵甚至不可能的。强化学习（Reinforcement Learning, RL）提供了一种替代范式：智能体（Agent）通过与环境的交互，收集奖励（Reward）信号，并以此为导向优化自身的行为。

Williams 给出了用采样回报更新随机策略参数的 REINFORCE 算法 [[Williams, 1992]](https://doi.org/10.1007/BF00992696)。Sutton 等人随后给出策略梯度定理，并讨论结合函数近似价值函数的 actor–critic 形式 [[Sutton et al., 1999]](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html)。本节从概率论推导策略梯度，并说明价值基线如何降低梯度估计方差；这两篇引用分别对应算法估计器与定理。

## 策略与轨迹的数学描述

在我们引入任何复杂的张量运算之前，首先必须对智能体与环境的交互过程进行严格的数学定义。
假设我们处于一个马尔可夫决策过程（MDP）中，时间由离散的步长 $t=0, 1, 2, \dots$ 组成。

### 策略函数的定义

(**定义策略（Policy）作为条件概率分布**)
我们将策略定义为在给定当前状态 $s$ 的情况下，选择某个动作 $a$ 的条件概率分布。如果策略由参数 $\theta$（例如神经网络的权重）参数化，我们将其记为：

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

强化学习的终极目标是找到一组最优参数 $\theta^*$，使得轨迹的总回报的期望值（Expected Return）最大化。我们将这个目标函数记为 $J(\theta)$。

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

这个简单的公式是整个策略梯度算法的灵魂所在。回到我们的目标函数梯度：

$$
\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \int P(\tau | \theta) R(\tau) d\tau \\
&= \int \nabla_\theta P(\tau | \theta) R(\tau) d\tau \\
&= \int P(\tau|\theta) \nabla_\theta \log P(\tau|\theta) R(\tau) d\tau \\
&= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau|\theta) R(\tau) \right]
\end{aligned}
$$

这里我们做了一次极其关键的转换：将原本对概率分布本身的求导，转化为了对“某个随机变量求期望”的形式。因为在实际应用中，我们不可能穷举所有的轨迹来积分，但我们可以通过让智能体在环境中运行多次来**采样**轨迹，进而利用蒙特卡洛方法（Monte Carlo method）来无偏地估计这个期望值。

### 拆解轨迹概率的对数梯度

接着，我们需要计算 $\nabla_\theta \log P(\tau|\theta)$。对该公式两边取对数，乘积将变成求和：

$$
\log P(\tau|\theta) = \log P(s_0) + \sum_{t=0}^{T-1} \left( \log \pi_\theta(a_t|s_t) + \log P(s_{t+1}|s_t, a_t) \right)
$$

现在，对等式两边关于 $\theta$ 求梯度。奇妙的事情发生了：在这个式子中，初始状态分布 $P(s_0)$ 和环境的状态转移概率 $P(s_{t+1}|s_t, a_t)$ 完全与参数 $\theta$ 无关！因此，它们的梯度统统为零。

$$
\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

将该公式代回该公式，我们得到了经典的 REINFORCE 策略梯度公式：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau) \right]
$$

::: warning 注意
这是一个在强化学习中少有的、允许我们在不知道环境转移概率（即所谓 Model-free）的情况下，仍然可以计算目标函数梯度的公式。这也是 REINFORCE 算法名称中 "Score Function Estimator" 的由来。
:::

## 因果性与奖励截断

该公式虽然形式严密，但在方差（Variance）上存在巨大的问题。让我们仔细观察公式中的项 $\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)$。它表示时间步 $t$ 采取动作 $a_t$ 的对数概率梯度，乘以整条轨迹的**总回报** $R(\tau)$。

从基本的逻辑因果律（Causality）出发：在时刻 $t$ 采取的动作 $a_t$，绝对不可能影响时刻 $t$ 之前的奖励 $r_{1}, r_{2}, \dots, r_t$。动作只能影响未来的奖励。因此，用整条轨迹的总回报来评估某个中间动作的好坏，引入了大量无意义的噪声。

我们可以严格证明（篇幅所限，此处不展开纯代数推导），对于任意 $t' < t$，有 $\mathbb{E} [ \nabla_\theta \log \pi_\theta(a_t|s_t) \gamma^{t'} r_{t'+1} ] = 0$。因此，我们可以心安理得地将总回报 $R(\tau)$ 替换为从时刻 $t$ 开始的**未来累积回报**（Return to go），通常记为 $G_t$：

$$
G_t = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'+1}
$$

由此，策略梯度公式被改良为：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]
$$

这个公式直观地表达了强化学习的本质原理：如果一个动作导致了较高的未来累积回报 $G_t > 0$，那么其对数概率的梯度方向就会被放大，网络在下一次遇到类似状态时，输出该动作的概率就会提高；反之亦然。

## 价值函数的引入与基线技巧

尽管因果性截断降低了一部分方差，但蒙特卡洛采样的本质决定了 $G_t$ 的波动仍然极大。为了进一步稳定训练，我们需要引入**价值函数（Value Function）**的概念。

### 状态价值与动作价值

(**定义状态价值函数（State Value Function）**)
状态价值 $V^\pi(s)$ 衡量了智能体处于状态 $s$，并遵循策略 $\pi$ 进行决策，直至回合结束所能期望获得的累积回报：

$$
V^\pi(s) = \mathbb{E}_{\tau \sim \pi} [G_t | S_t = s]
$$

(**定义动作价值函数（Action Value Function）**)
动作价值 $Q^\pi(s, a)$ 衡量了智能体处于状态 $s$，**首先采取动作 $a$**，随后严格遵循策略 $\pi$ 所能期望获得的累积回报：

$$
Q^\pi(s, a) = \mathbb{E}_{\tau \sim \pi} [G_t | S_t = s, A_t = a]
$$

两者的关系非常直接：处于状态 $s$ 的期望价值，就是选择各种动作能够带来的期望价值，按照策略给出的概率分布进行的加权平均。这也是全概率公式的一种体现。

$$
V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a)
$$

### 引入基线（Baseline）

在该公式中，我们用 $G_t$ 作为评估动作好坏的权重。然而，如果环境的所有奖励都是非常大的正数（例如总是给 +100 到 +200 之间的奖励），那么所有的 $G_t$ 都会很大，策略梯度会不加区分地尝试提高所有动作的概率。

> 唯一的精炼类比：假设你是一家公司的老板（策略网络），你需要给员工（动作）发奖金来鼓励他们。如果公司整体效益很好，每个员工都发了一百万奖金（$G_t$ 很大），这并不能反映出哪个员工的贡献最突出。更好的方法是设定一个“公司平均绩效”（基线）。如果某个员工的业绩超过了平均绩效，我们才重点提拔他（提高概率），否则即使他赚了钱，只要低于平均，我们也应该削减他的奖金（相对降低概率）。

在数学上，我们可以证明，在策略梯度的计算式中减去任意一个不依赖于具体动作 $a_t$ 的基线函数 $b(s_t)$，都不会改变梯度的无偏期望值。最自然、最合理的基线选择，正是状态价值函数 $V^\pi(s_t)$。

我们定义**优势函数（Advantage Function）** 为动作价值与状态价值之差：

$$
A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)
$$

利用优势函数，最终的、同时具备极低方差和无偏特性的策略梯度定理（Actor-Critic 架构的理论基础）可以写为：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^\pi(s_t, a_t) \right]
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
    # 标准化回报（引入基线的最简单替代方案，降低方差）
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def reinforce_update(policy_net, optimizer, saved_log_probs, rewards):
    # 计算带有标准化基线的 G_t
    returns = compute_returns(rewards)

    policy_loss = []
    # 遍历轨迹中的每一步，计算对数梯度与回报的乘积
    for log_prob, G in zip(saved_log_probs, returns):
        # 策略梯度的实现：我们希望最大化目标函数 J，即最小化 -J
        # 由于 PyTorch 的优化器执行的是梯度下降（Gradient Descent）
        # 因此我们在前向附加一个负号
        policy_loss.append(-log_prob * G)

    # 对时间步求和，反向传播
    policy_loss = torch.stack(policy_loss).sum()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
```

在上述代码中，我们定义了策略网络，并在 `reinforce_update` 中展示了该公式的实现逻辑。值得注意的是，我们将原本在该公式中由价值网络估算状态价值的严谨过程，替换为了在张量维度上直接计算所有采样的 `returns` 的均值和标准差。这是一种极简的、计算代价低廉的基线技巧，虽然不如 Actor-Critic 方法那般精准，但依然大幅提升了朴素 REINFORCE 算法的稳定性。

## 小结

在本节中，我们详细拆解了强化学习中的策略梯度定理：

1. 从马尔可夫决策过程的轨迹分布出发，我们定义了由策略网络参数化的目标期望函数。
2. 通过引入复合函数求导中经典的**对数导数技巧**，我们将目标函数的梯度转化为可以从环境中采样的期望形式，克服了无法获知环境状态转移概率分布的难题。
3. 遵循严格的因果时间序律，我们使用未来累积回报取代了轨迹总回报。
4. 为了抑制蒙特卡洛采样带来的巨大方差，我们引入了状态价值、动作价值的严格定义，并通过推导优势函数，引入了用于修正权重的**基线（Baseline）**概念。

这些数学基础是深度强化学习后续所有高级算法（诸如 PPO、TRPO、SAC）不可或缺的底层逻辑。理解并能够自行手推上述的每一步公式转换，将为您理解复杂环境下的决策智能打下最坚实的基石。
