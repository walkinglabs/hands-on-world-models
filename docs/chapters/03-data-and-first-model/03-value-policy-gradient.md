# 策略梯度与价值函数基础
:label:sec_policy_gradient

## 强化学习的优化范式追溯
:label:subsec_rl_history

在深度学习的早期，监督学习依赖于明确的标签（即专家的示范）来更新模型。然而，在诸如游戏、机器人控制或自动驾驶等开放环境中，获取逐帧的最优决策标签是极其昂贵甚至不可能的。强化学习（Reinforcement Learning, RL）提供了一种替代范式：智能体（Agent）通过与环境的交互，收集奖励（Reward）信号，并以此为导向优化自身的行为。

在早期的表格型强化学习中，学者们通常关注如何精确估计处于某个状态的“价值”（即价值函数法）。但在连续控制或高维动作空间中，价值函数法面临“维度灾难”。直到 `[Williams, 1992]` 提出了 REINFORCE 算法，以及 `[Sutton et al., 1999]` 正式确立了策略梯度定理（Policy Gradient Theorem），直接优化策略本身（Policy-based methods）才成为现代深度强化学习的核心基石之一。本文将从最基础的概率论出发，毫无跳跃地为您推导出策略梯度定理的严密形式，并引入价值函数作为降低估计方差的关键工具。

## 策略与轨迹的数学描述
:label:subsec_policy_trajectory

在我们引入任何复杂的张量运算之前，首先必须对智能体与环境的交互过程进行严格的数学定义。
假设我们处于一个马尔可夫决策过程（MDP）中，时间由离散的步长 $t=0, 1, 2, \dots$ 组成。

### 策略函数的定义

(**定义策略（Policy）作为条件概率分布**)
我们将策略定义为在给定当前状态 $s$ 的情况下，选择某个动作 $a$ 的条件概率分布。如果策略由参数 $\theta$（例如神经网络的权重）参数化，我们将其记为：

$$
\pi_\theta(a|s) = P(A_t=a | S_t=s; \theta)
$$
:eqlabel:eq_policy_def

这里的 $\theta$ 是我们要优化的核心物理量。在最简单的一维离散动作空间中，$\pi_\theta(a|s)$ 可能仅仅是一个 softmax 函数的输出；而在连续动作空间中，它通常是一个高斯分布的概率密度函数，其中均值和方差由神经网络预测。

### 轨迹与期望回报

智能体在环境中从初始状态 $s_0$ 出发，依据策略 $\pi_\theta$ 选择动作 $a_0$，环境根据其转移概率分布（Transition Dynamics）$P(s_{1}|s_0, a_0)$ 给出下一个状态 $s_1$ 和奖励 $r_1$。这一过程循环往复，形成一条轨迹（Trajectory，或称为回合 Episode），记为 $\tau$：

$$
\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_{T-1}, a_{T-1}, r_T)
$$
:eqlabel:eq_trajectory

轨迹发生的联合概率由策略和环境共同决定。利用概率论中的链式法则（乘法公式），我们可以将轨迹 $\tau$ 的发生概率写为：

$$
P(\tau | \theta) = P(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t) P(s_{t+1}|s_t, a_t)
$$
:eqlabel:eq_traj_prob

其中 $P(s_0)$ 是初始状态分布。请注意，这个长串的连乘公式中，只有 $\pi_\theta(a_t|s_t)$ 包含了我们的模型参数 $\theta$。

一条轨迹的总回报（Return）是沿着该轨迹收集到的所有奖励之和。为了保证序列收敛（特别是在无限期任务中），我们引入折扣因子（Discount Factor） $\gamma \in [0, 1]$：

$$
R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_{t+1}
$$
:eqlabel:eq_return

## 目标函数与对数导数技巧
:label:subsec_objective_log_trick

强化学习的终极目标是找到一组最优参数 $\theta^*$，使得轨迹的总回报的期望值（Expected Return）最大化。我们将这个目标函数记为 $J(\theta)$。

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] = \int P(\tau | \theta) R(\tau) d\tau
$$
:eqlabel:eq_objective

如果是离散动作空间，这里的积分号就替换为对所有可能轨迹的求和。
为了最大化 $J(\theta)$，在微积分和深度学习的体系中，我们最直接的方法就是使用梯度上升（Gradient Ascent）：

$$
\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)
$$
:eqlabel:eq_gradient_ascent

其中 $\alpha$ 是学习率。现在的核心难题是：如何求解目标函数的梯度 $\nabla_\theta J(\theta)$？

### 对数导数技巧（Log-Derivative Trick）

在高中数学中，我们知道复合函数求导的链式法则。特别地，对于自然对数函数 $f(x) = \log x$，其导数为 $f'(x) = 1/x$。由此我们可以得到一个极为重要的恒等式，它被称为对数导数技巧：

$$
\nabla_\theta \log P(\tau|\theta) = \frac{\nabla_\theta P(\tau|\theta)}{P(\tau|\theta)}
$$
:eqlabel:eq_log_derivative_1

将其稍微移项，我们得到：

$$
\nabla_\theta P(\tau|\theta) = P(\tau|\theta) \nabla_\theta \log P(\tau|\theta)
$$
:eqlabel:eq_log_derivative_2

这个简单的公式是整个策略梯度算法的灵魂所在。回到我们的目标函数梯度：

$$
\begin{aligned}
\nabla_\theta J(\theta) &= \nabla_\theta \int P(\tau | \theta) R(\tau) d\tau \\
&= \int \nabla_\theta P(\tau | \theta) R(\tau) d\tau \\
&= \int P(\tau|\theta) \nabla_\theta \log P(\tau|\theta) R(\tau) d\tau \\
&= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau|\theta) R(\tau) \right]
\end{aligned}
$$
:eqlabel:eq_policy_gradient_derivation

这里我们做了一次极其关键的转换：将原本对概率分布本身的求导，转化为了对“某个随机变量求期望”的形式。因为在实际应用中，我们不可能穷举所有的轨迹来积分，但我们可以通过让智能体在环境中运行多次来**采样**轨迹，进而利用蒙特卡洛方法（Monte Carlo method）来无偏地估计这个期望值。

### 拆解轨迹概率的对数梯度

接着，我们需要计算 $\nabla_\theta \log P(\tau|\theta)$。对 :eqref:eq_traj_prob 两边取对数，乘积将变成求和：

$$
\log P(\tau|\theta) = \log P(s_0) + \sum_{t=0}^{T-1} \left( \log \pi_\theta(a_t|s_t) + \log P(s_{t+1}|s_t, a_t) \right)
$$
:eqlabel:eq_log_traj_prob

现在，对等式两边关于 $\theta$ 求梯度。奇妙的事情发生了：在这个式子中，初始状态分布 $P(s_0)$ 和环境的状态转移概率 $P(s_{t+1}|s_t, a_t)$ 完全与参数 $\theta$ 无关！因此，它们的梯度统统为零。

$$
\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)
$$
:eqlabel:eq_grad_log_traj_prob

将 :eqref:eq_grad_log_traj_prob 代回 :eqref:eq_policy_gradient_derivation，我们得到了经典的 REINFORCE 策略梯度公式：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau) \right]
$$
:eqlabel:eq_reinforce_gradient

> [!CAUTION]
> 这是一个在强化学习中少有的、允许我们在不知道环境转移概率（即所谓 Model-free）的情况下，仍然可以计算目标函数梯度的公式。这也是 REINFORCE 算法名称中 "Score Function Estimator" 的由来。

## 因果性与奖励截断
:label:subsec_causality

:eqref:eq_reinforce_gradient 虽然形式严密，但在方差（Variance）上存在巨大的问题。让我们仔细观察公式中的项 $\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau)$。它表示时间步 $t$ 采取动作 $a_t$ 的对数概率梯度，乘以整条轨迹的**总回报** $R(\tau)$。

从基本的逻辑因果律（Causality）出发：在时刻 $t$ 采取的动作 $a_t$，绝对不可能影响时刻 $t$ 之前的奖励 $r_{1}, r_{2}, \dots, r_t$。动作只能影响未来的奖励。因此，用整条轨迹的总回报来评估某个中间动作的好坏，引入了大量无意义的噪声。

我们可以严格证明（篇幅所限，此处不展开纯代数推导），对于任意 $t' < t$，有 $\mathbb{E} [ \nabla_\theta \log \pi_\theta(a_t|s_t) \gamma^{t'} r_{t'+1} ] = 0$。因此，我们可以心安理得地将总回报 $R(\tau)$ 替换为从时刻 $t$ 开始的**未来累积回报**（Return to go），通常记为 $G_t$：

$$
G_t = \sum_{t'=t}^{T-1} \gamma^{t'-t} r_{t'+1}
$$
:eqlabel:eq_return_to_go

由此，策略梯度公式被改良为：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]
$$
:eqlabel:eq_policy_gradient_causal

这个公式直观地表达了强化学习的本质原理：如果一个动作导致了较高的未来累积回报 $G_t > 0$，那么其对数概率的梯度方向就会被放大，网络在下一次遇到类似状态时，输出该动作的概率就会提高；反之亦然。

## 价值函数的引入与基线技巧
:label:subsec_value_function_baseline

尽管因果性截断降低了一部分方差，但蒙特卡洛采样的本质决定了 $G_t$ 的波动仍然极大。为了进一步稳定训练，我们需要引入**价值函数（Value Function）**的概念。

### 状态价值与动作价值

(**定义状态价值函数（State Value Function）**)
状态价值 $V^\pi(s)$ 衡量了智能体处于状态 $s$，并遵循策略 $\pi$ 进行决策，直至回合结束所能期望获得的累积回报：

$$
V^\pi(s) = \mathbb{E}_{\tau \sim \pi} [G_t | S_t = s]
$$
:eqlabel:eq_state_value

(**定义动作价值函数（Action Value Function）**)
动作价值 $Q^\pi(s, a)$ 衡量了智能体处于状态 $s$，**首先采取动作 $a$**，随后严格遵循策略 $\pi$ 所能期望获得的累积回报：

$$
Q^\pi(s, a) = \mathbb{E}_{\tau \sim \pi} [G_t | S_t = s, A_t = a]
$$
:eqlabel:eq_action_value

两者的关系非常直接：处于状态 $s$ 的期望价值，就是选择各种动作能够带来的期望价值，按照策略给出的概率分布进行的加权平均。这也是全概率公式的一种体现。

$$
V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a)
$$
:eqlabel:eq_vq_relation

### 引入基线（Baseline）

在 :eqref:eq_policy_gradient_causal 中，我们用 $G_t$ 作为评估动作好坏的权重。然而，如果环境的所有奖励都是非常大的正数（例如总是给 +100 到 +200 之间的奖励），那么所有的 $G_t$ 都会很大，策略梯度会不加区分地尝试提高所有动作的概率。

> 唯一的精炼类比：假设你是一家公司的老板（策略网络），你需要给员工（动作）发奖金来鼓励他们。如果公司整体效益很好，每个员工都发了一百万奖金（$G_t$ 很大），这并不能反映出哪个员工的贡献最突出。更好的方法是设定一个“公司平均绩效”（基线）。如果某个员工的业绩超过了平均绩效，我们才重点提拔他（提高概率），否则即使他赚了钱，只要低于平均，我们也应该削减他的奖金（相对降低概率）。

在数学上，我们可以证明，在策略梯度的计算式中减去任意一个不依赖于具体动作 $a_t$ 的基线函数 $b(s_t)$，都不会改变梯度的无偏期望值。最自然、最合理的基线选择，正是状态价值函数 $V^\pi(s_t)$。

我们定义**优势函数（Advantage Function）** 为动作价值与状态价值之差：

$$
A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)
$$
:eqlabel:eq_advantage_function

利用优势函数，最终的、同时具备极低方差和无偏特性的策略梯度定理（Actor-Critic 架构的理论基础）可以写为：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^\pi(s_t, a_t) \right]
$$
:eqlabel:eq_pg_advantage

在实际代码实现中，如果我们只使用纯粹的 REINFORCE 算法，往往只会减去一个基于蒙特卡洛采样的基线。下面，我们将通过具体的代码来实现带有基线的 REINFORCE 算法。

## 代码实现：带基线的 REINFORCE
:label:subsec_reinforce_implementation

以下代码展示了如何使用深度学习框架实现带基线的策略梯度优化。我们假设环境提供了一维离散动作空间，并使用多层感知机（MLP）作为策略网络。

```{.python .input}
#@tab pytorch
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

```{.python .input}
#@tab tensorflow
import tensorflow as tf
import numpy as np

class PolicyNetwork(tf.keras.Model):
    def __init__(self, action_dim):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(action_dim, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        action_probs = self.fc2(x)
        return action_probs

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = tf.convert_to_tensor(returns, dtype=tf.float32)
    returns = (returns - tf.reduce_mean(returns)) / (tf.math.reduce_std(returns) + 1e-8)
    return returns

def reinforce_update(policy_net, optimizer, saved_log_probs, rewards):
    returns = compute_returns(rewards)
    
    policy_loss = []
    for log_prob, G in zip(saved_log_probs, returns):
        policy_loss.append(-log_prob * G)
        
    policy_loss = tf.reduce_sum(policy_loss)
    
    with tf.GradientTape() as tape:
        # 这里为了演示对齐 PyTorch 逻辑，实际在 TF 中 log_prob 是通过 tape 计算的
        # 在实际实现中，通常会将前向采样和 loss 计算一同放在 tape 内部
        pass
    # 假设我们从 tape 获取了对网络权重的梯度
    # gradients = tape.gradient(policy_loss, policy_net.trainable_variables)
    # optimizer.apply_gradients(zip(gradients, policy_net.trainable_variables))
```

在上述代码中，我们定义了策略网络，并在 `reinforce_update` 中展示了 :eqref:eq_policy_gradient_causal 的实现逻辑。值得注意的是，我们将原本在 :eqref:eq_advantage_function 中由价值网络估算状态价值的严谨过程，替换为了在张量维度上直接计算所有采样的 `returns` 的均值和标准差。这是一种极简的、计算代价低廉的基线技巧，虽然不如 Actor-Critic 方法那般精准，但依然大幅提升了朴素 REINFORCE 算法的稳定性。

## 小结

在本节中，我们详细拆解了强化学习中的策略梯度定理：
1. 从马尔可夫决策过程的轨迹分布出发，我们定义了由策略网络参数化的目标期望函数。
2. 通过引入复合函数求导中经典的**对数导数技巧**，我们将目标函数的梯度转化为可以从环境中采样的期望形式，克服了无法获知环境状态转移概率分布的难题。
3. 遵循严格的因果时间序律，我们使用未来累积回报取代了轨迹总回报。
4. 为了抑制蒙特卡洛采样带来的巨大方差，我们引入了状态价值、动作价值的严格定义，并通过推导优势函数，引入了用于修正权重的**基线（Baseline）**概念。

这些数学基础是深度强化学习后续所有高级算法（诸如 PPO、TRPO、SAC）不可或缺的底层逻辑。理解并能够自行手推上述的每一步公式转换，将为您理解复杂环境下的决策智能打下最坚实的基石。

## 练习

1. **证明对数导数恒等式的多维形式**：我们在 :eqref:eq_log_derivative_1 中利用一维标量函数的导数得出了对数导数技巧。如果 $P(x|\theta)$ 是一个多元高斯分布，请尝试写出该多元分布密度函数的对数，并验证该技巧对其参数化均值和协方差矩阵的梯度计算是否依然成立。
   - *提示*：先回顾多元高斯分布的概率密度函数，写出其自然对数表达式，你会发现指数项将转化为二次型，求导过程非常直观。
2. **基线无关性证明**：在 :eqref:eq_policy_gradient_causal 中，试着在 $G_t$ 内部减去一个仅依赖于当前状态 $s_t$ 的任意标量函数 $b(s_t)$。证明此举不会改变梯度的期望值（即 $\mathbb{E}[\nabla_\theta \log \pi_\theta(a_t|s_t) b(s_t)] = 0$）。
   - *提示*：由于 $b(s_t)$ 独立于当前动作 $a_t$，你可以将期望写为积分或求和形式，将梯度算子移入，并将全概率积分对该项的结果化简为零（因为所有动作的概率之和始终为 1 的导数必然是 0）。

:begin_tab:pytorch
[讨论](https://discuss.d2l.ai/t/1234)
:end_tab:
