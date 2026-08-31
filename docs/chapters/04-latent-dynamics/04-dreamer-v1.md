# Dreamer V1：在潜在想象中学习

## 引言与学术背景

强化学习（Reinforcement Learning）在过去十多年中取得了令人瞩目的成就，特别是在高维感知输入（如图像）的任务中。然而，传统的无模型（Model-Free）强化学习算法（例如 DQN、PPO 和 SAC）通常需要海量的与环境交互的经验才能收敛。这种低下的样本效率在许多现实世界的物理任务中是不可接受的，因为收集真实数据的成本极其高昂且耗时。

基于模型（Model-Based）的强化学习长期研究如何用学到的模型辅助决策。Dyna 把真实经验更新与模型生成的规划更新结合起来 [[Sutton, 1990]](https://dl.acm.org/doi/10.5555/645530.658292)。PlaNet 随后提出**循环状态空间模型**（Recurrent State Space Model, RSSM），在紧凑的潜在空间中学习图像观测的动作条件动力学，并在每个环境步使用交叉熵方法（CEM）在线搜索动作序列 [[Hafner et al., 2019]](https://arxiv.org/abs/1811.04551)。这种反复采样和评估候选序列的方式，比直接执行一个参数化策略需要更多在线计算。

Hafner 等人随后提出 Dreamer（现常称为 DreamerV1）[[Hafner et al., 2020]](https://arxiv.org/abs/1912.01603)。Dreamer 不在部署时运行 CEM，而是在世界模型生成的潜在想象轨迹上训练参数化的 actor 与 critic。实际交互时，模型先更新潜在状态，再由 actor 输出动作。训练 actor 时，价值估计的梯度可以沿想象轨迹和学到的动力学反向传播；论文在其视觉控制基准上报告了较高的数据效率。

在本节中，我们将详细剖析 DreamerV1 的理论基础和算法机制，逐步推导从潜在空间动力学到 $\lambda$-回报（$\lambda$-return）的解析梯度计算过程，并最终通过代码实现一个微型的 Dreamer 架构。

## 世界模型的回顾：循环状态空间模型

在深入策略学习之前，我们需要先在数学上严谨地定义我们所处的“潜在想象空间”。Dreamer 沿用了 PlaNet 提出的 RSSM 作为其世界模型的基础。为了照顾初学者，我们从最简单的离散时间动态系统开始。

在经典物理学中，如果我们知道一个小球当前的位置 $x_t$ 和速度 $v_t$，并对其施加一个力（动作） $a_t$，我们可以通过牛顿运动定律确定地计算出它在下一时刻的状态 $(x_{t+1}, v_{t+1})$。这里的 $(x_t, v_t)$ 就是系统的**真实状态**。

但在现实复杂的强化学习任务中，我们通常只能观察到高维且包含噪声的图像（像素），我们将其称为观测值（Observation） $o_t$。这些观测值并未直接暴露环境的真实状态，因此我们需要一个神经网络将历史的观测和动作压缩为一个低维的潜在状态表示 $s_t$。

假设状态是完全离散标量，那么一步的时间演化可以被最简单地写为：
$s_{t+1} = f(s_t, a_t)$

但这忽略了环境固有的随机性（Stochasticity）。为了让模型能够应对不确定性，我们需要状态转移服从某个概率分布。Dreamer 中的潜在状态模型包含以下三个核心的概率分布：

1. **状态转移模型（Transition Model）**：描述在给定当前潜在状态和动作的情况下，环境如何向下一时刻演化。
   $$p(s_{t+1} \mid s_t, a_t)$$

2. **观测模型（Observation Model / Decoder）**：从潜在状态重建当前观测值，其主要作用是提供重建损失以训练潜在空间，从而保证潜在状态蕴含了当前环境的所有视觉信息。
   $$q(o_t \mid s_t)$$

3. **奖励模型（Reward Model）**：从潜在状态预测在当前时刻能够获得的奖励。
   $$q(r_t \mid s_t)$$

值得注意的是，RSSM 将潜在状态 $s_t$ 分成了**确定性**（Deterministic）和**随机性**（Stochastic）两部分，这也就是为什么被称为“循环状态空间”。确定性部分通常由一个门控循环单元（GRU）维护，用于长期的历史记忆；随机部分则通常建模为多变量高斯分布（Gaussian Distribution），用于捕捉环境在某一时步发生的不可预测事件。为了保持符号的整洁，在接下来的价值推导中，我们依然将它们统称为 $s_t$。

## 潜在想象中的多步价值评估

现在，假设我们已经使用观测数据训练好了一个精准的世界模型。接下来，我们需要在这个模型的大脑里进行“做梦（Dreaming）”——也就是在不与真实环境产生任何交互的情况下，仅凭借潜在模型来推演未来。

在时间步 $t$，真实环境向智能体提供了一个观测 $o_t$。通过世界模型的编码器，我们将这个时刻固定为一个初始的潜在状态 $s_t$。从这一步开始，我们让策略网络和世界模型相互配合，向未来推演 $H$ 步，我们用变量 $\tau$ 来索引这个**想象**的视野空间（即 $\tau = t, t+1, \dots, t+H$）。

我们的目标是训练一个**行动者网络（Actor）**，它输出动作分布 $q_{\phi}(a_\tau \mid s_\tau)$（由参数 $\phi$ 参数化）。但要训练行动者，我们需要知道“在状态 $s_\tau$ 采取动作 $a_\tau$ 有多好”，这就需要评估累积奖励。

### 从单步到 $k$ 步的价值（回报）推导

最简单的评估方式是直接将预测的奖励相加。但在长期任务中，我们需要考虑折扣因子 $\gamma \in [0, 1)$。
假设我们现在处于想象轨迹的某一个状态 $s_\tau$。我们如何衡量这个状态的价值呢？

如果不考虑后续的探索，直接使用一个**评论家网络（Critic）** $v_{\psi}(s)$（由参数 $\psi$ 参数化）来预测状态的价值，我们就可以得到所谓的单步（1-step）回报：

$$V^{1}_{\tau} \approx \mathbb{E} \big[ r(s_\tau) + \gamma v_{\psi}(s_{\tau+1}) \big]$$

其中，奖励 $r(s_\tau)$ 是世界模型中奖励预测网络的直接输出（取期望均值），而 $s_{\tau+1}$ 则是状态转移模型推演出的下一个状态。

单步回报极其依赖评论家网络的初始准确度。如果评论家由于训练不充分而存在严重偏差，行动者就会被误导。既然我们在想象空间中推演了 $H$ 步，为什么不多看几步呢？于是，我们可以将展开的步数扩展到 $k$ 步（$1 \leq k \leq H - (\tau - t)$）：

$$V^{k}_{\tau} = \sum_{n=0}^{k-1} \gamma^n r(s_{\tau+n}) + \gamma^k v_{\psi}(s_{\tau+k})$$

该公式是一个非常优雅的表达。我们可以这样理解：前 $k$ 步的奖励都是我们在世界模型的“梦境”中**显式预测**出来的，它们往往比评论家给出的一个抽象概括要准确（在世界模型训练良好的前提下）。只有到了第 $k$ 步的尽头时，我们才无奈地使用评论家 $v_{\psi}(s_{\tau+k})$ 来对更遥远、甚至超出视野的未来进行“兜底”。

### 指数衰减的 $\lambda$-回报

那么，最佳的步数 $k$ 到底应该是多少？
如果 $k$ 太小，模型容易受到评论家偏差的影响（偏差大，Bias）；如果 $k$ 很大接近 $H$，由于世界模型多步自回归预测会累积误差，且动作采样引入了大量随机性，最终的回报估计方差会急剧上升（方差大，Variance）。

为了在偏差和方差之间取得完美的平衡，Dreamer 引入了类似于强化学习基础概念中 TD($\lambda$) 的思想：对所有可能的 $k$ 步回报进行加权平均。具体而言，它使用一个参数 $\lambda \in [0, 1]$ 来决定权重，权重随着 $k$ 的增加而呈指数递减。

我们正式定义 $\lambda$-回报（$\lambda$-return） $V^{\lambda}_{\tau}$。为了避免陷入复杂的连加符号迷宫中，让我们用一个极其简洁直观的递归公式来定义它：

$$V^{\lambda}_{\tau} = r(s_\tau) + \gamma \Big( (1-\lambda) v_{\psi}(s_{\tau+1}) + \lambda V^{\lambda}_{\tau+1} \Big)$$

这其实就是在说：站在状态 $s_\tau$ 的角度，下一步的回报期望是由两部分组成的混血儿。其中一部分是评论家的稳妥预测 $v_{\psi}(s_{\tau+1})$（权重为 $1-\lambda$），另一部分是我们将梦想继续延伸一步得到的更长远的 $\lambda$-回报估计 $V^{\lambda}_{\tau+1}$（权重为 $\lambda$）。这种递归形式在编程实现时极其高效，只需从时间轴末端（即 $\tau = t+H$）向前反推即可。

通过计算整条想象轨迹上每个时间步的 $\lambda$-回报，我们就获得了一个兼顾偏差与方差的高质量评估基准，它将直接用于指导行动者和评论家的更新。

## 解析梯度：通过动力学模型反向传播

在传统的无模型强化学习（如 PPO）中，由于真实环境是一个不可微（Non-differentiable）的黑盒，当行动者网络尝试最大化累积回报时，必须依赖对数似然技巧（REINFORCE，即策略梯度定理）来估计梯度。这种梯度估计充满了噪声，导致样本效率低下。

而在 Dreamer 中，因为整个 $H$ 步的推演是完全在**神经网络构成的世界模型**中完成的，这个“假环境”是端到端全可微的（Differentiable）！这就允许我们使用一种更直接、更精准的方式——**解析梯度（Analytic Gradients）**来进行策略优化。

::: info 说明
想象我们的大脑不仅能预测打台球时球的轨迹（世界模型），还能敏锐地感知到，当我们手部肌肉的发力角度微调0.1度时，球最终落袋的概率会随之产生确切数学规律上的变化（可导的梯度）。在这个“梦境”里，我们不需要真的去推几千次杆来统计概率（无模型RL的做法），而是可以在大脑里通过解析方程，直接求得最完美的发力角度。这是潜在想象中最具威力的一环。
:::

为了实现反向传播通过状态的转移，我们需要应用**重参数化技巧（Reparameterization Trick）**。
行动者网络输出动作的分布参数（例如高斯分布的均值 $\mu_\phi$ 和标准差 $\sigma_\phi$）。我们将动作 $a_\tau$ 采样过程重写为：

$$a_\tau = \mu_\phi(s_\tau) + \sigma_\phi(s_\tau) \cdot \xi, \quad \xi \sim \mathcal{N}(0, I)$$

同样地，状态的随机转移也可以重参数化为一个可导的函数：
$$s_{\tau+1} = f_\theta(s_\tau, a_\tau, \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)$$

此时，整个轨迹的回报 $V^{\lambda}_{\tau}$ 变成了一个巨大的、由可导算子组成的计算图，其叶子节点包含了行动者的参数 $\phi$。

### 行动者（Actor）的损失函数

行动者网络的目标非常纯粹：最大化在想象轨迹中所有的初始状态 $s_\tau$ 上计算出的 $\lambda$-回报期望。因此，行动者的损失函数定义为回报的负值：

$$\mathcal{L}_{\text{actor}}(\phi) = - \mathbb{E} \left[ \sum_{\tau=t}^{t+H-1} V^{\lambda}_{\tau} \right]$$

在实际更新时，由于重参数化技巧，PyTorch 等深度学习框架的自动微分引擎可以直接从 $V^{\lambda}_{\tau}$ 出发，沿着时间维度穿过奖励模型、转移动力学模型，最终将精确的梯度 $\nabla_\phi \mathcal{L}_{\text{actor}}$ 流入行动者网络的参数 $\phi$ 中。

### 评论家（Critic）的损失函数

与此同时，评论家网络的作用是尽可能准确地预测状态价值，以帮助行动者计算回报。我们使用我们精心计算的 $\lambda$-回报作为它的回归目标（Target）。评论家的损失函数是预测值与 $\lambda$-回报目标之间的均方误差（MSE）：

$$\mathcal{L}_{\text{critic}}(\psi) = \mathbb{E} \left[ \sum_{\tau=t}^{t+H-1} \frac{1}{2} \big( v_{\psi}(s_\tau) - \text{stop\_gradient}(V^{\lambda}_{\tau}) \big)^2 \right]$$

注意公式中的 `stop_gradient` 操作。这意味着在训练评论家时，我们将 $V^{\lambda}_{\tau}$ 视为一个固定的真实标签，不需要通过它再去反向传播梯度。我们只更新评论家自身的参数 $\psi$。

## 代码实现

下面，我们将通过一段精简的代码，演示在 PyTorch 中如何实现在潜空间中推演并计算 $\lambda$-回报这一 DreamerV1 的核心过程。

(**首先，我们定义世界模型的桩（Stub）以及行动者与评论家网络。**) 为了聚焦于强化学习本身的算法原理，我们假设状态转移模型 `transition_model` 已经训练完毕，它能够接收当前状态和动作，通过重参数化返回下一个潜在状态。

```python
import torch
import torch.nn as nn
import torch.distributions as td

class DummyWorldModel(nn.Module):
    """一个简化的世界模型，仅用于演示计算图连接"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 简单使用线性层模拟复杂的动力学
        self.dynamics = nn.Linear(state_dim + action_dim, state_dim)
        self.reward_predictor = nn.Linear(state_dim, 1)

    def step(self, state, action):
        # 模拟下一个状态生成，并包含重参数化噪声
        next_state_mean = self.dynamics(torch.cat([state, action], dim=-1))
        # 简单的重参数化，方差恒定为0.1
        noise = torch.randn_like(next_state_mean) * 0.1
        next_state = next_state_mean + noise
        return next_state

    def predict_reward(self, state):
        return self.reward_predictor(state)

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ELU(),
            nn.Linear(64, action_dim * 2) # 输出均值和方差
        )

    def forward(self, state):
        out = self.net(state)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        # 限制方差范围以求稳定
        std = torch.exp(torch.clamp(log_std, min=-5, max=2))
        return mean, std

    def get_action_reparameterized(self, state):
        mean, std = self.forward(state)
        # 在PyTorch中，使用 rsample 进行带重参数化技巧的采样
        dist = td.Normal(mean, std)
        # 为简单起见，不应用Tanh激活带来的分布变换修正
        action = dist.rsample()
        return action

class CriticNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )
    def forward(self, state):
        return self.net(state)
```

(**接下来，我们实现潜在想象的核心循环：前向展开多步，计算递归的 $\lambda$-回报，并更新策略网络。**)

```python
def train_imagination_step(
    world_model, actor, critic, start_states,
    actor_optimizer, critic_optimizer, horizon=15, gamma=0.99, lam=0.95):

    # start_states 维度: (batch_size, state_dim)，是从重播缓冲采样后经过编码器得到的真实起点。
    batch_size, state_dim = start_states.shape

    states = [start_states]
    actions = []
    rewards = []

    # 1. 展开想象序列 (Rollout in Latent Space)
    # 不切断计算图，让梯度能一直流过所有时间步
    curr_state = start_states
    for t in range(horizon):
        action = actor.get_action_reparameterized(curr_state)
        next_state = world_model.step(curr_state, action)
        reward = world_model.predict_reward(next_state)

        actions.append(action)
        rewards.append(reward)
        states.append(next_state)
        curr_state = next_state

    # 转换为张量，维度: (horizon, batch_size, ...)
    states_tensor = torch.stack(states)   # (H+1, B, D)
    rewards_tensor = torch.stack(rewards) # (H, B, 1)

    # 2. 计算价值估计
    values = critic(states_tensor) # 形状: (H+1, B, 1)

    # 3. 递归计算 lambda 回报
    # V^\lambda_t = r_t + \gamma * ((1 - \lambda) * v_{t+1} + \lambda * V^\lambda_{t+1})
    lambda_returns = []

    # 从最后一个时间步向后递归
    # 对于最后一步（第 H 步），没有长远的预测，只能直接使用 critic
    last_val = values[-1]

    for t in reversed(range(horizon)):
        # 核心递归逻辑
        # 注意: values[t+1] 是网络在 t+1 状态的预测值
        last_val = rewards_tensor[t] + gamma * ((1 - lam) * values[t+1] + lam * last_val)
        # 将结果插入列表开头以保持时间顺序
        lambda_returns.insert(0, last_val)

    lambda_returns = torch.stack(lambda_returns).detach() # (H, B, 1) 作为评论家目标，切断梯度

    # 4. 计算 Actor 和 Critic 损失
    # 截取对应的时间步状态（0到H-1）
    states_to_evaluate = states_tensor[:-1]

    # 评论家损失：MSE (预测值, 目标值)
    pred_values = critic(states_to_evaluate.detach()) # 训练Critic时不对状态流进行反向传播
    critic_loss = torch.mean(0.5 * (pred_values - lambda_returns)**2)

    # 行动者损失：最大化回报等价于最小化负回报
    # 在本实现中，我们直接将生成的状态送入critic计算平均值，因为生成这些状态已经穿过了actor。
    actor_loss = -torch.mean(critic(states_to_evaluate))

    # 5. 反向传播更新
    actor_optimizer.zero_grad()
    actor_loss.backward(retain_graph=True)
    actor_optimizer.step()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    return actor_loss.item(), critic_loss.item()
```

在上面的代码中，我们揭示了一个极其容易混淆的细节：在计算 `lambda_returns` 作为基准信号时，我们将其使用了 `.detach()` 截断（防止梯度倒流干扰 Critic 或导致计算图死锁）。而在更新 Actor 时，我们必须保留生成 `states_to_evaluate` 的计算图。因为这些**未来的状态是由当前的 Actor 的动作所生成的**，让价值网络 `critic()` 对其进行打分，能够将“如何改变动作以到达更高价值的状态”这一数学推导顺着网络链路反向传播回 Actor 本身。

## 总结

- **DreamerV1** 标志着基于模型的强化学习（Model-based RL）的一个重要转折点。它证明了纯粹在潜在空间中学习策略不仅是可行的，而且是极其高效的。
- 通过引入**指数衰减的 $\lambda$-回报**，算法能在依赖模型自回归的“高方差推演”与直接依赖价值网络的“高偏差估计”之间寻找到理论上的最佳平衡。
- 全程可微的世界模型赋予了我们使用**解析梯度（Analytic Gradients）**反向传播的能力。这意味着模型能够通过深层的网络动力学，直接告诉策略应该向哪个方向微调动作才能获得最丰厚的未来回报。这种通过想象学习的范式，深刻地模拟了智能生命规划未来的行为逻辑。
