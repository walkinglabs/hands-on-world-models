# 4.4 DreamerV1：在潜在想象中学习

## 引言与学术背景

无模型（Model-Free）强化学习直接从环境经验更新策略或价值函数，通常需要较多交互。对真实机器人而言，数据采集还受到时间、磨损和安全约束。基于模型的方法尝试复用已有经验，在学到的动力学中评估更多未来；Dreamer 的核心选择，就是在这种潜在想象中训练策略。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/04-dreamer-v1/opening-dreamer-quadruped.gif" alt="Dreamer 学到的四足策略连续奔跑，先展示潜在想象训练最终如何变成环境中的真实动作。" width="86%">

_图 4.4-1：Dreamer 学到的四足策略连续奔跑，先展示潜在想象训练最终如何变成环境中的真实动作。 出处：Danijar Hafner；Timothy Lillicrap；Jimmy Ba；Mohammad Norouzi，[Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)（2020），Official behavior GIF: Quadruped Run。_

</div>

基于模型（Model-Based）的强化学习长期研究如何用学到的模型辅助决策。Dyna 把真实经验更新与模型生成的规划更新结合起来 [[Sutton, 1990]](https://dl.acm.org/doi/10.5555/645530.658292)。PlaNet 随后提出**循环状态空间模型**（Recurrent State Space Model, RSSM），在紧凑的潜在空间中学习图像观测的动作条件动力学，并在每个环境步使用交叉熵方法（CEM）在线搜索动作序列 [[Hafner et al., 2019]](https://arxiv.org/abs/1811.04551)。这种反复采样和评估候选序列的方式，比直接执行一个参数化策略需要更多在线计算。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/04-dreamer-v1/planet-fig5.png" alt="PlaNet 的六任务曲线把在线 CEM 规划的效果和数据收集条件放进实验语境，衬托 Dreamer 改学参数化策略的动机。" width="86%">

_图 4.4-2：PlaNet 的六任务曲线把在线 CEM 规划的效果和数据收集条件放进实验语境，衬托 Dreamer 改学参数化策略的动机。 出处：Danijar Hafner；Timothy Lillicrap；Ian Fischer；Ruben Villegas；David Ha；Honglak Lee；James Davidson，[Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)（2019），Figure 5。_

</div>

Hafner 等人随后提出 Dreamer（现常称为 DreamerV1）[[Hafner et al., 2020]](https://arxiv.org/abs/1912.01603)。Dreamer 不在部署时运行 CEM，而是在世界模型生成的潜在想象轨迹上训练参数化的 actor 与 critic。实际交互时，模型先更新潜在状态，再由 actor 输出动作。训练 actor 时，价值估计的梯度可以沿想象轨迹和学到的动力学反向传播；论文在其视觉控制基准上报告了较高的数据效率。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/04-dreamer-v1/pilco-fig7.png" alt="PILCO 的倒立摆结果把成功率与真实交互秒数并列，给出 Dreamer 之前模型式强化学习追求数据效率的实证背景。" width="86%">

_图 4.4-3：PILCO 的倒立摆结果把成功率与真实交互秒数并列，给出 Dreamer 之前模型式强化学习追求数据效率的实证背景。 出处：Marc Peter Deisenroth；Diether Fox；Carl Edward Rasmussen，[PILCO: A Model-Based and Data-Efficient Approach to Policy Search](https://doi.org/10.1109/TPAMI.2013.218)（2015），Figure 7。_

</div>

本节说明 DreamerV1 如何从 RSSM 状态出发生成想象轨迹，用 $\lambda$-回报训练价值模型，并把回报梯度沿可微动力学传回 actor。最后给出只覆盖这条核心计算路径的教学代码。

## 世界模型的回顾：循环状态空间模型

Dreamer 沿用 PlaNet 的 RSSM。先用简化符号回顾它提供哪些概率模型。

在经典物理学中，如果我们知道一个小球当前的位置 $x_t$ 和速度 $v_t$，并对其施加一个力（动作） $a_t$，我们可以通过牛顿运动定律确定地计算出它在下一时刻的状态 $(x_{t+1}, v_{t+1})$。这里的 $(x_t, v_t)$ 就是系统的**真实状态**。

但在现实复杂的强化学习任务中，我们通常只能观察到高维且包含噪声的图像（像素），我们将其称为观测值（Observation） $o_t$。这些观测值并未直接暴露环境的真实状态，因此我们需要一个神经网络将历史的观测和动作压缩为一个低维的潜在状态表示 $s_t$。

若暂时忽略随机性，一步状态转移可写为：
$s_{t+1} = f(s_t, a_t)$

但这忽略了环境固有的随机性（Stochasticity）。为了让模型能够应对不确定性，我们需要状态转移服从某个概率分布。Dreamer 中的潜在状态模型包含以下三个核心的概率分布：

1. **状态转移模型（Transition Model）**：描述在给定当前潜在状态和动作的情况下，环境如何向下一时刻演化。
   $$p(s_{t+1} \mid s_t, a_t)$$

2. **观测模型（Observation Model / Decoder）**：从潜在状态解释当前观测，为表征学习提供信号；潜在状态不必保留所有视觉细节。
   $$p_\theta(o_t \mid s_t)$$

3. **奖励模型（Reward Model）**：从潜在状态预测奖励分布。
   $$p_\theta(r_t \mid s_t)$$

值得注意的是，RSSM 将潜在状态 $s_t$ 分成了**确定性**（Deterministic）和**随机性**（Stochastic）两部分，这也就是为什么被称为“循环状态空间”。确定性部分通常由一个门控循环单元（GRU）维护，用于长期的历史记忆；随机部分则通常建模为多变量高斯分布（Gaussian Distribution），用于捕捉环境在某一时步发生的不可预测事件。为了保持符号的整洁，在接下来的价值推导中，我们依然将它们统称为 $s_t$。

## 潜在想象中的多步价值评估

<div align="center">
  <img src="/figures/04-latent-dynamics/source/04-dreamer-v1/dreamer-fig4.png" alt="Dreamer 在不同想象视野下的控制曲线显示，价值自举降低了策略学习对有限展开长度的敏感性。" width="86%">

_图 4.4-4：Dreamer 在不同想象视野下的控制曲线显示，价值自举降低了策略学习对有限展开长度的敏感性。 出处：Danijar Hafner；Timothy Lillicrap；Jimmy Ba；Mohammad Norouzi，[Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)（2020），Figure 4。_

</div>

假设世界模型已经用回放数据训练到可用于短期滚动。所谓“想象”，就是从真实序列推断出的潜在状态出发，不读取新的未来观测，仅用先验动力学与 actor 生成后续状态和动作。

在时间步 $t$，真实环境向智能体提供了一个观测 $o_t$。通过世界模型的编码器，我们将这个时刻固定为一个初始的潜在状态 $s_t$。从这一步开始，我们让策略网络和世界模型相互配合，向未来推演 $H$ 步，我们用变量 $\tau$ 来索引这个**想象**的视野空间（即 $\tau = t, t+1, \dots, t+H$）。

我们的目标是训练一个**行动者网络（Actor）**，它输出动作分布 $q_{\phi}(a_\tau \mid s_\tau)$（由参数 $\phi$ 参数化）。但要训练行动者，我们需要知道“在状态 $s_\tau$ 采取动作 $a_\tau$ 有多好”，这就需要评估累积奖励。

### 从单步到 $k$ 步的价值（回报）推导

最简单的评估方式是直接将预测的奖励相加。但在长期任务中，我们需要考虑折扣因子 $\gamma \in [0, 1)$。
假设我们现在处于想象轨迹的某一个状态 $s_\tau$。我们如何衡量这个状态的价值呢？

如果不考虑后续的探索，直接使用一个**评论家网络（Critic）** $v_{\psi}(s)$（由参数 $\psi$ 参数化）来预测状态的价值，我们就可以得到所谓的单步（1-step）回报：

$$V^{1}_{\tau} \approx \mathbb{E} \big[ r(s_\tau) + \gamma v_{\psi}(s_{\tau+1}) \big]$$

其中，奖励 $r(s_\tau)$ 是世界模型中奖励预测网络的直接输出（取期望均值），而 $s_{\tau+1}$ 则是状态转移模型推演出的下一个状态。

一步回报高度依赖评论家的 bootstrap 估计。若向前展开 $k$ 步，再在末端使用评论家，可得到：

$$V^{k}_{\tau} = \sum_{n=0}^{k-1} \gamma^n r(s_{\tau+n}) + \gamma^k v_{\psi}(s_{\tau+k})$$

前 $k$ 步使用世界模型预测的奖励，末端用 $v_\psi(s_{\tau+k})$ 估计剩余回报。增大 $k$ 会减少对近端 bootstrap 的依赖，却会增加对模型长滚动的依赖；哪一项更准取决于训练阶段与状态区域。

### 指数衰减的 $\lambda$-回报

那么，最佳的步数 $k$ 到底应该是多少？
如果 $k$ 太小，模型容易受到评论家偏差的影响（偏差大，Bias）；如果 $k$ 很大接近 $H$，由于世界模型多步自回归预测会累积误差，且动作采样引入了大量随机性，最终的回报估计方差会急剧上升（方差大，Variance）。

Dreamer 采用 TD($\lambda$) 风格的加权组合，在短期 bootstrap 与较长模型展开之间调节。参数 $\lambda\in[0,1]$ 控制更长回报的权重。

$\lambda$-回报可递归写成：

$$V^{\lambda}_{\tau} = r(s_\tau) + \gamma \Big( (1-\lambda) v_{\psi}(s_{\tau+1}) + \lambda V^{\lambda}_{\tau+1} \Big)$$

递归项在下一状态的价值预测与更长的 $\lambda$-回报之间插值。实现时从想象视界末端向前计算即可；末端仍需要价值网络提供 bootstrap。

通过计算整条想象轨迹上每个时间步的 $\lambda$-回报，我们就获得了一个兼顾偏差与方差的高质量评估基准，它将直接用于指导行动者和评论家的更新。

## 解析梯度：通过动力学模型反向传播

真实环境通常不能直接对动作求导，因此无模型 actor 常使用似然比策略梯度等估计器。它们不要求环境可微，但梯度方差可能较高。

Dreamer 的连续潜在动力学与重参数化动作可组成可微计算图，因此可使用论文所称的**解析梯度（Analytic Gradients）**更新 actor。这类梯度通常方差较低，但会继承世界模型偏差，并不等同于真实环境回报的精确梯度。

::: info 说明
以打台球为例：世界模型预测动作变化会怎样影响后续状态和回报。若动作采样与潜在动力学都采用可重参数化的连续分布，梯度就可以沿想象轨迹反向传到 actor。它提供的是学得模型内部的局部改进方向，而不是现实环境中的精确最优解。
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

自动微分可以从 $V^\lambda_\tau$ 沿奖励模型、转移模型和重参数化动作回到 actor 参数。这里得到的是对学到模型目标的路径梯度；训练 actor 时，世界模型和 critic 通常作为固定函数使用。

### 评论家（Critic）的损失函数

与此同时，评论家网络的作用是尽可能准确地预测状态价值，以帮助行动者计算回报。我们使用我们精心计算的 $\lambda$-回报作为它的回归目标（Target）。评论家的损失函数是预测值与 $\lambda$-回报目标之间的均方误差（MSE）：

$$\mathcal{L}_{\text{critic}}(\psi) = \mathbb{E} \left[ \sum_{\tau=t}^{t+H-1} \frac{1}{2} \big( v_{\psi}(s_\tau) - \text{stop\_gradient}(V^{\lambda}_{\tau}) \big)^2 \right]$$

<div align="center"><img src="/figures/04-latent-dynamics/latex/04-dreamer-v1/critic-stop-gradient-target.png" alt="评论家预测与停止梯度的 lambda 回报计算平方误差，梯度只返回评论家参数" width="86%">

_图 4.4-5：λ-return 在 critic 更新中充当固定回归目标；平方误差的梯度只返回 v_ψ，不能穿过停止梯度端改变目标。_

</div>

注意公式中的 `stop_gradient` 操作。这意味着在训练评论家时，我们将 $V^{\lambda}_{\tau}$ 视为一个固定的真实标签，不需要通过它再去反向传播梯度。我们只更新评论家自身的参数 $\psi$。

## 代码实现

下面，我们将通过一段精简的代码，演示在 PyTorch 中如何实现在潜空间中推演并计算 $\lambda$-回报这一 DreamerV1 的核心过程。

先定义简化世界模型、actor 与 critic。假设世界模型已训练完毕；代码省略 RSSM 的确定性/随机状态拆分、继续概率和动作边界变换。

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

接着展开潜在轨迹并递归计算 $\lambda$-回报。

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

    lambda_returns = torch.stack(lambda_returns) # (H, B, 1)，保留 actor 的路径梯度

    # 4. 计算 Actor 和 Critic 损失
    # 截取对应的时间步状态（0到H-1）
    states_to_evaluate = states_tensor[:-1]

    # 评论家损失：MSE (预测值, 目标值)
    pred_values = critic(states_to_evaluate.detach()) # 训练Critic时不对状态流进行反向传播
    critic_targets = lambda_returns.detach()
    critic_loss = torch.mean(0.5 * (pred_values - critic_targets)**2)

    # 行动者损失：最大化可微的想象 lambda 回报
    actor_loss = -torch.mean(lambda_returns)

    # 5. 反向传播更新
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    return actor_loss.item(), critic_loss.item()
```

同一组 $\lambda$-回报承担两个角色：作为 critic 标签时使用 `detach()`，避免目标随 critic 一起移动；作为 actor 目标时保留计算图，使梯度穿过想象状态、奖励与动作。示例只让 `actor_optimizer` 更新 actor，但仍会为固定世界模型和 critic 计算临时梯度；生产实现通常显式冻结它们的参数以节省内存。

## 总结

- **DreamerV1** 用参数化 actor 取代 PlaNet 部署时的 CEM，并在潜在想象轨迹上训练 actor–critic。
- **$\lambda$-回报**在较短 bootstrap 与较长模型滚动之间插值；它提供可调权衡，而非普遍最优的固定答案。
- 可微世界模型允许路径梯度穿过想象轨迹，但策略质量仍受模型覆盖范围与长期误差限制。
