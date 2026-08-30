# 世界模型与具身控制的闭环

具身智能（Embodied AI）的核心挑战在于，智能体必须在一个高度复杂、充满不确定性且代价高昂的物理世界中进行连续的感知与决策。传统的无模型强化学习（Model-Free RL）方法依赖于在真实环境中的海量试错，这在硬件磨损、安全风险以及数据采集效率等方面均存在不可逾越的瓶颈。

Ha 和 Schmidhuber 展示了从像素数据学习潜在动力学，并在模型生成的轨迹中训练控制器的可行性 [[Ha & Schmidhuber, 2018]](https://arxiv.org/abs/1803.10122)。RSSM 最初由 PlaNet 引入；Dreamer 在此基础上使用潜在想象轨迹训练 actor 与 critic [[Hafner et al., 2020]](https://arxiv.org/abs/1912.01603)。这些结果来自特定游戏与控制基准，构成了“视觉—模型—策略”闭环的实例，而不是对所有复杂具身任务的完整证明。

本节我们将深入探讨世界模型是如何在具身控制中形成闭环的。我们将从最基础的物理运动学出发，逐步推导出隐空间中的状态转移概率，最终在可微的“梦境”中实现策略的精确求解。

## 从真实物理世界到隐空间投影

在任何控制问题中，我们首先需要理解环境是如何随着时间演化的。假设我们在高中的物理课上研究一个在光滑水平面上做直线运动的滑块。

如果我们知道滑块在时刻 $t$ 的位置 $x_t$ 和速度 $v_t$，并对其施加一个恒定的加速度（控制量）$a_t$，那么经过一个极短的时间间隔 $\Delta t$ 后，滑块的新位置 $x_{t+1}$ 和新速度 $v_{t+1}$ 可以通过最基础的运动学公式精确计算：

$$
\begin{aligned}
v_{t+1} &= v_t + a_t \Delta t \\
x_{t+1} &= x_t + v_t \Delta t + \frac{1}{2} a_t \Delta t^2
\end{aligned}
$$

在这个简单的系统中，滑块的真实物理状态是由位置和速度完全描述的。我们可以将这两个标量组合成一个状态向量 $\mathbf{s}_t = [x_t, v_t]^\top$。此时，上述物理规律可以抽象为一个确定性的转移函数 $f$：

$$
\mathbf{s}_{t+1} = f(\mathbf{s}_t, \mathbf{a}_t)
$$

然而，在具身控制的真实场景（例如四足机器人或多自由度机械臂）中，系统的状态极其庞杂。更严峻的是，智能体通常无法直接获得完美的真实物理状态 $\mathbf{s}_t$。它所能依赖的，只有传感器（如RGB摄像头）传回的极其高维、充满噪声的观测数据（Observation）$\mathbf{o}_t$。

面对数以万计的像素，直接在像素空间中寻找类似该公式的转移函数是极其困难且低效的。因此，世界模型的第一个核心步骤是空间维度的降维。我们引入一个基于变分自编码器（VAE）思想的编码器模型 $E_\phi$，将高维的观测数据 $\mathbf{o}_t$ 压缩到一个低维且紧凑的隐状态（Latent State）空间中：

$$
\mathbf{z}_t \sim q_\phi(\mathbf{z}_t \mid \mathbf{o}_t)
$$

这里，$\mathbf{z}_t$ 就是智能体大脑中对当前世界局势的“抽象表征”。它不再是具体的像素，而是环境关键物理属性（如物体的相对位置、姿态及其不确定性）的低维概率分布。

## 循环状态空间模型（RSSM）：梦境的引擎

既然有了隐状态 $\mathbf{z}_t$，我们是否可以直接在隐空间中构建简单的马尔可夫转移函数 $\mathbf{z}_{t+1} = f(\mathbf{z}_t, \mathbf{a}_t)$ 呢？

答案是否定的。原因在于马尔可夫性（Markov Property）在单帧投影中遭到破坏。单帧图像的隐状态 $\mathbf{z}_t$ 无法提供速度、加速度等时序动态信息。此外，真实物理环境充满了由摩擦力扰动或传感器缺陷导致的随机性。

为了解决这个问题，循环状态空间模型（Recurrent State Space Model, RSSM）被提出。RSSM 将世界模型的状态一分为二：确定性隐状态（Deterministic Hidden State）$\mathbf{h}_t$ 和随机性隐状态（Stochastic Latent State）$\mathbf{z}_t$。

> 我们可以将这种确定性与随机性的解耦，视为“经典力学轨道”与“量子不确定性”的结合。在经典力学中，如果知道初始状态和受力情况，物体未来的轨迹是唯一确定的（这对应于模型中的确定性隐状态 $\mathbf{h}_t$，它像一个拥有记忆的循环核心，负责传递系统宏观上的历史演化规律）。然而，在复杂的真实环境中，微小的扰动和观测噪声使得具体的物理状态不可避免地呈现出概率分布的特性（这对应于随机状态 $\mathbf{z}_t$）。世界模型通过 $\mathbf{h}_t$ 描绘一条宏观的确定性演化轨道，并在每个时刻用 $\mathbf{z}_t$ 来捕捉无法被轨道完全决定的局部随机性。

RSSM 的内部动力学机制可以通过以下严谨的概率模型定义：

1. **确定性序列推断（Sequence Model）**：利用循环神经网络（如 GRU），基于过去的历史信息和动作，更新确定性状态：

   $$
   \mathbf{h}_t = f_\theta(\mathbf{h}_{t-1}, \mathbf{z}_{t-1}, \mathbf{a}_{t-1})
   $$

2. **先验动态模型（Dynamics Predictor）**：在没有接收当前真实物理观测的情况下，仅凭大脑中的历史记忆推测当前的随机状态：

   $$
   \hat{\mathbf{z}}_t \sim p_\theta(\hat{\mathbf{z}}_t \mid \mathbf{h}_t)
   $$

3. **后验表征模型（Representation Model）**：当接收到当前真实的视觉观测 $\mathbf{o}_t$ 后，结合历史记忆，得出对当前真实世界更精准的概率认知（用于修正内部模型和现实对齐）：
   $$
   \mathbf{z}_t \sim q_\phi(\mathbf{z}_t \mid \mathbf{h}_t, \mathbf{o}_t)
   $$

除此之外，具身控制还必须衡量行为的优劣，世界模型需要预测基于当前状态所能获得的奖励（Reward）：

$$
r_t \sim p_\theta(r_t \mid \mathbf{h}_t, \mathbf{z}_t)
$$

在世界模型的离线训练阶段，目标是使得未见观测的“先验预测”尽可能逼近包含观测事实的“后验认知”。因此，不仅要最小化图像重建和奖励预测的误差，还需要最小化先验分布和后验分布之间的 KL 散度（Kullback-Leibler Divergence）。

## 隐空间中的想象与解析梯度优化

一旦世界模型（引擎）收敛，智能体就可以切断与真实世界传感器的连接，闭上眼睛在“梦境”中学习如何决策。这是闭环的最后一块拼图。

在一个具体的时刻 $t$，智能体根据真实的视觉观测 $\mathbf{o}_t$ 计算出当前的后验状态 $\mathbf{z}_t$ 和历史特征 $\mathbf{h}_t$。从这个初始状态出发，智能体完全利用该公式和该公式在隐空间中向未来展开推演。

假设我们要优化的动作策略可以表示为一个概率分布 $\mathbf{a}_\tau \sim \pi_\psi(\mathbf{a}_\tau \mid \hat{\mathbf{z}}_\tau, \mathbf{h}_\tau)$，参数为 $\psi$。在梦境中，我们将时间轴从当前真实时刻 $\tau = t$ 展开到未来截断时刻 $\tau = t + H$。此时，智能体大脑中生成的虚拟轨迹（Trajectory）为：

$$
(\mathbf{h}_t, \mathbf{z}_t), \mathbf{a}_t, r_t, (\mathbf{h}_{t+1}, \hat{\mathbf{z}}_{t+1}), \mathbf{a}_{t+1}, r_{t+1}, \dots, (\mathbf{h}_{t+H}, \hat{\mathbf{z}}_{t+H})
$$

强化学习的终极目标是最大化累积回报。考虑到未来收益的衰减效应，我们引入极限级数求和中的折扣因子（Discount Factor）$\gamma \in (0, 1)$。该想象轨迹的价值函数（Value Function）被定义为对未来衰减奖励之和的期望估计：

$$
V(\mathbf{z}_\tau, \mathbf{h}_\tau) \approx \mathbb{E}_{p_\theta, \pi_\psi} \left[ \sum_{k=0}^{H-\tau} \gamma^k r_{\tau+k} + \gamma^{H-\tau+1} v_\xi(\hat{\mathbf{z}}_{t+H+1}, \mathbf{h}_{t+H+1}) \right]
$$

该公式中尾部的 $v_\xi$ 是一个独立训练的价值网络，用于在截断视野 $H$ 之外近似无穷远处的长尾收益。

传统无模型强化学习最深沉的痛点在于：真实的物理世界是一个极其复杂的黑盒系统。从执行动作 $\mathbf{a}_t$ 到环境反馈状态与奖励，这一客观物理演化过程在数学上是**不可导的**。这使得我们只能依赖计算代价昂贵且方差巨大的策略梯度（Policy Gradient）算法来摸索优化方向。

但在世界模型的控制闭环中，情况发生了彻底的数学质变。梦境中的环境，完全是由我们已经训练好的神经网络矩阵（动力学网络 $f_\theta$、先验推断 $p_\theta$、奖励映射 $p_\theta$）组合而成的。这意味着，动作流经整个“世界”的每一帧画面、每一笔奖励预测，本质上都是一系列标准的张量乘法和非线性激活函数，**这一切都是连续可微的！**

因此，我们不需要进行任何试错式的梯度近似，微积分中的多元函数链式法则（Chain Rule）能够直接穿透整个梦境时空，极其精确地计算出奖励目标的变动对策略参数 $\psi$ 的导数：

$$
\nabla_\psi \mathbb{E}[V] = \mathbb{E} \left[ \sum_{k=0}^{H} \gamma^k \nabla_{\mathbf{a}} r_{\tau+k} \cdot \nabla_\psi \pi_\psi(\mathbf{a}_{\tau+k}) \right]
$$

[**我们将这种求解范式称为解析梯度（Analytic Gradient）。在实现时，我们只需通过深度学习框架的自动微分引擎，反向传播穿越“梦境”（即世界模型的连续演化计算图），直接更新策略网络（Actor）的权重参数。**]

## 代码实现：在梦境计算图中推演闭环

下面，我们将通过框架代码构建 RSSM 的核心闭环过程。这段实现精炼地展示了如何维持确定性与随机性状态的双轨更新，并在计算图内展开梦境以实现完全可微的策略优化。

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.distributions import Normal

class RSSMCell(nn.Module):
    """循环状态空间模型的核心单元 (梦境引擎)"""
    def __init__(self, action_dim, hidden_dim=200, latent_dim=30):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 确定性状态更新网络 (GRU核心)
        # 输入维度: 动作空间 + 前一时刻随机隐状态
        self.gru = nn.GRUCell(action_dim + latent_dim, hidden_dim)

        # 先验网络 (Prior / Dynamics): p(z_t | h_t)
        self.prior_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        self.prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.prior_std = nn.Linear(hidden_dim, latent_dim)

        # 奖励预测网络
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward_prior(self, h_t):
        # [计算当前状态下向未来推演的概率分布参数]
        feat = self.prior_mlp(h_t)
        mean = self.prior_mean(feat)
        # 利用 softplus 函数强制标准差严格为正，并附加底噪避免坍缩
        std = nn.functional.softplus(self.prior_std(feat)) + 0.1
        return mean, std

    def step(self, action, h_prev, z_prev):
        # [步骤1: 计算宏观确定性隐状态 h_t]
        # action 维度: (batch, action_dim)
        # z_prev 维度: (batch, latent_dim)
        gru_input = torch.cat([action, z_prev], dim=-1)
        h_t = self.gru(gru_input, h_prev)

        # [步骤2: 利用先验网络预测微观随机状态 z_t]
        mean, std = self.forward_prior(h_t)

        # 采用重参数化技巧 (Reparameterization Trick) 采样 z_t
        # 保证在采样操作切断计算图后，依然能向 mean 和 std 回传梯度
        dist = Normal(mean, std)
        z_t = dist.rsample()

        # [步骤3: 预测智能体在当前世界格局下获得的奖赏]
        state_feature = torch.cat([h_t, z_t], dim=-1)
        reward_pred = self.reward_predictor(state_feature)

        return h_t, z_t, reward_pred


class Actor(nn.Module):
    """在梦境中输出控制信号的策略网络"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 200),
            nn.ELU(),
            nn.Linear(200, action_dim),
            nn.Tanh() # 强制动作空间约束在 [-1, 1] 物理有效范围内
        )

    def forward(self, state_feature):
        # 依赖完整的状态特征 (h_t 与 z_t 的拼接) 做出动作决策
        return self.net(state_feature)

# [初始化闭环组件]
batch_size = 4
action_dim = 2
hidden_dim = 200
latent_dim = 30
horizon = 15 # 定义梦境推演的时间视野长度 H

rssm = RSSMCell(action_dim, hidden_dim, latent_dim)
actor = Actor(hidden_dim + latent_dim, action_dim)
optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

# 假设我们在真实的具身环境中，通过视觉观测刚刚建立起对当前 t 时刻局势的认知
h_t = torch.zeros(batch_size, hidden_dim)
z_t = torch.zeros(batch_size, latent_dim)

# [开始闭合控制环：Latent Imagination Loop]
total_predicted_reward = 0

# 在循环内部，智能体切断传感器，完全靠神经网络在时间轴上向未来延展梦境
for t in range(horizon):
    state_feature = torch.cat([h_t, z_t], dim=-1)

    # 策略网络给出一个连续动作 (注意：此处前向传播维持了完整的微分轨迹)
    action = actor(state_feature)

    # 世界引擎承接动作，演化出下一步的时空状态和奖励
    h_t, z_t, reward = rssm.step(action, h_t, z_t)

    total_predicted_reward = total_predicted_reward + reward

# 计算策略目标，我们期望最大化梦境中的总期望奖励
# 由于需要执行梯度下降，这里直接取负值作为损失函数
loss = -total_predicted_reward.mean()

# 清空梯度
optimizer.zero_grad()
# 解析梯度可以直接通过整个循环的 15 步动态推演链条，精确定位到策略网络的参数上！
loss.backward()
optimizer.step()

print("梦境推演与策略反向传播完成！")
```

观察上述代码中 `loss.backward()` 这一震撼人心的调用。世界模型的绝对威力就在于：环境反馈的采样瓶颈被优雅地转化为张量的连乘运算。智能体的实体骨骼和马达并未在物理世界中移动分毫，它仅仅依靠神经网络中参数矩阵的激活与流转，就历经了动作执行后引发的未来时空演化，并以最高效的梯度下降法优化了自身的控制行为。

## 小结

- 具身控制的世界模型闭环完整涵盖了观测降维、序列动态预测以及隐空间内的极速策略规划三个步骤。
- 循环状态空间模型（RSSM）通过结构上解耦确定性历史状态传递与随机状态建模，完美契合了复杂物理系统的动态时序规律。
- 将环境转移严格建模为可微分的网络架构的最大红利在于，在策略求解阶段，算法脱离了方差巨大的采样估算，实现了微积分层面的“解析梯度”反向传播，极大提升了训练样本效率。
