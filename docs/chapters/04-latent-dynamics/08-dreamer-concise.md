# Dreamer 智能体的简洁实现

在深度强化学习的发展历程中，智能体如何高效地从环境中获取经验一直是一个核心难题。早期的无模型（Model-Free）强化学习算法，虽然在特定任务上取得了超越人类的表现，但往往需要数以千万计的交互步数。这就好比一个完全不具备物理直觉的婴儿，必须通过无数次摔倒才能学会走路。为了打破这种对环境交互的极度依赖，研究者们将目光转向了基于模型的强化学习（Model-Based Reinforcement Learning）。

Dreamer [[Hafner et al., 2020]](https://arxiv.org/abs/1912.01603) 建立在 PlaNet [[Hafner et al., 2019]](https://arxiv.org/abs/1811.04551) 的潜在动力学之上。它在学到的世界模型中生成潜在想象轨迹，并用这些轨迹训练 actor 与 critic；论文在视觉连续控制任务上报告了较高的数据效率。这里的 Dreamer 引用应指向 _Dream to Control_，不能误连到 PlaNet。

在本节中，我们将剥开 Dreamer 复杂的工程外衣，从最基础的物理运动规律出发，一步步严谨地推导出循环状态空间模型（Recurrent State Space Model, RSSM）的数学本质，并最终给出 Dreamer 智能体的核心简洁实现。

## 从物理递推到隐状态的动力学

在我们探讨复杂的深度学习模型之前，让我们先回到高中物理中最经典的抛体运动。假设我们要追踪一个在空中飞行的篮球，如果已知它在 $t$ 时刻的位置 $p_t$ 和速度 $v_t$，我们可以利用运动学公式极其精确地预测它在 $t+1$ 时刻的状态：

$$
p_{t+1} = p_t + v_t \Delta t + \frac{1}{2} a \Delta t^2
$$

在这里，$(p_t, v_t)$ 构成了这个物理系统的“状态”（State）。只要我们掌握了状态以及物理规律（转移函数），我们就能预测未来。然而，在强化学习面临的高维环境（例如连续的像素输入）中，我们无法直接获取这种完美的低维状态，只能得到包含大量冗余信息的观测值（Observation）$o_t$。

因此，Dreamer 试图寻找一个从高维观测 $o_t$ 到低维隐状态（Latent State）$s_t$ 的映射，并在隐状态空间中建立类似于该公式的转移规律。

但这还不够。真实世界充满了不确定性：一阵突如其来的风可能会改变篮球的轨迹。如果我们仅仅使用一个确定性的函数来预测未来，随着时间的推移，微小的误差将会迅速累积，导致“梦境”坍塌。为了解决这个问题，我们需要引入概率模型。我们不再预测一个绝对的 $s_{t+1}$，而是预测 $s_{t+1}$ 的概率分布。

## 循环状态空间模型 (RSSM) 的数学推导

Dreamer 的核心世界模型被称为循环状态空间模型（RSSM）。RSSM 巧妙地将确定性的循环神经网络（RNN）和随机的状态推断结合在一起。

首先，我们将隐状态拆分为两部分：一个确定性的状态（Deterministic State）$h_t$，和一个随机的状态（Stochastic State）$z_t$。

确定性状态 $h_t$ 负责捕捉长期的历史记忆，它通过一个标准的 GRU（门控循环单元）进行更新：

$$
h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})
$$

在这里，$a_{t-1}$ 是智能体在上一时刻采取的动作。该公式描述了系统底层的物理惯性。

接下来，我们需要确定随机状态 $z_t$。在模型训练阶段（我们称之为后验推断），智能体不仅能“回想”起历史 $h_t$，还能“看”到真实的当前观测 $o_t$。因此，后验概率分布 $q_\phi(z_t \mid h_t, o_t)$ 结合了对过去的记忆和当前的现实。

$$
z_t \sim q_\phi(z_t \mid h_t, o_t)
$$

然而，在智能体做“梦”（即规划未来）的时候，它是无法看到未来的真实观测 $o_{t}$ 的。它只能完全依赖内部模型去猜测。这就引出了先验概率分布（Prior Distribution）$p_\theta(z_t \mid h_t)$：

$$
\hat{z}_t \sim p_\theta(z_t \mid h_t)
$$

> [!NOTE]
> 想象你在一条熟悉的但没有路灯的走廊里蒙眼行走。确定性状态 $h_t$ 就像是你大脑中对之前走过步数的记忆和方向感；先验分布 $p_\theta(z_t \mid h_t)$ 是你根据记忆预测自己当前可能所处的位置分布。而当你稍微睁开一点眼睛，看到周围微弱的轮廓 $o_t$ 时，你结合记忆和所见，得出的更加确信的位置分布，就是后验分布 $q_\phi(z_t \mid h_t, o_t)$。Dreamer 训练世界模型的目标之一，就是让闭着眼睛的预测（先验）尽可能接近睁开眼睛的认知（后验）。

### 变分下界 (ELBO) 与损失函数

为了训练这样一个包含隐变量的生成模型，我们需要最大化观测数据的对数似然 $\log p(o_{1:T} \mid a_{1:T})$。由于直接计算含有隐变量的积分是极其困难的（难以进行解析计算），我们通过引入变分分布 $q_\phi$ 来构造变分下界（Evidence Lower BOund, ELBO）。

考虑单步的生成过程，我们要重构当前的观测 $o_t$ 和奖励 $r_t$。世界模型的总损失函数 $\mathcal{L}_{model}$ 由三个部分组成：

1. **重构损失 (Reconstruction Loss)**：智能体必须能够从隐状态 $(h_t, z_t)$ 中还原出真实的观测。
2. **奖励预测损失 (Reward Prediction Loss)**：智能体需要预测当前状态下能获得的奖励。
3. **动态 KL 散度损失 (Dynamics KL Loss)**：如前所述，我们要拉近先验分布和后验分布的距离。

我们给出标量形式下最基本的 KL 散度定义：

$$
D_{\text{KL}}(q \parallel p) = \sum q(x) \log \frac{q(x)}{p(x)}
$$

将其推广到我们的序列多维隐状态，并结合重构项，我们可以写出 RSSM 在时间步 $t$ 的严谨目标函数（我们要最小化它的负值）：

$$
\mathcal{L}_t = \underbrace{-\ln p_\theta(o_t \mid h_t, z_t)}_{\text{观测重构}} \underbrace{-\ln p_\theta(r_t \mid h_t, z_t)}_{\text{奖励预测}} + \underbrace{\beta D_{\text{KL}}\left( q_\phi(z_t \mid h_t, o_t) \parallel p_\theta(z_t \mid h_t) \right)}_{\text{动力学约束}}
$$

该公式中的三个项分别对应了图像解码器、奖励预测器和概率推断模块的优化目标。

## 隐空间中的策略优化

一旦世界模型（RSSM）训练收敛，我们就可以利用它来学习策略。与传统方法不同，Dreamer 的 Actor-Critic 网络的输入不再是高维的像素，而是低维的拼接隐状态 $s_t = [h_t, z_t]$。

在每一次策略更新时，我们从历史经验池中抽取一个真实的隐状态 $s_\tau$ 作为起点。然后，**完全不与真实环境交互**，我们使用先验网络 $p_\theta(z_t \mid h_t)$ 和策略网络 $\pi_\psi(a_t \mid s_t)$ 在隐空间中向前展开 $H$ 步，生成一段由“梦境”组成的轨迹。

为了优化动作，我们需要评估每个隐状态的价值。设价值网络为 $v_\xi(s_t)$。Dreamer 使用了指数加权的回报估计（类似于 $\text{TD}(\lambda)$）：

$$
V_t^\lambda = r_t + \gamma \left( (1 - \lambda) v_\xi(s_{t+1}) + \lambda V_{t+1}^\lambda \right)
$$

该公式是一个后向递推公式，它结合了单步的奖励预测、对下一状态的价值估计，以及对更长远未来的展望。

Actor 网络 $\pi_\psi$ 的目标是最大化预期的价值：

$$
\mathcal{L}_{\text{actor}} = -\sum_{t=\tau}^{\tau+H} \mathbb{E}[V_t^\lambda]
$$

Critic 网络 $v_\xi$ 的目标是使得其预测的价值逼近 $V_t^\lambda$：

$$
\mathcal{L}_{\text{critic}} = \frac{1}{2} \sum_{t=\tau}^{\tau+H} \left( v_\xi(s_t) - V_t^\lambda \right)^2
$$

## 极简代码实现

接下来，我们将使用纯粹的张量操作，来呈现 Dreamer 核心组件的简洁实现。我们首先定义多层感知机（MLP）作为基础的构建块。

(**我们定义基础的重参数化技巧，以使得随机采样过程可导。**)

```{.python .input}
#@tab pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReparameterizedGaussian(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        mean_std = self.fc(x)
        # 将输出分为均值和对数标准差
        mean, log_std = torch.chunk(mean_std, 2, dim=-1)
        # 限制标准差的范围以保证数值稳定性
        std = F.softplus(log_std) + 0.1
        # 重参数化采样
        noise = torch.randn_like(mean)
        sample = mean + noise * std
        return mean, std, sample
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
from tensorflow.keras import layers

class ReparameterizedGaussian(tf.keras.Model):
    def __init__(self, output_dim):
        super().__init__()
        self.fc = layers.Dense(output_dim * 2)

    def call(self, x):
        mean_std = self.fc(x)
        mean, log_std = tf.split(mean_std, 2, axis=-1)
        std = tf.math.softplus(log_std) + 0.1
        noise = tf.random.normal(tf.shape(mean))
        sample = mean + noise * std
        return mean, std, sample
```

(**接下来实现循环状态空间模型（RSSM）的核心类。**) 这包含了确定性路径（GRU）和两条随机推断路径（先验与后验）。

```{.python .input}
#@tab pytorch
class RSSM(nn.Module):
    def __init__(self, action_dim, obs_embed_dim, deter_dim, stoch_dim):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        # 确定性状态更新，由前一个随机状态和动作输入
        self.cell = nn.GRUCell(stoch_dim + action_dim, deter_dim)

        # 先验网络：基于确定性记忆，预测接下来的随机状态
        self.prior = ReparameterizedGaussian(deter_dim, stoch_dim)

        # 后验网络：结合观测特征和确定性记忆，推断准确的随机状态
        self.posterior = ReparameterizedGaussian(deter_dim + obs_embed_dim, stoch_dim)

    def forward_prior(self, prev_stoch, prev_action, prev_deter):
        # 拼接隐状态与动作
        x = torch.cat([prev_stoch, prev_action], dim=-1)
        # 更新确定性状态 h_t
        deter = self.cell(x, prev_deter)
        # 计算先验分布 p(z_t | h_t)
        mean, std, stoch = self.prior(deter)
        return deter, stoch, mean, std

    def forward_posterior(self, deter, obs_embed):
        # 结合观测特征计算后验分布 q(z_t | h_t, o_t)
        x = torch.cat([deter, obs_embed], dim=-1)
        mean, std, stoch = self.posterior(x)
        return stoch, mean, std
```

```{.python .input}
#@tab tensorflow
class RSSM(tf.keras.Model):
    def __init__(self, action_dim, obs_embed_dim, deter_dim, stoch_dim):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        self.cell = layers.GRUCell(deter_dim)
        self.prior = ReparameterizedGaussian(stoch_dim)
        self.posterior = ReparameterizedGaussian(stoch_dim)

    def forward_prior(self, prev_stoch, prev_action, prev_deter):
        x = tf.concat([prev_stoch, prev_action], axis=-1)
        deter, _ = self.cell(x, [prev_deter])
        mean, std, stoch = self.prior(deter)
        return deter, stoch, mean, std

    def forward_posterior(self, deter, obs_embed):
        x = tf.concat([deter, obs_embed], axis=-1)
        mean, std, stoch = self.posterior(x)
        return stoch, mean, std
```

在训练世界模型时，我们需要通过时间步展开这个网络，并分别利用后验网络生成表示（用于重构观测），以及利用先验网络计算 KL 散度以约束模型预测未来的能力。策略网络的训练则仅依赖于 `forward_prior` 在隐空间中向前做梦推演。通过这些模块的协同，Dreamer 在无需直接操作真实环境物理状态的前提下，仅凭想象便掌握了复杂的控制任务。

## 小结

- 基于模型的强化学习通过构建世界模型（如 RSSM）来摆脱对高频环境交互的依赖。
- RSSM 分离了确定性状态与随机状态，完美契合了记忆与未来不确定性的物理和概率特性。
- Dreamer 通过在隐空间中计算变分下界（ELBO）来联合训练编码器、解码器和动态模型，并在“梦境”内完成了类似于 $\lambda$-return 的价值估计和策略梯度更新。
