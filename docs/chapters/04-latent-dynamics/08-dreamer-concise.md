# 4.8 Dreamer 智能体的简洁实现

强化学习中的一个核心问题，是怎样用较少的环境交互学会有效策略。无模型方法直接从经验学习策略或价值；基于模型的方法则先学习环境如何变化，再利用这个模型辅助决策。两类方法各有适用范围，Dreamer 属于后者。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/08-dreamer-concise/opening-dreamer-cup.gif" alt="Dreamer 控制杯子接住落球，展示潜在想象训练不仅产生运动，还能形成需要时序协调的目标行为。" width="86%">

_图 4.8-1：Dreamer 控制杯子接住落球，展示潜在想象训练不仅产生运动，还能形成需要时序协调的目标行为。 出处：Danijar Hafner；Timothy Lillicrap；Jimmy Ba；Mohammad Norouzi，[Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)（2020），Official behavior GIF: Cup Catch。_

</div>

Dreamer [[Hafner et al., 2020]](https://arxiv.org/abs/1912.01603) 建立在 PlaNet [[Hafner et al., 2019]](https://arxiv.org/abs/1811.04551) 的潜在动力学之上。它在学到的世界模型中生成潜在想象轨迹，并用这些轨迹训练 actor 与 critic；论文在视觉连续控制任务上报告了较高的数据效率。这里的 Dreamer 引用应指向 _Dream to Control_，不能误连到 PlaNet。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/08-dreamer-concise/planet-fig4.png" alt="PlaNet 的逐任务学习曲线给出 Dreamer 所继承潜在规划器在视觉控制基准上的前身表现。" width="86%">

_图 4.8-2：PlaNet 的逐任务学习曲线给出 Dreamer 所继承潜在规划器在视觉控制基准上的前身表现。 出处：Danijar Hafner；Timothy Lillicrap；Ian Fischer；Ruben Villegas；David Ha；Honglak Lee；James Davidson，[Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)（2019），Figure 4。_

</div>

本节先复习 RSSM 的计算顺序，再说明 Dreamer 如何在潜在想象轨迹上训练 actor 和 critic。代码只实现连续高斯 RSSM 的最小骨架，便于观察接口；它不是可直接复现实验结果的完整 Dreamer 智能体。

## 从物理递推到隐状态的动力学

先看一个熟悉的递推例子。忽略空气阻力且假设加速度恒定时，篮球的位置可以近似写成：

$$
p_{t+1} = p_t + v_t \Delta t + \frac{1}{2} a \Delta t^2
$$

这里，$(p_t, v_t)$ 是这个简化系统的状态。强化学习环境往往不给出这样恰到好处的低维状态，而只提供图像等高维观测 $o_t$。

因此，Dreamer 试图寻找一个从高维观测 $o_t$ 到低维隐状态（Latent State）$s_t$ 的映射，并在隐状态空间中建立类似于该公式的转移规律。

环境可能包含随机性、部分可观测性和模型尚未学会的因素。为表达这些不确定性，RSSM 不只输出一个随机状态点，而是参数化它的概率分布。确定性模型并非一定会“坍塌”，但单点预测不足以描述多种可能的未来。

## 循环状态空间模型 (RSSM) 的数学推导

Dreamer 的核心世界模型是 RSSM，它把确定性的循环记忆与随机状态推断结合起来。

首先，我们将隐状态拆分为两部分：一个确定性的状态（Deterministic State）$h_t$，和一个随机的状态（Stochastic State）$z_t$。

确定性状态 $h_t$ 保存对历史的有限维摘要，并由 GRU 更新：

$$
h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})
$$

其中，$a_{t-1}$ 是上一时刻的动作。这个递推式描述的是模型学到的潜在动力学，不保证等同于环境的真实物理定律。

接下来，我们需要确定随机状态 $z_t$。在模型训练阶段（我们称之为后验推断），智能体不仅能“回想”起历史 $h_t$，还能“看”到真实的当前观测 $o_t$。因此，后验概率分布 $q_\phi(z_t \mid h_t, o_t)$ 结合了对过去的记忆和当前的现实。

$$
z_t \sim q_\phi(z_t \mid h_t, o_t)
$$

然而，在智能体做“梦”（即规划未来）的时候，它是无法看到未来的真实观测 $o_{t}$ 的。它只能完全依赖内部模型去猜测。这就引出了先验概率分布（Prior Distribution）$p_\theta(z_t \mid h_t)$：

$$
\hat{z}_t \sim p_\theta(z_t \mid h_t)
$$

::: info 说明
想象你在一条熟悉的但没有路灯的走廊里蒙眼行走。确定性状态 $h_t$ 就像是你大脑中对之前走过步数的记忆和方向感；先验分布 $p_\theta(z_t \mid h_t)$ 是你根据记忆预测自己当前可能所处的位置分布。而当你稍微睁开一点眼睛，看到周围微弱的轮廓 $o_t$ 时，你结合记忆和所见，得出的更加确信的位置分布，就是后验分布 $q_\phi(z_t \mid h_t, o_t)$。Dreamer 训练世界模型的目标之一，就是让闭着眼睛的预测（先验）尽可能接近睁开眼睛的认知（后验）。
:::

### 变分下界 (ELBO) 与损失函数

为了训练包含隐变量的生成模型，我们希望最大化观测数据的对数似然 $\log p(o_{1:T} \mid a_{1:T})$。由于对隐变量积分通常没有可用的解析解，我们引入变分分布 $q_\phi$，构造证据下界（Evidence Lower Bound，ELBO）。

考虑单步的生成过程，我们要重构当前的观测 $o_t$ 和奖励 $r_t$。世界模型的总损失函数 $\mathcal{L}_{model}$ 由三个部分组成：

<div align="center">
  <img src="/figures/04-latent-dynamics/source/08-dreamer-concise/dreamerv3-fig3.png" alt="DreamerV3 的训练总览把世界模型、想象 actor–critic 与真实数据回放连接起来，展示简洁实现所抽取主路径的后续版本。" width="86%">

_图 4.8-3：DreamerV3 的训练总览把世界模型、想象 actor–critic 与真实数据回放连接起来，展示简洁实现所抽取主路径的后续版本。 出处：Danijar Hafner；Jurgis Pasukonis；Jimmy Ba；Timothy Lillicrap，[Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)（2023），Figure 3。_

</div>

1. **重构损失 (Reconstruction Loss)**：从隐状态 $(h_t, z_t)$ 预测当前观测。
2. **奖励预测损失 (Reward Prediction Loss)**：智能体需要预测当前状态下能获得的奖励。
3. **动态 KL 散度损失 (Dynamics KL Loss)**：如前所述，我们要拉近先验分布和后验分布的距离。

我们给出标量形式下最基本的 KL 散度定义：

$$
D_{\text{KL}}(q \parallel p) = \sum q(x) \log \frac{q(x)}{p(x)}
$$

把它用于序列潜变量后，可以写出下面这个简化的单步负 ELBO；实际 Dreamer 还会根据版本加入继续概率、KL 平衡等项：

$$
\mathcal{L}_t = \underbrace{-\ln p_\theta(o_t \mid h_t, z_t)}_{\text{观测重构}} + \underbrace{-\ln p_\theta(r_t \mid h_t, z_t)}_{\text{奖励预测}} + \underbrace{\beta D_{\text{KL}}\left( q_\phi(z_t \mid h_t, o_t) \parallel p_\theta(z_t \mid h_t) \right)}_{\text{动力学约束}}
$$

该公式中的三个项分别对应了图像解码器、奖励预测器和概率推断模块的优化目标。

## 隐空间中的策略优化

世界模型经过更新后，Dreamer 从经验回放中推断出的状态出发，在潜在空间生成想象轨迹，并用它们更新策略。Actor-Critic 的输入是拼接状态 $s_t=[h_t,z_t]$，而不是原始像素。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/08-dreamer-concise/dreamer-fig10.png" alt="Dreamer 在连续控制任务中比较解析梯度、强化学习估计与在线规划，说明 actor 学习替代部署时搜索的实验依据。" width="86%">

_图 4.8-4：Dreamer 在连续控制任务中比较解析梯度、强化学习估计与在线规划，说明 actor 学习替代部署时搜索的实验依据。 出处：Danijar Hafner；Timothy Lillicrap；Jimmy Ba；Mohammad Norouzi，[Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)（2020），Figure 10。_

</div>

每次策略更新从经验回放对应的后验状态 $s_\tau$ 出发，随后暂时不调用真实环境，而是用先验 $p_\theta(z_t\mid h_t)$ 和策略 $\pi_\psi(a_t\mid s_t)$ 向前展开 $H$ 步。这段轨迹是模型预测，不是环境中的真实轨迹。

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

先用重参数化技巧实现一个对角高斯输出层，使梯度能够经过采样表达式传播。

<div align="center"><img src="/figures/04-latent-dynamics/latex/08-dreamer-concise/gaussian-stat-split-reparam.png" alt="网络输出拆分为均值与尺度参数，尺度经 softplus 变成正标准差，再与标准噪声重参数化得到样本" width="86%">

_图 4.8-5：高斯头把 2D_z 个输出等分为均值与尺度参数；正值变换稳定标准差，随机性由独立噪声承担，因此样本仍能对网络参数求导。本文根据上述实现绘制。_

</div>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReparameterizedGaussian(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        mean_scale = self.fc(x)
        # 将输出分为均值和未经约束的尺度参数
        mean, raw_scale = torch.chunk(mean_scale, 2, dim=-1)
        # 限制标准差的范围以保证数值稳定性
        std = F.softplus(raw_scale) + 0.1
        # 重参数化采样
        noise = torch.randn_like(mean)
        sample = mean + noise * std
        return mean, std, sample
```

接下来实现 RSSM 的核心类，其中包含确定性路径（GRU）和先验、后验两条随机推断路径。

```python
class RSSM(nn.Module):
    def __init__(self, action_dim, obs_embed_dim, deter_dim, stoch_dim):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        # 确定性状态更新，由前一个随机状态和动作输入
        self.cell = nn.GRUCell(stoch_dim + action_dim, deter_dim)

        # 先验网络：基于确定性记忆，预测接下来的随机状态
        self.prior = ReparameterizedGaussian(deter_dim, stoch_dim)

        # 后验网络：结合观测特征和确定性记忆，近似推断随机状态
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

训练世界模型时，需要沿时间展开这个网络：后验样本用于重构观测和预测奖励，先验与后验之间的 KL 则约束没有观测时的预测。训练策略时，`forward_prior` 负责生成想象中的后续状态。完整实现还需要编码器、解码器、奖励与继续概率模型、actor、critic、$\lambda$-return、经验回放以及各优化器；这里的代码只展示 RSSM 接口。

## 小结

- **基于模型的强化学习**利用学到的动力学提高经验利用率，但仍需要真实交互收集和更新数据。
- **RSSM** 用确定性状态摘要历史，用随机状态表示当前潜变量及其不确定性。
- **Dreamer** 用变分目标训练世界模型，再在潜在想象轨迹上计算 $\lambda$-return，并更新 actor 与 critic。
