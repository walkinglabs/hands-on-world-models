# 基于世界模型想象的强化学习

在深度强化学习的发展历程中，样本效率（Sample Efficiency）始终是一个难以逾越的瓶颈。传统的无模型（Model-Free）强化学习算法，例如深度Q网络（DQN）或近端策略优化（PPO），通常需要与环境进行数百万次甚至上亿次的交互，才能学习到有效的策略。在电子游戏中这种试错成本尚可接受，但在现实世界的机器人控制、自动驾驶等领域，高昂的试错代价使得传统无模型方法举步维艰。

为了解决这一问题，研究者们将目光投向了内部模拟机制。在面对新任务时，智能体不需要在现实中进行无数次试错，而是能够在内部表示中“想象”不同动作可能带来的后果，并在想象中规划最优路径。这种在内部认知模型中进行模拟与策略优化的思想，催生了基于世界模型想象的强化学习（Imagination-based Reinforcement Learning）。本节我们将深入探讨这一前沿方向，从其学术渊源出发，严谨地推导隐空间动力学模型的数学本质，并最终实现一个能够在“梦境”中学习的智能体。

## 学术溯源与理论动机

基于模型（Model-Based）的强化学习思想可以追溯到上世纪90年代Richard Sutton提出的Dyna架构 [[Sutton, 1990]](https://dl.acm.org/doi/10.5555/645530.658292)。Dyna的核心思想是利用智能体与真实环境交互收集的数据来训练一个环境动力学模型，随后智能体不仅在真实环境中学习，也在环境模型生成的模拟数据中学习。然而，早期的环境模型主要针对低维、离散的状态空间，难以处理高维的图像输入。

Ha 与 Schmidhuber 在《World Models》中用变分自编码器（VAE）压缩图像，再用混合密度循环网络（MDN-RNN）学习动作条件潜在动力学 [[Ha & Schmidhuber, 2018]](https://arxiv.org/abs/1803.10122)。CarRacing 的控制器在真实环境中优化；完全在模型“梦境”中训练并迁回真实环境的实验对应 VizDoom Take Cover。这个结果证明了特定任务上的可行性，不是对任意视觉控制任务的保证。

Hafner 等人在 PlaNet 中提出 RSSM，Dreamer 随后用这一模型生成潜在想象轨迹，并通过价值估计与动力学梯度训练策略 [[Hafner et al., 2020]](https://arxiv.org/abs/1912.01603)。DreamerV2 与 DreamerV3 又扩展到离散潜变量和跨领域统一超参数 [[Hafner et al., 2021]](https://arxiv.org/abs/2010.02193); [[Hafner et al., 2023]](https://arxiv.org/abs/2301.04104)。这些论文在各自报告的基准上与强无模型方法比较，但不能概括为在样本效率和最终性能上“首次全面超越”所有无模型算法。

## 动力学系统与隐空间映射

在正式引入复杂的变分推断之前，我们先从马尔可夫决策过程（MDP）的最基础动力学转移出发。

在一个确定性的动力学系统中，如果已知当前时刻 $t$ 的状态向量 $s_t$ 以及智能体执行的动作向量 $a_t$，系统的下一时刻状态 $s_{t+1}$ 可以由一个确定性的非线性函数精确映射得出：

$$s_{t+1} = f(s_t, a_t)$$

然而，真实世界往往充满了不可观测的系统噪声和随机扰动。因此，必须将确定性的映射矩阵升级为概率分布映射形式。给定当前状态 $s_t$ 和动作 $a_t$，下一状态 $s_{t+1}$ 服从一个状态转移概率分布：

$$s_{t+1} \sim p(s_{t+1} \mid s_t, a_t)$$

在传统基于模型的方法中，如果我们能够参数化并学习到这个概率分布 $p$，我们就可以利用它来展开多步预测。但是，当观测空间是高维张量（例如 $64 \times 64 \times 3$ 的RGB图像）时，直接在高维连续流形上拟合条件概率极具挑战，像素级别的微小预测误差会在多步自回归计算中呈指数级发散。

为了应对维数灾难，世界模型引入了隐空间映射（Latent Space Mapping）机制。我们将高维的观测数据 $o_t$ 投影到一个低维紧致的隐状态向量 $z_t \in \mathbb{R}^d$ 中。动力学的时序演化不再在高维像素张量空间中进行，而是完全转移到这个低维黎曼流形中：

$$z_{t+1} \sim p(z_{t+1} \mid z_t, a_t)$$

只要隐状态序列分布充分近似了推断当前奖励 $r_t$ 和重构原始观测 $o_t$ 的充分统计量，智能体就能够在极低计算开销下执行大规模的并行轨迹采样。

## 循环状态空间模型（RSSM）的严谨推导

隐空间连续动力学建模的难点在于：如何解耦环境演化的确定性动态结构与不可预测的随机过程？Hafner等人提出的RSSM通过确定性循环单元和随机变分后验的联合建模提供了严格的数学闭环。

RSSM将潜状态隐式解构为正交的两部分：一个确定性的非马尔可夫隐藏状态 $h_t$（由带有门控机制的循环神经网络递归维护）和一个马尔可夫随机后验状态 $z_t$。其联合生成分布严格服从以下因子分解过程。

给定初始条件和一系列动作序列 $a_{1:T}$，观测序列 $o_{1:T}$ 与奖励序列 $r_{1:T}$ 及对应隐序列 $z_{1:T}$ 的联合似然模型为：

$$p(o_{1:T}, r_{1:T}, z_{1:T} \mid a_{1:T}) = \prod_{t=1}^T p(o_t \mid h_t, z_t) p(r_t \mid h_t, z_t) p(z_t \mid h_t)$$

其中，确定性隐藏状态轨迹 $h_t$ 根据如下非线性递归方程更新：

$$h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})$$

在该公式的连乘项中，定义了系统核心的基础测度：

1. **先验转移模型（Prior Dynamics）**：$p_\theta(z_t \mid h_t)$，负责在缺少当前观测信息的情况下，沿着时间轴预测随机变量分布的演化。
2. **观测重构模型（Observation Model）**：$p_\theta(o_t \mid h_t, z_t)$，定义从低维隐流形到高维观测流形的映射。
3. **奖励反馈模型（Reward Model）**：$p_\theta(r_t \mid h_t, z_t)$，提供单步标量反馈预测。

在优化阶段，自然的目标是最大化真实观测与奖励的边缘对数似然 $\ln p(o_{1:T}, r_{1:T} \mid a_{1:T})$。由于多重高维随机变量的直接积分 $\int z_{1:T} dz$ 在计算上是不可解的，我们必须引入变分推断（Variational Inference）框架。构建一个能够访问全局信息的近似后验推断分布，我们要求其不仅依赖于循环历史信息，同时显式以当前真实观测 $o_t$ 为条件：

$$q_\phi(z_t \mid h_t, o_t)$$

> [!NOTE]
> 对于变分下界推导这一反直觉的高阶过程，我们可以借助一个精炼的类比：优化真实边缘似然如同试图精确重构一个漫长演化的远古生态系统，几乎不可能；而变分推断中的后验模型 $q$ 就像一位找到了部分当代化石残片（真实观测 $o$）的考古学家。通过最大化证据下界（ELBO），考古学家迫使自己不看化石做出的预测（先验 $p$）与看着化石做出的分析（后验 $q$）逐步对齐，从而在不需要穷举所有可能性的情况下，逼近那段未知的演化动力学真相。

运用詹森不等式（Jensen's Inequality），将对数函数的凸组合极值向内投射，我们获得时间序列联合分布的变分证据下界（ELBO）：

$$
\begin{aligned}
\ln p(o_{1:T}, r_{1:T} \mid a_{1:T}) &= \ln \mathbb{E}_{q_\phi(z_{1:T} \mid o_{1:T}, a_{1:T})} \left[ \frac{p(o_{1:T}, r_{1:T}, z_{1:T} \mid a_{1:T})}{q_\phi(z_{1:T} \mid o_{1:T}, a_{1:T})} \right] \\
&\ge \mathbb{E}_{q_\phi} \left[ \ln \frac{p(o_{1:T}, r_{1:T}, z_{1:T} \mid a_{1:T})}{q_\phi(z_{1:T} \mid o_{1:T}, a_{1:T})} \right] \\
&= \sum_{t=1}^T \mathbb{E}_{q_\phi} \Big[ \underbrace{\ln p_\theta(o_t \mid h_t, z_t)}_{\text{重构对数似然}} + \underbrace{\ln p_\theta(r_t \mid h_t, z_t)}_{\text{反馈对数似然}} - \underbrace{\text{KL}\big( q_\phi(z_t \mid h_t, o_t) \,\|\, p_\theta(z_t \mid h_t) \big)}_{\text{动力学分布散度}} \Big]
\end{aligned}
$$

该公式从信息论和统计力学双重维度锁定了RSSM的优化流形：

1. **重构对数似然**强制隐空间 $z_t$ 必须保留足以映射回原始输入度规的信息容量。
2. **反馈对数似然**确保表示空间严格包含与长期回报最大化直接相关的价值表征。
3. **KL散度项**（Kullback-Leibler Divergence）是一个强正则化算子，它约束开环预测的先验动力学必须紧紧跟随具有闭环观测修正的后验动力学。只有散度收敛，智能体在未来断开真实环境、进行纯粹“做梦”时的序列展开分布才具有拓扑一致性。

## 在隐空间中展开梦境与策略寻优

当动力学模型（参数为 $\theta$ 和 $\phi$）拟合收敛后，环境的真实动态规律已被提纯至连续函数的参数中。我们随即进入策略优化的“做梦”阶段（Dreaming Phase）。

在此阶段，智能体与物理环境绝对隔离。初始化过程从经验回放池中随机无偏采样一个后验隐状态 $z_\tau$。随后，策略网络（Actor）基于此纯代数状态空间输出动作，先验转移模型则代替物理规律，自回归地计算未来的时空演化轨迹：

$$a_t \sim \pi_\psi(a_t \mid \hat{s}_t), \quad \hat{z}_{t+1} \sim p_\theta(\hat{z}_{t+1} \mid \hat{h}_{t+1}), \quad \text{其中 } \hat{h}_{t+1} = f_\theta(\hat{h}_t, \hat{z}_t, a_t)$$

约定俗成地，采用上标 $\hat{\cdot}$ 标注该张量系在纯计算图内演化所得。为符号紧凑起见，令 $\hat{s}_t = (\hat{h}_t, \hat{z}_t)$。

在获取了基于全微分计算图的虚拟轨迹后，目标是针对策略参数 $\psi$ 进行无偏梯度下降。在标准Actor-Critic无模型范式中，由于状态转移算子属于不可导的物理/黑盒黑箱过程，仅能使用高方差的似然比率梯度（REINFORCE）估算。而在世界模型下，每一条链路皆为平滑的非线性激活函数复合。

假定隐状态 $z$ 被配置为对角协方差多元高斯分布，借由位置尺度族分布特有的重参数化技巧（Reparameterization Trick），可将其拆解为确定性仿射变换与标准正态白噪声 $\epsilon$ 的阿达马乘积：$\hat{z}_{t+1} = \mu_\theta(\hat{h}_{t+1}) + \sigma_\theta(\hat{h}_{t+1}) \odot \epsilon$。这赋予了状态对动作的直接偏导通路，任何标量目标泛函 $V(\hat{s}_t)$ 关于策略参数 $\psi$ 的解析梯度皆可通过反向传播时间链（BPTT）严格求出：

$$\frac{\partial V(\hat{s}_t)}{\partial \psi} = \sum_{\tau=t}^{t+H} \gamma^{\tau-t} \frac{\partial \mathbb{E}[r_\tau]}{\partial \hat{s}_\tau} \frac{\partial \hat{s}_\tau}{\partial \psi}$$

低方差解析梯度的大量应用，奠定了Imagination RL惊人收敛速度的基础。为消除有限视野截断带来的偏差，体系内会同步维持一个参数为 $\xi$ 的价值基线网络（Critic）$v_\xi(\hat{s}_t)$，预测截断层级以外的期望回报。多步展开的目标价值算子可通过 $\lambda$-return 公式进行带指数衰减权重的修正累积：

$$V_t^\lambda = \hat{r}_t + \gamma \begin{cases} (1-\lambda) v_\xi(\hat{s}_{t+1}) + \lambda V_{t+1}^\lambda, & t < H \\ v_\xi(\hat{s}_{t+1}), & t = H \end{cases}$$

最终的交替梯度下降法则退化为：Actor $\pi_\psi$ 朝着扩大 $V_t^\lambda$ 解析梯度的方向迭代，Critic $v_\xi$ 则极小化其输出预测与自举目标 $V_t^\lambda$ 间的均方勒贝格范数。

## 代码实现与深度剖析

我们将使用深度学习张量框架严格还原前述算子的数学结构。首先实现处于核心枢纽地位的 `RSSMCell` 类。该类内部封装了递归动态方程、分布参数映射以及前后验散度计算逻辑。

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.distributions import Normal

class RSSMCell(nn.Module):
    def __init__(self, action_dim, hidden_dim=200, latent_dim=30):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 确定性动力学方程: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        self.gru = nn.GRUCell(hidden_dim + latent_dim + action_dim, hidden_dim)

        # 先验预测算子: p(z_t | h_t)
        self.prior_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )

        # 后验推断算子: q(z_t | h_t, o_t) (约定o_t已被视觉流形编码器映射为向量e_t)
        self.posterior_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )

    def _split_dist(self, params):
        # [解构对角高斯分布参数矩阵]
        mean, log_std = torch.chunk(params, 2, dim=-1)
        # 为维持极值梯度稳定，将标准差值域硬截断至稳定区间
        std = torch.clamp(log_std, min=-5.0, max=2.0).exp()
        return Normal(mean, std)

    def forward(self, h_prev, z_prev, a_prev, obs_embed=None):
        """
        前向传播动力学单步演化算子。
        闭环阶段 (obs_embed 存在): 联合推断后验分布以重构观测并计算散度项。
        开环阶段 (obs_embed 空缺): 纯粹依据张量图先验展开未来序列。
        """
        # [沿时间轴递归展开非线性门控算子]
        rnn_input = torch.cat([h_prev, z_prev, a_prev], dim=-1)
        h_t = self.gru(rnn_input, h_prev)

        # [前向推导先验测度流形]
        prior_params = self.prior_mlp(h_t)
        prior_dist = self._split_dist(prior_params)

        post_dist = None
        # 遵循重参数化公理执行可导采样
        z_t = prior_dist.rsample()

        if obs_embed is not None:
            # [后验分布矫正过程仅激活于包含真实数据的训练回环]
            post_input = torch.cat([h_t, obs_embed], dim=-1)
            post_params = self.posterior_mlp(post_input)
            post_dist = self._split_dist(post_params)
            # 在拟合阶段，状态张量强制修正为后验信息源
            z_t = post_dist.rsample()

        return h_t, z_t, prior_dist, post_dist
```

在策略网络的反向传播架构中，我们切断一切调用 `env.step` 的物理调用序列，转而使用 `RSSMCell` 输出搭建动态计算图。以下模块刻画了基于全微积分链展开的策略网络迭代过程：

```{.python .input}
#@tab pytorch
class ActorCriticInDream(nn.Module):
    def __init__(self, rssm, actor, critic, horizon=15):
        super().__init__()
        self.rssm = rssm
        self.actor = actor
        self.critic = critic
        self.horizon = horizon

    def forward(self, init_h, init_z):
        h, z = init_h, init_z

        states = []

        # [在纯运算图中进行固定视界的自回归想象展演]
        for t in range(self.horizon):
            state_feature = torch.cat([h, z], dim=-1)
            states.append(state_feature)

            # [注入随机策略分布并应用重参数化技术抽取动作张量]
            action = self.actor(state_feature)

            # 驱使动力学模型执行纯先验状态转移
            h, z, _, _ = self.rssm(h, z, action, obs_embed=None)

            # 此处省略奖励模型的单步评估算子
            # reward = self.reward_model(torch.cat([h, z], dim=-1))

        # 最终阶段利用广义优势及截断递归累加算出目标价值标量
        # 解析梯度则由张量引擎中的 .backward() 指令隐式穿越整个时序拓扑结构
        pass
```

这种系统级工程架构揭示了一个深刻的事实：环境交互中本来离散断层且黑盒的部分，已经被严格地连续映射成为了具有完整雅可比矩阵（Jacobian Matrix）的张量复合函数。这即是利用世界模型极大幅度提升样本利用律的理论根源。

## 小结

在本节中，我们通过分析高阶马尔可夫演化算子的非线性与随机性缺陷，推演出了必须构建紧凑隐状态流形的数学必然性。基于随机变分后验方法和对数似然凸性的处理，我们严格导出了约束RSSM动力学的最优变分界（ELBO）。并结合微积分时间序列展开和重参数化技巧，详细证明了在张量计算网络进行平滑梯度流动的高效性。在下述章节内，我们将探索如何在此连续隐空间架构中引入基于分类分布和强化树的离散算子模块。
