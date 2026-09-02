# 8.5 基于世界模型想象的强化学习

假设一次真实机械臂抓取需要 10 秒，而模型在 GPU 中展开一条 15 步潜在轨迹只需很短时间。真实交互仍用来校正模型，策略更新则可以复用这些数据，在潜在空间中尝试更多动作。这种做法称为基于模型的**想象学习**。

<div align="center">
<img src="/figures/08-robot-sim/source/05-imagination-rl-rise/dreamer-fig2.png" alt="Dreamer 在多种视觉控制任务中从像素学习行为，展示潜在想象训练覆盖的实际任务范围。" width="86%">

_图 8.5-1：Dreamer 在多种视觉控制任务中从像素学习行为，展示潜在想象训练覆盖的实际任务范围。 出处：Danijar Hafner et al.，[Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)（2020），Figure 2。_
</div>

想象不会免费增加真实信息：模型没见过的接触或物体，仍可能被预测错。它的价值在于把已经收集的数据转化为可微、可重复的训练轨迹。本节以 RSSM 和 Dreamer 为主线说明这一过程。

## 学术溯源与理论动机

基于模型（Model-Based）的强化学习思想可以追溯到上世纪90年代Richard Sutton提出的Dyna架构 [[Sutton, 1990]](https://dl.acm.org/doi/10.5555/645530.658292)。Dyna的核心思想是利用智能体与真实环境交互收集的数据来训练一个环境动力学模型，随后智能体不仅在真实环境中学习，也在环境模型生成的模拟数据中学习。然而，早期的环境模型主要针对低维、离散的状态空间，难以处理高维的图像输入。

Ha 与 Schmidhuber 在《World Models》中用变分自编码器（VAE）压缩图像，再用混合密度循环网络（MDN-RNN）学习动作条件潜在动力学 [[Ha & Schmidhuber, 2018]](https://arxiv.org/abs/1803.10122)。CarRacing 的控制器在真实环境中优化；完全在模型“梦境”中训练并迁回真实环境的实验对应 VizDoom Take Cover。这个结果证明了特定任务上的可行性，不是对任意视觉控制任务的保证。

<div align="center">
<img src="/figures/08-robot-sim/source/05-imagination-rl-rise/worldmodels-fig7.png" alt="World Models 的梦境 rollout 把控制器置于学习到的潜在环境中训练，展示早期“内部模拟”路线。" width="86%">

_图 8.5-2：World Models 的梦境 rollout 把控制器置于学习到的潜在环境中训练，展示早期“内部模拟”路线。 出处：David Ha；Jürgen Schmidhuber，[World Models](https://arxiv.org/abs/1803.10122)（2018），Figure 7。_
</div>

Hafner 等人在 PlaNet 中提出 RSSM，Dreamer 随后用这一模型生成潜在想象轨迹，并通过价值估计与动力学梯度训练策略 [[Hafner et al., 2020]](https://arxiv.org/abs/1912.01603)。DreamerV2 与 DreamerV3 又扩展到离散潜变量和跨领域统一超参数 [[Hafner et al., 2021]](https://arxiv.org/abs/2010.02193); [[Hafner et al., 2023]](https://arxiv.org/abs/2301.04104)。这些论文在各自报告的基准上与强无模型方法比较，但不能概括为在样本效率和最终性能上“首次全面超越”所有无模型算法。

<div align="center">
<img src="/figures/08-robot-sim/source/05-imagination-rl-rise/planet-fig6.png" alt="PlaNet 在多个连续控制任务上的学习曲线展示 RSSM 与潜在空间规划的样本效率。" width="86%">

_图 8.5-3：PlaNet 在多个连续控制任务上的学习曲线展示 RSSM 与潜在空间规划的样本效率。 出处：Danijar Hafner et al.，[Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)（2019），Figure 6。_
</div>

## 动力学系统与隐空间映射

在正式引入复杂的变分推断之前，我们先从马尔可夫决策过程（MDP）的最基础动力学转移出发。

在一个确定性的动力学系统中，如果已知当前时刻 $t$ 的状态向量 $s_t$ 以及智能体执行的动作向量 $a_t$，系统的下一时刻状态 $s_{t+1}$ 可以由一个确定性的非线性函数精确映射得出：

$$s_{t+1} = f(s_t, a_t)$$

现实系统还包含难以观测的噪声和随机扰动，因此常用条件概率分布描述状态转移。给定当前状态 $s_t$ 和动作 $a_t$，下一状态 $s_{t+1}$ 服从：

$$s_{t+1} \sim p(s_{t+1} \mid s_t, a_t)$$

学到转移分布后，就可以展开多步预测。但当观测是 $64 \times 64 \times 3$ 的 RGB 图像时，直接预测每个像素既昂贵，又会把容量用于纹理等难以预测的细节。多步误差可能不断累积，却不一定按固定指数规律增长。

世界模型先把观测 $o_t$ 编码为较紧凑的潜在状态 $z_t \in \mathbb{R}^d$，再在潜在空间中预测：

$$z_{t+1} \sim p(z_{t+1} \mid z_t, a_t)$$

潜在状态需要保留预测奖励、终止和未来观测所需的信息。维度较低通常能降低展开成本，但表示是否充分要由训练目标与下游表现检验。

## 循环状态空间模型（RSSM）的严谨推导

隐空间动力学需要同时表达可由历史预测的结构与难以预测的随机变化。Hafner 等人提出的 RSSM 将确定性循环状态和随机潜变量结合起来。

RSSM 把潜在状态写成两部分：确定性循环状态 $h_t$ 汇总历史，随机状态 $z_t$ 表达当前不确定信息。它们不是数学上正交的子空间，而是承担不同建模角色的变量。

给定初始条件和一系列动作序列 $a_{1:T}$，观测序列 $o_{1:T}$ 与奖励序列 $r_{1:T}$ 及对应隐序列 $z_{1:T}$ 的联合似然模型为：

$$p(o_{1:T}, r_{1:T}, z_{1:T} \mid a_{1:T}) = \prod_{t=1}^T p(o_t \mid h_t, z_t) p(r_t \mid h_t, z_t) p(z_t \mid h_t)$$

<div align="center">
<img src="/figures/08-robot-sim/latex/05-imagination-rl-rise/rssm-observation-reward-factor.png" alt="每个时间步的潜变量先验、观测似然与奖励似然构成一个因子三元组并跨时间相乘" width="86%">

_图 8.5-4：每个时间步都乘入潜变量先验、观测似然与奖励似然；观测和奖励两项共享同一潜在状态条件。_
</div>

其中，确定性隐藏状态轨迹 $h_t$ 根据如下非线性递归方程更新：

$$h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})$$

这个分解包含三类条件分布：

1. **先验转移模型（Prior Dynamics）**：$p_\theta(z_t \mid h_t)$，负责在缺少当前观测信息的情况下，沿着时间轴预测随机变量分布的演化。
2. **观测重构模型（Observation Model）**：$p_\theta(o_t \mid h_t, z_t)$，定义从低维隐流形到高维观测流形的映射。
3. **奖励反馈模型（Reward Model）**：$p_\theta(r_t \mid h_t, z_t)$，提供单步标量反馈预测。

训练时希望最大化观测与奖励的边缘对数似然。潜变量积分通常无法直接计算，因此引入近似后验；它使用循环历史 $h_t$ 与当前观测 $o_t$：

$$q_\phi(z_t \mid h_t, o_t)$$

先验 $p_\theta(z_t\mid h_t)$ 在想象阶段不看当前图像，后验 $q_\phi(z_t\mid h_t,o_t)$ 在训练阶段看得到图像。KL 项让两者接近，使模型在移除真实观测后仍能沿先验展开。

运用詹森不等式（Jensen's Inequality），将对数函数的凸组合极值向内投射，我们获得时间序列联合分布的变分证据下界（ELBO）：

$$
\begin{aligned}
\ln p(o_{1:T}, r_{1:T} \mid a_{1:T}) &= \ln \mathbb{E}_{q_\phi(z_{1:T} \mid o_{1:T}, a_{1:T})} \left[ \frac{p(o_{1:T}, r_{1:T}, z_{1:T} \mid a_{1:T})}{q_\phi(z_{1:T} \mid o_{1:T}, a_{1:T})} \right] \\
&\ge \mathbb{E}_{q_\phi} \left[ \ln \frac{p(o_{1:T}, r_{1:T}, z_{1:T} \mid a_{1:T})}{q_\phi(z_{1:T} \mid o_{1:T}, a_{1:T})} \right] \\
&= \sum_{t=1}^T \mathbb{E}_{q_\phi} \Big[ \underbrace{\ln p_\theta(o_t \mid h_t, z_t)}_{\text{重构对数似然}} + \underbrace{\ln p_\theta(r_t \mid h_t, z_t)}_{\text{反馈对数似然}} - \underbrace{\text{KL}\big( q_\phi(z_t \mid h_t, o_t) \,\|\, p_\theta(z_t \mid h_t) \big)}_{\text{动力学分布散度}} \Big]
\end{aligned}
$$

这个下界包含三项：

1. **重构对数似然**鼓励潜变量保留预测观测所需的信息。
2. **奖励对数似然**鼓励状态保留与即时奖励有关的信息。
3. **KL 散度项**约束先验接近看过当前观测的后验。权重过强会压制信息，过弱则会让先验难以跟随后验。

## 在隐空间中展开梦境与策略寻优

<div align="center">
<img src="/figures/08-robot-sim/source/05-imagination-rl-rise/dreamerv3-fig4.png" alt="DreamerV3 的多步预测将真实帧、重建和开放循环潜在预测并列，展示想象轨迹保持任务信息的程度。" width="86%">

_图 8.5-5：DreamerV3 的多步预测将真实帧、重建和开放循环潜在预测并列，展示想象轨迹保持任务信息的程度。 出处：Danijar Hafner et al.，[Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)（2023），Figure 4。_
</div>

当动力学模型（参数为 $\theta$ 和 $\phi$）拟合收敛后，环境的真实动态规律已被提纯至连续函数的参数中。我们随即进入策略优化的“做梦”阶段（Dreaming Phase）。

想象轨迹从回放数据编码出的后验状态开始。随后 actor 输出动作，先验转移模型递归产生未来潜在状态；这段展开不再调用环境，但仍依赖从真实数据学到的模型：

$$a_t \sim \pi_\psi(a_t \mid \hat{s}_t), \quad \hat{z}_{t+1} \sim p_\theta(\hat{z}_{t+1} \mid \hat{h}_{t+1}), \quad \text{其中 } \hat{h}_{t+1} = f_\theta(\hat{h}_t, \hat{z}_t, a_t)$$

约定俗成地，采用上标 $\hat{\cdot}$ 标注该张量系在纯计算图内演化所得。为符号紧凑起见，令 $\hat{s}_t = (\hat{h}_t, \hat{z}_t)$。

如果动作与潜在状态采用可重参数化分布，actor 可以沿世界模型反向传播梯度。这个梯度对**学习到的模型目标**可导，但相对真实环境可能有模型偏差，因此不能称为无偏真实策略梯度。

以对角高斯潜变量为例，重参数化写成 $\hat{z}_{t+1}=\mu_\theta(\hat{h}_{t+1})+\sigma_\theta(\hat{h}_{t+1})\odot\epsilon$。固定噪声样本后，张量框架可以对这条有限长度计算图执行 BPTT：

$$\frac{\partial V(\hat{s}_t)}{\partial \psi} = \sum_{\tau=t}^{t+H} \gamma^{\tau-t} \frac{\partial \mathbb{E}[r_\tau]}{\partial \hat{s}_\tau} \frac{\partial \hat{s}_\tau}{\partial \psi}$$

价值网络 $v_\xi(\hat{s}_t)$ 估计想象视野之外的回报。$\lambda$-return 在短期预测与价值自举之间插值，能降低纯长视野展开的方差，但不会完全消除截断与模型偏差：

$$V_t^\lambda = \hat{r}_t + \gamma \begin{cases} (1-\lambda) v_\xi(\hat{s}_{t+1}) + \lambda V_{t+1}^\lambda, & t < H \\ v_\xi(\hat{s}_{t+1}), & t = H \end{cases}$$

训练时，actor 提高想象回报，critic 拟合停止梯度后的 $V_t^\lambda$ 目标。Dreamer 各版本在潜变量类型、损失细节和回报归一化上并不完全相同，代码实现需要对照具体版本。

## 代码实现与深度剖析

下面用张量代码实现一个简化的 `RSSMCell`，包含递归状态更新、先验与后验参数计算。

```python
import torch
from torch import nn
from torch.distributions import Normal

class RSSMCell(nn.Module):
    def __init__(self, action_dim, hidden_dim=200, latent_dim=30):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 确定性动力学方程: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)

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
        # h_prev 作为 GRU hidden，输入只拼接上一随机状态与动作
        rnn_input = torch.cat([z_prev, a_prev], dim=-1)
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

```python
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

        return torch.stack(states, dim=1)
```

这里返回的张量形状是 `(batch, horizon, hidden_dim + latent_dim)`。奖励模型、终止模型与 $\lambda$-return 尚未加入，因此这段代码只验证想象状态的计算图，不是完整 Dreamer 训练器。

## 小结

- RSSM 用确定性历史状态与随机状态共同表示潜在动力学。
- 训练阶段的后验使用当前观测，想象阶段的先验只能依赖历史与动作；KL 项连接这两条路径。
- actor 可以沿可微世界模型优化想象回报，但梯度会继承模型偏差。
- 价值自举与 $\lambda$-return 控制有限想象视野的偏差—方差折中。
