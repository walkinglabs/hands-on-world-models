# 4.2 循环状态空间模型（RSSM）

单帧自编码器可以压缩图像，却没有描述潜变量随动作如何变化。序列世界模型还要处理两个问题：过去观测中哪些信息应被保留，以及同一段历史之后是否存在多个合理未来。

PlaNet 引入了循环状态空间模型（Recurrent State Space Model, RSSM）[[Hafner et al., 2019]](https://arxiv.org/abs/1811.04551)，Dreamer 随后沿用 RSSM，并用潜在想象轨迹训练策略 [[Hafner et al., 2020]](https://arxiv.org/abs/1912.01603)。RSSM 把确定性的循环状态与每一步的随机潜变量结合起来。本节将从概率法则出发推导这一结构，并把它转化为可运行的代码。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/02-rssm/opening-rssm-quadruped.gif" alt="上方真实四足运动与下方 RSSM 开环预测同步推进，使“无新观测时仍在潜空间滚动未来”先变成可观察结果。" width="86%">

_图 4.2-1：上方真实四足运动与下方 RSSM 开环预测同步推进，使“无新观测时仍在潜空间滚动未来”先变成可观察结果。 出处：Danijar Hafner；Timothy Lillicrap；Jimmy Ba；Mohammad Norouzi，[Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)（2020），Official multi-step prediction GIF: Quadruped。_

</div>

## 历史背景与经典动态模型的局限性

先比较确定性和随机状态模型各自提供的信息。

早期的动态模型通常依赖于单纯的确定性循环神经网络（如 RNN、LSTM 或 GRU）。在这样的模型中，未来的隐状态完全由当前状态和当前动作决定。假设 $h_t$ 为时间步 $t$ 的确定性隐状态，$a_t$ 为动作，则动态转移可以表示为：

$$
h_t = f_\theta(h_{t-1}, a_{t-1})
$$

确定性状态每次只产生一个后继表示。若再配合像素 MSE 训练，它在多模态未来上容易得到折中的平均预测；但这不是所有确定性模型都必然模糊，结果还取决于输出分布与训练目标。

为了解决这个问题，研究者们引入了纯随机状态空间模型（SSM）。在 SSM 中，状态的转移不再是确定性的，而是一个概率分布：

$$
s_t \sim p(s_t \mid s_{t-1}, a_{t-1})
$$

随机状态可以表达多个可能后继，但每一步都要通过采样变量传递信息，训练时的梯度估计和长期记忆可能更困难。它并非天然无法记住长期信息；RSSM 的设计选择是用确定性递归路径承载稳定记忆，再用随机变量表示每一步的不确定部分。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/02-rssm/vrnn-fig1.png" alt="VRNN 原始概率图把每步先验、后验和循环状态放在同一时间链上，是 RSSM 随机路径的直接前史。" width="86%">

_图 4.2-2：VRNN 原始概率图把每步先验、后验和循环状态放在同一时间链上，是 RSSM 随机路径的直接前史。 出处：Junyoung Chung；Kyle Kastner；Laurent Dinh；Kratarth Goel；Aaron Courville；Yoshua Bengio，[A Recurrent Latent Variable Model for Sequential Data](https://arxiv.org/abs/1506.02216)（2015），Figure 1。_

</div>

RSSM 将隐状态拆成确定性成分 $h_t$ 与随机成分 $s_t$，让两条路径分别承担历史汇总和随机信息表达。它提供了一种有效权衡，而不是对所有任务都“完美”的固定分解。

## 从基础概率到隐状态转移

设观测序列为 $o_1,o_2,\dots,o_T$，动作序列为 $a_1,a_2,\dots,a_{T-1}$。生成模型要在给定动作的条件下解释观测序列。

直接在高维像素上建模联合分布代价很高，因此引入隐变量 $s_t$。它是模型为预测与重构学习出的内部表示，不等同于环境中完整、客观且唯一的真实状态。

在一阶马尔可夫假设与“观测只依赖当前隐状态”的条件下，联合分布可分解为：

<div align="center">
  <img src="/figures/04-latent-dynamics/source/02-rssm/dkf-fig1.png" alt="Deep Kalman Filter 并列生成转移与识别模型依赖，为正文中的序列联合分解提供一手概率图。" width="86%">

_图 4.2-3：Deep Kalman Filter 并列生成转移与识别模型依赖，为正文中的序列联合分解提供一手概率图。 出处：Rahul G. Krishnan；Uri Shalit；David Sontag，[Deep Kalman Filters](https://arxiv.org/abs/1511.05121)（2015），Figure 1。_

</div>

$$
p(s_{1:T}, o_{1:T} \mid a_{1:T-1}) = p(s_1) p(o_1 \mid s_1) \prod_{t=2}^T p(s_t \mid s_{t-1}, a_{t-1}) p(o_t \mid s_t)
$$

上述公式看似简单，但它准确地指明了构建世界模型所需的两个核心组件：

1. **转移模型（Transition Model）**：由 $p(s_t \mid s_{t-1}, a_{t-1})$ 刻画，用于预测环境的内部状态将如何演变。
2. **观测模型（Observation Model）**：由 $p(o_t \mid s_t)$ 刻画，用于将抽象的隐状态解码还原为具体的感官像素输入。

## RSSM 的核心结构与数学拆解

RSSM 进一步把每一时刻的模型状态拆成两个相关部分：

- **确定性状态 $h_t$**：由 GRU 等循环单元更新，汇总对后续预测有用的历史信息。
- **随机状态 $s_t$**：通常是一个服从高斯分布或离散分布的随机变量。它负责刻画当前时刻环境的不确定性。

时间步 $t$ 的计算分三步进行。

首先，基于上一个时间步的随机状态 $s_{t-1}$ 和动作 $a_{t-1}$，计算当前的确定性状态 $h_t$。这是一个纯粹的代数映射过程：

$$
h_t = f_{\text{RNN}}(h_{t-1}, s_{t-1}, a_{t-1})
$$

<div align="center"><img src="/figures/04-latent-dynamics/latex/02-rssm/rssm-causal-state-order.png" alt="上一时刻的确定性状态、随机状态和动作先更新 h_t，再由 h_t 产生当前随机状态先验" width="86%">

_图 4.2-4：时刻 t 的先验更新只读取 t−1 信息：先形成确定性记忆 h_t，再由它给出当前随机状态分布并采样。本文根据上式绘制。_

</div>

接着，仅根据 $h_t$ 得到尚未读取当前观测的**先验分布（Prior Distribution）**：

$$
p_\theta(s_t \mid h_t) = \mathcal{N}\!\left(\mu_{\text{prior}}(h_t), \operatorname{diag}(\sigma^2_{\text{prior}}(h_t))\right)
$$

训练时，编码器把当前观测 $o_t$ 变为特征 $e_t$，推断网络据此构造**后验分布（Posterior Distribution）**：

$$
q_\phi(s_t \mid h_t, e_t) = \mathcal{N}\!\left(\mu_{\text{post}}(h_t,e_t), \operatorname{diag}(\sigma^2_{\text{post}}(h_t,e_t))\right)
$$

后验用于把训练序列中的当前观测纳入状态估计；先验则在没有新观测的想象或规划阶段滚动生成未来。两者之间的 KL 项让先验学习逼近由观测辅助得到的后验。

## 变分下界与目标函数推导

直接计算观测序列的边际似然需要对所有潜变量轨迹积分；对非线性神经网络模型，这个积分通常没有可直接使用的闭式解：

$$
\log p(o_{1:T}) = \log \int p(o_{1:T}, s_{1:T}) \, ds_{1:T}
$$

> 💡 **类比：洞穴里的侦探与侧写**
> 这个数学困境就像是一个被困在黑暗洞穴里的侦探，只能通过墙上的光影变化（真实的观测 $o_t$）来推断洞外发生的事情（真实的隐状态 $s_t$）。由于可能有无数种洞外的情况都会产生完全相同的光影（在数学上这对应于高维积分），直接求出精确的客观真相（边缘似然）是不可行的。但是，如果侦探在脑海中建立了一个合理的猜测模型（变分后验 $q(s_t \mid o_t)$），并通过每次光影的验证来拉近猜测与直觉的距离（KL 散度），他就能不断提高对复杂环境的预测能力。这正是变分推断的精髓。

引入变分后验 $q(s_{1:T} \mid o_{1:T},a_{1:T-1})$ 后，可得到对数似然的证据下界（ELBO）：

$$
\log p(o_{1:T}) \ge \mathbb{E}_{q}\left[\sum_{t=1}^T \log p(o_t \mid s_t)\right] - \mathbb{E}_{q}\left[\sum_{t=1}^T D_{\text{KL}}(q(s_t \mid h_t, e_t) \parallel p(s_t \mid h_t))\right]
$$

这个简化写法包含两类主要项：

1. **观测对数似然** $\mathbb{E}_q[\log p(o_t \mid s_t)]$：鼓励后验状态保留解码观测所需的信息；它不要求逐像素“完美还原”。
2. **KL 散度** $D_{\text{KL}}(q(s_t \mid h_t,e_t)\parallel p(s_t \mid h_t))$：缩小观测辅助后验与历史先验之间的差异，使没有新观测时的先验滚动更接近训练期间的状态分布。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/02-rssm/planet-fig3a.png" alt="PlaNet 的标准训练子图用观测发射边和先验—后验 KL 边对应 RSSM 目标中的重构项与正则项。" width="86%">

_图 4.2-5：PlaNet 的标准训练子图用观测发射边和先验—后验 KL 边对应 RSSM 目标中的重构项与正则项。 出处：Danijar Hafner；Timothy Lillicrap；Ian Fischer；Ruben Villegas；David Ha；Honglak Lee；James Davidson，[Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)（2019），Figure 3(a)。_

</div>

## 模型实现与张量维度详解

下面实现连续对角高斯版本的 RSSM。`Normal.rsample()` 使用重参数化 $s_t=\mu+\sigma\odot\epsilon$，其中 $\epsilon\sim\mathcal{N}(0,\mathbf{I})$，从而保留关于分布参数的路径梯度。

核心模块如下。

```python
import torch
from torch import nn
from torch.distributions import Normal

class RSSMCore(nn.Module):
    def __init__(self, action_dim, state_dim, rnn_hidden_dim, embed_dim):
        """
        初始化 RSSM 核心模块。
        参数:
            action_dim (int): 动作空间的维度
            state_dim (int): 随机潜状态 s_t 的维度
            rnn_hidden_dim (int): 确定性状态 h_t (RNN隐藏状态) 的维度
            embed_dim (int): 图像观测编码 e_t 的维度
        """
        super().__init__()
        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        # 转移模型的核心 RNN，这里我们使用单层 GRU
        # 输入是拼接后的 s_{t-1} 和 a_{t-1}
        self.rnn = nn.GRUCell(state_dim + action_dim, rnn_hidden_dim)

        # 先验分布网络：从 h_t 映射到 s_t 的均值和对数标准差
        self.prior_net = nn.Sequential(
            nn.Linear(rnn_hidden_dim, rnn_hidden_dim),
            nn.ELU(),
            nn.Linear(rnn_hidden_dim, 2 * state_dim)
        )

        # 后验分布网络：从 h_t 和 e_t 的拼接映射到 s_t 的均值和对数标准差
        self.posterior_net = nn.Sequential(
            nn.Linear(rnn_hidden_dim + embed_dim, rnn_hidden_dim),
            nn.ELU(),
            nn.Linear(rnn_hidden_dim, 2 * state_dim)
        )

    def _build_dist(self, params):
        """
        根据网络输出构建对角高斯分布。
        参数 params 维度为 (batch_size, 2 * state_dim)
        """
        # 将输出切分为均值和对数标准差
        mu, log_std = torch.chunk(params, 2, dim=-1)
        # 裁剪对数标准差，避免数值尺度过大或过小
        std = torch.exp(torch.clamp(log_std, min=-5.0, max=2.0))
        return Normal(mu, std)

    def forward_prior(self, h_prev, s_prev, action):
        """
        前向先验推断（在想象阶段使用）。
        计算确定性状态的更新以及先验分布。
        """
        # 将上一时刻的潜状态和动作拼接
        rnn_input = torch.cat([s_prev, action], dim=-1)
        # 更新确定性状态 h_t
        h_t = self.rnn(rnn_input, h_prev)

        # 计算先验分布参数
        prior_params = self.prior_net(h_t)
        prior_dist = self._build_dist(prior_params)

        # 使用重参数化技巧进行采样
        s_t = prior_dist.rsample()

        return h_t, s_t, prior_dist

    def forward_posterior(self, h_prev, s_prev, action, obs_embed):
        """
        前向后验推断（在训练阶段使用）。
        计算完整的推断过程，并返回后验和先验分布以便在外部计算 KL 散度损失。
        """
        # 1. 首先计算确定性状态 h_t 与先验分布
        h_t, _, prior_dist = self.forward_prior(h_prev, s_prev, action)

        # 2. 结合观测编码 e_t 计算后验分布
        post_input = torch.cat([h_t, obs_embed], dim=-1)
        post_params = self.posterior_net(post_input)
        post_dist = self._build_dist(post_params)

        # 3. 从后验分布中提取重参数化采样 s_t
        s_t = post_dist.rsample()

        return h_t, s_t, prior_dist, post_dist
```

训练完整序列时，需要在外部沿时间维循环调用 `forward_posterior`。这样 $h_t$ 会递归更新，但也会增加反向传播的显存开销；实际系统常结合截断、序列分块或并行化技巧。这里省略了观测解码器、奖励模型、终止模型和完整损失归约，因此代码只覆盖 RSSM 状态更新核心。

## 小结

本节从状态空间模型的联合分布出发，拆解了 RSSM 的确定性递归状态 $h_t$、随机状态 $s_t$、观测先验与观测后验。ELBO 同时训练观测模型与潜在转移。两条状态路径让模型可以分别汇总历史和表达不确定性，但能保留多长记忆、预测多远，仍取决于容量、数据和训练设置。
