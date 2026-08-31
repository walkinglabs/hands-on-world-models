# 循环状态空间模型 (RSSM)

在深度强化学习与世界模型的发展历程中，如何从高维的图像观测中学习到一个紧凑且能够准确预测未来的动态模型，一直是一个核心难题。在前面章节中，我们已经了解了如何使用自编码器将单帧图像压缩为低维潜变量。然而，真实世界是随时间演变的，智能体必须能够基于过去的历史预测未来的状态演化。

PlaNet 引入了循环状态空间模型（Recurrent State Space Model, RSSM）[[Hafner et al., 2019]](https://arxiv.org/abs/1811.04551)，Dreamer 随后沿用 RSSM，并用潜在想象轨迹训练策略 [[Hafner et al., 2020]](https://arxiv.org/abs/1912.01603)。RSSM 把确定性的循环状态与每一步的随机潜变量结合起来。本节将从概率法则出发推导这一结构，并把它转化为可运行的代码。

## 历史背景与经典动态模型的局限性

为了理解为什么我们需要 RSSM，我们必须先审视以前的方法在面对复杂非确定性环境时的局限性。

早期的动态模型通常依赖于单纯的确定性循环神经网络（如 RNN、LSTM 或 GRU）。在这样的模型中，未来的隐状态完全由当前状态和当前动作决定。假设 $h_t$ 为时间步 $t$ 的确定性隐状态，$a_t$ 为动作，则动态转移可以表示为：

$$
h_t = f_\theta(h_{t-1}, a_{t-1})
$$

这种方法的致命弱点在于：真实世界充满了随机性与不可观测的混淆变量。如果一个纯确定性的模型试图预测多条可能的分支未来，它最终只能预测出所有可能性的平均值（在图像重建上表现为模糊的重影）。

为了解决这个问题，研究者们引入了纯随机状态空间模型（SSM）。在 SSM 中，状态的转移不再是确定性的，而是一个概率分布：

$$
s_t \sim p(s_t \mid s_{t-1}, a_{t-1})
$$

纯随机模型能够描绘多种可能的未来，但在实际训练中，由于每个时间步都引入了采样噪声，模型极难记住跨越数百个时间步的长期信息。在深度学习的早期，受限于硬件算力与变分推断技术的成熟度，纯随机模型的长期预测往往会随着时间步的增加而崩溃。

RSSM 的核心学术贡献在于：它将隐状态拆解为一个确定性成分 $h_t$ 和一个随机成分 $s_t$，从而在记忆长期历史和捕捉即时不确定性之间找到了完美的平衡。

## 从基础概率到隐状态转移

在深入复杂的张量运算之前，让我们回到高中学过的基础概率论。假设我们有一系列观测变量 $o_1, o_2, \dots, o_T$（比如环境传回的图像序列）以及一系列动作 $a_1, a_2, \dots, a_{T-1}$。我们希望构建一个模型，使其能够最大化产生这些观测序列的概率。

直接对高维连续变量 $o_t$ 建模其联合分布极其困难。因此，我们引入隐变量（Latent Variables） $s_t$。隐变量是我们假设的存在于系统内部，能够用低维向量完整描述系统当前客观情况的抽象表示。

根据高中概率论中的条件概率与全概率公式，结合马尔可夫假设（未来状态仅取决于当前状态），如果我们知道系统的初始隐状态分布 $p(s_1)$，以及状态随时间演化的转移概率 $p(s_t \mid s_{t-1}, a_{t-1})$，并且观测仅仅依赖于当前隐状态 $p(o_t \mid s_t)$，那么整个序列的联合概率可以被严密地分解为连续的乘积形式：

$$
p(s_{1:T}, o_{1:T} \mid a_{1:T-1}) = p(s_1) p(o_1 \mid s_1) \prod_{t=2}^T p(s_t \mid s_{t-1}, a_{t-1}) p(o_t \mid s_t)
$$

上述公式看似简单，但它准确地指明了构建世界模型所需的两个核心组件：

1. **转移模型（Transition Model）**：由 $p(s_t \mid s_{t-1}, a_{t-1})$ 刻画，用于预测环境的内部状态将如何演变。
2. **观测模型（Observation Model）**：由 $p(o_t \mid s_t)$ 刻画，用于将抽象的隐状态解码还原为具体的感官像素输入。

## RSSM 的核心结构与数学拆解

RSSM 对标准状态空间模型中单一的转移概率 $p(s_t \mid s_{t-1}, a_{t-1})$ 进行了深度的结构化改造。在 RSSM 中，每一时刻的系统状态被明确拆分为两个相互关联的连续张量：

- **确定性状态 $h_t$**：通常是一个较高维度的向量（例如 200 维或 512 维）。它是通过 GRU（门控循环单元）更新的，负责记忆长期的历史信息。
- **随机状态 $s_t$**：通常是一个服从高斯分布或离散分布的随机变量。它负责刻画当前时刻环境的不确定性。

时间步 $t$ 的状态计算被严格拆分为以下三个连续的步骤：

首先，基于上一个时间步的随机状态 $s_{t-1}$ 和动作 $a_{t-1}$，计算当前的确定性状态 $h_t$。这是一个纯粹的代数映射过程：

$$
h_t = f_{\text{RNN}}(h_{t-1}, s_{t-1}, a_{t-1})
$$

接着，模型需要预测一个没有看过当前实际观测的**先验分布（Prior Distribution）**。这个分布仅仅基于 $h_t$ 计算得出，代表了模型基于过去的经验对现在将要发生的事情的预测：

$$
\hat{p}(s_t) \sim \mathcal{N}(\mu_{\text{prior}}(h_t), \sigma_{\text{prior}}(h_t))
$$

最后，当模型真正接收到环境在时间步 $t$ 传来的真实观测 $o_t$（实际操作中通常是通过卷积神经网络提取出的一维特征 $e_t$）时，模型会将先验知识与当前观测相融合，计算出更加准确的**后验分布（Posterior Distribution）**：

$$
q(s_t \mid h_t, e_t) \sim \mathcal{N}(\mu_{\text{post}}(h_t, e_t), \sigma_{\text{post}}(h_t, e_t))
$$

在这里，先验分布用于在没有观测反馈的“想象”阶段（Planning 阶段）生成未来的模拟轨迹；而后验分布则用于在训练阶段，利用真实的观测数据纠正模型内部的信念。

## 变分下界与目标函数推导

我们已经定义了模型的结构与张量的流动方式，但如何使用反向传播算法来优化这些神经网络参数呢？直接最大化观测序列的边缘似然 $\log p(o_{1:T})$ 包含了一个极其棘手且无法计算的积分（我们需要对所有可能的潜变量演化轨迹 $s_{1:T}$ 积分）：

$$
\log p(o_{1:T}) = \log \int p(o_{1:T}, s_{1:T}) \, ds_{1:T}
$$

> 💡 **类比：洞穴里的侦探与侧写**
> 这个数学困境就像是一个被困在黑暗洞穴里的侦探，只能通过墙上的光影变化（真实的观测 $o_t$）来推断洞外发生的事情（真实的隐状态 $s_t$）。由于可能有无数种洞外的情况都会产生完全相同的光影（在数学上这对应于高维积分），直接求出精确的客观真相（边缘似然）是不可行的。但是，如果侦探在脑海中建立了一个合理的猜测模型（变分后验 $q(s_t \mid o_t)$），并通过每次光影的验证来拉近猜测与直觉的距离（KL 散度），他就能不断提高对复杂环境的预测能力。这正是变分推断的精髓。

为了避开这个积分，我们使用变分推断。我们引入上文定义的变分后验分布 $q(s_{1:T} \mid o_{1:T}, a_{1:T-1})$，利用 Jensen 不等式，我们可以严格推导出对数似然的证据下界（ELBO，Evidence Lower Bound）：

$$
\log p(o_{1:T}) \ge \mathbb{E}_{q}\left[\sum_{t=1}^T \log p(o_t \mid s_t)\right] - \mathbb{E}_{q}\left[\sum_{t=1}^T D_{\text{KL}}(q(s_t \mid h_t, e_t) \parallel p(s_t \mid h_t))\right]
$$

让我们仔细审视这个极其优美的目标函数公式，它恰好由两部分组成，并且对应着非常明确的物理与几何意义：

1. **重构损失（Reconstruction Loss）**：即 $\mathbb{E}_q[\log p(o_t \mid s_t)]$。模型必须能够从后验分布中采样的隐状态 $s_t$ 出发，通过解码器网络完美地还原出当前的原始观测图像 $o_t$。
2. **KL 散度正则化（KL Divergence）**：即 $D_{\text{KL}}(q(s_t \mid h_t, e_t) \parallel p(s_t \mid h_t))$。其作用是强迫在没有观测参与的先验分布 $p(s_t \mid h_t)$，在参数空间中尽可能地逼近融合了当前真实观测的后验分布 $q(s_t \mid h_t, e_t)$。只有这样，在未来脱离真实环境进行闭环推理运算时，先验分布才能产生合理且不偏离真实的预测。

## 模型实现与张量维度详解

现在，我们将上述严密的数学理论转化为具体的神经网络代码。在实现中，我们假设状态变量采用多维对角高斯分布。特别需要注意的是重参数化技巧（Reparameterization Trick）的使用，它使得随机采样操作变得可导：$s_t = \mu + \sigma \odot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$。这在 PyTorch 的 `Normal.rsample()` 中已被自动实现。

(**下面我们定义 RSSM 的核心模块代码。**)

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
        # 为保持数值稳定性，对对数标准差进行硬性裁剪，避免出现梯度爆炸或方差绝对为 0
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

这段核心代码精准地实现了我们在上文推导的数学公式。需要注意的是，由于强化学习序列中的时间跨度较长，在实际模型训练时，我们需要在外部添加一个时间维度的 `for` 循环，将批次内的序列一步步送入 `forward_posterior`。这保证了隐藏状态 `h_t` 能够随着时间的推移不断积累并更新历史记忆，同时这也意味着我们需要警惕长序列上反向传播时的显存消耗与梯度消失问题。

## 小结

在本节中，我们回顾了状态空间模型的演进过程，从高中概率乘法法则出发推导出序列隐状态的联合概率，并深入推导了 **RSSM 的确定性与随机性双流网络架构**。我们详细解释了最大化**证据下界（ELBO）**在优化整个序列推断模型中的核心作用，并给出了结构严密的 PyTorch 实现。通过引入 **$h_t$ 和 $s_t$ 的解耦设计**，RSSM 不仅能够利用 RNN 提供长达数百步的长期跨度记忆，同时还保留了对下一步高方差环境的敏锐建模能力。
