# 世界模型（World Models）绪论

> **本章导读**
>
> **讲什么：** 本章研究第一条完整路线：不在像素空间里直接推演，而是把观测压缩成潜在状态，再学习状态随动作如何变化。我们会沿着 World Models、RSSM、PlaNet、Dreamer 与 MuZero 的设计变化，比较“重建世界、规划动作、训练策略、只保留决策信息”这几种不同目标。
>
> **为什么要在潜空间中建模：** 一张赛车画面可能有几十万个像素，但决定转弯的往往只是道路形状、车速和姿态。逐像素预测不仅昂贵，还会把树叶抖动等无关细节当成主要任务；压缩后的状态若保留了行动所需的信息，就能用更小的模型展开更长的未来。
>
> **故事线：** `压缩观测并学习转移 → 用确定性记忆和随机状态组成 RSSM → 在潜空间搜索动作 → 在想象轨迹中训练策略 → 提高跨任务稳健性 → 只预测规划所需的奖励、价值与策略`

深度强化学习的早期突破主要来自无模型方法。例如，DQN 通过拟合动作价值函数，直接从 Atari 像素与游戏得分学习控制策略 [[Mnih et al., 2015]](https://doi.org/10.1038/nature14236)。这项结果针对离散动作的 Atari 游戏，并不覆盖连续控制；它同时也说明了一个工程问题：仅靠真实或仿真的环境交互往往需要大量样本。

在这样的背景下，Ha 和 Schmidhuber 给出了由视觉模型、记忆模型与控制器组成的 **World Models** 框架 [[Ha & Schmidhuber, 2018]](https://arxiv.org/abs/1803.10122)。它把表征学习、动力学建模与控制器优化拆成可以分别训练的组件。CarRacing 实验在真实环境中优化控制器，并使用世界模型提供的潜在特征；VizDoom 实验才把控制器完全放到学到的潜在环境中训练，再迁回真实游戏。论文没有证明该方法能以极少交互适用于任意复杂任务。

本章我们将从最基础的物理规律出发，严谨推演世界模型的数学结构，并剖析其为何必须依赖潜空间（Latent Space）与序列生成模型。

## 从基础物理到状态转移：预测的本质

在正式引入深度学习框架之前，让我们先回到高中物理中关于运动学最基础的描述。这是理解一切“动态预测”模型的起点。

假设有一个质点在直线上运动，已知其在 $t$ 时刻的位置为 $x_t \in \mathbb{R}$。如果我们对该质点施加一个动作（在此场景下，假设动作为提供一个恒定的速度 $v_t$），并在一个极小的时间间隔 $\Delta t$ 内保持不变，那么在 $t+1$ 时刻（即 $t + \Delta t$），质点的位置可以被唯一且精确地计算出来：

$$
x_{t+1} = x_t + v_t \Delta t
$$

在该公式中，我们实际上构建了一个极简的**环境模型**。它包含三个核心要素：

1. $x_t$：当前状态（State），在这里退化为一个标量。
2. $v_t$：智能体在 $t$ 时刻采取的动作（Action）。
3. $x_{t+1}$：动作施加于当前状态后，环境反馈出的未来状态。

在现代控制理论与强化学习中，我们将这种关系推广至高维向量空间。假设系统的状态由一个多维向量 $\mathbf{s}_t \in \mathbb{R}^n$ 描述，动作由向量 $\mathbf{a}_t \in \mathbb{R}^m$ 描述。一个确定性的状态转移函数（Deterministic Transition Function）可抽象地表示为：

$$
\mathbf{s}_{t+1} = f(\mathbf{s}_t, \mathbf{a}_t)
$$

然而，真实物理世界往往充满着观测噪声、隐藏变量（部分可观测性）以及环境本身的固有随机性。这就要求我们将确定性的函数映射升级为条件概率分布（Conditional Probability Distribution）。即在给定当前状态与动作的前提下，未来状态服从某一种概率分布：

$$
P(\mathbf{s}_{t+1} \mid \mathbf{s}_t, \mathbf{a}_t)
$$

深度学习视角下的“世界模型”，本质上就是用神经网络来高精度地拟合该公式所描述的这个条件概率分布。

## 维度的诅咒与潜空间降维

如果我们将环境直接设定为一个自动驾驶的仿真画面，此时观测空间（状态） $\mathbf{s}_t$ 是一张 $64 \times 64$ 的 RGB 图像。那么该状态空间的维度是 $D = 64 \times 64 \times 3 = 12288$。

如果我们试图直接在包含 $12288$ 个变量的原始像素空间中拟合转移概率分布 $P(\mathbf{s}_{t+1} \mid \mathbf{s}_t, \mathbf{a}_t)$，我们将面临极度严重的“维度诅咒”。更为关键的是，像素空间充满了对决策毫无意义的冗余信息。天空中云朵形状的细微变化、树叶随风摇曳的阴影，都会引起大量像素值的剧烈变化，但这对于车辆是否需要刹车完全没有影响。

因此，[[Ha & Schmidhuber, 2018]](https://arxiv.org/abs/1803.10122) 将预测系统优雅解耦为两个独立训练的模块：**视觉模型**（V Model）与**记忆模型**（M Model）。

### 视觉模型 (V Model)：空间压缩

视觉模型的任务是将高维的观测图像 $\mathbf{x}_t \in \mathbb{R}^D$ 映射到一个极其紧凑的低维潜空间（Latent Space）特征向量 $\mathbf{z}_t \in \mathbb{R}^d$，其中 $d \ll D$。

经典实现采用变分自编码器（Variational Autoencoder, VAE）[[Kingma & Welling, 2013]](https://arxiv.org/abs/1312.6114)。VAE 通过 KL 项把近似后验约束到标准正态先验附近，而不是让每个样本的潜变量都严格服从标准正态分布；这种正则化有助于得到较连续、便于采样的潜空间。其优化目标是最大化证据下界（ELBO）：

$$
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[-\log p_\theta(\mathbf{x}|\mathbf{z})] + D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \parallel \mathcal{N}(\mathbf{0}, \mathbf{I}))
$$

在该公式中，第一项为重建损失（保证 $\mathbf{z}$ 包含了恢复原始图像必需的信息），第二项为 KL 散度（约束潜空间的分布形态）。通过训练完毕的编码器 $q_\phi$，我们可以将每一帧极其庞大的像素矩阵 $\mathbf{x}_t$ 压缩为几十维的向量 $\mathbf{z}_t$。

### 记忆模型 (M Model)：时间序列与混合密度推导

经过 V 模型的处理后，环境的动态演化问题被完全转移到了低维的潜空间中。现在的目标是建模潜状态的演变律：

$$
P(\mathbf{z}_{t+1} \mid \mathbf{z}_{\le t}, \mathbf{a}_{\le t})
$$

这里的下标 $\le t$ 表示从 $0$ 时刻到 $t$ 时刻的完整历史轨迹。由于部分可观测性（单帧图像无法提供速度、加速度等时间衍生信息），我们需要利用循环神经网络（RNN）来压缩历史信息。设 RNN 在 $t$ 时刻的隐藏状态为 $\mathbf{h}_t \in \mathbb{R}^h$，它可以被视为对过去所有历史信息的聚合：

$$
\mathbf{h}_{t} = \text{RNN}(\mathbf{z}_{t}, \mathbf{a}_{t}, \mathbf{h}_{t-1})
$$

此时，上述转移概率可近似被化简为仅依赖于当前隐藏状态的条件分布：

$$
P(\mathbf{z}_{t+1} \mid \mathbf{z}_{\le t}, \mathbf{a}_{\le t}) \approx P(\mathbf{z}_{t+1} \mid \mathbf{h}_{t})
$$

为了极其严密地建模 $P(\mathbf{z}_{t+1} \mid \mathbf{h}_{t})$，我们需要回答一个问题：这个分布应该长什么样？

如果使用均方误差（MSE）作为损失函数来直接预测 $\mathbf{z}_{t+1}$，就隐式地假设了下一时刻的状态服从各向同性的单模态高斯分布，且我们只关心预测其均值。但这在现实中是不成立的。

::: info 理论抽象（仅有的一处类比）
我们可以将这种多模态预测，视作一位在十字路口观察的向导（即RNN的隐藏状态 $\mathbf{h}_t$）。向导知道车辆可能会向左转（概率 $\pi_1$，目标分布均值 $\mu_1$，路线不确定度 $\sigma_1$），也可能会向右转（概率 $\pi_2$），但他绝不会建议车辆“直直地撞向正前方的隔离墩”（这是两个不同决策所对应高斯分布的简单平均）。这种用多个带权重的正态分布来严密拼接、包络未知世界未来可能性的方式，正是**混合密度网络（MDN, [[Bishop, 1994]](https://www.microsoft.com/en-us/research/publication/mixture-density-networks/)）**的本质。
:::

让我们首先考察单维变量 $z \in \mathbb{R}$ 下的标准高斯分布：

$$
\mathcal{N}(z \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(z - \mu)^2}{2\sigma^2} \right)
$$

为了具备表达多模态（多种不同未来可能性）的能力，我们引入由 $K$ 个高斯分量叠加而成的高斯混合模型（Gaussian Mixture Model, GMM）：

$$
P(z) = \sum_{k=1}^K \pi_k \mathcal{N}(z \mid \mu_k, \sigma_k^2)
$$

其中混合权重满足 $\sum_{k=1}^K \pi_k = 1$ 且 $\pi_k \ge 0$。

将其拓展到我们维度为 $d$ 的潜状态空间向量 $\mathbf{z} \in \mathbb{R}^d$。在原始世界模型论文的严谨设定中，为了计算高效，假设在给定同一个混合分量 $k$ 时，潜变量各维度之间是条件独立的（即协方差矩阵为对角阵）。这使得多维的高斯联合分布可以拆解为单维高斯分布的连乘积：

$$
P(\mathbf{z}_{t+1} \mid \mathbf{h}_t) = \sum_{k=1}^K \pi_{k, t} \prod_{i=1}^d \mathcal{N}(z_{t+1,i} \mid \mu_{k,i,t}, \sigma_{k,i,t}^2)
$$

在上述该公式中，所有关于 $t$ 时刻的分布参数：混合权重 $\boldsymbol{\pi}_t$、各分量均值矩阵 $\boldsymbol{\mu}_t$ 以及方差矩阵 $\boldsymbol{\sigma}_t$，都必须由当前时刻的 RNN 隐藏状态 $\mathbf{h}_t$ 经过一个线性层严格映射得出：

$$
[\boldsymbol{\hat{\pi}}_t, \boldsymbol{\hat{\mu}}_t, \boldsymbol{\hat{\sigma}}_t] = W_o \mathbf{h}_t + \mathbf{b}_o
$$

在经过非线性激活以满足物理量约束（例如 $\pi$ 需要 Softmax 归一化，$\sigma$ 需要 Softplus 或 Exponential 保证严格为正）后，我们通过最小化负对数似然（Negative Log-Likelihood, NLL）来对 M 模型进行梯度下降优化：

$$
\mathcal{L}_{\text{MDN}} = \mathbb{E}_{t} \left[ -\log \left( \sum_{k=1}^K \pi_{k,t} \prod_{i=1}^d \frac{1}{\sqrt{2\pi\sigma_{k,i,t}^2}} \exp\left(-\frac{(z_{t+1,i} - \mu_{k,i,t})^2}{2\sigma_{k,i,t}^2}\right) \right) \right]
$$

## 架构与核心代码实现：MDN-RNN

在透彻理解了底层的严密推导后，我们将使用 PyTorch 构建记忆模型（M Model）最核心的 `MDNRNN` 模块。

为了使张量维度映射绝对清晰，我们预设潜变量维度 $d=32$，RNN隐藏维度 $h=256$，混合高斯分量数 $K=5$。
该模块接收 $t$ 时刻的动作序列和潜变量序列，在内部推演 RNN 的隐状态，最终输出下一步潜在分布的参数。

(**我们首先定义MDN-RNN的前向传播逻辑与损失计算**)

```python
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

class MDNRNN(nn.Module):
    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=256, num_gaussians=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians

        # RNN 核心组件：接收拼接后的 (z_t, a_t) 向量
        self.rnn = nn.LSTM(input_size=latent_dim + action_dim,
                           hidden_size=hidden_dim,
                           batch_first=True)

        # MDN 输出层映射：将 hidden_dim 映射到 GMM 所需的所有参数
        # 需要输出每个维度的 mu, sigma，以及每一个高斯簇的概率权重 pi
        self.fc_pi = nn.Linear(hidden_dim, num_gaussians)
        self.fc_mu = nn.Linear(hidden_dim, num_gaussians * latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, num_gaussians * latent_dim)

    def forward(self, z, action, hidden=None):
        """
        前向传播函数。
        输入参数：
            z: [batch_size, seq_len, latent_dim] - 当前时间步的观测潜变量
            action: [batch_size, seq_len, action_dim] - 当前时间步的动作
            hidden: (h_0, c_0) - RNN 的初始隐状态，默认全零
        """
        batch_size, seq_len, _ = z.size()

        # 将潜状态与动作在特征维度拼接：[batch_size, seq_len, latent_dim + action_dim]
        rnn_input = torch.cat([z, action], dim=-1)

        # rnn_out 的形状为 [batch_size, seq_len, hidden_dim]
        rnn_out, hidden = self.rnn(rnn_input, hidden)

        # 计算 pi: [batch_size, seq_len, num_gaussians]
        # 使用 softmax 保证各个高斯分量权重总和严格为 1
        pi = F.softmax(self.fc_pi(rnn_out), dim=-1)

        # 计算 mu: [batch_size, seq_len, num_gaussians, latent_dim]
        mu = self.fc_mu(rnn_out)
        mu = mu.view(batch_size, seq_len, self.num_gaussians, self.latent_dim)

        # 计算 sigma: [batch_size, seq_len, num_gaussians, latent_dim]
        # 使用 exp（或 softplus）保证方差参数严格大于 0，并具备良好的数值稳定性
        sigma = torch.exp(self.fc_sigma(rnn_out))
        sigma = sigma.view(batch_size, seq_len, self.num_gaussians, self.latent_dim)

        return pi, mu, sigma, hidden
```

在获得了由 `MDNRNN` 吐出的三个分布参数张量 `pi`、`mu` 和 `sigma` 后，我们需要计算对应的负对数似然损失。我们在设计损失函数计算时必须采取极为严谨的数值稳定措施。

(**实现混合密度网络的对数似然损失计算逻辑**)

```python
def mdn_loss(pi, mu, sigma, target_z):
    """
    严谨计算公式(4.1.9)对应的 MDN 负对数似然损失。
    输入参数：
        pi: [batch_size, seq_len, num_gaussians]
        mu: [batch_size, seq_len, num_gaussians, latent_dim]
        sigma: [batch_size, seq_len, num_gaussians, latent_dim]
        target_z: [batch_size, seq_len, latent_dim] - 目标分布，即 z_{t+1}
    """
    batch_size, seq_len, num_gaussians, latent_dim = mu.size()

    # 扩展 target_z 以匹配 mu 和 sigma 的多模态高斯簇维度
    # 形状变为：[batch_size, seq_len, num_gaussians, latent_dim]
    target_z = target_z.unsqueeze(2).expand_as(mu)

    # 利用 PyTorch 内置的 Normal 分布对象获取对数概率密度
    normal_dist = Normal(loc=mu, scale=sigma)

    # 计算在每个高斯分布下的 log P(z|mu, sigma)
    # 形状为 [batch_size, seq_len, num_gaussians, latent_dim]
    log_prob_per_dim = normal_dist.log_prob(target_z)

    # 因为假设各个特征维度(latent_dim)之间条件独立，概率连乘等价于 log 域的求和
    # 形状变为 [batch_size, seq_len, num_gaussians]
    log_prob_per_gaussian = torch.sum(log_prob_per_dim, dim=-1)

    # 将 pi 转换为对数空间以保证数值稳定：log(pi)
    log_pi = torch.log(pi + 1e-8)  # 加上极小量防止 log(0)

    # 计算 log(pi * N) = log(pi) + log(N)
    log_pi_times_prob = log_pi + log_prob_per_gaussian

    # 核心数学推导的最后一步：使用 logsumexp 技巧合并所有的高斯分量
    # 相当于 log(\sum_k pi_k * P_k(z))
    # 形状变为 [batch_size, seq_len]
    log_prob_final = torch.logsumexp(log_pi_times_prob, dim=-1)

    # 负对数似然 (NLL) 需要取反并对所有时间步和批次求均值
    nll_loss = -torch.mean(log_prob_final)

    return nll_loss
```

## 小结

至此，我们已经详尽推演了世界模型在架构层面的基础逻辑与核心数学范式。从简单的一维状态迁移方程起步，我们将问题逐渐升维，直到触碰到原始高维像素空间所面临的巨大挑战，进而引入了通过 **VAE（V模型）**降维到连续潜空间，并利用**混合密度神经网络 MDN-RNN（M模型）**来进行多模态时序预测的整体思想。

在下一节中，我们将深入探讨在获得了强大的“脑内模拟器”之后，**控制器（Controller）**如何利用进化的手段去优化最终的动作策略。
