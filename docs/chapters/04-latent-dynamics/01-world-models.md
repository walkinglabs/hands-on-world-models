# 4.1 World Models：压缩观测并预测潜在未来

> **本章导读**
>
> **讲什么：** 本章研究第一条完整路线：不在像素空间里直接推演，而是把观测压缩成潜在状态，再学习状态随动作如何变化。我们会沿着 World Models、RSSM、PlaNet、Dreamer 与 MuZero 的设计变化，比较“重建世界、规划动作、训练策略、只保留决策信息”这几种不同目标。
>
> **为什么要在潜空间中建模：** 一张赛车画面可能有几十万个像素，但决定转弯的往往只是道路形状、车速和姿态。逐像素预测不仅昂贵，还会把树叶抖动等无关细节当成主要任务；压缩后的状态若保留了行动所需的信息，就能用更小的模型展开更长的未来。
>
> **故事线：** `压缩观测并学习转移 → 用确定性记忆和随机状态组成 RSSM → 在潜空间搜索动作 → 在想象轨迹中训练策略 → 提高跨任务稳健性 → 只预测规划所需的奖励、价值与策略`

深度强化学习的早期突破主要来自无模型方法。例如，DQN 通过拟合动作价值函数，直接从 Atari 像素与游戏得分学习控制策略 [[Mnih et al., 2015]](https://doi.org/10.1038/nature14236)。这项结果针对离散动作的 Atari 游戏，并不覆盖连续控制；它同时也说明了一个工程问题：仅靠真实或仿真的环境交互往往需要大量样本。

在这样的背景下，Ha 和 Schmidhuber 给出了由视觉模型、记忆模型与控制器组成的 **World Models** 框架 [[Ha & Schmidhuber, 2018]](https://arxiv.org/abs/1803.10122)。它把表征学习、动力学建模与控制器优化拆成可以分别训练的组件。CarRacing 实验在真实环境中优化控制器，并使用世界模型提供的潜在特征；VizDoom 实验才把控制器完全放到学到的潜在环境中训练，再迁回真实游戏。论文没有证明该方法能以极少交互适用于任意复杂任务。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/01-world-models/wm-fig10.png" alt="同一帧 CarRacing 画面经 VAE 压缩再解码后，道路走向和车辆位置仍被保留，说明控制相关信息可以进入紧凑潜变量。" width="86%">

_图 4.1-1：同一帧 CarRacing 画面经 VAE 压缩再解码后，道路走向和车辆位置仍被保留，说明控制相关信息可以进入紧凑潜变量。 出处：David Ha；Jürgen Schmidhuber，[World Models](https://arxiv.org/abs/1803.10122)（2018），Figure 10。_

</div>

本节从状态转移出发，说明为什么可以先把图像压缩到潜空间（Latent Space），再用序列模型预测潜变量的变化。这里介绍的是 Ha 与 Schmidhuber 的具体架构，不代表所有世界模型都必须采用 VAE 与 RNN。

## 从基础物理到状态转移：预测的本质

先看一个最简单的离散时间动力学例子。

假设质点在 $t$ 时刻的位置为 $x_t \in \mathbb{R}$，并在时间间隔 $\Delta t$ 内保持速度 $v_t$ 不变，则欧拉离散化给出：

$$
x_{t+1} = x_t + v_t \Delta t
$$

这个简化的**环境模型**包含三个量：

1. $x_t$：当前状态（State），在这里退化为一个标量。
2. $v_t$：该时间段内的速度；在控制问题中，它可由动作间接影响，但不必等同于动作本身。
3. $x_{t+1}$：动作施加于当前状态后，环境反馈出的未来状态。

在现代控制理论与强化学习中，我们将这种关系推广至高维向量空间。假设系统的状态由一个多维向量 $\mathbf{s}_t \in \mathbb{R}^n$ 描述，动作由向量 $\mathbf{a}_t \in \mathbb{R}^m$ 描述。一个确定性的状态转移函数（Deterministic Transition Function）可抽象地表示为：

$$
\mathbf{s}_{t+1} = f(\mathbf{s}_t, \mathbf{a}_t)
$$

若系统存在观测噪声、隐藏变量或随机转移，单个确定性映射就不足以描述全部可能结果。此时可建模条件分布：

$$
P(\mathbf{s}_{t+1} \mid \mathbf{s}_t, \mathbf{a}_t)
$$

在强化学习中，世界模型通常学习状态转移，并可能同时预测观测、奖励或终止信号。它不必重建所有像素，只需保留下游预测或决策需要的信息。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/01-world-models/e2c-fig1.png" alt="E2C 把图像编码、局部线性潜在转移与图像解码连成控制模型，展示潜空间动力学在 World Models 之前的控制脉络。" width="86%">

_图 4.1-2：E2C 把图像编码、局部线性潜在转移与图像解码连成控制模型，展示潜空间动力学在 World Models 之前的控制脉络。 出处：Manuel Watter；Jost Tobias Springenberg；Joschka Boedecker；Martin Riedmiller，[Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images](https://arxiv.org/abs/1506.07365)（2015），Figure 1。_

</div>

## 维度的诅咒与潜空间降维

如果我们将环境直接设定为一个自动驾驶的仿真画面，此时观测空间（状态） $\mathbf{s}_t$ 是一张 $64 \times 64$ 的 RGB 图像。那么该状态空间的维度是 $D = 64 \times 64 \times 3 = 12288$。

直接预测 12288 个像素变量需要较大的模型与计算量，而且像素误差会同等惩罚任务相关和无关的细节。若道路边缘、车速和姿态决定控制，背景纹理的微小变化却未必值得分配同样的预测容量。

World Models 将感知与预测拆成两个分别训练的模块：**视觉模型**（V Model）和**记忆模型**（M Model）。

### 视觉模型 (V Model)：空间压缩

视觉模型把高维观测 $\mathbf{x}_t \in \mathbb{R}^D$ 编码为较低维的潜变量 $\mathbf{z}_t \in \mathbb{R}^d$，其中 $d \ll D$。

原论文使用变分自编码器（Variational Autoencoder, VAE）[[Kingma & Welling, 2013]](https://arxiv.org/abs/1312.6114)。VAE 通过 KL 项把近似后验约束到标准正态先验附近，而不是让每个样本的潜变量都严格服从标准正态分布。若把训练目标写成需要最小化的负 ELBO，可表示为：

$$
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[-\log p_\theta(\mathbf{x}|\mathbf{z})] + D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \parallel \mathcal{N}(\mathbf{0}, \mathbf{I}))
$$

第一项是负重构对数似然，鼓励潜变量保留解码观测所需的信息；第二项约束近似后验与先验之间的差异。编码器随后把每一帧像素观测压缩为潜变量 $\mathbf{z}_t$。

### 记忆模型 (M Model)：时间序列与混合密度推导

经过 V 模型编码后，记忆模型在潜空间中预测下一步：

$$
P(\mathbf{z}_{t+1} \mid \mathbf{z}_{\le t}, \mathbf{a}_{\le t})
$$

下标 $\le t$ 表示截至 $t$ 的轨迹。在部分可观测环境中，单帧通常不足以推断速度等量，因此 RNN 用有限维隐藏状态 $\mathbf{h}_t$ 汇总与预测有关的历史；它不保证无损保存全部过去：

<div align="center">
  <img src="/figures/04-latent-dynamics/source/01-world-models/vrnn-fig3.png" alt="VRNN 与 RNN-GMM 的语音序列样本对照显示，逐时刻随机潜变量能够表达序列中的局部变化。" width="86%">

_图 4.1-3：VRNN 与 RNN-GMM 的语音序列样本对照显示，逐时刻随机潜变量能够表达序列中的局部变化。 出处：Junyoung Chung；Kyle Kastner；Laurent Dinh；Kratarth Goel；Aaron Courville；Yoshua Bengio，[A Recurrent Latent Variable Model for Sequential Data](https://arxiv.org/abs/1506.02216)（2015），Figure 3。_

</div>

$$
\mathbf{h}_{t} = \text{RNN}(\mathbf{z}_{t}, \mathbf{a}_{t}, \mathbf{h}_{t-1})
$$

此时，上述转移概率可近似被化简为仅依赖于当前隐藏状态的条件分布：

$$
P(\mathbf{z}_{t+1} \mid \mathbf{z}_{\le t}, \mathbf{a}_{\le t}) \approx P(\mathbf{z}_{t+1} \mid \mathbf{h}_{t})
$$

接下来需要为 $P(\mathbf{z}_{t+1} \mid \mathbf{h}_{t})$ 选择合适的分布族。

若用 MSE 直接预测 $\mathbf{z}_{t+1}$，在概率解释下相当于采用固定方差的单峰高斯似然并学习其均值。面对多个可能未来，这个假设可能产生折中的平均预测。

::: info 理论抽象（仅有的一处类比）
我们可以将这种多模态预测，视作一位在十字路口观察的向导（即RNN的隐藏状态 $\mathbf{h}_t$）。向导知道车辆可能会向左转（概率 $\pi_1$，目标分布均值 $\mu_1$，路线不确定度 $\sigma_1$），也可能会向右转（概率 $\pi_2$），但他绝不会建议车辆“直直地撞向正前方的隔离墩”（这是两个不同决策所对应高斯分布的简单平均）。这种用多个带权重的正态分布来严密拼接、包络未知世界未来可能性的方式，正是**混合密度网络（MDN, [[Bishop, 1994]](https://www.microsoft.com/en-us/research/publication/mixture-density-networks/)）**的本质。
:::

<div align="center">
  <img src="/figures/04-latent-dynamics/source/01-world-models/wm-fig23.png" alt="World Models 附录将 RNN 时间展开与每步高斯混合输出并列，直接呈现 MDN-RNN 如何为下一潜变量给出多模态分布。" width="86%">

_图 4.1-4：World Models 附录将 RNN 时间展开与每步高斯混合输出并列，直接呈现 MDN-RNN 如何为下一潜变量给出多模态分布。 出处：David Ha；Jürgen Schmidhuber，[World Models](https://arxiv.org/abs/1803.10122)（2018），Figure 23。_

</div>

先写出一维高斯密度：

$$
\mathcal{N}(z \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(z - \mu)^2}{2\sigma^2} \right)
$$

为了具备表达多模态（多种不同未来可能性）的能力，我们引入由 $K$ 个高斯分量叠加而成的高斯混合模型（Gaussian Mixture Model, GMM）：

$$
P(z) = \sum_{k=1}^K \pi_k \mathcal{N}(z \mid \mu_k, \sigma_k^2)
$$

其中混合权重满足 $\sum_{k=1}^K \pi_k = 1$ 且 $\pi_k \ge 0$。

下面采用一个共享混合分量、分量内对角协方差的简化 MDN。给定分量 $k$ 时，各潜变量维度条件独立，因此联合密度可写成一维密度的乘积。需要注意，原始 World Models 的 MDN-RNN 实现细节与这个教学版参数化并不完全相同。

$$
P(\mathbf{z}_{t+1} \mid \mathbf{h}_t) = \sum_{k=1}^K \pi_{k, t} \prod_{i=1}^d \mathcal{N}(z_{t+1,i} \mid \mu_{k,i,t}, \sigma_{k,i,t}^2)
$$

混合权重 $\boldsymbol{\pi}_t$、均值 $\boldsymbol{\mu}_t$ 和标准差 $\boldsymbol{\sigma}_t$ 都由当前 RNN 隐藏状态 $\mathbf{h}_t$ 经输出层得到：

$$
[\boldsymbol{\hat{\pi}}_t, \boldsymbol{\hat{\mu}}_t, \boldsymbol{\hat{\sigma}}_t] = W_o \mathbf{h}_t + \mathbf{b}_o
$$

Softmax 使 $\pi$ 非负且和为 1，Softplus 或指数函数使标准差为正。训练时最小化负对数似然（Negative Log-Likelihood, NLL）：

$$
\mathcal{L}_{\text{MDN}} = \mathbb{E}_{t} \left[ -\log \left( \sum_{k=1}^K \pi_{k,t} \prod_{i=1}^d \frac{1}{\sqrt{2\pi\sigma_{k,i,t}^2}} \exp\left(-\frac{(z_{t+1,i} - \mu_{k,i,t})^2}{2\sigma_{k,i,t}^2}\right) \right) \right]
$$

<div align="center"><img src="/figures/04-latent-dynamics/latex/01-world-models/mdn-log-reduction-order.png" alt="MDN 先沿潜变量维累加对数高斯密度，再加入混合权重并沿混合分量执行 log-sum-exp" width="86%">

_图 4.1-5：分量内的维度乘积在对数域先化为求和；加入各分量的 log 权重后，才在混合分量轴上执行 log-sum-exp。本文根据上式绘制。_

</div>

## 架构与核心代码实现：MDN-RNN

下面用 PyTorch 构建教学版 `MDNRNN`。

设潜变量维度 $d=32$、RNN 隐藏维度 $h=256$、高斯分量数 $K=5$。
该模块接收 $t$ 时刻的动作序列和潜变量序列，在内部推演 RNN 的隐状态，最终输出下一步潜在分布的参数。

先定义 MDN-RNN 的前向传播。

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
        # Softplus 保证标准差为正；下限避免退化为零方差
        sigma = F.softplus(self.fc_sigma(rnn_out)) + 1e-4
        sigma = sigma.view(batch_size, seq_len, self.num_gaussians, self.latent_dim)

        return pi, mu, sigma, hidden
```

得到 `pi`、`mu` 和 `sigma` 后，在对数域计算混合分布的负对数似然，以避免直接连乘小概率造成下溢。

损失函数如下。

```python
def mdn_loss(pi, mu, sigma, target_z):
    """
    计算教学版对角高斯混合模型的负对数似然。
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

本节从状态转移出发，介绍了 World Models 的两级表示：**VAE（V 模型）**压缩单帧观测，**MDN-RNN（M 模型）**根据历史潜变量和动作预测下一潜变量分布。教学代码使用共享混合分量的对角高斯，重点是展示张量形状与对数似然计算，不应视为原论文实现的逐行复刻。

下一节转向 RSSM，观察确定性记忆与随机状态如何在同一个序列模型中协同工作。
