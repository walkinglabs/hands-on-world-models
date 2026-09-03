# 5.1 随机视频预测：从单一未来到概率分布

> **本章导读**
>
> **讲什么：** 本章换一种交付形式：让世界模型直接生成我们可以观看的未来画面，并让动作控制画面如何继续。我们将从随机视频预测出发，经过视频词元化、自回归与扩散生成，再处理实时推理，最后组装一个受动作控制的视频小世界。
>
> **为什么单一的下一帧预测不够：** 同一辆车驶到路口，可以左转、右转或直行；如果模型只能输出一个未来，常会把几种可能平均成模糊画面。交互式世界还要求“换一个动作，未来也随之改变”，并且生成速度必须跟得上操作，因此随机性、动作条件和实时性缺一不可。
>
> **故事线：** `从确定性下一帧走向随机未来 → 把长视频压缩成词元 → 用自回归或扩散生成时空内容 → 用缓存降低逐帧开销 → 注入动作并检验未来是否真的可控`

## 本章总览

<div align="center">

<img src="/figures/05-interactive-video/latex/01-video-prediction-svg/chapter-overview.png" alt="第 5 章学习路线：从 SVG 视频预测到可控交互视频生成" width="100%">

_第 5 章学习路线：串联随机视频预测、时空词元、扩散模型、流式加速与动作控制。_

</div>

视频预测（Video Prediction）给模型一段历史画面，要求它生成接下来可能出现的图像序列。它既要延续可预测的运动，又要表达遮挡、碰撞和未观测因素带来的多种未来。

<div align="center">
<img src="/figures/05-interactive-video/source/01-video-prediction-svg/svg-fig3.png" alt="同一段弹跳数字历史产生清晰但不同的后续轨迹，直观呈现随机视频模型对多种未来的采样。" width="86%">

_图 5.1-1：同一段弹跳数字历史产生清晰但不同的后续轨迹，直观呈现随机视频模型对多种未来的采样。 出处：Emily Denton；Rob Fergus，[Stochastic Video Generation with a Learned Prior](https://arxiv.org/abs/1802.07687)（2018），Figure 3。_
</div>

本节以随机视频生成模型（Stochastic Video Generation，SVG）[[Denton & Fergus, 2018]](https://arxiv.org/abs/1802.07687) 为主线，从单点预测过渡到逐时刻潜变量，并实现一个只保留先验、后验与循环状态的教学单元。

## 视频预测的历史脉络与挑战

长短期记忆网络（LSTM）用门控记忆单元改善循环网络中的长程梯度传播 [[Hochreiter & Schmidhuber, 1997]](https://doi.org/10.1162/neco.1997.9.8.1735)。ConvLSTM 又把输入到状态、状态到状态的变换改为卷积，并在降水临近预报任务上验证时空序列建模 [[Shi et al., 2015]](https://arxiv.org/abs/1506.04214)。这些结构后来被用于视频预测；但两篇原论文分别支撑循环记忆和卷积递归结构，不能单独证明某种架构是所有视频任务的“标准方案”。

<div align="center">
<img src="/figures/05-interactive-video/source/01-video-prediction-svg/convlstm-fig2.png" alt="ConvLSTM 的卷积门控单元保留二维空间结构，说明视频递归模型如何在时间更新中处理局部邻域。" width="86%">

_图 5.1-2：ConvLSTM 的卷积门控单元保留二维空间结构，说明视频递归模型如何在时间更新中处理局部邻域。 出处：Xingjian Shi et al.，[Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214)（2015），Figure 2。_
</div>

单点预测与像素 MSE 组合时容易产生**模糊性（Blurriness）**。例如，滚动的杯子既可能向左跌落，也可能向右跌落；条件均值可能把两种结果叠成重影。模糊并不是所有确定性网络的必然结果，它取决于输出分布、损失函数和数据中的多模态程度。

<div align="center">
<img src="/figures/05-interactive-video/source/01-video-prediction-svg/sv2p-fig1.png" alt="确定性预测把多种方向平均成模糊形状，而随机样本保持单一清晰运动方向。" width="86%">

_图 5.1-3：确定性预测把多种方向平均成模糊形状，而随机样本保持单一清晰运动方向。 出处：Mohammad Babaeizadeh et al.，[Stochastic Variational Video Prediction](https://arxiv.org/abs/1710.11252)（2018），Figure 1。_
</div>

Denton 与 Fergus 提出的 SVG 在每个时间步引入随机潜变量，使同一历史条件能够采样出不同的未来。论文比较了固定先验和学习先验等变体，并用多样性与预测质量共同评估结果。

## 从运动学到概率生成模型

先用一个运动学例子理解确定趋势与未建模扰动的区别。

### 确定性系统

假设在时刻 $t-1$ 已知小球的位置 $s_{t-1}$ 和速度 $v_{t-1}$。在匀速近似下，下一时刻的位置为：

$$s_t = s_{t-1} + v_{t-1} \cdot \Delta t$$

在这里，$s_t$ 完全由过去的状态决定。我们可以将这种确定性的演化用一个更一般的函数 $f_{\theta}$ 来表示，其中 $\theta$ 是系统参数（如深度神经网络的权重）：

$$\mathbf{x}_t = f_{\theta}(\mathbf{x}_{1}, \mathbf{x}_{2}, \dots, \mathbf{x}_{t-1})$$

其中 $\mathbf{x}_t \in \mathbb{R}^{C \times H \times W}$ 表示时刻 $t$ 的图像帧张量，三个维度依次对应通道、高度和宽度。

### 引入随机隐变量

为了表示历史帧没有决定的因素，可以在每个时间步引入潜变量 $z_t\in\mathbb{R}^d$。在固定先验变体中它来自标准正态分布；在 SVG-LP 中，先验参数由历史状态预测，因此会随时间和上下文变化。

此时，生成时刻 $t$ 图像的过程就不再是一个固定的函数映射，而是从一个条件概率分布中进行采样：

$$\mathbf{x}_t \sim p_{\theta}(\mathbf{x}_t \mid \mathbf{x}_{<t}, z_t)$$

<div align="center">
<img src="/figures/05-interactive-video/latex/01-video-prediction-svg/stochastic-future-branching.png" alt="固定同一段历史并改变时刻 t 的随机潜变量样本，会从同一条件分布得到多个不同未来帧" width="86%">

_图 5.1-4：历史帧保持不变时，不同的 z_t 样本沿同一条件生成分布产生不同未来，从而避免把多种可能性压成一张平均帧。_
</div>

其中，$\mathbf{x}_{<t}$ 表示第 $1$ 帧到第 $t-1$ 帧的历史观测。给定同一段历史，改变 $z_t$ 就能得到不同的条件样本。

### 时序先验与后验分布

在视频生成中，不同时间步的随机扰动 $z_t$ 并不是完全孤立的。为了让网络学会如何合理地猜测未来的扰动，我们需要定义两个概率分布：

1. **先验分布（Prior Distribution） $p_{\psi}(z_t \mid \mathbf{x}_{<t})$**：在只看到历史帧 $\mathbf{x}_{<t}$ 的情况下，网络对时刻 $t$ 的扰动所作出的预测。
2. **近似后验（Approximate Posterior） $q_{\phi}(z_t \mid \mathbf{x}_{\leq t})$**：训练时额外看到目标帧 $\mathbf{x}_t$，据此推断有助于解释该帧的潜变量分布。它是学习到的近似分布，不是可观测的“真实扰动”。

在训练阶段，后验网络负责提取真实的隐变量以重建图像；而在推理（预测未来）阶段，由于我们不知道未来的 $\mathbf{x}_t$，我们只能依赖先验网络来采样 $z_t$。因此，训练的目标之一就是让先验分布尽可能地逼近后验分布。

## 变分下界与损失函数

::: info 说明
可以把后验看成训练时拥有“答案线索”的老师：它同时读取历史帧和目标帧，为潜变量提供较有信息量的分布。先验只能读取历史帧。KL 项让两者靠近，使测试时没有目标帧可看时，模型仍能从先验采样。
:::

<div align="center">
<img src="/figures/05-interactive-video/source/01-video-prediction-svg/svg-fig2.png" alt="SVG-LP 的训练与生成图分开标出后验、学习先验和逐帧预测器，直接对应训练看未来、测试只看历史的差别。" width="86%">

_图 5.1-5：SVG-LP 的训练与生成图分开标出后验、学习先验和逐帧预测器，直接对应训练看未来、测试只看历史的差别。 出处：Emily Denton；Rob Fergus，[Stochastic Video Generation with a Learned Prior](https://arxiv.org/abs/1802.07687)（2018），Figure 2。_
</div>

我们希望最大化视频序列 $\mathbf{x}_{1:T}$ 的边缘对数似然 $\log p_{\theta}(\mathbf{x}_{1:T})$。由于对高维隐变量积分通常不可解析，我们改为最大化证据下界（Evidence Lower Bound，ELBO）。下面给出简化的单步形式：

$$\mathcal{L}_t = \mathbb{E}_{q_{\phi}(z_t \mid \mathbf{x}_{\leq t})} \left[ \log p_{\theta}(\mathbf{x}_t \mid \mathbf{x}_{<t}, z_t) \right] - \beta D_{\text{KL}} \left( q_{\phi}(z_t \mid \mathbf{x}_{\leq t}) \,\|\, p_{\psi}(z_t \mid \mathbf{x}_{<t}) \right)$$

两项分别承担不同作用：

- 第一项：**重构似然（Reconstruction Likelihood）**。后验网络基于截至 $t$ 的信息采样 $z_t$，生成器再用它和历史状态预测 $\mathbf{x}_t$。固定方差高斯似然对应平方误差；拉普拉斯似然对应 L1，二者不能混为同一个分布假设。
- 第二项：**KL散度（Kullback-Leibler Divergence）**。它衡量了先验分布与后验分布之间的差异。$\beta$ 是一个超参数（借鉴了 $\beta$-VAE 的思想），用于调节模型在“记忆特定帧细节”与“泛化随机性”之间的平衡。

在序列的整体训练中，我们将每一个时间步的损失相加，即可得到完整的序列损失。

## 网络架构与张量流转

SVG网络（具体来说是其变体SVG-LP，Learned Prior）由四个核心组件构成。为了清晰展示高维张量的流转过程，我们假设批次大小为 $B$，图像序列长度为 $T$，通道数为 $C$，高宽均为 $H, W$。

1. **帧编码器（Frame Encoder）**：通常是一个卷积神经网络（CNN）。输入单帧图像 $\mathbf{x}_t \in \mathbb{R}^{B \times C \times H \times W}$，输出低维空间特征表达 $h_t \in \mathbb{R}^{B \times d_h}$。
2. **循环核心（Recurrent Core）**：通常是LSTM单元。它接收过去的特征，维护一个隐藏状态变量 $\mathbf{c}_t \in \mathbb{R}^{B \times d_c}$，代表了对确定性历史的编码。
3. **推断模块（Inference Module）**：由多层感知机（MLP）构成。先验网络接受历史隐状态 $\mathbf{c}_{t-1}$ 输出先验的高斯分布参数 $(\mu_{p}, \sigma_{p})$；后验网络同时接受 $\mathbf{c}_{t-1}$ 和当前帧特征 $h_t$，输出后验参数 $(\mu_{q}, \sigma_{q})$。两者输出的均值和方差张量维度均为 $\mathbb{R}^{B \times d_z}$。
4. **帧预测器与解码器**：循环预测器结合前一帧特征、潜变量与历史状态，再由解码器映射回像素空间，得到 $\hat{\mathbf{x}}_t \in \mathbb{R}^{B \times C \times H \times W}$。原论文还使用编码器—解码器和跳跃连接；下面的代码不复现这些视觉模块。

下面用 PyTorch 写出 SVG 核心逻辑，重点放在潜变量采样和循环状态更新。为保持代码简洁，省略卷积编解码器、跳跃连接和完整损失，因此这是教学骨架而不是论文复现。

```python
import torch
from torch import nn

class SVGCell(nn.Module):
    def __init__(self, dim_h, dim_z, dim_c):
        """
        初始化SVG的时间步单元
        参数:
        dim_h: 图像特征编码维度
        dim_z: 随机隐变量维度
        dim_c: LSTM隐藏状态维度
        """
        super(SVGCell, self).__init__()
        self.dim_z = dim_z

        # 确定性动力学核心：LSTM
        # 输入维度为：历史特征(dim_h) + 隐变量(dim_z)
        self.lstm = nn.LSTMCell(dim_h + dim_z, dim_c)

        # 先验网络: p(z_t | x_<t)
        # 仅依赖LSTM的历史状态输出
        self.prior_net = nn.Sequential(
            nn.Linear(dim_c, 128),
            nn.ReLU(),
            nn.Linear(128, dim_z * 2) # 输出均值和对数方差
        )

        # 后验网络: q(z_t | x_<=t)
        # 依赖LSTM历史状态与当前帧特征
        self.posterior_net = nn.Sequential(
            nn.Linear(dim_c + dim_h, 128),
            nn.ReLU(),
            nn.Linear(128, dim_z * 2)
        )

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧 (Reparameterization Trick)
        从 N(mu, sigma^2) 中采样等价于 mu + sigma * epsilon, epsilon ~ N(0, 1)
        这使得梯度可以反向传播通过采样节点
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, h_t, h_t_minus_1, hidden_state):
        """
        前向传播一个时间步
        参数:
        h_t: 当前时刻真实帧的编码特征 [B, dim_h] (仅训练时可用)
        h_t_minus_1: 上一时刻帧的编码特征 [B, dim_h]
        hidden_state: LSTM的隐状态元组 (h_lstm, c_lstm)
        """
        h_lstm, c_lstm = hidden_state

        # 1. 计算先验分布参数
        prior_out = self.prior_net(h_lstm)
        mu_p, logvar_p = torch.split(prior_out, self.dim_z, dim=1)

        # 2. 计算后验分布参数 (仅在训练阶段有意义)
        post_input = torch.cat([h_lstm, h_t], dim=1)
        post_out = self.posterior_net(post_input)
        mu_q, logvar_q = torch.split(post_out, self.dim_z, dim=1)

        # 3. 从后验分布中采样 z_t
        z_t = self.reparameterize(mu_q, logvar_q)

        # 4. 更新确定性状态
        # 结合上一步的图像特征与当前步的扰动送入LSTM
        lstm_input = torch.cat([h_t_minus_1, z_t], dim=1)
        next_hidden_state = self.lstm(lstm_input, hidden_state)

        return z_t, mu_p, logvar_p, mu_q, logvar_q, next_hidden_state
```

这里使用**重参数化技巧（Reparameterization Trick）** [[Kingma & Welling, 2013]](https://arxiv.org/abs/1312.6114)。若直接把 $z\sim\mathcal{N}(\mu,\sigma^2)$ 当作随机采样节点，普通反向传播不能得到样本相对 $\mu,\sigma$ 的路径导数。把它改写为 $z=\mu+\sigma\epsilon$、$\epsilon\sim\mathcal{N}(0,1)$ 后，随机性被移到与参数无关的噪声变量上，梯度便可沿确定性计算路径传播。
