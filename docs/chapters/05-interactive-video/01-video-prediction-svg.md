# 视频预测网络基础与随机视频生成
:label:sec_video_prediction_svg

预测未来是智能体理解世界的基石。对于人类而言，当我们抛出一枚硬币时，即使硬币还在空中翻滚，我们的大脑已经能够粗略预测它下落的轨迹与最终触地的瞬间。在深度学习中，这种“预测未来”的能力往往被抽象为视频预测（Video Prediction）任务：给定一系列历史图像帧，模型需要生成未来可能出现的图像序列。

在这节课中，我们将深入探讨视频预测的理论基础，并详细剖析随机视频生成网络（Stochastic Video Generation, SVG）`[Denton & Fergus, 2018]`的设计哲学。我们将从最直观的物理运动学出发，逐步推导到高维张量的概率生成模型。

## 视频预测的历史脉络与挑战
:label:subsec_video_prediction_history

早期的视频预测模型大多基于确定性（Deterministic）假设。研究者们通常采用循环神经网络（RNN）及其变体，如长短期记忆网络（LSTM）`[Hochreiter & Schmidhuber, 1997]`。当图像的二维空间结构需要被保留时，卷积长短期记忆网络（ConvLSTM）`[Xingjian et al., 2015]`成为了标准的解决方案。在当时的硬件计算力限制下，确定性模型通过最小化预测帧与真实未来帧之间的均方误差（Mean Squared Error, MSE）来优化网络权重。

然而，确定性模型很快遇到了一个难以逾越的瓶颈：**模糊性（Blurriness）**。真实世界充满了随机性与不可控的扰动。假设桌子上有一个正在滚动的玻璃杯，它可能向左掉落，也可能向右掉落。如果模型只能给出一个确定性的预测，为了最小化MSE，网络会倾向于输出所有可能结果的平均值——即同时向左和向右掉落的重影。这种“平均化”策略导致了生成的未来帧随着时间推移变得越来越模糊，丧失了所有的纹理细节。

为了解决这一问题，Denton与Fergus在2018年提出了随机视频生成网络（SVG）。他们引入了随时间变化的隐变量（Latent Variable），将视频预测从一个“确定性回归问题”转变为了一个“概率分布采样问题”，从而使得模型能够生成清晰且具有多样性的未来帧。

## 从运动学到概率生成模型
:label:subsec_svg_kinematics_to_prob

为了理解SVG的数学机制，我们不妨先回到高中物理中的运动学。

### 确定性系统
假设我们在时刻 $t-1$ 观察到一个小球的位置 $s_{t-1}$ 和速度 $v_{t-1}$。如果不考虑空气阻力等任何随机因素，下一时刻 $t$ 的位置可以被精确计算：

$$s_t = s_{t-1} + v_{t-1} \cdot \Delta t$$
:eqlabel:eq_deterministic_motion

在这里，$s_t$ 完全由过去的状态决定。我们可以将这种确定性的演化用一个更一般的函数 $f_{\theta}$ 来表示，其中 $\theta$ 是系统参数（如深度神经网络的权重）：

$$\mathbf{x}_t = f_{\theta}(\mathbf{x}_{1}, \mathbf{x}_{2}, \dots, \mathbf{x}_{t-1})$$
:eqlabel:eq_deterministic_nn

在公式 :eqref:eq_deterministic_nn 中，$\mathbf{x}_t \in \mathbb{R}^{C \times H \times W}$ 表示时刻 $t$ 的图像帧张量，包含通道数、高度和宽度。

### 引入随机隐变量
如前文所述，完美的确定性是不存在的。为了模拟世界的不确定性，我们在每一个时间步 $t$ 引入一个服从标准正态分布的高斯噪声，称之为隐变量 $z_t \in \mathbb{R}^d$。这个隐变量 $z_t$ 就像是一阵随机吹来的微风，或者是不可观测的微小扰动。

此时，生成时刻 $t$ 图像的过程就不再是一个固定的函数映射，而是从一个条件概率分布中进行采样：

$$\mathbf{x}_t \sim p_{\theta}(\mathbf{x}_t \mid \mathbf{x}_{<t}, z_t)$$
:eqlabel:eq_stochastic_generation

其中，$\mathbf{x}_{<t}$ 表示从第 $1$ 帧到第 $t-1$ 帧的所有历史观测。公式 :eqref:eq_stochastic_generation 深刻地揭示了SVG模型的核心思想：未来是由确定的历史 $\mathbf{x}_{<t}$ 与随机的扰动 $z_t$ 共同决定的。

### 时序先验与后验分布
在视频生成中，不同时间步的随机扰动 $z_t$ 并不是完全孤立的。为了让网络学会如何合理地猜测未来的扰动，我们需要定义两个概率分布：

1. **先验分布（Prior Distribution） $p_{\psi}(z_t \mid \mathbf{x}_{<t})$**：在只看到历史帧 $\mathbf{x}_{<t}$ 的情况下，网络对时刻 $t$ 的扰动所作出的预测。
2. **后验分布（Posterior Distribution） $q_{\phi}(z_t \mid \mathbf{x}_{\leq t})$**：在已经看到（或称为“作弊”看到）了目标帧 $\mathbf{x}_t$ 的情况下，网络推断出的导致这一结果的真实扰动。

在训练阶段，后验网络负责提取真实的隐变量以重建图像；而在推理（预测未来）阶段，由于我们不知道未来的 $\mathbf{x}_t$，我们只能依赖先验网络来采样 $z_t$。因此，训练的目标之一就是让先验分布尽可能地逼近后验分布。

## 变分下界与损失函数
:label:subsec_svg_elbo

> [!NOTE]
> 我们在这里使用一个教育学的类比来帮助理解变分推断的核心思想：假设你是一位考古学家（先验网络 $p_{\psi}$），试图根据古代遗迹（历史帧 $\mathbf{x}_{<t}$）预测明天会发掘出什么文物（分布 $z_t$）。而你的导师（后验网络 $q_{\phi}$）已经看到了明天发掘出的文物照片（当前帧 $\mathbf{x}_t$），因此导师能给出极其准确的判断。在长期的训练中，你要尽可能让自己的预测（先验）与导师的判断（后验）保持一致（即最小化KL散度），从而在未来导师不在（没有目标帧）时，你也能做出合理的预测。

我们希望最大化观察到真实视频序列 $\mathbf{x}_{1:T}$ 的边缘对数似然 $\log p_{\theta}(\mathbf{x}_{1:T})$。由于直接计算包含了隐变量 $z_{1:T}$ 的积分是不可解的，我们转向最大化其证据下界（Evidence Lower Bound, ELBO）。对于单一时间步 $t$，变分下界可以拆解为重构项与正则化项：

$$\mathcal{L}_t = \mathbb{E}_{q_{\phi}(z_t \mid \mathbf{x}_{\leq t})} \left[ \log p_{\theta}(\mathbf{x}_t \mid \mathbf{x}_{<t}, z_t) \right] - \beta D_{\text{KL}} \left( q_{\phi}(z_t \mid \mathbf{x}_{\leq t}) \,\|\, p_{\psi}(z_t \mid \mathbf{x}_{<t}) \right)$$
:eqlabel:eq_svg_loss_step

让我们极其严谨地拆解公式 :eqref:eq_svg_loss_step 中的每一项：
- 第一项：**重构似然（Reconstruction Likelihood）**。后验网络 $q_{\phi}$ 基于直到当前时刻 $t$ 的所有信息提取隐特征 $z_t$，生成器 $p_{\theta}$ 用它和历史状态还原图像 $\mathbf{x}_t$。由于我们通常假设像素服从高斯分布，最大化对数似然等价于最小化均方误差（MSE）或平均绝对误差（L1 Loss）。
- 第二项：**KL散度（Kullback-Leibler Divergence）**。它衡量了先验分布与后验分布之间的差异。$\beta$ 是一个超参数（借鉴了 $\beta$-VAE 的思想），用于调节模型在“记忆特定帧细节”与“泛化随机性”之间的平衡。

在序列的整体训练中，我们将每一个时间步的损失相加，即可得到完整的序列损失。

## 网络架构与张量流转
:label:subsec_svg_architecture

SVG网络（具体来说是其变体SVG-LP，Learned Prior）由四个核心组件构成。为了清晰展示高维张量的流转过程，我们假设批次大小为 $B$，图像序列长度为 $T$，通道数为 $C$，高宽均为 $H, W$。

1. **帧编码器（Frame Encoder）**：通常是一个卷积神经网络（CNN）。输入单帧图像 $\mathbf{x}_t \in \mathbb{R}^{B \times C \times H \times W}$，输出低维空间特征表达 $h_t \in \mathbb{R}^{B \times d_h}$。
2. **循环核心（Recurrent Core）**：通常是LSTM单元。它接收过去的特征，维护一个隐藏状态变量 $\mathbf{c}_t \in \mathbb{R}^{B \times d_c}$，代表了对确定性历史的编码。
3. **推断模块（Inference Module）**：由多层感知机（MLP）构成。先验网络接受历史隐状态 $\mathbf{c}_{t-1}$ 输出先验的高斯分布参数 $(\mu_{p}, \sigma_{p})$；后验网络同时接受 $\mathbf{c}_{t-1}$ 和当前帧特征 $h_t$，输出后验参数 $(\mu_{q}, \sigma_{q})$。两者输出的均值和方差张量维度均为 $\mathbb{R}^{B \times d_z}$。
4. **帧解码器（Frame Decoder）**：通过转置卷积（Transposed CNN）将隐藏特征 $h_{t-1}$ 与采样的隐变量 $z_t$ 进行拼接后，映射回高维像素空间，得到预测帧 $\hat{\mathbf{x}}_t \in \mathbb{R}^{B \times C \times H \times W}$。

(**定义SVG推断网络与生成核心的伪实现**)

下面我们展示在PyTorch中实现SVG核心逻辑的代码。我们将重点放在隐变量的采样与KL散度的计算上。为了保持代码简洁，我们省略了卷积层编解码器的具体定义，聚焦于时序动态变化。

```{.python .input}
#@tab pytorch
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

```{.python .input}
#@tab tensorflow
import tensorflow as tf

class SVGCell(tf.keras.layers.Layer):
    def __init__(self, dim_h, dim_z, dim_c):
        super(SVGCell, self).__init__()
        self.dim_z = dim_z
        self.lstm = tf.keras.layers.LSTMCell(dim_c)
        
        self.prior_net = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(dim_z * 2)
        ])
        
        self.posterior_net = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(dim_z * 2)
        ])

    def reparameterize(self, mu, logvar):
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=tf.shape(std))
        return mu + eps * std

    def call(self, h_t, h_t_minus_1, hidden_state):
        h_lstm, c_lstm = hidden_state
        
        prior_out = self.prior_net(h_lstm)
        mu_p, logvar_p = tf.split(prior_out, num_or_size_splits=2, axis=1)
        
        post_input = tf.concat([h_lstm, h_t], axis=1)
        post_out = self.posterior_net(post_input)
        mu_q, logvar_q = tf.split(post_out, num_or_size_splits=2, axis=1)
        
        z_t = self.reparameterize(mu_q, logvar_q)
        
        lstm_input = tf.concat([h_t_minus_1, z_t], axis=1)
        _, next_hidden_state = self.lstm(lstm_input, states=hidden_state)
        
        return z_t, mu_p, logvar_p, mu_q, logvar_q, next_hidden_state
```

在这段代码中，最关键的一步是**重参数化技巧（Reparameterization Trick）**`[Kingma & Welling, 2013]`。直接从高斯分布 $\mathcal{N}(\mu, \sigma^2)$ 中采样是一个不可导的操作，这会阻断神经网络基于梯度下降的端到端反向传播。通过将其改写为可导的确定性路径与不可导的常数噪声源 $\epsilon \sim \mathcal{N}(0, 1)$ 相结合，我们精巧地绕过了这个数学障碍。

## 练习

1. 在公式 :eqref:eq_svg_loss_step 中，我们使用了高斯分布假设。如果先验 $p_{\psi}$ 和后验 $q_{\phi}$ 都是多变量对角高斯分布，请查阅相关数学知识，写出 $D_{\text{KL}}(q \| p)$ 的解析解表达式。
   - *提示*：对于两个高斯分布 $\mathcal{N}_0(\mu_0, \sigma_0^2)$ 和 $\mathcal{N}_1(\mu_1, \sigma_1^2)$，KL散度公式为 $\frac{1}{2} \left[ \log \frac{\sigma_1^2}{\sigma_0^2} + \frac{\sigma_0^2 + (\mu_0 - \mu_1)^2}{\sigma_1^2} - 1 \right]$。
2. 为什么在训练阶段我们需要向 LSTM 输入上一帧的特征 $h_{t-1}$ 而非当前帧 $h_t$？
   - *提示*：考虑时间序列预测任务中严格的因果关系，如果在时刻 $t$ 的状态更新中使用了时刻 $t$ 的信息，是否构成了“信息泄露（Information Leakage）”？
3. 如果我们将 KL 散度前面的权重系数 $\beta$ 设置得非常大（例如 $\beta=1000$），会对生成的视频产生什么影响？
   - *提示*：回忆 $\beta$ 的作用。当网络被迫使后验分布无条件地紧贴先验分布（通常是标准正态分布）时，隐变量 $z_t$ 还能否保留当前帧的有效结构信息？这与“后验崩塌（Posterior Collapse）”现象有何关联？
4. 修改上述 PyTorch 或 TensorFlow 代码中的 `SVGCell` 类，使其在推理（Inference）模式下运行。在推理时，由于没有未来帧的真实观测 $h_t$，代码逻辑应当如何调整？
   - *提示*：隐变量 $z_t$ 需要直接从先验分布计算得到的 $\mu_p$ 和 $\log\sigma_p$ 中进行重参数化采样。

:begin_tab:pytorch
[讨论](https://discuss.d2l.ai/t/5001)
:end_tab:
