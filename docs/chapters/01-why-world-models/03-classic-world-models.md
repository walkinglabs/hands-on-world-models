# 1.3 经典世界模型理论回顾

我们所处的世界是高维、充满噪声且遵循着复杂物理规律的。人类之所以能够在这样的世界中迅速学会驾驶汽车、打网球甚至在月球上行走，是因为我们在大脑中构建了一个抽象的“世界模型”。在这个内部模型中，我们能够预测动作的结果，并在想象中进行规划。

在本节中，我们将严格追溯这一思想在深度学习领域的工程实现。我们将从学术脉络出发，逐步推导 Ha 和 Schmidhuber 于 2018 年提出的经典世界模型架构 [[Ha & Schmidhuber, 2018]](https://arxiv.org/abs/1803.10122)。我们会深入剖析该架构的核心数学原理，将复杂的张量运算拆解为高中物理和基础统计学的直观概念，并最终在代码层面构建这一经典系统。

## 1.3.1 学术脉络与时代背景

利用内部模型来进行学习和预测的思想，并非深度学习时代的产物。早在 1990 年，强化学习领域的先驱 Richard Sutton 就提出了著名的 Dyna 架构 [[Sutton, 1990]](https://dl.acm.org/doi/10.5555/645530.658292)。Dyna 架构的核心理念是：智能体不仅应该通过与真实环境的交互来学习（无模型强化学习），还应该利用已经学到的经验来构建一个关于环境的转移模型（Model），进而在这个“虚拟环境”中进行规划和策略更新。

然而，在 20 世纪 90 年代，受限于计算能力和算法的发展，早期的基于模型的强化学习（Model-Based RL）只能处理离散状态或低维度的简单连续状态（例如几个坐标和速度值）。当面对真实世界中诸如摄像头捕捉到的高维像素图像时，传统的马尔可夫决策过程（MDP）转移模型便束手无策。因为预测下一张 1024 $\times$ 768 的彩色图像，其维度爆炸和像素间的复杂非线性依赖，使得直接对图像级动力学进行建模成为不可能完成的任务。

这一僵局直到变分自编码器（VAE）和循环神经网络（RNN）成熟后才被打破。2018年，David Ha 和 Jürgen Schmidhuber 发表了标志性的论文 *World Models*。他们巧妙地将复杂的环境建模拆解为三个独立但高度协同的模块：
1. **视觉模型（V模型）**：将高维像素压缩为低维潜空间（Latent Space）的概率表示。
2. **记忆与动力学模型（M模型）**：在低维潜空间中，基于历史状态和动作，对未来的状态进行概率预测。
3. **控制器模型（C模型）**：仅接收低维状态特征，负责输出动作以最大化累计回报。

这种解耦不仅极大地降低了动力学建模的难度，还赋予了智能体在“梦境”（闭环想象）中训练自身策略的能力。接下来，我们将沿着数据流动的方向，逐一拆解这些模块的数学本质。

## 1.3.2 V模型：从高维观测到低维潜空间表示

考虑一个简单的物理场景：一辆汽车在一条直线上行驶。要描述这辆汽车的瞬时位置，我们只需要一个标量坐标 $z \in \mathbb{R}$。然而，自动驾驶系统接收到的观测数据 $x$，却是包含汽车、树木、天空和道路的数万个像素点构成的图像。V模型的作用，就是找到一个映射关系，将这数万维的冗余图像 $x$ 重新映射回能够刻画系统本质属性的低维状态 $z$。

如果环境是完全确定的，且传感器没有噪声，我们可以简单地寻找一个确定的函数映射 $z = f(x)$。但真实世界充满了不确定性。当我们仅根据当前模糊或带有噪声的一帧图像 $x$ 提取状态时，我们对“汽车究竟在哪里”的估计不应当是一个确定的点，而应当是一个概率分布。

在高中数学中，描述一个带有误差的连续变量的最常见方式是高斯分布（正态分布）。因此，在标量情况下，我们假设状态 $z$ 服从以 $\mu$ 为均值，$\sigma^2$ 为方差的正态分布：
$$
p(z | x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{(z - \mu)^2}{2\sigma^2} \right)
$$

在这个公式中，$\mu$ 代表了我们对汽车位置的**最可能估计**，而 $\sigma^2$ 代表了我们对这个估计的**不确定度**。

当状态不再是单一标量，而是包含位置、颜色、姿态等多个属性的 $D$ 维向量 $\mathbf{z} = [z_1, z_2, \dots, z_D]^\top$ 时，我们通常假设这些特征在潜空间中是相互独立的（即协方差矩阵为对角阵）。此时，我们需要通过一个深度卷积神经网络（CNN）构成的编码器来计算图像 $x$ 对应的均值向量 $\boldsymbol{\mu} \in \mathbb{R}^D$ 和方差向量 $\boldsymbol{\sigma}^2 \in \mathbb{R}^D$。

为了能够使用梯度下降法优化这个概率模型，我们需要从分布 $\mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ 中采样出一个具体的 $\mathbf{z}$ 传递给下游。然而，“采样”这一操作是随机的，不可导的。为此，V模型（即变分自编码器 VAE）采用了著名的**重参数化技巧（Reparameterization Trick）**。

在标量情形下，重参数化技巧可以直观地理解为：与其直接从均值为 $\mu$、标准差为 $\sigma$ 的分布中抽取 $z$，不如我们先从标准正态分布 $\mathcal{N}(0,1)$ 中抽取一个“标准随机噪声” $\epsilon$。然后，将这个标准噪声放大 $\sigma$ 倍（匹配我们预估的不确定度），再将其平移 $\mu$（匹配我们的预估中心）。其标量表达式为：
$$
z = \mu + \sigma \cdot \epsilon \quad \text{其中} \quad \epsilon \sim \mathcal{N}(0, 1)
$$

由于推导过程严丝合缝，我们可以自然地将其推广到 $D$ 维向量形式。此时 $\boldsymbol{\mu}$ 和 $\boldsymbol{\sigma}$ 为向量，$\odot$ 表示逐元素相乘：
$$
\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon} \quad \text{其中} \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

通过上述推导，V模型将高维图像成功转化为低维的、连续的且包含不确定性的状态向量 $\mathbf{z}$，为后续的时间序列建模提供了高质量的原材料。

## 1.3.3 M模型：混合密度网络与时间演化

拥有了低维表示 $\mathbf{z}_t$ 后，我们来到了世界模型中最核心、也最体现物理本质的部分：预测未来。

假设我们在时刻 $t$ 观察到了状态 $z_t$，并且执行了一个动作 $a_t$（例如踩下油门）。我们希望预测下一时刻的状态 $z_{t+1}$。如果这是经典的高中运动学问题，对于一个做匀加速直线的物体，我们有确定的位移公式。然而，智能体所处的环境往往是高度随机的：地面可能有摩擦力突变，侧面可能突然刮起大风。

### 从单一高斯到混合高斯 (MDN)
面对不确定性，我们第一反应是继续使用单一高斯分布来描述 $z_{t+1}$，即预测下一时刻均值 $\mu_{t+1}$ 和方差 $\sigma_{t+1}^2$。但这种假设在真实世界中会面临严重缺陷。

设想这样一个场景：汽车行驶到了一个没有直行道路的 T 字路口。面对前方的墙壁，如果此时汽车的方向盘是回正的（直行），它在下一时刻将不可避免地向左滑动或向右滑动（概率各占 50%）。
如果我们强制使用单一高斯分布来拟合这种未来，网络会计算出“向左滑动”和“向右滑动”的平均值——结果就是预测汽车将**直接穿墙而过（直行）**。这在物理上是荒谬的。

为了解决这种**多模态（Multimodal）**的预测问题，我们需要引入混合密度网络（Mixture Density Network, MDN）。MDN 的核心思想是：未来的可能性不是由一个高斯钟形曲线描述的，而是由多个不同形状的高斯曲线叠加而成的。

对于标量预测，具有 $K$ 个混合成分的概率密度函数表示为：
$$
P(z_{t+1} | z_t, a_t) = \sum_{k=1}^{K} \pi_{k} \mathcal{N}(z_{t+1} | \mu_k, \sigma_k^2)
$$

在这里，每个 $k$ 代表一种可能的“宏观物理演化路径”（例如 $k=1$ 代表向左侧滑，$k=2$ 代表向右侧滑）。
- $\pi_k$ 是**混合权重**，代表了第 $k$ 种路径发生的概率，且必须满足 $\sum_{k=1}^K \pi_k = 1$。
- $\mu_k$ 是第 $k$ 种路径下预测的具体均值。
- $\sigma_k^2$ 是该路径下方差（微观物理不确定性）。

### 引入循环神经网络 (RNN)
单独根据当前的一帧状态 $z_t$ 来预测 $z_{t+1}$ 是不够的。物理学告诉我们，要知道物体的下一时刻位置，我们不仅需要当前的位置，还需要当前的速度、加速度等高阶信息。由于 $z_t$ 仅是单帧图像的编码，它不包含速度。

为了恢复动力学信息，我们需要在时间序列上引入历史记忆。RNN（或其变体 LSTM）就是用来累积这段历史的。在时刻 $t$，RNN 维护着一个隐藏状态 $\mathbf{h}_t$，它压缩了从 $0$ 到 $t$ 时刻的所有历史观察和动作序列。

我们将 RNN 与 MDN 结合，就得到了 M 模型的完整动力学方程：
首先，通过 RNN 更新记忆：
$$
\mathbf{h}_t = f_{\text{RNN}}(\mathbf{h}_{t-1}, [\mathbf{z}_t, \mathbf{a}_t])
$$

随后，使用简单的线性映射层，从当前的隐藏状态 $\mathbf{h}_t$ 中提取出多元混合高斯分布所需的参数（对于维度为 $D$ 的状态，且有 $K$ 个混合成分）：
$$
\boldsymbol{\pi}_t, \boldsymbol{\mu}_t, \boldsymbol{\sigma}_t = \text{Linear}(\mathbf{h}_t)
$$

这里的预测概率密度扩展到多维向量空间，在各维度独立的假设下，表示为：
$$
P(\mathbf{z}_{t+1} | \mathbf{a}_t, \mathbf{z}_t, \mathbf{h}_t) = \sum_{k=1}^{K} \pi_{t}^{(k)} \prod_{d=1}^{D} \mathcal{N} \left( z_{t+1}^{(d)} | \mu_{t}^{(k, d)}, \left( \sigma_{t}^{(k, d)} \right)^2 \right)
$$

这就构成了经典世界模型中最关键的时间演化引擎。

## 1.3.4 C模型：基于内部想象的决策与训练

当 V 模型提供了状态的抽象压缩，M 模型掌握了世界演化的规律后，控制策略的任务就变得惊人地简单。在 Ha 和 Schmidhuber 的设计中，C 模型仅仅是一个极简的单层线性网络（后接激活函数）：
$$
\mathbf{a}_t = \text{tanh} \left( \mathbf{W}_c [\mathbf{z}_t, \mathbf{h}_t] + \mathbf{b}_c \right)
$$

为什么在如此复杂的自动驾驶或游戏中，控制器可以设计得如此简陋？因为感知（V）和预测（M）这两个最消耗算力和参数量的任务已经被前置解决。控制器的输入不再是杂乱无章的像素，而是经过高度结构化的当前物理状态 $\mathbf{z}_t$ 和历史动态趋势 $\mathbf{h}_t$。

更为重要的是，世界模型赋予了智能体“脱离真实环境进行学习”的能力。

> 在此，我们引入本章唯一的一个类比来帮助理解“内部世界想象”与“强化学习”的结合：
> 想象一位即将在明天参加跨栏决赛的田径运动员。在比赛前夜，她闭上眼睛，在脑海中（潜空间）模拟起跑、奔跑、起跳的完整过程（M模型的时间展开）。在脑海中，她不需要真实地消耗体力或承担受伤的风险（无真实环境采样），但依然能够根据脑海中预判的跨栏位置（预测的未来状态）来调整自己的发力策略（C模型的参数更新）。这种在“梦境”中进行的训练，正是模型能够极大地减少真实环境交互次数的核心原因。

由于 M 模型能够根据动作 $a_t$ 生成逼真的下一状态 $z_{t+1}$，我们可以让 C 模型完全在 M 模型的展开序列中进行训练。这种无需与真实环境反复交互的范式，极大地推动了样本效率（Sample Efficiency）的提升。

## 1.3.5 代码实现：构建经典的M模型 (MDN-RNN)

基于我们在相关章节中的理论推导，(**我们将构建一个MDN-RNN模块**)。在这个模块中，我们将明确标注所有的张量形状，特别是当融合了批次（Batch Size）、混合成分（Num Mixtures）和潜空间维度（Z Dim）时带来的多维张量变换。

```{.python .input}
#@tab pytorch
import torch
from torch import nn
from torch.nn import functional as F

class MDNRNN(nn.Module):
    def __init__(self, z_dim, action_dim, hidden_dim, num_mixtures):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_mixtures = num_mixtures
        self.z_dim = z_dim
        
        # RNN 单元接收当前时刻的状态和动作
        # 输入维度: z_dim + action_dim
        self.rnn = nn.LSTMCell(input_size=z_dim + action_dim, hidden_size=hidden_dim)
        
        # MDN 线性输出层
        # 输出 pi: 共有 num_mixtures 个权重
        self.fc_pi = nn.Linear(hidden_dim, num_mixtures)
        # 输出 mu: 每个混合成分需要预测 z_dim 维度的均值
        self.fc_mu = nn.Linear(hidden_dim, num_mixtures * z_dim)
        # 输出 sigma: 每个混合成分需要预测 z_dim 维度的标准差
        self.fc_sigma = nn.Linear(hidden_dim, num_mixtures * z_dim)

    def forward(self, z, action, hidden_state):
        """
        参数:
        z: (batch_size, z_dim) 当前时刻潜状态
        action: (batch_size, action_dim) 当前时刻动作
        hidden_state: (hx, cx) RNN的历史隐藏状态，两者形状均为 (batch_size, hidden_dim)
        """
        # 沿着特征维度拼接状态和动作
        # rnn_input 形状: (batch_size, z_dim + action_dim)
        rnn_input = torch.cat([z, action], dim=1)
        
        # 通过 LSTM 更新记忆
        hx, cx = self.rnn(rnn_input, hidden_state)
        
        # 计算混合权重，使用 Softmax 保证其和为 1
        # pi 形状: (batch_size, num_mixtures)
        pi = F.softmax(self.fc_pi(hx), dim=-1)
        
        # 计算均值并重塑张量形状，将混合成分与空间维度分离
        # mu 形状: (batch_size, num_mixtures, z_dim)
        mu = self.fc_mu(hx).view(-1, self.num_mixtures, self.z_dim)
        
        # 计算标准差，利用 exp 函数保证标准差始终为正数
        # sigma 形状: (batch_size, num_mixtures, z_dim)
        sigma = torch.exp(self.fc_sigma(hx)).view(-1, self.num_mixtures, self.z_dim)
        
        return pi, mu, sigma, (hx, cx)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
from tensorflow.keras import layers

class MDNRNN(tf.keras.Model):
    def __init__(self, z_dim, action_dim, hidden_dim, num_mixtures):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_mixtures = num_mixtures
        self.z_dim = z_dim
        
        # 定义 LSTM 单元
        self.rnn_cell = layers.LSTMCell(hidden_dim)
        
        # MDN 的线性输出层
        self.fc_pi = layers.Dense(num_mixtures)
        self.fc_mu = layers.Dense(num_mixtures * z_dim)
        self.fc_sigma = layers.Dense(num_mixtures * z_dim)

    def call(self, z, action, states):
        """
        参数:
        z: (batch_size, z_dim)
        action: (batch_size, action_dim)
        states: 包含 hx 和 cx 的列表
        """
        rnn_input = tf.concat([z, action], axis=-1)
        
        # 更新 LSTM 状态
        # hx 为新隐藏状态, new_states 包含传给下一步的记忆
        hx, new_states = self.rnn_cell(rnn_input, states)
        
        # pi 的计算
        pi = tf.nn.softmax(self.fc_pi(hx), axis=-1)
        
        # 动态获取 batch_size 并调整形状
        batch_size = tf.shape(hx)[0]
        mu = tf.reshape(self.fc_mu(hx), [batch_size, self.num_mixtures, self.z_dim])
        
        # 保证 sigma 为正数
        sigma = tf.exp(tf.reshape(self.fc_sigma(hx), [batch_size, self.num_mixtures, self.z_dim]))
        
        return pi, mu, sigma, new_states
```

上述代码精确再现了核心的推导逻辑，不仅处理了时序上的传递，更通过合理的张量重塑（Reshape/View）将多模态不确定性参数化。

## 1.3.6 练习
1. 在推导前文的向量形式混合密度公式时，我们假设了潜状态各个维度在给定条件下是相互独立的。如果这一假设不成立，我们应该如何修改协方差的表示？
   - *提示：思考高中统计中关于相关系数和协方差矩阵对角化的问题。*
2. 请结合张量维度解释，为什么 `fc_mu` 和 `fc_sigma` 的输出维度必须是 `num_mixtures * z_dim`？
   - *提示：如果一个物体在 3 维空间运动，且未来有 5 种不同的走向（模态），那么共需要多少个坐标值来完整描述这些走向的中心位置？*
3. 在代码实现中，我们计算标准差时使用了 `exp` 函数：`torch.exp(self.fc_sigma(hx))`。除了 `exp` 之外，还有什么操作既能保证网络输出大于零，又能提供更平稳的梯度？
   - *提示：可以参考 Softplus 函数。*
