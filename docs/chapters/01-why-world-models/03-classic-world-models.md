# 1.3 经典世界模型理论回顾

经典 _World Models_ 研究了一个具体问题：当观测是 CarRacing 或 VizDoom 的像素画面时，能否先把图像压缩，再在低维空间预测未来，最后只用一个很小的控制器完成任务？

本节沿数据流拆解 Ha 和 Schmidhuber 于 2018 年提出的架构 [[Ha & Schmidhuber, 2018]](https://arxiv.org/abs/1803.10122)：VAE 负责视觉压缩，MDN-RNN 负责潜空间动力学，线性控制器负责动作选择。

## 1.3.1 学术脉络与时代背景

利用内部模型来进行学习和预测的思想，并非深度学习时代的产物。早在 1990 年，强化学习领域的先驱 Richard Sutton 就提出了著名的 Dyna 架构 [[Sutton, 1990]](https://dl.acm.org/doi/10.5555/645530.658292)。Dyna 架构的核心理念是：智能体不仅应该通过与真实环境的交互来学习（无模型强化学习），还应该利用已经学到的经验来构建一个关于环境的转移模型（Model），进而在这个“虚拟环境”中进行规划和策略更新。

早期的基于模型强化学习（Model-Based RL）主要在离散状态或低维连续状态上验证。像素观测的维度更高，背景纹理、遮挡与运动又相互耦合，直接学习图像级动力学需要更多数据与计算。

2018 年，David Ha 和 Jürgen Schmidhuber 在 _World Models_ 中把视觉控制拆成三个模块：

1. **视觉模型（V模型）**：将高维像素压缩为低维潜空间（Latent Space）的概率表示。
2. **记忆与动力学模型（M模型）**：在低维潜空间中，基于历史状态和动作，对未来的状态进行概率预测。
3. **控制器模型（C模型）**：仅接收低维状态特征，负责输出动作以最大化累计回报。

<div align="center">
  <img src="/figures/01-why-world-models/source/03-classic-world-models/vae-fig1.png" alt="VAE 原论文的图模型区分生成路径与近似后验路径，为 V 模型的随机潜变量编码提供理论来源。" width="86%">

_图 1.3-1：VAE 原论文的图模型区分生成路径与近似后验路径，为 V 模型的随机潜变量编码提供理论来源。 出处：Diederik P. Kingma; Max Welling，[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)（2014），Figure 1。_

</div>

这种解耦把图像重建、时间预测和动作优化分开训练。论文在 VizDoom 中演示了在模型生成轨迹中训练控制器；CarRacing 控制器则是在真实游戏环境中优化，两项实验需要区分。

## 1.3.2 V模型：从高维观测到低维潜空间表示

考虑一个简单的物理场景：一辆汽车在一条直线上行驶。要描述这辆汽车的瞬时位置，我们只需要一个标量坐标 $z \in \mathbb{R}$。然而，自动驾驶系统接收到的观测数据 $x$，却是包含汽车、树木、天空和道路的数万个像素点构成的图像。V模型的作用，就是找到一个映射关系，将这数万维的冗余图像 $x$ 重新映射回能够刻画系统本质属性的低维状态 $z$。

如果环境是完全确定的，且传感器没有噪声，我们可以简单地寻找一个确定的函数映射 $z = f(x)$。但真实世界充满了不确定性。当我们仅根据当前模糊或带有噪声的一帧图像 $x$ 提取状态时，我们对“汽车究竟在哪里”的估计不应当是一个确定的点，而应当是一个概率分布。

在初等代数中，描述一个带有误差的连续变量的最常见方式是高斯分布（正态分布）。因此，在标量情况下，我们假设状态 $z$ 服从以 $\mu$ 为均值，$\sigma^2$ 为方差的正态分布：

$$
p(z | x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp \left( -\frac{(z - \mu)^2}{2\sigma^2} \right)
$$

在这个公式中，$\mu$ 代表了我们对汽车位置的**最可能估计**，而 $\sigma^2$ 代表了我们对这个估计的**不确定度**。

当潜变量是 $D$ 维向量 $\mathbf{z} = [z_1, z_2, \dots, z_D]^\top$ 时，常用对角协方差近似后验。这是为了简化计算的建模假设，不表示学到的各维一定对应位置、颜色或姿态，也不保证它们在真实数据中独立。编码器输出均值向量 $\boldsymbol{\mu} \in \mathbb{R}^D$ 和方差向量 $\boldsymbol{\sigma}^2 \in \mathbb{R}^D$。

为了用梯度下降优化这个概率模型，需要从 $\mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ 中得到一个具体的 $\mathbf{z}$。直接把随机采样节点放在计算图中，无法沿普通样本路径得到分布参数的梯度；VAE 因而采用**重参数化技巧（Reparameterization Trick）**。

在标量情形下，重参数化技巧可以直观地理解为：与其直接从均值为 $\mu$、标准差为 $\sigma$ 的分布中抽取 $z$，不如我们先从标准正态分布 $\mathcal{N}(0,1)$ 中抽取一个“标准随机噪声” $\epsilon$。然后，将这个标准噪声放大 $\sigma$ 倍（匹配我们预估的不确定度），再将其平移 $\mu$（匹配我们的预估中心）。其标量表达式为：

$$
z = \mu + \sigma \cdot \epsilon \quad \text{其中} \quad \epsilon \sim \mathcal{N}(0, 1)
$$

同样的写法可以推广到 $D$ 维向量形式。此时 $\boldsymbol{\mu}$ 和 $\boldsymbol{\sigma}$ 为向量，$\odot$ 表示逐元素相乘：

$$
\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon} \quad \text{其中} \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

V 模型由此把高维图像表示为低维随机变量 $\mathbf{z}$。表示是否保留了控制所需信息，还要由重建效果与后续任务共同检验。

## 1.3.3 M模型：混合密度网络与时间演化

<div align="center">
  <img src="/figures/01-why-world-models/source/03-classic-world-models/world-models-fig6.png" alt="World Models 的 MDN-RNN 图显示循环隐藏状态如何参数化下一潜变量的混合分布。" width="86%">

_图 1.3-2：World Models 的 MDN-RNN 图显示循环隐藏状态如何参数化下一潜变量的混合分布。 出处：David Ha; Jürgen Schmidhuber，[World Models](https://arxiv.org/abs/1803.10122)（2018），Figure 6。_

</div>

拥有了低维表示 $\mathbf{z}_t$ 后，我们来到了世界模型中最核心、也最体现物理本质的部分：预测未来。

假设我们在时刻 $t$ 观察到了状态 $z_t$，并且执行了一个动作 $a_t$（例如踩下油门）。我们希望预测下一时刻的状态 $z_{t+1}$。如果这是经典的初等运动学问题，对于一个做匀加速直线的物体，我们有确定的位移公式。然而，智能体所处的环境往往是高度随机的：地面可能有摩擦力突变，侧面可能突然刮起大风。

### 从单一高斯到混合高斯 (MDN)

<div align="center">
  <img src="/figures/01-why-world-models/source/03-classic-world-models/graves-fig10.png" alt="Graves 的混合密度输出热图展示同一时刻可以同时保留多个可能的下一笔方向与位置。" width="86%">

_图 1.3-3：Graves 的混合密度输出热图展示同一时刻可以同时保留多个可能的下一笔方向与位置。 出处：Alex Graves，[Generating Sequences With Recurrent Neural Networks](https://arxiv.org/abs/1308.0850)（2013），Figure 10。_

</div>

面对不确定性，我们第一反应是继续使用单一高斯分布来描述 $z_{t+1}$，即预测下一时刻均值 $\mu_{t+1}$ 和方差 $\sigma_{t+1}^2$。但这种假设在真实世界中会面临严重缺陷。

设想观测中有一辆暂时被遮挡的汽车：它下一刻可能从障碍物左侧出现，也可能从右侧出现。若用单一高斯拟合两个分离模态，均值可能落在两条合理轨迹之间，反而不对应任何真实结果。

为了解决这种**多模态（Multimodal）**的预测问题，我们需要引入混合密度网络（Mixture Density Network, MDN）。MDN 的核心思想是：未来的可能性不是由一个高斯钟形曲线描述的，而是由多个不同形状的高斯曲线叠加而成的。

对于标量预测，具有 $K$ 个混合成分的概率密度函数表示为：

$$
P(z_{t+1} | z_t, a_t) = \sum_{k=1}^{K} \pi_{k} \mathcal{N}(z_{t+1} | \mu_k, \sigma_k^2)
$$

<div align="center"><img src="/figures/01-why-world-models/latex/03-classic-world-models/mdn-component-then-sample.png" alt="先按混合权重选择高斯成分，再从被选成分采样下一潜状态" width="86%">

_图 1.3-4：π 决定宏观分支的选择概率；选定 k 后，μ_k 与 σ_k 再描述该分支内部的不确定性。_

</div>

在这里，每个 $k$ 代表一种可能的“宏观物理演化路径”（例如 $k=1$ 代表向左侧滑，$k=2$ 代表向右侧滑）。

- $\pi_k$ 是**混合权重**，代表了第 $k$ 种路径发生的概率，且必须满足 $\sum_{k=1}^K \pi_k = 1$。
- $\mu_k$ 是第 $k$ 种路径下预测的具体均值。
- $\sigma_k^2$ 是该路径下方差（微观物理不确定性）。

### 引入循环神经网络 (RNN)

单独根据当前的一帧状态 $z_t$ 来预测 $z_{t+1}$ 是不够的。物理学告诉我们，要知道物体的下一时刻位置，我们不仅需要当前的位置，还需要当前的速度、加速度等高阶信息。由于 $z_t$ 仅是单帧图像的编码，它不包含速度。

为了恢复动力学信息，需要在时间序列上引入历史记忆。RNN（或其变体 LSTM）在时刻 $t$ 维护隐藏状态 $\mathbf{h}_t$，尝试把此前的观测和动作压缩为定长向量；压缩是否充分由数据、容量与训练目标决定。

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

当 V 模型提供低维表示、M 模型提供历史特征后，控制器可以使用较小的参数量。在 Ha 和 Schmidhuber 的设计中，C 模型是一个单层线性网络（后接激活函数）：

$$
\mathbf{a}_t = \text{tanh} \left( \mathbf{W}_c [\mathbf{z}_t, \mathbf{h}_t] + \mathbf{b}_c \right)
$$

为什么在如此复杂的自动驾驶或游戏中，控制器可以设计得如此简陋？因为感知（V）和预测（M）这两个最消耗算力和参数量的任务已经被前置解决。控制器的输入不再是杂乱无章的像素，而是经过高度结构化的当前物理状态 $\mathbf{z}_t$ 和历史动态趋势 $\mathbf{h}_t$。

学到的 M 模型还可以生成闭环潜变量轨迹，供控制器反复试验候选动作。这样能够减少一部分真实环境交互，但只在模型覆盖良好的区域内可靠。经典论文使用 CMA-ES 优化小型控制器，并在 VizDoom 实验中把控制器放入生成环境训练；这不是所有任务都采用的固定流程。

## 1.3.5 代码实现：构建经典的M模型 (MDN-RNN)

下面构建一个 MDN-RNN 模块，并标注批次、混合成分和潜空间维度对应的张量形状。

```python
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

        # Softplus 保证标准差为正，并避免 exp 在大输入下快速溢出
        # sigma 形状: (batch_size, num_mixtures, z_dim)
        sigma = F.softplus(self.fc_sigma(hx)).view(-1, self.num_mixtures, self.z_dim) + 1e-4

        return pi, mu, sigma, (hx, cx)
```

上述代码展示了时序更新和 MDN 参数的张量形状。完整训练还需要用数值稳定的混合对数似然计算损失，并处理序列初始状态与终止信号。

## 1.3.6 练习

1. 在推导前文的向量形式混合密度公式时，我们假设了潜状态各个维度在给定条件下是相互独立的。如果这一假设不成立，我们应该如何修改协方差的表示？
   - _提示：思考初等统计中关于相关系数和协方差矩阵对角化的问题。_
2. 请结合张量维度解释，为什么 `fc_mu` 和 `fc_sigma` 的输出维度必须是 `num_mixtures * z_dim`？
   - _提示：如果一个物体在 3 维空间运动，且未来有 5 种不同的走向（模态），那么共需要多少个坐标值来完整描述这些走向的中心位置？_
3. 代码使用 `softplus(raw_sigma) + 1e-4` 计算标准差。为什么还需要后面的微小常数？
   - _提示：考虑概率密度中的除法、对数与接近零的标准差。_
