# 联合嵌入预测架构（JEPA）基础理论

自监督学习（Self-Supervised Learning, SSL）在深度学习的黄金十年中占据了举足轻重的地位。然而，当我们试图让机器像人类一样理解世界时，传统的学习范式暴露出了深层的局限性。2022年，Yann LeCun 在其具有里程碑意义的论文 _A Path Towards Autonomous Machine Intelligence_ [[LeCun, 2022]](https://openreview.net/forum?id=BZ5a1r-kVsf) 中正式提出了联合嵌入预测架构（Joint Embedding Predictive Architecture, JEPA）。这一架构试图回答一个基础且深刻的问题：机器应该在哪个空间中预测未来或补全缺失的信息？

在本节中，我们将从自监督学习的演进脉络出发，逐步剖析 JEPA 的物理直觉与数学基础，并最终在代码层面构建这一架构的核心组件。

## 历史脉络：从重构到表征预测

在 JEPA 诞生之前，主流的自监督学习大致可以分为两类：生成式架构（Generative Architectures）和联合嵌入架构（Joint Embedding Architectures）。

掩码自编码器 MAE 通过重构被遮挡图像块的像素学习表征 [[He et al., 2022]](https://arxiv.org/abs/2111.06377)。LeCun 的 JEPA 立场则认为，物理世界包含许多难以预测且未必与任务相关的像素细节，因此更适合在抽象表征空间预测 [[LeCun, 2022]](https://openreview.net/forum?id=BZ5a1r-kVsf)。前一句是 MAE 的方法，后一句是 JEPA 的设计主张，不能都归到 MAE 论文名下。

另一方面，联合嵌入架构直接约束不同视角的表征。若只最小化正样本之间的距离，模型确实可能把所有输入映射为同一常数；SimCLR 则使用批内负样本和对比损失来排除这种平凡解 [[Chen et al., 2020]](https://arxiv.org/abs/2002.05709)。因此，表示坍塌是联合嵌入方法需要解决的问题，而不是 SimCLR “引入”的问题。

JEPA 的提出，是对这两种范式的超越。它保留了联合嵌入架构在高维抽象空间中操作的优势，同时引入了生成式架构中的“预测”机制。不同的是，JEPA 的预测发生在抽象的嵌入空间，而非原始的观测空间。

## 物理直觉与高中数学映射：信号、噪声与状态转移

为了严谨地理解 JEPA，我们首先将问题降维到最基础的高中物理与数学模型中。

假设我们在观察一个沿直线运动的物体。在时刻 $t$，物体的实际位置是 $p_t$，但我们的测量仪器存在误差，因此我们观测到的位置是 $x_t = p_t + \epsilon_t$，其中 $\epsilon_t$ 是随机噪声。

如果我们想要预测下一时刻 $t+1$ 的观测值 $x_{t+1}$，我们需要同时预测物体的真实运动规律以及未来的噪声 $\epsilon_{t+1}$。从统计学的角度看，这是极其困难甚至是不可能的，因为 $\epsilon_{t+1}$ 是不可预测的。

JEPA 的核心思想在于，我们不应该试图预测观测值 $x_{t+1}$。相反，我们应该构建一个“编码器”（Encoder），其作用是过滤掉观测中的噪声，提取出真实的状态 $p_t$。数学上，我们可以将这个过程定义为一个函数映射 $s_t = f(x_t)$，我们希望 $s_t$ 尽可能接近 $p_t$。

随后，我们在状态空间（而非观测空间）中进行预测：寻找一个预测器（Predictor）函数 $g$，使得 $g(s_t, \Delta t) \approx s_{t+1}$。

这就是 JEPA 的本质：**在过滤了不确定性和无关细节的抽象状态空间中进行条件预测**。

## 核心架构与数学形式化

现在，我们将上述直觉推广到高维向量空间与深度神经网络。

在 JEPA 框架中，存在三个核心的神经网络模块：

1. **上下文编码器（Context Encoder）** $E_\theta$：接收部分可观测的上下文信号 $x$，并输出其表示 $s_x = E_\theta(x)$。
2. **目标编码器（Target Encoder）** $E_\phi$：接收目标信号 $y$（这通常是 $x$ 的空间或时间延伸，或者同一实体的另一视角），并输出其表示 $s_y = E_\phi(y)$。
3. **预测器（Predictor）** $P_\psi$：接收上下文表示 $s_x$ 和一个隐变量（或条件变量） $z$，试图预测目标表示：$\hat{s}_y = P_\psi(s_x, z)$。

### 标量空间的极简损失

我们先从最简单的一维标量空间看起。假设 $s_y, \hat{s}_y \in \mathbb{R}$。衡量预测值与真实目标之间差异的最直观方式是距离。我们可以定义一个简单的平方误差损失：

$$
L = \frac{1}{2} (\hat{s}_y - s_y)^2
$$

此时，我们要优化网络参数 $\theta, \phi, \psi$ 以最小化 $L$。但如果我们直接优化所有参数，系统极易陷入一个平凡解：$E_\theta(x) = c, E_\phi(y) = c$（其中 $c$ 为任意常数）。此时 $\hat{s}_y = c, s_y = c$，损失完美地降为 0，但这并没有学习到任何关于数据的有用表征。这就是前文提到的信息坍塌。

### 向量空间与非对称架构

为了推广到多维向量空间 $\mathbb{R}^d$，并将系统的稳定性纳入考量，我们需要引入更精细的机制。

假设现在的表示向量为 $\mathbf{s}_y, \mathbf{\hat{s}}_y \in \mathbb{R}^d$。基于 $L_2$ 范数的距离损失函数形式如下：

$$
L = \|\mathbf{\hat{s}}_y - \mathbf{s}_y\|_2^2 = \sum_{i=1}^d (\hat{s}_{y, i} - s_{y, i})^2
$$

在这里，$\mathbf{\hat{s}}_y = P_\psi(E_\theta(x), z)$，而 $\mathbf{s}_y = E_\phi(y)$。

为了避免信息坍塌，现代 JEPA 变体（如 I-JEPA [[Assran et al., 2023]](https://arxiv.org/abs/2301.08243)）通常采用一种**非对称（Asymmetric）**的参数更新策略。具体而言，目标编码器 $E_\phi$ 的参数 $\phi$ **不通过**梯度下降直接更新。相反，它是上下文编码器参数 $\theta$ 的指数移动平均（Exponential Moving Average, EMA）：

$$
\phi \leftarrow \tau \phi + (1 - \tau) \theta
$$

其中 $\tau \in [0, 1]$ 且通常接近 1（例如 0.996）。慢速目标编码器、停止梯度、预测器以及掩码策略共同构成 I-JEPA 的非对称训练机制。EMA 提供了稳定目标，但单独使用 EMA 并不能从数学上保证任何架构都不会坍塌。

## 潜变量 $z$ 的深层含义与变分视角

> 假设你正在观察一片在狂风中飘落的树叶。生成式模型试图精确预测树叶在每一微秒的坐标、姿态甚至叶脉上的反光——这项任务几乎注定失败，因为系统中存在巨大的不可控变数（风速的微小扰动）。而 JEPA 则采取了截然不同的策略：它提取出“树叶正在下落”这一核心状态，然后引入一个控制变量 $z$。如果 $z$ 代表“重力影响”，它就能预测出树叶整体向下的趋势；如果 $z$ 代表“特定的强侧风”，它就能预测出树叶向一侧偏移。$z$ 吸收了所有那些我们无法从当前上下文中推断，但对目标状态有决定性影响的信息。

在纯数学意义上，潜变量 $z$ 的存在是为了处理现实世界中“多对多”的映射关系。对于同一个上下文 $x$，未来可能存在多种合理的演化 $y$。如果没有 $z$，预测器 $P_\psi(s_x)$ 只能被迫输出所有可能未来的平均值（这往往是一个模糊且不现实的均值状态）。

引入 $z$ 后，预测器变为 $P_\psi(s_x, z)$。在理想的能量模型（Energy-Based Model）框架下，系统的能量函数定义为：

$$
E(x, y, z) = C(x, y) + D(P_\psi(E_\theta(x), z), E_\phi(y))
$$

其中 $C(x, y)$ 衡量观测变量本身的代价，而 $D$ 则是我们在隐空间中的预测误差。为了使得对应于真实观测对 $(x, y)$ 的能量最低，我们需要对隐变量 $z$ 进行推断，即寻找：

$$
z^* = \arg\min_{z \in \mathcal{Z}} D(P_\psi(E_\theta(x), z), E_\phi(y))
$$

在实际的端到端训练中（特别是在计算机视觉中），隐变量 $z$ 常常被实例化为目标位置或掩码的显式编码（例如目标图像块的位置嵌入），从而免去了复杂的在线优化过程。

## 代码实现：构建 JEPA 的核心骨架

接下来，我们将使用纯粹的张量操作来实现一个 JEPA 的最小骨架。

(**我们首先定义上下文编码器、目标编码器和预测器。**)
在这个简化的例子中，我们使用多层感知机（MLP）来替代复杂的 Transformer 骨干网络，以专注于 JEPA 独有的架构设计。

```{.python .input}
#@tab pytorch
import torch
from torch import nn
import copy

class Encoder(nn.Module):
    """一个简化的编码器，用于将原始输入映射到抽象表示空间。"""
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class Predictor(nn.Module):
    """预测器：接收上下文表示和隐变量 z（此处简化为直接拼接），预测目标表示。"""
    def __init__(self, embed_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + z_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, s_x, z):
        # 将隐变量 z 或条件变量与上下文表示在特征维度拼接
        sz = torch.cat([s_x, z], dim=-1)
        return self.net(sz)
```

(**现在，我们将这些组件组合成完整的 JEPA 模型，并实现非对称的指数移动平均（EMA）更新机制。**)

```{.python .input}
#@tab pytorch
class JEPA(nn.Module):
    def __init__(self, input_dim, embed_dim, z_dim, ema_tau=0.99):
        super().__init__()
        self.ema_tau = ema_tau

        # 上下文编码器 (可训练)
        self.context_encoder = Encoder(input_dim, embed_dim)

        # 目标编码器 (不通过反向传播训练)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # 预测器 (可训练)
        self.predictor = Predictor(embed_dim, z_dim)

    def update_target_encoder(self):
        """执行目标编码器参数的指数移动平均更新。"""
        with torch.no_grad():
            for param_q, param_k in zip(self.context_encoder.parameters(),
                                        self.target_encoder.parameters()):
                param_k.data.mul_(self.ema_tau).add_(param_q.data, alpha=1.0 - self.ema_tau)

    def forward(self, x, y, z):
        """
        x: 上下文输入 (Batch, InputDim)
        y: 目标输入 (Batch, InputDim)
        z: 条件/隐变量 (Batch, ZDim)
        """
        # 1. 计算上下文表示
        s_x = self.context_encoder(x)

        # 2. 计算目标表示 (注意使用 no_grad，防止梯度回传)
        with torch.no_grad():
            s_y = self.target_encoder(y)

        # 3. 在潜空间中进行预测
        s_y_hat = self.predictor(s_x, z)

        return s_y_hat, s_y
```

在训练循环中，每次前向传播并计算出 `s_y_hat` 和 `s_y` 的 L2 损失后，我们只对 `context_encoder` 和 `predictor` 进行反向传播更新，随后必须显式调用 `update_target_encoder` 来更新目标的表征。这一精妙的设计，使得模型能够在一个不断演化但相对稳定的流形中进行预测，从而有效规避了表征坍塌。

## 总结

联合嵌入预测架构（JEPA）代表了机器智能迈向更高级抽象能力的关键一步。通过放弃对细枝末节的执着重构，JEPA 拥抱了在特征空间中进行预测的范式。借助于目标编码器的非对称 EMA 更新与隐变量机制，它不仅提供了一个计算高效的学习框架，也为处理现实世界中内在的不确定性奠定了数学基础。
