# 联合嵌入预测架构（JEPA）基础理论
:label:sec_jepa_foundation

自监督学习（Self-Supervised Learning, SSL）在深度学习的黄金十年中占据了举足轻重的地位。然而，当我们试图让机器像人类一样理解世界时，传统的学习范式暴露出了深层的局限性。2022年，Yann LeCun 在其具有里程碑意义的论文 *A Path Towards Autonomous Machine Intelligence* `[LeCun, 2022]` 中正式提出了联合嵌入预测架构（Joint Embedding Predictive Architecture, JEPA）。这一架构试图回答一个基础且深刻的问题：机器应该在哪个空间中预测未来或补全缺失的信息？

在本节中，我们将从自监督学习的演进脉络出发，逐步剖析 JEPA 的物理直觉与数学基础，并最终在代码层面构建这一架构的核心组件。

## 历史脉络：从重构到表征预测
:label:subsec_jepa_history

在 JEPA 诞生之前，主流的自监督学习大致可以分为两类：生成式架构（Generative Architectures）和联合嵌入架构（Joint Embedding Architectures）。

生成式架构（例如掩码自编码器 MAE `[He et al., 2022]` 或自回归语言模型）通过在输入空间（如像素或词元）中预测缺失的部分来进行学习。这种方法的优势在于其通用性和简单的优化目标，但其致命弱点在于：物理世界中充满了不可预测的细枝末节（例如风中飘动的树叶的精确轨迹）。强迫模型在像素级别重构这些不可预测的噪声，不仅浪费了巨大的计算资源，也阻碍了模型学习更高层级的抽象语义。

另一方面，早期的联合嵌入架构（如 SimCLR `[Chen et al., 2020]`）通过最大化同一图像的不同视角的表征相似度来进行学习。这种方法成功地摒弃了像素级重构，但也引入了所谓的“表示坍塌”（Representation Collapse）问题——模型可能学会将所有输入映射到一个常数向量以轻易地最小化损失。

JEPA 的提出，是对这两种范式的超越。它保留了联合嵌入架构在高维抽象空间中操作的优势，同时引入了生成式架构中的“预测”机制。不同的是，JEPA 的预测发生在抽象的嵌入空间，而非原始的观测空间。

## 物理直觉与高中数学映射：信号、噪声与状态转移
:label:subsec_jepa_intuition

为了严谨地理解 JEPA，我们首先将问题降维到最基础的高中物理与数学模型中。

假设我们在观察一个沿直线运动的物体。在时刻 $t$，物体的实际位置是 $p_t$，但我们的测量仪器存在误差，因此我们观测到的位置是 $x_t = p_t + \epsilon_t$，其中 $\epsilon_t$ 是随机噪声。

如果我们想要预测下一时刻 $t+1$ 的观测值 $x_{t+1}$，我们需要同时预测物体的真实运动规律以及未来的噪声 $\epsilon_{t+1}$。从统计学的角度看，这是极其困难甚至是不可能的，因为 $\epsilon_{t+1}$ 是不可预测的。

JEPA 的核心思想在于，我们不应该试图预测观测值 $x_{t+1}$。相反，我们应该构建一个“编码器”（Encoder），其作用是过滤掉观测中的噪声，提取出真实的状态 $p_t$。数学上，我们可以将这个过程定义为一个函数映射 $s_t = f(x_t)$，我们希望 $s_t$ 尽可能接近 $p_t$。

随后，我们在状态空间（而非观测空间）中进行预测：寻找一个预测器（Predictor）函数 $g$，使得 $g(s_t, \Delta t) \approx s_{t+1}$。

这就是 JEPA 的本质：**在过滤了不确定性和无关细节的抽象状态空间中进行条件预测**。

## 核心架构与数学形式化
:label:subsec_jepa_math

现在，我们将上述直觉推广到高维向量空间与深度神经网络。

在 JEPA 框架中，存在三个核心的神经网络模块：
1. **上下文编码器（Context Encoder）** $E_\theta$：接收部分可观测的上下文信号 $x$，并输出其表示 $s_x = E_\theta(x)$。
2. **目标编码器（Target Encoder）** $E_\phi$：接收目标信号 $y$（这通常是 $x$ 的空间或时间延伸，或者同一实体的另一视角），并输出其表示 $s_y = E_\phi(y)$。
3. **预测器（Predictor）** $P_\psi$：接收上下文表示 $s_x$ 和一个隐变量（或条件变量） $z$，试图预测目标表示：$\hat{s}_y = P_\psi(s_x, z)$。

### 标量空间的极简损失
:label:subsec_jepa_loss_scalar

我们先从最简单的一维标量空间看起。假设 $s_y, \hat{s}_y \in \mathbb{R}$。衡量预测值与真实目标之间差异的最直观方式是距离。我们可以定义一个简单的平方误差损失：

$$
L = \frac{1}{2} (\hat{s}_y - s_y)^2
$$
:eqlabel:eq_jepa_scalar_loss

此时，我们要优化网络参数 $\theta, \phi, \psi$ 以最小化 $L$。但如果我们直接优化所有参数，系统极易陷入一个平凡解：$E_\theta(x) = c, E_\phi(y) = c$（其中 $c$ 为任意常数）。此时 $\hat{s}_y = c, s_y = c$，损失完美地降为 0，但这并没有学习到任何关于数据的有用表征。这就是前文提到的信息坍塌。

### 向量空间与非对称架构
:label:subsec_jepa_loss_vector

为了推广到多维向量空间 $\mathbb{R}^d$，并将系统的稳定性纳入考量，我们需要引入更精细的机制。

假设现在的表示向量为 $\mathbf{s}_y, \mathbf{\hat{s}}_y \in \mathbb{R}^d$。基于 $L_2$ 范数的距离损失函数形式如下：

$$
L = \|\mathbf{\hat{s}}_y - \mathbf{s}_y\|_2^2 = \sum_{i=1}^d (\hat{s}_{y, i} - s_{y, i})^2
$$
:eqlabel:eq_jepa_vector_loss

在这里，$\mathbf{\hat{s}}_y = P_\psi(E_\theta(x), z)$，而 $\mathbf{s}_y = E_\phi(y)$。

为了避免信息坍塌，现代 JEPA 变体（如 I-JEPA `[Assran et al., 2023]`）通常采用一种**非对称（Asymmetric）**的参数更新策略。具体而言，目标编码器 $E_\phi$ 的参数 $\phi$ **不通过**梯度下降直接更新。相反，它是上下文编码器参数 $\theta$ 的指数移动平均（Exponential Moving Average, EMA）：

$$
\phi \leftarrow \tau \phi + (1 - \tau) \theta
$$
:eqlabel:eq_jepa_ema

其中 $\tau \in [0, 1]$ 且通常接近 1（例如 0.996）。这种非对称性打破了网络向常数解坍塌的梯度流动路径，因为目标表征生成器成为了一个相对缓慢移动的“教师”网络，而非可以被随意优化以匹配预测的“学生”。

## 潜变量 $z$ 的深层含义与变分视角
:label:subsec_jepa_latent

> 假设你正在观察一片在狂风中飘落的树叶。生成式模型试图精确预测树叶在每一微秒的坐标、姿态甚至叶脉上的反光——这项任务几乎注定失败，因为系统中存在巨大的不可控变数（风速的微小扰动）。而 JEPA 则采取了截然不同的策略：它提取出“树叶正在下落”这一核心状态，然后引入一个控制变量 $z$。如果 $z$ 代表“重力影响”，它就能预测出树叶整体向下的趋势；如果 $z$ 代表“特定的强侧风”，它就能预测出树叶向一侧偏移。$z$ 吸收了所有那些我们无法从当前上下文中推断，但对目标状态有决定性影响的信息。

在纯数学意义上，潜变量 $z$ 的存在是为了处理现实世界中“多对多”的映射关系。对于同一个上下文 $x$，未来可能存在多种合理的演化 $y$。如果没有 $z$，预测器 $P_\psi(s_x)$ 只能被迫输出所有可能未来的平均值（这往往是一个模糊且不现实的均值状态）。

引入 $z$ 后，预测器变为 $P_\psi(s_x, z)$。在理想的能量模型（Energy-Based Model）框架下，系统的能量函数定义为：

$$
E(x, y, z) = C(x, y) + D(P_\psi(E_\theta(x), z), E_\phi(y))
$$
:eqlabel:eq_jepa_energy

其中 $C(x, y)$ 衡量观测变量本身的代价，而 $D$ 则是我们在隐空间中的预测误差。为了使得对应于真实观测对 $(x, y)$ 的能量最低，我们需要对隐变量 $z$ 进行推断，即寻找：

$$
z^* = \arg\min_{z \in \mathcal{Z}} D(P_\psi(E_\theta(x), z), E_\phi(y))
$$
:eqlabel:eq_jepa_z_inference

在实际的端到端训练中（特别是在计算机视觉中），隐变量 $z$ 常常被实例化为目标位置或掩码的显式编码（例如目标图像块的位置嵌入），从而免去了复杂的在线优化过程。

## 代码实现：构建 JEPA 的核心骨架
:label:subsec_jepa_code

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

## 练习

1. **梯度截断与坍塌**：在实现 JEPA 的 `forward` 方法时，如果我们在计算 `s_y = self.target_encoder(y)` 时忘记使用 `torch.no_grad()`，并使得目标编码器的参数也可以被直接优化，系统会发生什么？
   *提示：考虑前文中关于 $E_\theta(x) = c, E_\phi(y) = c$ 平凡解的数学推导。*
2. **动量参数 $\tau$ 的极限情况**：考察公式 :eqref:`eq_jepa_ema`，当 $\tau = 0$ 和 $\tau = 1$ 时，目标编码器的行为分别退化成了什么？这两种极端情况对训练稳定性有何影响？
   *提示：$\tau=0$ 意味着完全同步，$\tau=1$ 意味着完全冻结。*
3. **隐变量 $z$ 的维度**：如果 $z$ 的维度极高，并且其容量大到足以记住目标 $y$ 中的所有信息，预测器 $P_\psi$ 会发生什么样的退化？
   *提示：从信息论的角度思考，如果 $z$ 本身可以无损重构 $y$，上下文 $s_x$ 的信息还会被利用吗？*

:begin_tab:pytorch
[讨论](https://discuss.d2l.ai/t/1234)
:end_tab:
