# 联合嵌入预测架构（JEPA）基础理论

> **本章导读**
>
> **讲什么：** 本章研究另一条路线：模型不重建每个像素，而是预测目标画面中较抽象的特征。我们会先建立联合嵌入预测架构，再观察“所有输入都映射到同一个特征”这种坍塌解，随后用掩码、目标网络和指数移动平均稳定训练，并把动作加入特征动力学。
>
> **为什么可以不预测像素：** 遮住视频下一帧的一角时，树叶具体偏向哪边可能无法确定，但“道路仍在前方、行人正在靠近”仍然可以预测。若任务关心的是规划与控制，逼模型还原每个不可预测的纹理会浪费容量；然而只预测特征又容易找到没有信息的常数答案，因此必须同时解决表示坍塌。
>
> **故事线：** `从像素重构改为特征预测 → 用掩码制造上下文与目标 → 暴露常数表示的坍塌解 → 用不对称目标网络与 EMA 稳定学习 → 加入动作并展开多步未来`

## 本章总览

<div align="center">

<img src="/figures/06-jepa/latex/01-jepa-foundation/chapter-overview.png" alt="第 6 章学习路线：从 JEPA 特征预测到具身感知与规划" width="100%">

_第 6 章学习路线：从非生成式特征预测出发，经过防坍塌与动量自举，走向动作条件推演与规划。_

</div>

自监督学习（Self-Supervised Learning, SSL）从数据本身构造训练目标。像素重构是一种选择，预测另一部分数据的表征则是另一种选择。LeCun 在 _A Path Towards Autonomous Machine Intelligence_ 中系统阐述了联合嵌入预测架构（Joint-Embedding Predictive Architecture, JEPA）的设想 [[LeCun, 2022]](https://openreview.net/forum?id=BZ5a1r-kVsf)：给定上下文，在表征空间预测目标，而不要求还原目标的每个观测细节。

<div align="center">
  <img src="/figures/06-jepa/source/01-jepa-foundation/data2vec-fig1.png" alt="data2vec 在语音、图像和文本上共享教师表征预测流程，展示潜在目标预测并不限于一种模态。" width="86%">

_图 6.1-1：data2vec 在语音、图像和文本上共享教师表征预测流程，展示潜在目标预测并不限于一种模态。 出处：Alexei Baevski et al.，[data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language](https://arxiv.org/abs/2202.03555)（2022），Figure 1。_

</div>

本节先比较像素重构与表征预测，再定义上下文编码器、目标编码器和预测器，最后实现一个最小训练骨架。

## 历史脉络：从重构到表征预测

在 JEPA 诞生之前，主流的自监督学习大致可以分为两类：生成式架构（Generative Architectures）和联合嵌入架构（Joint Embedding Architectures）。

<div align="center">
  <img src="/figures/06-jepa/source/01-jepa-foundation/ijepa-fig2.png" alt="生成式、联合嵌入与联合嵌入预测三类架构的并列图，说明 JEPA 把监督信号从像素移到表征关系。" width="86%">

_图 6.1-2：生成式、联合嵌入与联合嵌入预测三类架构的并列图，说明 JEPA 把监督信号从像素移到表征关系。 出处：Mahmoud Assran et al.，[Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)（2023），Figure 2。_

</div>

掩码自编码器 MAE 通过重构被遮挡图像块的像素学习表征 [[He et al., 2022]](https://arxiv.org/abs/2111.06377)。LeCun 的 JEPA 立场则认为，物理世界包含许多难以预测且未必与任务相关的像素细节，因此更适合在抽象表征空间预测 [[LeCun, 2022]](https://openreview.net/forum?id=BZ5a1r-kVsf)。前一句是 MAE 的方法，后一句是 JEPA 的设计主张，不能都归到 MAE 论文名下。

<div align="center">
  <img src="/figures/06-jepa/source/01-jepa-foundation/mae-fig2.png" alt="MAE 的遮挡图、像素重建与原图对照，具体展示像素重构路线需要恢复的视觉细节。" width="86%">

_图 6.1-3：MAE 的遮挡图、像素重建与原图对照，具体展示像素重构路线需要恢复的视觉细节。 出处：Kaiming He et al.，[Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)（2022），Figure 2。_

</div>

另一方面，联合嵌入架构直接约束不同视角的表征。若只最小化正样本之间的距离，模型确实可能把所有输入映射为同一常数；SimCLR 则使用批内负样本和对比损失来排除这种平凡解 [[Chen et al., 2020]](https://arxiv.org/abs/2002.05709)。因此，表示坍塌是联合嵌入方法需要解决的问题，而不是 SimCLR “引入”的问题。

JEPA 把“联合嵌入”与“预测”放进同一个框架：上下文和目标分别编码，预测器从上下文表征推断目标表征。与像素重构方法的主要区别是预测目标所在的空间，而不是简单地把两类方法排成高低关系。

## 物理直觉与高中数学映射：信号、噪声与状态转移

先用一维运动说明表征预测想舍弃什么信息。

假设我们在观察一个沿直线运动的物体。在时刻 $t$，物体的实际位置是 $p_t$，但我们的测量仪器存在误差，因此我们观测到的位置是 $x_t = p_t + \epsilon_t$，其中 $\epsilon_t$ 是随机噪声。

若直接预测 $x_{t+1}$，模型既要描述位置变化，也会因损失函数而受到未来测量噪声 $\epsilon_{t+1}$ 的影响。独立噪声的具体取值无法由当前观测确定，平方误差下通常只能学到其条件均值。

JEPA 选择先编码观测 $s_t=f(x_t)$，再预测未来表征。训练希望表征保留可预测、对下游任务有用的结构，并弱化不稳定细节；这是一项目标，不意味着编码器天然等于真实物理状态 $p_t$，还需要实验验证。

随后，我们在状态空间（而非观测空间）中进行预测：寻找一个预测器（Predictor）函数 $g$，使得 $g(s_t, \Delta t) \approx s_{t+1}$。

这一取舍可以概括为：**在表征空间做条件预测，让学习目标少受不可预测观测细节支配。**

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

若两端编码器都能通过同一个距离损失自由更新，$E_\theta(x)=c$、$E_\phi(y)=c$ 会使损失为 0，却不保留输入信息。这说明常数映射是目标函数允许的平凡解；实际优化是否走到这里还取决于完整架构与训练设置。

### 向量空间与非对称架构

为了推广到多维向量空间 $\mathbb{R}^d$，并将系统的稳定性纳入考量，我们需要引入更精细的机制。

假设现在的表示向量为 $\mathbf{s}_y, \mathbf{\hat{s}}_y \in \mathbb{R}^d$。基于 $L_2$ 范数的距离损失函数形式如下：

$$
L = \|\mathbf{\hat{s}}_y - \mathbf{s}_y\|_2^2 = \sum_{i=1}^d (\hat{s}_{y, i} - s_{y, i})^2
$$

<div align="center">
  <img src="/figures/06-jepa/latex/01-jepa-foundation/vector-distance-reduction.png" alt="预测向量与目标向量逐维相减平方，再沿特征维求和" width="86%">

_图 6.1-4：每个特征坐标先形成差值并平方，随后沿 i=1,…,d 归约，得到单个表征距离。_

</div>

在这里，$\mathbf{\hat{s}}_y = P_\psi(E_\theta(x), z)$，而 $\mathbf{s}_y = E_\phi(y)$。

为了避免信息坍塌，现代 JEPA 变体（如 I-JEPA [[Assran et al., 2023]](https://arxiv.org/abs/2301.08243)）通常采用一种**非对称（Asymmetric）**的参数更新策略。具体而言，目标编码器 $E_\phi$ 的参数 $\phi$ **不通过**梯度下降直接更新。相反，它是上下文编码器参数 $\theta$ 的指数移动平均（Exponential Moving Average, EMA）：

$$
\phi \leftarrow \tau \phi + (1 - \tau) \theta
$$

其中 $\tau \in [0, 1]$ 且通常接近 1（例如 0.996）。慢速目标编码器、停止梯度、预测器以及掩码策略共同构成 I-JEPA 的非对称训练机制。EMA 提供了稳定目标，但单独使用 EMA 并不能从数学上保证任何架构都不会坍塌。

## 条件变量与一般 JEPA 中的潜变量

> 只看树叶当前一帧，下一时刻可能向左也可能向右。若另有风向或动作条件，就可把它作为 $z$ 输入预测器；若影响未来的因素不可观测，一般 JEPA 设想也允许用潜变量表示多种相容结果。具体实现是否显式建模这种潜变量，要看方法本身。

在纯数学意义上，潜变量 $z$ 的存在是为了处理现实世界中“多对多”的映射关系。对于同一个上下文 $x$，未来可能存在多种合理的演化 $y$。如果没有 $z$，预测器 $P_\psi(s_x)$ 只能被迫输出所有可能未来的平均值（这往往是一个模糊且不现实的均值状态）。

引入 $z$ 后，预测器变为 $P_\psi(s_x, z)$。在理想的能量模型（Energy-Based Model）框架下，系统的能量函数定义为：

$$
E(x, y, z) = C(x, y) + D(P_\psi(E_\theta(x), z), E_\phi(y))
$$

其中 $C(x, y)$ 衡量观测变量本身的代价，而 $D$ 则是我们在隐空间中的预测误差。为了使得对应于真实观测对 $(x, y)$ 的能量最低，我们需要对隐变量 $z$ 进行推断，即寻找：

$$
z^* = \arg\min_{z \in \mathcal{Z}} D(P_\psi(E_\theta(x), z), E_\phi(y))
$$

<div align="center">
  <img src="/figures/06-jepa/latex/01-jepa-foundation/latent-condition-argmin.png" alt="同一上下文经多个候选条件产生预测，并以到目标表征的距离选择 z 星" width="86%">

_图 6.1-5：x 与 y 固定时，每个候选 z 产生一个预测表征；内层 argmin 选择距离目标编码最小的 z\*。_

</div>

在 I-JEPA 这类图像实现中，预测器接收目标块的位置 token；它是已知条件，不应与表示不可观测不确定性的随机潜变量混为一谈。这里的统一符号 $z$ 只表示“预测器还需读取的条件”。

## 代码实现：构建 JEPA 的核心骨架

下面用张量操作实现一个最小骨架。

为突出分支关系，示例用多层感知机（MLP）代替 Transformer 骨干。

```python
import torch
from torch import nn
import copy

class Encoder(nn.Module):
    """一个简化的编码器，用于将原始输入映射到抽象表示空间。"""
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
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
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embed_dim)
        )

    def forward(self, s_x, z):
        # 将隐变量 z 或条件变量与上下文表示在特征维度拼接
        sz = torch.cat([s_x, z], dim=-1)
        return self.net(sz)
```

再把三个组件组合起来，并实现非对称的 EMA 更新。

```python
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
                param_k.mul_(self.ema_tau).add_(param_q, alpha=1.0 - self.ema_tau)

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

训练时只对 `context_encoder` 和 `predictor` 反向传播，再显式调用 `update_target_encoder`。这样目标分支变化得比在线分支慢。该最小代码展示数据流，但没有复现 I-JEPA 的块采样、位置 token、归一化与完整训练配置，因此不能单凭这个示例断言一定不会坍塌。

## 总结

JEPA 的核心选择是让预测误差发生在表征空间。上下文编码器提供已知信息，目标编码器产生训练目标，预测器结合位置、动作或其他条件输出目标表征。非对称更新用于稳定目标；潜变量是否存在、怎样处理多种未来，则由具体 JEPA 实现决定。
