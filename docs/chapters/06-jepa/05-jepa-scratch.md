# JEPA 表征学习模块的从零开始实现

自监督学习包含多种训练目标。掩码自编码器 MAE 重构被遮挡图像块的像素 [[He et al., 2022]](https://arxiv.org/abs/2111.06377)；SimCLR、MoCo 等对比方法则在表征空间中拉近正样本、区分负样本。LeCun 的 JEPA 立场认为，对不可预测像素细节进行精确重构可能把容量花在与语义任务无关的信息上；这是 JEPA 的设计动机，不能仅由 MAE 论文反向证明。

LeCun 提出了联合嵌入预测架构（Joint-Embedding Predictive Architecture, JEPA）的总体设想 [[LeCun, 2022]](https://openreview.net/forum?id=BZ5a1r-kVsf)。I-JEPA 随后把它实现为图像自监督学习方法：根据上下文块的表征预测目标块表征，不重构像素，也不使用显式负样本，并在论文所报告的图像分类、低样本和迁移评测中验证表征质量 [[Assran et al., 2023]](https://arxiv.org/abs/2301.08243)。

在本节中，我们将完全从零开始，使用基础的张量操作和基本的神经网络层，构建一个 JEPA 表征学习模块。我们将从其数学原理出发，逐步推导并实现其核心组件：上下文编码器（Context Encoder）、目标编码器（Target Encoder）以及预测器（Predictor）。

## 自监督学习的范式转移：抽象空间中的预测

为了更好地理解为什么要在表征空间进行预测，我们先从一个简单的物理学观察出发。假设我们在观察一个从斜坡上滚下的小球。如果我们使用生成式的思维，我们需要预测下一刻小球表面的每一道反光、每一处划痕，以及背景中扬起的灰尘；这对应于巨大的计算负担。

在经典力学中，我们并不会这样做。我们会将小球抽象为一个质点，只关心它的质量 $m$、当前位置 $x$ 和速度 $v$。基于当前的观测（上下文），我们通过牛顿运动定律预测它未来的位置和速度（目标表征）。这种抽象极大地过滤了无关的视觉噪声。

JEPA 正是这种抽象思维在神经网络中的直接体现。它不强迫网络去重现“划痕和反光”，而是让网络学习一个映射，将复杂的原始高维输入映射到一个低维、紧凑的表征空间，并在该空间内进行动力学或空间结构上的预测。

## JEPA 架构的数学形式化

JEPA 的训练过程可以被严谨地描述为两个不同视角的特征提取，以及它们之间在特定条件下的回归问题。

### 场景设定与简单标量推导

假设我们的数据是一维的标量序列，例如某个传感器随时间采集的温度数据 $x_1, x_2, \ldots, x_T$。
我们拥有过去的观测上下文 $x_c = \{x_1, x_2, x_3\}$，并希望预测未来的目标 $x_y = x_5$。

在 JEPA 中，我们首先使用一个非线性函数 $f$（编码器）将这些观测值转换为隐状态（表征）：
$$s_c = f(x_c)$$
$$s_y = f(x_y)$$

此时，我们引入一个预测器 $g$。预测器不能仅仅凭借上下文 $s_c$ 就随意输出，它必须知道我们希望预测**什么位置**的目标。因此，预测器还需要接收一个指示变量 $z$（例如时间差 $\Delta t = 2$ 或目标的位置索引）。预测过程表示为：
$$\hat{s}_y = g(s_c, z)$$

我们希望预测的表征 $\hat{s}_y$ 尽可能逼近真实计算出的目标表征 $s_y$。最直接的衡量标准是均方误差（MSE）：
$$L = (\hat{s}_y - s_y)^2$$

### 矩阵与张量化表达

现在，我们将上述简单的一维序列推广到高维张量，例如图像或高维时间序列。令输入为 $\mathbf{X} \in \mathbb{R}^{N \times D}$，其中 $N$ 是序列长度（或图像分块的数量），$D$ 是特征维度。

我们将输入拆分为两个不重叠或部分重叠的集合：上下文区域矩阵 $\mathbf{X}_c \in \mathbb{R}^{N_c \times D}$ 和目标区域矩阵 $\mathbf{X}_y \in \mathbb{R}^{N_y \times D}$。

在深度学习中，编码器 $f$ 通常由参数化的神经网络构成，例如具有权重 $\theta$ 的 Vision Transformer。因此，我们将上下文编码器记为 $f_\theta$。目标编码器本应具有相同的权重，但为了防止表示坍塌（Representation Collapse，即网络输出一个常数以使误差恒为0），我们使用另一组权重 $\bar{\theta}$ 来参数化目标编码器，记为 $f_{\bar{\theta}}$。

于是，我们得到矩阵形式的表征：
$$\mathbf{S}_c = f_\theta(\mathbf{X}_c) \in \mathbb{R}^{N_c \times d}$$
$$\mathbf{S}_y = f_{\bar{\theta}}(\mathbf{X}_y) \in \mathbb{R}^{N_y \times d}$$

其中 $d$ 是表征空间的维度。

预测器 $g_\phi$ 由参数 $\phi$ 构成，它结合上下文表征 $\mathbf{S}_c$ 和目标区域的位置编码矩阵 $\mathbf{Z} \in \mathbb{R}^{N_y \times d}$，输出对目标表征的预测：
$$\hat{\mathbf{S}}_y = g_\phi(\mathbf{S}_c, \mathbf{Z}) \in \mathbb{R}^{N_y \times d}$$

最终的损失函数在特征维度和目标块的数量上取均方误差：
$$\mathcal{L}(\theta, \phi) = \frac{1}{N_y} \sum_{i=1}^{N_y} \|\hat{\mathbf{s}}_{y, i} - \mathbf{s}_{y, i}\|_2^2$$

::: warning 注意
在 JEPA 的优化过程中，这是一个极其核心的非对称操作：**损失函数 $\mathcal{L}$ 只对上下文编码器的参数 $\theta$ 和预测器的参数 $\phi$ 计算梯度并更新**。目标编码器的参数 $\bar{\theta}$ 被视为常数（Stop-Gradient），绝不能通过反向传播更新。这也是打破对称性、防止网络坍塌到平凡解（Trivial Solution）的根本保证。
:::

为了让目标编码器能够提供高质量、一致的表征目标，参数 $\bar{\theta}$ 采用指数移动平均（Exponential Moving Average, EMA）的方式，根据 $\theta$ 的历史值进行平滑更新：
$$\bar{\theta} \leftarrow \tau \bar{\theta} + (1 - \tau) \theta$$

其中 $\tau \in [0, 1)$ 是动量衰减率，通常取接近 $1$ 的值（如 $0.996$）。

## 从零实现 JEPA 的核心组件

理解了上述严密的数学表述后，我们将逐步使用代码将这些公式转化为计算图。为了聚焦于 JEPA 机制本身，这里我们使用多层感知机（MLP）作为编码器与预测器的基础模块。在实际的 I-JEPA 中，它们通常由更复杂的 Transformer 块构成，但基本的数据流转逻辑是完全一致的。

(**首先，我们导入必要的库。**)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
```

### 基础块的定义

我们首先定义一个通用的多层感知机（MLP）块，它将承担这两个公式中非线性映射的重任。

(**我们定义一个带有残差连接的 MLP 模块，作为特征提取的基础。**)

```python
class MLPBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return x + self.net(self.norm(x))
```

### 上下文与目标编码器

接下来，我们基于上述基础模块构建编码器。正如前文的编码器公式所示，输入数据首先映射到维度为 `d` 的表征空间。

(**我们定义编码器架构。**)

```python
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, mlp_dim, num_layers=3):
        super().__init__()
        # 将输入投影到隐含特征维度 (d)
        self.proj = nn.Linear(input_dim, hidden_dim)
        # 堆叠多个基础特征提取块
        self.blocks = nn.ModuleList([
            MLPBlock(hidden_dim, mlp_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x 形状: (批量大小, 序列长度/块数, 输入特征维度)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
```

### 预测器 (Predictor)

预测器是 JEPA 区别于其他架构的核心。它必须接收上下文表征 $S_c$ 以及指示目标位置的变量 $Z$。
在实践中，一种常见的处理方法是将目标的位置编码 $Z$ 直接拼接或相加到上下文表征上，然后再通过预测器网络。在这里，为了简化说明，我们将表示“目标条件”的变量 $Z$（比如期望预测的未来时间步或空间索引映射成的向量）与上下文特征进行拼接。

(**我们实现预测器，它根据上下文和位置指示预测目标。**)

```python
class Predictor(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, num_layers=2):
        super().__init__()
        # 输入维度是 hidden_dim (上下文) + hidden_dim (位置编码 Z)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_dim),
            nn.GELU(),
            *[MLPBlock(mlp_dim, mlp_dim) for _ in range(num_layers - 1)],
            nn.LayerNorm(mlp_dim),
            nn.Linear(mlp_dim, hidden_dim)
        )

    def forward(self, context_repr, target_position_encoding):
        # context_repr 形状: (批量大小, hidden_dim)
        # target_position_encoding 形状: (批量大小, hidden_dim)

        # 在特征维度进行拼接
        x = torch.cat([context_repr, target_position_encoding], dim=-1)
        # 输出形状将再次回到 (批量大小, hidden_dim)
        return self.net(x)
```

### 整合 JEPA 模型与 EMA 机制

现在，我们将上下文编码器、目标编码器和预测器组合成一个完整的 JEPA 模块。这里我们需要特别注意该公式中的 EMA 逻辑的实现。在初始化时，目标编码器是上下文编码器的精确副本。在每次前向传播或优化后，我们需要平滑地更新目标编码器的权重。

(**实现包含 EMA 动量更新的完整 JEPA 架构。**)

```python
class JEPAModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, mlp_dim, tau=0.996):
        super().__init__()
        self.tau = tau

        # 1. 实例化上下文编码器 (参数 theta)
        self.context_encoder = Encoder(input_dim, hidden_dim, mlp_dim)

        # 2. 实例化目标编码器 (参数 bar_theta)，并初始化为与 context_encoder 相同
        self.target_encoder = copy.deepcopy(self.context_encoder)
        # 冻结目标编码器的参数，使其不参与反向传播
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # 3. 实例化预测器 (参数 phi)
        self.predictor = Predictor(hidden_dim, mlp_dim)

    def forward(self, x_context, x_target, z_target_pos):
        """
        x_context: 上下文数据 (Batch, N_c, input_dim)
        x_target: 目标数据 (Batch, N_y, input_dim)
        z_target_pos: 目标数据对应的位置信息编码 (Batch, N_y, hidden_dim)
        """
        # 计算上下文表征 S_c，由于是序列形式，我们池化取平均以得到单个向量
        # 在实际实现（如 I-JEPA）中，这会更加复杂（例如使用注意力机制合并信息）
        s_c_seq = self.context_encoder(x_context)
        s_c = s_c_seq.mean(dim=1) # 形状: (Batch, hidden_dim)

        # 扩展 s_c 以匹配目标序列长度进行逐个预测
        # 形状变为 (Batch, N_y, hidden_dim)
        s_c_expanded = s_c.unsqueeze(1).expand(-1, z_target_pos.size(1), -1)

        # 预测目标表征 \hat{S}_y
        s_y_hat = self.predictor(s_c_expanded, z_target_pos)

        # 使用目标编码器计算真实的目标表征 S_y
        # 使用 torch.no_grad() 确保没有任何梯度流向 target_encoder
        with torch.no_grad():
            s_y = self.target_encoder(x_target)

        return s_y_hat, s_y

    @torch.no_grad()
    def update_target_encoder(self):
        """执行指数移动平均 (EMA) 更新 \bar{\theta} <- \tau \bar{\theta} + (1 - \tau) \theta"""
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(self.tau).add_((1.0 - self.tau) * param_q.detach().data)
```

## 损失函数与训练过程

在拥有了完整的架构后，我们可以编写单次迭代的训练逻辑。训练过程严格遵循我们推导的公式：计算前向传播得到预测表征 $\hat{\mathbf{S}}_y$ 和目标表征 $\mathbf{S}_y$，根据该公式计算均方误差（MSE），通过反向传播仅更新 $\theta$ 和 $\phi$，最后调用该公式更新 $\bar{\theta}$。

(**演示一个训练步骤（单步更新）的数据流转。**)

```python
# 模拟一些随机输入数据
batch_size = 8
N_c = 10 # 上下文序列长度
N_y = 4  # 预测目标序列长度
input_dim = 64
hidden_dim = 128

x_context = torch.randn(batch_size, N_c, input_dim)
x_target = torch.randn(batch_size, N_y, input_dim)
# 假设我们通过某种方式提取到了目标位置的向量表示 Z
z_target_pos = torch.randn(batch_size, N_y, hidden_dim)

# 初始化模型与优化器
jepa = JEPAModel(input_dim=input_dim, hidden_dim=hidden_dim, mlp_dim=256)
optimizer = torch.optim.Adam(
    list(jepa.context_encoder.parameters()) + list(jepa.predictor.parameters()),
    lr=1e-4
)

# 训练迭代单步
jepa.train()
optimizer.zero_grad()

# 1. 前向传播
s_y_hat, s_y = jepa(x_context, x_target, z_target_pos)

# 2. 计算表征空间的 MSE 损失
loss = F.mse_loss(s_y_hat, s_y)

# 3. 反向传播更新 \theta (context_encoder) 和 \phi (predictor)
loss.backward()
optimizer.step()

# 4. 指数移动平均更新 \bar{\theta} (target_encoder)
jepa.update_target_encoder()

print(f"训练步完成，表征预测损失: {loss.item():.4f}")
```

在这个机制中，目标表征 $\mathbf{S}_y$ 由平滑更新的目标编码器生成，起到了一个动态但相对稳定的教师（Teacher）作用。而上下文编码器和预测器则是学生（Student），它们致力于根据片段信息和相对位置去重构“教师”心目中的完整世界。由于在表征空间内进行重构，网络无需理会那些无用的高频像素信息，从而拥有了更深层的语义抽象能力。

## 小结

- 传统生成式与对比式自监督方法在解决高频噪声和依赖数据增强上存在根本瓶颈。
- JEPA 提供了一种优雅的范式转移：不再直接预测原始空间的未知信息，而是在**高度抽象的特征空间**内基于给定的位置先验去预测目标区域的表征。
- 非对称的架构设计是保证 JEPA 不发生表征坍塌的物理基石：对预测器和上下文编码器执行梯度下降，但对目标编码器严格使用**停止梯度**（Stop-Gradient）并配合**指数移动平均**（EMA）进行平滑更新。
- 预测器不仅接收上下文信息，必须还要接收目标的位置条件变量 $Z$ 才能进行精准推断。
