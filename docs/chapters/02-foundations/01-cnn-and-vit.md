# 视觉基础模型：从卷积神经网络到视觉变换器

## 引言与历史追溯

在计算机视觉的发展中，手工特征与神经网络曾长期并行演进。LeCun 等人的早期卷积网络已能从数据中学习用于手写数字识别的特征 [[LeCun et al., 1989]](https://doi.org/10.1162/neco.1989.1.4.541)；2012 年，AlexNet 又在 ImageNet 分类任务上显著降低了错误率 [[Krizhevsky et al., 2012]](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)。这两项工作分别展示了卷积网络的早期可行性与大规模视觉任务上的突破。

CNN 的成功很大程度上归功于其内置的**归纳偏置**（Inductive Bias）——特别是**平移不变性**（Translation Invariance）和**局部性**（Locality）。这些偏置使得 CNN 能够在相对较少的数据集上高效学习。然而，随着数据规模的指数级增长和计算能力的提升，研究人员开始质疑这种强偏置是否限制了模型的上限。

2020 年，Dosovitskiy 等人提出视觉变换器（Vision Transformer, ViT） [[Dosovitskiy et al., 2020]](https://arxiv.org/abs/2010.11929)，这一工作沿用了自然语言 Transformer 的序列建模结构 [[Vaswani et al., 2017]](https://arxiv.org/abs/1706.03762)。ViT 将图像分割为一系列小块（Patches），再用自注意力聚合全局信息。原论文报告，在足够大规模的数据上预训练后，ViT 在多项图像分类基准上达到或超过当时的卷积网络基线；这项结论依赖论文中的预训练规模和评测设置。

在本章中，我们将从最基础的数学定义出发，逐步推导并实现这两种奠定了现代视觉基础模型地位的核心架构。

## 卷积神经网络的数学原理

### 从全连接层到卷积

要理解卷积神经网络，我们首先需要考察为什么不能直接使用多层感知机（MLP）来处理图像。假设我们有一张尺寸为 $H \times W$ 的二维黑白图像，它可以自然地表示为一个矩阵 $\mathbf{X} \in \mathbb{R}^{H \times W}$。如果使用全连接层，我们需要将图像展平为一个长度为 $H \times W$ 的一维向量 $\mathbf{x}$。
全连接层的输出隐藏表示 $\mathbf{h}$ 同样可以被组织为 $H \times W$ 的矩阵，其对应的标量计算可以严格表示为：

$$
h_{i, j} = \sum_{k=1}^{H} \sum_{l=1}^{W} W_{i, j, k, l} x_{k, l} + b_{i, j}
$$

在这里，权重 $\mathbf{W}$ 是一个四阶张量，包含 $(H \times W) \times (H \times W)$ 个参数。对于一张仅仅是 $1000 \times 1000$ 像素的图像，这个全连接层的权重参数量将达到惊人的 $10^{12}$，这不仅会耗尽任何现代计算机的内存，更会导致模型极易过拟合。

为了减少参数并利用图像的内在几何结构，我们引入两个重要的原则：

1. **平移不变性**：在图像中，一个物体无论出现在左上角还是右下角，模型对其特征的响应机制应该是相同的。这意味着权重张量 $\mathbf{W}$ 不应该依赖于输出的绝对物理位置 $(i, j)$，而只应该依赖于输入和输出位置的相对偏移量。我们令 $k = i + a$ 和 $l = j + b$，则权重可以严格重写为 $V_{a, b} = W_{i, j, i+a, j+b}$。
2. **局部性**：图像中的像素通常只与其周围邻近的像素有较强的物理和统计相关性。因此，我们在计算 $h_{i,j}$ 时，只需考察距离 $(i,j)$ 较近的输入像素，即限制偏移量 $a$ 和 $b$ 的范围在 $[-\Delta, \Delta]$ 之间。

结合上述两个原则，该公式可以被极大地简化为：

$$
h_{i, j} = \sum_{a=-\Delta}^{\Delta} \sum_{b=-\Delta}^{\Delta} V_{a, b} x_{i+a, j+b} + b
$$

这就是二维**互相关**（Cross-Correlation）运算的严格数学定义。在深度学习文献中，它通常被直接称为“卷积”（尽管在纯数学定义中，卷积要求对权重进行翻转，但这并不影响其在神经网络中的参数学习）。参数张量 $\mathbf{V}$ 被称为**卷积核**（Convolutional Kernel）或**滤波器**（Filter），其大小仅为 $(2\Delta + 1) \times (2\Delta + 1)$，且在图像的所有位置严格共享。

### 多通道的张量运算

在实际应用中，图像往往不仅是一个二维矩阵，而是具有颜色通道的三维张量，例如 RGB 图像表示为 $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$，其中 $C=3$。为了处理多通道输入，我们需要为每个输入通道分配一个二维卷积核，并将它们的结果相加。

假设输入有 $C_{in}$ 个通道，输出我们需要产生 $C_{out}$ 个通道的特征图。对于每一个输出通道 $d \in \{1, \dots, C_{out}\}$，其在位置 $(i, j)$ 的输出标量值为：

$$
h_{d, i, j} = \sum_{c=1}^{C_{in}} \sum_{a=-\Delta}^{\Delta} \sum_{b=-\Delta}^{\Delta} V_{d, c, a, b} x_{c, i+a, j+b} + b_d
$$

此时，权重张量 $\mathbf{V}$ 的维度变为 $\mathbb{R}^{C_{out} \times C_{in} \times K_h \times K_w}$，其中 $K_h$ 和 $K_w$ 为卷积核的高度和宽度。这种多通道的卷积操作使得网络不仅能够捕获空间上的局部特征，还能够在通道维度融合更加抽象的信息。

(**我们可以使用 PyTorch 来实现并观察一个多通道卷积层计算后的张量维度：**)

```{.python .input}
#@tab pytorch
import torch
import torch.nn as nn

# 创建一个具有 3 个输入通道、16 个输出通道、卷积核大小为 3x3 的卷积层
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
# 构造一个形状为 (批量大小, 通道数, 高度, 宽度) 的随机张量
# 这里我们假设批量大小为 4，图像分辨率为 224x224
X = torch.randn(4, 3, 224, 224)

# 前向传播
Y = conv_layer(X)
print(f"输入形状: {X.shape}")
print(f"输出形状: {Y.shape}")
print(f"权重形状: {conv_layer.weight.shape}")
```

## 经典 CNN 架构：残差网络 (ResNet)

随着网络层数增加，简单堆叠网络可能出现训练误差反而升高的退化问题。He 等人提出残差网络，并用恒等快捷连接使网络学习残差函数；原论文展示了这种结构能够训练更深的图像分类网络 [[He et al., 2015]](https://arxiv.org/abs/1512.03385)。这项引用直接支持退化问题与残差结构，不应笼统表述为解决了所有梯度消失问题。

### 残差连接的数学机制

假设我们将网络中的某个块（由若干个卷积层组成）拟合为一个非线性映射 $\mathcal{F}(\mathbf{x})$。在传统的网络设计中，该块的输出直接就是 $\mathcal{F}(\mathbf{x})$。然而，ResNet 引入了一个严格的跳跃连接（Skip Connection），要求该网络块去拟合残差映射 $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$，因此其实际输出变为了：

$$
\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}
$$

在反向传播计算梯度时，假设最终的标量损失函数为 $\mathcal{L}$，根据链式法则，我们对其求导：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \left( \frac{\partial \mathcal{F}(\mathbf{x})}{\partial \mathbf{x}} + \mathbf{I} \right)
$$

这里的 $\mathbf{I}$ 是单位矩阵。该公式揭示了残差网络为何能够避免梯度消失的核心数学机制：由于加法项 $\mathbf{I}$ 的存在，即使在极深的网络中梯度 $\frac{\partial \mathcal{F}}{\partial \mathbf{x}}$ 趋近于零（即经过多次卷积导致信号衰减），由后方传来的梯度 $\frac{\partial \mathcal{L}}{\partial \mathbf{y}}$ 也总能通过单位矩阵直接、无损地传递回上一层，保证了底层网络参数的有效更新。

(**下面我们实现一个标准的残差块：**)

```{.python .input}
#@tab pytorch
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 第一个卷积层，可能改变步幅以调整空间维度
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 卷积用于调整跳跃连接的通道数和分辨率，以保证能够正确相加
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity  # 严格执行残差相加的数学映射
        out = self.relu(out)
        return out
```

## 视觉变换器 (Vision Transformer, ViT)

虽然卷积在提取局部特征上极其高效，但要捕获距离较远的两个像素之间的关联，CNN 需要堆叠非常深的层，使得其感受野（Receptive Field）能够扩大到覆盖整个图像。相比之下，视觉变换器（ViT）摒弃了局部性假设，通过自注意力机制一步到位地建立了全局依赖。

### 图像块的嵌入与序列化

为了让最初为处理自然语言离散序列而设计的 Transformer 架构能够处理连续且高维的图像，ViT 首先将三维的图像张量转换成一维的词元（Token）序列。

假设输入图像为 $\mathbf{X} \in \mathbb{R}^{C \times H \times W}$，我们给定一个固定大小的图像块维度 $P \times P$。我们将图像在空间维度上严格划分为不重叠的二维块。块的数量为 $N = \frac{H \times W}{P^2}$。此时，图像可以被重塑为一系列展平的二维块序列 $\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$。

接着，我们通过一个可学习的线性投影矩阵 $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 将这些高维的展平块映射到一个统一的隐藏维度 $D$ 中：

$$
\mathbf{z}_0 = [\mathbf{x}_{class}; \, \mathbf{x}_{p}^1 \mathbf{E}; \, \mathbf{x}_{p}^2 \mathbf{E}; \, \dots; \, \mathbf{x}_{p}^N \mathbf{E}] + \mathbf{E}_{pos}
$$

在这里，$\mathbf{x}_{class} \in \mathbb{R}^{1 \times D}$ 是一个特殊的可学习向量，被称为分类标记（Class Token），它的最终输出状态将被用作整个图像的全局聚合表示。$\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$ 则是位置编码矩阵。与 CNN 固有的位置敏感性不同，Transformer 架构及其基于点积的注意力机制本身是对序列元素的绝对排列位置不可知的（Permutation Invariant）。如果不显式地注入位置信息，模型将无法分辨图像块的相对空间关系。

(**在 PyTorch 中，我们可以使用步幅等于卷积核大小的普通卷积操作，以极其高效且数学上等价的方式实现块嵌入：**)

```{.python .input}
#@tab pytorch
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 使用不重叠的卷积直接提取特征并隐式完成线性投影
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x 的输入形状: (B, C, H, W)
        x = self.proj(x)  # 输出形状: (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # 将空间维度合并并展平: (B, embed_dim, N)
        x = x.transpose(1, 2)  # 转换为标准的序列格式: (B, N, embed_dim)
        return x
```

### 多头自注意力机制 (Multi-Head Self-Attention)

Transformer 架构的灵魂在于自注意力机制。为了建立其严谨的数学基础，我们首先考察单头自注意力的标量形式推导。

给定序列中的某个词元 $\mathbf{x}_i \in \mathbb{R}^D$（为简化符号，假设它已经是投影后的向量），我们需要计算它与序列中其他所有词元 $\mathbf{x}_j$ 之间的相关性。自注意力机制将每个输入投射为三个不同的角色：查询（Query）、键（Key）和值（Value）。

我们引入三个可学习的权重矩阵 $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V \in \mathbb{R}^{D \times D_h}$，其中 $D_h$ 为注意力头的内部特征维度。对于第 $i$ 个位置的查询向量 $\mathbf{q}_i = \mathbf{x}_i \mathbf{W}^Q$ 和第 $j$ 个位置的键向量 $\mathbf{k}_j = \mathbf{x}_j \mathbf{W}^K$，它们之间的注意力得分计算为严格的内积：

$$
s_{i,j} = \mathbf{q}_i \cdot \mathbf{k}_j^T = \sum_{d=1}^{D_h} q_{i,d} k_{j,d}
$$

由于内积的值域会随着特征维度 $D_h$ 的增加而呈指数级剧烈波动，这将导致后续的 Softmax 激活函数迅速进入梯度极小饱和区（即梯度趋近于零），因此我们必须引入缩放因子 $\sqrt{D_h}$。随后使用 Softmax 函数对得分进行概率归一化，得到注意力权重：

$$
\alpha_{i,j} = \frac{\exp(s_{i,j} / \sqrt{D_h})}{\sum_{m=1}^{N+1} \exp(s_{i,m} / \sqrt{D_h})}
$$

最终，第 $i$ 个位置的输出特征是由所有位置的值向量 $\mathbf{v}_j = \mathbf{x}_j \mathbf{W}^V$ 按注意力权重概率进行加权线性组合而成：

$$
\mathbf{o}_i = \sum_{j=1}^{N+1} \alpha_{i,j} \mathbf{v}_j
$$

若我们将上述操作进行矢量化，转换为标准的矩阵形式，设查询、键、值矩阵分别为 $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{(N+1) \times D_h}$，则全局注意力计算公式被优美而凝练地表达为：

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{D_h}}\right)\mathbf{V}
$$

多头注意力（Multi-Head Attention）则是将上述过程在多个并行的子空间中执行 $h$ 次，每次使用独立初始化的投影矩阵。最后将所有头输出的矩阵在隐藏维度上拼接，并通过一个线性层进行最终投影。

> [!NOTE]
> 在极端复杂的长程依赖建模中，自注意力机制在物理上可被看作一种“基于内容的全局软寻址系统”：查询 $\mathbf{Q}$ 在全图范围内并发检索与其高度匹配的键 $\mathbf{K}$，并在建立高概率连接后提取并融合对应的值 $\mathbf{V}$。这与 CNN 通过卷积核感受野在固定网格上层层向上传递局部信号的机制有着根本性的不同。

(**下面是包含多头注意力与前馈网络的 ViT 编码器块的核心实现：**)

```{.python .input}
#@tab pytorch
class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        # 多头注意力层
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)
        # 多层感知机 (MLP)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        # 带有前置层归一化与残差连接的注意力层
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # 带有残差连接的 MLP 层
        x = x + self.mlp(self.norm2(x))
        return x
```

## 小结

在本章中，我们详尽探讨了现代计算机视觉中两大至关重要的基础模型架构。卷积神经网络（CNN）通过严格的局部卷积操作和权重空间共享，确立了强大的空间几何归纳偏置，在中小规模数据集的训练中具有极高的参数效率与稳定性。视觉变换器（ViT）则打破了局部网格感受野的限制，通过将连续图像离散化建模为词元序列，并施加全局自注意力矩阵乘法运算，在全局特征的融合上展现出了无与伦比的上限。
