# 5.2 视频 Tokenizer 与 VideoPoet：把画面变成词元

视频比图像多出时间维度，原始数据量随帧数、分辨率与通道数的乘积增长；自注意力对词元数的开销还可能呈平方增长，但不能笼统称为“指数级”。VideoPoet 把多种视听模态表示为离散词元，并用大型自回归语言模型式架构统一生成 [[Kondratyuk et al., 2023]](https://arxiv.org/abs/2312.14125)；其视频词元来自 MAGVIT-v2 [[Yu et al., 2023]](https://arxiv.org/abs/2310.05737)。本节将讨论视频分词器与自回归生成的对应关系。

<div align="center">
<img src="/figures/05-interactive-video/source/02-tokenizer-videopoet/videopoet-fig1.png" alt="VideoPoet 从文本、图像、深度和视频条件生成多类视听输出，展示统一离散序列接口的实际任务跨度。" width="86%">

_图 5.2-1：VideoPoet 从文本、图像、深度和视频条件生成多类视听输出，展示统一离散序列接口的实际任务跨度。 出处：Dan Kondratyuk et al.，[VideoPoet: A Large Language Model for Zero-Shot Video Generation](https://arxiv.org/abs/2312.14125)（2023），Figure 1。_
</div>

## 历史背景与维度灾难

在高中物理中，我们学过运动学，知道所谓的速度和加速度，都是描述物体在连续时间上的空间位置变化。视频的本质，正是在离散的时间间隔上，对连续物理世界的空间快照进行采样。

若一段视频时长为 $t$ 秒、帧率为 $f$，每帧高 $H$、宽 $W$，那么 RGB 视频可以表示为：

$$V \in \mathbb{R}^{T \times H \times W \times 3}$$

其中 $T=t\times f$。一段 10 秒、30 帧/秒、$1080\times1920$ 的视频含约 $1.86\times10^9$ 个像素标量。模型可以分块或流式处理原始视频，但直接把每个像素都当作长序列词元会带来很高的存储与计算开销；若再离散化每个变量，组合状态数还会随维度指数增长。

视频在时间和空间上通常存在冗余，因此可以先映射到更紧凑的潜在空间。压缩是有损的：Tokenizer 的目标是在减少词元数的同时，尽量保留后续生成与重建需要的信息。

<div align="center">
<img src="/figures/05-interactive-video/source/02-tokenizer-videopoet/magvitv2-fig2.png" alt="MAGVIT-v2 对比三种因果视频编码器，显示时间下采样与空间下采样怎样组合成视觉词元。" width="86%">

_图 5.2-2：MAGVIT-v2 对比三种因果视频编码器，显示时间下采样与空间下采样怎样组合成视觉词元。 出处：Lijun Yu et al.，[Language Model Beats Diffusion — Tokenizer is Key to Visual Generation](https://arxiv.org/abs/2310.05737)（2023），Figure 2。_
</div>

## 视频 Tokenizer：时空维度的量化自编码

视频 Tokenizer 的目标是将高维连续视频张量 $V$ 转换为一维的离散整数序列 $S$。这个过程分为两步：首先是时空联合下采样，其次是向量量化（Vector Quantization）。

### 时空编码器（Encoder）与下采样

我们先从最简单的二维平面考虑。假设我们有一个标量序列，我们可以通过移动平均来提取特征。同理，对于具有时间维度的3D张量，我们使用三维卷积网络（3D-CNN）。三维卷积核不仅在高度 $H$ 和宽度 $W$ 上滑动，还在时间 $T$ 上滑动。

假设编码器记作 $\mathcal{E}$，它将原始视频 $V$ 映射为一个隐状态张量 $Z$：

$$Z = \mathcal{E}(V)$$

如果下采样率在时间、高度、宽度维度上分别为 $s_t, s_h, s_w$，那么潜在张量 $Z$ 的维度将会变为：

$$Z \in \mathbb{R}^{T' \times H' \times W' \times d}$$

其中 $T' = \frac{T}{s_t}$，$H' = \frac{H}{s_h}$，$W' = \frac{W}{s_w}$，而 $d$ 是编码器输出特征向量的通道维度。此时，原视频被划分为一个个微小的“时空立方体”（Spatiotemporal Patches），每个立方体由一个长度为 $d$ 的特征向量表示。

### 向量量化（Vector Quantization）与码本

潜在表示 $Z$ 仍是连续实数。Transformer 本身处理连续向量，但 VideoPoet 采用离散词元接口，因此需要把每个潜在向量映射为有限词汇表中的索引。

在标准的VQ-VAE [[van den Oord et al., 2017]](https://arxiv.org/abs/1711.00937) 框架中，我们定义一个可学习的“字典”或“码本”（Codebook） $\mathcal{C}$。码本包含了 $K$ 个标准参考向量，每个参考向量的长度也是 $d$：

$$\mathcal{C} = \{ e_1, e_2, \dots, e_K \} \subset \mathbb{R}^d$$

对于 $Z$ 中的任意一个空间-时间位置处的特征向量 $z_{t,h,w} \in \mathbb{R}^d$，我们遍历码本中的所有向量，找到与它欧几里得距离最接近的那个参考向量 $e_k$。这就是量化操作 $Q$：

$$k^* = \arg\min_{k \in \{1, 2, \dots, K\}} \| z_{t,h,w} - e_k \|_2^2$$

量化后的特征向量被替换为码本中的对应向量，即 $\hat{z}_{t,h,w} = e_{k^*}$。同时只需记录其码本索引 $k^*$。所有位置量化后，连续张量就得到一个整数索引矩阵；再按约定顺序展平，便得到供序列模型处理的离散序列 $S$。

### 无查找表量化 (Lookup-Free Quantization)

传统向量量化需要显式码本和最近邻搜索，也可能出现码本利用率不足。MAGVIT-v2 引入无查找表量化（Lookup-Free Quantization，LFQ），用各维符号的组合隐式定义码字，从而支持很大的词汇表 [[Yu et al., 2023]](https://arxiv.org/abs/2310.05737)。这改善了计算和利用率，但不保证训练中完全没有表示退化。

<div align="center">
<img src="/figures/05-interactive-video/source/02-tokenizer-videopoet/fsq-fig1.png" alt="FSQ 将连续编码逐维限制并取整，提供不依赖最近邻码本的另一种离散化机制。" width="86%">

_图 5.2-3：FSQ 将连续编码逐维限制并取整，提供不依赖最近邻码本的另一种离散化机制。 出处：Fabian Mentzer et al.，[Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505)（2023），Figure 1。_
</div>

> **LFQ 机制的类比（极简解释）**
> 传统的VQ好比你在图书馆里逐一比对 $K$ 本书，找到最像的一本，这种全量查找在 $K$ 达到百万级别时计算量极大；而 LFQ 则好比对你的特征向量做一连串是或否的二元选择题。每个维度你只需判断它是正还是负，就自动确定了它属于哪一个类别。

LFQ 不保存显式字典矩阵。对特征向量 $z\in\mathbb{R}^d$，每一维先按符号量化为 $-1$ 或 $+1$；为了计算整数索引，再把这两种取值映射为比特 $0$ 或 $1$。

设特征向量的第 $j$ 个分量为 $z^{(j)}$，量化函数定义为：

$$q(z^{(j)}) = \begin{cases} 1, & \text{if } z^{(j)} > 0 \\ 0, & \text{otherwise} \end{cases}$$

如此一来，整个特征向量被转化为一个由 $0$ 和 $1$ 组成的 $d$ 维布尔向量 $b \in \{0, 1\}^d$。这个布尔向量实际上可以被直接视作一个二进制编码。我们只需将其转换为对应的十进制整数索引 $I$：

$$I = \sum_{j=1}^d q(z^{(j)}) \times 2^{j-1}$$

<div align="center">
<img src="/figures/05-interactive-video/latex/02-tokenizer-videopoet/lfq-bits-to-index.png" alt="连续潜向量各通道按正负号量化为比特，再按二进制位权求和得到整数词元索引" width="86%">

_图 5.2-4：LFQ 先把每个通道阈值化为一位 0/1，再按通道位置赋予二进制位权；例如 1、0、1、1 对应索引 13。本文根据上式绘制。_
</div>

若 $d=18$，可表示的二进制组合数为 $2^{18}=262{,}144$。模型不需要维护同等大小的显式向量表，也不需要对所有码字做最近邻搜索；实际 MAGVIT-v2 还结合熵相关正则等训练设计来提高码字利用率。

## VideoPoet：万物皆为自回归序列预测

MAGVIT-v2 把视频压缩为整数词元后，VideoPoet 可以用解码器型 Transformer 统一建模文本、图像、视频与音频词元。这里的“统一”指共享序列模型与任务接口，并不意味着视觉编解码器或模态专用 Tokenizer 被取消。

### 自回归生成的主导地位

VideoPoet 的核心生成模型采用自回归（Autoregressive，AR）范式。对离散视频词元序列 $S=(s_1,s_2,\dots,s_N)$，概率链式法则给出：

$$P(S) = P(s_1, s_2, \dots, s_N) = \prod_{i=1}^N P(s_i \mid s_1, s_2, \dots, s_{i-1})$$

其中 $P(s_i \mid s_{<i})$ 表示在给定前 $i-1$ 个标记的历史信息下，预测第 $i$ 个标记的条件概率。

在 VideoPoet 中，模型接收由文本提示、起始图像词元或历史视频词元等模态拼接而成的上下文，再逐个生成目标词元。因果注意力使 $s_i$ 只能读取左侧历史，防止训练时泄漏未来答案；生成是否符合物理现实，还取决于数据和模型学到的规律。

### 统一序列打包与模态融合

在实际工程中，我们要处理包含多种模态的任务，比如“文本到视频”（Text-to-Video）。假设我们有文本序列（由文本 Tokenizer，如 T5，分词得到）记为 $C_{text}$，我们有希望生成的视频序列记为 $S_{video}$。

VideoPoet 的做法是将所有模态的离散化序列进行首尾拼接（Concatenation）。为了让 Transformer 模型能够区分当前正在处理哪一种模态的数据，必须引入模态标识符（Modality Embeddings）和特定的任务提示（Task Prompts）。拼接后的输入序列往往形如：

<div align="center">
<img src="/figures/05-interactive-video/source/02-tokenizer-videopoet/videopoet-fig2.png" alt="VideoPoet 的序列布局把任务前缀、条件词元和目标词元排成统一自回归训练序列。" width="86%">

_图 5.2-5：VideoPoet 的序列布局把任务前缀、条件词元和目标词元排成统一自回归训练序列。 出处：Dan Kondratyuk et al.，[VideoPoet: A Large Language Model for Zero-Shot Video Generation](https://arxiv.org/abs/2312.14125)（2023），Figure 2。_
</div>

$$X = [ \text{<BOS>}, C_{text}, \text{<VID>}, S_{video}, \text{<EOS>} ]$$

这里的 $\text{<VID>}$ 是一个特殊的分割标记（Separator Token），告诉模型接下来要预测的词汇属于视频码本。然后，整个序列 $X$ 将被送入层叠的 Transformer Decoder 块中。

每一个 Transformer 块执行多头自注意力计算（Multi-Head Self-Attention）。对于输入矩阵 $\mathbf{X}$，首先进行线性映射得到查询（$\mathbf{Q}$）、键（$\mathbf{K}$）和值（$\mathbf{V}$）：

$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{X} \mathbf{W}_V$$

自注意力的核心在于利用查询与键的点积来衡量不同标记之间的相关性。为了保证自回归生成的因果性，必须引入一个下三角的掩码矩阵（Mask Matrix） $\mathbf{M}$，使得当前标记无法“看到”未来的标记：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V}$$

训练时，模型最小化目标词元的负对数似然，等价于最大化其条件对数似然。视频中的运动、外观变化和镜头模式由训练数据与模型共同学习；这不保证模型得到真实物理规律，也不排除 Tokenizer、任务前缀和模态词汇等专门设计。

## 代码实现：构建简易无查找表量化器 (LFQ)

在这一部分，我们将通过代码演示如何实现基于 MAGVIT-v2 核心思想的简化版 Lookup-Free Quantization (LFQ)。虽然实际生产中的模型包含了复杂的三维卷积残差网络和熵惩罚项（Entropy Penalty），但 LFQ 的核心量化逻辑却异常简洁。

下面的量化器把连续特征二值化，并用直通估计器（Straight-Through Estimator，STE）近似梯度。表达式 `x + (q(x) - x).detach()` 在前向取量化值，在反向把梯度近似为恒等映射；这是有偏估计，不是符号函数的真实导数。

```python
import torch
from torch import nn

class LookupFreeQuantizer(nn.Module):
    def __init__(self, codebook_dim):
        """
        初始化LFQ模块。
        codebook_dim: 潜在特征的通道维度 d。
                      隐式码本大小将为 2^d。
        """
        super().__init__()
        self.codebook_dim = codebook_dim
        # 创建一个2的幂次权重向量，用于将二进制编码转换为十进制索引
        # weight = [1, 2, 4, 8, ..., 2^(d-1)]
        powers = torch.arange(codebook_dim, dtype=torch.long)
        self.register_buffer('binary_weights', 2 ** powers)

    def forward(self, z):
        """
        前向传播函数。
        输入 z 的维度为 (Batch, Time, Height, Width, Channels)。
        """
        # z: (B, T, H, W, C)，其中 C 等于 codebook_dim

        # 1. 二值化量化 (Binarization)
        # 将 z 转换为 -1 或 1
        z_quantized = torch.sign(z)

        # 处理恰好为 0 的异常值，强制其为 1
        z_quantized = z_quantized + (z_quantized == 0).float()

        # 2. 直通估计器 (Straight-Through Estimator)
        # 在正向传播时，z_ste 的值等于 z_quantized。
        # 在反向传播时，z_quantized - z 被切断梯度，梯度直接传给原始 z。
        z_ste = z + (z_quantized - z).detach()

        # 3. 将 -1, 1 映射为 0, 1 二进制布尔分布
        binary_indices = (z_quantized > 0).float()

        # 4. 计算整数索引：布尔张量与权重内积
        # 最终得到的 indices 维度为 (B, T, H, W)，值域为 [0, 2^d - 1]
        indices = torch.sum(binary_indices * self.binary_weights, dim=-1).long()

        return z_ste, indices

# 模拟一个经过 3D 编码器提取出的微型连续潜在特征图
# 维度：(Batch=2, Time=4, Height=8, Width=8, Channels=8)
latent_features = torch.randn(2, 4, 8, 8, 8)
quantizer = LookupFreeQuantizer(codebook_dim=8)

quantized_features, token_indices = quantizer(latent_features)
print("量化后特征图的形状:", quantized_features.shape)
print("离散 Token 索引序列的形状:", token_indices.shape)
print("部分离散 Token 的值:", token_indices[0, 0, 0, :5])
print(f"最大可能索引值 (码本大小-1): {2**8 - 1}")
```

这个示例省略了编码器、解码器、熵正则和分布式训练，只展示“符号组合如何变成整数索引”。与显式 VQ 相比，它避免了对大型码本逐项计算欧氏距离。

## 小结

- 视频生成的核心挑战在于极高的维度，通过**三维卷积结合向量量化**，我们可以将连续的高维视频压缩为一维离散符号序列。
- **无查找表量化（LFQ）**用二值组合替代显式码本搜索，并配合正则项改善大词汇表的利用率。
- **VideoPoet** 把多种模态转换为离散词元，用共享的自回归 Transformer 完成不同条件生成任务。
