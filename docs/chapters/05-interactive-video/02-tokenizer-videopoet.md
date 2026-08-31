# 视频 Tokenizer 与 VideoPoet 架构

视频比图像多出时间维度，原始数据量随帧数、分辨率与通道数的乘积增长；自注意力对词元数的开销还可能呈平方增长，但不能笼统称为“指数级”。VideoPoet 把多种视听模态表示为离散词元，并用大型自回归语言模型式架构统一生成 [[Kondratyuk et al., 2023]](https://arxiv.org/abs/2312.14125)；其视频词元来自 MAGVIT-v2 [[Yu et al., 2023]](https://arxiv.org/abs/2310.05737)。本节将讨论视频分词器与自回归生成的对应关系。

## 历史背景与维度灾难

在高中物理中，我们学过运动学，知道所谓的速度和加速度，都是描述物体在连续时间上的空间位置变化。视频的本质，正是在离散的时间间隔上，对连续物理世界的空间快照进行采样。

如果我们有一段时长为 $t$ 秒、帧率为 $f$ 帧/秒的视频，每一帧的空间分辨率为高度 $H$ 和宽度 $W$。由于每个像素包含红、绿、蓝（RGB）三个通道，这部视频可以被严谨地表示为一个高维张量（Tensor）。我们定义原始视频张量 $V$，其维度可以表示为：

$$V \in \mathbb{R}^{T \times H \times W \times 3}$$

其中 $T = t \times f$ 为总帧数。试想一段仅仅 $10$ 秒、帧率 $30$、分辨率为 $1080 \times 1920$ 的短视频，其包含的像素标量总数达到了 $10 \times 30 \times 1080 \times 1920 \times 3 \approx 1.86 \times 10^9$。如果我们要使用神经网络直接对这样一个近20亿个变量的联合概率分布进行建模，在当下的算力条件下是绝对不可能的。这种随着特征维度增加，状态空间呈现指数级爆炸的现象，被称为“维度灾难”（Curse of Dimensionality）。

为了解决这一问题，深度学习领域的先驱们提出了特征压缩的思想。既然相邻两帧图像中的背景往往是一模一样的（高度的时序冗余），且同一帧内相邻像素的颜色也往往相近（高度的空间冗余），我们完全可以把视频映射到一个低维的潜在空间（Latent Space）中去。在这个低维空间里，我们不仅保留了视频的核心语义结构，还大幅降低了数据量。而视频 Tokenizer，正是连接原始像素世界与低维语义空间的桥梁。

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

到目前为止，潜在表示 $Z$ 中的每一个元素仍然是连续的实数。而语言模型只能处理离散的词汇表（Vocabulary）。因此，我们需要将 $Z$ 中的连续向量“翻译”成离散的单词。

在标准的VQ-VAE [[van den Oord et al., 2017]](https://arxiv.org/abs/1711.00937) 框架中，我们定义一个可学习的“字典”或“码本”（Codebook） $\mathcal{C}$。码本包含了 $K$ 个标准参考向量，每个参考向量的长度也是 $d$：

$$\mathcal{C} = \{ e_1, e_2, \dots, e_K \} \subset \mathbb{R}^d$$

对于 $Z$ 中的任意一个空间-时间位置处的特征向量 $z_{t,h,w} \in \mathbb{R}^d$，我们遍历码本中的所有向量，找到与它欧几里得距离最接近的那个参考向量 $e_k$。这就是量化操作 $Q$：

$$k^* = \arg\min_{k \in \{1, 2, \dots, K\}} \| z_{t,h,w} - e_k \|_2^2$$

量化后的特征向量被替换为码本中的那个对应向量，即 $\hat{z}_{t,h,w} = e_{k^*}$。同时，我们可以仅仅记录这个向量在码本中的索引（Index），即整数 $k^*$。通过将所有位置的特征都替换为对应的索引，我们就把原先的连续张量彻底变成了一个由整数组成的离散矩阵。为了后续输入语言模型，我们将其展平为一个一维的离散序列 $S$。

### 无查找表量化 (Lookup-Free Quantization)

上述传统的向量量化方法存在一个致命问题：码本崩溃（Codebook Collapse）。当字典容量 $K$ 很大时，网络往往只会使用字典里极小一部分的向量，导致字典的大部分处于“死状态”（Dead Codes）。为了容纳高质量的视频生成，MAGVIT-v2提出了无查找表量化（Lookup-Free Quantization, LFQ）技术。

> **LFQ 机制的类比（极简解释）**
> 传统的VQ好比你在图书馆里逐一比对 $K$ 本书，找到最像的一本，这种全量查找在 $K$ 达到百万级别时计算量极大；而 LFQ 则好比对你的特征向量做一连串是或否的二元选择题。每个维度你只需判断它是正还是负，就自动确定了它属于哪一个类别。

在 LFQ 中，我们彻底抛弃了显式的字典矩阵。假设我们的潜在特征维度依然是 $d$。对于特征向量 $z \in \mathbb{R}^d$，LFQ 的做法极为暴力且有效——直接利用符号函数（Sign function）将向量的每一位硬性二值化为 $+1$ 或 $-1$ 或 $0,1$。

设特征向量的第 $j$ 个分量为 $z^{(j)}$，量化函数定义为：

$$q(z^{(j)}) = \begin{cases} 1, & \text{if } z^{(j)} > 0 \\ 0, & \text{otherwise} \end{cases}$$

如此一来，整个特征向量被转化为一个由 $0$ 和 $1$ 组成的 $d$ 维布尔向量 $b \in \{0, 1\}^d$。这个布尔向量实际上可以被直接视作一个二进制编码。我们只需将其转换为对应的十进制整数索引 $I$：

$$I = \sum_{j=1}^d q(z^{(j)}) \times 2^{j-1}$$

这种优雅的数学设计带来了巨大的好处：如果我们让通道数 $d = 18$，那么隐式的字典大小 $K$ 将达到 $2^{18} = 262,144$。网络不需要维护一个庞大的字典张量，也不需要执行昂贵的距离搜索（Argmin），量化过程变成了一种纯粹的位运算。这种设计极大地提高了 Tokenizer 在处理高频视频细节时的词汇表容量。

## VideoPoet：万物皆为自回归序列预测

当我们拥有了强大的 MAGVIT-v2 视频 Tokenizer，连续视频世界就被完全离散化为了一串整数序列（Tokens）。这使得我们可以彻底抛弃为了图像定制的卷积网络或扩散模型，转而拥抱自然语言处理领域绝对的霸主：解码器型（Decoder-only）Transformer架构。这也是 VideoPoet 成功的核心所在。

### 自回归生成的主导地位

VideoPoet 采用自回归（Autoregressive, AR）范式作为其唯一的生成引擎。在高中排列组合中，我们知道一个联合事件发生的概率等于各步条件概率的乘积。对于一个由离散视频标记组成的序列 $S = (s_1, s_2, \dots, s_N)$，其联合概率分布可以通过链式法则精确地展开为：

$$P(S) = P(s_1, s_2, \dots, s_N) = \prod_{i=1}^N P(s_i \mid s_1, s_2, \dots, s_{i-1})$$

其中 $P(s_i \mid s_{<i})$ 表示在给定前 $i-1$ 个标记的历史信息下，预测第 $i$ 个标记的条件概率。

在 VideoPoet 中，模型接收来自各个模态（文本提示、起始图像标记、过去的一段视频标记）拼接而成的上下文序列，然后逐个生成未来的视频标记。由于每一个标记 $s_i$ 的预测仅仅依赖于其左侧的历史标记，这种严格的因果注意力机制（Causal Attention）确保了时间方向的单向性和物理现实的一致性。

### 统一序列打包与模态融合

在实际工程中，我们要处理包含多种模态的任务，比如“文本到视频”（Text-to-Video）。假设我们有文本序列（由文本 Tokenizer，如 T5，分词得到）记为 $C_{text}$，我们有希望生成的视频序列记为 $S_{video}$。

VideoPoet 的做法是将所有模态的离散化序列进行首尾拼接（Concatenation）。为了让 Transformer 模型能够区分当前正在处理哪一种模态的数据，必须引入模态标识符（Modality Embeddings）和特定的任务提示（Task Prompts）。拼接后的输入序列往往形如：

$$X = [ \text{<BOS>}, C_{text}, \text{<VID>}, S_{video}, \text{<EOS>} ]$$

这里的 $\text{<VID>}$ 是一个特殊的分割标记（Separator Token），告诉模型接下来要预测的词汇属于视频码本。然后，整个序列 $X$ 将被送入层叠的 Transformer Decoder 块中。

每一个 Transformer 块执行多头自注意力计算（Multi-Head Self-Attention）。对于输入矩阵 $\mathbf{X}$，首先进行线性映射得到查询（$\mathbf{Q}$）、键（$\mathbf{K}$）和值（$\mathbf{V}$）：

$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X} \mathbf{W}_K, \quad \mathbf{W} = \mathbf{X} \mathbf{W}_V$$

自注意力的核心在于利用查询与键的点积来衡量不同标记之间的相关性。为了保证自回归生成的因果性，必须引入一个下三角的掩码矩阵（Mask Matrix） $\mathbf{M}$，使得当前标记无法“看到”未来的标记：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V}$$

在这套纯粹的下一个词预测（Next-token Prediction）框架下，VideoPoet 仅通过最大化负对数似然损失（Negative Log-Likelihood）即可端到端地完成训练。没有任何针对视频物理先验的特化设计，所有的动态规律、光影变化和相机运镜，全部被隐含在海量视频序列的联合概率分布之中。这种“大道至简”的设计哲学，正是当今多模态大模型的演进方向。

## 代码实现：构建简易无查找表量化器 (LFQ)

在这一部分，我们将通过代码演示如何实现基于 MAGVIT-v2 核心思想的简化版 Lookup-Free Quantization (LFQ)。虽然实际生产中的模型包含了复杂的三维卷积残差网络和熵惩罚项（Entropy Penalty），但 LFQ 的核心量化逻辑却异常简洁。

我们将定义一个量化器模块，它将连续特征硬性二值化，并支持梯度的直通估计器（Straight-Through Estimator, STE）。因为符号函数 $q(x) = \text{sign}(x)$ 在各处导数为零，无法使用标准反向传播，我们将通过 `x + (q(x) - x).detach()` 这一精妙的操作，在正向传播时保留量化效果，反向传播时将梯度完美绕过量化步骤。

(**构建基于布尔化与直通估计器的 LFQ 模块**)

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
        powers = torch.arange(codebook_dim, dtype=torch.float32)
        self.register_buffer('binary_weights', 2 ** powers)

    def forward(self, z):
        """
        前向传播函数。
        输入 z 的维度应为 (Batch, Channels, T, H, W)，此处设通道维度在前。
        """
        # 将输入移动通道维到最后，方便后续处理
        # z: (B, T, H, W, C) 假设此时 C 等于 codebook_dim

        # 1. 严格的二值化量化 (Binarization)
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

通过如上代码可以看到，传统 VQ 需要在内存中持久化巨大的查表矩阵并计算欧式距离，而基于 LFQ 思想的 Tokenizer 仅仅使用了最基础的符号操作，就在微秒内得到了具备极大表征容量的 Token 离散索引。

## 小结

- 视频生成的核心挑战在于极高的维度，通过**三维卷积结合向量量化**，我们可以将连续的高维视频压缩为一维离散符号序列。
- **无查找表量化（LFQ）**巧妙地利用正负号的二进制属性替代了昂贵的欧氏距离搜索与字典查表，有效避免了字典崩塌问题，支撑起极大的潜在表征词汇量。
- **VideoPoet** 将视频生成完全转化为经典的自回归语言模型任务。在这种框架下，文本、图像、音频和视频等一切模态最终都可以拼接到一个统一的上下文张量中，依赖 Transformer 强大的因果自注意力机制进行下一个词预测。
