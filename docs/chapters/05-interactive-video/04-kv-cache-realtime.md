# KV Cache 与视频生成的实时推理加速

## 引言与学术脉络

Transformer 自回归生成需要反复访问历史词元 [[Vaswani et al., 2017]](https://arxiv.org/abs/1706.03762)。视频把高度、宽度与时间三个维度离散化后，词元数会随三者的乘积增长，而不是“指数级爆炸”。键值缓存可以避免在每一步重复计算旧词元的键和值 [[Pope et al., 2022]](https://arxiv.org/abs/2211.05102)；PagedAttention 则主要改善服务系统中 KV 缓存的内存分配与共享 [[Kwon et al., 2023]](https://arxiv.org/abs/2309.06180)。它们能降低延迟和内存浪费，但原论文并未证明所有视频模型都能因此从秒级变为毫秒级。

本节将从最基础的序列求和思想出发，严格推导自回归解码中的冗余计算，并详细解析 KV Cache 的数学机制及其在视频时空维度上的扩展。

## 自回归解码的效率瓶颈

让我们回想高中数学中的数列求和问题。假设我们需要计算一个数列的前 $t$ 项和 $S_t = \sum_{i=1}^{t} a_i$。
如果我们已经知道了前 $t-1$ 项的和 $S_{t-1}$，那么计算 $S_t$ 只需要简单的加法：
$$S_t = S_{t-1} + a_t$$

如果我们在每一步计算 $S_t$ 时，都愚蠢地从第一项 $a_1$ 一直加到 $a_t$，那么计算第 $t$ 项的时间复杂度将是 $O(t)$，计算前 $T$ 项的总时间复杂度将是 $O(T^2)$。这种“从头重算”的做法，恰恰就是朴素的 Transformer 模型在生成序列时所犯的错误。

在自回归模型中，给定前 $t-1$ 个词元的序列 $x_{1:t-1}$，模型需要预测第 $t$ 个词元 $x_t$。在标准的多头自注意力（Multi-Head Self-Attention）机制中，输入序列首先被映射为查询（Query）、键（Key）和值（Value）矩阵。

设我们在第 $t$ 步的输入是一个形状为 $t \times d$ 的矩阵 $\mathbf{X}_t \in \mathbb{R}^{t \times d}$，其中 $d$ 是隐藏层的特征维度。通过线性变换，我们得到：
$$\mathbf{Q}_t = \mathbf{X}_t \mathbf{W}_q, \quad \mathbf{K}_t = \mathbf{X}_t \mathbf{W}_k, \quad \mathbf{V}_t = \mathbf{X}_t \mathbf{W}_v$$

其中 $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v \in \mathbb{R}^{d \times d_k}$ 是学习到的权重矩阵。此时，注意力输出可严格表示为：
$$\text{Attention}(\mathbf{Q}_t, \mathbf{K}_t, \mathbf{V}_t) = \text{softmax}\left(\frac{\mathbf{Q}_t \mathbf{K}_t^\top}{\sqrt{d_k}}\right) \mathbf{V}_t$$

在这个公式中，$\mathbf{K}_t$ 和 $\mathbf{V}_t$ 是由前 $t$ 个词元的完整历史计算得来的。当我们要生成第 $t+1$ 个词元时，输入的矩阵变为 $\mathbf{X}_{t+1} \in \mathbb{R}^{(t+1) \times d}$。若按照上述公式直接计算，我们会重新计算前 $t$ 个词元的 $\mathbf{K}$ 和 $\mathbf{V}$。随着 $t$ 的增长，计算量与 $t^2$ 成正比，这在视频生成（动辄数万个词元）中是绝对无法接受的。

## 键值缓存 (KV Cache) 的严密推导

为了消除上述冗余，我们需要对自注意力机制进行数学上的解耦。

考虑在生成第 $t$ 个词元时，我们实际上只关心序列的最后一个查询向量 $\mathbf{q}_t \in \mathbb{R}^{1 \times d_k}$ 对历史所有键 $\mathbf{K}_t$ 的注意力响应。矩阵 $\mathbf{X}_t$ 可以被分块（Block-partitioned）为前 $t-1$ 个词元的历史部分 $\mathbf{X}_{<t} \in \mathbb{R}^{(t-1) \times d}$ 和当前步骤的新词元 $\mathbf{x}_t \in \mathbb{R}^{1 \times d}$：
$$\mathbf{X}_t = \begin{bmatrix} \mathbf{X}_{<t} \\ \mathbf{x}_t \end{bmatrix}$$

因此，键矩阵 $\mathbf{K}_t$ 和值矩阵 $\mathbf{V}_t$ 也自然地可以写成分块形式：
$$\mathbf{K}_t = \begin{bmatrix} \mathbf{K}_{<t} \\ \mathbf{k}_t \end{bmatrix} = \begin{bmatrix} \mathbf{X}_{<t} \mathbf{W}_k \\ \mathbf{x}_t \mathbf{W}_k \end{bmatrix}$$

$$\mathbf{V}_t = \begin{bmatrix} \mathbf{V}_{<t} \\ \mathbf{v}_t \end{bmatrix} = \begin{bmatrix} \mathbf{X}_{<t} \mathbf{W}_v \\ \mathbf{x}_t \mathbf{W}_v \end{bmatrix}$$

对于查询，由于自回归解码的因果性（Causal property），历史词元的输出在之前的步骤中已经计算完毕并固定下来。当前步骤只需处理最新的查询向量 $\mathbf{q}_t = \mathbf{x}_t \mathbf{W}_q$。

此时，第 $t$ 步的注意力得分向量 $\mathbf{s}_t \in \mathbb{R}^{1 \times t}$ 为：
$$\mathbf{s}_t = \mathbf{q}_t \mathbf{K}_t^\top = \mathbf{q}_t \begin{bmatrix} \mathbf{K}_{<t}^\top & \mathbf{k}_t^\top \end{bmatrix} = \begin{bmatrix} \mathbf{q}_t \mathbf{K}_{<t}^\top & \mathbf{q}_t \mathbf{k}_t^\top \end{bmatrix}$$

应用 softmax 函数后得到注意力权重 $\mathbf{\alpha}_t = \text{softmax}\left(\frac{\mathbf{s}_t}{\sqrt{d_k}}\right)$。最终的注意力输出向量 $\mathbf{o}_t \in \mathbb{R}^{1 \times d_k}$ 为：
$$\mathbf{o}_t = \mathbf{\alpha}_t \mathbf{V}_t = \mathbf{\alpha}_t \begin{bmatrix} \mathbf{V}_{<t} \\ \mathbf{v}_t \end{bmatrix} = \sum_{i=1}^{t} \alpha_{t,i} \mathbf{v}_i$$

通过这两个公式，我们发现 $\mathbf{K}_{<t}$ 和 $\mathbf{V}_{<t}$ 的值完全等同于在第 $t-1$ 步时计算的结果。因此，只要我们在显存中开辟一块空间，将之前计算的 $\mathbf{k}_i$ 和 $\mathbf{v}_i$ 缓存下来，在第 $t$ 步时只需计算新的 $\mathbf{q}_t$、$\mathbf{k}_t$ 和 $\mathbf{v}_t$。然后将 $\mathbf{k}_t$ 和 $\mathbf{v}_t$ 拼接到缓存中，即可完成整个自注意力计算。
这就是 KV Cache 的核心数学原理，它将每一步的计算复杂度从 $O(t^2)$ 严格降低到了 $O(t)$。

> 唯一的精炼类比：
> 可以将朴素的自回归解码想象为一个完全没有记忆力的人在朗读一本不断翻页的书，每次为了读出新的一页，他必须从第一页开始大声朗读。而引入 KV Cache 后，这个人拥有了短期记忆（显存），他只需在脑海中回忆之前的故事情节（缓存），然后结合眼前的这一页（当前 Token），就能直接继续讲下去。

## 视频生成中的时空 KV Cache

文本生成中的 KV Cache 仅仅处理一维的时间序列。但在视频生成中，视频数据被三维张量所描述（帧时间 $T$、高度 $H$、宽度 $W$）。

假设我们将视频压缩为潜在表示（Latent representation），每个时空补丁（Patch）被视为一个词元。若帧数为 $F$，每帧的高度和宽度方向上分别有 $h$ 和 $w$ 个补丁，则总的序列长度将达到 $L = F \times h \times w$。
例如，生成一段包含 16 帧、特征图分辨率为 $32 \times 32$ 的视频，其序列长度 $L = 16 \times 32 \times 32 = 16384$。这比普通的文本序列长几个数量级。

此时，KV Cache 会面临严重的**显存墙（Memory Wall）**问题。每个浮点数占用 2 个字节（FP16），一个注意力头的缓存大小不仅与隐藏维度相关，更与序列长度 $L$ 成正比。随着序列长度急剧增加，显存往往被迅速耗尽。

为了解决 KV 缓存的碎片化与重复存储问题，vLLM 提出了**分页注意力（PagedAttention）**机制 [[Kwon et al., 2023]](https://arxiv.org/abs/2309.06180)。这一工作针对大语言模型服务；把它迁移到长上下文视频模型，是基于相同内存问题的工程推广。
在操作系统中，内存被划分为固定大小的页（Pages）以减少碎片；分页注意力同样将连续的键值向量划分为独立的块（Blocks）。
设每个块能容纳 $B$ 个词元的键值对，则键缓存 $\mathbf{K}$ 可以表示为一系列非连续的张量块的集合：
$$\mathcal{K} = \{ \mathbf{K}^{(1)}, \mathbf{K}^{(2)}, \dots, \mathbf{K}^{(M)} \}$$

其中 $M = \lceil \frac{L}{B} \rceil$，每个 $\mathbf{K}^{(m)} \in \mathbb{R}^{B \times d_k}$。
在注意力计算时，模型无需分配一整块连续的显存空间，而是通过一个专门的块表（Block Table）来记录逻辑上的时空词元序列与物理显存块的映射关系。当视频的自回归生成跨越到新的时间帧或空间行时，系统动态分配新的物理块，从而极大地提高了显存利用率，并支持了更高分辨率视频的实时推理。

## 代码实现与张量分析

下面，我们将从零开始实现一个带有 KV Cache 的因果自注意力层。

(**在实现代码时，我们需要密切关注张量的形状变化。**) 特别是缓存中键和值序列长度的动态拼接。

```{.python .input}
#@tab pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KVCacheSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        """
        x: 当前输入的张量，形状为 (batch_size, seq_len, d_model)
           如果是第一步，seq_len 为提示序列长度；
           如果是后续自回归步，seq_len 通常为 1。
        kv_cache: 一个元组 (k_cache, v_cache)，包含过去缓存的键和值。
                  它们各自的形状通常为 (batch_size, num_heads, past_seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape

        # 1. 计算当前步骤的 Q, K, V
        # 形状变为 (batch_size, seq_len, num_heads, d_k)，然后转置以利用多头并行
        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # 此时 q, k, v 的形状: (batch_size, num_heads, seq_len, d_k)

        # 2. KV 缓存的管理
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # 将过去的缓存与当前步计算的 k, v 在序列维度（dim=2）上拼接
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
            # 拼接后 k, v 形状: (batch_size, num_heads, past_seq_len + seq_len, d_k)

        # 更新当前的缓存以供下一步使用
        new_kv_cache = (k, v)

        # 3. 计算注意力得分
        # q 的最后两个维度: (seq_len, d_k)
        # k 转置后的最后两个维度: (d_k, past_seq_len + seq_len)
        # 矩阵乘法后 scores 形状: (batch_size, num_heads, seq_len, past_seq_len + seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 因果掩码 (如果是自回归生成阶段且 seq_len=1，掩码是不必要的)
        # 这里为了完整性，在 seq_len > 1 (如 Prefill 阶段) 时加上掩码
        if seq_len > 1:
            # past_seq_len 取决于拼接后的总长度减去当前长度
            total_seq_len = k.size(2)
            mask = torch.tril(torch.ones(seq_len, total_seq_len)).unsqueeze(0).unsqueeze(0).to(x.device)
            # 将下三角外的位置设为负无穷
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)

        # 4. 聚合值向量
        # attn_weights 形状: (batch_size, num_heads, seq_len, past_seq_len + seq_len)
        # v 形状: (batch_size, num_heads, past_seq_len + seq_len, d_k)
        # output 形状: (batch_size, num_heads, seq_len, d_k)
        out = torch.matmul(attn_weights, v)

        # 重塑并经过输出线性层
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)

        return out, new_kv_cache
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
import math

class KVCacheSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = tf.keras.layers.Dense(d_model)
        self.W_k = tf.keras.layers.Dense(d_model)
        self.W_v = tf.keras.layers.Dense(d_model)
        self.W_o = tf.keras.layers.Dense(d_model)

    def call(self, x, kv_cache=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # 计算当前步骤的 Q, K, V
        q = tf.reshape(self.W_q(x), (batch_size, seq_len, self.num_heads, self.d_k))
        k = tf.reshape(self.W_k(x), (batch_size, seq_len, self.num_heads, self.d_k))
        v = tf.reshape(self.W_v(x), (batch_size, seq_len, self.num_heads, self.d_k))

        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        # KV 缓存的管理
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = tf.concat([k_cache, k], axis=2)
            v = tf.concat([v_cache, v], axis=2)

        new_kv_cache = (k, v)

        # 计算注意力得分
        scores = tf.matmul(q, k, transpose_b=True) / math.sqrt(float(self.d_k))

        if seq_len is not None and seq_len > 1:
            total_seq_len = tf.shape(k)[2]
            # 动态生成下三角矩阵以做因果掩码
            mask = tf.linalg.band_part(tf.ones((seq_len, total_seq_len)), -1, 0)
            mask = tf.reshape(mask, (1, 1, seq_len, total_seq_len))
            # 将 0 的位置填充大负数
            scores = tf.where(tf.equal(mask, 1), scores, tf.fill(tf.shape(scores), -1e9))

        attn_weights = tf.nn.softmax(scores, axis=-1)

        # 聚合值向量
        out = tf.matmul(attn_weights, v)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (batch_size, seq_len, self.d_model))

        out = self.W_o(out)
        return out, new_kv_cache
```

在推理阶段，我们通常将计算过程严谨地划分为两部分：

1. **预填充阶段（Prefill Stage）**：一次性将初始条件或上下文提示（Prompt）完整输入模型，利用大矩阵乘法并行计算出最初始的、完整的历史 KV Cache。
2. **解码阶段（Decoding Stage）**：处于真正的自回归循环中，此时 `seq_len=1`，模型仅利用最新生成的一个时空词元作为输入。借助已经构建好的缓存，模型将矩阵乘法退化为极速的矩阵-向量乘法，从而高速产出下一词元。

## 小结

- 朴素自回归解码的时间复杂度随着序列长度呈现出严峻的二次方增长，使得大规模视频生成推理举步维艰。
- 借由自注意力机制内在的线性组合与解耦性质，我们严格推导出键值缓存（KV Cache）不仅是计算上的近似，而是数学上精确等效的计算优化，大幅压低了自回归生成的计算时延。
- 视频数据的三维特性导致时空词元激增，随之而来的显存墙问题进一步催生了类似操作系统管理内存的分页注意力（PagedAttention）技术。这对于深入理解与部署前沿的实时视频生成模型至关重要。
