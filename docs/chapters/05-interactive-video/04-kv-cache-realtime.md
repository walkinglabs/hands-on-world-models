# 5.4 KV Cache：减少自回归视频的重复计算

## 引言与学术脉络

Transformer 自回归生成需要反复访问历史词元 [[Vaswani et al., 2017]](https://arxiv.org/abs/1706.03762)。视频把高度、宽度与时间三个维度离散化后，词元数会随三者的乘积增长，而不是“指数级爆炸”。键值缓存可以避免在每一步重复计算旧词元的键和值 [[Pope et al., 2022]](https://arxiv.org/abs/2211.05102)；PagedAttention 则主要改善服务系统中 KV 缓存的内存分配与共享 [[Kwon et al., 2023]](https://arxiv.org/abs/2309.06180)。它们能降低延迟和内存浪费，但原论文并未证明所有视频模型都能因此从秒级变为毫秒级。

<div align="center">
<img src="/figures/05-interactive-video/source/04-kv-cache-realtime/vllm-fig1.png" alt="vLLM 的显存布局显示服务时 KV cache 会随并发请求动态占据大量空间，使缓存管理成为实际吞吐瓶颈。" width="86%">

_图 5.4-1：vLLM 的显存布局显示服务时 KV cache 会随并发请求动态占据大量空间，使缓存管理成为实际吞吐瓶颈。 出处：Woosuk Kwon et al.，[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)（2023），Figure 1。_
</div>

本节从序列递推出发，推导自回归解码中的重复计算，再实现一个支持 prefill 和单步 decode 的 KV Cache 注意力层。PagedAttention 的原始证据来自大语言模型服务，把它用于视频系统时需要单独验证。

## 自回归解码的效率瓶颈

让我们回想高中数学中的数列求和问题。假设我们需要计算一个数列的前 $t$ 项和 $S_t = \sum_{i=1}^{t} a_i$。
如果我们已经知道了前 $t-1$ 项的和 $S_{t-1}$，那么计算 $S_t$ 只需要简单的加法：
$$S_t = S_{t-1} + a_t$$

若每一步都从 $a_1$ 重新加到 $a_t$，第 $t$ 步为 $O(t)$，前 $T$ 步合计为 $O(T^2)$。Transformer 的具体复杂度不同，但“旧结果被重复计算”的递推问题相同。

<div align="center">
<img src="/figures/05-interactive-video/source/04-kv-cache-realtime/txl-fig2.png" alt="Transformer-XL 的跨段记忆把前一段隐藏状态复用于后一段，展示缓存历史表示的早期结构。" width="86%">

_图 5.4-2：Transformer-XL 的跨段记忆把前一段隐藏状态复用于后一段，展示缓存历史表示的早期结构。 出处：Zihang Dai et al.，[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)（2019），Figure 2。_
</div>

在自回归模型中，给定前 $t-1$ 个词元的序列 $x_{1:t-1}$，模型需要预测第 $t$ 个词元 $x_t$。在标准的多头自注意力（Multi-Head Self-Attention）机制中，输入序列首先被映射为查询（Query）、键（Key）和值（Value）矩阵。

设我们在第 $t$ 步的输入是一个形状为 $t \times d$ 的矩阵 $\mathbf{X}_t \in \mathbb{R}^{t \times d}$，其中 $d$ 是隐藏层的特征维度。通过线性变换，我们得到：
$$\mathbf{Q}_t = \mathbf{X}_t \mathbf{W}_q, \quad \mathbf{K}_t = \mathbf{X}_t \mathbf{W}_k, \quad \mathbf{V}_t = \mathbf{X}_t \mathbf{W}_v$$

其中 $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v \in \mathbb{R}^{d \times d_k}$ 是学习到的权重矩阵。注意力输出为：
$$\text{Attention}(\mathbf{Q}_t, \mathbf{K}_t, \mathbf{V}_t) = \text{softmax}\left(\frac{\mathbf{Q}_t \mathbf{K}_t^\top}{\sqrt{d_k}}\right) \mathbf{V}_t$$

若每一步都把完整前缀重新送入模型，除了重复计算旧词元的 K/V 投影，还会重算前缀内部的注意力。单层注意力在长度 $t$ 时约为 $O(t^2)$，累计到长度 $T$ 可达 $O(T^3)$；具体运行时间还受模型宽度、硬件与实现影响。

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

<div align="center">
<img src="/figures/05-interactive-video/latex/04-kv-cache-realtime/kv-cache-append-current-row.png" alt="当前查询复用历史键值缓存，只追加当前键值行并对全部历史位置计算注意力" width="86%">

_图 5.4-3：第 t 步只追加新的 k_t 与 v_t；当前 q_t 仍与缓存后的全部键形成长度 t 的得分，并对全部值精确加权。本文根据上式绘制。_
</div>

应用 softmax 函数后得到注意力权重 $\mathbf{\alpha}_t = \text{softmax}\left(\frac{\mathbf{s}_t}{\sqrt{d_k}}\right)$。最终的注意力输出向量 $\mathbf{o}_t \in \mathbb{R}^{1 \times d_k}$ 为：
$$\mathbf{o}_t = \mathbf{\alpha}_t \mathbf{V}_t = \mathbf{\alpha}_t \begin{bmatrix} \mathbf{V}_{<t} \\ \mathbf{v}_t \end{bmatrix} = \sum_{i=1}^{t} \alpha_{t,i} \mathbf{v}_i$$

通过这两个公式，我们发现 $\mathbf{K}_{<t}$ 和 $\mathbf{V}_{<t}$ 的值完全等同于在第 $t-1$ 步时计算的结果。因此，只要我们在显存中开辟一块空间，将之前计算的 $\mathbf{k}_i$ 和 $\mathbf{v}_i$ 缓存下来，在第 $t$ 步时只需计算新的 $\mathbf{q}_t$、$\mathbf{k}_t$ 和 $\mathbf{v}_t$。然后将 $\mathbf{k}_t$ 和 $\mathbf{v}_t$ 拼接到缓存中，即可完成整个自注意力计算。
这就是 KV Cache 的核心：第 $t$ 个解码步只计算新词元的 Q/K/V，并让一个新查询读取 $t$ 个缓存键值。就注意力随序列长度的主项而言，单步从约 $O(t^2)$ 降为 $O(t)$，累计从约 $O(T^3)$ 降为 $O(T^2)$；缓存同时引入了 $O(T)$ 的每层显存开销。

> 唯一的精炼类比：
> 可以将朴素的自回归解码想象为一个完全没有记忆力的人在朗读一本不断翻页的书，每次为了读出新的一页，他必须从第一页开始大声朗读。而引入 KV Cache 后，这个人拥有了短期记忆（显存），他只需在脑海中回忆之前的故事情节（缓存），然后结合眼前的这一页（当前 Token），就能直接继续讲下去。

## 视频生成中的时空 KV Cache

无论词元来自文本还是视频，KV Cache 最终缓存的都是模型所采用的一维解码顺序。视频的不同之处在于，词元通常来自时间、高度和宽度三个轴，展平后序列可能更长。

<div align="center">
<img src="/figures/05-interactive-video/source/04-kv-cache-realtime/stream-fig1.png" alt="StreamingLLM 对比稠密、窗口和保留 attention sink 的缓存策略，展示固定缓存预算下的长序列生成。" width="86%">

_图 5.4-4：StreamingLLM 对比稠密、窗口和保留 attention sink 的缓存策略，展示固定缓存预算下的长序列生成。 出处：Guangxuan Xiao et al.，[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)（2023），Figure 1。_
</div>

假设我们将视频压缩为潜在表示（Latent representation），每个时空补丁（Patch）被视为一个词元。若帧数为 $F$，每帧的高度和宽度方向上分别有 $h$ 和 $w$ 个补丁，则总的序列长度将达到 $L = F \times h \times w$。
例如，若每个潜在网格位置都是一个词元，16 帧、每帧 $32\times32$ 个网格位置会得到 $L=16{,}384$。实际模型可能进一步分块、并行生成或采用不同顺序，因此该数字只是示例。

此时，KV Cache 会面临严重的**显存墙（Memory Wall）**问题。每个浮点数占用 2 个字节（FP16），一个注意力头的缓存大小不仅与隐藏维度相关，更与序列长度 $L$ 成正比。随着序列长度急剧增加，显存往往被迅速耗尽。

<div align="center">
<img src="/figures/05-interactive-video/source/04-kv-cache-realtime/flash-fig1.png" alt="FlashAttention 用分块把注意力计算留在片上存储，区分了计算复用之外的显存读写瓶颈。" width="86%">

_图 5.4-5：FlashAttention 用分块把注意力计算留在片上存储，区分了计算复用之外的显存读写瓶颈。 出处：Tri Dao et al.，[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)（2022），Figure 1。_
</div>

为减少服务系统中的缓存碎片和预留浪费，vLLM 提出了**分页注意力（PagedAttention）** [[Kwon et al., 2023]](https://arxiv.org/abs/2309.06180)。它还可在共享前缀、并行采样等情形复用物理块。原论文针对语言模型；把同一内存管理思想迁移到自回归视频模型，是工程类比而不是论文已证明的结论。
在操作系统中，内存被划分为固定大小的页（Pages）以减少碎片；分页注意力同样将连续的键值向量划分为独立的块（Blocks）。
设每个块能容纳 $B$ 个词元的键值对，则键缓存 $\mathbf{K}$ 可以表示为一系列非连续的张量块的集合：
$$\mathcal{K} = \{ \mathbf{K}^{(1)}, \mathbf{K}^{(2)}, \dots, \mathbf{K}^{(M)} \}$$

其中 $M = \lceil \frac{L}{B} \rceil$，每个 $\mathbf{K}^{(m)} \in \mathbb{R}^{B \times d_k}$。
块表记录逻辑词元位置与物理显存块之间的映射，序列增长时按需分配新块。它改善的是缓存分配与共享，不会减少注意力必须读取的历史长度，也不能单独保证“实时”视频生成。

## 代码实现与张量分析

下面，我们将从零开始实现一个带有 KV Cache 的因果自注意力层。

实现时要重点检查缓存中键和值的序列长度，以及 prefill 阶段因果掩码相对历史缓存的偏移。

```python
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

        # 因果掩码：单步 decode 时当前查询可读取全部缓存，无需显式掩码。
        # prefill 或分块 decode 时，要把查询位置偏移 past_seq_len。
        if seq_len > 1:
            total_seq_len = k.size(2)
            past_seq_len = total_seq_len - seq_len
            query_pos = past_seq_len + torch.arange(seq_len, device=x.device)
            key_pos = torch.arange(total_seq_len, device=x.device)
            mask = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
            mask = mask.unsqueeze(0).unsqueeze(0)
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

在推理阶段，通常把计算分成两部分：

1. **预填充阶段（Prefill Stage）**：一次性将初始条件或上下文提示（Prompt）完整输入模型，利用大矩阵乘法并行计算出最初始的、完整的历史 KV Cache。
2. **解码阶段（Decoding Stage）**：通常令 `seq_len=1`，只输入最新词元。注意力变为一个查询与全部历史键值的矩阵—向量计算，仍会随缓存长度线性增长。

## 小结

- 朴素解码会在每一步重算完整前缀；KV Cache 保存各层旧词元的 K/V，只处理新增词元。
- 在确定性推理、相同数值精度和位置处理下，缓存与重算前缀在数学上等价；工程实现仍可能因精度和内核产生细小数值差异。
- PagedAttention 管理 KV 缓存的物理内存，不压缩历史内容，也不自动把任意视频模型变成实时系统。
