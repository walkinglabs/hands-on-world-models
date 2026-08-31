# 2.2 序列模型：从循环神经网络到Transformer

在现实世界中，大量的数据不仅具有空间结构，更具备明确的时间先后顺序或序列依赖性。例如，金融市场的股票价格走势、一段完整的语音信号、自然语言中的句子，乃至于强化学习中智能体在环境中的连续状态轨迹。在这些场景中，经典统计机器学习中“独立同分布（Independent and Identically Distributed, i.i.d.）”的基础假设被打破了。当前时刻发生的事情，往往由过去的无穷多个历史瞬间所共同决定。如何从具备长程依赖关系的序列数据中提取有效表征，一直是深度学习历史上的核心难题。

在本节中，我们将首先追溯序列建模的统计学起源，从基础的高中概率论出发，推导出处理序列数据的核心数学框架。接着，我们将回顾循环神经网络（Recurrent Neural Networks, RNN）[[Elman, 1990]](https://doi.org/10.1016/0364-0213(90)90002-E) 的设计哲学，剖析其如何维持“隐状态”以记忆历史信息。然而，RNN 由于其时序展开特性，在训练效率和长程梯度传播上面临挑战 [[Hochreiter & Schmidhuber, 1997]](https://doi.org/10.1162/neco.1997.9.8.1735)。由此，我们将过渡到 Transformer [[Vaswani et al., 2017]](https://arxiv.org/abs/1706.03762)，讨论自注意力机制（Self-Attention）如何通过投影与加权聚合来建模序列。

## 2.2.1 序列数据的统计学视角

### 联合概率与条件概率的链式法则

为了建立对序列数据的数学描述，让我们先回到高中数学中的概率论基础。假设我们有一枚硬币，连续抛掷 $T$ 次，每次的结果记为 $x_t$。如果每次抛掷都是独立的，那么产生特定序列的联合概率仅仅是边缘概率的乘积：$P(x_1, x_2) = P(x_1)P(x_2)$。但在自然语言或股票价格中，第 $t$ 个词或价格 $x_t$ 显然依赖于前 $t-1$ 个词或价格。

对于任意长度为 $T$ 的序列 $(x_1, x_2, \ldots, x_T)$，其同时出现的联合概率 $P(x_1, x_2, \ldots, x_T)$ 可以通过条件概率的链式法则（Chain Rule of Probability）进行极其严格且无损的展开：

$$
P(x_1, x_2, \ldots, x_T) = P(x_1) \cdot P(x_2 \mid x_1) \cdot P(x_3 \mid x_1, x_2) \cdots P(x_T \mid x_1, x_2, \ldots, x_{T-1})
$$

这个公式的形式非常优美：它告诉我们，理解一个序列的整体分布，等价于学习在给定所有历史信息的情况下，预测下一步观测值 $x_t$ 的条件分布 $P(x_t \mid x_1, \ldots, x_{t-1})$。这也是当今所有自回归（Autoregressive）生成模型（包括 GPT 系列）的最基础理论基石。

### 马尔可夫假设与自回归模型

直接建模该公式面临一个致命的问题：随着时间步 $t$ 的增加，条件 $x_1, \ldots, x_{t-1}$ 的长度在不断增长。如果要穷举所有可能的历史组合，所需的计算量和参数量将随时间呈指数级爆炸。在计算机算力与存储极其受限的早期，统计学家们不得不做出妥协。

最经典的妥协方案是引入马尔可夫假设（Markov Assumption）：假设当前时刻的状态仅仅依赖于过去有限的 $\tau$ 个时刻，而与更早的历史无关。在自然语言处理中，这被称为 $N$-gram 模型（其中 $N = \tau + 1$）。如果取 $\tau = 1$（即一阶马尔可夫模型），条件概率将被极大地简化为：

$$
P(x_t \mid x_1, \ldots, x_{t-1}) \approx P(x_t \mid x_{t-1})
$$

尽管马尔可夫假设使得模型变得可计算，但它的缺陷同样明显：它人为地切断了序列的长程依赖（Long-range Dependency）。例如，在句子“他来自法国，精通各种文学和艺术，并且能说一口流利的[填空]”中，要填出“法语”，模型必须回忆起远在句子开头的“法国”。固定的截断窗口 $\tau$ 无法处理这种跨越长距离的逻辑关联。我们需要一种能够动态维持并更新全局历史信息的机制。

## 2.2.2 循环神经网络（RNN）的数学推导

为了打破固定窗口大小的限制，循环神经网络引入了一个极其深刻的数学概念：**隐状态（Hidden State）**。

### 从标量到张量：隐状态的诞生

让我们先用高中物理中的运动学来建立直觉。假设我们要追踪一个正在做复杂曲线运动的粒子。在任意时刻 $t$，粒子的当前位置 $x_t$ 无法单独决定下一时刻的位置 $x_{t+1}$，我们还需要知道它的速度。在这里，“位置和速度的集合”就可以看作是粒子的“状态” $h_t$。只要我们掌握了状态 $h_t$，并且知道当前的受力情况（输入），我们就能根据牛顿运动定律（状态转移方程）推演出下一个状态 $h_{t+1}$。

在深度学习中，我们将这个直觉抽象为数学函数。定义隐状态 $\mathbf{h}_t$ 来存储序列直到时间步 $t$ 的所有历史信息。当前时刻的隐状态 $\mathbf{h}_t$，由上一时刻的隐状态 $\mathbf{h}_{t-1}$ 和当前时刻的外部输入 $\mathbf{x}_t$ 共同决定。我们用一个非线性函数 $f$ 来表示这个状态转移过程：

$$
\mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1})
$$

现在，我们通过严谨的矩阵运算来具体实例化这个非线性函数 $f$。假设在时间步 $t$，小批量输入 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$（其中 $n$ 为批量大小，$d$ 为输入维度）。我们设上一时刻的隐状态为 $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$（其中 $h$ 为隐藏单元的数量）。循环神经网络的核心计算公式如下：

$$
\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh} + \mathbf{b}_h)
$$

其中：

- $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$ 是输入到隐状态的权重矩阵；
- $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ 是隐状态到隐状态（即时间步之间传递记忆）的权重矩阵；
- $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$ 是偏置参数；
- $\phi$ 是非线性激活函数，在传统 RNN 中通常采用 $\tanh$ 函数，以保证隐状态的数值范围被稳定限制在 $[-1, 1]$ 之间。

有了当前的隐状态 $\mathbf{H}_t$，我们就可以通过另一个线性变换来预测输出 $\mathbf{O}_t \in \mathbb{R}^{n \times q}$（例如下一个词的概率分布，其中 $q$ 是输出的词表大小）：

$$
\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q
$$

这里 $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$ 和 $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ 分别是隐状态到输出的权重矩阵和偏置参数。需要特别强调的是，RNN 的一个核心特性是**参数共享（Parameter Sharing）**：对于任意时间步 $t$，权重矩阵 $\mathbf{W}_{xh}, \mathbf{W}_{hh}, \mathbf{W}_{hq}$ 都是完全相同的。这种设计不仅极大地减少了模型参数量，还赋予了模型处理任意长度序列的能力。

### 沿时间反向传播（BPTT）与梯度消失

尽管 RNN 的前向传播公式看起来简洁优雅，但其在优化时却面临严重的数学困难。在 RNN 中，我们通常使用“沿时间反向传播”（Backpropagation Through Time, BPTT）来计算梯度。本质上，BPTT 就是将该公式在时间轴上展开，然后应用微积分中的链式法则。

假设我们要计算最终输出关于初始隐状态 $\mathbf{h}_0$ 的梯度，链式法则会产生一长串偏导数的连乘：

$$
\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_0} = \prod_{t=1}^T \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}
$$

由于 $\mathbf{h}_t = \phi(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \dots)$，每一项偏导数 $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}$ 都包含权重矩阵 $\mathbf{W}_{hh}$ 和激活函数的导数。这意味着，上述连乘相当于 $\mathbf{W}_{hh}$ 连续相乘了 $T$ 次。从线性代数的特征值分解角度来看，如果 $\mathbf{W}_{hh}$ 的最大特征值绝对值小于 $1$，经过 $T$ 次方后，梯度将呈指数级衰减至零，这被称为**梯度消失（Vanishing Gradient）**；反之，若大于 $1$，则会导致**梯度爆炸（Exploding Gradient）**。梯度消失使得 RNN 难以在训练中真正捕捉到相隔几十上百个时间步的长期依赖。

(**为了深刻理解RNN的机制，我们尝试在PyTorch中从零开始实现单步RNN。**)

```python
import torch
from torch import nn
from torch.nn import functional as F

class RNNStep(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # 严谨对应公式 eqref:eq_rnn_step 的参数定义
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, X, H_prev):
        # X: (batch_size, input_size)
        # H_prev: (batch_size, hidden_size)
        # 矩阵乘法并相加：计算当前步的预激活值
        pre_activation = torch.matmul(X, self.W_xh) + torch.matmul(H_prev, self.W_hh) + self.b_h
        # 使用 tanh 作为非线性激活函数，保持数值稳定
        H_curr = torch.tanh(pre_activation)
        return H_curr

# 测试一个简单的小批量数据
batch_size, input_size, hidden_size = 32, 128, 256
rnn_step = RNNStep(input_size, hidden_size)
X_t = torch.randn(batch_size, input_size)
H_prev = torch.zeros(batch_size, hidden_size)

# 执行单步前向传播
H_t = rnn_step(X_t, H_prev)
print(f"H_t shape: {H_t.shape}") # 预期输出: torch.Size([32, 256])
```

## 2.2.3 注意力机制与Transformer架构

为缓解传统 RNN 的长程依赖与梯度传播问题，研究者提出了长短期记忆网络（LSTM, [[Hochreiter & Schmidhuber, 1997]](https://doi.org/10.1162/neco.1997.9.8.1735)）和门控循环单元（GRU, [[Cho et al., 2014]](https://arxiv.org/abs/1406.1078)）。它们用门控机制控制信息流。不过，在标准 RNN 中，当前状态 $h_t$ 仍依赖上一步状态 $h_{t-1}$，因此训练时难以在时间维度上完全并行。

2017 年，Vaswani 等人发表了 _Attention Is All You Need_ [[Vaswani et al., 2017]](https://arxiv.org/abs/1706.03762)，提出不使用循环结构、主要依赖注意力与前馈网络的 Transformer。训练时，给定完整输入序列后，各位置的注意力计算可以并行；自回归解码时仍需按词元逐步生成，不能概括为消除了所有时序依赖。

### 自注意力（Self-Attention）的几何直觉与严密推导

在介绍复杂的矩阵运算之前，我们先剥离表象，回到高中向量几何，寻找注意力机制最核心的直觉来源。

对于两个向量 $\mathbf{a}$ 和 $\mathbf{b}$，它们的内积（Dot Product）定义为 $\mathbf{a}^\top \mathbf{b} = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$。当两个向量长度固定时，夹角 $\theta$ 越小（方向越趋于一致），内积越大。因此，**内积可以作为衡量两个高维向量之间相似度（Similarity）或相关性的严谨数学度量**。

在自注意力机制中，我们将序列中的每一个元素（如词元）投影到三个不同的向量空间，分别赋予它们三种不同的身份角色：

1. **查询向量（Query, $\mathbf{q}$）**：代表该元素正在寻找什么样的信息。
2. **键向量（Key, $\mathbf{k}$）**：代表该元素包含了什么样的信息特征。
3. **值向量（Value, $\mathbf{v}$）**：代表该元素实际提供的内容实质。

> 只有在此刻面对全篇最抽象的交互机制时，我们才允许使用一次极其克制的类比来辅助理解：在一个由序列元素构成的学术会议上，你（作为一个查询向量 $\mathbf{q}_i$）希望寻找特定领域的专家来补充你的论文背景。你身上带着名牌（你的键向量 $\mathbf{k}_i$）向全场展示你的研究特征，同时你也在检视会场中其他所有人的名牌（他人的键向量 $\mathbf{k}_j$）。当你发现某人的 $\mathbf{k}_j$ 与你的查询意图 $\mathbf{q}_i$ 在高维空间中内积很大（高度相似）时，你会分配给此人一个极高的注意力权重 $a_{ij}$，并大量吸收他提供的内容解答（值向量 $\mathbf{v}_j$）。自注意力机制，就是让序列中的每一个元素同时扮演寻问者和解答者的角色，通过两两之间的全局匹配，重塑各自的表征。

让我们严格写出这个过程的数学公式。对于序列中第 $i$ 个元素的查询 $\mathbf{q}_i$ 和第 $j$ 个元素的键 $\mathbf{k}_j$，它们的原始注意力打分（Attention Score）为：

$$
s_{i,j} = \mathbf{q}_i^\top \mathbf{k}_j
$$

为了将这些原始打分转化为概率分布（权重之和为1），我们对其施加 Softmax 函数。同时，当向量维度 $d_k$ 很大时，内积的值容易变得极大，导致 Softmax 函数进入梯度极小（饱和）的区域。因此，我们需要除以 $\sqrt{d_k}$ 进行缩放平滑。最终，元素 $i$ 注意到元素 $j$ 的概率权重为：

$$
a_{i,j} = \frac{\exp(s_{i,j} / \sqrt{d_k})}{\sum_{m=1}^T \exp(s_{i,m} / \sqrt{d_k})}
$$

最后，元素 $i$ 的新表征 $\mathbf{z}_i$ 是全场所有值向量 $\mathbf{v}_j$ 的概率加权求和：

$$
\mathbf{z}_i = \sum_{j=1}^T a_{i,j} \mathbf{v}_j
$$

**从向量到矩阵：高度并行的威力**

上述逐个元素的标量求和公式虽然直观，但在实践中却极其缓慢。Transformer 最大的创举在于，它将整个序列长度为 $T$ 的所有查询拼成一个矩阵 $\mathbf{Q} \in \mathbb{R}^{T \times d_k}$，所有的键拼成 $\mathbf{K} \in \mathbb{R}^{T \times d_k}$，所有的值拼成 $\mathbf{V} \in \mathbb{R}^{T \times d_v}$。这样，时间步之间的两两互动，被一次性压缩为了纯粹的矩阵乘法，完全消除了时序上的循环依赖：

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

由于矩阵乘法在现代 GPU 上被优化到了极致，这一公式使得 Transformer 能够在处理长序列时展现出比 RNN 惊人得多的计算效率。

### 多头注意力（Multi-Head Attention）与位置编码

单一的自注意力机制往往只能捕捉序列中某一种维度的关联（例如仅仅关注语法结构或者仅仅关注情感倾向）。为了让模型拥有从多个独立子空间提取特征的能力，Transformer 引入了多头注意力（Multi-Head Attention）。它将原始的 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 通过不同的权重矩阵投影 $h$ 次，分别执行 $h$ 次独立的注意力计算，最后将结果拼接（Concatenate）并通过线性映射合并。

然而，细心的读者可能已经发现了一个严峻的数学漏洞：在该公式的矩阵乘法中，并不包含任何序列元素的相对或绝对位置信息。如果你打乱输入序列的顺序，输出的表征只是一起打乱，而数值完全不变。自注意力机制天生是一个“词袋（Bag-of-Words）”操作。

为了弥补序列顺序信息的缺失，Transformer 采用了正弦和余弦函数的**位置编码（Positional Encoding）**，强制注入绝对位置信号。对于位置 $pos$ 和维度 $2i$ 或 $2i+1$，位置编码定义为：

$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin(pos / 10000^{2i/d_{\text{model}}}) \\
PE_{(pos, 2i+1)} &= \cos(pos / 10000^{2i/d_{\text{model}}})
\end{aligned}
$$

这种极其精妙的三角函数设计，不仅通过不同频率的周期变化唯一确定了绝对位置，还利用三角函数的和差化积公式，在理论上使得相对位置的计算可以通过线性变换实现。

(**代码实现：利用PyTorch构建缩放点积注意力**)

```python
import math
import torch
from torch import nn

def masked_softmax(X, valid_lens):
    """通过在掩码位置填充极小值来执行掩蔽 softmax 操作，常用于处理变长序列"""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 展平以便于掩蔽
        X = X.reshape(-1, shape[-1])
        mask = torch.arange((shape[-1]), dtype=torch.float32,
                            device=X.device)[None, :] < valid_lens[:, None]
        # 将无效位置填充为非常小的值（接近负无穷），使得 softmax 后的概率趋近于0
        X[~mask] = -1e6
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    """严谨实现的缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        # queries 的形状：(batch_size, num_queries, d)
        # keys 的形状：(batch_size, num_kv_pairs, d)
        # values 的形状：(batch_size, num_kv_pairs, value_dimension)
        d = queries.shape[-1]

        # 执行矩阵乘法 QK^T，并除以 sqrt(d) 进行稳定缩放
        # transpose(1, 2) 实现了矩阵转置，形状变为 (batch_size, num_queries, num_kv_pairs)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)

        # 应用 softmax 获取概率分布权重
        self.attention_weights = masked_softmax(scores, valid_lens)

        # 将概率权重与 values 矩阵相乘
        return torch.bmm(self.dropout(self.attention_weights), values)

# 创建小批量测试张量
queries = torch.normal(0, 1, (2, 1, 64))
keys = torch.normal(0, 1, (2, 10, 64))
values = torch.normal(0, 1, (2, 10, 128))
valid_lens = torch.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
attention.eval() # 评估模式，关闭 dropout
context = attention(queries, keys, values, valid_lens)
print(f"注意力输出形状: {context.shape}") # 预期输出: torch.Size([2, 1, 128])
```

## 2.2.4 小结

本节我们完成了一次从经典统计到现代深度学习的跨越。我们首先通过高中概率论的条件概率链式法则，建立了序列预测的基础理论框架，并指出了马尔可夫假设在建模长程依赖时的局限性。为了克服这些局限，循环神经网络（RNN）被提出，利用隐状态 $\mathbf{h}_t$ 持续累积历史信息。然而，RNN的自回归展开导致的反向传播梯度消失和无法并行化计算，成为了限制其规模扩张的核心瓶颈。

Transformer 架构以一种极其激进的视角重构了序列建模范式。通过完全抛弃时序上的循序渐进，采用全局感受野的“缩放点积自注意力机制（Scaled Dot-Product Self-Attention）”，Transformer 使得任意两个相隔甚远的序列元素能在单次操作中直接交互。结合严谨的矩阵运算和精妙的正弦余弦位置编码，它不仅彻底解决了长程依赖问题，更释放了 GPU 无与伦比的并行计算潜能，最终奠定了现代大语言模型（LLMs）的底层基石。

## 2.2.5 练习

1. 回顾该公式，假设 $\mathbf{W}_{hh}$ 是一个对角矩阵，且对角线元素全部为 $0.5$。经过 $100$ 个时间步的沿时间反向传播，最初一步的梯度将衰减到原始大小的多少？这说明了 RNN 训练的什么问题？
   - **提示**：计算 $0.5^{100}$，并结合深度学习中数值下溢的概念进行思考。
2. 在 Transformer 的缩放点积注意力该公式中，为什么我们必须除以 $\sqrt{d_k}$？
   - **提示**：假设 $\mathbf{q}$ 和 $\mathbf{k}$ 的元素都是均值为 $0$、方差为 $1$ 的独立随机变量。利用高中统计学中独立变量乘积与求和的期望与方差公式，推导 $\mathbf{q}^\top \mathbf{k}$ 的方差变化，思考如果不除以 $\sqrt{d_k}$，随着维度增加，Softmax 函数的输入分布会发生怎样的严重偏移。
3. 位置编码该公式采用了三角函数。请尝试用高中数学的三角函数和差公式推导：对于任意固定的偏移量 $k$，$PE_{(pos+k)}$ 能否表示为 $PE_{(pos)}$ 的线性函数？
   - **提示**：展开 $\sin(\omega(pos+k))$ 和 $\cos(\omega(pos+k))$，寻找它们与 $\sin(\omega pos)$ 和 $\cos(\omega pos)$ 的线性关系。
