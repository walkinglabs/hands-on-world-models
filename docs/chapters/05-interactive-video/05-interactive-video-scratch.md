# 交互式视频生成模块的从零开始实现

在前面的章节中，我们已经探讨了静态图像的潜在空间表征以及无条件视频生成的基础架构。然而，真正的世界模型（World Models）必须具备对环境做出响应的能力。在本节中，我们将深入探讨并从零开始实现一个**交互式视频生成模块**（Interactive Video Generation Module）。该模块的核心任务是：给定过去的视频帧序列和一系列控制动作（如按键、摇杆输入或连续控制指令），模型需要预测并生成符合物理规律及动作逻辑的未来视频帧。

## 历史背景与学术脉络

动作条件视频预测与模型式规划密切相关。Oh 等人在 Atari 上展示了根据动作预测未来画面的方法 [[Oh et al., 2015]](https://arxiv.org/abs/1507.08750)；这些画面仍是高维像素，只是场景和动作空间相对受限。Finn 等人把动作条件预测用于真实机器人交互视频 [[Finn et al., 2016]](https://arxiv.org/abs/1605.07157)，Babaeizadeh 等人则显式引入随机潜变量来表达同一过去对应多种未来的情形 [[Babaeizadeh et al., 2017]](https://arxiv.org/abs/1710.11252)。

近年来，潜在动力学出现了不同实现路线。Dreamer 在连续或离散的潜在状态中学习动作条件动力学，用于想象训练 [[Hafner et al., 2020]](https://arxiv.org/abs/1912.01603)；它并不是基于离散视频词元的 Transformer。Genie 则从无动作标签的视频中学习潜在动作，并在离散视频词元上生成可交互轨迹 [[Bruce et al., 2024]](https://arxiv.org/abs/2402.15391)。因此，本节采用“视觉词元 + 因果注意力”时，主要借鉴的是后一类自回归交互视频模型。

## 交互式生成的数学构型

为了确保理论的严谨性，我们从最基础的条件概率模型起步。假设我们需要描述一个简单的物理现象，例如一个小球的位置 $x$ 随时间 $t$ 的变化。在高中物理中，如果已知初速度和加速度，我们可以用确定性方程 $x(t) = x_0 + v_0 t + \frac{1}{2} a t^2$ 来计算小球的轨迹。

然而，在交互式视频生成中，系统是极其复杂的，并且受到外部输入的影响。我们不再具有完美的物理方程，而是只能通过观测数据来推断状态的转移规律。定义 $x_t$ 为第 $t$ 个时间步的视频帧观测，$a_t$ 为在该时间步施加的动作指令。我们的目标是建立一个概率模型，估计在给定历史观测序列 $x_{1:t}$ 和历史动作序列 $a_{1:t}$ 的条件下，下一个时间步观测 $x_{t+1}$ 的条件概率分布：

$$P(x_{t+1} \mid x_1, x_2, \ldots, x_t, a_1, a_2, \ldots, a_t)$$

根据概率论中的链式法则，整个长度为 $T$ 的交互式视频序列的联合概率分布可以严格展开为条件概率的连乘积：

$$P(x_1, \ldots, x_T \mid a_1, \ldots, a_{T-1}) = \prod_{t=1}^{T} P(x_t \mid x_{<t}, a_{<t})$$

在此公式中，每一次帧的生成都严格依赖于**严格发生在此之前**的所有帧和动作。这种时间上的不对称性，要求我们在模型架构中必须引入因果掩码（Causal Masking），以阻断任何从未来向过去的信息流动。

## 动作条件与时空序列的融合策略

在现代深度学习中，高分辨率的视频帧 $x_t$ 通常不会直接在像素级被处理。我们在先前的章节中介绍过空间自编码器（如 VQ-VAE），它可以将每一帧 $x_t$ 压缩为一组离散的潜在标记（Latent Tokens）。设每帧可以被编码为 $S$ 个标记的集合 $Z_t = \{z_{t,1}, z_{t,2}, \ldots, z_{t,S}\}$。

此时，视频序列不再是一个一维的帧序列，而是一个嵌套的时空矩阵。为了将其输入到自回归序列模型中，我们必须将其展平（Flatten）为一维序列。一种严谨且有效的方式是按照时间优先、空间次之的顺序进行光栅化扫描（Raster Scan）。

更关键的是动作 $a_t$ 的注入。动作本质上是连接时间步 $t$ 和时间步 $t+1$ 的桥梁。因此，在序列排布上，我们将动作标记（Action Token）显式地插入在相邻两帧的视觉标记之间。设展平后的序列为 $\mathcal{U}$，其结构定义为：

$$\mathcal{U} = [Z_1, a_1, Z_2, a_2, \ldots, Z_{T-1}, a_{T-1}, Z_T]$$

通过这种交错排列（Interleaving），我们可以强制自回归模型在预测帧 $Z_{t+1}$ 的首个空间标记时，必须不仅关注历史帧，还要读取到紧邻的动作指令 $a_t$。

## 自回归核心：带掩码的因果注意力机制

我们使用 Transformer 作为序列建模的核心骨干。在处理序列 $\mathcal{U}$ 时，为了计算第 $i$ 个元素的隐含表示，模型通过注意力机制计算该元素与序列中其他元素的关联度。

设输入序列的嵌入矩阵为 $\mathbf{H} \in \mathbb{R}^{N \times D}$，其中 $N$ 是序列总长度，$D$ 是隐含层维度。我们通过线性映射得到查询矩阵 $\mathbf{Q}$、键矩阵 $\mathbf{K}$ 和值矩阵 $\mathbf{V}$：

$$\mathbf{Q} = \mathbf{H} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{H} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{H} \mathbf{W}_V$$

标准自注意力机制计算 $\mathbf{Q}$ 与 $\mathbf{K}$ 的点积来衡量相似度。然而，对于预测任务，我们必须施加严格的因果性：第 $i$ 个位置只能观察到位置 $j \le i$ 的信息。为此，我们引入一个下三角掩码矩阵 $\mathbf{M} \in \mathbb{R}^{N \times N}$，其定义如下：

$$
\mathbf{M}_{i,j} = \begin{cases}
0 & \text{if } j \le i \\
-\infty & \text{if } j > i
\end{cases}
$$

带掩码的缩放点积注意力（Masked Scaled Dot-Product Attention）的严格数学形式为：

$$\mathrm{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{M}) = \mathrm{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V}$$

当 $j > i$ 时，矩阵相加使得相应的对数几率趋近于 $-\infty$，在经过 $\mathrm{softmax}$ 归一化后，其注意力权重将严格等于零，从而在物理学意义上隔绝了未来的“信息泄露”。

::: info 类比思考：时间的不可逆性
我们可以将这种因果掩码视为热力学第二定律在信息流中的具象化体现。时间箭头不可逆，模型在计算当前状态的演化时，其所处的“光锥”（Light Cone）内部绝对不包含来自未来的任何微小光子或信息。这正是 $\mathbf{M}$ 矩阵中 $-\infty$ 所施加的刚性物理边界。
:::

## 代码实现：交互式视频生成器的构建

现在，我们通过代码严格实现上述数学和几何构造。首先，我们将构建因果注意力模块和 Transformer 块。考虑到篇幅和细节，代码将包含详细的类型注释和维度推导。

(**我们将首先实现多头因果自注意力层，注意掩码矩阵的应用。**)

```python
import torch
from torch import nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    """带因果掩码的多头自注意力机制"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "嵌入维度必须能被注意力头数整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 将 Q, K, V 的线性映射合并为一个权重矩阵以提升计算效率
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x 形状: (批量大小 B, 序列长度 N, 嵌入维度 D)
        B, N, D = x.size()

        # 线性映射并分离为 Q, K, V
        # 形状变化: (B, N, 3D) -> (B, N, 3, num_heads, head_dim) -> 转置为 (3, B, num_heads, N, head_dim)
        qkv = self.c_attn(x).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # 每一个的形状: (B, num_heads, N, head_dim)

        # 计算缩放点积注意力，并动态生成因果掩码 (利用 PyTorch 的 is_causal 标志)
        # 这里等价于数学公式中的加上下三角矩阵 M
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True
        ) # y 形状: (B, num_heads, N, head_dim)

        # 将多个头拼接回去
        y = y.transpose(1, 2).contiguous().view(B, N, D)

        # 输出线性投影与残差丢弃
        return self.resid_dropout(self.c_proj(y))
```

接下来，我们基于上述注意力机制构建标准的 Transformer 块。

(**在这个块中，我们交替使用层归一化（Layer Normalization）和残差连接（Residual Connections），以保障深层网络梯度反向传播的稳定性。**)

```python
class TransformerBlock(nn.Module):
    """标准的自回归 Transformer 块"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 遵循 Pre-LayerNorm 架构设计
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

### 序列拼接与完整生成器

为了实现该公式中的序列交错排布，我们需要精心设计位置编码。视频具有内在的时空二维结构，而一维的绝对位置编码往往会破坏这一结构。因此，我们将为每一个空间标记分配一个由“时间步索引”和“空间位置索引”联合决定的复合嵌入。动作标记同样需要融入序列中。

(**下面的类封装了标记映射、位置编码、交错拼接以及最终的多层 Transformer 前向传播过程。**)

```python
class InteractiveVideoGenerator(nn.Module):
    """交互式视频生成器的核心模块"""
    def __init__(self, vocab_size, action_dim, embed_dim, num_layers, num_heads,
                 tokens_per_frame, max_frames, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.tokens_per_frame = tokens_per_frame

        # 视觉标记的词表嵌入层
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)

        # 动作输入的线性映射层（假设动作为连续向量）
        self.action_proj = nn.Linear(action_dim, embed_dim)

        # 空间位置编码 (0 到 tokens_per_frame - 1)
        self.spatial_pos_emb = nn.Parameter(torch.zeros(1, tokens_per_frame, embed_dim))
        # 时间位置编码 (0 到 max_frames - 1)
        self.temporal_pos_emb = nn.Parameter(torch.zeros(1, max_frames, embed_dim))
        # 为动作分配一种特殊的位置/类型标识向量
        self.action_type_emb = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        # 堆叠 Transformer 块
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        # 最终预测视觉标记词表的线性头
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.spatial_pos_emb, std=0.02)
        nn.init.normal_(self.temporal_pos_emb, std=0.02)
        nn.init.normal_(self.action_type_emb, std=0.02)

    def forward(self, visual_tokens, actions):
        """
        visual_tokens: 形状 (B, T, S) 包含各帧的离散标记索引
        actions: 形状 (B, T-1, action_dim) 包含帧间的动作输入
        """
        B, T, S = visual_tokens.size()

        # 1. 提取视觉嵌入并加入时空位置编码
        # token_embeddings 形状: (B, T, S, D)
        token_embeddings = self.tok_emb(visual_tokens)

        # 添加空间与时间位置编码，利用广播机制
        # (1, T, 1, D) + (1, 1, S, D) -> (1, T, S, D)
        spatial_emb = self.spatial_pos_emb.unsqueeze(1).expand(-1, T, -1, -1)
        temporal_emb = self.temporal_pos_emb[:, :T, :].unsqueeze(2).expand(-1, -1, S, -1)

        visual_repr = token_embeddings + spatial_emb + temporal_emb

        # 2. 处理动作嵌入
        # action_repr 形状: (B, T-1, 1, D)
        if T > 1:
            action_embeddings = self.action_proj(actions)
            action_temporal_emb = self.temporal_pos_emb[:, :T-1, :]
            # 动作用自身的时间编码加上动作专属类别标识
            action_repr = action_embeddings + action_temporal_emb + self.action_type_emb
            action_repr = action_repr.unsqueeze(2) # 变为 (B, T-1, 1, D)

        # 3. 交错重组序列
        # 我们需要将序列排布为: [Z_1, a_1, Z_2, a_2, ..., Z_T]
        sequence = []
        for t in range(T):
            sequence.append(visual_repr[:, t, :, :]) # (B, S, D)
            if t < T - 1:
                sequence.append(action_repr[:, t, :, :]) # (B, 1, D)

        # 沿序列维度拼接
        # 总长度 N = T * S + (T - 1)
        h = torch.cat(sequence, dim=1)
        h = self.dropout(h)

        # 4. 通过自回归 Transformer
        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        logits = self.lm_head(h) # (B, N, vocab_size)

        return logits
```

## 损失函数与模型训练

在自回归建模框架下，我们的训练目标是最大化序列对数似然度（Log-Likelihood）。由于动作在我们的设定中是作为条件给定的，我们不需要去预测动作，而只关注视觉标记。

在给定逻辑回归输出（Logits）的情况下，对于预测序列中的第 $k$ 个视觉标记（在展平序列中的真实值设为 $y_k$），我们采用标准的交叉熵损失函数（Cross-Entropy Loss）：

$$\mathcal{L} = -\frac{1}{N_{vis}} \sum_{k=1}^{N_{vis}} \log \frac{\exp(\mathbf{logits}_{k, y_k})}{\sum_{v=1}^{V} \exp(\mathbf{logits}_{k, v})}$$

其中 $V$ 是词表大小（`vocab_size`），$N_{vis}$ 是序列中所有视觉标记的总数。在实现损失函数时，我们需要小心地处理索引对齐问题。序列中位置 $i$ 的隐含层输出用于预测位置 $i+1$ 的标记。同时，我们要利用掩码（Mask）将预测动作标记处的损失过滤掉，仅计算视觉标记上的梯度。

(**以下演示如何从展平的 logits 序列中提取对应的视觉标记预测，并与目标标签计算交叉熵损失。**)

```python
def calculate_loss(logits, visual_tokens, tokens_per_frame):
    """
    计算序列的交叉熵损失，剔除动作标记对应的输出
    logits: (B, N, vocab_size) 包含序列的所有输出
    visual_tokens: (B, T, S) 目标的真实视觉标记
    """
    B, T, S = visual_tokens.size()

    # 构造目标序列，将 visual_tokens 展平，并将预测目标向左偏移一位 (Shift left)
    # 对于位置 i，预测目标是位置 i+1 的输入

    # 我们先在序列维度找到对应的目标索引
    target_indices = []
    # 第一个帧不需要预测（或者说它通常用作初始条件），损失计算从帧内的转移及后续帧开始
    # 为简单起见，我们对所有后续有效的视觉标记计算预测损失

    logits_for_loss = []
    targets_for_loss = []

    current_idx = 0
    for t in range(T):
        if t < T - 1:
            # 帧 t 内的每一个 token 预测下一个 token
            # 最后一个 token 预测动作 (我们不计算动作预测损失)
            logits_for_loss.append(logits[:, current_idx : current_idx + S - 1, :])
            targets_for_loss.append(visual_tokens[:, t, 1:S])

            # 动作 token 处的预测，目标是帧 t+1 的第一个视觉 token
            current_idx += S # 跳过帧视觉特征，到达动作 token
            logits_for_loss.append(logits[:, current_idx : current_idx + 1, :])
            targets_for_loss.append(visual_tokens[:, t+1, 0:1])

            current_idx += 1 # 跳过动作 token
        else:
            # 最后一帧内的预测
            logits_for_loss.append(logits[:, current_idx : current_idx + S - 1, :])
            targets_for_loss.append(visual_tokens[:, t, 1:S])

    # 拼接所有的预测和目标
    logits_flat = torch.cat(logits_for_loss, dim=1).reshape(-1, logits.size(-1))
    targets_flat = torch.cat(targets_for_loss, dim=1).reshape(-1)

    # 标准交叉熵损失
    loss = F.cross_entropy(logits_flat, targets_flat)
    return loss
```

通过如上设计，我们完整实现了从多模态输入（动作指令与离散视频流）、序列重组、带掩码前向计算到损失反向传播的端到端架构。这就构成了一个基础却具备完全表征能力的交互式视频世界模型的核心引擎。

## 小结

- **交互式视频生成**可以被严谨地映射为一个给定历史帧和外部动作序列的条件联合概率预测问题。
- 在 Transformer 架构下，通过将离散的视觉标记和连续或离散的动作标记交错排列，模型能够自发地学习**时间动态和动作干预结果**。
- **带掩码的因果自注意力机制（Causal Self-Attention）**是确保预测严谨性、阻止未来信息渗透进入当前推断的绝对数学防线。
