# 5.5 从零实现动作条件视频生成器

前面几节讨论的视频模型大多只回答“接下来可能出现什么”。交互式模型还要回答一个更具体的问题：如果智能体此刻向左、加速或抓取，画面会怎样变化？本节从一个教学版模块出发，把离散视频词元和动作交错排成序列，再用因果 Transformer 预测后续词元。它展示的是核心数据流，并不复刻某个大型系统的全部训练配方。

<div align="center">
<img src="/figures/05-interactive-video/source/05-interactive-video-scratch/genie-fig1.png" alt="Genie 把照片、草图和生成图变成可逐步操控的平台世界，展示动作改变后续画面的真实交互目标。" width="86%">

_图 5.5-1：Genie 把照片、草图和生成图变成可逐步操控的平台世界，展示动作改变后续画面的真实交互目标。 出处：Jake Bruce et al.，[Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391)（2024），Figure 1。_
</div>

## 历史背景与学术脉络

动作条件视频预测与模型式规划密切相关。Oh 等人在 Atari 上展示了根据动作预测未来画面的方法 [[Oh et al., 2015]](https://arxiv.org/abs/1507.08750)；这些画面仍是高维像素，只是场景和动作空间相对受限。Finn 等人把动作条件预测用于真实机器人交互视频 [[Finn et al., 2016]](https://arxiv.org/abs/1605.07157)，Babaeizadeh 等人则显式引入随机潜变量来表达同一过去对应多种未来的情形 [[Babaeizadeh et al., 2017]](https://arxiv.org/abs/1710.11252)。

<div align="center">
<img src="/figures/05-interactive-video/source/05-interactive-video-scratch/gamengen-fig1.png" alt="GameNGen 在 20 FPS 下响应玩家动作生成 DOOM 画面，说明动作条件视频世界模型能够进入实时闭环。" width="86%">

_图 5.5-2：GameNGen 在 20 FPS 下响应玩家动作生成 DOOM 画面，说明动作条件视频世界模型能够进入实时闭环。 出处：Dani Valevski et al.，[Diffusion Models Are Real-Time Game Engines](https://arxiv.org/abs/2408.14837)（2024），Figure 1。_
</div>

近年来，潜在动力学出现了不同实现路线。Dreamer 在连续或离散的潜在状态中学习动作条件动力学，用于想象训练 [[Hafner et al., 2020]](https://arxiv.org/abs/1912.01603)；它并不是基于离散视频词元的 Transformer。Genie 则从无动作标签的视频中学习潜在动作，并在离散视频词元上生成可交互轨迹 [[Bruce et al., 2024]](https://arxiv.org/abs/2402.15391)。因此，本节采用“视觉词元 + 因果注意力”时，主要借鉴的是后一类自回归交互视频模型。

<div align="center">
<img src="/figures/05-interactive-video/source/05-interactive-video-scratch/iris-fig1.png" alt="IRIS 将离散帧词元与动作交错输入 Transformer，并在想象轨迹中驱动策略。" width="86%">

_图 5.5-3：IRIS 将离散帧词元与动作交错输入 Transformer，并在想象轨迹中驱动策略。 出处：Vincent Micheli et al.，[Transformers are Sample-Efficient World Models](https://arxiv.org/abs/2209.00588)（2023），Figure 1。_
</div>

## 交互式生成的数学构型

先从条件概率模型看这件事。对于一个做匀加速运动的小球，如果初始位置、速度和加速度都已知，可以用 $x(t) = x_0 + v_0 t + \frac{1}{2} a t^2$ 直接算出轨迹。

真实视频中的状态无法由一条已知方程完整描述，还会受到动作和未观测因素影响。记 $x_t$ 为第 $t$ 个时间步的画面，$a_t$ 为画面之后执行的动作。模型要估计的是：给定截至当前的观测和动作，下一帧可能是什么。

<div align="center">
<img src="/figures/05-interactive-video/source/05-interactive-video-scratch/diamond-fig1.png" alt="DIAMOND 以动作条件扩散世界模型逐帧展开策略轨迹，提供非自回归词元路线的直接对照。" width="86%">

_图 5.5-4：DIAMOND 以动作条件扩散世界模型逐帧展开策略轨迹，提供非自回归词元路线的直接对照。 出处：Daniel Alonso et al.，[Diffusion for World Modeling: Visual Details Matter in Atari](https://arxiv.org/abs/2405.12399)（2024），Figure 1。_
</div>

$$P(x_{t+1} \mid x_1, x_2, \ldots, x_t, a_1, a_2, \ldots, a_t)$$

根据概率的链式法则，长度为 $T$ 的条件序列可以写成：

$$P(x_1, \ldots, x_T \mid a_1, \ldots, a_{T-1}) = \prod_{t=1}^{T} P(x_t \mid x_{<t}, a_{<t})$$

这里采用的是自回归分解：生成 $x_t$ 时只能使用更早的帧和动作。训练 Transformer 时，因果掩码负责挡住序列右侧尚未生成的信息。

## 动作条件与时空序列的融合策略

在现代深度学习中，高分辨率的视频帧 $x_t$ 通常不会直接在像素级被处理。我们在先前的章节中介绍过空间自编码器（如 VQ-VAE），它可以将每一帧 $x_t$ 压缩为一组离散的潜在标记（Latent Tokens）。设每帧可以被编码为 $S$ 个标记的集合 $Z_t = \{z_{t,1}, z_{t,2}, \ldots, z_{t,S}\}$。

编码后的视频具有“时间 × 空间位置”的二维索引。输入普通自回归 Transformer 前，需要约定一种一维顺序。下面采用时间优先、帧内按空间位置排列的光栅顺序；这是一种简单选择，并非唯一选择。

动作 $a_t$ 描述从时刻 $t$ 到 $t+1$ 的干预。为了让它出现在下一帧之前，可以把动作标记插在相邻两帧的视觉词元之间：

$$\mathcal{U} = [Z_1, a_1, Z_2, a_2, \ldots, Z_{T-1}, a_{T-1}, Z_T]$$

<div align="center">
<img src="/figures/05-interactive-video/latex/05-interactive-video-scratch/interleaved-action-causal-visibility.png" alt="动作词元置于相邻视频块之间，下三角掩码让下一视频块可见该动作并屏蔽未来动作" width="86%">

_图 5.5-5：把动作词元插在当前帧词元与下一帧词元之间后，因果掩码允许下一帧读取该动作，同时把尚未发生的动作严格置于不可见区。_
</div>

这样，预测 $Z_{t+1}$ 的第一个词元时，模型可见历史画面和刚刚执行的 $a_t$。模型是否真正学会动作后果仍取决于数据覆盖、训练目标和模型容量，序列排布本身并不提供保证。

## 自回归核心：带掩码的因果注意力机制

我们使用 Transformer 作为序列建模的核心骨干。在处理序列 $\mathcal{U}$ 时，为了计算第 $i$ 个元素的隐含表示，模型通过注意力机制计算该元素与序列中其他元素的关联度。

设输入序列的嵌入矩阵为 $\mathbf{H} \in \mathbb{R}^{N \times D}$，其中 $N$ 是序列总长度，$D$ 是隐含层维度。我们通过线性映射得到查询矩阵 $\mathbf{Q}$、键矩阵 $\mathbf{K}$ 和值矩阵 $\mathbf{V}$：

$$\mathbf{Q} = \mathbf{H} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{H} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{H} \mathbf{W}_V$$

标准自注意力用 $\mathbf{Q}$ 与 $\mathbf{K}$ 的点积计算权重。自回归训练还要限制第 $i$ 个位置只能读取 $j \le i$ 的位置，因此加入下三角掩码 $\mathbf{M} \in \mathbb{R}^{N \times N}$：

$$
\mathbf{M}_{i,j} = \begin{cases}
0 & \text{if } j \le i \\
-\infty & \text{if } j > i
\end{cases}
$$

带掩码的缩放点积注意力写作：

$$\mathrm{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{M}) = \mathrm{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V}$$

当 $j > i$ 时，对应分数被设为 $-\infty$；经过 $\mathrm{softmax}$ 后，其权重为零。因此当前位置无法从未来位置读取信息。

::: info 因果掩码约束的是什么？
它约束的是训练计算图中的信息可见性，而不是在模型中写入物理定律。模型看不到未来词元，但仍可能学到错误的动力学关系。
:::

## 代码实现：交互式视频生成器的构建

下面实现一个便于检查张量形状的教学版本。首先构建因果注意力模块和 Transformer 块。

先实现多头因果自注意力层，重点看掩码的应用。

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

这个块采用 Pre-LayerNorm，并在注意力和前馈网络外各放置一条残差连接。

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

为了实现交错排布，还要告诉模型每个视觉词元来自哪个时间和空间位置。这里把时间嵌入与空间嵌入相加；动作使用对应时间嵌入和单独的类型嵌入。一维位置编码也能工作，这里只是显式保留了两类索引。

下面的类把词元映射、位置编码、交错拼接和多层 Transformer 串在一起。

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
        if S != self.tokens_per_frame:
            raise ValueError(f"期望每帧 {self.tokens_per_frame} 个词元，实际得到 {S}")
        if T > self.temporal_pos_emb.size(1):
            raise ValueError("输入帧数超过 max_frames")
        if actions.shape[:2] != (B, max(T - 1, 0)):
            raise ValueError("actions 的前两维必须是 (B, T-1)")

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

自回归训练最大化视觉词元的条件对数似然，等价地最小化负对数似然。动作在这里作为已知条件输入，因此不要求模型预测动作。

在给定逻辑回归输出（Logits）的情况下，对于预测序列中的第 $k$ 个视觉标记（在展平序列中的真实值设为 $y_k$），我们采用标准的交叉熵损失函数（Cross-Entropy Loss）：

$$\mathcal{L} = -\frac{1}{N_{vis}} \sum_{k=1}^{N_{vis}} \log \frac{\exp(\mathbf{logits}_{k, y_k})}{\sum_{v=1}^{V} \exp(\mathbf{logits}_{k, v})}$$

其中 $V$ 是词表大小，$N_{vis}$ 是实际纳入损失的目标数。序列位置 $i$ 的输出预测位置 $i+1$：帧内位置预测下一个视觉词元，动作位置预测下一帧的第一个词元，而指向动作词元的位置不计损失。在这份实现里，$N_{vis}=T(S-1)+(T-1)$，初始帧的首个词元由外部上下文给定。

下面从展平后的输出中选出这些位置，再计算交叉熵。

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

至此，视觉词元与动作完成了序列重组、因果前向计算和视觉词元损失计算。若要变成可用的视频世界模型，还需要训练好的 tokenizer、自回归采样循环、长序列缓存、数据管线，以及针对随机未来的建模与评测。

## 小结

- **交互式视频生成**可以写成给定历史画面与动作的条件序列预测问题。
- 交错排列让预测下一帧时能够读取相应动作，但动作响应仍需从数据中学习。
- **因果自注意力**阻止训练时读取未来词元；它解决信息泄漏问题，不保证动力学本身正确。
