# 可控视频生成模块的简洁实现

在深入探讨了无条件视频生成的理论与工程细节之后，我们自然而然地面临一个更为核心的问题：如何让模型按照我们的意图生成特定的视频内容？对于构建“世界模型”（World Models）而言，这种基于意图的条件生成——特别是基于动作（Action）或文本指令（Text Prompt）的生成——是实现智能体与环境交互的基石。

在本节中，我们将从概率生成模型的条件分布讲起，逐步拆解并实现一个现代的可控视频生成模块。我们将追溯可控生成的学术脉络，从简单的标量条件注入，一路推导至当前工业界广泛采用的交叉注意力机制（Cross-Attention）与无分类器引导（Classifier-Free Guidance）。

## 可控生成的学术脉络与概率学基础

在深度学习的早期探索中，使生成模型具备可控性往往依赖于直接拼接条件变量。例如，在条件生成对抗网络（Conditional GANs）[[Mirza & Osindero, 2014]](https://arxiv.org/abs/1411.1784)中，研究者通过将类别标签直接拼接到生成器和判别器的输入特征中来控制输出。随着扩散模型（Diffusion Models）[[Ho et al., 2020]](https://arxiv.org/abs/2006.11239)的崛起，去噪扩散概率模型（DDPM）展示了极其强大的无条件生成能力。为了将这种能力扩展至可控生成，随后提出的潜在扩散模型（Latent Diffusion Models）[[Rombach et al., 2022]](https://arxiv.org/abs/2112.10752)和ControlNet[[Zhang & Agrawala, 2023]](https://arxiv.org/abs/2302.05543)，彻底奠定了现代条件注入的基础。在视频领域，如Stable Video Diffusion[[Blattmann et al., 2023]](https://arxiv.org/abs/2311.15127)，则将这些条件控制机制延展到了时间维度。

从纯粹的概率学视角来看，无条件视频生成旨在拟合真实视频数据的边缘概率分布 $p(\mathbf{x})$。而可控生成，本质上是将目标转化为拟合一个条件概率分布 $p(\mathbf{x} \mid \mathbf{c})$，其中 $\mathbf{c}$ 代表控制信号。在世界模型中，这个控制信号 $\mathbf{c}$ 通常是智能体在时间步 $t$ 执行的动作序列（Action Sequence）或高维条件图。

## 条件注入的数学推导：从标量到张量

为了严谨地理解控制信号是如何影响视频生成的，我们不能直接跳跃到复杂的注意力机制。相反，让我们退回到最基础的物理运动场景，从一个初中物理中的一维质点运动开始推导。

### 标量场景：一维质点的运动控制

假设我们正在模拟一个一维空间中质点的运动。质点在当前时间步的位置为一个标量状态 $x \in \mathbb{R}$。我们希望控制质点在下一个时间步的位置 $y \in \mathbb{R}$，而我们的控制信号（例如施加的速度指令）也是一个标量 $c \in \mathbb{R}$。

根据基本的运动学规律，最简单的控制关系是线性的。下一个状态 $y$ 可以表示为当前状态的维持与控制信号的叠加。如果我们引入一组可学习的权重参数，这种关系可以被严谨地表述为：

$$y = w_x x + w_c c + b$$

其中，$w_x, w_c \in \mathbb{R}$ 分别是状态特征和控制信号的权重，$b \in \mathbb{R}$ 是偏置项。该公式告诉我们，控制信号 $c$ 是通过**加法（Addition）**的形式直接注入到目标状态中的。在早期的神经网络中，这等价于将 $x$ 和 $c$ 进行拼接（Concatenation）后通过一个全连接层。

### 向量场景：多维特征空间的映射

在真实的视频生成中，图像的特征不可能只用一个标量来表示。假设我们将一帧图像压缩成了一个 $d$ 维的潜在特征向量 $\mathbf{x} \in \mathbb{R}^d$，而我们的控制动作（例如摇杆的前后左右推力）被编码为一个 $k$ 维的向量 $\mathbf{c} \in \mathbb{R}^k$。

要将 $k$ 维的控制信号注入到 $d$ 维的特征中，我们需要将标量该公式泛化到向量空间。这要求我们引入一个变换矩阵 $\mathbf{W}_c \in \mathbb{R}^{d \times k}$，将控制向量投影到与视觉特征相同的维度空间中：

$$\mathbf{y} = \mathbf{W}_x \mathbf{x} + \mathbf{W}_c \mathbf{c} + \mathbf{b}$$

该公式就是经典的“特征拼接投影”（Feature Concatenation and Projection）的严格数学表达。由于矩阵乘法是线性变换，这种加性注入方式虽然简单，但存在严重的局限性：**控制信号 $\mathbf{c}$ 对输出 $\mathbf{y}$ 的影响是全局恒定的，它无法根据视觉特征 $\mathbf{x}$ 本身的内容动态地调整控制强度。**

### 张量场景：从加性注入到动态匹配（注意力机制前奏）

为了克服加性注入的局限性，我们需要一种能够度量特征向量 $\mathbf{x}$ 与控制向量 $\mathbf{c}$ 之间“相关性”的机制。这在几何上可以通过向量的内积（Dot Product）来实现。

假设我们希望控制信号根据其与当前视觉特征的相似度来动态决定注入的强度。我们可以计算两者的内积 $\alpha = \mathbf{x}^\top \mathbf{W}_{match} \mathbf{c}$，其中 $\alpha \in \mathbb{R}$ 是一个标量权重，代表了匹配程度。然后，我们将控制信号加权后注入：

$$\mathbf{y} = \mathbf{x} + \alpha (\mathbf{W}_v \mathbf{c})$$

在该公式中，如果视觉特征与控制信号高度匹配（内积大），控制信号就会被强烈地注入；反之则被忽略。这一由加法转向乘法（动态加权）的演进，正是现代交叉注意力机制（Cross-Attention）的核心数学基石。

## 交叉注意力机制的严谨解构

视频是一系列图像帧的序列，这就意味着我们的视觉特征实际上是一个张量。令视觉特征序列为 $\mathbf{X} \in \mathbb{R}^{N \times d}$（$N$ 是空间或时间序列的长度，$d$ 是特征维度），控制信号序列为 $\mathbf{C} \in \mathbb{R}^{M \times k}$（例如 $M$ 个文本词向量或动作序列）。

我们将基于该公式的内积思想，严密推导标准的交叉注意力机制。首先，我们用三个不同的参数矩阵对输入进行线性变换，分别生成查询（Query）、键（Key）和值（Value）。这里，**查询来自于视觉特征，而键和值来自于控制信号**：

$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{C} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{C} \mathbf{W}_V$$

其中，$\mathbf{W}_Q \in \mathbb{R}^{d \times d_h}$, $\mathbf{W}_K \in \mathbb{R}^{k \times d_h}$, $\mathbf{W}_V \in \mathbb{R}^{k \times d_v}$。此时，$\mathbf{Q} \in \mathbb{R}^{N \times d_h}$ 和 $\mathbf{K} \in \mathbb{R}^{M \times d_h}$ 被投影到了相同的隐含维度 $d_h$。

接下来，我们需要计算每一个视觉特征元素与每一个控制元素的匹配程度。通过矩阵乘法 $\mathbf{Q} \mathbf{K}^\top \in \mathbb{R}^{N \times M}$，我们一次性计算出了所有 $N \times M$ 个组合的内积相似度。为了防止内积值随维度 $d_h$ 增大而导致梯度消失，我们除以缩放因子 $\sqrt{d_h}$，并在每一行应用 Softmax 函数，得到归一化的注意力权重矩阵 $\mathbf{A}$：

$$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}}\right) \in \mathbb{R}^{N \times M}$$

最终的输出是将注意力矩阵 $\mathbf{A}$ 应用于值矩阵 $\mathbf{V}$，完成对控制信号的动态提取：

$$\text{CrossAttention}(\mathbf{X}, \mathbf{C}) = \mathbf{A} \mathbf{V} \in \mathbb{R}^{N \times d_v}$$

> [!NOTE]
> **极其克制的唯一类比**
> 如果我们必须用一个生活中的物理过程来比喻交叉注意力机制：想象你（视觉特征 $\mathbf{X}$）走进一个庞大的中药铺（控制信号集合 $\mathbf{C}$）。你手里拿着一张具体的药方（生成查询 $\mathbf{Q}$），而药铺里的每一个药屉外都贴着标签（生成键 $\mathbf{K}$）。你逐一比对药方上的名字和药屉上的标签（计算点积 $\mathbf{Q}\mathbf{K}^\top$ 并通过 Softmax 决定匹配度 $\mathbf{A}$）。匹配度越高的药屉，你从里面抓取的药材（提取值 $\mathbf{V}$）就越多。最终你带走的是所有药材按比例混合后的一包新药（输出 $\mathbf{A}\mathbf{V}$）。这种“基于查询的加权检索”过程，保证了生成模型能够极其精准地捕捉到与当前视觉区域最相关的控制指令，而不是盲目地接受所有外部输入。

## 无分类器引导（Classifier-Free Guidance）

仅有交叉注意力并不能保证模型严格遵循控制信号。无分类器引导（Classifier-Free Guidance, CFG）在训练时随机丢弃条件，推理时组合条件与无条件预测，从而调节样本质量与条件一致性 [[Ho & Salimans, 2022]](https://arxiv.org/abs/2207.12598)。它被许多扩散生成系统采用，但公开资料不足以支持“所有顶级视频系统都使用 CFG”这一绝对说法。

CFG的数学本质源自对条件概率的贝叶斯重写。根据贝叶斯定理，条件数据分布的对数似然梯度（即得分函数 Score Function）可以分解为：

$$\nabla_{\mathbf{x}} \log p(\mathbf{x} | c) = \nabla_{\mathbf{x}} \log p(\mathbf{x}) + \nabla_{\mathbf{x}} \log p(c | \mathbf{x})$$

在这个公式中，右侧第二项 $\nabla_{\mathbf{x}} \log p(c | \mathbf{x})$ 可以被视为一种“引导力”，它促使生成的图像 $\mathbf{x}$ 更符合条件 $c$。CFG的核心洞察是：我们可以人为地放大这种引导力。通过引入一个控制强度的标量系数 $w > 0$，我们定义一个修正后的引导得分：

$$\nabla_{\mathbf{x}} \log p_w(\mathbf{x} | c) = \nabla_{\mathbf{x}} \log p(\mathbf{x}) + (1 + w) \nabla_{\mathbf{x}} \log p(c | \mathbf{x})$$

现在，我们将该公式中的隐式引导项 $\nabla_{\mathbf{x}} \log p(c | \mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x} | c) - \nabla_{\mathbf{x}} \log p(\mathbf{x})$ 代入到该公式中，得到：

$$\nabla_{\mathbf{x}} \log p_w(\mathbf{x} | c) = (1 + w) \nabla_{\mathbf{x}} \log p(\mathbf{x} | c) - w \nabla_{\mathbf{x}} \log p(\mathbf{x})$$

在扩散模型的框架下，得分函数被参数化的噪声预测网络 $\epsilon_\theta$ 所等价替代。因此，在推理阶段，模型预测的最终噪声 $\tilde{\epsilon}$ 为：

$$\tilde{\epsilon}_\theta(\mathbf{x}_t, t, c) = (1 + w) \epsilon_\theta(\mathbf{x}_t, t, c) - w \epsilon_\theta(\mathbf{x}_t, t, \emptyset)$$

该公式优雅地告诉我们：只需要一个同时支持条件输入 $c$ 和空条件 $\emptyset$ 的单一网络，通过取条件预测与无条件预测的线性外推（Extrapolation，因为 $1+w > 1$ 且系数和为1），就能在不引入额外分类器网络的情况下，极其强劲地提升控制信号的服从度。

## 简洁实现

基于上述严谨的数学推导，我们现在开始用代码实现这个可控视频生成模块。我们将首先实现交叉注意力机制，然后将其集成到一个微型的条件去噪网络块中。

(**我们定义交叉注意力层**)，严格按照这两个公式进行矩阵运算。为了工程上的高效，我们通常使用多头注意力机制（Multi-Head Attention），即在多个子空间中独立执行交叉注意力，最后将结果拼接。

```{.python .input}
#@tab pytorch
import torch
from torch import nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    """交叉注意力机制的简洁实现"""
    def __init__(self, visual_dim, control_dim, num_heads=8, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        
        # 视觉特征生成 Query
        self.to_q = nn.Linear(visual_dim, inner_dim, bias=False)
        # 控制信号生成 Key 和 Value
        self.to_k = nn.Linear(control_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(control_dim, inner_dim, bias=False)
        
        # 最终的输出投影
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, visual_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x, context):
        """
        x: 视觉特征，形状为 (batch_size, sequence_length, visual_dim)
        context: 控制信号，形状为 (batch_size, context_length, control_dim)
        """
        batch_size = x.shape[0]
        
        # 计算 Q, K, V
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # 将张量重塑为多头形式: (batch, seq_len, heads, head_dim) -> (batch, heads, seq_len, head_dim)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力权重: 公式 (4)
        # q: (B, H, N, d_h), k.transpose: (B, H, d_h, M) -> sim: (B, H, N, M)
        sim = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(sim, dim=-1)
        
        # 应用注意力矩阵提取特征: 公式 (5)
        # attn: (B, H, N, M), v: (B, H, M, d_v) -> out: (B, H, N, d_v)
        out = torch.matmul(attn, v)
        
        # 还原张量形状并投影回视觉维度
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        return self.to_out(out)
```

接下来，(**我们将交叉注意力层嵌入到生成网络的残差块中**)。在现代扩散模型（如 DiT 或 U-Net）中，自注意力机制负责视觉特征内部的时空一致性，而交叉注意力模块则专门负责吸收外部条件。

```{.python .input}
#@tab pytorch
class ConditionalVideoBlock(nn.Module):
    """带条件控制的视频生成残差块"""
    def __init__(self, visual_dim, control_dim, num_heads=8):
        super().__init__()
        # 时空自注意力（简化为处理被展平的序列）
        self.self_attn = nn.MultiheadAttention(embed_dim=visual_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(visual_dim)
        
        # 交叉注意力用于注入控制条件
        self.cross_attn = CrossAttention(visual_dim=visual_dim, control_dim=control_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(visual_dim)
        
        # 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(visual_dim, visual_dim * 4),
            nn.GELU(),
            nn.Linear(visual_dim * 4, visual_dim)
        )
        self.norm3 = nn.LayerNorm(visual_dim)

    def forward(self, x, context):
        """
        前向传播：先处理自注意力，再处理交叉条件注意力。
        """
        # 1. 自注意力：维持视觉连贯性
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # 2. 交叉注意力：注入外部控制（如动作或指令）
        x = x + self.cross_attn(self.norm2(x), context)
        
        # 3. 特征映射
        x = x + self.ffn(self.norm3(x))
        return x
```

为了支持我们在相关章节中讨论的无分类器引导（CFG），(**我们在模型的前向推理逻辑中必须同时计算条件生成和无条件生成的预测值**)。

```{.python .input}
#@tab pytorch
def classifier_free_guidance_step(model_block, x_t, context, unconditional_context, guidance_scale=7.5):
    """
    无分类器引导的单步推理。
    x_t: 当前时间步的含噪视觉序列
    context: 真实的控制信号 (C)
    unconditional_context: 空白控制信号 (空集)
    """
    # 获取条件预测 eps(x_t, c)
    eps_cond = model_block(x_t, context)
    
    # 获取无条件预测 eps(x_t, empty)
    eps_uncond = model_block(x_t, unconditional_context)
    
    # 执行无分类器引导外推公式：公式 (8)
    eps_cfg = (1 + guidance_scale) * eps_cond - guidance_scale * eps_uncond
    
    return eps_cfg
```

## 训练目标

在训练阶段，我们需要网络不仅学会从输入中去噪，还要学会将去噪过程与控制信号相关联。在实践中，为了支持无分类器引导，我们需要以一定概率（例如 $10\%$ 到 $20\%$）随机将控制信号替换为空标记（Null Token）。其训练目标的损失函数可以表示为：

$$ \mathcal{L} = \mathbb{E}_{\mathbf{x}_0, c, \epsilon \sim \mathcal{N}(0, \mathbf{I}), t} \left[ \left\| \epsilon - \epsilon_\theta(\mathbf{x}_t, t, c_\phi) \right\|^2_2 \right] $$

其中 $c_\phi$ 有 $(1 - p_{uncond})$ 的概率是真实控制信号 $c$，有 $p_{uncond}$ 的概率是被屏蔽的空信号 $\emptyset$。这确保了单个模型 $\epsilon_\theta$ 在同一个权重空间中同时学会了 $p(\mathbf{x} \mid c)$ 和 $p(\mathbf{x})$ 的去噪能力。

## 小结

* 可控视频生成的核心是将无条件的边缘概率分布估计转化为条件分布估计。
* 从简单的标量加性注入，到向量空间的拼接，再到现代基于内积度量匹配度的交叉注意力机制，条件注入的方法在数学演进上日益严密且具有表达力。
* 交叉注意力允许特征序列的每个元素根据查询 $\mathbf{Q}$ 动态检索控制序列的键 $\mathbf{K}$ 并聚合值 $\mathbf{V}$。
* 无分类器引导（CFG）利用贝叶斯法则，通过推断阶段的线性外推，极其有效地放大了控制信号的权重，克服了条件坍塌问题。
