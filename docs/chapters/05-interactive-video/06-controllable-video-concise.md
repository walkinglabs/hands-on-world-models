# 5.6 可控视频生成：交叉注意力与 CFG

无条件视频模型回答“可能出现什么”，可控视频模型还要回答“在这个动作、文本或参考图条件下，会出现什么”。条件可能描述内容，也可能规定运动、姿态或相机轨迹。对世界模型而言，动作条件尤其重要，因为它把智能体的选择与后续观测连接起来。

<div align="center">
<img src="/figures/05-interactive-video/source/06-controllable-video-concise/motionctrl-fig1.png" alt="MotionCtrl 的样例分别改变相机运动、物体轨迹及两者组合，展示可控视频所要求的条件响应。" width="86%">

_图 5.6-1：MotionCtrl 的样例分别改变相机运动、物体轨迹及两者组合，展示可控视频所要求的条件响应。 出处：Zhou et al.，[MotionCtrl: A Unified and Flexible Motion Controller for Video Generation](https://arxiv.org/abs/2312.03641)（2023），Figure 1。_
</div>

本节从条件分布出发，先看加性条件注入，再实现交叉注意力（Cross-Attention）和无分类器引导（Classifier-Free Guidance, CFG）。代码用于解释这两个机制，不对应某个完整视频系统。

## 可控生成的学术脉络与概率学基础

条件生成对抗网络把类别等条件输入提供给生成器与判别器 [[Mirza & Osindero, 2014]](https://arxiv.org/abs/1411.1784)。DDPM 建立了基于逐步去噪的生成过程 [[Ho et al., 2020]](https://arxiv.org/abs/2006.11239)；潜在扩散模型进一步在潜空间中使用交叉注意力接收文本等条件 [[Rombach et al., 2022]](https://arxiv.org/abs/2112.10752)，ControlNet 则为预训练扩散模型增加边缘、深度和姿态等空间控制 [[Zhang & Agrawala, 2023]](https://arxiv.org/abs/2302.05543)。Stable Video Diffusion 研究的是图像条件潜在视频扩散及其数据、训练策略 [[Blattmann et al., 2023]](https://arxiv.org/abs/2311.15127)。这些论文分别支撑不同的条件形式，不能合并成一个笼统的“统一控制机制”。

<div align="center">
<img src="/figures/05-interactive-video/source/06-controllable-video-concise/controlnet-fig1.png" alt="ControlNet 用边缘条件约束扩散生成，为空间条件如何改变输出提供直接证据。" width="86%">

_图 5.6-2：ControlNet 用边缘条件约束扩散生成，为空间条件如何改变输出提供直接证据。 出处：Lvmin Zhang；Maneesh Agrawala，[Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)（2023），Figure 1。_
</div>

从概率角度看，无条件生成拟合边缘分布 $p(\mathbf{x})$，可控生成拟合条件分布 $p(\mathbf{x} \mid \mathbf{c})$。$\mathbf{c}$ 可以是动作序列、文本、图像、深度图或其他结构化信号；不同条件适合不同的编码器和注入位置。

<div align="center">
<img src="/figures/05-interactive-video/source/06-controllable-video-concise/dragnuwa-fig1.png" alt="DragNUWA 将文本、起始图像和轨迹组合，分别控制视频语义、外观与运动路径。" width="86%">

_图 5.6-3：DragNUWA 将文本、起始图像和轨迹组合，分别控制视频语义、外观与运动路径。 出处：Shengming Yin et al.，[DragNUWA: Fine-grained Control in Video Generation by Integrating Text, Image, and Trajectory](https://arxiv.org/abs/2308.08089)（2023），Figure 1。_
</div>

## 条件注入的数学推导：从标量到张量

先用一维状态说明条件注入，再把它推广到向量和序列。

### 标量场景：一维质点的运动控制

假设我们正在模拟一个一维空间中质点的运动。质点在当前时间步的位置为一个标量状态 $x \in \mathbb{R}$。我们希望控制质点在下一个时间步的位置 $y \in \mathbb{R}$，而我们的控制信号（例如施加的速度指令）也是一个标量 $c \in \mathbb{R}$。

一个最简单的可学习模型，是把当前状态与控制信号分别做线性变换后相加：

$$y = w_x x + w_c c + b$$

其中 $w_x,w_c$ 是可学习权重，$b$ 是偏置。它也可写成对拼接向量 $[x;c]$ 的一次线性映射。

### 向量场景：多维特征空间的映射

在真实的视频生成中，图像的特征不可能只用一个标量来表示。假设我们将一帧图像压缩成了一个 $d$ 维的潜在特征向量 $\mathbf{x} \in \mathbb{R}^d$，而我们的控制动作（例如摇杆的前后左右推力）被编码为一个 $k$ 维的向量 $\mathbf{c} \in \mathbb{R}^k$。

把标量推广到向量后，用 $\mathbf{W}_c \in \mathbb{R}^{d \times k}$ 将控制向量投影到视觉特征维度：

$$\mathbf{y} = \mathbf{W}_x \mathbf{x} + \mathbf{W}_c \mathbf{c} + \mathbf{b}$$

这仍可看作拼接后的线性投影。它为整个视觉向量注入同一份控制表示，却没有显式机制让不同空间或时间位置选择不同的条件片段。

### 张量场景：从加性注入到动态匹配（注意力机制前奏）

为了克服加性注入的局限性，我们需要一种能够度量特征向量 $\mathbf{x}$ 与控制向量 $\mathbf{c}$ 之间“相关性”的机制。这在几何上可以通过向量的内积（Dot Product）来实现。

假设我们希望控制信号根据其与当前视觉特征的相似度来动态决定注入的强度。我们可以计算两者的内积 $\alpha = \mathbf{x}^\top \mathbf{W}_{match} \mathbf{c}$，其中 $\alpha \in \mathbb{R}$ 是一个标量权重，代表了匹配程度。然后，我们将控制信号加权后注入：

$$\mathbf{y} = \mathbf{x} + \alpha (\mathbf{W}_v \mathbf{c})$$

内积越大，对应控制表示的权重越高。交叉注意力把单个控制向量扩展为一组键和值，让每个视觉位置都能从中进行加权检索。

## 交叉注意力机制

视频是一系列图像帧的序列，这就意味着我们的视觉特征实际上是一个张量。令视觉特征序列为 $\mathbf{X} \in \mathbb{R}^{N \times d}$（$N$ 是空间或时间序列的长度，$d$ 是特征维度），控制信号序列为 $\mathbf{C} \in \mathbb{R}^{M \times k}$（例如 $M$ 个文本词向量或动作序列）。

首先用三个参数矩阵生成查询（Query）、键（Key）和值（Value）。这里查询来自视觉特征，键和值来自控制序列：

$$\mathbf{Q} = \mathbf{X} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{C} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{C} \mathbf{W}_V$$

其中，$\mathbf{W}_Q \in \mathbb{R}^{d \times d_h}$, $\mathbf{W}_K \in \mathbb{R}^{k \times d_h}$, $\mathbf{W}_V \in \mathbb{R}^{k \times d_v}$。此时，$\mathbf{Q} \in \mathbb{R}^{N \times d_h}$ 和 $\mathbf{K} \in \mathbb{R}^{M \times d_h}$ 被投影到了相同的隐含维度 $d_h$。

矩阵乘法 $\mathbf{Q} \mathbf{K}^\top \in \mathbb{R}^{N \times M}$ 一次计算所有位置与条件元素的内积。点积的方差会随维度 $d_h$ 增大。除以 $\sqrt{d_h}$ 可控制分数尺度，减少 Softmax 过早饱和，再逐行归一化得到权重矩阵 $\mathbf{A}$：

$$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_h}}\right) \in \mathbb{R}^{N \times M}$$

最终的输出是将注意力矩阵 $\mathbf{A}$ 应用于值矩阵 $\mathbf{V}$，完成对控制信号的动态提取：

$$\text{CrossAttention}(\mathbf{X}, \mathbf{C}) = \mathbf{A} \mathbf{V} \in \mathbb{R}^{N \times d_v}$$

::: info 把交叉注意力看成检索
每个视觉位置提出一个查询，与控制序列的键逐一比较，再按权重汇总相应的值。它提供了“按位置选择条件”的通道，但是否选中正确条件仍要通过训练学习。
:::

## 无分类器引导（Classifier-Free Guidance）

仅有交叉注意力并不能保证模型严格遵循控制信号。无分类器引导（Classifier-Free Guidance, CFG）在训练时随机丢弃条件，推理时组合条件与无条件预测，从而调节样本质量与条件一致性 [[Ho & Salimans, 2022]](https://arxiv.org/abs/2207.12598)。它被许多扩散生成系统采用，但公开资料不足以支持“所有顶级视频系统都使用 CFG”这一绝对说法。

<div align="center">
<img src="/figures/05-interactive-video/source/06-controllable-video-concise/cfg-fig1.png" alt="分类器无关引导原图从左到右增大引导强度，显示条件一致性增强同时也改变样本多样性与饱和度。" width="86%">

_图 5.6-4：分类器无关引导原图从左到右增大引导强度，显示条件一致性增强同时也改变样本多样性与饱和度。 出处：Jonathan Ho；Tim Salimans，[Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)（2022），Figure 1。_
</div>

从贝叶斯关系出发，条件分布的得分可以分解为：

$$\nabla_{\mathbf{x}} \log p(\mathbf{x} | c) = \nabla_{\mathbf{x}} \log p(\mathbf{x}) + \nabla_{\mathbf{x}} \log p(c | \mathbf{x})$$

右侧第二项反映条件 $c$ 对样本方向的影响。若用 $w\ge 0$ 放大这部分，得到：

$$\nabla_{\mathbf{x}} \log p_w(\mathbf{x} | c) = \nabla_{\mathbf{x}} \log p(\mathbf{x}) + (1 + w) \nabla_{\mathbf{x}} \log p(c | \mathbf{x})$$

再代入 $\nabla_{\mathbf{x}} \log p(c | \mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x} | c) - \nabla_{\mathbf{x}} \log p(\mathbf{x})$：

$$\nabla_{\mathbf{x}} \log p_w(\mathbf{x} | c) = (1 + w) \nabla_{\mathbf{x}} \log p(\mathbf{x} | c) - w \nabla_{\mathbf{x}} \log p(\mathbf{x})$$

在采用噪声预测参数化的扩散模型中，相同的线性组合可写为：

$$\tilde{\epsilon}_\theta(\mathbf{x}_t, t, c) = (1 + w) \epsilon_\theta(\mathbf{x}_t, t, c) - w \epsilon_\theta(\mathbf{x}_t, t, \emptyset)$$

<div align="center">
<img src="/figures/05-interactive-video/latex/06-controllable-video-concise/cfg-affine-extrapolation.png" alt="CFG 从无条件预测指向条件预测并继续外推，正文额外引导量 w 与代码尺度 s 满足 s 等于 1 加 w" width="86%">

_图 5.6-5：正文用 w 表示越过条件预测后的额外外推量，代码常用 s 从无条件预测开始计量；两者满足 s=1+w，所以 w=0 与 s=1 都等于条件预测。本文根据上式绘制。_
</div>

因此，同一个网络只要同时学过条件输入 $c$ 和空条件 $\emptyset$，推理时就能在两次预测之间做线性外推。增大 $w$ 往往提高条件一致性，但过大也可能降低多样性、放大伪影或使颜色过饱和。

## 简洁实现

下面先实现交叉注意力，再把它放进一个简化的条件去噪块。

交叉注意力采用多头形式：多个子空间分别执行检索，再拼接输出。

```python
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

接着把交叉注意力放进残差块。这里用自注意力混合视觉词元，用交叉注意力读取外部条件；真实 U-Net 或 DiT 的层次结构会更完整。

```python
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

CFG 推理需要条件预测和无条件预测。工程上通常把两份输入沿 batch 维拼接，以一次前向计算得到两者；下面分开书写，便于理解。

```python
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

    # 常见约定：s=1 时退化为条件预测；s>1 时向条件方向外推
    eps_cfg = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    return eps_cfg
```

## 训练目标

在训练阶段，我们需要网络不仅学会从输入中去噪，还要学会将去噪过程与控制信号相关联。在实践中，为了支持无分类器引导，我们需要以一定概率（例如 $10\%$ 到 $20\%$）随机将控制信号替换为空标记（Null Token）。其训练目标的损失函数可以表示为：

$$ \mathcal{L} = \mathbb{E}_{\mathbf{x}_0, c, \epsilon \sim \mathcal{N}(0, \mathbf{I}), t} \left[ \left\| \epsilon - \epsilon_\theta(\mathbf{x}_t, t, c_\phi) \right\|^2_2 \right] $$

其中 $c_\phi$ 以概率 $1-p_{uncond}$ 取真实条件 $c$，以概率 $p_{uncond}$ 取空条件 $\emptyset$。这样，同一组参数会同时接触条件与无条件去噪样本，为推理时的两次预测提供基础。

## 小结

- 可控视频生成的核心是将无条件的边缘概率分布估计转化为**条件分布估计**。
- 加性投影提供全局条件，交叉注意力则允许不同视觉位置检索不同的条件片段。
- **交叉注意力**允许特征序列的每个元素根据查询 $\mathbf{Q}$ 动态检索控制序列的键 $\mathbf{K}$ 并聚合值 $\mathbf{V}$。
- **无分类器引导（CFG）**在条件与无条件预测之间做线性外推，用引导强度交换条件一致性、多样性与稳定性。
