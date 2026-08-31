# 5.3 扩散模型在视频生成中的应用（以 Sora 为例）

在前面的章节中，我们已经探讨了生成模型在离散文本和静态图像领域的突破。然而，当我们试图将这些成功经验向视频领域迁移时，往往会遭遇前所未有的阻力。视频并非仅仅是图像在时间轴上的简单堆叠，它包含了复杂的物理规律、物体间的时空交互，以及长程的因果一致性。近年来，以 Sora 为代表的视频生成模型向我们展示了惊人的物理世界模拟能力。本节将从最基础的标量扩散过程起步，抽丝剥茧地推导扩散模型（Diffusion Models）是如何与 Transformer 架构深度融合，最终演化为能够处理复杂时空张量的视频生成范式的。

## 5.3.1 视频生成的学术追溯与高维诅咒

在深入技术细节之前，我们有必要回顾一下视频生成领域的学术脉络。早期的视频生成多依赖于生成对抗网络（GANs）或是自回归模型（Autoregressive Models）。然而，这些方法在处理高分辨率和长视频时，往往会遭遇“高维诅咒”（Curse of Dimensionality）。具体而言，随着时间维度 $T$ 的增加，状态空间的可能组合呈指数级爆炸。GANs 极易在此过程中产生模式崩溃（Mode Collapse）及训练不稳定的问题；而自回归模型（例如 VideoGPT）需要进行极大规模的逐像素（Raster-scan）光栅式自回归预测，在长达数万甚至数十万的序列长度（$T \times H \times W$）前，生成速度极其缓慢且容易累积误差。

2020 年，Ho 等人提出去噪扩散概率模型（Denoising Diffusion Probabilistic Models, DDPM）[[Ho et al., 2020]](https://arxiv.org/abs/2006.11239)。它用一个马尔可夫链逐步向数据加入高斯噪声，再训练神经网络学习逆过程。Peebles 与 Xie 随后提出 Diffusion Transformer（DiT），用 Transformer 替代常见的 U-Net 去噪骨干，并观察到计算量增加时样本质量持续改善 [[Peebles & Xie, 2023]](https://arxiv.org/abs/2212.09748)。Sora 的技术报告把视频压缩为时空潜变量块，并用 Transformer 处理这些块；报告展示了最长一分钟的视频，但没有给出“物理保真度已被解决”的定量证明 [[OpenAI, 2024]](https://openai.com/index/video-generation-models-as-world-simulators/)。

## 5.3.2 扩散模型基础理论：从标量到时空张量

为了理解视频扩散模型，我们绝不能一开始就陷入复杂的四维张量运算中。让我们将视角降维，回到高中统计学中最基础的单变量场景。

假设我们有一个一维标量变量 $x_0$，它代表某一个像素在某一帧的精确灰度值。扩散模型的前向过程（Forward Process）可以看作是随着时间步 $t$（注意这里的 $t$ 是人为引入的扩散步数，并非视频的时间轴）逐步向 $x_0$ 中加入微小的随机扰动。

在每一个离散的时间步 $t \in \{1, 2, \dots, T_{diff}\}$，我们定义状态更新的线性递推公式为：

$$x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t$$

其中，$\beta_t \in (0, 1)$ 是预先设定的方差超参数（也称为噪声表，Noise Schedule），而 $\epsilon_t \sim \mathcal{N}(0, 1)$ 是从标准正态分布中采样的高斯噪声。这个公式非常直观：当前状态 $x_t$ 是前一状态 $x_{t-1}$ 的衰减版本与新加入噪声的线性组合。

为了能够在训练时直接跳跃到任意时间步 $t$，我们需要将递推公式展开。令 $\alpha_t = 1 - \beta_t$，上述递推公式可以写为 $x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_t$。将其向回展开一步，可以得到：

$$x_t = \sqrt{\alpha_t} (\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_{t-1}} \epsilon_{t-1}) + \sqrt{1 - \alpha_t} \epsilon_t$$

$$x_t = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t (1 - \alpha_{t-1})} \epsilon_{t-1} + \sqrt{1 - \alpha_t} \epsilon_t$$

此时，我们需要借助高中数学中独立正态变量相加的方差可加性原理。由于 $\epsilon_{t-1}$ 和 $\epsilon_t$ 是独立且服从标准正态分布的变量，它们线性组合的方差等于各自系数平方的和：$\alpha_t (1 - \alpha_{t-1}) + (1 - \alpha_t) = 1 - \alpha_t \alpha_{t-1}$。因此，后两项可以合并为一个新的高斯噪声。通过不断向回递归，令 $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$，我们可以严谨推导出最终的解析表达式：

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

其中 $\epsilon \sim \mathcal{N}(0, 1)$。由于 $\bar{\alpha}_t$ 随着 $t$ 的增大逐渐趋近于 0，标量 $x_t$ 最终将完全退化为纯粹的随机噪声。

当我们从单一像素过渡到宏观的视频序列时，上述标量公式可以直接推广到高维张量。定义原始视频为张量 $\mathbf{V}_0 \in \mathbb{R}^{C \times T \times H \times W}$，其中 $C$ 为通道数（例如 RGB 图像为 3），$T$ 为视频帧数，$H$ 和 $W$ 分别为单帧图像的高度和宽度。此时，视频扩散模型的前向过程严谨地表达为：

$$\mathbf{V}_t = \sqrt{\bar{\alpha}_t} \mathbf{V}_0 + \sqrt{1 - \bar{\alpha}_t} \mathbf{E}$$

这里，噪声张量 $\mathbf{E}$ 具有与 $\mathbf{V}_0$ 完全相同的多维形状，且其内部的每一个元素均独立服从标准正态分布。去噪模型（即我们要训练的神经网络）的任务就是接收带噪张量 $\mathbf{V}_t$ 和时间步 $t$，预测出被加入的噪声 $\mathbf{E}$。

## 5.3.3 时空补丁（Spacetime Patches）：视频的降维艺术

直接在全分辨率的视频张量 $\mathbf{V}_0$ 上进行自注意力计算是极其昂贵且不现实的。Transformer 的核心计算复杂度随序列长度呈平方级（$O(N^2)$）增长，而在高分辨率视频中，逐像素的序列长度为 $T \times H \times W$，这是一个计算设备无法承受的天文数字。

Vision Transformer 把二维图像划分为补丁并作为序列处理 [[Dosovitskiy et al., 2020]](https://arxiv.org/abs/2010.11929)。Sora 的公开技术报告则把压缩后的视频潜变量切成时空补丁，并把不同分辨率、宽高比和时长的数据表示为补丁序列 [[OpenAI, 2024]](https://openai.com/index/video-generation-models-as-world-simulators/)。前一篇论文提供二维补丁化的先例，后一份报告才直接支持 Sora 的具体表示。

我们可以利用高中立体几何中“切分长方体”的直观思想来理解时空补丁。假设视频张量在空间（$H \times W$）上被均匀划分为尺寸为 $h_p \times w_p$ 的小块，同时在时间轴（$T$）上被切分为长度为 $t_p$ 的片段。那么，每一个剥离出来的时空补丁在本质上就是一个尺寸为 $C \times t_p \times h_p \times w_p$ 的局部长方体。

通过这种严格的几何切分，整个庞大的视频张量被降维并转换为一个由局部补丁组成的序列，其最终的序列长度 $N$（即序列中的元素个数）被大幅缩减为：

$$N = \left( \frac{T}{t_p} \right) \times \left( \frac{H}{h_p} \right) \times \left( \frac{W}{w_p} \right)$$

每一个被切分出来的时空补丁首先会被展平（Flatten）为一维向量，随后通过一个线性投影矩阵 $\mathbf{W}_{proj}$ 映射为隐藏层维度为 $D$ 的潜在向量（Latent Vector），并加上空间与时间的三维位置编码（3D Positional Encoding），以向模型保留该补丁在原始长方体张量中的绝对位置信息。

(**下面我们用代码展示如何构建一个简化的时空补丁嵌入层。**)

```python
import torch
import torch.nn as nn

class SpacetimePatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        """
        patch_size: 一个包含 (t_p, h_p, w_p) 的整数元组
        """
        super().__init__()
        self.patch_size = patch_size
        # 使用非重叠的三维卷积来实现时空补丁的提取与线性映射投影
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x 的输入形状: (批量大小, 通道数, 帧数, 高度, 宽度) -> (B, C, T, H, W)
        x = self.proj(x)
        # 经过 3D 卷积后，x 的形状降维为 (B, embed_dim, T_prime, H_prime, W_prime)
        # 将时空网格维度完全展平为单一的序列维度
        B, D, T_prime, H_prime, W_prime = x.shape
        x = x.view(B, D, -1).transpose(1, 2)
        # 最终输出的形状为: (B, N, D)，其中 N = T_prime * H_prime * W_prime
        return x
```

## 5.3.4 联合时空自注意力机制与 DiT 架构

在成功将连续的视频转化为长度为 $N$ 的离散向量序列后，接下来的核心挑战是：模型必须学习如何逆转该公式，即在每一步预测出被加入的高斯噪声 $\mathbf{E}$。

传统的 2D 图像扩散模型多采用 U-Net 架构，而 Sora 全面倒向了 Transformer 架构（即 DiT，Diffusion Transformer）。对于视频而言，模型不仅需要捕捉单帧画面内部的空间纹理结构，更需要理解不同帧之间物体运动的时间轴因果演变。在这里，全联合时空自注意力（Full Joint Spatiotemporal Attention）机制发挥了至关重要的作用。

在多头自注意力（Multi-Head Self-Attention, MHSA）的严格数学定义中，序列张量 $\mathbf{Z} \in \mathbb{R}^{N \times D}$ 中的每个补丁都会被一组权重矩阵映射为查询（Query）、键（Key）和值（Value）：

$$\mathbf{Q} = \mathbf{Z} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{Z} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{Z} \mathbf{W}_V$$

注意力的全局交互权重分配由点积操作计算得出，公式如下：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}$$

> 💡 **精炼类比：**
> 我们可以将这种联合时空注意力机制视为一间“多维度的高级会议室”。在这个会议室里，每一个“与会者”（代表局部时空信息的补丁向量）不仅能实时听到身边邻座（即在空间上相邻的图像补丁）的发言，还能同步调取并聆听昨天或明天同一座位（即在时间轴上相邻的帧补丁）的录音记录。通过这种不遗漏任何方向的全局信息交换，模型能够在生成画面中一只奔跑的猎犬时，既保证它在当前帧的四肢比例精准协调（空间结构的物理保真），又能确保它在后续帧中的奔跑轨迹严格符合重力加速度和骨骼肌肉的运动学规律（时间连贯的物理保真）。

## 5.3.5 潜在空间的深度压缩（Video VAE）

直接在像素空间处理长视频的成本很高。潜在扩散先把图像压缩到低维潜空间，再执行扩散过程 [[Rombach et al., 2022]](https://arxiv.org/abs/2112.10752)；Sora 也使用独立训练的视频压缩网络把视频映射到低维潜空间，但公开报告没有说明其实现与 Latent Diffusion 完全相同 [[OpenAI, 2024]](https://openai.com/index/video-generation-models-as-world-simulators/)。

在视频领域，这一技术表现为预先训练一个极度强大的三维视频变分自编码器（Video VAE）。对于极高维度的输入像素视频 $\mathbf{V}_{pixel}$，编码器 $\mathcal{E}$ 将其降采样映射到一个更为紧凑和密集的潜在空间（Latent Space）：

$$\mathbf{Z}_0 = \mathcal{E}(\mathbf{V}_{pixel})$$

例如，编码器可能在空间维度上进行 8 倍的降采样，在时间维度上进行 4 倍的降采样。这意味着原本庞大的视觉冗余信息被极大地剥离，留下了高度语义化的核心特征表示。我们在前面该公式中描述的所有前向加噪、网络去噪过程，都完全发生在这个小尺寸的潜在张量 $\mathbf{Z}_0$ 上。

只有当扩散模型完全去噪，在潜在空间生成了清晰干净的潜在表示 $\hat{\mathbf{Z}}_0$ 后，解码器 $\mathcal{D}$ 才会出马，将其高保真地渲染回人类可见的像素长方体：

$$\hat{\mathbf{V}}_{pixel} = \mathcal{D}(\hat{\mathbf{Z}}_0)$$

这种两阶段的分离设计，是视频模型能够在有限算力下逼近物理世界模拟的工程基石。

## 5.3.6 代码实现：构建极简版 Video DiT 块

在这一小节，我们将把前面推导的注意力该公式落实到具体的代码逻辑中，编写一个包含层归一化（Layer Normalization）和前馈网络（MLP）的标准 Diffusion Transformer Block。

(**我们用 PyTorch 定义一个简化的 Video DiT 块。**)

```python
class VideoDiTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        # 实例化多头自注意力层，对应该公式        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        # 典型的前馈神经网络，用于对局部特征进行非线性高维映射
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x, t_emb):
        """
        x: 降维后的视频补丁序列，形状为 (B, N, D)
        t_emb: 当前时间步和条件特征（如文本提示）的融合嵌入，形状为 (B, D)
        """
        # 在严谨的工业级 DiT 架构中，时间步嵌入通常被用作自适应层归一化 (AdaLN)
        # 的动态尺度和平移参数。在此为了降低代码复杂度并突出核心逻辑，
        # 我们做简化处理，将其通过广播机制直接加到序列变量上
        x_with_cond = x + t_emb.unsqueeze(1)

        # 步骤 1：利用全序列联合自注意力机制进行时空维度的全局信息交互
        attn_out, _ = self.attn(
            self.norm1(x_with_cond),
            self.norm1(x_with_cond),
            self.norm1(x_with_cond)
        )
        x = x + attn_out  # 残差连接

        # 步骤 2：通过前馈网络完成每个序列元素的独立非线性变换
        x = x + self.mlp(self.norm2(x))
        return x
```

## 5.3.7 小结

在本节的深入探讨中，我们从一维基础标量的随机漫步方程出发，严谨推导了去噪扩散模型在物理方差上的递推解析闭式（Closed-form Expression）。为了应对视频长序列生成所带来的高维计算诅咒，我们引入了时空补丁（Spacetime Patches），通过立体几何的三维切分思想，成功将庞大且冗杂的四维视频张量降维为了 Transformer 架构能够游刃有余地处理的一维离散序列。最后，结合潜在张量深度压缩技术（Video VAE）和严谨的全时空联合多头自注意力机制，Sora 及类似架构展示了当代人工智能在理解、解构甚至模拟宏观物理规律方面不可估量的巨大潜力。

## 5.3.8 练习

1. 请仔细回顾该公式的数学推导全过程。假设某个标量的初始值 $x_0 = 0.5$，前两个时间步的方差超参数设定为 $\beta_1 = 0.1, \beta_2 = 0.2$。请你纯手工计算出 $x_2$ 这个随机变量的理论均值和理论方差。
   - **提示**：充分利用独立正态分布相加时均值与方差满足线性可加性的统计学规律。不要跳步，请先分布计算出 $\alpha_1, \alpha_2$ 以及累积连乘项 $\bar{\alpha}_2$。
2. 假设给定的待处理原始视频总共有 $16$ 帧，单帧分辨率为 $256 \times 256$，颜色通道数 $C=3$。如果在补丁嵌入层将时空补丁的超参数尺寸硬编码设定为 $t_p=2, h_p=16, w_p=16$，请你计算出展平后的 Transformer 输入序列长度 $N$ 究竟是多少？
   - **提示**：找到文章中给出的数学该公式，并直接代入已知数值进行除法和连乘运算。
3. 为什么在处理视频张量的扩散模型中，通常需要在自注意力层的输入阶段，强行同时注入“空间位置编码”和“时间位置编码”？如果你作为架构师，鲁莽地去掉了时间位置编码，模型在生成视频时，画面可能会出现什么极为怪异的现象？
   - **提示**：核心思考点在于标准 Transformer 中自注意力机制（Self-Attention）本身是具备排列不变性（Permutation Invariance）的。
