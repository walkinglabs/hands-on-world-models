# 5.3 扩散模型在视频生成中的应用（以 Sora 为例）

视频比静态图像多出时间维度，模型不仅要生成每一帧的外观，还要维持跨帧的物体、相机和场景一致性。Sora 的公开报告把视频压缩成时空补丁，并用扩散 Transformer 生成这些补丁 [[OpenAI, 2024]](https://openai.com/index/video-generation-models-as-world-simulators/)。本节从标量扩散公式出发，解释它怎样推广到视频潜变量；公开资料没有披露的训练和网络细节不会被当作 Sora 的确定实现。

<div align="center">
<img src="/figures/05-interactive-video/source/03-diffusion-sora/latte-fig1.png" alt="Latte 的多组生成帧展示扩散 Transformer 需要同时维持主体外观、动作与跨帧背景一致性。" width="86%">

_图 5.3-1：Latte 的多组生成帧展示扩散 Transformer 需要同时维持主体外观、动作与跨帧背景一致性。 出处：Xin Ma et al.，[Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03048)（2024），Figure 1。_
</div>

## 5.3.1 视频生成的学术追溯与高维诅咒

早期视频生成广泛探索了 GAN 与自回归模型。GAN 可能出现模式覆盖不足和训练不稳定；自回归模型则要按既定顺序逐个生成词元，推理难以沿序列维并行。VideoGPT 等方法先压缩视频再对离散潜变量建模，并非简单地逐像素扫描；即便如此，视频词元数仍随时间和空间分辨率增长。

2020 年，Ho 等人提出去噪扩散概率模型（Denoising Diffusion Probabilistic Models, DDPM）[[Ho et al., 2020]](https://arxiv.org/abs/2006.11239)。它用一个马尔可夫链逐步向数据加入高斯噪声，再训练神经网络学习逆过程。Peebles 与 Xie 随后提出 Diffusion Transformer（DiT），用 Transformer 替代常见的 U-Net 去噪骨干，并观察到计算量增加时样本质量持续改善 [[Peebles & Xie, 2023]](https://arxiv.org/abs/2212.09748)。Sora 的技术报告把视频压缩为时空潜变量块，并用 Transformer 处理这些块；报告展示了最长一分钟的视频，但没有给出“物理保真度已被解决”的定量证明 [[OpenAI, 2024]](https://openai.com/index/video-generation-models-as-world-simulators/)。

<div align="center">
<img src="/figures/05-interactive-video/source/03-diffusion-sora/dit-fig3.png" alt="DiT 将潜变量切成补丁送入 Transformer，并以条件化模块完成扩散去噪，是 Sora 所属扩散 Transformer 路线的一手前身。" width="86%">

_图 5.3-2：DiT 将潜变量切成补丁送入 Transformer，并以条件化模块完成扩散去噪，是 Sora 所属扩散 Transformer 路线的一手前身。 出处：William Peebles；Saining Xie，[Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)（2023），Figure 3。_
</div>

## 5.3.2 扩散模型基础理论：从标量到时空张量

先从一个标量开始，再把同一公式推广到视频张量。

假设我们有一个一维标量变量 $x_0$，它代表某一个像素在某一帧的精确灰度值。扩散模型的前向过程（Forward Process）可以看作是随着时间步 $t$（注意这里的 $t$ 是人为引入的扩散步数，并非视频的时间轴）逐步向 $x_0$ 中加入微小的随机扰动。

在每一个离散的时间步 $t \in \{1, 2, \dots, T_{diff}\}$，我们定义状态更新的线性递推公式为：

$$x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_t$$

其中，$\beta_t \in (0, 1)$ 是预先设定的方差超参数（也称为噪声表，Noise Schedule），而 $\epsilon_t \sim \mathcal{N}(0, 1)$ 是从标准正态分布中采样的高斯噪声。这个公式非常直观：当前状态 $x_t$ 是前一状态 $x_{t-1}$ 的衰减版本与新加入噪声的线性组合。

为了能够在训练时直接跳跃到任意时间步 $t$，我们需要将递推公式展开。令 $\alpha_t = 1 - \beta_t$，上述递推公式可以写为 $x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon_t$。将其向回展开一步，可以得到：

$$x_t = \sqrt{\alpha_t} (\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_{t-1}} \epsilon_{t-1}) + \sqrt{1 - \alpha_t} \epsilon_t$$

$$x_t = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t (1 - \alpha_{t-1})} \epsilon_{t-1} + \sqrt{1 - \alpha_t} \epsilon_t$$

独立正态变量线性组合后的方差等于各项方差之和，因此后两项可以合并成一个新的高斯噪声。递归展开并令 $\bar{\alpha}_t=\prod_{i=1}^t\alpha_i$，得到：

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

其中 $\epsilon\sim\mathcal{N}(0,1)$。噪声日程通常选择使终点分布接近标准高斯；在有限步数下是否“完全”等于纯噪声取决于具体日程。

从单一标量推广到视频张量时，记原始视频为 $\mathbf{V}_0 \in \mathbb{R}^{C \times T \times H \times W}$，其中 $C$ 为通道数，$T$ 为帧数，$H$ 和 $W$ 为画面尺寸。前向加噪过程写作：

$$\mathbf{V}_t = \sqrt{\bar{\alpha}_t} \mathbf{V}_0 + \sqrt{1 - \bar{\alpha}_t} \mathbf{E}$$

<div align="center">
<img src="/figures/05-interactive-video/latex/03-diffusion-sora/video-diffusion-two-time-axes.png" alt="一个扩散步的两项标量系数作用于完整视频张量，而帧时间是张量内部的独立索引轴" width="86%">

_图 5.3-3：扩散步 t 只选择一对全张量共享的混合系数；视频的帧索引则位于张量内部，不能与扩散时间混为一谈。_
</div>

这里，噪声张量 $\mathbf{E}$ 具有与 $\mathbf{V}_0$ 完全相同的多维形状，且其内部的每一个元素均独立服从标准正态分布。去噪模型（即我们要训练的神经网络）的任务就是接收带噪张量 $\mathbf{V}_t$ 和时间步 $t$，预测出被加入的噪声 $\mathbf{E}$。

## 5.3.3 时空补丁（Spacetime Patches）：视频的降维艺术

直接在全分辨率视频像素上做全局自注意力成本很高，因为注意力矩阵随序列长度按 $O(N^2)$ 增长。潜空间压缩与补丁化共同减少 $N$。

Vision Transformer 把二维图像划分为补丁并作为序列处理 [[Dosovitskiy et al., 2020]](https://arxiv.org/abs/2010.11929)。Sora 的公开技术报告则把压缩后的视频潜变量切成时空补丁，并把不同分辨率、宽高比和时长的数据表示为补丁序列 [[OpenAI, 2024]](https://openai.com/index/video-generation-models-as-world-simulators/)。前一篇论文提供二维补丁化的先例，后一份报告才直接支持 Sora 的具体表示。

<div align="center">
<img src="/figures/05-interactive-video/source/03-diffusion-sora/vivit-fig3.png" alt="ViViT 的 tubelet 图把连续帧切成不重叠时空块并线性嵌入，直接展示二维补丁向视频时空补丁的推广。" width="86%">

_图 5.3-4：ViViT 的 tubelet 图把连续帧切成不重叠时空块并线性嵌入，直接展示二维补丁向视频时空补丁的推广。 出处：Anurag Arnab et al.，[ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691)（2021），Figure 3。_
</div>

我们可以利用高中立体几何中“切分长方体”的直观思想来理解时空补丁。假设视频张量在空间（$H \times W$）上被均匀划分为尺寸为 $h_p \times w_p$ 的小块，同时在时间轴（$T$）上被切分为长度为 $t_p$ 的片段。那么，每一个剥离出来的时空补丁在本质上就是一个尺寸为 $C \times t_p \times h_p \times w_p$ 的局部长方体。

若三个维度都能被补丁尺寸整除，不使用重叠或填充，序列长度为：

$$N = \left( \frac{T}{t_p} \right) \times \left( \frac{H}{h_p} \right) \times \left( \frac{W}{w_p} \right)$$

每个时空补丁被展平并投影到隐藏维度 $D$。模型还需要某种位置表示来区分时间和空间位置；Sora 报告没有公开其位置编码的具体实现。

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

补丁嵌入得到的是长度为 $N$ 的**连续向量序列**，不是离散整数词元。去噪网络接收带噪补丁、扩散时间步和条件信息，预测噪声或其他等价参数化目标。

许多图像扩散模型使用 U-Net，DiT 则用 Transformer 处理潜在补丁。视频模型可以采用联合时空注意力，也可以分解空间与时间注意力；Sora 报告只明确说明使用时空补丁与 Transformer，没有公开足够细节来断言其采用下面这一个具体变体。扩散去噪通常同时访问整段带噪视频，因此这里也不是自回归意义上的因果注意力。

在多头自注意力（Multi-Head Self-Attention, MHSA）中，序列张量 $\mathbf{Z} \in \mathbb{R}^{N \times D}$ 中的每个补丁经线性映射得到查询（Query）、键（Key）和值（Value）：

$$\mathbf{Q} = \mathbf{Z} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{Z} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{Z} \mathbf{W}_V$$

注意力的全局交互权重分配由点积操作计算得出，公式如下：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}$$

联合时空注意力让一个补丁直接读取其他帧与其他空间位置的表示，因而有机会协调外观和运动。但注意力连接只提供信息通路，不保证生成结果遵守重力、碰撞或骨骼约束。

## 5.3.5 潜在空间的深度压缩（Video VAE）

直接在像素空间处理长视频的成本很高。潜在扩散先把图像压缩到低维潜空间，再执行扩散过程 [[Rombach et al., 2022]](https://arxiv.org/abs/2112.10752)；Sora 也使用独立训练的视频压缩网络把视频映射到低维潜空间，但公开报告没有说明其实现与 Latent Diffusion 完全相同 [[OpenAI, 2024]](https://openai.com/index/video-generation-models-as-world-simulators/)。

<div align="center">
<img src="/figures/05-interactive-video/source/03-diffusion-sora/ldm-fig3.png" alt="潜在扩散的感知压缩与潜空间生成两阶段结构说明为何可在较小表示上运行昂贵去噪网络。" width="86%">

_图 5.3-5：潜在扩散的感知压缩与潜空间生成两阶段结构说明为何可在较小表示上运行昂贵去噪网络。 出处：Robin Rombach et al.，[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)（2022），Figure 3。_
</div>

在视频领域，可以预训练视频压缩网络，把像素视频 $\mathbf{V}_{pixel}$ 映射到更紧凑的潜在空间：

$$\mathbf{Z}_0 = \mathcal{E}(\mathbf{V}_{pixel})$$

编码器可以同时压缩空间和时间维度。压缩比例是模型设计，不应从 Sora 的公开报告中臆测具体数值；潜在表示也可能保留纹理等低层信息，不只是“核心语义”。扩散过程随后在较小的潜在张量上进行。

采样结束得到潜在表示 $\hat{\mathbf{Z}}_0$ 后，解码器 $\mathcal{D}$ 将其还原到像素空间；重建质量受压缩模型能力限制：

$$\hat{\mathbf{V}}_{pixel} = \mathcal{D}(\hat{\mathbf{Z}}_0)$$

这种两阶段设计把视觉压缩与生成建模分开，显著减少去噪网络处理的时空元素数。

## 5.3.6 代码实现：构建极简版 Video DiT 块

下面把注意力计算写进代码，构造一个包含层归一化和前馈网络的简化 Diffusion Transformer Block。

```python
class VideoDiTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        # 实例化多头自注意力层
        self.attn = nn.MultiheadAttention(
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
        # 完整 DiT 常用时间步嵌入调制自适应层归一化 (AdaLN)
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

本节从标量前向扩散得到任意噪声步的闭式采样公式，再把它推广到视频潜变量。视频压缩降低潜变量网格大小，时空补丁把网格变成连续向量序列，Transformer 负责在这些补丁之间交换信息。Sora 的公开视频展示了较强的生成能力，同时其报告也列出物理交互、因果关系和长视频一致性等失败模式；这些仍是需要评估的问题。

## 5.3.8 练习

1. 回顾前向加噪过程。假设标量初值 $x_0 = 0.5$，前两个时间步的方差超参数为 $\beta_1 = 0.1, \beta_2 = 0.2$。手工计算随机变量 $x_2$ 的理论均值和方差。
   - **提示**：充分利用独立正态分布相加时均值与方差满足线性可加性的统计学规律。不要跳步，请先分布计算出 $\alpha_1, \alpha_2$ 以及累积连乘项 $\bar{\alpha}_2$。
2. 假设给定的待处理原始视频总共有 $16$ 帧，单帧分辨率为 $256 \times 256$，颜色通道数 $C=3$。如果在补丁嵌入层将时空补丁的超参数尺寸硬编码设定为 $t_p=2, h_p=16, w_p=16$，请你计算出展平后的 Transformer 输入序列长度 $N$ 究竟是多少？
   - **提示**：使用文中的补丁数量表达式，代入各维尺寸进行除法和连乘。
3. 为什么在处理视频张量的扩散模型中，通常需要在自注意力层的输入阶段，强行同时注入“空间位置编码”和“时间位置编码”？如果你作为架构师，鲁莽地去掉了时间位置编码，模型在生成视频时，画面可能会出现什么极为怪异的现象？
   - **提示**：核心思考点在于标准 Transformer 中自注意力机制（Self-Attention）本身是具备排列不变性（Permutation Invariance）的。
