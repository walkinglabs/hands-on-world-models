# 5.3 视频扩散模型、DiT 架构与 Sora 物理先验

在生成式人工智能的发展史中，2024 年初 OpenAI 发布的 **Sora** 如同一道划破天际的闪电，将世界模型的概念推向了全球科技界的风口浪尖。

Sora 展现出了一种令人震撼的“物理世界模拟能力”：在长达 60 秒的高清视频生成中，镜头在复杂的三维城市中穿梭旋转，被遮挡的建筑物在转弯后能够准确重现（三维空间恒常性）；咖啡杯在跌落碎裂时，液体会自发地按照流体力学在桌面上扩散渗透；帆船在激流中航行时，船体与水面波浪之间能够产生精确的流体动力学浮力反馈。

OpenAI 在技术报告中将其定义为 **“物理世界的通用模拟器（World Simulators）”**。

这一技术奇迹背后的核心架构，正是 **扩散 Transformer（Diffusion Transformer, DiT）** 与 **三维时空潜变量切片（Spatiotemporal Patches）**。

通过彻底抛弃传统的 2D/3D U-Net 卷积骨架，全面拥抱纯自注意力 Transformer，DiT 展现出了无可比拟的**算力扩展律（Scaling Law）**——随着模型参数量与训练 Token 数量的指数级暴增，模型自发地在神经网络内部涌现出了深邃的初等物理世界先验规律！

本节我们将从初等三维空间几何坐标变换出发，严密推导 DiT 的 3D Patch 展开公式、adaLN-Zero 自适应层归一化调制机制与 3D 旋转位置编码（3D-RoPE），并使用纯底层 PyTorch 从零手写一个时空 DiT 核心生成块。

<div align="center">

<img src="/figures/05-interactive-video/source/03-diffusion-sora/latte-fig1.png" alt="Diffusion Transformer (DiT) 架构：将潜特征切分为 Patch 序列，并结合自适应层归一化 (adaLN-Zero) 实现扩散去噪。" width="86%">

_图 5.3-1：Diffusion Transformer (DiT) 架构：将潜特征切分为 Patch 序列，并结合自适应层归一化 (adaLN-Zero) 实现扩散去噪。 出处：[Scalable Diffusion Models with Transformers，William Peebles & Saining Xie，2023](https://arxiv.org/abs/2212.09748)。_

</div>

---

## 5.3.1 物理与几何基石：从 U-Net 感受野局限到 DiT 全局时空注意力

要理解 DiT 相比传统扩散模型的架构飞跃，我们首先必须审视传统卷积 U-Net 在处理长程物理视频时的先天软肋。

### 1. 传统 2D/3D U-Net 的“局部视野短板”
传统的视频扩散模型（如早期 Video LDM）依赖多层卷积核提取特征。
- 卷积核的局部感受野使得模型很难在远距离的时空两端建立直接联系；
- 当一辆汽车在第 1 秒驶入隧道被完全遮挡，并在第 10 秒从隧道另一端驶出时，卷积网络早已遗忘了汽车最初的颜色、车牌与型号，导致汽车驶出时发生了荒谬的“变形换车”。

### 2. DiT 的全时空 Patch 展开（Spatiotemporal Patches as Tokens）
DiT 将四维潜在视频张量 $\mathbf{Z} \in \mathbb{R}^{C \times T \times H \times W}$ 视为一个连续的三维物理时空立方体：
- 沿着时间轴以步长 $p_t$、高度轴步长 $p_h$、宽度轴步长 $p_w$ 进行网格切割（如 $p_t = 2, p_h = 2, p_w = 2$）；
- 切出的每一个小立方块被展平为一维向量，并通过线性层映射为一个标准的 Transformer 词元（Token）；
- 全局自注意力使得任意一个时刻的任意像素点，都能够在单个计算层内与全时空其他所有位置进行无死角的直接特征交互，彻底攻克了三维长程物理一致性难题！

<div align="center">

<img src="/figures/05-interactive-video/latex/03-diffusion-sora/video-diffusion-two-time-axes.png" alt="DiT 核心 Block 架构：adaLN-Zero 调制六大动态参数与零初始化残差直通" width="86%">

_图 5.3-2：DiT 核心 Block 架构：adaLN-Zero 调制六大动态参数与零初始化残差直通。_

</div>

---

## 5.3.2 核心数学推导一：3D Spatiotemporal Patch 切片与 adaLN-Zero 零初始化

在 DiT 架构中，如何将扩散时间步 $t$ 与文本/动作控制条件 $\mathbf{c}$ 注入到百亿参数的深层网络中？

<div align="center">

<img src="/figures/05-interactive-video/source/03-diffusion-sora/latte-fig1.png" alt="DiT 在不同模型规模 (G/2, B/4) 下展示随着计算量增加性能单调提升的 Scaling Law 曲线。" width="86%">

_图 5.3-3：DiT 在不同模型规模 (G/2, B/4) 下展示随着计算量增加性能单调提升的 Scaling Law 曲线。 出处：[Scalable Diffusion Models with Transformers，William Peebles & Saining Xie，2023](https://arxiv.org/abs/2212.09748)。_

</div>

### 1. 3D Patch 词元序列长度初等代数计算
设输入潜在视频张量维度为 $(T, H, W)$，切片大小为 $(p_t, p_h, p_w)$。
总词元序列长度 $N$ 严格满足初等三维网格剖分公式：

$$N = \frac{T}{p_t} \times \frac{H}{p_h} \times \frac{W}{p_w}$$

每个词元的输入向量维度为 $D_{\text{in}} = C \cdot p_t \cdot p_h \cdot p_w$。

### 2. adaLN-Zero（自适应层归一化与零初始化门控）
传统的 Cross-Attention 注入条件计算量庞大。DiT 采用了极简而高效的 **adaLN-Zero** 机制：
通过一个轻量级 MLP，根据扩散时间步嵌入与条件向量 $\mathbf{y} = \text{Embedding}(t) + \text{Embedding}(\mathbf{c})$，一次性回归预测出 6 个关键调制标量参数：

$$(\boldsymbol{\gamma}_1, \; \boldsymbol{\beta}_1, \; \boldsymbol{\alpha}_1, \; \boldsymbol{\gamma}_2, \; \boldsymbol{\beta}_2, \; \boldsymbol{\alpha}_2) = \text{MLP}(\mathbf{y})$$

DiT 核心计算块的数学方程为：

$$\mathbf{x}' = \mathbf{x} + \boldsymbol{\alpha}_1 \odot \text{MultiHeadAttention}\left( \boldsymbol{\gamma}_1 \odot \text{LayerNorm}(\mathbf{x}) + \boldsymbol{\beta}_1 \right)$$

$$\mathbf{x}_{\text{out}} = \mathbf{x}' + \boldsymbol{\alpha}_2 \odot \text{FeedForwardNetwork}\left( \boldsymbol{\gamma}_2 \odot \text{LayerNorm}(\mathbf{x}') + \boldsymbol{\beta}_2 \right)$$

### 3. adaLN-Zero 零初始化手算数值算例
在模型刚初始化的第 0 步，将 MLP 输出 $\boldsymbol{\alpha}_1, \boldsymbol{\alpha}_2$ 的最后一层权重与偏置全部**初始化为严格的 0**（$\boldsymbol{\alpha}_1 = \mathbf{0}, \boldsymbol{\alpha}_2 = \mathbf{0}$），而将缩放因子初始化为 1（$\boldsymbol{\gamma} = \mathbf{1}, \boldsymbol{\beta} = \mathbf{0}$）。

设输入特征向量为 $\mathbf{x} = [3.0, -1.0]^\top$。经过注意力与 FFN 计算后产生了大震荡的中间特征 $\mathbf{F} = [100.0, -50.0]^\top$。
我们来手动计算此时的最终输出：
$$\mathbf{x}_{\text{out}} = \mathbf{x} + \boldsymbol{\alpha}_1 \odot \mathbf{F} = \begin{bmatrix} 3.0 \\ -1.0 \end{bmatrix} + \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix} \odot \begin{bmatrix} 100.0 \\ -50.0 \end{bmatrix} = \begin{bmatrix} 3.0 \\ -1.0 \end{bmatrix} + \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 3.0 \\ -1.0 \end{bmatrix} = \mathbf{x}$$

初等代数的几步加减乘除生动揭示了 adaLN-Zero 的设计精妙：在训练开始的第一瞬间，深达数十层的 DiT 骨干网络全部**恒等退化为一个纯粹的直通连接（Identity Map）**！梯度能够以 $100\%$ 的原始强度直接贯穿整个网络，彻底杜绝了深层百亿大模型初始阶段的数值发散！

<details>
<summary><b>深入推导：adaLN-Zero 动态调制在无穷深度网络收敛性下的利普希茨条件数稳定性证明（点击展开查看完整推导）</b></summary>

将深度为 $L$ 的残差动力学系统建模为非线性常微分方程（ODE）离散化序列 $\mathbf{x}_{l+1} = \mathbf{x}_l + \alpha_l f(\mathbf{x}_l, \theta_l)$。
整个系统的雅可比条件数满足上界：
$$\kappa\left( \frac{\partial \mathbf{x}_L}{\partial \mathbf{x}_0} \right) \le \prod_{l=1}^L (1 + \|\alpha_l\| \cdot \|J_{f_l}\|) \le \exp\left( \sum_{l=1}^L \|\alpha_l\| \cdot \|J_{f_l}\| \right)$$
当零初始化 $\alpha_l = 0$ 时，初始李雅普诺夫指数严格为 0，雅可比矩阵严格退化为恒等矩阵 $\mathbf{I}$。随着参数平滑演进，梯度流的谱半径始终被锚定在单位圆盘附近，从数学上根除了梯度爆炸。
</details>

---

## 5.3.3 核心数学推导二：3D 旋转位置编码 (3D-RoPE)

在连续视频中，每个 Patch 词元同时拥有时间坐标 $t \in [0, T)$、垂直高度坐标 $h \in [0, H)$ 与水平宽度坐标 $w \in [0, W)$。

为了让注意力机制直接感知三维物理空间中的**相对距离与相对位移**，现代视频生成普遍采用 **三维旋转位置编码（3D-RoPE）**。

将隐藏特征通道切分为三个正交子空间 $D = D_t + D_h + D_w$：
对于在时空坐标 $(t, h, w)$ 处的特征向量，分别沿三个正交轴执行二维复数吉文斯旋转（Givens Rotation）：

$$\mathbf{R}_{3D}(t, h, w) = \text{diag}\left( \mathbf{R}_{\theta_t}(t), \; \mathbf{R}_{\theta_h}(h), \; \mathbf{R}_{\theta_w}(w) \right)$$

任意两个时空词元之间的自注意力点积直接内化了其相对时间差 $\Delta t$ 与相对空间距离 $(\Delta h, \Delta w)$：

$$\langle \mathbf{R}_{3D}(t_1, h_1, w_1) \mathbf{q}, \; \mathbf{R}_{3D}(t_2, h_2, w_2) \mathbf{k} \rangle = g(\mathbf{q}, \mathbf{k}, \; t_1 - t_2, \; h_1 - h_2, \; w_1 - w_2)$$

这种严密的初等几何旋转不变性，赋予了 DiT 生成长距离刚体平移与摄像机旋转轨道的天然物理先验！

<details>
<summary><b>深入推导：三维正交李代数旋转群 $\text{SO}(3)$ 在相对空间位置内积不变性下的严格几何证明（点击展开查看完整推导）</b></summary>

设旋转生成元为反对称矩阵 $\mathbf{J} \in \mathfrak{so}(3)$。根据欧拉公式，一维旋转矩阵满足正交群性质 $\mathbf{R}(\theta)^\top = \mathbf{R}(-\theta)$ 与同态加法映射 $\mathbf{R}(\theta_1)^\top \mathbf{R}(\theta_2) = \mathbf{R}(\theta_2 - \theta_1)$。
对三维正交直和分解空间 $\mathbb{R}^D = \mathbb{R}^{D_t} \oplus \mathbb{R}^{D_h} \oplus \mathbb{R}^{D_w}$：
$$\langle \mathbf{R}_{3D}(\mathbf{p}_1) \mathbf{q}, \; \mathbf{R}_{3D}(\mathbf{p}_2) \mathbf{k} \rangle = \mathbf{q}^\top \mathbf{R}_{3D}(\mathbf{p}_1)^\top \mathbf{R}_{3D}(\mathbf{p}_2) \mathbf{k} = \mathbf{q}^\top \mathbf{R}_{3D}(\mathbf{p}_2 - \mathbf{p}_1) \mathbf{k}$$
严格证明了自注意力权重仅与相对时空位移矢量 $\Delta \mathbf{p} = \mathbf{p}_2 - \mathbf{p}_1$ 显式相关，建立了欧几里得时空的相对论平移不变性。
</details>

---

## 5.3.4 纯底层 PyTorch 代码实现：从零手写时空 DiT 基础块与 adaLN-Zero 调制引擎

下面我们使用纯底层 PyTorch 算子实现完整的 3D 时空 Patch 展开层、adaLN-Zero 动态调制层与 DiT 核心去噪块。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed3D(nn.Module):
    """
    3D 时空 Patch 展开层: (B, C, T, H, W) -> (B, N, d_model)
    """
    def __init__(self, patch_size: tuple = (2, 2, 2), in_c: int = 4, d_model: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_c, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, d_model, T_p, H_p, W_p)
        x = self.proj(x)
        # 展平为 (B, N, d_model)
        x = x.flatten(2).transpose(1, 2)
        return x

class DiTBlock3D(nn.Module):
    """
    DiT 核心时空自注意力块
    内置 adaLN-Zero 动态条件调制与零初始化门控
    """
    def __init__(self, d_model: int = 64, nhead: int = 4, d_cond: int = 32):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

        # adaLN 调制网络: 从条件向量生成 (gamma1, beta1, alpha1, gamma2, beta2, alpha2) 共 6 组参数
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_cond, 6 * d_model)
        )
        # 零初始化最后一层权重与偏置 (adaLN-Zero 核心)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, N, d_model) 时空词元序列
        :param cond: (B, d_cond) 时间步与条件嵌入
        """
        # 1. 生成调制参数
        mod = self.adaLN_modulation(cond).unsqueeze(1) # (B, 1, 6 * d_model)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)

        # 2. 调制自注意力层 (含 alpha1 零门控)
        norm_x1 = self.norm1(x) * (1.0 + gamma1) + beta1
        attn_out, _ = self.attn(norm_x1, norm_x1, norm_x1)
        x = x + alpha1 * attn_out

        # 3. 调制前馈网络 (含 alpha2 零门控)
        norm_x2 = self.norm2(x) * (1.0 + gamma2) + beta2
        mlp_out = self.mlp(norm_x2)
        x = x + alpha2 * mlp_out

        return x

# ===================================================================
# 单元测试与零初始化恒等映射校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 2
    T_len, H_len, W_len = 4, 8, 8
    d_model = 64
    d_cond = 32

    # 1. 3D Patch 展开测试
    patch_embed = PatchEmbed3D(patch_size=(2, 2, 2), in_c=4, d_model=d_model)
    dummy_latent_video = torch.randn(batch_size, 4, T_len, H_len, W_len)
    tokens = patch_embed(dummy_latent_video)

    expected_tokens = (T_len // 2) * (H_len // 2) * (W_len // 2) # 2 * 4 * 4 = 32
    print(f"[DiT Test] 输入潜在视频形状: {dummy_latent_video.shape}")
    print(f"[DiT Test] 3D Patch 词元序列形状: {tokens.shape} (期望长度: {expected_tokens})")

    # 2. adaLN-Zero 恒等映射测试
    dit_block = DiTBlock3D(d_model=d_model, nhead=4, d_cond=d_cond)
    dummy_cond = torch.randn(batch_size, d_cond)

    out_tokens = dit_block(tokens, dummy_cond)

    # 验证在初始阶段输出是否完全严格等于输入 (恒等映射)
    max_diff = torch.max(torch.abs(out_tokens - tokens)).item()
    print(f"[DiT Test] 初始前向输出与输入的绝对差值最大值: {max_diff:.8f}")

    assert tokens.shape == (batch_size, expected_tokens, d_model), "3D Patch 词元维度不符！"
    assert max_diff < 1e-6, "adaLN-Zero 初始阶段未满足恒等映射条件！"
    print("✓ 3D 时空 Patch 展开、adaLN-Zero 零初始化与 DiT 核心去噪块单测全部通过！")
```

---

## 5.3.5 本节小结

回顾本节内容，我们掌握了现代视频物理模拟器的核心架构：
1. **纯时空注意力 Scaling Law**：将四维视频切分为 3D Patch 序列并输入全注意力 Transformer，突破了卷积感受野的物理限制；
2. **adaLN-Zero 稳定定海神针**：利用零初始化门控将深层百亿参数大模型的初始态化简为恒等映射，攻克了深层扩散训练的发散死穴；
3. **空间相对几何先验**：3D-RoPE 旋转位置编码将欧几里得时空的相对因果直接内化于注意力点积中，为物理世界的高保真模拟奠定了坚实的几何底座。
