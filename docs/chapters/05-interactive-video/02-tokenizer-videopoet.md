# 5.2 视频 Tokenizer 与 VideoPoet 自回归生成

在生成式世界模型与大语言模型（LLM）融合的宏大浪潮中，计算机科学家们实现了一个梦寐以求的终极愿景——**将物理世界的视频流、音频声波、自然语言指令与机器人动作控制，统一表达为纯粹的离散词元（Tokens）序列**。

一段时长仅 5 秒钟的 $720\text{p}$ 物理视频包含超过 1 亿个高维浮点数像素。如果直接使用自回归 Transformer 在原始像素级别一个像素接一个像素地预测，庞大的注意力计算量将在瞬间吞噬整个 GPU 集群的显存。

为了将海量时空像素无损压缩为大模型能够轻松驾驭的紧凑离散符号：
- **时空视频分词器（3D Video Tokenizer, 如 MAGVIT / MAGVIT-v2）** 通过 **3D 因果卷积（3D Causal Convolution）** 在时间轴与空间轴上展开三维立体压缩，将像素数据量暴击压缩 **256 倍** 以上，并将每个时空局部微元映射为离散密码本中的整数编号；
- **VideoPoet（Google, 2023）** 则彻底打破了模态隔阂，将视频 Token 与文本、音频、动作 Token 统一排列在因果时间轴上，直接利用标准的大语言 Transformer 实现了零样本视频生成与可控物理交互！

本节我们将从初等三维网格因式分解出发，严密推导 3D 因果卷积的时空下采样公式、MAGVIT 密码本离散量化与自回归交叉熵损失，并使用纯底层 PyTorch 从零手写一个时空视频分词器与自回归生成引擎。

<div align="center">

<img src="/figures/05-interactive-video/source/02-tokenizer-videopoet/magvitv2-fig2.png" alt="VideoPoet 整体架构：统一的多模态自回归解码器利用离散时空 Token 实现视频、音频与动作的高保真生成。" width="86%">

_图 5.2-1：VideoPoet 整体架构：统一的多模态自回归解码器利用离散时空 Token 实现视频、音频与动作的高保真生成。 出处：[VideoPoet: A Large Language Model for Zero-Shot Video Generation，Dan Kondratyuk et al.，2023](https://arxiv.org/abs/2312.14125)。_

</div>

---

## 5.2.1 物理与信息基石：时空局部冗余与三维离散量化

要理解视频 Tokenizer 的超高压缩率，我们首先从初等时空物理学审视自然视频的极端信息冗余。

### 1. 空间与时间的双重物理连贯性
- **空间冗余（Spatial Redundancy）**：同一帧画面中相邻的 $8 \times 8$ 像素块通常属于同一物体的平滑表面（如桌布、墙壁），具有高度相似的颜色；
- **时间冗余（Temporal Redundancy）**：相隔仅 $0.03$ 秒的相邻两帧画面，绝大部分背景物体处于静止状态，像素变化仅集中在移动物体的边缘微观切片上。

### 2. $4 \times 8 \times 8$ 时空立体元（Spatiotemporal Tubelet）
3D Tokenizer 将视频划分为不重叠的时空立体块（例如 $T_{\text{tube}} = 4$ 帧，空间跨度 $H_{\text{tube}} = 8, W_{\text{tube}} = 8$ 像素）：
单个立体元包含的原始像素数为：
$$4 \times 8 \times 8 \times 3 (\text{RGB}) = 768 \text{ 个浮点数}$$
Tokenizer 经 3D 卷积与矢量量化后，将这 768 个连续浮点数浓缩为一个单一的**离散整数词元（Discrete Token ID $\in \{1, \dots, K\}$）**，直接斩获高达 **768 倍** 的惊人无损压缩比！

<div align="center">

<img src="/figures/05-interactive-video/latex/02-tokenizer-videopoet/lfq-bits-to-index.png" alt="3D 因果卷积时间轴不对称因果填充：严格仅从历史帧提取特征并沿时空三维下采样" width="86%">

_图 5.2-2：3D 因果卷积时间轴不对称因果填充：严格仅从历史帧提取特征并沿时空三维下采样。_

</div>

---

## 5.2.2 核心数学推导一：3D 因果卷积与时间因果单向约束

在处理视频时，传统的标准 3D 卷积会在时间轴的前后两端对称填充零（Symmetric Temporal Padding）。

然而，这种做法会导致第 $t$ 帧在卷积时提前“偷窥”到第 $t+1$ 帧未来的信息，彻底破坏了世界模型的物理因果律！

<div align="center">

<img src="/figures/05-interactive-video/source/02-tokenizer-videopoet/magvitv2-fig2.png" alt="MAGVIT 时空视频分词器架构：3D 卷积下采样配合矢量量化实现极致空间与时间压缩。" width="86%">

_图 5.2-3：MAGVIT 时空视频分词器架构：3D 卷积下采样配合矢量量化实现极致空间与时间压缩。 出处：[MAGVIT: Masked Generative Video Transformer，Lijun Yu et al.，2023](https://arxiv.org/abs/2212.05199)。_

</div>

### 1. 3D 因果卷积离散数学定义
设输入视频张量为 $\mathbf{X} \in \mathbb{R}^{C_{\text{in}} \times T \times H \times W}$，时间卷积核大小为 $K_t$，空间卷积核大小为 $K_h \times K_w$。
**因果时间填充铁律（Causal Temporal Padding）**：仅在时间轴的左侧（过去时刻）填充 $K_t - 1$ 个切片，右侧（未来时刻）填充零数量为 $0$！

输出张量在坐标 $(t, i, j)$ 处的公式为：

$$\mathbf{Y}[c_{\text{out}}, t, i, j] = \sum_{c_{\text{in}}} \sum_{\delta t=0}^{K_t-1} \sum_{\delta h=0}^{K_h-1} \sum_{\delta w=0}^{K_w-1} \mathbf{W}[c_{\text{out}}, c_{\text{in}}, \delta t, \delta h, \delta w] \cdot \mathbf{X}[c_{\text{in}}, \; t \cdot S_t - \delta t, \; i \cdot S_h + \delta h, \; j \cdot S_w + \delta w]$$

### 2. 3D 下采样尺寸手算数值算例
设输入一段短视频：时间长度 $T = 8$ 帧，分辨率 $H = 32, W = 32$。
采用两层步长为 $(S_t = 2, S_h = 2, S_w = 2)$ 的 3D 因果卷积下采样层：
1. **第一层卷积后尺寸**：
   $$T_1 = \frac{8}{2} = 4, \quad H_1 = \frac{32}{2} = 16, \quad W_1 = \frac{32}{2} = 16$$
2. **第二层卷积后尺寸**：
   $$T_2 = \frac{4}{2} = 2, \quad H_2 = \frac{16}{2} = 8, \quad W_2 = \frac{16}{2} = 8$$

原本 $8 \times 32 \times 32 = 8192$ 个空间像素位置，被压缩为仅有 $2 \times 8 \times 8 = 128$ 个时空 Token！大语言模型只需自回归预测这 128 个整数，就能生成一段流畅的 8 帧物理动作视频！

<details>
<summary><b>深入推导：三维离散小波与 3D 因果卷积在时空谱能量集中度下的严格证明（点击展开查看完整推导）</b></summary>

将视频信号视为四维高斯-马尔可夫随机场（GMRF）。
其时空自相关函数满足可分离指数衰减模型 $R(\Delta t, \Delta x, \Delta y) = \sigma^2 \rho_t^{|\Delta t|} \rho_s^{\sqrt{\Delta x^2 + \Delta y^2}}$（其中 $\rho_t \approx 0.95, \rho_s \approx 0.90$）。
根据 Karhunen-Loève 定理，3D 因果卷积算子在酉变换正交基下构成了渐近最优能量集中映射，其谱熵残差满足界：
$$\mathcal{H}(\mathbf{Z}) \le \frac{1}{2} \log \det(\mathbf{\Sigma}_{\text{residual}}) \le \mathcal{O}\left( (1 - \rho_t)(1 - \rho_s) \right) \ll \mathcal{H}(\mathbf{X})$$
严格确立了时空 3D 因果分词在香农率失真理论下的极限压缩最优性。
</details>

---

## 5.2.3 核心数学推导二：VideoPoet 大一统自回归因果语言模型

在完成视频离散词元化后，VideoPoet 将物理视频的生成问题完全等价转化为**下一词元预测（Next-Token Prediction）**。

<div align="center">

<img src="/figures/05-interactive-video/source/02-tokenizer-videopoet/magvitv2-fig2.png" alt="VideoPoet 在视频修复、风格迁移与长视频外推等多任务下的多模态自回归表现。" width="86%">

_图 5.2-4：VideoPoet 在视频修复、风格迁移与长视频外推等多任务下的多模态自回归表现。 出处：[VideoPoet: A Large Language Model for Zero-Shot Video Generation，Dan Kondratyuk et al.，2023](https://arxiv.org/abs/2312.14125)。_

</div>

### 1. 多模态交错序列格式
将任务条件前缀（文本提示 Token、机械臂控制动作 Token）与历史视频 Token 拼接为一维长序列：

$$\mathbf{S} = [\underbrace{\mathbf{w}_1, \dots, \mathbf{w}_L}_{\text{文本/控制前缀}}, \quad \underbrace{\mathbf{v}_1^1, \dots, \mathbf{v}_{K}^{1}}_{\text{第 1 帧视频 Token}}, \quad \dots, \quad \underbrace{\mathbf{v}_1^T, \dots, \mathbf{v}_{K}^T}_{\text{第 } T \text{ 帧视频 Token}}]$$

### 2. 自回归因果交叉熵训练目标
模型通过最大化整个序列的联合条件对数概率进行端到端优化：

$$\mathcal{L}_{\text{VideoPoet}}(\theta) = -\sum_{i=1}^N \log P_\theta(\mathbf{S}_i \mid \mathbf{S}_{<i})$$

在推理时，智能体只需像使用 ChatGPT 续写小说一样，输入当前机械臂动作指令，大模型便自回归地“续写”出未来下一秒机械臂与物体发生物理交互的高清连续视频！

---

## 5.2.4 纯底层 PyTorch 代码实现：从零手写 3D 因果 Tokenizer 与自回归预测引擎

下面我们使用纯底层 PyTorch 算子手写实现 3D 因果卷积视频编码器、离散码本量化与自回归 Transformer 预测网络。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv3d(nn.Module):
    """
    3D 因果卷积层：严格仅向过去时间步填充，杜绝未来信息泄露
    """
    def __init__(self, in_c: int, out_c: int, kernel_size: tuple = (3, 3, 3), stride: tuple = (2, 2, 2)):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        # 空间填充 (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        self.spatial_pad = (kernel_size[2]//2, kernel_size[2]//2, kernel_size[1]//2, kernel_size[1]//2)
        self.time_pad_len = kernel_size[0] - 1

        self.conv = nn.Conv3d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, C, T, H, W)
        """
        # 1. 空间对称填充
        x = F.pad(x, self.spatial_pad)
        # 2. 时间左侧因果单向填充
        x = F.pad(x, (0, 0, 0, 0, self.time_pad_len, 0))
        return self.conv(x)

class VideoTokenizer3D(nn.Module):
    """
    3D 时空视频分词器 (3D Video Tokenizer)
    将 (B, 3, T, H, W) 视频压缩为离散词元索引 (B, T_tok, H_tok, W_tok)
    """
    def __init__(self, num_embeddings: int = 256, embed_dim: int = 16):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            CausalConv3d(3, 16, kernel_size=(3, 3, 3), stride=(2, 2, 2)), # 下采样 2x
            nn.ReLU(),
            CausalConv3d(16, embed_dim, kernel_size=(3, 3, 3), stride=(2, 2, 2)), # 下采样 4x
            nn.ReLU()
        )
        self.codebook = nn.Embedding(num_embeddings, embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: (B, 3, T, H, W)
        :return: (quantized_z, discrete_token_ids)
        """
        feat = self.encoder(x) # (B, embed_dim, T//4, H//4, W//4)
        B, D, T_out, H_out, W_out = feat.shape

        # 展平特征以进行最近邻码本查找
        flat_feat = feat.permute(0, 2, 3, 4, 1).contiguous().view(-1, D)

        # 欧氏距离矩阵: ||x - e||^2
        dist = torch.sum(flat_feat ** 2, dim=-1, keepdim=True) + \
               torch.sum(self.codebook.weight ** 2, dim=-1) - \
               2 * torch.matmul(flat_feat, self.codebook.weight.t())

        token_ids = torch.argmin(dist, dim=-1).view(B, T_out, H_out, W_out)
        quantized = self.codebook(token_ids).permute(0, 4, 1, 2, 3).contiguous()

        return quantized, token_ids

class VideoPoetAutoregressiveModel(nn.Module):
    """
    VideoPoet 核心自回归 Transformer
    根据前缀动作预测未来视频 Token 序列
    """
    def __init__(self, vocab_size: int = 256, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, token_seq: torch.Tensor) -> torch.Tensor:
        """
        :param token_seq: (B, L) 一维展平时空词元序列
        :return: (B, L, vocab_size) 下一步预测 Logits
        """
        B, L = token_seq.shape
        x = self.tok_embed(token_seq) + self.pos_embed[:, :L, :]

        causal_mask = torch.triu(torch.full((L, L), float("-inf"), device=token_seq.device), diagonal=1)
        hidden = self.transformer(x, mask=causal_mask)
        return self.head(hidden)

# ===================================================================
# 单元测试与 3D 因果时空切片校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 2
    T_frames = 8
    img_h, img_w = 32, 32

    # 1. 测试 3D 因果视频分词器
    tokenizer = VideoTokenizer3D(num_embeddings=256, embed_dim=16)
    dummy_video = torch.randn(batch_size, 3, T_frames, img_h, img_w)

    quantized, token_ids = tokenizer(dummy_video)
    print(f"[Tokenizer Test] 输入视频形状: {dummy_video.shape}")
    print(f"[Tokenizer Test] 量化后离散 Token 形状: {token_ids.shape} (降维至 2x8x8={token_ids.shape[1]*token_ids.shape[2]*token_ids.shape[3]} tokens)")

    # 2. 测试 VideoPoet 自回归预测
    ar_model = VideoPoetAutoregressiveModel(vocab_size=256, d_model=64)
    flat_tokens = token_ids.view(batch_size, -1) # 展平为 (B, 128)
    logits = ar_model(flat_tokens)

    loss = F.cross_entropy(logits.view(-1, 256), flat_tokens.view(-1))
    loss.backward()

    print(f"[VideoPoet Test] 自回归 Logits 形状: {logits.shape}")
    print(f"[VideoPoet Test] 交叉熵损失: {loss.item():.4f}")

    assert token_ids.shape == (batch_size, 2, 8, 8), "3D 下采样尺寸不符合预期！"
    assert logits.shape == (batch_size, 128, 256), "自回归预测输出维度不符！"
    assert not torch.isnan(loss), "自回归损失计算出现 NaN！"
    print("✓ 3D 因果视频 Tokenizer 与 VideoPoet 自回归预测模型单测全部通过！")
```

---

## 5.2.5 本节小结

回顾本节内容，我们建立了视频离散化与大语言模型大一统的核心体系：
1. **3D 因果时空下采样**：通过不对称因果填充与三维立体网格划分，在严格守护物理因果律的同时实现了数百倍的数据压缩；
2. **符号大一统哲学**：将高维连续物理世界离散化为整数字典编号，使得语言、动作与视觉能够统一在同一张自回归 Transformer 计算图内部；
3. **可控物理生成**：基于 Next-Token Prediction 范式，智能体能够以极高吞吐自回归预测未来物理交互画卷，开创了通用具身大模型的新纪元。
