# Video-JEPA（V-JEPA）的简洁实现

在深入探究了图像级别的联合嵌入预测架构（I-JEPA）之后，我们自然而然地会将目光投向时间维度。物理世界并不是静止的图像集合，而是连续演化的动态系统。本节，我们将详细探讨并实现 Video-JEPA（V-JEPA）[Bardes et al., 2024]，这是由 Yann LeCun 提出的联合嵌入预测架构（JEPA）在视频领域的重磅延伸。

## 历史脉络与学术背景

在深度学习的早期，视频理解大多依赖于 3D 卷积神经网络（如 C3D）或是结合了时间序列模型（如 LSTM）的 2D 卷积网络。随着 Transformer [Vaswani et al., 2017] 在自然语言处理领域的巨大成功，研究者们迅速将其引入到视觉领域。Vision Transformer（ViT）[Dosovitskiy et al., 2020] 证明了将图像切分为块（Patch）并进行自注意力计算的有效性。

随后，在自监督学习（Self-Supervised Learning, SSL）领域，掩码自编码器（Masked Autoencoders, MAE）[He et al., 2021] 和 VideoMAE [Tong et al., 2022] 展现了通过遮挡部分数据并要求模型在**像素空间**重建它们的强大能力。然而，直接在像素空间进行预测存在一个致命的理论瓶颈：真实世界的视频包含了大量不可预测的高频噪声（例如水面的波纹、树叶的微小摆动）。迫使模型花费巨大的算力去精准重建这些毫无语义价值的随机细节，不仅效率低下，而且会阻碍模型学习到真正的高级语义和物理规律。

正是在这样的学术背景下，V-JEPA 应运而生。它放弃了在像素空间的逐点重建，转而要求模型在**抽象的特征表示空间**中预测缺失的视频片段。这种转变，不仅极大地提升了训练效率，更使其在特征提取上展现出了更强的泛化能力。

## 从静态二维到动态三维：数据的降维解析

为了理解 V-JEPA 的输入机制，我们首先需要将视频这一复杂的多媒体形态，降维拆解为高中生即可理解的数学对象。

在初等几何中，我们知道一个平面可以由二维笛卡尔坐标系 $(x, y)$ 来描述。一幅静态的彩色图像，在忽略颜色通道的意义下，可以看作是一个定义在二维平面上的函数 $f(x, y)$，或者离散化为一个矩阵 $I \in \mathbb{R}^{H \times W}$，其中 $H$ 和 $W$ 分别代表图像的高度和宽度。

视频只不过是在这个二维平面上增加了一个时间维度 $t$。因此，一段持续的视频可以被严格定义为一个三维张量 $V \in \mathbb{R}^{T \times H \times W}$（这里依然暂时忽略 RGB 三个颜色通道 $C$ 以简化理解，实际张量为 $\mathbb{R}^{T \times C \times H \times W}$）。

在 V-JEPA 中，为了让 Transformer 能够处理这个庞大的三维张量，我们不能像早期的图像模型那样逐个像素地输入。我们需要将其“粗粒度化”。

设原始视频的时间帧数为 $T$，高度为 $H$，宽度为 $W$。我们定义空间块的大小为 $p_h \times p_w$，时间块的大小（即连续的帧数）为 $p_t$。
通过这种切分，原本连续的三维时空被划分为一个个独立的“时空立方体”（Spatio-temporal Tubelets）。
在这个三维网格中，沿着时间轴的块数为 $N_t = \frac{T}{p_t}$，沿着高度和宽度的块数分别为 $N_h = \frac{H}{p_h}$ 和 $N_w = \frac{W}{p_w}$。
最终，一个视频将被转化为 $N = N_t \times N_h \times N_w$ 个词元（Tokens），每个词元代表一个时空局部区域的信息。

## 核心机制的数学推导与严格定义

V-JEPA 的核心思想是：给定视频的一个部分上下文（Context），预测该视频中被遮挡部分（Target）在隐空间中的特征表示。

### 1. 目标与上下文的严格划分

设完整的视频经过上述切分并线性映射后，表示为一个词元序列 $X = \{x_1, x_2, \dots, x_N\}$。
在每一轮训练中，我们首先随机采样若干个时空连续的区块作为目标集合（Target blocks），记其索引集合为 $\mathcal{T}$。
随后，我们从剩余的区域中采样一个较大的区块作为上下文集合（Context block），记其索引集合为 $\mathcal{C}$。
显然，这两个集合在时空位置上是不相交的，即 $\mathcal{T} \cap \mathcal{C} = \emptyset$。

### 2. 编码与隐空间映射

我们定义两个神经网络：上下文编码器（Context Encoder）$f_{\theta_c}$ 和目标编码器（Target Encoder）$f_{\theta_t}$。它们通常具有相同的网络结构（例如标准的 Vision Transformer），但参数不同。

我们首先考察一个最简单的标量情形。假设输入只是单一的变量 $x_c$ 和 $x_t$，编码器仅仅是一个标量函数。上下文特征就是 $h_c = f_{\theta_c}(x_c)$，目标特征就是 $h_t = f_{\theta_t}(x_t)$。
顺理成章地，推广到矩阵和序列的形式，我们将上下文序列 $X_{\mathcal{C}}$ 输入上下文编码器，得到隐状态表示：
$$
H_{\mathcal{C}} = f_{\theta_c}(X_{\mathcal{C}}) 
$$

同理，我们将目标序列 $X_{\mathcal{T}}$ 输入目标编码器，得到它在隐空间的目标真实值（Ground Truth）：
$$
H_{\mathcal{T}} = f_{\theta_t}(X_{\mathcal{T}})
$$

### 3. 位置条件预测

预测器（Predictor）$g_{\phi}$ 的任务是，根据上下文的特征 $H_{\mathcal{C}}$，以及我们**想要预测的目标的具体时空位置信息** $P_{\mathcal{T}}$，来预测目标在隐空间的特征。

$$
\hat{H}_{\mathcal{T}} = g_{\phi}(H_{\mathcal{C}}, P_{\mathcal{T}})
$$

> [!quote] 唯一的类比：关于特征预测与参数指数移动平均 (EMA)
> V-JEPA 的架构最反直觉的地方在于：为什么需要两个编码器，并且目标编码器的参数不通过梯度下降更新？
> 我们可以用一个极度克制的博弈学类比来理解。想象上下文编码器（及预测器）是一个“学生”，而目标编码器是一个“导师”。如果导师和学生一起通过梯度下降来纠正错误，系统极易陷入“特征坍塌”（Feature Collapse）——导师为了让学生永远猜对，干脆把所有的视频内容都映射为一个常数 $0$。
> 为了防止这种作弊，V-JEPA 规定导师（目标编码器 $\theta_t$）的权重只能是学生（上下文编码器 $\theta_c$）历史权重的缓慢积累，即通过指数移动平均（EMA）更新：$\theta_t \leftarrow \tau \theta_t + (1 - \tau) \theta_c$。导师的判断标准不随当前这道题的对错而急剧改变，迫使学生必须真正学到如何根据上下文推断缺失的时空信息。

### 4. 目标函数

既然我们在特征空间进行预测，损失函数自然是预测特征 $\hat{H}_{\mathcal{T}}$ 与目标编码器给出的“导师特征” $H_{\mathcal{T}}$ 之间的距离。为了保持严谨性，我们使用归一化后的均方误差（MSE）：

$$
\mathcal{L} = \frac{1}{|\mathcal{T}|} \sum_{i \in \mathcal{T}} \left\| \hat{h}_i - h_i \right\|_2^2
$$

其中 $\hat{h}_i$ 是 $\hat{H}_{\mathcal{T}}$ 中的第 $i$ 个词元， $h_i$ 是 $H_{\mathcal{T}}$ 中的对应词元。通过最小化 $\mathcal{L}$，模型只能更新预测器 $\phi$ 和上下文编码器 $\theta_c$ 的参数。

## 核心网络架构与前向传播实现

在理解了严密的数学推导后，我们将使用 PyTorch 来构建这个系统的简洁版本。为了保持代码的教科书般的清晰，我们将分模块实现。

### 1. 时空分块与嵌入 (Tubelet Embedding)

首先，我们需要将输入的 4D 张量 `(B, C, T, H, W)` 转换为序列 `(B, N, D)`。我们通过 3D 卷积来实现这一时空切块和线性映射。

```{.python .input}
#@tab pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class TubeletEmbedding(nn.Module):
    """
    视频的时空分块与嵌入层。
    通过 3D 卷积将连续的时间和空间像素转化为离散的特征词元。
    """
    def __init__(self, in_channels=3, embed_dim=768, tubelet_size=(2, 16, 16)):
        super().__init__()
        # 使用 3D 卷积进行不重叠的滑动窗口提取，步长等于核大小
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=tubelet_size,
            stride=tubelet_size
        )

    def forward(self, x):
        # [**执行 3D 卷积映射，将像素空间转为特征空间**]
        # 输入维度: (B, C, T, H, W) -> 输出维度: (B, embed_dim, T', H', W')
        x = self.proj(x)
        # 展平空间和时间维度，准备输入 Transformer
        # (B, embed_dim, T', H', W') -> (B, embed_dim, N) -> (B, N, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x
```

### 2. 三维位置编码 (3D Positional Encoding)

在序列化之后，模型丢失了所有的时空结构信息。对于视频，我们不仅要告诉模型“这个块在什么位置”，还要告诉它“这个块在哪个时刻”。绝对三维位置编码是不可或缺的。为了简洁，在此实现中我们将位置编码作为一个可学习的绝对参数，在实际的大规模应用中往往使用正弦余弦分离编码以应对可变长度。

```{.python .input}
#@tab pytorch
def get_3d_sincos_pos_embed(embed_dim, grid_size, t_size):
    """
    这是一个占位函数，说明 3D 位置编码的生成逻辑。
    出于简洁实现的目的，我们在主网络中将使用可学习的绝对位置参数代替。
    """
    pass
```

### 3. V-JEPA 主干网络 (V-JEPA Backbone)

V-JEPA 的主体由上下文编码器（学生）、目标编码器（导师）以及预测器组成。这里我们直接构建完整的架构。为了保证独立性，我们引入基础的 Transformer 块。由于篇幅限制，这里直接使用 PyTorch 自带的 TransformerEncoderLayer。

```{.python .input}
#@tab pytorch
class VJEPAModel(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 num_frames=16, 
                 tubelet_size=2,
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12,
                 predictor_embed_dim=384,
                 predictor_depth=6):
        super().__init__()
        
        # 1. 初始化时空嵌入层
        self.patch_embed = TubeletEmbedding(
            in_channels=3, 
            embed_dim=embed_dim, 
            tubelet_size=(tubelet_size, patch_size, patch_size)
        )
        
        # 计算序列总长度 N = (T / t) * (H / p) * (W / p)
        self.num_patches = (num_frames // tubelet_size) * ((img_size // patch_size) ** 2)
        
        # 2. 声明可学习的 3D 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
        
        # 3. 构建上下文编码器 (学生)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 4. 构建目标编码器 (导师)，其结构与上下文编码器完全一致
        self.target_encoder = copy.deepcopy(self.context_encoder)
        # [**锁定目标编码器的梯度，防止它被优化器直接更新**]
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        # 5. 构建预测器
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, predictor_embed_dim))
        
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=predictor_embed_dim,
            nhead=num_heads // 2,
            dim_feedforward=predictor_embed_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.predictor = nn.TransformerEncoder(predictor_layer, num_layers=predictor_depth)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim)

    def update_target_encoder(self, momentum=0.996):
        """
        [**利用动量机制 (EMA) 缓慢更新目标编码器的参数**]
        这是防止模型发生表征坍塌的核心技巧。
        """
        with torch.no_grad():
            for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(momentum).add_((1.0 - momentum) * param_q.detach().data)

    def forward(self, x, context_mask, target_mask):
        """
        前向传播计算图
        参数:
            x: 原始视频张量 (B, 3, T, H, W)
            context_mask: 上下文部分的布尔掩码 (B, N)
            target_mask: 目标部分的布尔掩码 (B, N)
        """
        B = x.shape[0]
        
        # 1. 时空块嵌入并加上位置编码
        x_embed = self.patch_embed(x) + self.pos_embed
        
        # 2. 目标特征提取 (仅用于产生 Ground Truth，不需要计算梯度)
        with torch.no_grad():
            # 获取完整的目标特征
            target_full_features = self.target_encoder(x_embed)
            # 通过 target_mask 筛选出真正的目标特征
            # 为了简洁，这里假设掩码后每个 batch 的有效序列长度相同，实际实现中需要复杂的 gather 操作
            # 这里仅展示语义层面的张量操作
            target_features = target_full_features[target_mask].view(B, -1, target_full_features.shape[-1])
            
        # 3. 上下文特征提取
        # 仅将未被遮挡的上下文送入编码器，这极大地节省了计算量
        context_x = x_embed[context_mask].view(B, -1, x_embed.shape[-1])
        context_features = self.context_encoder(context_x)
        
        # 4. 预测阶段
        # 降维以减少预测器的计算开销
        context_features = self.predictor_embed(context_features)
        
        # 构造预测器的输入：上下文特征 + 遮挡标志 (Mask Tokens)
        num_targets = target_features.shape[1]
        mask_tokens = self.mask_token.repeat(B, num_targets, 1)
        
        # [**为需要预测的 Mask Token 注入它们本来对应的位置编码**]
        # 这是 Predictor 能够知道“要预测哪里”的唯一途径
        target_pos_embed = self.predictor_pos_embed.repeat(B, 1, 1)[target_mask].view(B, -1, self.predictor_pos_embed.shape[-1])
        mask_tokens = mask_tokens + target_pos_embed
        
        # 拼接上下文与掩码，送入预测器
        predictor_input = torch.cat([context_features, mask_tokens], dim=1)
        predicted_features = self.predictor(predictor_input)
        
        # 提取对应于掩码部分的输出，并映射回原始维度
        predicted_target_features = predicted_features[:, -num_targets:]
        predicted_target_features = self.predictor_proj(predicted_target_features)
        
        # 5. 计算损失 (均方误差)
        loss = F.mse_loss(predicted_target_features, target_features)
        
        return loss
```

### 代码推导的严谨性注记

在上述代码中，有几处为了与纯粹的数学推导对齐而设计的精密巧思值得读者反复推敲：
1. **梯度的阻断**：目标编码器的参数必须强制设为 `requires_grad = False`。模型唯一的学习信号来自于 `F.mse_loss` 反向传播给预测器和上下文编码器的梯度。
2. **位置编码的时机**：注意预测器的输入设计。上下文特征在进入预测器时**没有**再次加上位置编码（它们在最开始已经加过了，特征内部已隐含位置信息），而 `mask_token` **必须**加上它试图重建的时空位置编码。这完美印证了公式 :eqref:`eq_vjepa_predictor` 中 $P_{\mathcal{T}}$ 作为预测条件的核心地位。

## 结语

通过本节的探讨，我们完成了一次从视频的物理维度到抽象语义维度的穿越。V-JEPA 摒弃了执着于重建像素级细节的执念，证明了“预测高维抽象特征”才是通向对时空动态规律深刻理解的正确途径。这种架构的优美之处在于其简洁的数学表达和极为高效的训练过程。
