# 视觉-语言-动作模型与RT-X

给机器人一张当前画面和一句“把可乐罐放进抽屉”，策略必须把语言里的目标、图像里的物体位置和机器人的动作空间联系起来。视觉—语言—动作模型（Vision-Language-Action Model, VLA）用统一模型完成这种条件动作预测；Open X-Embodiment/RT-X 进一步把多机构、不同形态的机器人数据放进联合训练框架。

<div align="center">

<img src="/figures/07-robot-policy/source/07-vla-rtx/rtx-fig1.png" alt="Open X-Embodiment 汇集多机构机器人与真实任务，展示 RT-X 的跨具身对象。" width="86%">

_图 7.7-1：Open X-Embodiment 汇集多机构机器人与真实任务，展示 RT-X 的跨具身对象。 出处：[Open X-Embodiment: Robotic Learning Datasets and RT-X Models，Open X-Embodiment Collaboration，2023](https://arxiv.org/abs/2310.08864)。_

</div>

本节以 RT-1 [[Brohan et al., 2022]](https://arxiv.org/abs/2212.06817)、RT-2 [[Brohan et al., 2023]](https://arxiv.org/abs/2307.15818) 和 Open X-Embodiment/RT-X [[Open X-Embodiment Collaboration, 2023]](https://arxiv.org/abs/2310.08864) 为主线，说明动作离散化、条件序列建模和跨机器人数据统一。不同 VLA 的动作表示并不完全相同，下面的离散词元方案只对应采用这一路线的模型。

## 历史脉络与学术背景

经典机器人系统常把感知、任务规划、运动规划和底层控制拆成不同模块。这样的结构便于验证和施加约束，但感知模型或规则没有覆盖的环境变化，仍可能让整条链路失效。

随后，基于卷积神经网络（CNN）的端到端行为克隆开始直接从观测预测动作。许多早期系统局限在单一机器人、实验室环境和有限指令集。Transformer 提供了统一处理长序列与多种词元的工具 [[Vaswani et al., 2017]](https://arxiv.org/abs/1706.03762)，机器人研究由此尝试把感知、指令和动作放进同一序列模型。

RT-1 展示了用 Transformer 统一处理图像、语言和离散动作的机器人策略。RT-2 把预训练视觉—语言模型与机器人动作数据共同微调，研究网络知识能否迁移到控制任务。Open X-Embodiment 汇集了 22 种机器人、超过一百万条轨迹；RT-X 实验表明，多机器人联合训练在论文评测设置中可以带来正迁移，但这不保证任意机器人组合都不会出现负迁移。

## 动作空间的离散化：从连续物理量到语言词表

在高中物理中，我们习惯于将物体的运动描述为连续的实数变量（如坐标 $x, y, z$ 和速度 $v$）。传统的机器人控制也将动作输出建模为连续的连续向量 $\mathbf{a} \in \mathbb{R}^d$，并使用均方误差（MSE）作为回归任务的损失函数。

<div align="center">

<img src="/figures/07-robot-policy/source/07-vla-rtx/rt2-fig2.png" alt="RT-2 的真实泛化案例说明语言知识怎样通过动作 token 落到机器人行为。" width="86%">

_图 7.7-2：RT-2 的真实泛化案例说明语言知识怎样通过动作 token 落到机器人行为。 出处：[RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control，Anthony Brohan et al.，2023](https://arxiv.org/abs/2307.15818)。_

</div>

RT-1、RT-2 等模型把连续物理动作量化为离散词元，从而复用分类式的序列预测目标。离散分布可以为多个动作区间分配概率，但是否优于连续回归取决于数据、量化精度和模型设计，不能仅由损失函数形式决定。

(**为了将连续动作转化为可以输入 Transformer 的词元（Tokens），我们需要进行统一的离散化操作。**)

### 标量动作的离散化

假设机器人的某个控制指令（例如机械臂末端执行器在 $X$ 轴的位移）是一个标量 $a \in \mathbb{R}$。在实际物理系统中，执行器的运动能力总是有限的，因此该动作必然被约束在一个物理边界内，即 $a \in [a_{\min}, a_{\max}]$。

为了将其离散化为 $N$ 个独立的区间（Bins），我们首先对 $a$ 进行归一化，使其映射到 $[0, 1]$ 之间：

$$ \tilde{a} = \frac{a - a_{\min}}{a_{\max} - a_{\min}} $$

接着，我们将归一化后的值乘以 $(N-1)$，并向下取整，得到该动作所属的离散类别标签 $k$（$k \in \{0, 1, \dots, N-1\}$）：

$$ k = \operatorname{round}\!\left(\tilde{a}(N-1)\right) $$

这里还要先把越界值裁剪到 $[0,1]$。量化后的整数 $k$ 可以作为动作词元的类别标签。

### 向量化动作的离散化

在实际的机械臂控制中，动作不仅包括三维平移（$X, Y, Z$），还包括三维旋转（如欧拉角或四元数表示的角度变化）以及夹爪（Gripper）的开合程度。假设在时刻 $t$，机器人的完整动作是一个 $D$ 维向量 $\mathbf{a}_t = [a_{t,1}, a_{t,2}, \dots, a_{t,D}]^\top \in \mathbb{R}^D$。

我们可以利用哈达玛积（Hadamard Product，即逐元素乘法）和基本的向量运算，将该公式严谨地推广到高维向量空间。设 $\mathbf{a}_{\min}$ 和 $\mathbf{a}_{\max}$ 分别为各维度的边界向量，则归一化动作向量 $\tilde{\mathbf{a}}_t$ 可以表示为：

$$ \tilde{\mathbf{a}}_t = (\mathbf{a}_t - \mathbf{a}_{\min}) \oslash (\mathbf{a}_{\max} - \mathbf{a}_{\min}) $$

其中 $\oslash$ 表示逐元素除法。进而，动作的离散标签向量 $\mathbf{k}_t \in \mathbb{Z}^D$ 为：

$$ \mathbf{k}_t = \operatorname{round}\!\left(\tilde{\mathbf{a}}_t \odot (N-1)\right) $$

最终，这 $D$ 个离散值将被映射到一个预先定义的动作词表中，就像一句话中的 $D$ 个单词一样，送入 Transformer 进行序列建模。

## VLA 模型的联合概率分布建模

在完成了动作的“词元化”之后，我们就可以使用自回归（Autoregressive）的方式来描述机器人的决策过程。

假设给定了自然语言指令序列 $L = (l_1, l_2, \dots, l_m)$，以及直至当前时刻 $t$ 的历史图像观测序列 $\mathcal{I}_t = (I_1, I_2, \dots, I_t)$。我们的目标是预测当前时刻的动作向量序列 $\mathbf{k}_t = (k_{t,1}, k_{t,2}, \dots, k_{t,D})$。

根据概率论中的链式法则（Chain Rule），联合概率分布可以分解为一系列条件概率的乘积：

$$ P(\mathbf{k}_t \mid \mathcal{I}_t, L; \boldsymbol{\theta}) = \prod_{d=1}^{D} P(k_{t,d} \mid k_{t,<d}, \mathcal{I}_t, L; \boldsymbol{\theta}) $$

其中 $\boldsymbol{\theta}$ 为 VLA 模型的参数。在训练阶段，我们使用极大似然估计（Maximum Likelihood Estimation），即最小化负对数似然（Negative Log-Likelihood）损失函数：

$$ \mathcal{L}(\boldsymbol{\theta}) = - \sum_{t=1}^{T} \sum_{d=1}^{D} \log P(k_{t,d} \mid k_{t,<d}, \mathcal{I}_t, L; \boldsymbol{\theta}) $$

这个分解把一次动作预测写成“下一个词元”目标。闭环控制还包含观测更新、动作反量化、控制频率与安全约束，不能只由该概率分解概括。

## RT-1：基于 FiLM 的跨模态融合

<div align="center">

<img src="/figures/07-robot-policy/source/07-vla-rtx/rt1-fig5.png" alt="RT-1 的多条真实执行轨迹将语言指令、视觉输入与离散动作闭环相连。" width="86%">

_图 7.7-3：RT-1 的多条真实执行轨迹将语言指令、视觉输入与离散动作闭环相连。 出处：[RT-1: Robotics Transformer for Real-World Control at Scale，Anthony Brohan et al.，2022](https://arxiv.org/abs/2212.06817)。_

</div>

在 RT-1 架构中，如何有效地将语言指令 $L$ 的语义信息“注入”到视觉特征提取器中，是模型设计的关键。研究者并没有采用计算复杂度较高的跨注意力（Cross-Attention）机制，而是巧妙地借用了 **FiLM (Feature-wise Linear Modulation)** 机制。

FiLM 的直觉很直接：语言嵌入为每个视觉通道产生一个缩放量和偏置量，因此同一幅图在不同指令下会得到不同的条件特征。它是一种软调制，并不保证模型只保留与指令相关的区域。

具体而言，给定视觉网络某一层输出的特征图 $\mathbf{F} \in \mathbb{R}^{C \times H \times W}$（其中 $C$ 为通道数），以及语言指令的嵌入向量 $\mathbf{e}_L \in \mathbb{R}^{d_L}$。FiLM 层首先通过两层全连接网络，将 $\mathbf{e}_L$ 映射为两个 $C$ 维的仿射变换向量 $\boldsymbol{\gamma}$ 和 $\boldsymbol{\beta}$：

$$ \boldsymbol{\gamma} = \mathbf{W}_\gamma \mathbf{e}_L + \mathbf{b}_\gamma, \quad \boldsymbol{\beta} = \mathbf{W}_\beta \mathbf{e}_L + \mathbf{b}_\beta $$

然后，在每个通道 $c \in \{1, \dots, C\}$ 上，对特征图进行空间一致的仿射调制：

$$ \mathbf{F}'_{c, h, w} = \gamma_c \cdot \mathbf{F}_{c, h, w} + \beta_c
$$

<div align="center">

<img src="/figures/07-robot-policy/latex/07-vla-rtx/film-channel-broadcast.png" alt="语言生成的通道缩放和平移参数广播到全部空间位置" width="86%">

_图 7.7-4：语言嵌入生成每个通道的 γ 与 β；固定通道 c 后，同一对标量会广播到全部 H×W 位置。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

经过 FiLM 调制后，视觉特征图会被展平为一维的词元序列（Token Sequence），并与动作词元序列拼接，送入仅包含解码器的 Transformer（Decoder-only Transformer）中。

## 代码实现：从零构建微型 VLA 模型

为了更深入地理解，我们将用代码实现一个简化的 VLA 架构。由于完整的 RT 模型涉及数十亿参数，我们在此构建一个极简版本，包含：

1. 动作离散化器（Action Tokenizer）
2. 简化的图像-语言特征融合模块
3. 基于 Transformer 的自回归策略解码器

(**首先，我们实现动作空间的离散化与去离散化。**)

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

class ActionTokenizer:
    def __init__(self, action_min, action_max, vocab_size=256):
        """
        动作离散化器
        action_min: List[float] 或 Tensor，动作各维度的下界
        action_max: List[float] 或 Tensor，动作各维度的上界
        vocab_size: int, 离散化的区间数量
        """
        self.action_min = torch.tensor(action_min, dtype=torch.float32)
        self.action_max = torch.tensor(action_max, dtype=torch.float32)
        self.vocab_size = vocab_size

    def tokenize(self, action: torch.Tensor) -> torch.Tensor:
        """
        将连续动作转换为离散词元
        action: [batch_size, action_dim]
        """
        # 确保 action 和 min/max 在同一设备上
        self.action_min = self.action_min.to(action.device)
        self.action_max = self.action_max.to(action.device)

        # 归一化到 [0, 1] 之间
        normalized_action = (action - self.action_min) / (self.action_max - self.action_min)
        # 裁剪以防越界
        normalized_action = torch.clamp(normalized_action, 0.0, 1.0)

        # 离散化为整数标签
        tokenized = torch.round(normalized_action * (self.vocab_size - 1)).long()
        return tokenized

    def detokenize(self, tokenized: torch.Tensor) -> torch.Tensor:
        """
        将离散词元还原为连续动作向量
        """
        self.action_min = self.action_min.to(tokenized.device)
        self.action_max = self.action_max.to(tokenized.device)

        normalized_action = tokenized.float() / (self.vocab_size - 1)
        action = normalized_action * (self.action_max - self.action_min) + self.action_min
        return action
```

(**接下来，我们实现极为关键的 FiLM 层。**)

```python
class FiLMLayer(nn.Module):
    def __init__(self, lang_dim, channels):
        super(FiLMLayer, self).__init__()
        # 语言特征映射到 γ 和 β
        self.fc_gamma = nn.Linear(lang_dim, channels)
        self.fc_beta = nn.Linear(lang_dim, channels)

    def forward(self, x, lang_emb):
        """
        x: 视觉特征图 [batch_size, channels, H, W]
        lang_emb: 语言嵌入向量 [batch_size, lang_dim]
        """
        # 计算缩放和偏置系数，形状变为 [batch_size, channels, 1, 1]
        gamma = self.fc_gamma(lang_emb).unsqueeze(-1).unsqueeze(-1)
        beta = self.fc_beta(lang_emb).unsqueeze(-1).unsqueeze(-1)

        # 仿射调制
        return gamma * x + beta
```

(**最后，我们将各部分组装成一个极简的 VLA Transformer。**)

```python
class TinyVLAModel(nn.Module):
    def __init__(self, action_dim, lang_dim=128, img_channels=64, vocab_size=256, d_model=256, n_heads=4, num_layers=4):
        super(TinyVLAModel, self).__init__()
        self.action_dim = action_dim
        self.vocab_size = vocab_size

        # 视觉前端（使用卷积层模拟）
        self.vision_conv = nn.Conv2d(3, img_channels, kernel_size=3, stride=2, padding=1)
        self.film = FiLMLayer(lang_dim, img_channels)
        self.vision_proj = nn.Linear(img_channels, d_model)

        # 动作嵌入
        self.action_emb = nn.Embedding(vocab_size, d_model)

        # Transformer 解码器
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # 动作预测头
        self.action_head = nn.Linear(d_model, vocab_size)

    def forward(self, image, lang_emb, action_tokens=None):
        """
        image: [batch_size, 3, H, W]
        lang_emb: [batch_size, lang_dim]
        action_tokens: [batch_size, action_len] (训练时使用真实标签，推理时自回归生成)
        """
        batch_size = image.size(0)

        # 1. 视觉-语言特征融合
        v_feat = F.relu(self.vision_conv(image)) # [B, C, H', W']
        v_feat = self.film(v_feat, lang_emb)

        # 展平为序列 [B, H'*W', C]
        v_seq = v_feat.flatten(2).transpose(1, 2)
        v_seq = self.vision_proj(v_seq) # [B, seq_len, d_model]

        # 2. 拼接序列
        if action_tokens is not None:
            a_seq = self.action_emb(action_tokens) # [B, action_len, d_model]
            # 输入序列: [视觉词元序列, 动作词元序列]
            transformer_input = torch.cat([v_seq, a_seq], dim=1)
        else:
            transformer_input = v_seq

        # 3. 因果掩码防止动作位置读取未来动作标签
        seq_len = transformer_input.size(1)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=image.device),
            diagonal=1,
        )
        output = self.transformer(transformer_input, mask=causal_mask)

        # 4. 预测下一个动作词元的概率分布
        # 取对应于动作位置的输出特征
        action_logits = self.action_head(output)
        return action_logits
```

## RT-X 与开放跨具身数据集

随着模型规模的扩大，单一机器人收集的数据量很快成为了性能瓶颈。RT-X 的核心贡献在于提出了一种**跨具身（Cross-Embodiment）**的数据统一范式。

<div align="center">

<img src="/figures/07-robot-policy/source/07-vla-rtx/rtx-fig3.png" alt="RT-1-X 与 RT-2-X 在统一跨具身数据上分别延续机器人 Transformer 与 VLA 路线。" width="86%">

_图 7.7-5：RT-1-X 与 RT-2-X 在统一跨具身数据上分别延续机器人 Transformer 与 VLA 路线。 出处：[Open X-Embodiment: Robotic Learning Datasets and RT-X Models，Open X-Embodiment Collaboration，2023](https://arxiv.org/abs/2310.08864)。_

</div>

不同机器人的连杆长度、关节数量和控制频率不同。Open X-Embodiment 为数据交换定义了统一格式，并在训练 RT-X 时尽量把动作转成夹爪坐标系中的七维表示：三维平移、三维旋转和夹爪动作；缺失的维度可以用零填充。

这七维量在不同数据源中可能表示位置变化、速度或夹爪状态，论文的数据转换规则会尽量对齐语义。底层控制器如何把末端动作变成关节命令由具体机器人决定，并不是数据集统一层自动提供的一套通用逆运动学。统一格式降低了联合训练门槛，却没有消除动力学、频率和相机视角之间的差异。

## 小结

- 一部分 **VLA** 将连续控制量离散化，再用自回归序列模型预测动作词元；这不是所有 VLA 的统一定义。
- 离散交叉熵能够表示多个动作区间的概率，但量化也会引入误差。
- **FiLM 机制**提供了一种轻量级且有效的跨模态特征调制方法，使得语言指令能够在视觉特征提取的早期介入。
- **跨具身学习**依赖数据格式与动作语义的对齐。Open X-Embodiment 汇集的是百万级轨迹，硬件差异仍需在模型、数据和底层控制中处理。

$$
$$
