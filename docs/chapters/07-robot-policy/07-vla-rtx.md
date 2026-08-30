# 视觉-语言-动作模型与RT-X
:label:sec_vla_rtx

在人类探索通用人工智能（Artificial General Intelligence, AGI）的历程中，如果说大型语言模型（Large Language Models, LLMs）赋予了机器“认知”与“推理”的大脑，视觉-语言模型（Vision-Language Models, VLMs）为机器装配了“观察”世界的双眼，那么视觉-语言-动作模型（Vision-Language-Action Models, VLA）则是迈向具身智能（Embodied AI）的关键一步——它赋予了机器在物理世界中“行动”的躯干与四肢。

本节将深入探讨机器人策略（Robot Policy）学习范式的演进，特别是从传统的孤立控制算法到统一的 Transformer 架构的跨越。我们将重点剖析 Robotics Transformer 及其衍生模型（RT-1 `[Brohan et al., 2022]`, RT-2 `[Brohan et al., 2023]`, 以及跨具身的 RT-X `[Padalkar et al., 2023]`），并从最基础的数学概念出发，逐步构建起 VLA 模型的严谨理论框架与代码实现。

## 历史脉络与学术背景

在深度学习全面介入机器人控制之前，机器人策略主要依赖于经典控制理论与状态机（State Machines）。研究者需要利用动力学方程，计算关节的力矩与逆运动学（Inverse Kinematics）。这种方法的局限性在于，它对环境的非结构化变化极其敏感。

随后，基于卷积神经网络（CNNs）的端到端（End-to-End）行为克隆（Behavioral Cloning）开始兴起。然而，早期的模仿学习模型通常只能在单一机器人形态（Embodiment）、单一实验室环境和有限的指令集下工作。当 Transformer 架构在自然语言处理任务中展现出惊人的泛化能力 `[Vaswani et al., 2017]` 后，具身智能领域的研究者开始思考：我们能否将机器人的“感知-决策-行动”循环，抽象为一种类似语言翻译的序列建模问题？

RT-1 首先证明了将图像、语言和动作统一到同一个 Transformer 架构中的可行性；而 RT-2 则进一步回答了另一个深刻的问题：基于互联网海量文本与图像训练的视觉-语言模型（VLM），其蕴含的世界知识能否直接迁移到物理世界的机器人控制中？RT-X 项目（隶属于 Open X-Embodiment）则打破了硬件壁垒，证明了在多种完全不同的机器人硬件上联合训练单一模型，不仅不会导致严重的负迁移（Negative Transfer），反而能产生正向的跨具身（Cross-Embodiment）泛化能力。

## 动作空间的离散化：从连续物理量到语言词表

在高中物理中，我们习惯于将物体的运动描述为连续的实数变量（如坐标 $x, y, z$ 和速度 $v$）。传统的机器人控制也将动作输出建模为连续的连续向量 $\mathbf{a} \in \mathbb{R}^d$，并使用均方误差（MSE）作为回归任务的损失函数。

然而，在 VLA 框架中，我们将连续的物理动作强行“降维”并映射到了离散的词表（Vocabulary）空间中。这样做的根本原因是：**离散化的交叉熵损失（Cross-Entropy Loss）在处理多峰分布（Multi-modal Distribution）时，比连续均方误差表现出更高的稳定性与表达能力**。

(**为了将连续动作转化为可以输入 Transformer 的词元（Tokens），我们需要进行统一的离散化操作。**)

### 标量动作的离散化

假设机器人的某个控制指令（例如机械臂末端执行器在 $X$ 轴的位移）是一个标量 $a \in \mathbb{R}$。在实际物理系统中，执行器的运动能力总是有限的，因此该动作必然被约束在一个物理边界内，即 $a \in [a_{\min}, a_{\max}]$。

为了将其离散化为 $N$ 个独立的区间（Bins），我们首先对 $a$ 进行归一化，使其映射到 $[0, 1]$ 之间：

$$ \tilde{a} = \frac{a - a_{\min}}{a_{\max} - a_{\min}} $$
:eqlabel:eq_action_normalize_scalar

接着，我们将归一化后的值乘以 $(N-1)$，并向下取整，得到该动作所属的离散类别标签 $k$（$k \in \{0, 1, \dots, N-1\}$）：

$$ k = \lfloor \tilde{a} \times (N - 1) \rceil $$
:eqlabel:eq_action_discretize_scalar

在这里，符号 $\lfloor \cdot \rceil$ 表示就近取整运算。通过这种方式，原本连续的物理量 $a$ 就变成了一个可以用独热编码（One-hot Encoding）表示的分类变量。

### 向量化动作的离散化

在实际的机械臂控制中，动作不仅包括三维平移（$X, Y, Z$），还包括三维旋转（如欧拉角或四元数表示的角度变化）以及夹爪（Gripper）的开合程度。假设在时刻 $t$，机器人的完整动作是一个 $D$ 维向量 $\mathbf{a}_t = [a_{t,1}, a_{t,2}, \dots, a_{t,D}]^\top \in \mathbb{R}^D$。

我们可以利用哈达玛积（Hadamard Product，即逐元素乘法）和基本的向量运算，将 :eqref:eq_action_discretize_scalar 严谨地推广到高维向量空间。设 $\mathbf{a}_{\min}$ 和 $\mathbf{a}_{\max}$ 分别为各维度的边界向量，则归一化动作向量 $\tilde{\mathbf{a}}_t$ 可以表示为：

$$ \tilde{\mathbf{a}}_t = (\mathbf{a}_t - \mathbf{a}_{\min}) \oslash (\mathbf{a}_{\max} - \mathbf{a}_{\min}) $$
:eqlabel:eq_action_normalize_vector

其中 $\oslash$ 表示逐元素除法。进而，动作的离散标签向量 $\mathbf{k}_t \in \mathbb{Z}^D$ 为：

$$ \mathbf{k}_t = \lfloor \tilde{\mathbf{a}}_t \odot (N - 1) \rceil $$
:eqlabel:eq_action_discretize_vector

最终，这 $D$ 个离散值将被映射到一个预先定义的动作词表中，就像一句话中的 $D$ 个单词一样，送入 Transformer 进行序列建模。

## VLA 模型的联合概率分布建模

在完成了动作的“词元化”之后，我们就可以使用自回归（Autoregressive）的方式来描述机器人的决策过程。

假设给定了自然语言指令序列 $L = (l_1, l_2, \dots, l_m)$，以及直至当前时刻 $t$ 的历史图像观测序列 $\mathcal{I}_t = (I_1, I_2, \dots, I_t)$。我们的目标是预测当前时刻的动作向量序列 $\mathbf{k}_t = (k_{t,1}, k_{t,2}, \dots, k_{t,D})$。

根据概率论中的链式法则（Chain Rule），联合概率分布可以分解为一系列条件概率的乘积：

$$ P(\mathbf{k}_t \mid \mathcal{I}_t, L; \boldsymbol{\theta}) = \prod_{d=1}^{D} P(k_{t,d} \mid k_{t,<d}, \mathcal{I}_t, L; \boldsymbol{\theta}) $$
:eqlabel:eq_autoregressive_action

其中 $\boldsymbol{\theta}$ 为 VLA 模型的参数。在训练阶段，我们使用极大似然估计（Maximum Likelihood Estimation），即最小化负对数似然（Negative Log-Likelihood）损失函数：

$$ \mathcal{L}(\boldsymbol{\theta}) = - \sum_{t=1}^{T} \sum_{d=1}^{D} \log P(k_{t,d} \mid k_{t,<d}, \mathcal{I}_t, L; \boldsymbol{\theta}) $$
:eqlabel:eq_vla_loss

通过公式 :eqref:eq_vla_loss，我们将复杂的机械臂闭环控制任务，严丝合缝地转换为了标准的自然语言处理中的“下一个词元预测”（Next-Token Prediction）任务。

## RT-1：基于 FiLM 的跨模态融合

在 RT-1 架构中，如何有效地将语言指令 $L$ 的语义信息“注入”到视觉特征提取器中，是模型设计的关键。研究者并没有采用计算复杂度较高的跨注意力（Cross-Attention）机制，而是巧妙地借用了 **FiLM (Feature-wise Linear Modulation)** 机制。

> [!NOTE]
> **唯一一次类比：FiLM 机制的直觉**
> 我们可以将计算机视觉网络（如 ResNet 或 EfficientNet）看作是一个能够提取各种频率和形状特征的“全频段收音机”。自然语言指令（例如“抓取红色的苹果”）就像是一个“调谐器”（Tuner）。FiLM 机制通过语言特征直接生成缩放系数（Scale）和偏置系数（Shift），作用于视觉特征图的各个通道上。这相当于语言指令在“抑制”与红色苹果无关的通道特征，同时“放大”与红色、苹果形状相关的通道特征，从而在特征提取的最早期就完成了注意力的硬性聚焦。

具体而言，给定视觉网络某一层输出的特征图 $\mathbf{F} \in \mathbb{R}^{C \times H \times W}$（其中 $C$ 为通道数），以及语言指令的嵌入向量 $\mathbf{e}_L \in \mathbb{R}^{d_L}$。FiLM 层首先通过两层全连接网络，将 $\mathbf{e}_L$ 映射为两个 $C$ 维的仿射变换向量 $\boldsymbol{\gamma}$ 和 $\boldsymbol{\beta}$：

$$ \boldsymbol{\gamma} = \mathbf{W}_\gamma \mathbf{e}_L + \mathbf{b}_\gamma, \quad \boldsymbol{\beta} = \mathbf{W}_\beta \mathbf{e}_L + \mathbf{b}_\beta $$
:eqlabel:eq_film_params

然后，在每个通道 $c \in \{1, \dots, C\}$ 上，对特征图进行空间一致的仿射调制：

$$ \mathbf{F}'_{c, h, w} = \gamma_c \cdot \mathbf{F}_{c, h, w} + \beta_c $$
:eqlabel:eq_film_modulation

经过 FiLM 调制后，视觉特征图会被展平为一维的词元序列（Token Sequence），并与动作词元序列拼接，送入仅包含解码器的 Transformer（Decoder-only Transformer）中。

## 代码实现：从零构建微型 VLA 模型

为了更深入地理解，我们将用代码实现一个简化的 VLA 架构。由于完整的 RT 模型涉及数十亿参数，我们在此构建一个极简版本，包含：
1. 动作离散化器（Action Tokenizer）
2. 简化的图像-语言特征融合模块
3. 基于 Transformer 的自回归策略解码器

(**首先，我们实现动作空间的离散化与去离散化。**)

```{.python .input}
#@tab pytorch
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

```{.python .input}
#@tab pytorch
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

```{.python .input}
#@tab pytorch
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
            
        # 3. Transformer 处理 (此处简化，未加入因果掩码，仅用于示意特征流动)
        output = self.transformer(transformer_input)
        
        # 4. 预测下一个动作词元的概率分布
        # 取对应于动作位置的输出特征
        action_logits = self.action_head(output)
        return action_logits
```

## RT-X 与开放跨具身数据集

随着模型规模的扩大，单一机器人收集的数据量很快成为了性能瓶颈。RT-X 的核心贡献在于提出了一种**跨具身（Cross-Embodiment）**的数据统一范式。

不同机器人的连杆长度、关节数量和控制频率大相径庭。例如，Franka Emika 拥有 7 个自由度（DoF），而 UR5 仅有 6 个。为了将这些异构数据送入同一个模型中训练，RT-X 定义了一种统一的动作空间表示法。

具体而言，研究者没有选择直接控制底层关节角度，而是统一控制末端执行器（End-Effector）在三维笛卡尔空间中的姿态变化（6 DoF 位姿，包括平移 $\Delta x, \Delta y, \Delta z$ 和旋转 $\Delta \text{roll}, \Delta \text{pitch}, \Delta \text{yaw}$），以及一个额外的标量用于控制夹爪状态。通过反向运动学（IK），模型输出的统一笛卡尔动作可以被转化为针对不同底盘硬件的专属关节控制信号。这一数学变换使得大规模、跨平台数据的协同微调（Co-finetuning）成为可能，极大地推动了通用机器人的发展进程。

## 小结

* 视觉-语言-动作模型（VLA）通过将连续的控制信号离散化，成功将具身智能领域的规划与控制任务转化为自然语言处理中的自回归序列生成问题。
* 动作序列的交叉熵损失，相较于均方误差，能够更好地建模具有多模态特性的动作分布。
* FiLM 机制提供了一种轻量级且有效的跨模态特征调制方法，使得语言指令能够在视觉特征提取的早期介入。
* 跨具身（Cross-Embodiment）学习通过统一坐标系下的逆运动学变换，打破了硬件壁垒，使得在千万级不同机器人轨迹上的联合训练成为可能。

## 练习

1. 在标量动作离散化公式 :eqref:eq_action_normalize_scalar 中，如果系统的物理传感器发生故障，导致输入的观测动作 $a$ 略微超出了预定义的 $[a_{\min}, a_{\max}]$ 范围，在代码实现中应如何处理，以防止程序崩溃或产生无效的分类标签？
   - *提示*：回想我们在代码实现中使用了 `torch.clamp` 操作，思考它在数学定义上的等价表达。
2. 假设我们使用了包含 256 个区间的词表对动作进行离散化。如果我们需要模型进行极高精度的插孔任务，离散化可能会带来什么问题？你能想出一种在不显著增加词表大小的前提下，提高动作精度的数学方案吗？
   - *提示*：思考如何将一个大的数值分解为“高位（粗调）”和“低位（微调）”的组合，类似于浮点数的尾数和指数。
3. 推导公式 :eqref:eq_film_modulation 的反向传播梯度。假设后续网络计算得到了关于 $\mathbf{F}'_{c, h, w}$ 的梯度 $\frac{\partial \mathcal{L}}{\partial \mathbf{F}'_{c, h, w}}$，写出关于 $\boldsymbol{\gamma}_c$ 和原始特征 $\mathbf{F}_{c, h, w}$ 的偏导数表达式。
   - *提示*：应用基础的微积分乘积法则和链式法则。

:begin_tab:pytorch
[讨论](https://discuss.d2l.ai/t/1234)
:end_tab:
