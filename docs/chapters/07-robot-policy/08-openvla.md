# OpenVLA：开源具身大模型

在探讨了RT系列模型（如RT-1与RT-2）之后，我们进入了一个由大语言模型（LLM）主导的具身智能新阶段。RT-2向我们展示了将视觉-语言大模型（Vision-Language Model, VLM）直接用于输出机器人动作的巨大潜力 [[Brohan et al., 2023]](https://arxiv.org/abs/2307.15818)。然而，这类模型通常依赖于闭源的大规模专有架构，其高昂的训练成本和闭门造车的生态限制了整个具身智能社区的发展。为了打破这一壁垒，OpenVLA [[Kim et al., 2024]](https://arxiv.org/abs/2406.09246) 应运而生。作为一个拥有70亿参数的完全开源的视觉-语言-动作（Vision-Language-Action, VLA）模型，OpenVLA不仅在多项机器人操作基准测试中展现出卓越的性能，还为研究者提供了一套基于低秩自适应（LoRA）的高效微调范式。

本节我们将深入解构OpenVLA的核心设计思想。我们将从动作序列的自回归建模出发，严格推导其如何将连续的物理动作映射为语言模型的离散词表，并探讨其如何利用几何降维的思想实现高效的模型微调。

## 视觉-语言-动作建模的自回归表述

大语言模型（如Llama 2或GPT系列）的核心数学形式是**自回归生成**（Autoregressive Generation）。给定一段离散的文本序列，模型通过极大化下一个词的条件概率来进行训练。OpenVLA继承了这一优雅的范式，并将其扩展到了多模态与控制领域。

在具身控制场景中，机器人在离散时间步 $t$ 需要根据当前的视觉观察图像 $x_{\text{img}}$ 和人类提供的文本指令 $x_{\text{text}}$，输出一个多维度的动作向量 $\mathbf{a}_t$。
假设动作向量包含 $D$ 个维度（例如末端执行器的三维坐标、三维旋转姿态以及夹爪的开合程度），即 $\mathbf{a}_t = [a_{t}^{(1)}, a_{t}^{(2)}, \dots, a_{t}^{(D)}]^{\top}$。

我们希望寻找一个策略模型 $\pi$，使得动作序列的条件概率最大化：

$$ P(\mathbf{a}_t \mid x_{\text{img}}, x_{\text{text}}) $$

由于语言模型本质上是处理离散序列的，我们需要将动作向量 $\mathbf{a}_t$ 的各个维度依次展开，并将其视为句子中的“词语”。如果我们假设各个动作维度的生成依赖于前面的维度，概率分布可以根据链式法则严格拆解为：

$$ P(\mathbf{a}_t \mid x_{\text{img}}, x_{\text{text}}) = \prod_{d=1}^{D} P(a_{t}^{(d)} \mid x_{\text{img}}, x_{\text{text}}, a_{t}^{(1)}, \dots, a_{t}^{(d-1)}) $$

这种拆解将一个高维的联合概率密度估计问题，转化为了 $D$ 个一维条件概率分布的序列预测问题。现在，模型只需要在每一步预测动作的某一个维度。这与预测句子的下一个单词在数学形式上达到了完美的统一。

## 连续物理动作的离散化（Tokenization）

由于语言模型的输出空间是预定义好的离散词汇表（Vocabulary），而物理世界中机器人的动作通常是连续的实数（例如关节角度的弧度值或移动距离的米数），我们必须在连续动作与离散词汇之间建立一座严格的数学桥梁。这被称为动作的**离散化**或**词元化**（Action Tokenization）。

### 一维标量的均匀量化

首先，我们考虑一个最简单的场景：假设机器人的夹爪开合度用一个标量 $v$ 表示，其物理范围已知为 $[v_{\min}, v_{\max}]$。我们的目标是将这个连续区间切分为 $B$ 个均匀的离散“桶”（Bins），用整数索引 $k \in \{0, 1, \dots, B-1\}$ 来表示。

第一步，我们需要通过仿射变换（Affine Transformation）将真实的物理量 $v$ 映射到 $[0, 1]$ 的标准区间。我们定义归一化函数：

$$ v_{\text{norm}} = \frac{v - v_{\min}}{v_{\max} - v_{\min}} $$

显然，当 $v = v_{\min}$ 时，$v_{\text{norm}} = 0$；当 $v = v_{\max}$ 时，$v_{\text{norm}} = 1$。

第二步，我们将 $[0, 1]$ 区间放大到离散桶的索引范围 $[0, B-1]$，并通过就近取整操作得到最终的离散类别 $k$：

$$ k = \text{round}(v_{\text{norm}} \times (B-1)) $$

其中，$\text{round}(\cdot)$ 表示四舍五入到最接近的整数。此时，连续的标量 $v$ 就成功转化为了一个离散的整数 $k$。OpenVLA 将这 $B$ 个整数作为特殊的词元（Action Tokens）直接追加到大语言模型的词汇表中。

### 反向映射与量化误差

当语言模型输出一个动作词元索引 $k$ 时，我们需要将其还原为机器人的连续执行指令。这个逆过程（Detokenization）是一个精确的代数求逆步骤，但由于我们之前使用了舍入函数，会不可避免地引入误差。

通过代数变换重组这两个公式，我们可以推导出还原后的连续动作近似值 $\hat{v}$：

$$ \hat{v} = \left( \frac{k}{B-1} \right) (v_{\max} - v_{\min}) + v_{\min} $$

量化引入的最大绝对误差（Quantization Error）由相邻两个桶代表的物理间隔的一半决定：

$$ \epsilon_{\max} = \frac{v_{\max} - v_{\min}}{2(B-1)} $$

这在物理上意味着：只要我们设置的桶数量 $B$ 足够大（OpenVLA 默认设为 256），离散化带来的误差 $\epsilon_{\max}$ 就会极度缩小，从而对最终的物理控制影响微乎其微。

### 矢量化与多维拓展

对于多维动作向量 $\mathbf{a}_t \in \mathbb{R}^D$，我们通常在所有维度上共享相同的动作词汇表空间。但在实际情况中，不同物理维度的数据分布（例如坐标系 X 轴的位移与末端夹爪的开度）可能相差极大。因此，我们需要对每个维度 $d$ 单独统计其在数据集中的极值 $a_{\min}^{(d)}$ 和 $a_{\max}^{(d)}$，进而通过矢量化形式并行完成整个动作向量的离散化。

## OpenVLA 的网络架构与特征融合

OpenVLA 的架构由三个核心组件构成：视觉编码器（Vision Encoder）、视觉-语言投影层（Projector）和大语言模型主干（LLM Backbone）。其本质是将图像映射为语言模型能够理解的“视觉词汇”，进而触发大语言模型的自回归推理。

1. **视觉编码器**：OpenVLA 采用了多尺度特征提取的方法。它将图像切分为 $N$ 个不重叠的图像块（Patches），并通过视觉变换器（如 SigLIP 或 DINOv2）提取特征矩阵 $\mathbf{X}_{\text{vis}} \in \mathbb{R}^{N \times d_{\text{vis}}}$。
2. **投影层**：由于视觉特征的维度 $d_{\text{vis}}$ 与语言模型的词嵌入维度 $d_{\text{llm}}$ 不匹配，我们需要引入一个多层感知机（MLP）将其投影到相同的空间：
   $$ \mathbf{H}_{\text{vis}} = \text{MLP}(\mathbf{X}_{\text{vis}}) \in \mathbb{R}^{N \times d_{\text{llm}}} $$
3. **特征拼接与推理**：将文本指令通过嵌入层转化为矩阵 $\mathbf{H}_{\text{text}} \in \mathbb{R}^{M \times d_{\text{llm}}}$ 后，在序列维度上与视觉特征拼接，形成完整的输入序列 $[\mathbf{H}_{\text{vis}}; \mathbf{H}_{\text{text}}]$。随后，Llama 2 主干网络将在此基础上自回归地生成动作词元。

## 低秩自适应（LoRA）：高效微调的几何视角

拥有 70 亿参数的 OpenVLA 若直接进行全参数微调（Full Fine-Tuning），将对显存和计算资源造成极大的挑战。OpenVLA 选择采用低秩自适应（Low-Rank Adaptation, LoRA）技术 [[Hu et al., 2021]](https://arxiv.org/abs/2106.09685)，使得普通实验室甚至个人研究者也能在特定的机器人任务上对其进行高效微调。

让我们从线性变换的几何视角来严格拆解 LoRA 的原理。在大模型的前馈网络中，核心运算是矩阵乘法。设预训练的权重矩阵为 $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，输入向量为 $\mathbf{x} \in \mathbb{R}^{d_{\text{in}}}$，则线性投影的输出为：

$$ \mathbf{h} = \mathbf{W}_0 \mathbf{x} $$

在微调过程中，我们需要寻找一个更新量 $\Delta \mathbf{W}$，使得新的权重矩阵为 $\mathbf{W} = \mathbf{W}_0 + \Delta \mathbf{W}$。全参数微调需要更新 $\Delta \mathbf{W}$ 中的 $d_{\text{out}} \times d_{\text{in}}$ 个参数，这往往是百万级别的标量。

然而，深度学习研究中的一个重要发现是：预训练大模型在适应特定下游任务（如抓取某个特定物体）时，其权重矩阵的有效更新实际上处于一个非常低维的子空间（Intrinsic Subspace）中。

基于这一洞察，LoRA 强制约束更新矩阵 $\Delta \mathbf{W}$ 的秩（Rank）不超过常数 $r$，且 $r \ll \min(d_{\text{out}}, d_{\text{in}})$。根据线性代数的矩阵分解原理，任何秩为 $r$ 的矩阵均可以分解为两个低秩矩阵的乘积：

$$ \Delta \mathbf{W} = \mathbf{A} \mathbf{B} $$

其中，$\mathbf{B} \in \mathbb{R}^{r \times d_{\text{in}}}$ 将原始高维特征投影到低维子空间，而 $\mathbf{A} \in \mathbb{R}^{d_{\text{out}} \times r}$ 将低维特征重新映射回高维的目标空间。前向传播公式也随之被重写为两条独立的数据流之和：

$$ \mathbf{h} = \mathbf{W}_0 \mathbf{x} + \mathbf{A} \mathbf{B} \mathbf{x} $$

在训练期间，**我们冻结庞大的预训练权重 $\mathbf{W}_0$ 的梯度，仅对小矩阵 $\mathbf{A}$ 和 $\mathbf{B}$ 进行梯度下降**。这种几何降维的思想将需要更新的参数量从 $\mathcal{O}(d_{\text{out}} d_{\text{in}})$ 急剧压缩到了 $\mathcal{O}(r(d_{\text{out}} + d_{\text{in}}))$。在实际部署 OpenVLA 时，这能节省超过 80% 的显存占用。

## 代码实现

(**我们将利用 PyTorch 搭建 OpenVLA 的核心模块**)，包括动作的离散化处理器、视觉-语言投影层，以及简化的条件自回归生成流程。

```{.python .input}
#@tab pytorch
import torch
import torch.nn as nn
from typing import Tuple

class ActionTokenizer:
    def __init__(self, num_bins: int = 256, action_min: float = -1.0, action_max: float = 1.0):
        """
        初始化动作离散化器。
        假设所有动作维度已经被预处理为统一的 [action_min, action_max] 范围内。
        """
        self.num_bins = num_bins
        self.action_min = action_min
        self.action_max = action_max

    def tokenize(self, continuous_actions: torch.Tensor) -> torch.Tensor:
        """
        将连续的动作向量离散化为整数词元索引。
        """
        # 将输入裁剪到合法范围内，防止越界
        actions_clipped = torch.clamp(continuous_actions, self.action_min, self.action_max)
        
        # 应用等式 (7.8.2)：归一化到 [0, 1]
        norm_actions = (actions_clipped - self.action_min) / (self.action_max - self.action_min)
        
        # 应用等式 (7.8.3)：缩放并就近取整
        bin_indices = torch.round(norm_actions * (self.num_bins - 1))
        
        return bin_indices.long()

    def detokenize(self, bin_indices: torch.Tensor) -> torch.Tensor:
        """
        将整数词元索引还原为连续的物理动作。
        """
        # 应用等式 (7.8.4)：反向映射
        continuous_actions = (bin_indices.float() / (self.num_bins - 1)) * \
                             (self.action_max - self.action_min) + self.action_min
        return continuous_actions

class VisionLanguageProjector(nn.Module):
    def __init__(self, vis_dim: int = 1152, llm_dim: int = 4096):
        """
        视觉到语言空间的投影层，即公式 (7.8.6)。
        这里使用一个两层 MLP。
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(vis_dim, llm_dim, bias=False),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim, bias=False)
        )

    def forward(self, x_vis: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_vis)

class SimpleOpenVLA(nn.Module):
    def __init__(self, vocab_size: int, action_bins: int, llm_dim: int = 4096):
        super().__init__()
        # 视觉特征投影层
        self.projector = VisionLanguageProjector(vis_dim=1152, llm_dim=llm_dim)
        
        # 语言与动作词元共享的嵌入层
        # 总词表大小 = 文本词表大小 + 动作桶的数量
        self.total_vocab_size = vocab_size + action_bins
        self.embedding = nn.Embedding(self.total_vocab_size, llm_dim)
        
        # 简化的 LLM 主干网络（此处用标准 Transformer 编码器模拟自回归主干）
        decoder_layer = nn.TransformerEncoderLayer(d_model=llm_dim, nhead=8, batch_first=True)
        self.llm_backbone = nn.TransformerEncoder(decoder_layer, num_layers=4)
        
        # 输出分类头
        self.lm_head = nn.Linear(llm_dim, self.total_vocab_size, bias=False)

    def forward(self, vis_features: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将视觉与文本对齐，预测动作词元。
        vis_features: [batch_size, num_patches, vis_dim]
        text_tokens: [batch_size, seq_len]
        """
        # 1. 投影视觉特征到 LLM 维度
        vis_emb = self.projector(vis_features) # [batch, num_patches, llm_dim]
        
        # 2. 获取文本的词嵌入
        text_emb = self.embedding(text_tokens) # [batch, seq_len, llm_dim]
        
        # 3. 序列拼接：[视觉特征; 文本指令特征]
        combined_emb = torch.cat([vis_emb, text_emb], dim=1)
        
        # 4. 通过大模型的处理
        hidden_states = self.llm_backbone(combined_emb)
        
        # 5. 预测下一个词元的 logits分布
        logits = self.lm_head(hidden_states) # [batch, combined_seq_len, total_vocab_size]
        
        return logits
```

## 小结

- OpenVLA 通过统一的语言建模范式，将机器人动作空间精确地映射为大模型的扩展离散词汇表，实现了视觉-语言-动作的端到端自回归生成。
- 连续物理动作的离散化通过归一化、缩放与舍入操作完成，这引入了与离散区间成正比的固有误差，但通过增大区间数（如256桶）可将误差缩小至物理可容忍的范围。
- 通过引入低秩自适应（LoRA）技术，我们能够将高维的权重更新分解为两个低维矩阵的乘积，极大地降低了微调大参数量模型时的硬件门槛与计算复杂度。
