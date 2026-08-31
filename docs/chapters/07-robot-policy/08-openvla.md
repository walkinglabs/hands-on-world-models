# OpenVLA：开源具身大模型

<div align="center">

<img src="/figures/07-robot-policy/source/08-openvla/openvla-fig1.png" alt="OpenVLA 在多种真实机器人与任务上执行语言条件控制，呈现开源模型的实际对象边界。" width="86%">

_图 7.8-1：OpenVLA 在多种真实机器人与任务上执行语言条件控制，呈现开源模型的实际对象边界。 出处：[OpenVLA: An Open-Source Vision-Language-Action Model，Moo Jin Kim et al.，2024](https://arxiv.org/abs/2406.09246)。_

</div>

OpenVLA 接收一张机器人视角图像和一条语言指令，例如“拿起蓝色杯子”，随后生成七个动作维度对应的离散词元，再还原成连续控制量。它是一个 70 亿参数的开源视觉—语言—动作模型，基于 Open X-Embodiment 的 97 万条机器人轨迹训练 [[Kim et al., 2024]](https://arxiv.org/abs/2406.09246)。开放权重、代码和微调流程，让研究者可以在自己的机器人数据上复现实验并检查模型边界。

本节从自回归动作建模出发，说明 OpenVLA 如何量化连续动作、融合两种视觉特征，并用 LoRA 降低任务适配的训练参数量。

## 视觉-语言-动作建模的自回归表述

大语言模型（如 Llama 2）的核心训练形式是**自回归生成**。给定一段离散序列，模型学习下一个词元的条件概率。OpenVLA 沿用这一形式，并把输出空间扩展到机器人动作词元。

在具身控制场景中，机器人在离散时间步 $t$ 需要根据当前的视觉观察图像 $x_{\text{img}}$ 和人类提供的文本指令 $x_{\text{text}}$，输出一个多维度的动作向量 $\mathbf{a}_t$。
假设动作向量包含 $D$ 个维度（例如末端执行器的三维坐标、三维旋转姿态以及夹爪的开合程度），即 $\mathbf{a}_t = [a_{t}^{(1)}, a_{t}^{(2)}, \dots, a_{t}^{(D)}]^{\top}$。

我们希望寻找一个策略模型 $\pi$，使得动作序列的条件概率最大化：

$$ P(\mathbf{a}_t \mid x_{\text{img}}, x_{\text{text}}) $$

由于语言模型本质上是处理离散序列的，我们需要将动作向量 $\mathbf{a}_t$ 的各个维度依次展开，并将其视为句子中的“词语”。如果我们假设各个动作维度的生成依赖于前面的维度，概率分布可以根据链式法则严格拆解为：

$$ P(\mathbf{a}_t \mid x_{\text{img}}, x_{\text{text}}) = \prod_{d=1}^{D} P(a_{t}^{(d)} \mid x_{\text{img}}, x_{\text{text}}, a_{t}^{(1)}, \dots, a_{t}^{(d-1)}) $$

这种拆解将高维联合分布写成 $D$ 个条件分布的序列预测问题。它与语言模型的下一个词元预测采用相同的概率分解，但动作词元还需要满足物理范围、控制频率和安全约束。

## 连续物理动作的离散化（Tokenization）

由于语言模型的输出空间是预定义好的离散词汇表（Vocabulary），而物理世界中机器人的动作通常是连续的实数（例如关节角度的弧度值或移动距离的米数），我们必须在连续动作与离散词汇之间建立一座严格的数学桥梁。这被称为动作的**离散化**或**词元化**（Action Tokenization）。

### 一维标量的均匀量化

先考虑夹爪开合度标量 $v$。OpenVLA 按数据集和动作维度统计第 1、99 百分位数，并把它们作为鲁棒边界 $[v_{\min}, v_{\max}]$；超出边界的少量值会被裁剪。随后把区间切分为 $B$ 个均匀桶，用整数 $k \in \{0,1,\dots,B-1\}$ 表示。

第一步，我们需要通过仿射变换（Affine Transformation）将真实的物理量 $v$ 映射到 $[0, 1]$ 的标准区间。我们定义归一化函数：

$$ v_{\text{norm}} = \frac{v - v_{\min}}{v_{\max} - v_{\min}} $$

显然，当 $v = v_{\min}$ 时，$v_{\text{norm}} = 0$；当 $v = v_{\max}$ 时，$v_{\text{norm}} = 1$。

第二步，我们将 $[0, 1]$ 区间放大到离散桶的索引范围 $[0, B-1]$，并通过就近取整操作得到最终的离散类别 $k$：

$$ k = \text{round}(v_{\text{norm}} \times (B-1)) $$

其中，$\text{round}(\cdot)$ 表示就近取整。OpenVLA 使用 256 个桶，并复用 Llama 2 词表中使用频率最低的 256 个词元作为动作词元，而不是简单扩充一个全新的词表区间。

### 反向映射与量化误差

当语言模型输出一个动作词元索引 $k$ 时，我们需要将其还原为机器人的连续执行指令。这个逆过程（Detokenization）是一个精确的代数求逆步骤，但由于我们之前使用了舍入函数，会不可避免地引入误差。

通过代数变换重组这两个公式，我们可以推导出还原后的连续动作近似值 $\hat{v}$：

$$ \hat{v} = \left( \frac{k}{B-1} \right) (v_{\max} - v_{\min}) + v_{\min} $$

量化引入的最大绝对误差（Quantization Error）由相邻两个桶代表的物理间隔的一半决定：

$$ \epsilon_{\max} = \frac{v_{\max} - v_{\min}}{2(B-1)} $$

增加桶数会减小单维量化误差上界，但同时提高分类分辨率。量化误差是否能被具体机器人容忍，还取决于动作尺度、控制频率和底层控制器。

### 矢量化与多维拓展

对于多维动作向量 $\mathbf{a}_t \in \mathbb{R}^D$，不同维度仍可共享同一组动作词元，但归一化统计量不能混用。OpenVLA 对每个维度 $d$ 分别统计第 1、99 百分位数 $q_{01}^{(d)}$ 和 $q_{99}^{(d)}$，再并行完成整个动作向量的归一化与离散化。

## OpenVLA 的网络架构与特征融合

<div align="center">

<img src="/figures/07-robot-policy/source/08-openvla/openvla-fig2.png" alt="OpenVLA 将 DINOv2、SigLIP、Llama 2 与动作解码器串联为端到端 VLA。" width="86%">

_图 7.8-2：OpenVLA 将 DINOv2、SigLIP、Llama 2 与动作解码器串联为端到端 VLA。 出处：[OpenVLA: An Open-Source Vision-Language-Action Model，Moo Jin Kim et al.，2024](https://arxiv.org/abs/2406.09246)。_

</div>

OpenVLA 的架构由三个核心组件构成：视觉编码器（Vision Encoder）、视觉-语言投影层（Projector）和大语言模型主干（LLM Backbone）。其本质是将图像映射为语言模型能够理解的“视觉词汇”，进而触发大语言模型的自回归推理。

1. **视觉编码器**：OpenVLA 同时使用 SigLIP 与 DINOv2。两路特征分别提供语言对齐和细粒度视觉结构线索，拼接后形成视觉特征矩阵 $\mathbf{X}_{\text{vis}} \in \mathbb{R}^{N \times d_{\text{vis}}}$。
2. **投影层**：由于视觉特征的维度 $d_{\text{vis}}$ 与语言模型的词嵌入维度 $d_{\text{llm}}$ 不匹配，我们需要引入一个多层感知机（MLP）将其投影到相同的空间：
   $$ \mathbf{H}_{\text{vis}} = \text{MLP}(\mathbf{X}_{\text{vis}}) \in \mathbb{R}^{N \times d_{\text{llm}}} $$
3. **特征拼接与推理**：将文本指令通过嵌入层转化为矩阵 $\mathbf{H}_{\text{text}} \in \mathbb{R}^{M \times d_{\text{llm}}}$ 后，在序列维度上与视觉特征拼接，形成完整的输入序列 $[\mathbf{H}_{\text{vis}}; \mathbf{H}_{\text{text}}]$。随后，Llama 2 主干网络将在此基础上自回归地生成动作词元。

## 低秩自适应（LoRA）：高效微调的几何视角

<div align="center">

<img src="/figures/07-robot-policy/source/08-openvla/lora-fig1.png" alt="LoRA 用低秩矩阵分解参数更新，在冻结主权重时实现高效适配。" width="86%">

_图 7.8-3：LoRA 用低秩矩阵分解参数更新，在冻结主权重时实现高效适配。 出处：[LoRA: Low-Rank Adaptation of Large Language Models，Edward J. Hu et al.，2021](https://arxiv.org/abs/2106.09685)。_

</div>

拥有 70 亿参数的 OpenVLA 若直接进行全参数微调（Full Fine-Tuning），将对显存和计算资源造成极大的挑战。OpenVLA 选择采用低秩自适应（Low-Rank Adaptation, LoRA）技术 [[Hu et al., 2021]](https://arxiv.org/abs/2106.09685)，使得普通实验室甚至个人研究者也能在特定的机器人任务上对其进行高效微调。

<div align="center">

<img src="/figures/07-robot-policy/source/08-openvla/openvla-fig5.png" alt="新机器人平台上的适配任务比较全量与参数高效微调的实际效果。" width="86%">

_图 7.8-4：新机器人平台上的适配任务比较全量与参数高效微调的实际效果。 出处：[OpenVLA: An Open-Source Vision-Language-Action Model，Moo Jin Kim et al.，2024](https://arxiv.org/abs/2406.09246)。_

</div>

让我们从线性变换的几何视角来严格拆解 LoRA 的原理。在大模型的前馈网络中，核心运算是矩阵乘法。设预训练的权重矩阵为 $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，输入向量为 $\mathbf{x} \in \mathbb{R}^{d_{\text{in}}}$，则线性投影的输出为：

$$ \mathbf{h} = \mathbf{W}_0 \mathbf{x} $$

在微调过程中，我们需要寻找一个更新量 $\Delta \mathbf{W}$，使得新的权重矩阵为 $\mathbf{W} = \mathbf{W}_0 + \Delta \mathbf{W}$。全参数微调需要更新 $\Delta \mathbf{W}$ 中的 $d_{\text{out}} \times d_{\text{in}}$ 个参数，这往往是百万级别的标量。

LoRA 假设任务适配所需的权重更新可以用低秩矩阵近似。这个假设在许多任务上有效，但秩 $r$ 是否足够仍需由验证结果决定。

基于这一洞察，LoRA 强制约束更新矩阵 $\Delta \mathbf{W}$ 的秩（Rank）不超过常数 $r$，且 $r \ll \min(d_{\text{out}}, d_{\text{in}})$。根据线性代数的矩阵分解原理，任何秩为 $r$ 的矩阵均可以分解为两个低秩矩阵的乘积：

$$ \Delta \mathbf{W} = \mathbf{A} \mathbf{B} $$

其中，$\mathbf{B} \in \mathbb{R}^{r \times d_{\text{in}}}$ 将原始高维特征投影到低维子空间，而 $\mathbf{A} \in \mathbb{R}^{d_{\text{out}} \times r}$ 将低维特征重新映射回高维的目标空间。前向传播公式也随之被重写为两条独立的数据流之和：

$$ \mathbf{h} = \mathbf{W}_0 \mathbf{x} + \mathbf{A} \mathbf{B} \mathbf{x}
$$

<div align="center">

<img src="/figures/07-robot-policy/latex/08-openvla/lora-low-rank-branch.png" alt="冻结基座支路与可训练低秩支路同形相加" width="86%">

_图 7.8-5：低秩支路先把 d_in 压到 r，再升回 d_out，与冻结基座输出相加；反向梯度只进入 A、B。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

训练时冻结预训练权重 $\mathbf{W}_0$，只更新小矩阵 $\mathbf{A}$ 和 $\mathbf{B}$。这样，需要更新的参数量从 $\mathcal{O}(d_{\text{out}} d_{\text{in}})$ 降到 $\mathcal{O}(r(d_{\text{out}} + d_{\text{in}}))$。OpenVLA 论文中的 LoRA 配置只训练约 1.4% 的参数；实际显存节省还受优化器、激活和量化设置影响。

## 代码实现

(**我们将利用 PyTorch 搭建 OpenVLA 的核心模块**)，包括动作的离散化处理器、视觉-语言投影层，以及简化的条件自回归生成流程。

```python
import torch
import torch.nn as nn
from typing import Tuple

class ActionTokenizer:
    def __init__(self, num_bins: int = 256, action_min: float = -1.0, action_max: float = 1.0):
        """
        初始化动作离散化器。
        教学实现假设动作已归一化到共享范围。真实 OpenVLA 使用每维 q01/q99 统计量。
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

        # 4. 因果掩码保证当前位置不能读取未来词元
        seq_len = combined_emb.size(1)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=combined_emb.device),
            diagonal=1,
        )
        hidden_states = self.llm_backbone(combined_emb, mask=causal_mask)

        # 5. 预测下一个词元的 logits分布
        logits = self.lm_head(hidden_states) # [batch, combined_seq_len, total_vocab_size]

        return logits
```

## 小结

- **OpenVLA** 结合 SigLIP、DINOv2、投影层与 Llama 2 7B，自回归生成动作词元。
- 动作按数据集、按维度使用第 1、99 百分位数归一化，再均匀量化为 256 桶；桶数降低量化误差，但不能单独保证控制精度。
- **LoRA** 用两个低秩矩阵表示权重更新。论文配置仅训练约 1.4% 的参数，适合资源受限的任务适配。

$$
$$
