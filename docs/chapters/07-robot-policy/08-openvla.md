# 7.8 OpenVLA：开源视觉-语言-动作模型

在上一节中，我们见证了 RT-1、RT-2 以及 RT-X 如何开辟将大语言模型先验直接注入机器人动作空间的革命性道路。然而，RT-2 等工业界模型拥有数百亿庞大参数，且其核心权重与训练代码均未公开。对于广大学术研究者与机器人初创团队而言，在单一工控机或消费级 GPU 上微调和部署这类巨型闭源模型几乎是一项不可能完成的任务。

2024 年，由斯坦福大学、伯克利加州大学等多所顶尖高校联合推出了 **OpenVLA**（Open Vision-Language-Action Model）。OpenVLA 不仅完全开源了其 70 亿参数（7B）的模型权重与全套跨具身训练管线，更在视觉编码表征、稳健动作分词与参数高效微调（PEFT）三大关键维度上做出了极具启发性的架构创新。

<div align="center">

<img src="/figures/07-robot-policy/source/08-openvla/openvla-fig1.png" alt="OpenVLA 结合双视觉编码器与 Llama 2 骨干，支持多机器人平台与高效微调。" width="86%">

_图 7.8-1：OpenVLA 结合双视觉编码器与 Llama 2 骨干，支持多机器人平台与高效微调。 出处：[OpenVLA: An Open-Source Vision-Language-Action Model，Moo Jin Kim et al.，2024](https://arxiv.org/abs/2406.09246)。_

</div>

---

## 7.8.1 物理与生理基石：人类双视觉通路与开源具身智能演进

要理解 OpenVLA 在视觉感知设计上的精妙之处，我们首先需要从人类双眼如何观察世界与引导双手的生物生理学机制讲起。

### 1. 生物视觉系统的双通路假说（Two-Streams Hypothesis）
当我们看着一个排球迎面飞来时，人类的大脑其实在以极高的速度同时解答两道截然不同的题目：
- 第一道是**语文与常识识别题**：“眼前这个物体是一个白红蓝相间的排球，而不是一个沉重的铅球或一只飞鸟”（负责回答**“它是什么”**）；
- 第二道是**立体几何与物理测距题**：“这个球当前距离我的双手还有 $45\text{ 厘米}$，正以大约 $5\text{ m/s}$ 的速度向斜下方飞行，我的双臂需要以 $30^\circ$ 仰角并拢垫击”（负责回答**“它在哪里、该怎么动”**）。

在神经生物学中，这两道题目分别由大脑中两条平行的神经纤维回路分工解答：
- **腹侧通路（Ventral Stream，又称“What”通路）**：延伸至大脑颞叶皮层，专门负责高级物体的**语义概念识别与分类**；
- **背侧通路（Dorsal Stream，又称“Where/How”通路）**：延伸至大脑顶叶皮层，专门负责捕捉物体的**三维空间位置、深度距离、几何轮廓与肌肉运动引导**。

在过去的多模态大模型研究中，绝大多数视觉编码器（如著名的 CLIP）几乎把全部技能点都加在了“What”语义通路上——它们在数亿张网页图文上训练，非常擅长在看到一张照片时给出“这是一只猫”的文字判断；但如果你问它“这只猫的爪子距离桌角精确相差几毫米”，纯语义编码器就会彻底失灵。如果直接把这种“近视眼”编码器装在机器人上，机器人往往能够“认出杯子”，却总因为“看不清杯柄的精确深度”而频繁抓空。

### 2. Prismatic 双编码器互补机制
为了让机器人同时具备“博学的常识”与“精准的空间视力”，OpenVLA 提出了 **Prismatic 双视觉融合架构**：同时引入了两个互补的预训练视觉主干网络：
1. **SigLIP（充当“语义学者”，负责 What 通路）**：通过海量图文对比学习训练，赋予机器人识别成千上万种日常物体名称与功能的语义常识；
2. **DINOv2（充当“几何工匠”，负责 Where/How 通路）**：通过无文字标签的纯图像空间几何自监督训练，能够精确感知微观像素级的空间深度、边缘轮廓与机械臂夹爪的精确位置。

<div align="center">

<img src="/figures/07-robot-policy/source/08-openvla/openvla-fig2.png" alt="SigLIP 与 DINOv2 提取的特征在通道维度拼接后投影至 LLM 维度" width="86%">

_图 7.8-2：SigLIP 与 DINOv2 提取的特征在通道维度拼接后投影至 LLM 维度。_

</div>

---

## 7.8.2 核心架构一：Prismatic 多源视觉融合机制

我们来一步步推演 Prismatic 架构如何将两种异构视觉特征无缝拼接并输入大语言模型。

设工作台摄像头输入的一张 RGB 彩色图片为 $I \in \mathbb{R}^{3 \times H \times W}$（例如分辨率为 $224 \times 224$ 像素）。就像把一张大拼图切割为若干个 $14 \times 14$ 像素的正方形小方块一样（在计算机视觉中称为**图像块 Patches**），整张图片共被切分为：

$$N_{\text{patches}} = \left(\frac{H}{14}\right) \times \left(\frac{W}{14}\right)$$

两路视觉编码器分别对每一个小方块提取特征：
- **SigLIP 提取的语义特征向量**：$\mathbf{Z}_{\text{SigLIP}} \in \mathbb{R}^{N_{\text{patches}} \times D_{\text{SigLIP}}}$（特征维度 $D_{\text{SigLIP}} = 1152$）；
- **DINOv2 提取的空间几何特征向量**：$\mathbf{Z}_{\text{DINOv2}} \in \mathbb{R}^{N_{\text{patches}} \times D_{\text{DINOv2}}}$（特征维度 $D_{\text{DINOv2}} = 1152$）。

### 1. 特征通道拼接与多层感知机（MLP）投影
对于第 $i$ 个图像块，我们将它的“语义特征”与“空间几何特征”像拼火车车厢一样首尾拼接在一起：

$$\mathbf{z}_i = [\mathbf{z}_{\text{SigLIP}, i}^\top, \mathbf{z}_{\text{DINOv2}, i}^\top]^\top \in \mathbb{R}^{D_{\text{SigLIP}} + D_{\text{DINOv2}}}$$

拼接后的总维度为 $1152 + 1152 = 2304$。

随后，通过一个带有非线性激活函数 GELU 的两层多层感知机（Projector），将融合特征线性转换至与大语言模型（如 Llama 2 7B）完全相同的词向量隐藏层维度 $D_{\text{LLM}} = 4096$：

$$\mathbf{h}_{\text{vis}, i} = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{z}_i) \in \mathbb{R}^{D_{\text{LLM}}}$$

> **公式符号逐一拆解**：
> - $\mathbf{W}_1 \in \mathbb{R}^{D_{\text{LLM}} \times (D_{\text{SigLIP}} + D_{\text{DINOv2}})}$：第一层线性变换矩阵；
> - $\text{GELU}(x) = x \cdot \Phi(x)$：高斯误差线性单元激活函数（$\Phi(x)$ 为标准正态分布的累积概率函数）；
> - $\mathbf{W}_2 \in \mathbb{R}^{D_{\text{LLM}} \times D_{\text{LLM}}}$：第二层线性变换矩阵；
> - $\mathbf{h}_{\text{vis}, i} \in \mathbb{R}^{D_{\text{LLM}}}$：最终生成的第 $i$ 个视觉词元向量，它现在可以像普通的文字单词一样直接输入给大语言模型了。

<details>
<summary><b>深入推导：自监督 DINOv2 空间补丁特征与对比学习 SigLIP 的特征正交性数学分析（点击展开查看完整推导）</b></summary>

考虑两个编码器在表示空间中的互信息（Mutual Information）与正交性。
SigLIP 优化的对比损失为全局图像-文本匹配：
$$\mathcal{L}_{\text{SigLIP}} = -\sum_{i} \log \frac{1}{1 + \exp\left(-t (\mathbf{v}_i^\top \mathbf{t}_i - b)\right)}$$
这促使其特征向量主要分布在语义相关的低维流形上，而对高频空间纹理具有不变性。
DINOv2 则基于知识蒸馏自监督学习局部 Patch 对应关系：
$$\mathcal{L}_{\text{DINO}} = - \sum_{k} P_{\text{teacher}}(k \mid \mathbf{x}) \log P_{\text{student}}(k \mid \mathbf{x}')$$
DINOv2 的自注意力图（Self-Attention Maps）保留了精准的物体边界与深度几何。两者在流形切空间上的正交互补性满足：
$$\cos(\theta_{\text{feat}}) = \frac{\langle \mathbf{z}_{\text{SigLIP}}, \mathbf{z}_{\text{DINOv2}} \rangle}{\|\mathbf{z}_{\text{SigLIP}}\|_2 \|\mathbf{z}_{\text{DINOv2}}\|_2} \approx 0$$
拼接后的联合表征同时最大化了任务相关的语义互信息 $I(\mathbf{Z}_{\text{vis}}; \text{Task})$ 与几何控制互信息 $I(\mathbf{Z}_{\text{vis}}; \mathbf{a}_{\text{target}})$。
</details>

---

## 7.8.3 核心架构二：分位数归一化动作量化（Quantile Action Tokenization）

在将连续物理动作划分为离散分桶时，我们在上一节采用了区间最大值 $a_{\max}$ 和最小值 $a_{\min}$ 进行线性归一化。但在面对来自全球数十家实验室的海量杂乱数据集时，这种简单粗暴的“极值归一化”会引发灾难性的数值失真。

### 1. 为什么绝对极值归一化会失效？
在基础统计学中，我们知道平均值极易受到极端异常值（Outliers）的拉扯。

在长达数万小时的人工遥操作轨迹中，由于示教员偶发的误触开关或急停碰撞，数据集中不可避免地存在极少数异常外点（例如某一次意外碰撞导致机械臂瞬时读数达到了正常速度的 10 倍）。

如果直接取数据绝对最大值 $a_{\max}$ 和最小值 $a_{\min}$：
$$a_{\text{span}} = a_{\max} - a_{\min} \gg \text{正常作业区间}$$
原本跨度仅有 $1.0\text{ 米}$ 的正常工作区间，会被拉大到 $10\text{ 米}$ 的分母中。这导致 $99.9\%$ 的正常动作全部被拥挤地压缩在 256 个桶中仅有的 $3 \sim 5$ 个桶内！原本 $2\text{ 毫米}$ 的高精度量化分辨率瞬间退化为十几厘米的粗糙大步，机器人动作立刻失控。

### 2. 稳健分位数归一化（Robust Quantile Normalization）
在大型统一考试与排位统计中，为了制定稳定的排位分数线，统计老师通常不会看极个别因为缺考或答题卡填错产生的 0 分，而是看“全校排名前 1% 的高分线”与“排名前 99% 的基础线”。

在数学统计中，这种把所有数据从小到大排序后处于特定百分比位置的数值被称为**分位数（Quantiles）**。

OpenVLA 采用数据集统计出的**第 1 百分位数（$q_{01}$）**与**第 99 百分位数（$q_{99}$）**作为可靠的物理边界，将中间 $98\%$ 的核心正常动作均匀铺满整个 256 个桶：

1. **边界截断（将极端外点收敛至分位数边界）**：
   $$\hat{a}_j = \min(\max(a_j, q_{01, j}), q_{99, j})$$
2. **基于分位数的稳健线性归一化**：
   $$\tilde{a}_j = \frac{\hat{a}_j - q_{01, j}}{q_{99, j} - q_{01, j}} \in [0, 1]$$
3. **离散分桶**：
   $$\text{Token}_j = \text{round}\left(\tilde{a}_j \times (B - 1)\right) \in \{0, 1, \dots, B - 1\}$$

**手算代入算例**：
设某机械臂 $x$ 轴动作经统计，第 1 百分位数为 $q_{01} = -0.35\text{ m}$，第 99 百分位数为 $q_{99} = +0.45\text{ m}$（有效跨度为 $q_{99} - q_{01} = 0.80\text{ m}$），量化桶数 $B = 256$。假定当前动作值为 $a_x = +0.05\text{ m}$。

1. 检查截断区间：$-0.35 \le 0.05 \le 0.45$，数值在正常区间内；
2. 计算归一化比例：
   $$\tilde{a}_x = \frac{0.05 - (-0.35)}{0.45 - (-0.35)} = \frac{0.40}{0.80} = 0.50$$
3. 映射为词元索引：
   $$\text{Token}_x = \text{round}(0.50 \times 255) = \text{round}(127.5) = 128$$

通过剔除首尾 $1\%$ 的异常外点，256 个桶的利用率提升至近乎 $100\%$，每一格的分辨率都被用在了刀刃上。

<details>
<summary><b>深入推导：基于累积经验分布函数（ECDF）的稳健分位数估计与动作截断误差界（点击展开查看完整推导）</b></summary>

设离散动作样本为 $\{a^{(1)}, a^{(2)}, \dots, a^{(N)}\}$，其经验累积分布函数定义为：
$$\hat{F}_N(a) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(a^{(i)} \le a)$$
分位数 $q_p$ 严格满足 $\hat{F}_N(q_p) = p$。
将区间截断在 $[q_{01}, q_{99}]$ 内部，对于任意样本 $a$，截断误差引起的动作失真能量为：
$$\mathbb{E}[(a - \hat{a})^2] = \int_{-\infty}^{q_{01}} (a - q_{01})^2 dF(a) + \int_{q_{99}}^{+\infty} (a - q_{99})^2 dF(a)$$
由于外点概率测度极小（总和仅为 $2\%$），截断能量期望被严格压制在微小上界 $\epsilon \le 0.005 \sigma^2$ 之内，同时在 $[q_{01}, q_{99}]$ 核心区间内将分桶分辨率方差降低了数个数量级。
</details>

---

## 7.8.4 核心架构三：参数高效微调（LoRA）的低秩几何本质

拥有 70 亿参数的 OpenVLA 若直接进行全参数微调（Full Fine-Tuning），不仅需要数张高规格企业级显卡（单卡显存 $> 80\text{ GB}$），还会导致模型遗忘海量的预训练通用常识（即灾难性遗忘）。

OpenVLA 全面采用**低秩自适应（Low-Rank Adaptation, LoRA）**技术，使得普通研究者仅用一张消费级显卡（如 RTX 4090）即可在几小时内完成特定机器人任务的高效适配。

<div align="center">

<img src="/figures/07-robot-policy/source/08-openvla/lora-fig1.png" alt="LoRA 用低秩矩阵分解参数更新，在冻结主权重时实现高效适配。" width="86%">

_图 7.8-3：LoRA 用低秩矩阵分解参数更新，在冻结主权重时实现高效适配。 出处：[LoRA: Low-Rank Adaptation of Large Language Models，Edward J. Hu et al.，2021](https://arxiv.org/abs/2106.09685)。_

</div>

<div align="center">

<img src="/figures/07-robot-policy/source/08-openvla/openvla-fig5.png" alt="新机器人平台上的适配任务比较全量与参数高效微调的实际效果。" width="86%">

_图 7.8-4：新机器人平台上的适配任务比较全量与参数高效微调的实际效果。 出处：[OpenVLA: An Open-Source Vision-Language-Action Model，Moo Jin Kim et al.，2024](https://arxiv.org/abs/2406.09246)。_

</div>

### 1. 矩阵分解的初等代数直觉
在初等代数中解多元线性方程组时，经常会发现如果第三个方程恰好是前两个方程相加得到的，那么第三个方程并没有提供新的信息，方程组的“有效独立未知数”其实减少了。

在线性代数中，一个庞大矩阵内部真正独立起作用的有效自由度，被称为矩阵的**秩（Rank）**。

LoRA 提出一个深刻的洞察：**预训练大模型已经具备了极为强大的世界常识。为了让它学会‘在新的工作台上抓取特定的杯子’这一具体技能，其 70 亿参数需要的调整变化量 $\Delta \mathbf{W}$，其实只集中在极少数几个核心的有效方向（低秩子空间）上**。

<div align="center">

<img src="/figures/07-robot-policy/latex/08-openvla/lora-low-rank-branch.png" alt="冻结基座支路与可训练低秩支路同形相加" width="86%">

_图 7.8-5：低秩支路先把 d_in 压到 r，再升回 d_out，与冻结基座输出相加；反向梯度只进入 A、B。_

</div>

基于这一洞察，LoRA 将原本巨大的参数更新矩阵 $\Delta \mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 强制分解为两个极薄的小矩阵相乘：

$$\Delta \mathbf{W} = \mathbf{B} \mathbf{A}$$

其中：
- $\mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}}$：**降维压缩矩阵**，将输入特征从高维空间压缩到极低的秩 $r$ 空间（例如取 $r = 16$ 或 $r = 32$）；
- $\mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r}$：**升维还原矩阵**，将低维特征重新映射回原本的输出维度。

在前向传播过程中，庞大的原始预训练权重 $\mathbf{W}_0$ 被完全“冰封”（不计算梯度也不更新），模型输出由两条并行支路相加得到：

$$\mathbf{h} = \mathbf{W}_0 \mathbf{x} + \frac{\alpha}{r} (\mathbf{B} \mathbf{A}) \mathbf{x}$$

其中 $\alpha$ 是一个固定的缩放超参数。

### 2. 算力与显存节省的惊人代数对比
设大语言模型中某一层全连接权重维度为 $d_{\text{in}} = 4096, d_{\text{out}} = 4096$：
- **全参数微调**：需要更新的参数量为 $4096 \times 4096 = 16,777,216$（约 **$1677\text{ 万}$** 个浮点数）；
- **LoRA 低秩微调（取 $r = 16$）**：参数量为 $r \times d_{\text{in}} + d_{\text{out}} \times r = 16 \times 4096 + 4096 \times 16 = 131,072$（仅 **$13.1\text{ 万}$** 个参数）！

需要更新的参数量仅仅是全量的 **$0.78\%$**！在 OpenVLA 的官方实验中，仅微调全模型约 **$1.4\%$** 的参数，就在未见过的全新机器人形态上取得了与全量微调同等乃至更优的鲁棒抓取表现。

<details>
<summary><b>深入推导：LoRA 低秩矩阵分解的本征维度假说与奇异值分解（SVD）近似（点击展开查看完整推导）</b></summary>

根据矩阵奇异值分解（SVD）定理，任意秩为 $k$ 的权重更新矩阵 $\Delta \mathbf{W}$ 均可写为：
$$\Delta \mathbf{W} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$
根据 Eckart-Young-Mirsky 定理，对于任意设定的低秩阶数 $r < k$，截断 SVD 给出了在 Frobenius 范数意义下的最佳低秩逼近：
$$\min_{\text{rank}(\mathbf{M}) \le r} \|\Delta \mathbf{W} - \mathbf{M}\|_F = \sqrt{\sum_{i=r+1}^k \sigma_i^2}$$
在深度过参数化大模型中，由于参数间的高度共线性，奇异值谱 $\sigma_i$ 呈现出极其陡峭的幂律衰减（Power-law Decay）。当 $r \ge 16$ 时，残余奇异值能量之和占总能量的比重不足 $0.1\%$，因此低秩分解能够在几乎无信息损失的前提下完成任务适配。
</details>

---

## 7.8.5 纯底层 PyTorch 代码实现：OpenVLA 核心组件与端到端前向推理

下面我们使用纯底层 PyTorch 算子手写实现 OpenVLA 的核心模块，包括分位数动作离散化器、Prismatic 双视觉投影层、LoRA 线性层以及微型 OpenVLA 主干网络。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class OpenVLAActionTokenizer:
    """
    基于分位数统计的动作离散化与反量化器
    """
    def __init__(self, num_bins: int = 256, q01: torch.Tensor = None, q99: torch.Tensor = None):
        self.num_bins = num_bins
        # 默认 7 维动作的分位数边界
        self.q01 = q01 if q01 is not None else -torch.ones(7)
        self.q99 = q99 if q99 is not None else torch.ones(7)

    def tokenize(self, continuous_actions: torch.Tensor) -> torch.Tensor:
        """
        连续动作 -> 稳健离散词元索引
        :param continuous_actions: (B, 7)
        :return: (B, 7) 整数张量
        """
        self.q01 = self.q01.to(continuous_actions.device)
        self.q99 = self.q99.to(continuous_actions.device)

        # 1. 分位数截断
        clamped = torch.max(torch.min(continuous_actions, self.q99), self.q01)
        # 2. 稳健归一化到 [0, 1]
        norm_actions = (clamped - self.q01) / (self.q99 - self.q01 + 1e-8)
        # 3. 均匀分桶取整
        tokens = torch.round(norm_actions * (self.num_bins - 1)).long()
        return tokens

    def detokenize(self, bin_indices: torch.Tensor) -> torch.Tensor:
        """
        离散词元索引 -> 连续动作恢复
        """
        self.q01 = self.q01.to(bin_indices.device)
        self.q99 = self.q99.to(bin_indices.device)

        norm_actions = bin_indices.float() / (self.num_bins - 1)
        continuous_actions = norm_actions * (self.q99 - self.q01) + self.q01
        return continuous_actions

class PrismaticProjector(nn.Module):
    """
    Prismatic 双流视觉特征融合投影层
    将 SigLIP (语义) 与 DINOv2 (空间几何) 特征通道拼接并投影至 LLM 维度
    """
    def __init__(self, siglip_dim: int = 256, dinov2_dim: int = 256, llm_dim: int = 512):
        super().__init__()
        fused_dim = siglip_dim + dinov2_dim
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, llm_dim, bias=False),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim, bias=False)
        )

    def forward(self, siglip_feat: torch.Tensor, dinov2_feat: torch.Tensor) -> torch.Tensor:
        """
        :param siglip_feat: (B, num_patches, siglip_dim)
        :param dinov2_feat: (B, num_patches, dinov2_dim)
        :return: (B, num_patches, llm_dim)
        """
        # 通道维度拼接
        fused = torch.cat([siglip_feat, dinov2_feat], dim=-1)
        return self.mlp(fused)

class LoRALinear(nn.Module):
    """
    手写底层 LoRA (Low-Rank Adaptation) 线性层
    h = W_0 * x + (alpha / r) * B * A * x
    """
    def __init__(self, base_linear: nn.Linear, r: int = 16, lora_alpha: float = 32.0):
        super().__init__()
        self.base_linear = base_linear
        # 冻结原始基座权重
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        in_dim = base_linear.in_features
        out_dim = base_linear.out_features
        self.r = r
        self.scaling = lora_alpha / r

        # 初始化 A 为高斯分布，B 为全 0
        self.lora_A = nn.Parameter(torch.randn(r, in_dim) * (1.0 / r))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        # 低秩支路: x -> (x @ A^T) @ B^T
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

class SimpleOpenVLA(nn.Module):
    """
    微型 OpenVLA 演示模型
    """
    def __init__(self, text_vocab_size: int = 32000, action_bins: int = 256, llm_dim: int = 512):
        super().__init__()
        self.projector = PrismaticProjector(siglip_dim=256, dinov2_dim=256, llm_dim=llm_dim)
        self.total_vocab_size = text_vocab_size + action_bins
        self.embedding = nn.Embedding(self.total_vocab_size, llm_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=llm_dim, nhead=4, dim_feedforward=llm_dim * 2, batch_first=True
        )
        self.llm_backbone = nn.TransformerEncoder(decoder_layer, num_layers=2)
        self.lm_head = nn.Linear(llm_dim, self.total_vocab_size, bias=False)

    def forward(
        self,
        siglip_tokens: torch.Tensor,
        dinov2_tokens: torch.Tensor,
        text_action_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        前向计算
        """
        # 1. 视觉特征融合并投影
        vis_emb = self.projector(siglip_tokens, dinov2_tokens) # (B, N_patches, llm_dim)
        # 2. 文本/动作嵌入
        tok_emb = self.embedding(text_action_tokens) # (B, seq_len, llm_dim)

        # 3. 序列拼接
        seq = torch.cat([vis_emb, tok_emb], dim=1) # (B, N_patches + seq_len, llm_dim)

        # 4. 因果自回归掩码
        seq_len = seq.size(1)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=seq.device), diagonal=1
        )
        hidden = self.llm_backbone(seq, mask=causal_mask)
        logits = self.lm_head(hidden)
        return logits

# ===================================================================
# 单元测试与低秩参数量校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 2
    num_patches = 16
    action_bins = 256
    llm_dim = 512

    # 1. 测试分位数动作分词器
    q01 = torch.tensor([-0.50] * 7)
    q99 = torch.tensor([0.50] * 7)
    tokenizer = OpenVLAActionTokenizer(num_bins=action_bins, q01=q01, q99=q99)
    dummy_act = torch.tensor([[0.25, -0.10, 0.00, 0.50, -0.60, 0.10, 0.45]])
    toks = tokenizer.tokenize(dummy_act)
    recon = tokenizer.detokenize(toks)
    print(f"[OpenVLA Test] 输入动作: {dummy_act.numpy().round(3)}")
    print(f"[OpenVLA Test] 分位数离散索引: {toks.numpy()}")
    print(f"[OpenVLA Test] 最大重构误差: {(recon - dummy_act).abs().max().item():.6f}")

    # 2. 测试 LoRA 模块与参数量统计
    base_dense = nn.Linear(4096, 4096)
    lora_dense = LoRALinear(base_dense, r=16, lora_alpha=32.0)
    trainable_params = sum(p.numel() for p in lora_dense.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in lora_dense.parameters() if not p.requires_grad)
    print(f"[LoRA Test] 可训练参数量 (A + B): {trainable_params} ({trainable_params / frozen_params * 100:.2f}%)")
    print(f"[LoRA Test] 冻结基座参数量 (W_0): {frozen_params}")
    assert trainable_params == 16 * 4096 * 2, "LoRA 参数量计算不符！"

    # 3. 测试 OpenVLA 前向推理
    model = SimpleOpenVLA(text_vocab_size=32000, action_bins=action_bins, llm_dim=llm_dim)
    model.eval()

    dummy_siglip = torch.randn(batch_size, num_patches, 256)
    dummy_dinov2 = torch.randn(batch_size, num_patches, 256)
    dummy_input_tokens = torch.randint(0, 32000, (batch_size, 10))

    with torch.no_grad():
        out_logits = model(dummy_siglip, dummy_dinov2, dummy_input_tokens)

    print(f"[OpenVLA Test] 前向推理输出 Logits 形状: {out_logits.shape}")
    assert out_logits.shape == (batch_size, num_patches + 10, 32000 + action_bins)
    print("✓ OpenVLA 双通路视觉与 LoRA 微调单测全部通过！")
```

---

## 7.8.6 本节小结

回顾本节内容，我们建立了一条从生物视觉双通路走向现代开源大一统 VLA 模型的完整知识脉络：
1. **Prismatic 双视觉通路**：融合语义导向的 SigLIP 与空间几何导向的 DINOv2，克服了单一 CLIP 编码器空间定位迟钝的缺陷；
2. **稳健分位数动作量化**：通过第 1 与第 99 分位数截断，规避了海量多源数据中的极端外点干扰，最大化保留了 256 桶离散化的物理控制分辨率；
3. **低秩自适应（LoRA）的几何本质**：基于下游任务适配的低秩子空间假说，将参数微调量严格限制在 $1.4\%$ 左右，使大模型能够在低算力下敏捷迁移到新机器人形态。
