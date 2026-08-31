# 7.1 具身智能与多模态观测

> **本章导读**
>
> **讲什么：** 本章把问题从“预测会发生什么”转向“让机器人现在做什么”。我们先处理视觉、本体感觉、触觉与接触动力学，再从行为克隆出发，依次研究扩散策略、动作分块和视觉—语言—动作模型，最后把世界模型接回策略，用预测的后果检查候选动作。
>
> **为什么策略不能只看一张 RGB 图像：** 机械臂看见杯子，并不等于知道自己的关节位置、夹爪受力和杯子是否已经滑动。真实动作连续、常有多种正确做法，还会因一次小偏差进入训练数据未覆盖的状态；因此策略既要融合异构观测，也要处理多峰动作、长动作序列和闭环误差。
>
> **故事线：** `融合身体与环境观测 → 理解接触和全身控制约束 → 用行为克隆建立基线并观察分布偏移 → 用扩散与动作分块表达多种连续动作 → 用语言和大规模数据扩展任务 → 用世界模型检查动作后果`

先看一个抓杯子的控制时刻。相机图像给出杯子在桌面上的位置，关节编码器给出机械臂当前姿态，夹爪传感器则反映接触力。只有图像，策略不知道手臂能否到达；只有关节角，策略又不知道杯子在哪里。

<div align="center">

<img src="/figures/07-robot-policy/source/01-multimodal-observation/levine-fig1.png" alt="相机画面与机械臂构型共同进入视觉运动策略，输出直接驱动真实机器人。" width="86%">

_图 7.1-1：相机画面与机械臂构型共同进入视觉运动策略，输出直接驱动真实机器人。 出处：[End-to-End Training of Deep Visuomotor Policies，Sergey Levine; Chelsea Finn; Trevor Darrell; Pieter Abbeel，2016](https://arxiv.org/abs/1504.00702)。_

</div>

机器人需要把这些物理含义、维度和采样频率不同的信号组合起来。这样的输入称为**多模态观测**（Multimodal Observation）；强调智能体通过身体持续感知并作用于环境的研究范式，称为**具身智能**（Embodied AI）。

## 7.1.1 历史脉络与学术追溯

1986 年，Rodney Brooks 在论文《A Robust Layered Control System for a Mobile Robot》中提出包容体系结构（Subsumption Architecture），用分层的感觉—动作模块控制移动机器人 [[Brooks, 1986]](https://doi.org/10.1109/JRA.1986.1087032)。这项工作代表了一条重要路线：控制不必总从完整的符号世界模型开始，也可以由与环境紧密耦合的行为层组成。

Levine 等人用引导策略搜索训练深度视觉运动策略，使卷积网络根据相机图像与机器人构型输出电机转矩 [[Levine et al., 2016]](https://arxiv.org/abs/1504.00702)。这项工作直接展示了端到端视觉运动策略在多项机器人操作任务上的训练与执行；论文不需要承担“最早把 CNN 与强化学习结合”这一优先权判断。

Transformer 为序列中的跨位置信息交互提供了通用结构 [[Vaswani et al., 2017]](https://arxiv.org/abs/1706.03762)。RT-1 根据相机图像序列与自然语言任务描述预测离散化机器人动作 [[Brohan et al., 2022]](https://arxiv.org/abs/2212.06817)；RT-2 又把机器人动作表示为文本词元，并联合利用互联网视觉—语言数据与机器人轨迹训练视觉—语言—动作模型 [[Brohan et al., 2023]](https://arxiv.org/abs/2307.15818)。这两篇论文不能用来证明系统输入包含 RGB-D 或任意高维本体感受，因此这里只列出原文明确使用的模态。

<div align="center">

<img src="/figures/07-robot-policy/source/01-multimodal-observation/rt2-fig1.png" alt="RT-2 把机器人动作表示为语言 token，连接视觉语言推理与低层控制。" width="86%">

_图 7.1-2：RT-2 把机器人动作表示为语言 token，连接视觉语言推理与低层控制。 出处：[RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control，Anthony Brohan et al.，2023](https://arxiv.org/abs/2307.15818)。_

</div>

## 7.1.2 物理量的降维映射：从单摆到机器人状态空间

为了理解多模态观测的必要性，我们不妨先回到高中物理中最经典的单摆模型。

假设单摆的长度、重力参数和外部输入均已知。此时，摆角 $\theta$ 与角速度 $\dot{\theta}$ 构成一个足以继续积分动力学方程的状态；二者缺一不可，因为相同摆角可能对应向左或向右运动。

在机器人学中，这种对自身内在物理状态的测量，被称为**本体感受**（Proprioception）。对于一个拥有 $n$ 个自由度的机器人，其本体状态可以通过广义坐标 $\mathbf{q} \in \mathbb{R}^n$（例如各关节的角度）和广义速度 $\dot{\mathbf{q}} \in \mathbb{R}^n$（各关节的角速度）来严格定义。我们将其拼接为一个本体观测向量：

$$
\mathbf{o}_{\text{prop}} = [\mathbf{q}^\top, \dot{\mathbf{q}}^\top]^\top \in \mathbb{R}^{2n}
$$

抓取桌上的苹果时，机械臂不仅要知道自身关节角度，还要估计苹果的位置。这种对外部环境的感知称为**外感受**（Exteroception）。RGB 摄像头是常见的外感受器，输出可写成三维张量 $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$。

因此，在时间步 $t$，具身智能体所接收到的完整多模态观测 $\mathbf{o}_t$ 至少包含了视觉和本体两个模态：

$$
\mathbf{o}_t = \{ \mathbf{I}_t, \mathbf{o}_{\text{prop}, t} \}
$$

我们的目标是设计一个神经网络函数 $f_\theta$，将这个异构的观测集合映射为一个统一的低维稠密向量 $\mathbf{z}_t \in \mathbb{R}^d$，从而供下游的策略网络（Policy Network）计算具体的控制动作。

## 7.1.3 模态对齐与融合的数学推导

视觉图像 $\mathbf{I}$ 是高维张量，本体状态 $\mathbf{o}_{\text{prop}}$ 则是较短的向量。二者的形状和单位不同，不能直接逐元素相加。通常先用各自的编码器（Encoder）提取特征，再在兼容的表示空间中融合。

首先，我们分别独立地对两种模态进行编码：

$$
\mathbf{z}_{\text{vis}} = f_{\text{vis}}(\mathbf{I}; \theta_{\text{vis}}) \in \mathbb{R}^{d_v}
$$

$$
\mathbf{z}_{\text{prop}} = f_{\text{prop}}(\mathbf{o}_{\text{prop}}; \theta_{\text{prop}}) \in \mathbb{R}^{d_p}
$$

其中，$f_{\text{vis}}$ 通常是ResNet或Vision Transformer（ViT），而 $f_{\text{prop}}$ 通常是一个多层感知机（MLP）。

接下来，我们需要将 $\mathbf{z}_{\text{vis}}$ 和 $\mathbf{z}_{\text{prop}}$ 融合。最直观也是最简单的方法是**拼接（Concatenation）与线性投影**。

先看一维情况。设视觉编码器输出标量 $z_v \in \mathbb{R}$，本体编码器输出标量 $z_p \in \mathbb{R}$，线性融合就是分别加权后再加偏置：

$$
z = w_1 z_v + w_2 z_p + b
$$

将这个标量方程严格地推广到高维向量空间。我们将两个特征向量在特征维度上进行拼接，得到向量 $[\mathbf{z}_{\text{vis}}^\top, \mathbf{z}_{\text{prop}}^\top]^\top \in \mathbb{R}^{d_v + d_p}$。然后，我们应用一个权重矩阵 $\mathbf{W} \in \mathbb{R}^{d \times (d_v + d_p)}$ 进行线性投影，并经过一个非线性激活函数 $\sigma$：

$$
\mathbf{z}_{\text{fused}} = \sigma \left( \mathbf{W} \begin{bmatrix} \mathbf{z}_{\text{vis}} \\ \mathbf{z}_{\text{prop}} \end{bmatrix} + \mathbf{b} \right)
$$

这是一种**后期融合**（Late Fusion）。它实现简单，但同一组投影参数会用于所有样本；如果任务需要“由当前本体状态决定应读取哪个图像区域”，单次拼接不一定能显式表达这种选择关系。

## 7.1.4 跨模态注意力机制（Cross-Modal Attention）

在高度动态的物理交互中，静态融合往往是不够的。

以移动机器人变道为例，当前转向角和速度可以作为查询，图像中的前方道路、后视镜和邻车区域则提供候选视觉信息。跨模态注意力（Cross-Modal Attention）用本体特征计算查询，再对不同视觉区域分配随状态变化的权重。

<div align="center">

<img src="/figures/07-robot-policy/source/01-multimodal-observation/rt1-fig13.png" alt="RT-1 的注意力可视化展示不同层和头如何聚焦任务相关图像区域。" width="86%">

_图 7.1-3：RT-1 的注意力可视化展示不同层和头如何聚焦任务相关图像区域。 出处：[RT-1: Robotics Transformer for Real-World Control at Scale，Anthony Brohan et al.，2022](https://arxiv.org/abs/2212.06817)。_

</div>

我们不再将视觉图像编码为单一的全局向量，而是保留其空间结构，将其编码为 $N$ 个局部特征块（Patch Embeddings），即 $\mathbf{Z}_{\text{vis}} \in \mathbb{R}^{N \times d_v}$。

在这里，我们引入注意力机制。我们将机器人的本体特征 $\mathbf{z}_{\text{prop}}$ 视作查询向量（Query），而将视觉特征矩阵 $\mathbf{Z}_{\text{vis}}$ 视作键值对（Keys and Values）。

首先，我们看一个局部视觉块 $i$ 与本体查询之间的相关性。我们通过线性变换将它们投影到相同的维度 $d_k$ 中，计算点积来衡量相似度，并使用缩放因子 $\sqrt{d_k}$ 保证数值稳定性：

$$
e_i = \frac{(\mathbf{W}_q \mathbf{z}_{\text{prop}})^\top (\mathbf{W}_k \mathbf{z}_{\text{vis}, i})}{\sqrt{d_k}}
$$

为了将这个不受界的能量值 $e_i$ 转化为合法的概率分布，我们应用 Softmax 操作：

$$
\alpha_i = \frac{\exp(e_i)}{\sum_{j=1}^N \exp(e_j)}
$$

最后，我们用这些概率权重 $\alpha_i$ 对视觉值向量（Value vectors）进行加权求和，得到融合后的特征向量：

$$
\mathbf{z}_{\text{cross}} = \sum_{i=1}^N \alpha_i (\mathbf{W}_v \mathbf{z}_{\text{vis}, i})
$$

将上述步骤统一写成严格的矩阵乘法形式。令查询 $\mathbf{Q} \in \mathbb{R}^{1 \times d_k}$，键 $\mathbf{K} \in \mathbb{R}^{N \times d_k}$，值 $\mathbf{V} \in \mathbb{R}^{N \times d_v}$：

$$
\text{CrossAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax} \left( \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} \right) \mathbf{V} \in \mathbb{R}^{1 \times d_v}

$$

<div align="center">

<img src="/figures/07-robot-policy/latex/01-multimodal-observation/cross-attention-row-softmax.png" alt="单个本体查询沿视觉 patch 维做行 Softmax，再汇聚 Value" width="86%">

_图 7.1-4：单个本体查询与 N 个视觉键形成一行分数，Softmax 只沿 patch 维归一化，再用同组权重汇聚 Value。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

这样得到的视觉汇总会随本体状态变化。注意力权重可以提示模型正在使用哪些区域，但不能自动等同于因果解释。

## 7.1.5 代码实现：构建多模态观测编码器

下面用 PyTorch 实现一个视觉 CNN、本体 MLP 与拼接融合组成的最小编码器。

```python
import torch
from torch import nn

class MultiModalEncoder(nn.Module):
    def __init__(self, img_channels=3, prop_dim=14, vis_embed_dim=256,
                 prop_embed_dim=64, fused_dim=128):
        """
        参数:
            img_channels (int): 输入图像通道数
            prop_dim (int): 本体观测向量的原始维度 (例如7个关节的角度和速度)
            vis_embed_dim (int): 视觉特征提取后的维度
            prop_embed_dim (int): 本体特征提取后的维度
            fused_dim (int): 最终融合后的联合表示维度
        """
        super().__init__()

        # 1. 视觉编码器：使用一个简单的浅层CNN代替ResNet以简化演示
        self.vis_encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(vis_embed_dim),
            nn.LayerNorm(vis_embed_dim)
        )

        # 2. 本体编码器：使用两层MLP
        self.prop_encoder = nn.Sequential(
            nn.Linear(prop_dim, 128),
            nn.ReLU(),
            nn.Linear(128, prop_embed_dim),
            nn.LayerNorm(prop_embed_dim)
        )

        # 3. 融合层：拼接后通过MLP映射到目标维度
        self.fusion_mlp = nn.Sequential(
            nn.Linear(vis_embed_dim + prop_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, fused_dim)
        )

    def forward(self, img_obs, prop_obs):
        """
        参数:
            img_obs: 形状为 (B, C, H, W) 的图像张量
            prop_obs: 形状为 (B, prop_dim) 的本体状态向量
        返回:
            fused_feature: 形状为 (B, fused_dim) 的多模态融合特征
        """
        # 提取视觉特征
        z_vis = self.vis_encoder(img_obs)
        # 提取本体特征
        z_prop = self.prop_encoder(prop_obs)

        # 在特征维度(dim=1)进行拼接 [B, vis_embed_dim + prop_embed_dim]
        z_concat = torch.cat([z_vis, z_prop], dim=1)

        # 线性投影与非线性激活
        fused_feature = self.fusion_mlp(z_concat)

        return fused_feature

# 测试前向传播
encoder = MultiModalEncoder()
dummy_img = torch.randn(4, 3, 84, 84) # Batch size 4, 84x84 RGB图像
dummy_prop = torch.randn(4, 14)       # Batch size 4, 14维本体状态
output = encoder(dummy_img, dummy_prop)
print(f"融合特征的张量形状: {output.shape}")
```

## 7.1.6 小结

- 具身智能要求智能体处理与其躯体及环境物理交互相关的数据。**多模态观测**（主要是视觉外感受与关节本体感受）是构建具身策略网络的基础。
- 对于跨越维度和语义鸿沟的多源数据，我们必须通过各自专用的编码网络将其投影到统一的潜空间中。
- 简单的**拼接融合（Late Fusion）**实现简单但缺乏动态交互能力；**跨模态注意力机制（Cross-Modal Attention）**允许神经网络基于本体状态动态地对空间视觉特征进行加权选择。

## 7.1.7 练习

1. 在该公式中，如果我们要描述一台带有6自由度机械臂（每个关节可测角度和角速度）以及一个底盘（可测平面 $x, y$ 坐标、朝向角 $\psi$ 及其对应的速度）的移动机器人，其本体观测向量 $\mathbf{o}_{\text{prop}}$ 的维度是多少？
   - **提示**：分别计算机械臂和底盘的广义坐标和速度维度并求和。
2. 仔细观察代码实现中的 `MultiModalEncoder` 类。为什么在对 `z_vis` 和 `z_prop` 提取特征的最后一步，我们都加入了一个 `LayerNorm`（层归一化）操作？如果不加，在后续的拼接与线性映射中可能会引发什么数值优化问题？
   - **提示**：思考不同模态编码器初始输出权重的方差差异，以及这种差异在 $\mathbf{W} \mathbf{z}_{\text{concat}}$ 矩阵乘法中会导致梯度如何流动。
3. 如果我们希望将当前的**后期拼接融合**替换为相关章节提到的**跨模态注意力融合**，请写出将视觉卷积特征图（形状为 `[B, 64, 7, 7]`）转换为注意力键 $\mathbf{K}$ 和值 $\mathbf{V}$ 时，张量形状必须经历哪些重塑（Reshape）和转置操作？
