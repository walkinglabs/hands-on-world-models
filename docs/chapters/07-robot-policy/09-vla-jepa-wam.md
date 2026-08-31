# 基于 JEPA 的视觉-语言-动作模型（VLA-JEPA/WAM）

机器人推开抽屉时，真正影响动作的是把手位置、机械臂姿态和抽屉的运动状态；光照闪烁或背景里晃动的树叶通常不是控制目标。像素生成式世界模型需要预测大量视觉细节，而 JEPA 路线尝试改在表征空间预测未来，让模型有机会把容量集中到更稳定的结构上。这里的“有机会”很重要：网络不会仅凭架构就自动忽略所有无关信息。

2022 年，Yann LeCun 系统阐述了联合嵌入预测架构（Joint-Embedding Predictive Architecture, JEPA）[[LeCun, 2022]](https://openreview.net/forum?id=BZ5a1r-kVsf)，主张在表征空间中预测与任务有关的信息。V-JEPA 随后把这一思路用于视频表征学习 [[Bardes et al., 2024]](https://arxiv.org/abs/2404.08471)。把预测状态与生成动作结合的机器人模型仍是快速演进中的研究范式；“World Action Model（WAM）”是对多种相关架构的统称，而不是 V-JEPA 论文定义的单一标准模型 [[Wang et al., 2026]](https://arxiv.org/abs/2605.12090)。本节中的“VLA-JEPA”因此表示一种教学性的组合设计，不声称复现某篇同名论文。

<div align="center">

<img src="/figures/07-robot-policy/source/09-vla-jepa-wam/wam-fig1.png" alt="WAM 谱系图把显式、隐式、扩散式与联合 WAM 放入统一历史坐标。" width="86%">

_图 7.9-1：WAM 谱系图把显式、隐式、扩散式与联合 WAM 放入统一历史坐标。 出处：[World Action Models: The Next Frontier in Embodied AI，Siyin Wang et al.，2026](https://arxiv.org/abs/2605.12090)。_

</div>

本节从经典力学的状态变量出发，构造一个用于教学的动作条件 JEPA，并说明目标编码器、预测器和 EMA 更新各自解决什么问题。

## 从经典力学状态空间到隐变量抽象

<div align="center">

<img src="/figures/07-robot-policy/source/09-vla-jepa-wam/vjepa-fig1.png" alt="V-JEPA 的特征预测与冻结评估说明隐空间预测为何可避开像素细节。" width="86%">

_图 7.9-2：V-JEPA 的特征预测与冻结评估说明隐空间预测为何可避开像素细节。 出处：[Revisiting Feature Prediction for Learning Visual Representations from Video，Adrien Bardes et al.，2024](https://arxiv.org/abs/2404.08471)。_

</div>

先从经典力学的状态建模理解“预测表征而非像素”的动机。

在牛顿力学中，假设我们在考察一个质点在三维空间中的抛体运动。尽管质点可能由数以亿计的原子构成，或者它在不同的光照下呈现出截然不同的视觉反光，但我们要预测它在时间 $\Delta t$ 之后的空间演变，只需要抽取极少数的几个本质物理量：初始位置向量 $\mathbf{p}_0 \in \mathbb{R}^3$ 和速度向量 $\mathbf{v}_0 \in \mathbb{R}^3$。

在外力向量 $\mathbf{F}$（我们可以将其视为系统接收到的“动作” $a$）和重力加速度 $\mathbf{g}$ 的作用下，质点的新状态可以通过严密的动力学方程组直接求得：

$$
\mathbf{p}_1 = \mathbf{p}_0 + \mathbf{v}_0 \Delta t + \frac{1}{2} \left( \mathbf{g} + \frac{\mathbf{F}}{m} \right) \Delta t^2
$$

在质量 $m$、重力和外力都已知的这个简化模型里，$(\mathbf{p}_0, \mathbf{v}_0)$ 足以预测下一时刻；若空气阻力或接触条件未知，状态还需要补充。低维状态的意义，是保留预测所需变量，而不是复现质点表面的纹理。

具身机器人的学习问题只在“从高维观测提取可预测状态”这一点上与它相似，并不与理想质点模型等价。机器人通常从图像 $\mathbf{X}_t \in \mathbb{R}^{H \times W \times C}$ 和本体传感器估计环境状态，任务目标还可能由语言指令 $l$ 给出。

本节构造的教学模型使用深层神经网络把观测 $\mathbf{X}_t$ 映射到低维向量，并在该空间内做动作条件预测。这个向量是否真的等价于完整物理状态，必须通过探针、下游控制与反事实实验验证，不能由网络结构本身保证。

## VLA-JEPA 的数学体系基础

<div align="center">

<img src="/figures/07-robot-policy/source/09-vla-jepa-wam/wam-fig2.png" alt="WAM 路线图按表征、架构、学习与评估拆解动作世界模型的设计空间。" width="86%">

_图 7.9-3：WAM 路线图按表征、架构、学习与评估拆解动作世界模型的设计空间。 出处：[World Action Models: The Next Frontier in Embodied AI，Siyin Wang et al.，2026](https://arxiv.org/abs/2605.12090)。_

</div>

为了实现上述构想，VLA-JEPA 定义了一套由三个核心神经网络模块组成的数学闭环：上下文编码器（Context Encoder）$E_\theta$、目标编码器（Target Encoder）$E_{\bar{\theta}}$ 和预测器（Predictor）$P_\phi$。

### 1. 联合空间映射与上下文编码

给定时刻 $t$ 的图像观测 $\mathbf{X}_t \in \mathbb{R}^{H \times W \times C}$ 与语言指令序列向量 $\mathbf{l} \in \mathbb{R}^{D_l}$。上下文编码器 $E_\theta$ 需要捕捉图像的几何空间布局，并根据语言指令的语义焦点，提取出当前时刻的隐状态表达 $\mathbf{s}_t$：

$$
\mathbf{s}_t = E_\theta(\mathbf{X}_t, \mathbf{l}), \quad \mathbf{s}_t \in \mathbb{R}^{D_s}
$$

这里，$D_s$ 是隐空间维度。训练目标希望 $\mathbf{s}_t$ 保留有助于未来预测的结构，但它究竟编码了物体位姿、背景还是其他统计线索，需要额外实验验证。

类似地，环境在下一个时间步（或未来第 $k$ 步）的真实观测 $\mathbf{X}_{t+1}$ 被馈入目标编码器 $E_{\bar{\theta}}$，产生目标隐状态 $\bar{\mathbf{s}}_{t+1}$：

$$
\bar{\mathbf{s}}_{t+1} = E_{\bar{\theta}}(\mathbf{X}_{t+1}, \mathbf{l}), \quad \bar{\mathbf{s}}_{t+1} \in \mathbb{R}^{D_s}
$$

目标编码器参数 $\bar{\theta}$ 与上下文编码器参数 $\theta$ 的更新方式不同：前者不接受当前损失的反向传播，而由后者的指数移动平均更新。

### 2. 隐空间上的预测闭环

当系统获取了当前时刻的隐状态 $\mathbf{s}_t$ 后，机器人执行了特定的多维连续动作 $\mathbf{a}_t \in \mathbb{R}^{D_a}$（例如 $SE(3)$ 空间内的末端执行器位姿增量与夹爪开合度）。预测器 $P_\phi$ 则充当了该公式中物理方程的角色，负责在隐空间内进行时间向前的动态推演：

<div align="center">

<img src="/figures/07-robot-policy/source/09-vla-jepa-wam/wam-fig4.png" alt="世界模型可分别支持 VLA 模仿、强化学习、奖励建模与策略评估。" width="86%">

_图 7.9-4：世界模型可分别支持 VLA 模仿、强化学习、奖励建模与策略评估。 出处：[World Action Models: The Next Frontier in Embodied AI，Siyin Wang et al.，2026](https://arxiv.org/abs/2605.12090)。_

</div>

$$
\hat{\mathbf{s}}_{t+1} = P_\phi(\mathbf{s}_t, \mathbf{a}_t), \quad \hat{\mathbf{s}}_{t+1} \in \mathbb{R}^{D_s}
$$

## 打破特征坍缩博弈：不对称结构与指数移动平均

在定义了架构之后，我们自然需要衡量预测与真实目标之间的差距。在欧几里得隐空间中，最直接的损失度量是预测向量 $\hat{\mathbf{s}}_{t+1}$ 与目标向量 $\bar{\mathbf{s}}_{t+1}$ 之间的均方误差（MSE）：

$$
\mathcal{L}_{MSE}(\theta, \phi) = \frac{1}{D_s} \sum_{i=1}^{D_s} \left( \hat{\mathbf{s}}_{t+1}^{(i)} - \bar{\mathbf{s}}_{t+1}^{(i)} \right)^2 = \frac{1}{D_s}\left\| P_\phi(E_\theta(\mathbf{X}_t, \mathbf{l}), \mathbf{a}_t) - E_{\bar{\theta}}(\mathbf{X}_{t+1}, \mathbf{l}) \right\|_2^2
$$

这个目标存在一个需要防范的退化解，称为**特征坍缩（Feature Collapse）**。

如果两个编码器和预测器都直接追随同一个 MSE 目标，常量表示也是一个零损失解：任意输入都映射到同一向量，预测器再输出同一向量。此时损失很小，但表示不再区分环境状态。训练是否真的走向该解取决于架构和优化过程，不能简单断言会“迅速归零”。

上下文编码器与预测器接受梯度，目标编码器只缓慢跟随在线编码器，这种更新不对称使预测目标在单个训练步内保持固定。

为降低退化到平凡解的风险，本节采用 I-JEPA 风格的不对称参数更新。**上下文编码器 $\theta$ 与预测器 $\phi$ 通过梯度反向传播更新：**

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_{MSE}, \quad \phi \leftarrow \phi - \eta \nabla_\phi \mathcal{L}_{MSE}
$$

而**目标编码器 $\bar{\theta}$ 必须通过上下文编码器历史参数的指数移动平均（Exponential Moving Average, EMA）进行平滑更新**，其更新过程彻底阻断了由于当前 Batch 的 MSE 损失所产生的任何即时梯度：

$$
\bar{\theta} \leftarrow \tau \bar{\theta} + (1 - \tau) \theta

$$

<div align="center">

<img src="/figures/07-robot-policy/latex/09-vla-jepa-wam/ema-target-update.png" alt="旧目标参数和在线参数按 EMA 权重合成新目标参数" width="86%">

_图 7.9-5：目标参数不由当前损失反向更新，而是把旧目标与当前在线参数按 τ 和 1−τ 做跨步平滑。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

其中衰减率参数 $\tau \in [0.99, 1)$。$\bar{\theta}$ 提供随时间缓慢变化的目标，经验上有助于稳定训练。它与停止梯度、预测器和数据掩码共同降低坍塌风险，但不能单独给出“不发生坍塌”的数学保证。

## 基于 Transformer 的架构映射

在现代实现中，上述的向量表示通常会被延展为 Transformer 架构下的序列张量。假设输入图像通过 ViT（Vision Transformer）被切分为 $N$ 个不重叠的图块（Patch），时间窗口长度为 $T$。

<div align="center">

<img src="/figures/07-robot-policy/source/09-vla-jepa-wam/wam-fig5.png" alt="级联 WAM 的显式动作、隐式动作与几何提取三种结构承担不同接口角色。" width="86%">

_图 7.9-6：级联 WAM 的显式动作、隐式动作与几何提取三种结构承担不同接口角色。 出处：[World Action Models: The Next Frontier in Embodied AI，Siyin Wang et al.，2026](https://arxiv.org/abs/2605.12090)。_

</div>

设视觉观测张量 $\mathcal{X} \in \mathbb{R}^{T \times N \times D_{patch}}$，语言嵌入张量 $\mathcal{L} \in \mathbb{R}^{M \times D_{lang}}$（其中 $M$ 为语言指令的分词长度）。
在上下文编码器 $E_\theta$ 中，视觉和语言张量经过线性投影对齐维度后，会沿着序列维度拼接。多头自注意力机制（Self-Attention）将负责在它们之间建立密集的语义交互：

$$
\mathcal{S} = \text{Softmax}\left( \frac{\mathcal{Q} \mathcal{K}^T}{\sqrt{d_k}} \right) \mathcal{V}
$$

在一种可选的交叉注意力实现中，$\mathcal{Q}$ 来自视觉词元，$\mathcal{K},\mathcal{V}$ 来自视觉与语言词元。具体来源取决于网络设计，公式本身并不限定它们。多层 Transformer 输出时空融合状态 $\mathbf{S}_t \in \mathbb{R}^{N \times D_s}$。
同理，预测器 $P_\phi$ 也是一个多层因果 Transformer。连续动作向量 $\mathbf{a}_t$ 首先经过多层感知机（MLP）编码为动作 Token，并作为额外的序列元素前置于状态序列 $\mathbf{S}_t$。因果掩码（Causal Mask）保证了模型只能严格利用 $t$ 及之前的状态和动作来推演 $t+1$ 时刻的隐状态分布。

## 代码实现

下面的代码只保留 MLP 编码器、动作条件预测器和 EMA 更新，以便看清计算图。它不是前文 Transformer 架构的复现。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class VisionLanguageEncoder(nn.Module):
    """
    视觉-语言上下文编码器。
    在工业级实现中，此处通常是一个完整的 ViT 与跨模态 Transformer。
    为体现核心数学逻辑，这里采用深度多层感知机来模拟非线性特征提取与特征融合。
    """
    def __init__(self, img_dim=1024, lang_dim=512, latent_dim=256):
        super().__init__()
        # 将不同模态的特征映射到统一的语义维度
        self.img_proj = nn.Linear(img_dim, latent_dim)
        self.lang_proj = nn.Linear(lang_dim, latent_dim)

        # 融合网络，等价于注意力机制中的特征深度非线性混合
        self.fusion_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward(self, x, l):
        # x 形状: [Batch, img_dim], l 形状: [Batch, lang_dim]
        # [通过 GELU 激活函数引入非线性并统一维度]
        x_feat = F.gelu(self.img_proj(x))
        l_feat = F.gelu(self.lang_proj(l))

        # [沿特征维度拼接，进入多层感知机完成联合特征嵌入]
        fused = torch.cat([x_feat, l_feat], dim=-1)
        s_t = self.fusion_mlp(fused)
        # 输出形状: [Batch, latent_dim]
        return s_t

class ActionPredictor(nn.Module):
    """
    动作条件预测器 (Predictor $P_\\phi$)。
    在隐空间内根据当前状态和给定的物理动作，计算前向演变方程。
    """
    def __init__(self, latent_dim=256, act_dim=64):
        super().__init__()
        # 动作向量投影
        self.act_proj = nn.Linear(act_dim, latent_dim)

        # 深层前馈预测网络
        self.predictor_net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward(self, s_t, a_t):
        # [将低维物理动作投影到与视觉状态相同的高维语义空间]
        a_feat = F.gelu(self.act_proj(a_t))

        # [拼接状态与动作以进行未来状态的前向推演]
        combined = torch.cat([s_t, a_feat], dim=-1)
        s_t_next_pred = self.predictor_net(combined)
        return s_t_next_pred

class VLA_JEPA(nn.Module):
    """
    完整的 VLA-JEPA 顶层架构，封装编码器、预测器与不对称的动量更新机制。
    """
    def __init__(self, img_dim=1024, lang_dim=512, act_dim=64, latent_dim=256, ema_tau=0.996):
        super().__init__()
        self.ema_tau = ema_tau

        # 1. 初始化上下文编码器 \theta (在线网络，接受梯度)
        self.context_encoder = VisionLanguageEncoder(img_dim, lang_dim, latent_dim)

        # 2. 初始化预测器 \phi (在线网络，接受梯度)
        self.predictor = ActionPredictor(latent_dim, act_dim)

        # 3. 初始化目标编码器 \bar{\theta} (目标网络，脱离计算图)
        # 初始时刻采用完全一致的权重
        self.target_encoder = copy.deepcopy(self.context_encoder)

        # [在整个训练周期内冻结目标编码器的参数，强制关闭自动求导]
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_encoder(self):
        """
        核心函数：执行目标编码器的指数移动平均 (EMA) 更新。
        数学对应公式：\bar{\theta} <- \tau \bar{\theta} + (1 - \tau) \theta
        该方法应在每个训练 Step 后立刻被调用。
        """
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(self.ema_tau).add_((1 - self.ema_tau) * param_q.data)

    def forward(self, x_t, x_t_next, l, a_t):
        """
        前向传播计算图：
        x_t: 当前时刻图像
        x_t_next: 未来/下一时刻真实图像
        l: 任务语言指令
        a_t: 当前执行的动作
        """
        # [利用在线上下文编码器，提取当前状态的隐式表达]
        s_t = self.context_encoder(x_t, l)

        # [利用在线预测器推演下一时刻的预测表达]
        s_t_next_pred = self.predictor(s_t, a_t)

        # [开启无梯度上下文，通过历史累计网络提取真实的未来目标表达]
        with torch.no_grad():
            s_t_next_target = self.target_encoder(x_t_next, l)

        # 计算在 D_s 维度上的均方误差
        loss = F.mse_loss(s_t_next_pred, s_t_next_target)

        return loss, s_t_next_pred, s_t_next_target
```

## 小结

本节的 VLA-JEPA 是一个教学组合：上下文编码器从图像与语言得到状态表征，动作条件预测器预测未来表征，目标编码器由 EMA 更新。表征空间预测可以避免逐像素重构，但不会自动得到“物理本质”，也不会自动滤除全部背景。EMA、停止梯度、预测器和数据设计共同降低坍塌风险；最终仍要用下游控制、表征探针和反事实测试验证模型学到了什么。
