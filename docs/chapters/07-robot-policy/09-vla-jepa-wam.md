# 基于 JEPA 的视觉-语言-动作模型（VLA-JEPA/WAM）

在具身智能（Embodied AI）与机器人控制的演进历程中，如何让系统不仅能“感知”当前状态，还能理解物理世界的运作规律并据此做出精准的“动作预测”，一直是学术界的核心难题。传统端到端行为克隆（Behavior Cloning）往往缺乏对环境未来动态的前瞻能力，而基于生成式模型（如扩散模型或自动编码器）的世界模型则倾向于在像素级别进行严苛的重构验证。然而，真实世界充满了高度的偶然认知不确定性（Aleatoric Uncertainty）——风吹动的树叶、水面的反光、相机镜头的噪点。将宝贵的网络容量和算力浪费在预测这些与任务本身毫无关联的视觉高频细节上，从信息论的角度看是极为低效的。

2022 年，Yann LeCun 系统阐述了联合嵌入预测架构（Joint-Embedding Predictive Architecture, JEPA）[[LeCun, 2022]](https://openreview.net/forum?id=BZ5a1r-kVsf)，主张在表征空间中预测与任务有关的信息。V-JEPA 随后把这一思路用于视频表征学习 [[Bardes et al., 2024]](https://arxiv.org/abs/2404.08471)。把预测状态与生成动作结合的机器人模型仍是快速演进中的研究范式；“World Action Model（WAM）”是对多种相关架构的统称，而不是 V-JEPA 论文定义的单一标准模型 [[Wang et al., 2026]](https://arxiv.org/abs/2605.12090)。本节中的“VLA-JEPA”因此表示一种教学性的组合设计，不声称复现某篇同名论文。

本节将从经典力学中的系统状态变量建模起步，严格推导 VLA-JEPA 的数学体系与优化博弈机制，并展示如何用这一架构为机器人构建一个既严谨又极其高效的策略大脑。

## 从经典力学状态空间到隐变量抽象

为了深刻理解 JEPA 在数学上所追求的终极目标，我们暂且抛开深层神经网络的参数矩阵，回到经典力学中最基础的系统状态建模。

在牛顿力学中，假设我们在考察一个质点在三维空间中的抛体运动。尽管质点可能由数以亿计的原子构成，或者它在不同的光照下呈现出截然不同的视觉反光，但我们要预测它在时间 $\Delta t$ 之后的空间演变，只需要抽取极少数的几个本质物理量：初始位置向量 $\mathbf{p}_0 \in \mathbb{R}^3$ 和速度向量 $\mathbf{v}_0 \in \mathbb{R}^3$。

在外力向量 $\mathbf{F}$（我们可以将其视为系统接收到的“动作” $a$）和重力加速度 $\mathbf{g}$ 的作用下，质点的新状态可以通过严密的动力学方程组直接求得：

$$
\mathbf{p}_1 = \mathbf{p}_0 + \mathbf{v}_0 \Delta t + \frac{1}{2} \left( \mathbf{g} + \frac{\mathbf{F}}{m} \right) \Delta t^2
$$

在该公式中，$(\mathbf{p}_0, \mathbf{v}_0, m)$ 构成了这个物理系统的一组**完备的隐状态抽象**。在这个低维度的流形（Manifold）上，未来的演变仅仅是当前状态与外部动作的代数函数，而无需关心质点表面的微观纹理。

在具身机器人的 VLA 任务中，我们面临着完全等价的拓扑映射挑战。机器人无法直接获取环境完美的物理状态向量。它只能接收到维度极高的传感器观测数据矩阵 $\mathbf{X}_t \in \mathbb{R}^{H \times W \times C}$（高度、宽度及颜色通道）。同时，其任务意图并不总是简单的力学方程，而是由高维离散的自然语言指令序列 $l$ 所定义。

本节构造的教学模型使用深层神经网络把观测 $\mathbf{X}_t$ 映射到低维向量，并在该空间内做动作条件预测。这个向量是否真的等价于完整物理状态，必须通过探针、下游控制与反事实实验验证，不能由网络结构本身保证。

## VLA-JEPA 的数学体系基础

为了实现上述构想，VLA-JEPA 定义了一套由三个核心神经网络模块组成的数学闭环：上下文编码器（Context Encoder）$E_\theta$、目标编码器（Target Encoder）$E_{\bar{\theta}}$ 和预测器（Predictor）$P_\phi$。

### 1. 联合空间映射与上下文编码

给定时刻 $t$ 的图像观测 $\mathbf{X}_t \in \mathbb{R}^{H \times W \times C}$ 与语言指令序列向量 $\mathbf{l} \in \mathbb{R}^{D_l}$。上下文编码器 $E_\theta$ 需要捕捉图像的几何空间布局，并根据语言指令的语义焦点，提取出当前时刻的隐状态表达 $\mathbf{s}_t$：

$$
\mathbf{s}_t = E_\theta(\mathbf{X}_t, \mathbf{l}), \quad \mathbf{s}_t \in \mathbb{R}^{D_s}
$$

这里，$D_s$ 表示隐空间的特征维度。$\mathbf{s}_t$ 是一个高度浓缩的张量，它剔除了所有无助于任务规划的背景高频噪声，仅保留了物体的位姿、机械臂的空间距离以及语义交互焦点。

类似地，环境在下一个时间步（或未来第 $k$ 步）的真实观测 $\mathbf{X}_{t+1}$ 被馈入目标编码器 $E_{\bar{\theta}}$，产生目标隐状态 $\bar{\mathbf{s}}_{t+1}$：

$$
\bar{\mathbf{s}}_{t+1} = E_{\bar{\theta}}(\mathbf{X}_{t+1}, \mathbf{l}), \quad \bar{\mathbf{s}}_{t+1} \in \mathbb{R}^{D_s}
$$

(**请严谨区分目标编码器参数 $\bar{\theta}$ 与上下文编码器参数 $\theta$**)。在 VLA-JEPA 中，由于其非对比学习的底层逻辑，$\bar{\theta}$ 绝不是通过梯度反向传播直接优化的独立变量。

### 2. 隐空间上的预测闭环

当系统获取了当前时刻的隐状态 $\mathbf{s}_t$ 后，机器人执行了特定的多维连续动作 $\mathbf{a}_t \in \mathbb{R}^{D_a}$（例如 $SE(3)$ 空间内的末端执行器位姿增量与夹爪开合度）。预测器 $P_\phi$ 则充当了该公式中物理方程的角色，负责在隐空间内进行时间向前的动态推演：

$$
\hat{\mathbf{s}}_{t+1} = P_\phi(\mathbf{s}_t, \mathbf{a}_t), \quad \hat{\mathbf{s}}_{t+1} \in \mathbb{R}^{D_s}
$$

## 打破特征坍缩博弈：不对称结构与指数移动平均

在定义了架构之后，我们自然需要衡量预测与真实目标之间的差距。在欧几里得隐空间中，最直接的损失度量是预测向量 $\hat{\mathbf{s}}_{t+1}$ 与目标向量 $\bar{\mathbf{s}}_{t+1}$ 之间的均方误差（MSE）：

$$
\mathcal{L}_{MSE}(\theta, \phi) = \frac{1}{D_s} \sum_{i=1}^{D_s} \left( \hat{\mathbf{s}}_{t+1}^{(i)} - \bar{\mathbf{s}}_{t+1}^{(i)} \right)^2 = \left\| P_\phi(E_\theta(\mathbf{X}_t, \mathbf{l}), \mathbf{a}_t) - E_{\bar{\theta}}(\mathbf{X}_{t+1}, \mathbf{l}) \right\|_2^2
$$

然而，在该公式中潜藏着一个致命的优化陷阱，学术界称之为**特征坍缩（Feature Collapse）**。

由于 $E_\theta$, $E_{\bar{\theta}}$ 和 $P_\phi$ 都是参数化的、可自由更新的神经网络算子，如果允许网络通过梯度下降自由地最小化该损失，优化器将会迅速发现一条最省力的拓扑“捷径”：使得编码器对于任意输入的 $\mathbf{X}$ 均输出零向量（即 $\mathbf{s}_t = \mathbf{0}, \bar{\mathbf{s}}_{t+1} = \mathbf{0}$），且预测器恒等输出零向量。此时损失函数瞬间归零，但整个特征空间坍缩到了一个奇异的质点上，失去了任何表达环境演变的能力。

> 💡 **不对称知识蒸馏的博弈论视角**
>
> 我们可以将这种架构看作一场师生间的知识博弈。如果教师（目标编码器）和学生（上下文编码器与预测器）同时看着同一本无字的答案书，并且都被允许修改答案书的内容来证明“我们达成了一致”，那么最快达成一致的方法就是双双把书撕掉（坍缩为零）。为了防止这种情况，必须强制要求“教师的知识必须是过去历史中稳定积累的，且在此刻不可被随意篡改”。

为降低退化到平凡解的风险，本节采用 I-JEPA 风格的不对称参数更新。**上下文编码器 $\theta$ 与预测器 $\phi$ 通过梯度反向传播更新：**

$$
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_{MSE}, \quad \phi \leftarrow \phi - \eta \nabla_\phi \mathcal{L}_{MSE}
$$

而**目标编码器 $\bar{\theta}$ 必须通过上下文编码器历史参数的指数移动平均（Exponential Moving Average, EMA）进行平滑更新**，其更新过程彻底阻断了由于当前 Batch 的 MSE 损失所产生的任何即时梯度：

$$
\bar{\theta} \leftarrow \tau \bar{\theta} + (1 - \tau) \theta
$$

其中衰减率参数 $\tau \in [0.99, 1)$。$\bar{\theta}$ 提供随时间缓慢变化的目标，经验上有助于稳定训练。它与停止梯度、预测器和数据掩码共同降低坍塌风险，但不能单独给出“不发生坍塌”的数学保证。

## 基于 Transformer 的架构映射

在现代实现中，上述的向量表示通常会被延展为 Transformer 架构下的序列张量。假设输入图像通过 ViT（Vision Transformer）被切分为 $N$ 个不重叠的图块（Patch），时间窗口长度为 $T$。

设视觉观测张量 $\mathcal{X} \in \mathbb{R}^{T \times N \times D_{patch}}$，语言嵌入张量 $\mathcal{L} \in \mathbb{R}^{M \times D_{lang}}$（其中 $M$ 为语言指令的分词长度）。
在上下文编码器 $E_\theta$ 中，视觉和语言张量经过线性投影对齐维度后，会沿着序列维度拼接。多头自注意力机制（Self-Attention）将负责在它们之间建立密集的语义交互：

$$
\mathcal{S} = \text{Softmax}\left( \frac{\mathcal{Q} \mathcal{K}^T}{\sqrt{d_k}} \right) \mathcal{V}
$$

其中 $\mathcal{Q}$ 通常来自于视觉 Token，而 $\mathcal{K}, \mathcal{V}$ 则由视觉与语言 Token 的联合拼接提供。经过多层 Transformer 块处理后，输出的时空融合状态 $\mathbf{S}_t \in \mathbb{R}^{N \times D_s}$ 成为隐空间的严谨表达。
同理，预测器 $P_\phi$ 也是一个多层因果 Transformer。连续动作向量 $\mathbf{a}_t$ 首先经过多层感知机（MLP）编码为动作 Token，并作为额外的序列元素前置于状态序列 $\mathbf{S}_t$。因果掩码（Causal Mask）保证了模型只能严格利用 $t$ 及之前的状态和动作来推演 $t+1$ 时刻的隐状态分布。

## 代码实现

现在，让我们利用严谨的面向对象思想，将 VLA-JEPA 的上述数学推导一一落实到代码中。注意我们如何利用 `torch.no_grad()` 阻断目标编码器的梯度，并显式实现参数的 EMA 动量更新。

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

在本节中，我们详尽推导了基于 JEPA 架构的**视觉-语言-动作模型**。不同于传统的端到端控制或者深度依赖像素重构的扩散生成模型，VLA-JEPA 坚守了“在高度抽象的隐空间内预测事物物理本质”的设计信念。通过精心引入参数的**不对称 EMA 更新机制**，该架构从纯数学角度严谨地规避了表征空间的**特征坍缩**问题，构建了一个极具鲁棒性的世界模型。对于面临复杂随机环境的具身智能体而言，这种机制使其能够果断剔除无关的高频环境噪声，将最为核心的网络计算资源全部投入到环境语义与系统动力学的核心规律学习中。
