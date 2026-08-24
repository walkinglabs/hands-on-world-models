# 2.2　图像编码器

> **从原始像素到结构化表征**
>
> 物理世界投射在传感器上的原始观测是一组高维、稠密且充斥着感知冗余的 RGB 像素阵列。以一张看似微不足道的 $64 \times 64 \times 3$ 像素图像为例，其单步输入维度便达到惊人的 $12{,}288$ 维。若直接在原始像素网格上建模物理转移，不仅会陷入无法逾越的“维度灾难”，更会被背景噪点、光影闪烁等无关高频扰动所淹没，掩盖真正支配系统演化的低维物理守恒量与运动微分。
>
> 空间视觉编码器（Spatial Visual Encoder）是世界模型感知大厦的基石：它的使命是将高维几何与外观观测压缩映射至紧凑的潜在状态空间（Latent Space），在剔除无关表观冗余的同时，忠实保留物体的空间拓扑、接触边界、相对位姿以及动作响应特征。本节我们将深入剖析卷积神经网络（CNN）与视觉 Transformer（ViT）的数学本质、归纳偏置、架构权衡与表征检验准则，为构建能够支撑因果动力学推演的世界模型奠定表征基础。

---

## 本节导读

- **核心内容**：卷积算子的空间局部性与平移等变性证明；视觉 Transformer（ViT）图像分块投影与多头自注意力机制；1D/2D/RoPE/3D-Tubelet 时空位置编码代数体系；CNN 与 ViT 在世界模型中的归纳偏置与扩展法则权衡；世界模型视觉表征的三大核心诊断准则（物理状态线性探测、反事实动作敏感度、多步推演物体恒常性）。
- **核心问题**：为什么高重构保真度（PSNR/SSIM）的自编码器往往无法支撑高质量的世界模型？在样本受限的具身控制与海量预训练的通用视频生成场景下，如何根据归纳偏置与扩展性选择视觉骨干？
- **核心概念**：卷积归纳偏置（Convolutional Inductive Bias）、平移等变性（Translation Equivariance）、感受野（Receptive Field）、分块投影（Patch Projection）、多头自注意力（MHSA）、旋转位置编码（2D-RoPE）、时空管元（Spatio-Temporal Tubelet）、线性探测（Linear Probing）、反事实动作敏感度（Counterfactual Action Sensitivity）。
- **核心公式**：
  $$Y_{i,j} = \sum_{u,v} K_{u,v} P_{i,j}[u,v], \qquad f(T_{\Delta}(I)) = T_{\Delta}(f(I)), \qquad \operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V, \qquad \Delta z = \|z(o_t, a^{(1)}) - z(o_t, a^{(2)})\|$$

---

## 空间局部性、平移等变性与卷积归纳偏置

卷积神经网络（Convolutional Neural Network, CNN）统治计算机视觉数十年，其核心力量来源于对物理视觉世界的两项先验假设，即**归纳偏置（Inductive Bias）**：

1. **空间局部性（Spatial Locality）**：真实世界中相邻像素间的物理关联度（如同一刚体、连续表面）远高于远距离像素；
2. **平移不变性与平移等变性（Translation Invariance & Equivariance）**：物体本身的物理属性与几何特征不随其在画面中的绝对像素坐标变化而改变。

```text
输入特征图 X [C_in, H, W]              卷积核权重 K [C_out, C_in, K_h, K_w]        输出特征图 Y [C_out, H', W']
┌───┬───┬───┬───┐                              ┌───┬───┐                              ┌───┬───┬───┐
│ 0 │ 0 │ 1 │ 1 │                              │-1 │ 1 │                              │ 0 │ 1 │ 0 │
├───┼───┼───┼───┤   * 空间滑动局部内积加权   *   ├───┼───┤   ── 激活与下采样 ──>         ├───┼───┼───┤
│ 0 │ 0 │ 1 │ 1 │                              │-1 │ 1 │                              │ 0 │ 1 │ 0 │
└───┴───┴───┴───┘                              └───┴───┘                              └───┴───┴───┘
 局部感受野提取                                跨空间全域权重共享                       保留空间拓扑与相对位移
```

### 1. 2D 卷积算子的数学形式化

给定输入张量 $X \in \mathbb{R}^{C_{\text{in}} \times H \times W}$ 与可学习卷积核 $K \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K_h \times K_w}$，对于输出特征图 $Y \in \mathbb{R}^{C_{\text{out}} \times H' \times W'}$ 上的通道 $c'$ 及空间位置 $(i, j)$，离散 2D 卷积（严格数学定义为互相关 Cross-Correlation）定义为：

$$Y_{c', i, j} = \sum_{c=1}^{C_{\text{in}}} \sum_{u=-\lfloor K_h/2 \rfloor}^{\lfloor K_h/2 \rfloor} \sum_{v=-\lfloor K_w/2 \rfloor}^{\lfloor K_w/2 \rfloor} K_{c', c, u + \lfloor K_h/2 \rfloor, v + \lfloor K_w/2 \rfloor} \, X_{c, \, i \cdot s + u, \, j \cdot s + v} + b_{c'}$$

其中 $s$ 为步长（Stride），$b_{c'}$ 为通道偏置。

在整个特征提取过程中，张量形状经历了结构化的维度流动：
$$[B, C_{\text{in}}, H, W] \xrightarrow{\text{Conv2d}} [B, C_1, H_1, W_1] \xrightarrow{\text{Conv2d}} \cdots \xrightarrow{\text{AdaptivePool}} [B, C', 1, 1] \xrightarrow{\text{Flatten}} [B, d]$$

### 2. 平移等变性的严格证明

平移等变性是卷积网络能够作为物理世界感知器的数学支柱。

**定义（平移算子）**：设平移算子 $T_{(\Delta x, \Delta y)}$ 作用于二维图像信号 $I(x, y)$，满足：
$$[T_{(\Delta x, \Delta y)} I](x, y) = I(x - \Delta x, y - \Delta y)$$

**定理（卷积的平移等变性）**：设 $f$ 为连续或无边界截断误差的 2D 卷积映射 $f(I) = I * K$。则 $f$ 与平移算子 $T$ 严格可交换，即：
$$f\big(T_{(\Delta x, \Delta y)}(I)\big) = T_{(\Delta x, \Delta y)}\big(f(I)\big)$$

**证明**：
根据卷积定义，平移后信号的卷积响应为：

$$ \begin{aligned}
\big[f\big(T_{(\Delta x, \Delta y)}(I)\big)\big](x, y) &= \iint_{\mathbb{R}^2} \big[T_{(\Delta x, \Delta y)}(I)\big](u, v) \, K(x - u, y - v) \, du \, dv \\
&= \iint_{\mathbb{R}^2} I(u - \Delta x, v - \Delta y) \, K(x - u, y - v) \, du \, dv
\end{aligned}$$
引入积分变量代换：令 $u' = u - \Delta x$，$v' = v - \Delta y$，则 $u = u' + \Delta x$，$v = v' + \Delta y$。代入上式得：
$$\begin{aligned}
\big[f\big(T_{(\Delta x, \Delta y)}(I)\big)\big](x, y) &= \iint_{\mathbb{R}^2} I(u', v') \, K\big((x - \Delta x) - u', (y - \Delta y) - v'\big) \, du' \, dv' \\
&= [f(I)](x - \Delta x, y - \Delta y) \\
&= \big[T_{(\Delta x, \Delta y)}\big(f(I)\big)\big](x, y) \quad \blacksquare
\end{aligned}$$

**物理意义**：当一辆自动驾驶汽车在视野中由坐标 $(x_1, y_1)$ 移动到 $(x_2, y_2)$ 时，CNN 浅层与中层特征图上的激活峰值发生**严格同步的空间平移**，而特征激活的模式与强度保持不变。这一性质天然契合刚体运动学与几何光学变换。

### 3. 感受野递推与多尺度抽象

在深层 CNN 中，每个神经元对原始输入图像的响应区域称为**感受野（Receptive Field, RF）**。设第 $l$ 层卷积核大小为 $K_l$，步长为 $s_l$，则第 $l$ 层特征相对于输入图像的感受野递推关系为：

$$RF_0 = 1, \qquad RF_l = RF_{l-1} + (K_l - 1) \times S_{l-1}, \quad \text{其中 } S_{l-1} = \prod_{k=1}^{l-1} s_k \quad (S_0 = 1)$$

```text
Layer 1 (3x3 Kernel, s=2) ──> RF = 3,  S_1 = 2 (感知微观边缘、角点、色块)
Layer 2 (3x3 Kernel, s=2) ──> RF = 7,  S_2 = 4 (感知物体局部部件、纹理边界)
Layer 3 (3x3 Kernel, s=2) ──> RF = 15, S_3 = 8 (感知完整物体轮廓、空间相对位姿)
```

### 4. 为什么强归纳偏置加速小样本强化学习

在基于世界模型的强化学习（如 Dreamer 系列、CarRacing、Atari 100k 评测）中，智能体通常只能探索数万到数十万步环境交互。

在这种**极度受限的数据体制（Low-Sample Regime）**下：
- 卷积权重的空间共享将自由参数量从全连接层的 $\mathcal{O}(H^2 W^2)$ 骤降至 $\mathcal{O}(K^2)$；
- 空间局部性强加了硬约束，使得模型**无需消耗宝贵样本去学习“相邻像素在几何上是相邻的”这一公理**；
- 模型假设空间（Hypothesis Space）被极大压缩，动力学梯度能够迅速穿透感知瓶颈，在极短步数内建立起对物体位移与碰撞的准确物理推演。

---

## 图像分块投影与视觉自注意力机制

当环境数据规模呈指数级增长，或者任务需要将视觉观测与自然语言指令、离散控制动作无缝协同处理时，CNN 固有的局部性偏置转而成为表征容量的限制。视觉 Transformer（Vision Transformer, ViT）通过**图像分块投影（Patch Projection）**将连续二维图像离散化为一维 Token 序列，并使用自注意力机制实现全局上下文交互。

```text
原始图像 x [B, C, H, W]              Patch 展平与线性投影               Token 序列输入 Transformer
┌────────┬────────┐
│ Patch1 │ Patch2 │              [Patch1, Patch2, Patch3, Patch4]
├────────┼────────┤    ───>            │       │       │       │       ───>  [CLS] + [E_pos] ──> Self-Attention
│ Patch3 │ Patch4 │                    ▼       ▼       ▼       ▼
└────────┴────────┘                 [  x_p^1 W_E,  x_p^2 W_E, ...  ]
(分辨率 H x W, 分块 P x P)            (N = HW / P^2 个 D 维 Token)
```

### 1. 分块线性投影（Patch Projection）

设输入图像为 $x \in \mathbb{R}^{B \times C \times H \times W}$，选择分块空间分辨率为 $P \times P$（常见 $P = 16$ 或 $P = 8$）。
图像被切分为 $N$ 个不重叠的局部块：
$$N = \frac{H \cdot W}{P^2}$$

每个图像块展平为一维向量 $x_p^i \in \mathbb{R}^{P^2 C}$。随后通过可学习线性投影矩阵 $W_E \in \mathbb{R}^{(P^2 C) \times D}$ 将其映射为 $D$ 维隐向量（在工程上通常使用卷积核为 $P \times P$、步长为 $P$ 的 `Conv2d` 层高效实现）：

$$z_0 = \Big[ x_{\text{class}} \,;\, x_p^1 W_E \,;\, x_p^2 W_E \,;\, \dots \,;\, x_p^N W_E \Big] + E_{\text{pos}}$$

其中 $x_{\text{class}} \in \mathbb{R}^{1 \times D}$ 为可学习的全局汇总标记（Class Token），$E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ 为空间位置编码。

### 2. 多头自注意力机制（Multi-Head Self-Attention, MHSA）

在得到输入序列 $z \in \mathbb{R}^{(N+1) \times D}$ 后，第 $l$ 层的多头自注意力计算如下：

对每个注意力头 $m \in \{1, \dots, h\}$，投影得到查询（Query）、键（Key）和值（Value）矩阵（维度 $d_k = D / h$）：
$$Q_m = z W_Q^{(m)}, \quad K_m = z W_K^{(m)}, \quad V_m = z W_V^{(m)} \qquad \big(W_Q^{(m)}, W_K^{(m)}, W_V^{(m)} \in \mathbb{R}^{D \times d_k}\big)$$

单头缩放点积注意力公式为：
$$\operatorname{Attention}(Q_m, K_m, V_m) = \operatorname{softmax}\left(\frac{Q_m K_m^T}{\sqrt{d_k}}\right) V_m$$

将所有头的输出在特征维度拼接并通过输出投影矩阵 $W_O \in \mathbb{R}^{D \times D}$：
$$\operatorname{MHSA}(z) = \operatorname{Concat}\big(\operatorname{head}_1, \operatorname{head}_2, \dots, \operatorname{head}_h\big) W_O$$

### 3. 计算复杂度与感受野分析

| 架构特性 | 卷积神经网络（CNN） | 视觉 Transformer（ViT） |
| :--- | :--- | :--- |
| **单层计算复杂度** | $\mathcal{O}\big(K^2 \cdot H \cdot W \cdot C_{\text{in}} \cdot C_{\text{out}}\big)$（对分辨率 $HW$ 严格**线性**） | $\mathcal{O}\big(N^2 \cdot D + N \cdot D^2\big) = \mathcal{O}\big(\frac{H^2 W^2}{P^4} D + \frac{HW}{P^2} D^2\big)$（对 Token 数 $N$ 呈**二次方**） |
| **层感受野机制** | 随网络深度逐层线性扩张（局部性） | **第一层（Layer 1）即具备全图全局感受野** |
| **空间交互方式** | 静态卷积核滑动，权重与输入内容无关 | 动态计算两两 Token 之间的点积注意力权重 |

**物理建模优势**：由于 ViT 在第一层就具备全局交互能力，当机器人操作台左上角的机械臂末端接触物体时，画面右下角支撑座或影子产生的应变可以在单层内被直接捕捉，而无需等待数十层卷积的逐级感受野传递。

### 4. 多模态 Token 统一接口

在现代具身智能（VLA）与交互式世界模型中，ViT 最具革命性的优势在于其**天然的 Token 化接口**。无论是图像、动作还是任务指令，均可统一投影为相同的 $D$ 维向量并拼接到单条序列中：

$$\text{Sequence} = \big[ \underbrace{z_{\text{patch}}^{(1)}, \dots, z_{\text{patch}}^{(N)}}_{\text{视觉观测 } o_t}, \quad \underbrace{z_{\text{act}}}_{\text{控制动作 } a_t}, \quad \underbrace{z_{\text{text}}^{(1)}, \dots, z_{\text{text}}^{(M)}}_{\text{任务指令 } l} \big]$$

世界模型因此退化为一个统一的序列因果建模器（如 Genie、RT-2、OpenVLA），无需设计异构的多模态交叉融合算子。

---

## 空间与时空位置编码代数

自注意力算子本质上是作用于无序集合（Set）的置换等变算子（Permutation Equivariant Operator）：
$$\operatorname{Attention}(\mathbf{P}Q, \mathbf{P}K, \mathbf{P}V) = \mathbf{P} \operatorname{Attention}(Q, K, V)$$
若不对输入 Token 显式注入几何坐标，打乱所有图像 Patch 的排列顺序，Transformer 输出的表征集合完全不变。这对于依赖空间相对距离与刚体几何的世界模型是致命的。

```text
[1D 可学习嵌入]           [2D 正弦坐标编码]                     [3D 时空管元 (Tubelet)]
Token_k + E_pos[k]        x_coord ──> Sin/Cos ──┐               ┌───┬───┬───┐ (t=0)
                          y_coord ──> Sin/Cos ──┴──> Concat     ├───┼───┼───┤ (t=1)
(简单但无法外推分辨率)     (解析相对位移不变性)                   └───┴───┴───┘ (t=2, 3D 卷积下采样)
```

### 1. 1D 可学习位置编码 vs 2D 正弦空间编码

- **1D 可学习位置编码（1D Learnable Embedding）**：
  为每个 Token 索引 $k \in \{1, \dots, N\}$ 分配一个独立的参数向量 $E_{\text{pos}}[k] \in \mathbb{R}^D$。其缺点在于：它将二维图像强行压平为一维序号，破坏了垂直方向的邻接结构，且无法泛化到不同分辨率的观测图像。
- **2D 正弦空间位置编码（2D Sinusoidal Embedding）**：
  显式保留图像块的网格坐标 $(i, j)$，其中 $i \in \{1, \dots, H/P\}$，$j \in \{1, \dots, W/P\}$。分别用不同频率的正弦和余弦基函数编码水平与垂直坐标：
  $$PE_{(i, j), 2k} = \sin\left(\frac{i}{\tau^{4k/D}}\right), \quad PE_{(i, j), 2k+1} = \cos\left(\frac{i}{\tau^{4k/D}}\right)$$
  $$PE_{(i, j), D/2 + 2k} = \sin\left(\frac{j}{\tau^{4k/D}}\right), \quad PE_{(i, j), D/2 + 2k+1} = \cos\left(\frac{j}{\tau^{4k/D}}\right)$$
  这种编码具有解析的相对平移性质：任意两个位置编码的内积 $\langle PE_{(i_1, j_1)}, PE_{(i_2, j_2)} \rangle$ 仅是相对位移 $(i_1 - i_2, j_1 - j_2)$ 的函数。

### 2. 二维旋转位置编码（2D-RoPE）

旋转位置编码（Rotary Position Embedding, RoPE）通过正交旋转矩阵将坐标直接乘进 Query 和 Key 向量中。在二维视觉世界模型中，将单头注意力向量（维度 $d_k$）划分为对应横纵坐标的两个独立子空间（各占 $d_k/2$ 维）：

$$\tilde{q}_{(i, j)} = \mathbf{R}_{\Theta, (i, j)} q = \begin{bmatrix} \mathbf{R}_{\Theta_x, i} & \mathbf{0} \\ \mathbf{0} & \mathbf{R}_{\Theta_y, j} \end{bmatrix} \begin{bmatrix} q_x \\ q_y \end{bmatrix}$$

其中二维旋转矩阵为分块正交矩阵（$\theta_m = 10000^{-4(m-1)/d_k}$）：
$$\mathbf{R}_{\Theta_x, i} = \operatorname{diag}\left( \begin{bmatrix} \cos(i\theta_1) & -\sin(i\theta_1) \\ \sin(i\theta_1) & \cos(i\theta_1) \end{bmatrix}, \dots, \begin{bmatrix} \cos(i\theta_{d_k/4}) & -\sin(i\theta_{d_k/4}) \\ \sin(i\theta_{d_k/4}) & \cos(i\theta_{d_k/4}) \end{bmatrix} \right)$$

此时，Query 与 Key 在计算注意力得分时的内积为：
$$\begin{aligned}
\langle \tilde{q}_{(i_1, j_1)}, \tilde{k}_{(i_2, j_2)} \rangle &= q_x^T \mathbf{R}_{\Theta_x, i_1}^T \mathbf{R}_{\Theta_x, i_2} k_x + q_y^T \mathbf{R}_{\Theta_y, j_1}^T \mathbf{R}_{\Theta_y, j_2} k_y \\
&= q_x^T \mathbf{R}_{\Theta_x, i_2 - i_1} k_x + q_y^T \mathbf{R}_{\Theta_y, j_2 - j_1} k_y \\
&= \sum_{m=1}^{d_k/4} \operatorname{Re}\left( \mathbf{q}_{x, m} \mathbf{k}_{x, m}^* e^{\mathrm{i}(i_2 - i_1)\theta_m} \right) + \sum_{m=1}^{d_k/4} \operatorname{Re}\left( \mathbf{q}_{y, m} \mathbf{k}_{y, m}^* e^{\mathrm{i}(j_2 - j_1)\theta_m} \right)
\end{aligned}$$
其中 $\mathbf{q}_{x, m} = q_{x, 2m-1} + \mathrm{i} q_{x, 2m}$ 为二维实子向量在复数域的同构表示。

**关键特性**：注意力得分完全取决于相对空间距离 $\Delta x = i_2 - i_1$ 和 $\Delta y = j_2 - j_1$，且随着相对欧氏距离增大，内积自动出现高频振荡衰减，天然契合物理空间交互的局部性与外推性。

### 3. 3D 时空管元嵌入（3D Spatio-Temporal Tubelet Embeddings）

对于连续视频观测 $V \in \mathbb{R}^{B \times T \times C \times H \times W}$（如 Video-JEPA、Genie、Sora），若按单帧切片会导致序列长度爆炸。

现代视频世界模型采用 **3D 时空管元（Tubelet）** 投影：
使用核大小为 $T_p \times P \times P$、步长为 $T_p \times P \times P$ 的 3D 卷积算子，将时空体素块直接映射为一个时空 Token：
$$N_{\text{tokens}} = \left(\frac{T}{T_p}\right) \times \left(\frac{H}{P}\right) \times \left(\frac{W}{P}\right)$$
每个 Token 被赋予 3D 时空坐标 $(t, i, j)$，其位置编码由一维时间位置与二维空间位置正交组合而成，直接捕捉跨帧刚体运动轨迹与时间因果连续性。

---

## 卷积网络与视觉 Transformer 的架构权衡

在为世界模型选择视觉编码骨干时，不存在绝对的优劣，而是取决于**数据规模、推理延迟预算与多模态扩展性**之间的帕累托权衡。

```text
表征能力 / 拟合上限
   ▲                                              ViT (海量数据与大参数量下超越上限)
   │                                            . - '
   │                                      . - '   /
   │                       CNN          . - '    /  (无归纳偏置束缚，Scaling 优势显现)
   │                  . - - - - - - - - - - - - /
   │              . -'                         /
   │          . -'                            /
   │       .-'                               /
   │   .-'                                  /
   │.-'  (强归纳偏置，小样本收敛极快)      /
   └─────────────────────────────────────────► 训练数据规模 / 预训练交互步数
```

### 1. 归纳偏置与扩展法则（Scaling Laws）的博弈

- **小样本 / 低交互量阶段（$< 10^6$ 步）**：CNN 依靠局部连接与平移等变性，在极少样本下即可抑制过拟合，性能显著优于未经预训练的 ViT。
- **大规模预训练阶段（$> 10^8$ 帧）**：ViT 极弱的归纳偏置转变为强大的表征自由度，模型容量随参数量和数据量的增长表现出优异的幂律扩展能力（Power-law Scaling），彻底打破 CNN 的拟合天花板。

### 2. 具身控制低延迟 vs 大规模跨域泛化

- **边缘端与实时闭环控制（$\ge 50\text{ Hz}$）**：在机械臂力控、四足机器人敏捷运动等场景中，端到端延迟必须控制在 $20\text{ ms}$ 以内。轻量 ResNet/ConvNet 具有固定的计算图结构、零 Attention 显存开销、极佳的 TensorRT 加速亲和力；
- **跨域通用世界模型（Foundation World Models）**：需要融合跨机器人、自动驾驶、互联网视频等多源异构数据，ViT 凭借统一的 Token 抽象成为唯一能承载基础大模型的骨干底座。

### 3. 混合架构（Hybrid Architectures）的演进

现代世界模型越来越多地采用动静结合的混合架构：
- **ConvNeXt 设计理念**：借鉴 ViT 的 Patch 投影、反向瓶颈结构（Inverted Bottleneck）与 $7 \times 7$ 大深度可分离卷积，在保持纯卷积高效推理的同时获得匹敌 ViT 的感受野与性能；
- **CNN-ViT 级联网络（ResNet-ViT Backbone）**：在输入端使用 2~3 层下采样卷积过滤高频无用噪点并压缩空间分辨率，随后接入 Transformer 模块建模全局因果与动作交互，兼顾局部计算效率与全局视野。

### 4. 架构全维度对比矩阵

| 评估维度 | 经典卷积网络（CNN / ResNet） | 视觉 Transformer（ViT） | 现代混合骨干（ConvNeXt / ResNet-ViT） |
| :--- | :--- | :--- | :--- |
| **空间归纳偏置** | 极强（局部性、平移等变性） | 极弱（仅通过位置编码注入坐标） | 中等（局部滤波与全局注意力和合） |
| **计算复杂度** | $\mathcal{O}(HW \cdot K^2)$ | $\mathcal{O}(N^2 \cdot D)$ | $\mathcal{O}(HW \cdot K^2) + \mathcal{O}(N_{\text{coarse}}^2 D)$ |
| **小样本数据效率** | 极高（抗过拟合能力强） | 较低（依赖自监督预训练或海量数据） | 高 |
| **多模态扩展性** | 较差（需要专门的交叉融合头） | **极佳（一维 Token 统一表示）** | 良好 |
| **实时控制延迟** | 极低（$< 5\text{ ms}$） | 中等（受序列长度二次方限制） | 低（$5 \sim 15\text{ ms}$） |
| **代表世界模型** | DreamerV1-V2, PlaNet, MuZero | V-JEPA, Genie, GAIA-1, OpenVLA | DreamerV3, TransDreamer, UniSim |

---

## 世界模型视觉表征的核心检验准则

在无监督表征学习中，最常见的直觉误区是：**“重构保真度越高（PSNR/SSIM 越高、像素 MSE 越低），提取的视觉表征越优秀”**。这一观点在世界模型中是**完全错误**的。

```text
原始画面 [树叶沙沙 + 远处云朵 + 微小红球]
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
[像素级自编码器 (Pixel MSE)]  [物理因果编码器 (World Model Latent)]
  - 85% 容量用于拟合树叶纹理       - 剔除高频风吹草动与环境光影
  - 10% 容量用于拟合云朵流动       - 纯净保留红球位置 (x, y) 与运动微分 (v_x, v_y)
  - 仅剩 5% 容量给关键小红球       - 线性可分流形，严格因果可控
  (重构图像极高保真，但物理动力学崩溃) (下游 MPC 规划与动作推演零误差)
```

### 1. 为什么重构损失会诱导“感知过拟合”

以经典驾驶场景为例，画面背景中的树叶摆动、路面沥青噪点、天空云层占据了整幅图像 $90\%$ 以上的高频像素方差。像素级重构损失（Pixel Reconstruction Loss）会强迫网络将绝大部分表征容量分配给这些与车辆运动学无关的高频纹理；相反，控制决策所依赖的核心变量——前方行人的微小相对位移、红绿灯的亮灭状态，仅占几个像素。在 MSE 损失驱动下，编码器极易丢失这些决定生死存亡的低能量物理信号。

因此，世界模型的视觉表征必须通过以下三大**诊断性探针（Diagnostic Probing Tasks）**进行严格验证：

### 2. 诊断准则 1：物理状态线性探测（Linear Probing）

固定视觉编码器权重 $\phi$，仅训练一个简单的无偏置/有偏置线性回归层 $W \in \mathbb{R}^{d_{\text{state}} \times D}$：
$$\hat{s}_t^* = W z(o_t) + b$$
目标是预测底层的真实物理状态 $s_t^* = (x, y, v_x, v_y, \theta, \omega)$。

通过决定系数 $R^2$ 评估拟合质量：
$$R^2 = 1 - \frac{\sum_{i=1}^M \|s_t^{*(i)} - \hat{s}_t^{*(i)}\|^2}{\sum_{i=1}^M \|s_t^{*(i)} - \bar{s}^*\|^2}$$

- **理论判据**：若 $R^2 \ge 0.95$，说明非线性物理状态已经在潜在空间 $z$ 中被解耦为**全局线性可分流形**；下游动力学模型 $f(z_t, a_t)$ 无需花费额外层数去非线性“解码”物体的空间位置。

### 3. 诊断准则 2：反事实动作敏感度（Counterfactual Action Sensitivity）

在同一物理状态下施加不同的控制动作，潜在空间的状态演化必须产生统计显著的几何偏离：

$$\Delta z(a^{(1)}, a^{(2)}) = \big\| z(o_{t+1}^{(1)}) - z(o_{t+1}^{(2)}) \big\|_2 > \epsilon_{\text{threshold}}$$

- **失效模式（Action Collapse）**：许多视频预测模型在隐空间中 $\Delta z \approx 0$，即无论给什么控制指令，模型均输出惯性延续画面。合格的世界模型表征必须对控制动作具备严格的各向异性因果灵敏度。

### 4. 诊断准则 3：多步推演物体恒常性（Object Permanence）

当运动物体（如小球、障碍车辆）暂时移动到遮挡物背后或短暂滑出视野边缘时，在隐空间多步推演中，该物体的质量与动力学表征是否能够持续保留（而非在消失瞬间发生潜在表征湮灭）。

---

## 简洁实现：ConvEncoder 与 PatchViT 的特征提取与线性探测

下面的完整可运行代码实现了一个轻量级 `ConvEncoder` 与 `PatchViT`，并在一个带有确定性物理速度的弹跳小球环境中进行特征提取，最后通过 `LinearProbe` 诊断潜在表征对底层物理速度的线性解析能力。

```python
import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
# 1. 卷积编码器 (ConvEncoder): 维持局部归纳偏置
# =====================================================================
class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1),  # [B, 16, 32, 32]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),           # [B, 32, 16, 16]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),           # [B, 64, 8, 8]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                                    # [B, 64, 1, 1]
            nn.Flatten()                                                      # [B, 64]
        )
        self.proj = nn.Linear(64, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.proj(h)

# =====================================================================
# 2. 视觉 Transformer 编码器 (PatchViT): 分块投影与自注意力
# =====================================================================
class PatchViT(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # [B, C, H, W] -> [B, D, H/P, W/P] -> [B, N, D]
        patches = self.patch_proj(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1) + self.pos_embed
        out = self.transformer(tokens)
        return self.norm(out[:, 0])  # 提取 CLS Token 作为全局潜在表征

# =====================================================================
# 3. 诊断探针: 线性探测器 (Linear Probe)
# =====================================================================
class LinearProbe(nn.Module):
    """用于检验潜在表征 z 中是否线性包含物理状态量 (如速度 vx, vy)"""
    def __init__(self, latent_dim: int = 64, target_dim: int = 2):
        super().__init__()
        self.fc = nn.Linear(latent_dim, target_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)

# =====================================================================
# 4. 实验验证: 生成物理运动数据并评估探测能力
# =====================================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    B_total = 320
    H, W = 64, 64

    # 生成包含物理速度矢量的图像序列 (双帧差分光流特征)
    images = torch.zeros(B_total, 3, H, W)
    velocities = (torch.rand(B_total, 2) - 0.5) * 8.0  # 真实物理速度 vx, vy \in [-4.0, 4.0]

    for i in range(B_total):
        # 依据速度模拟物体在画面中的空间位移光斑
        cx = int(32 + velocities[i, 0].item() * 3.5)
        cy = int(32 + velocities[i, 1].item() * 3.5)
        cx = max(4, min(W - 5, cx))
        cy = max(4, min(H - 5, cy))
        images[i, 0, cy-3:cy+4, cx-3:cx+4] = 1.0  # 前景通道
        images[i, 1, 28:36, 28:36] = 0.3          # 静态参考物
        images[i, 2] = torch.randn(H, W) * 0.05    # 环境感知高频噪声

    # 划分训练集与测试集
    train_x, test_x = images[:240], images[240:]
    train_v, test_v = velocities[:240], velocities[240:]

    # 实例化两种视觉编码器
    conv_encoder = ConvEncoder(in_channels=3, latent_dim=64)
    vit_encoder = PatchViT(img_size=64, patch_size=8, in_channels=3, embed_dim=64)

    # 提取潜在表征 (测试阶段固定编码器)
    conv_encoder.eval()
    vit_encoder.eval()
    with torch.no_grad():
        z_train_conv = conv_encoder(train_x)
        z_test_conv = conv_encoder(test_x)
        z_train_vit = vit_encoder(train_x)
        z_test_vit = vit_encoder(test_x)

    print(f"输入图像尺寸: {images.shape}")
    print(f"ConvEncoder 输出潜在表征: {z_train_conv.shape}")
    print(f"PatchViT 输出潜在表征:     {z_train_vit.shape}")
    print("-" * 60)

    # 训练线性探测器解码真实速度
    def evaluate_probe(z_train: torch.Tensor, y_train: torch.Tensor, z_test: torch.Tensor, y_test: torch.Tensor, name: str):
        probe = LinearProbe(latent_dim=64, target_dim=2)
        optimizer = optim.Adam(probe.parameters(), lr=0.02)
        criterion = nn.MSELoss()

        for _ in range(200):
            optimizer.zero_grad()
            loss = criterion(probe(z_train), y_train)
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            preds = probe(z_test)
            test_mse = criterion(preds, y_test).item()
            ss_tot = torch.sum((y_test - y_test.mean(dim=0)) ** 2).item()
            ss_res = torch.sum((y_test - preds) ** 2).item()
            r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

        print(f"[{name}] 线性探测测试 MSE: {test_mse:.4f} | 物理速度解析 R^2 得分: {r2:.4f}")

    evaluate_probe(z_train_conv, train_v, z_test_conv, test_v, "ConvEncoder (未经动力学校准)")
    evaluate_probe(z_train_vit, train_v, z_test_vit, test_v, "PatchViT    (未经动力学校准)")
```

### 运行输出与诊断分析

```text
输入图像尺寸: torch.Size([320, 3, 64, 64])
ConvEncoder 输出潜在表征: torch.Size([240, 64])
PatchViT 输出潜在表征:     torch.Size([240, 64])
------------------------------------------------------------
[ConvEncoder (未经动力学校准)] 线性探测测试 MSE: 5.4310 | 物理速度解析 R^2 得分: -0.0155
[PatchViT    (未经动力学校准)] 线性探测测试 MSE: 5.4278 | 物理速度解析 R^2 得分: -0.0149
```

**核心实验结论**：
随机初始化的编码器无论架构是 CNN 还是 ViT，均无法让隐空间自发具备物理速度的线性可解耦性（$R^2 \approx 0$）。在后续章节中，我们将通过**时间差分自监督（JEPA）与动力学预测重构（RSSM）**对编码器施加端到端物理约束，使 $R^2$ 得分跃升至 $0.95$ 以上。

---

## 练习与思考

1. **平移等变性与置换不变性的动力学表征对比**：
   在连续物理动力学仿真中，小球碰撞后以匀速矢量 $\vec{v}$ 沿直线飞行。试从群论（Group Theory）角度严格证明：标准卷积网络对二维平移群 $(\mathbb{R}^2, +) \cong \mathrm{T}(2) \subset \mathrm{SE}(2)$ 的作用具备严格等变性，而未添加位置编码的标准自注意力算子具备置换群 $\mathcal{S}_N$ 的不变性。解释为什么若不加任何几何先验，纯自注意力网络需要消耗大量数据才能间接学会“刚体沿直线平移”这一基本物理规律？

2. **ViT Patch 离散化在亚像素接触力学检测中的理论极限**：
   考虑一个机械臂精密装配任务，抓夹指尖与工件接触面的微小形变仅跨越 $1.5$ 个物理像素（Sub-pixel Contact）。若采用分块大小为 $P=16$ 的 ViT 编码器，整个接触区域将被合并压入同一个 Patch Token 中。分析这种空间离散化对接触法向力估计的截断误差，并提出两种无须全局降低 $P$ 即可恢复亚像素高频几何的改进方案（如可变形注意力 Deformable Attention、多尺度 Patch 融合或 TokenLearner）。

3. **线性探测与互信息下界的数学形式化证明**：
   设真实物理状态为 $s_t^*$，观测图像为 $o_t$，潜在表征为 $z_t = \phi(o_t)$。若存在一个线性解码器 $W$ 使得预测均方误差 $\mathbb{E}\big[\|s_t^* - W z_t\|^2\big] \le \epsilon$。假设 $s_t^*$ 服从高斯分布，试利用香农信息论与数据处理不等式（Data Processing Inequality），推导潜在表征 $z_t$ 与真实状态 $s_t^*$ 之间互信息 $I(z_t; s_t^*)$ 的严格下界，并说明为什么线性探测误差是衡量表征充分性（Sufficiency）的有效代用指标。

4. **二维旋转位置编码（2D-RoPE）在相机视点变换下的几何解析性推导**：
   设车载相机经历连续俯仰（Pitch）与偏航（Yaw）角速度旋转，导致成像平面上的光流场满足仿射变换 $\mathbf{x}' = \mathbf{A} \mathbf{x} + \mathbf{b}$。试推导在 2D-RoPE 调制下，旋转变换前后 Query-Key 内积得分的解析闭式解。分析 2D-RoPE 在何种视点几何变换下能够保持内积的平移不变性，而在何种变换下（如尺度缩放、透视投影变形）会导致相对内积失真？

5. **像素级重构损失与特征级 JEPA 损失的优化动态解耦对比**：
   设观测图像 $o_t = s_{\text{phy}} + \eta_{\text{tex}}$，其中 $s_{\text{phy}}$ 是控制相关的低频物理信号，$\eta_{\text{tex}}$ 是高能量但无因果作用的环境纹理噪声（满足 $\operatorname{Var}(\eta_{\text{tex}}) \gg \operatorname{Var}(s_{\text{phy}})$）。试分析在最小化像素 MSE 损失 $\mathcal{L}_{\text{pixel}} = \|o_t - \operatorname{Dec}(\operatorname{Enc}(o_t))\|^2$ 与最小化潜在特征预测损失 $\mathcal{L}_{\text{JEPA}} = \|\operatorname{Enc}(o_{t+1}) - \operatorname{Pred}(\operatorname{Enc}(o_t), a_t)\|^2$ 时，编码器梯度更新方向的数学差异。为什么 JEPA 能够实现对无关纹理噪声的天然鲁棒性？

---

## 本节总结

- **原始像素充满感知冗余与维度灾难**，世界模型必须依靠空间视觉编码器将 $10^4$ 维度的像素阵列压缩为低维结构化潜在流形。
- **CNN 凭借局部连接与平移等变性**构筑了强大的几何归纳偏置，在低样本强化的局部物理控制中具备极高的样本效率；**ViT 凭借分块投影与自注意力机制**打破了感受野限制，提供了统一的多模态 Token 交互接口与卓越的规模扩展上限。
- **位置编码（1D / 2D-RoPE / 3D-Tubelet）** 是自注意力网络感知物理空间拓扑与时空连续轨迹的代数桥梁。
- **高重构保真度并不等价于高质量的世界模型表征**；必须通过物理状态线性探测、反事实动作敏感度与多步物体恒常性三大准则，检验隐空间是否真正具备支持未来动力学推演的因果可解耦性。

空间观测被编码为潜在 Token 后，下一个核心问题随之浮现：如何跨越时间维度，将连续历史 Token 压缩为具有长期因果记忆的动态信念状态？在下一节 [2.3 记忆与动力学](/chapters/02-foundations/03-memory-and-dynamics) 中，我们将深入剖析 RNN、GRU、Transformer 与循环状态空间模型（RSSM）的时序建模精髓。

---

## 参考文献

1. **Gradient-Based Learning Applied to Document Recognition** (LeCun et al., Proceedings of the IEEE, 1998) [[IEEE Xplore](https://ieeexplore.ieee.org/document/726791)]：系统确立现代卷积神经网络（LeNet）基础架构，证明权重共享与空间局部感受野的有效性。
2. **Deep Residual Learning for Image Recognition** (He et al., CVPR 2016) [[arXiv:1512.03385](https://arxiv.org/abs/1512.03385)]：提出残差连接（ResNet），解决深层神经网络梯度退化问题，成为世界模型最常用的卷积骨干之一。
3. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** (Dosovitskiy et al., ICLR 2021) [[arXiv:2010.11929](https://arxiv.org/abs/2010.11929)]：提出 Vision Transformer（ViT），开创将图像分块投影为 Token 序列直接送入 Transformer 编码的通用范式。
4. **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows** (Liu et al., ICCV 2021) [[arXiv:2103.14030](https://arxiv.org/abs/2103.14030)]：引入移动窗口自注意力，将自注意力计算复杂度降至与图像尺寸线性相关，兼具 CNN 层级结构与 ViT 全局建模能力。
5. **V-JEPA: Video Joint Embedding Predictive Architecture** (Bardes et al., 2024) [[arXiv:2404.08471](https://arxiv.org/abs/2404.08471)]：提出基于时空特征预测的无自回归视频表征学习架构，摒弃像素重构，专注高级物理特征演化。
6. **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., Neurocomputing, 2024) [[arXiv:2104.09864](https://arxiv.org/abs/2104.09864)]：提出旋转位置编码（RoPE），严格证明内积形式下的相对位置解析性质与距离衰减规律。
$$
