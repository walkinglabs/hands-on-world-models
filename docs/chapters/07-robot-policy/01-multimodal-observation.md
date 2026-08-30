# 7.1 具身智能与多模态观测

在前面的章节中，我们主要探讨了在纯视觉或纯文本环境下的深度学习模型。然而，真正的智能体（Agent）并非“缸中之脑”，它们生存在复杂的物理世界中，需要通过躯体与环境发生持续的物理交互。这种强调智能体与环境物理形态耦合的智能范式，被称为**具身智能**（Embodied AI）。

在具身智能的设定下，智能体（例如一台双足机器人或一条机械臂）在执行任务时，不仅能通过摄像头“看”到外部世界，还能通过关节编码器和力矩传感器“感受”到自身的姿态与受力。如何将这些维度、采样频率、物理语义截然不同的信息源有效地整合在一起，形成对当前状态的统一理解？这就是**多模态观测**（Multimodal Observation）所要解决的核心问题。

## 7.1.1 历史脉络与学术追溯

具身智能的思想可以追溯到人工智能的早期。1986年，Robotics领域的先驱Rodney Brooks在论文《A robust layered control system for a mobile robot》[Brooks, 1986] 中提出了包容体系结构（Subsumption Architecture），严厉批评了当时主流的“感知-建模-规划-行动”这种自上而下的符号计算范式。他主张智能应当直接从感觉运动（Sensorimotor）的交互中涌现。

随着深度学习的爆发，Levine等人在2016年的经典工作《End-to-end training of deep visuomotor policies》[Levine et al., 2016] 中，首次展示了如何将卷积神经网络（CNN）与强化学习结合，直接将原始像素和机器人的关节状态映射为电机的力矩输出。这项工作打破了传统机器人学中视觉感知与控制模块割裂的局面，确立了端到端多模态策略（Visuomotor Policy）的基础。

近年来，随着Transformer [Vaswani et al., 2017] 在多模态领域的成功，诸如RT-1 [Brohan et al., 2022] 和 RT-2 [Brohan et al., 2023] 等视觉-语言-动作（Vision-Language-Action, VLA）模型，进一步将多模态观测的边界扩展到了包含自然语言指令、RGB-D视觉流以及高维本体感受的大一统框架中。在这些系统中，多模态特征的对齐与融合能力，成为了决定机器人策略上限的最关键因素。

## 7.1.2 物理量的降维映射：从单摆到机器人状态空间

为了理解多模态观测的必要性，我们不妨先回到高中物理中最经典的单摆模型。

假设我们要完全描述一个单摆在某一时刻的物理状态，我们需要哪些信息？根据经典力学，我们只需要知道单摆当前的摆角 $\theta$ 和它的角速度 $\dot{\theta}$。只要知道了这两个标量，我们就能利用运动学和动力学方程预测它未来的所有行为。

在机器人学中，这种对自身内在物理状态的测量，被称为**本体感受**（Proprioception）。对于一个拥有 $n$ 个自由度的机器人，其本体状态可以通过广义坐标 $\mathbf{q} \in \mathbb{R}^n$（例如各关节的角度）和广义速度 $\dot{\mathbf{q}} \in \mathbb{R}^n$（各关节的角速度）来严格定义。我们将其拼接为一个本体观测向量：

$$
\mathbf{o}_{\text{prop}} = [\mathbf{q}^\top, \dot{\mathbf{q}}^\top]^\top \in \mathbb{R}^{2n}
$$

然而，机器人并不是在一个空无一物的真空中运动。假设我们要让机械臂去抓取桌子上的一个苹果，仅仅知道机械臂自身的关节角度显然是不够的，它还必须知道苹果在空间中的位置。这种对外部环境的感知，被称为**外感受**（Exteroception）。在现代机器人系统中，最常见的外感受器就是RGB摄像头，它提供了一个三维张量 $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$。

因此，在时间步 $t$，具身智能体所接收到的完整多模态观测 $\mathbf{o}_t$ 至少包含了视觉和本体两个模态：

$$
\mathbf{o}_t = \{ \mathbf{I}_t, \mathbf{o}_{\text{prop}, t} \}
$$

我们的目标是设计一个神经网络函数 $f_\theta$，将这个异构的观测集合映射为一个统一的低维稠密向量 $\mathbf{z}_t \in \mathbb{R}^d$，从而供下游的策略网络（Policy Network）计算具体的控制动作。

## 7.1.3 模态对齐与融合的数学推导

视觉图像 $\mathbf{I}$ 是一个具有极高空间冗余度的高维张量（通常有几十万甚至上百万个像素），而本体状态 $\mathbf{o}_{\text{prop}}$ 则是一个维度极低、但物理意义极其密集的向量。将它们直接相加显然是荒谬的。我们需要通过编码器（Encoder）将它们映射到同一个潜空间（Latent Space）中。

首先，我们分别独立地对两种模态进行编码：

$$
\mathbf{z}_{\text{vis}} = f_{\text{vis}}(\mathbf{I}; \theta_{\text{vis}}) \in \mathbb{R}^{d_v}
$$

$$
\mathbf{z}_{\text{prop}} = f_{\text{prop}}(\mathbf{o}_{\text{prop}}; \theta_{\text{prop}}) \in \mathbb{R}^{d_p}
$$

其中，$f_{\text{vis}}$ 通常是ResNet或Vision Transformer（ViT），而 $f_{\text{prop}}$ 通常是一个多层感知机（MLP）。

接下来，我们需要将 $\mathbf{z}_{\text{vis}}$ 和 $\mathbf{z}_{\text{prop}}$ 融合。最直观也是最简单的方法是**拼接（Concatenation）与线性投影**。

假设我们退化到最极端的一维情况，即视觉特征提取出了一个标量 $z_v \in \mathbb{R}$，本体特征提取出了一个标量 $z_p \in \mathbb{R}$。我们希望得到一个综合特征 $z \in \mathbb{R}$。最简单的线性融合就是对它们赋予不同的权重，并加上偏置：

$$
z = w_1 z_v + w_2 z_p + b
$$

将这个标量方程严格地推广到高维向量空间。我们将两个特征向量在特征维度上进行拼接，得到向量 $[\mathbf{z}_{\text{vis}}^\top, \mathbf{z}_{\text{prop}}^\top]^\top \in \mathbb{R}^{d_v + d_p}$。然后，我们应用一个权重矩阵 $\mathbf{W} \in \mathbb{R}^{d \times (d_v + d_p)}$ 进行线性投影，并经过一个非线性激活函数 $\sigma$：

$$
\mathbf{z}_{\text{fused}} = \sigma \left( \mathbf{W} \begin{bmatrix} \mathbf{z}_{\text{vis}} \\ \mathbf{z}_{\text{prop}} \end{bmatrix} + \mathbf{b} \right)
$$

这种被称为“后期融合（Late Fusion）”的策略在早期深度强化学习中非常普遍。然而，它的局限性在于：权重矩阵 $\mathbf{W}$ 在训练完成后是静态的，这意味着无论机器人处于何种姿态，视觉特征和本体特征之间的组合方式是不变的。

## 7.1.4 跨模态注意力机制（Cross-Modal Attention）

在高度动态的物理交互中，静态融合往往是不够的。

> 想象你正在驾驶汽车（本体感受：方向盘转角、车速）。当你在空旷直行时，你会关注正前方的路况；而当你打转向灯准备在高速公路上变道时（特定本体状态），你的注意力会自动集中在后视镜的特定区域（视觉状态的动态聚焦）。这种**由本体状态主导的、对视觉特征进行动态空间选择**的机制，在数学上可以通过跨模态注意力（Cross-Modal Attention）来严谨刻画。

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

通过这种方式，神经网络能够学会根据当前的机器人的关节状态，动态地“注视”图像中对其下一步动作最具指导意义的区域。

## 7.1.5 代码实现：构建多模态观测编码器

(**下面我们将基于PyTorch和TensorFlow，实现一个包含视觉CNN、本体MLP以及拼接融合机制的基础多模态观测编码器。**)

```{.python .input}
#@tab pytorch
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

```{.python .input}
#@tab tensorflow
import tensorflow as tf

class MultiModalEncoder(tf.keras.Model):
    def __init__(self, prop_dim=14, vis_embed_dim=256, 
                 prop_embed_dim=64, fused_dim=128):
        super().__init__()
        
        # 1. 视觉编码器
        self.vis_encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=8, strides=4, activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu'),
            tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(vis_embed_dim),
            tf.keras.layers.LayerNormalization(epsilon=1e-5)
        ])
        
        # 2. 本体编码器
        self.prop_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(prop_embed_dim),
            tf.keras.layers.LayerNormalization(epsilon=1e-5)
        ])
        
        # 3. 融合层
        self.fusion_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(fused_dim)
        ])

    def call(self, img_obs, prop_obs):
        # 提取视觉特征
        z_vis = self.vis_encoder(img_obs)
        # 提取本体特征
        z_prop = self.prop_encoder(prop_obs)
        
        # 拼接特征
        z_concat = tf.concat([z_vis, z_prop], axis=1)
        
        # 通过融合MLP
        fused_feature = self.fusion_mlp(z_concat)
        return fused_feature

# 测试前向传播
encoder = MultiModalEncoder()
# TensorFlow通常使用NHWC格式，此处假设输入图像维度为84x84x3
dummy_img = tf.random.normal((4, 84, 84, 3)) 
dummy_prop = tf.random.normal((4, 14))
output = encoder(dummy_img, dummy_prop)
print(f"融合特征的张量形状: {output.shape}")
```

## 7.1.6 小结

* 具身智能要求智能体处理与其躯体及环境物理交互相关的数据。多模态观测（主要是视觉外感受与关节本体感受）是构建具身策略网络的基础。
* 对于跨越维度和语义鸿沟的多源数据，我们必须通过各自专用的编码网络将其投影到统一的潜空间中。
* 简单的拼接融合（Late Fusion）实现简单但缺乏动态交互能力；跨模态注意力机制（Cross-Modal Attention）允许神经网络基于本体状态动态地对空间视觉特征进行加权选择。

## 7.1.7 练习

1. 在公式 :eqref:`eq_prop_vector` 中，如果我们要描述一台带有6自由度机械臂（每个关节可测角度和角速度）以及一个底盘（可测平面 $x, y$ 坐标、朝向角 $\psi$ 及其对应的速度）的移动机器人，其本体观测向量 $\mathbf{o}_{\text{prop}}$ 的维度是多少？
   * **提示**：分别计算机械臂和底盘的广义坐标和速度维度并求和。
2. 仔细观察代码实现中的 `MultiModalEncoder` 类。为什么在对 `z_vis` 和 `z_prop` 提取特征的最后一步，我们都加入了一个 `LayerNorm`（层归一化）操作？如果不加，在后续的拼接与线性映射中可能会引发什么数值优化问题？
   * **提示**：思考不同模态编码器初始输出权重的方差差异，以及这种差异在 $\mathbf{W} \mathbf{z}_{\text{concat}}$ 矩阵乘法中会导致梯度如何流动。
3. 如果我们希望将当前的**后期拼接融合**替换为 :numref:`sec_multimodal_observation` 提到的**跨模态注意力融合**，请写出将视觉卷积特征图（形状为 `[B, 64, 7, 7]`）转换为注意力键 $\mathbf{K}$ 和值 $\mathbf{V}$ 时，张量形状必须经历哪些重塑（Reshape）和转置操作？
