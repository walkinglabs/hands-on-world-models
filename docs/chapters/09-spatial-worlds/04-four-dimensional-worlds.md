# 4D 时空世界模型

Ha 和 Schmidhuber 的 World Models 从二维游戏画面学习每帧潜变量与时间动力学 [[Ha & Schmidhuber, 2018]](https://arxiv.org/abs/1803.10122)。该论文在 CarRacing 与 VizDoom 上验证控制，却没有显式维护相机几何、三维占用或可跨视角查询的场景表示。因此，它适合作为视觉潜在动力学的例子，不能直接用来证明二维潜变量在所有三维场景中必然产生物理错误。本节研究的 4D 表示，是在这一区别上进一步加入三维空间与时间结构。

生成式驾驶与视频模型开始显式处理时间和控制条件。例如，GAIA-1 根据驾驶视频、文本与动作生成未来画面 [[Anthony Hu et al., 2023]](https://arxiv.org/abs/2309.17080)，Stable Video Diffusion 在潜在视频生成上研究了训练与数据策略 [[Blattmann et al., 2023]](https://arxiv.org/abs/2311.15127)。但这两篇论文并没有显式维护 NeRF、3D 高斯或体素形式的三维几何，因此它们只能作为时序生成背景，不能作为“显式 4D 几何世界模型”已经成立的证据。本节随后构造的 4D 表示是一种教学性抽象。

## 从运动学场到四维张量表征

要透彻理解 4D 时空模型，我们无需一开始便面对复杂的深度神经网络，而应当从高中物理学中最简单的运动学出发。在中学课堂上，我们学习过质点的运动：假设一个质点在三维空间中，其位置可以用三维笛卡尔坐标系的向量 $\mathbf{p} = (x, y, z)$ 来表示。随着时间 $t$ 沿单向流动，该质点的位置成为时间的参数函数：

$$\mathbf{p}(t) = (x(t), y(t), z(t))$$

如果已知质点在初始时刻 $t_0$ 的位置 $\mathbf{p}_0$ 与恒定速度向量 $\mathbf{v}$，我们可通过严密的线性代数方程预测其未来状态：$\mathbf{p}(t) = \mathbf{p}_0 + \mathbf{v} \cdot (t - t_0)$。

然而，世界模型的预测对象并非单一质点，而是包含了空气流体、刚体碰撞、材质光影在内的整个复杂物理场（Physical Field）。我们需要从追踪“孤立的点”，跃升为对整个“连续空间”随时间演化的建模。我们将真实的物理世界严格定义为一个连续的 4D 时空映射函数 $F$。对于空间中任意一点 $\mathbf{p} \in \mathbb{R}^3$ 和任意时刻 $t \in \mathbb{R}$，$F$ 能够输出该时空坐标下所有的可观测物理量（如体密度 $\sigma$ 与辐射光度 $\mathbf{c}$）：

$$F: (\mathbf{p}, t) \rightarrow (\sigma, \mathbf{c})$$

由于现代计算设备的离散本质，我们无法直接将函数 $F$ 存入内存。我们必须对时间和空间同时进行离散化，将这个连续的场映射为一个高维张量表征（Tensor Representation）。在实际的 4D 世界模型中，我们在离散的时间步 $t$ 上提取一个三维特征体积（3D Feature Volume）作为当前的隐状态 $\mathbf{s}_t \in \mathbb{R}^{D \times H \times W \times C}$。其中 $D, H, W$ 为三维空间分辨率，$C$ 为特征通道数。在这个框架下，物理法则被“编码”为隐空间中高维张量序列之间的非线性状态转移函数。

## 变分推断下的时空演化

有了高维的张量表征后，我们需要构建预测其时间演化的动力学模型。假设状态演化遵循马尔可夫决策过程（MDP），即下一个状态 $\mathbf{s}_{t+1}$ 完全由当前状态 $\mathbf{s}_t$ 和外部干预动作 $\mathbf{a}_t$ 所决定。

> **变分推断与时空流体**
> 想象一条湍急的河流，河水的三维分布随时间剧烈变化。我们要在这个复杂流体中建立一种预测机制，既要保证局部水流不违反流体力学（局部一致性），又要保证整体河流的走向符合重力（全局约束）。在 4D 世界模型中，变分推断（Variational Inference）就扮演着这样的“约束”角色。它通过强制预测的时空隐状态分布向真实的后验分布靠拢，防止模型在长期预测时产生如同河流溃堤般发散的“幻觉”（Hallucination）。

设 $\mathbf{o}_{1:T}$ 为我们在 $1$ 到 $T$ 时刻内接收到的多视角观测序列（例如多摄像头的 2D 图像视频流），$\mathbf{a}_{1:T}$ 为对应的控制动作。我们希望模型通过内部的 4D 隐状态 $\mathbf{s}_{1:T}$，最大化观测序列的条件对数似然：

$$\log P(\mathbf{o}_{1:T} \mid \mathbf{a}_{1:T}) = \log \int P(\mathbf{o}_{1:T} \mid \mathbf{s}_{1:T}) P(\mathbf{s}_{1:T} \mid \mathbf{a}_{1:T}) \, d\mathbf{s}_{1:T}$$

直接计算上式中对所有可能 4D 路径积分的边际似然在数学上是极其棘手（Intractable）的。因此，我们引入一个变分后验分布（Variational Posterior） $Q(\mathbf{s}_{1:T} \mid \mathbf{o}_{1:T}, \mathbf{a}_{1:T})$ 来逼近真实的后验。利用詹森不等式（Jensen's Inequality），我们可以推导出一个严格的变分下界（Evidence Lower Bound, ELBO）：

$$ \log P(\mathbf{o}_{1:T} \mid \mathbf{a}_{1:T}) \geq \mathbb{E}_{Q} \left[ \sum_{t=1}^T \log P(\mathbf{o}_t \mid \mathbf{s}_t) \right] - \sum_{t=1}^T \mathbb{E}_{Q} \left[ D_{\text{KL}} \left( Q(\mathbf{s}_t \mid \mathbf{s}_{t-1}, \mathbf{o}_{\leq t}, \mathbf{a}_{< t}) \| P(\mathbf{s}_t \mid \mathbf{s}_{t-1}, \mathbf{a}_{t-1}) \right) \right] $$

让我们温柔地拆解该公式中蕴含的物理意义：

1. **三维空间渲染项（左侧期望项）**：$\log P(\mathbf{o}_t \mid \mathbf{s}_t)$ 描述了给定当前的 4D 特征体积 $\mathbf{s}_t$，系统将其解码并渲染回多视角二维观测 $\mathbf{o}_t$ 的能力。这一项强制模型在隐空间中维持极其严谨的**三维多视图几何一致性**（Multi-view Geometric Consistency）。如果 $\mathbf{s}_t$ 不能代表一个客观存在的 3D 结构，该似然项将急剧下降。
2. **时间动力学正则项（右侧 KL 散度项）**：该项惩罚了两个分布之间的差异。一个是利用上帝视角（当前及历史的所有 2D 观测与动作）推断得出的后验分布 $Q$；另一个是仅依据前一时刻状态 $\mathbf{s}_{t-1}$ 通过先验动力学网络预测得出的先验分布 $P$。KL 散度的最小化过程，本质上就是在“教导”模型的动力学推演网络学习真实世界中不可见的物理守恒与运动演变规律。

## 4D 时空神经网络的构建

要在代码中实现这种庞大的数学理论，模型通常需要解耦对空间维度与时间维度的处理。主流架构中往往交替使用三维空间卷积（捕捉局部几何属性，如表面的连续性和边缘）与时间注意力层（建模跨时间步的长距离物理因果关系）。

(**下面的代码定义了一个极度简化的 4D 时空动力学推演块。**) 为了凸显其对时空张量的处理逻辑，我们展示了该模块如何分离处理 3D 空间维度和 1D 时间维度。

```python
import torch
from torch import nn

class SpatialTemporalBlock(nn.Module):
    """一个简化的 4D 时空推演块，结合了 3D 空间卷积和时间注意力。"""
    def __init__(self, channels, num_heads):
        super().__init__()
        # 3D 卷积用于捕捉空间局部几何特征 (深度, 高度, 宽度)
        self.spatial_conv3d = nn.Conv3d(
            in_channels=channels, out_channels=channels,
            kernel_size=3, padding=1
        )
        # 多头注意力机制用于捕捉时间序列上的物理演化与长程依赖
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(channels)
        self.layer_norm2 = nn.LayerNorm(channels)

    def forward(self, s_t):
        """
        输入 s_t: 隐状态张量，形状为 (批量大小, 时间步, 通道数, 深度, 高度, 宽度)
                  即 (B, T, C, D, H, W)
        输出 s_next: 预测并更新后的下一层状态序列，形状不变
        """
        B, T, C, D, H, W = s_t.shape

        # 1. 空间特征提取
        # 将时间和批量维度折叠，强迫网络专注于独立的 3D 体积
        s_spatial = s_t.view(B * T, C, D, H, W)
        s_spatial_conv = self.spatial_conv3d(s_spatial)
        # 恢复原有形状，并应用残差连接以防止梯度消失
        s_spatial = s_spatial_conv.view(B, T, C, D, H, W) + s_t

        # 2. 时间特征演化
        # 为应用时间维度的注意力，我们将空间坐标视作独立的序列元素
        # 形状调整为 (B * D * H * W, T, C)
        s_temporal = s_spatial.permute(0, 3, 4, 5, 1, 2).reshape(-1, T, C)
        s_temporal = self.layer_norm1(s_temporal)

        # 时间自注意力计算：让同一空间坐标轴上的历史特征相互作用
        # 实际训练中通常需要提供 causal_mask 来防止未来信息的泄露
        attn_out, _ = self.temporal_attn(s_temporal, s_temporal, s_temporal)
        s_temporal = s_temporal + attn_out
        s_temporal = self.layer_norm2(s_temporal)

        # 将形状严密地逆向映射回 4D 张量格式 (B, T, C, D, H, W)
        s_next = s_temporal.view(B, D, H, W, T, C).permute(0, 4, 5, 1, 2, 3)
        return s_next
```

## 总结

**显式或半显式的三维结构**可以为遮挡、视角变化和空间查询提供更合适的归纳偏置；**时间动力学**则把静态重建扩展为未来预测。不过，这种表示并不会自动保证物理正确，也尚不能据此断言它已成为自动驾驶和机器人控制的统一核心范式。是否有效仍需用几何误差、多步预测、反事实一致性和**闭环控制结果**分别验证。
