# 9.4 4D 时空世界模型

Ha 和 Schmidhuber 的 World Models 从二维游戏画面学习每帧潜变量与时间动力学 [[Ha & Schmidhuber, 2018]](https://arxiv.org/abs/1803.10122)。该论文在 CarRacing 与 VizDoom 上验证控制，却没有显式维护相机几何、三维占用或可跨视角查询的场景表示。因此，它适合作为视觉潜在动力学的例子，不能直接用来证明二维潜变量在所有三维场景中必然产生物理错误。本节研究的 4D 表示，是在这一区别上进一步加入三维空间与时间结构。

<div align="center">
<img src="/figures/09-spatial-worlds/source/04-four-dimensional-worlds/dnerf-fig1.png" alt="D-NeRF 从不同时间与视角的动态场景图像重建连续时空辐射场，并合成未见时空位置的画面。" width="86%">

_图 9.4-1：D-NeRF 从不同时间与视角的动态场景图像重建连续时空辐射场，并合成未见时空位置的画面。 出处：Albert Pumarola et al.，[D-NeRF: Neural Radiance Fields for Dynamic Scenes](https://arxiv.org/abs/2011.13961)（2021），Figure 1。_
</div>

生成式驾驶与视频模型开始显式处理时间和控制条件。例如，GAIA-1 根据驾驶视频、文本与动作生成未来画面 [[Anthony Hu et al., 2023]](https://arxiv.org/abs/2309.17080)，Stable Video Diffusion 在潜在视频生成上研究了训练与数据策略 [[Blattmann et al., 2023]](https://arxiv.org/abs/2311.15127)。但这两篇论文并没有显式维护 NeRF、3D 高斯或体素形式的三维几何，因此它们只能作为时序生成背景，不能作为“显式 4D 几何世界模型”已经成立的证据。本节随后构造的 4D 表示是一种教学性抽象。

## 从运动轨迹到四维张量表征

先看一个移动质点。它在三维空间中的位置可写为 $\mathbf{p}=(x,y,z)$，随时间变化后成为函数：

$$\mathbf{p}(t) = (x(t), y(t), z(t))$$

若初始位置为 $\mathbf{p}_0$，速度在这段时间内近似不变，则 $\mathbf{p}(t)=\mathbf{p}_0+\mathbf{v}(t-t_0)$。

场景中有许多物体，而且可见外观也会随时间变化。一种连续表示是令函数 $F$ 接收空间位置与时间，输出体密度 $\sigma$ 和视角相关颜色 $\mathbf{c}$：

<div align="center">
<img src="/figures/09-spatial-worlds/source/04-four-dimensional-worlds/dnerf-fig3.png" alt="D-NeRF 用时间条件形变网络把动态观测映射到统一规范空间，再由规范 NeRF 解释颜色与密度。" width="86%">

_图 9.4-2：D-NeRF 用时间条件形变网络把动态观测映射到统一规范空间，再由规范 NeRF 解释颜色与密度。 出处：Albert Pumarola et al.，[D-NeRF: Neural Radiance Fields for Dynamic Scenes](https://arxiv.org/abs/2011.13961)（2021），Figure 3。_
</div>

$$F: (\mathbf{p}, t) \rightarrow (\sigma, \mathbf{c})$$

这是动态辐射场的一种抽象，并不包含“所有物理量”。另一种做法是在离散时间步 $t$ 上维护三维特征体 $\mathbf{s}_t \in \mathbb{R}^{D \times H \times W \times C}$，其中 $D,H,W$ 是空间分辨率，$C$ 是通道数。连续场便于在任意位置查询，离散网格便于使用卷积；两者也可以组合。

## 变分推断下的时空演化

接下来用一阶马尔可夫假设描述时间演化：下一状态分布只显式依赖当前状态和动作。这里讨论的是受控状态空间模型；只有再加入奖励与决策目标时，才构成完整的马尔可夫决策过程。

<div align="center">
<img src="/figures/09-spatial-worlds/source/04-four-dimensional-worlds/nsff-fig2.png" alt="NSFF 通过前后向场景流连接相邻时刻的三维点，并联合辐射场与遮挡权重重建动态视频。" width="86%">

_图 9.4-3：NSFF 通过前后向场景流连接相邻时刻的三维点，并联合辐射场与遮挡权重重建动态视频。 出处：Zhengqi Li et al.，[Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes](https://arxiv.org/abs/2011.13084)（2021），Figure 2。_
</div>

设 $\mathbf{o}_{1:T}$ 为我们在 $1$ 到 $T$ 时刻内接收到的多视角观测序列（例如多摄像头的 2D 图像视频流），$\mathbf{a}_{1:T}$ 为对应的控制动作。我们希望模型通过内部的 4D 隐状态 $\mathbf{s}_{1:T}$，最大化观测序列的条件对数似然：

$$\log P(\mathbf{o}_{1:T} \mid \mathbf{a}_{1:T}) = \log \int P(\mathbf{o}_{1:T} \mid \mathbf{s}_{1:T}) P(\mathbf{s}_{1:T} \mid \mathbf{a}_{1:T}) \, d\mathbf{s}_{1:T}$$

对所有潜在状态轨迹积分通常不可解析，因此引入变分后验 $Q(\mathbf{s}_{1:T}\mid\mathbf{o}_{1:T},\mathbf{a}_{1:T})$ 近似真实后验。利用詹森不等式可以得到证据下界（ELBO）：

$$ \log P(\mathbf{o}_{1:T} \mid \mathbf{a}_{1:T}) \geq \mathbb{E}_{Q} \left[ \sum_{t=1}^T \log P(\mathbf{o}_t \mid \mathbf{s}_t) \right] - \sum_{t=1}^T \mathbb{E}_{Q} \left[ D_{\text{KL}} \left( Q(\mathbf{s}_t \mid \mathbf{s}_{t-1}, \mathbf{o}_{\leq t}, \mathbf{a}_{< t}) \| P(\mathbf{s}_t \mid \mathbf{s}_{t-1}, \mathbf{a}_{t-1}) \right) \right] $$

这个目标包含两部分：

1. **观测似然** $\log P(\mathbf{o}_t\mid\mathbf{s}_t)$：要求潜在状态能够解释多视角观测。若解码器显式使用相机几何，这一项可以提供多视角一致性监督；仅靠似然本身并不保证潜变量一定对应真实三维结构。
2. **动力学 KL 项**：比较读取当前观测的后验与只依赖前一状态、动作的先验，使先验更接近训练时可推断出的状态分布。它约束预测分布，但不会自动产生物理守恒定律。

## 4D 时空神经网络的构建

一种简单结构是交替使用三维卷积和时间注意力：前者聚合局部空间邻域，后者让同一空间位置读取多个历史时间步。

<div align="center">
<img src="/figures/09-spatial-worlds/source/04-four-dimensional-worlds/4dgs-fig3.png" alt="4DGS 以规范三维高斯和时空形变场构成动态场景管线，实现可实时渲染的四维表示。" width="86%">

_图 9.4-4：4DGS 以规范三维高斯和时空形变场构成动态场景管线，实现可实时渲染的四维表示。 出处：Guanjun Wu et al.，[4D Gaussian Splatting for Real-Time Dynamic Scene Rendering](https://arxiv.org/abs/2310.08528)（2024），Figure 3。_
</div>

下面的模块只演示张量重排与因果时间注意力，不包含观测编码器、坐标对齐或概率状态。

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
        # 折叠时间和批量维度，对每个时间步独立做 3D 卷积
        s_spatial = s_t.reshape(B * T, C, D, H, W)
        s_spatial_conv = self.spatial_conv3d(s_spatial)
        # 恢复原有形状并添加残差
        s_spatial = s_spatial_conv.reshape(B, T, C, D, H, W) + s_t

        # 2. 时间特征演化
        # 为应用时间维度的注意力，我们将空间坐标视作独立的序列元素
        # 形状调整为 (B * D * H * W, T, C)
        s_temporal = s_spatial.permute(0, 3, 4, 5, 1, 2).reshape(-1, T, C)
        s_temporal = self.layer_norm1(s_temporal)

        # 上三角为 True，阻止时间步读取未来特征
        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=s_t.device), diagonal=1
        )
        attn_out, _ = self.temporal_attn(
            s_temporal, s_temporal, s_temporal, attn_mask=causal_mask
        )
        s_temporal = s_temporal + attn_out
        s_temporal = self.layer_norm2(s_temporal)

        # 还原为 (B, T, C, D, H, W)
        s_next = s_temporal.reshape(B, D, H, W, T, C).permute(0, 4, 5, 1, 2, 3)
        return s_next
```

<div align="center">
<img src="/figures/09-spatial-worlds/latex/04-four-dimensional-worlds/spacetime-axis-refactor.png" alt="六维时空状态在空间卷积时折叠批次与时间，在时间注意力时折叠批次与空间位置" width="86%">

_图 9.4-5：空间卷积把时间当作批次维，时间注意力则把每个空间位置当作独立序列批次；逆重排后恢复原六维布局。本文根据本节张量过程绘制。_
</div>

## 总结

**显式或半显式的三维结构**可以为遮挡、视角变化和空间查询提供更合适的归纳偏置；**时间动力学**则把静态重建扩展为未来预测。不过，这种表示并不会自动保证物理正确，也尚不能据此断言它已成为自动驾驶和机器人控制的统一核心范式。是否有效仍需用几何误差、多步预测、反事实一致性和**闭环控制结果**分别验证。
