# 9.4 四维时空建模与动态场景演化

在前面的章节中，我们深入探讨了静态三维空间的几何投影、BEV 栅格化与神经辐射场（NeRF/3DGS）渲染。然而，真实物理世界从来不是一座静止不动的石雕展馆——在自动驾驶街道上，有穿流不息的车辆与横穿马路的行人；在机器人作业台上，有被机械臂夹爪推移变形的软质物体。

如果仅用静态三维模型去描述世界，面对动态运动就会产生灾难性的“鬼影（Ghosting）”与模糊拉丝。

为了构建真正能够预测未来的世界模型，我们必须将时间维度 $t$ 作为第四个基本物理维度，将静态的三维空间扩展为**连续的四维时空流形（4D Spacetime Worlds）**。

本节我们将从非刚体形变与经典连续介质力学出发，推导动态辐射场（D-NeRF）、四维高斯（4DGS）与三维场景流（Scene Flow）的底层数学机制，并使用纯底层 PyTorch 算子实现时空形变与动态渲染引擎。

<div align="center">

<img src="/figures/09-spatial-worlds/source/04-four-dimensional-worlds/dnerf-fig1.png" alt="D-NeRF 从不同时间与视角的动态场景图像重建连续时空辐射场，并合成未见时空位置的画面。" width="86%">

_图 9.4-1：D-NeRF 从不同时间与视角的动态场景图像重建连续时空辐射场，并合成未见时空位置的画面。 出处：[D-NeRF: Neural Radiance Fields for Dynamic Scenes，Albert Pumarola et al.，2021](https://arxiv.org/abs/2011.13961)。_

</div>

---

## 9.4.1 物理与几何基石：规范空间与时间形变场的解耦哲学

要对四维时空中的动态物体进行数学建模，我们首先需要从经典运动学与弹性力学中汲取核心思想。

### 1. 规范空间（Canonical Space）与形变场（Deformation Field）
直接训练一个输入为四维坐标 $(X, Y, Z, t)$ 的庞大网络极其困难，因为在每一个微小的时刻 $t$，物体的几何拓扑都在变化，网络很容易陷入过拟合。

经典非刚体动力学提出了一种极具智慧的解耦思想——**规范参考系映射**：
- **规范空间（Canonical Space）**：定义一个“时间完全静止的标准姿态空间”（例如人体站立呈 T-pose 的基准状态）；在规范空间中，物体的几何轮廓与颜色是永恒不变的；
- **时空形变场（Deformation Field, $\Delta \mathbf{x}$）**：定义一个随时间连续变化的位移向量场。在任意时刻 $t$，空间中受力运动的物理点 $\mathbf{x}_t$，都可以通过形变场**映射回规范空间中的对应母体位置 $\mathbf{x}_{\text{canonical}}$**！

$$\mathbf{x}_{\text{canonical}} = \mathbf{x}_t + \Delta \mathbf{x}(\mathbf{x}_t, t)$$

<div align="center">

<img src="/figures/09-spatial-worlds/source/04-four-dimensional-worlds/dnerf-fig3.png" alt="D-NeRF 用时间条件形变网络把动态观测映射到统一规范空间，再由规范 NeRF 解释颜色与密度。" width="86%">

_图 9.4-2：D-NeRF 用时间条件形变网络把动态观测映射到统一规范空间，再由规范 NeRF 解释颜色与密度。 出处：[D-NeRF: Neural Radiance Fields for Dynamic Scenes，Albert Pumarola et al.，2021](https://arxiv.org/abs/2011.13961)。_

</div>

### 2. 三维场景流（Scene Flow）的瞬时速度场
在物理学中，空间点的三维位移对时间的导数定义了该点的瞬时运动速度向量，在计算机视觉中被称为**三维场景流（Scene Flow, $\mathbf{v}_t \in \mathbb{R}^3$）**：

$$\mathbf{v}_t(\mathbf{x}) = \frac{\partial \Delta \mathbf{x}(\mathbf{x}, t)}{\partial t}$$

场景流为动态世界模型赋予了直接的物理因果律——它不仅描述了“物体现在在哪里”，更精确量化了“物体下一毫秒将以何种速度飞向何方”。

<div align="center">

<img src="/figures/09-spatial-worlds/source/04-four-dimensional-worlds/nsff-fig2.png" alt="NSFF 通过前后向场景流连接相邻时刻的三维点，并联合辐射场与遮挡权重重建动态视频。" width="86%">

_图 9.4-3：NSFF 通过前后向场景流连接相邻时刻的三维点，并联合辐射场与遮挡权重重建动态视频。 出处：[Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes，Zhengqi Li et al.，2021](https://arxiv.org/abs/2011.13084)。_

</div>

---

## 9.4.2 核心数学推导一：时空形变网络与物理平滑正则化

我们来建立动态辐射场的完整正向推演与损失函数体系。

<div align="center">

<img src="/figures/09-spatial-worlds/source/04-four-dimensional-worlds/4dgs-fig3.png" alt="4DGS 以规范三维高斯和时空形变场构成动态场景管线，实现可实时渲染的四维表示。" width="86%">

_图 9.4-4：4DGS 以规范三维高斯和时空形变场构成动态场景管线，实现可实时渲染的四维表示。 出处：[4D Gaussian Splatting for Real-Time Dynamic Scene Rendering，Guanjun Wu et al.，2024](https://arxiv.org/abs/2310.08528)。_

</div>

### 1. 双阶段渲染计算图
对于在时刻 $t$ 发出的一条射线 $\mathbf{r}(s) = \mathbf{o}_t + s \mathbf{d}_t$ 上的任意采样点 $\mathbf{x}_t$：
1. **形变网络（Deformation MLP, $\Psi$）**：预测该点在当前时刻的三维空间位移增量：
   $$\Delta \mathbf{x}_t = \Psi(\mathbf{x}_t, t) \in \mathbb{R}^3$$
2. **规范坐标反演**：计算其在规范空间中的对应坐标：
   $$\mathbf{x}_{\text{can}} = \mathbf{x}_t + \Delta \mathbf{x}_t$$
3. **规范辐射场网络（Canonical MLP, $\Phi$）**：在规范空间中查询该点的物理密度 $\sigma$ 与发射颜色 $\mathbf{c}$：
   $$(\sigma, \mathbf{c}) = \Phi(\mathbf{x}_{\text{can}}, \mathbf{d}_t)$$
4. **体渲染积分**：将查询到的 $(\sigma, \mathbf{c})$ 代入上一节推导的体渲染求积公式，合成该时刻的最终图像。

### 2. 物理弹性平滑度正则化（Elastic Regularization）
真实世界中的刚体或弹性物体在运动时，相邻微元之间的距离不会发生剧烈撕裂。
为了防止神经网络产生非物理的剧烈空间畸变，系统引入了**空间梯度平滑度（Total Variation）**与**时间连续性**正则项：

$$\mathcal{L}_{\text{smooth}} = \frac{1}{N} \sum_{i=1}^N \left( \left\| \nabla_{\mathbf{x}} \Delta \mathbf{x}_t^{(i)} \right\|_F^2 + \lambda_t \left\| \frac{\partial \Delta \mathbf{x}_t^{(i)}}{\partial t} \right\|_2^2 \right)$$

**手算代入算例**：
设在规范空间中某物体中心位于原点 $\mathbf{x}_{\text{can}} = [0.0, 0.0, 0.0]^\top$。
在时刻 $t = 2.0\text{ s}$ 时，物体沿 $X$ 轴以 $1.5\text{ m/s}$ 匀速向右运动，且附带轻微的简谐振动位移 $\Delta x(t) = 1.5 t + 0.1 \sin(\pi t)$。
当前时刻观测到一个空间点 $\mathbf{x}_t = [3.0, 0.0, 0.0]^\top$。

1. 形变网络计算位移：
   $$\Delta x = -(1.5 \times 2.0 + 0.1 \sin(2\pi)) = -(3.0 + 0.0) = -3.0\text{ m}$$
2. 映射回规范空间坐标：
   $$\mathbf{x}_{\text{can}} = [3.0, 0.0, 0.0]^\top + [-3.0, 0.0, 0.0]^\top = [0.0, 0.0, 0.0]^\top$$
3. 规范网络查询：精准命中静止物体的中心几何，密度 $\sigma = 10.0$，颜色正常输出！

初等代数的手算结果极其优雅：通过一个简单的坐标减法，动态物体的时空演化被无缝还原为了静态场的稳定查询！

<details>
<summary><b>深入推导：非刚体连续介质力学格林-拉格朗日应变张量（Green-Lagrange Strain Tensor）数学推导（点击展开查看完整推导）</b></summary>

设形变映射为 $\boldsymbol{\phi}(\mathbf{X}) = \mathbf{X} + \mathbf{u}(\mathbf{X})$，其形变梯度张量（Deformation Gradient）为 $\mathbf{F} = \nabla_{\mathbf{X}} \boldsymbol{\phi} = \mathbf{I} + \nabla \mathbf{u}$。
右柯西-格林形变张量为 $\mathbf{C} = \mathbf{F}^\top \mathbf{F}$。
格林-拉格朗日应变张量（Green-Lagrange Strain Tensor）定义为：
$$\mathbf{E} = \frac{1}{2}(\mathbf{C} - \mathbf{I}) = \frac{1}{2}\left( \nabla \mathbf{u} + (\nabla \mathbf{u})^\top + (\nabla \mathbf{u})^\top \nabla \mathbf{u} \right)$$
在理想局部等距（Isometric）刚体旋转运动下，$\mathbf{F} \in SO(3) \implies \mathbf{C} = \mathbf{I} \implies \mathbf{E} = \mathbf{0}$。
惩罚应变能量 $\|\mathbf{E}\|_F^2$ 构成物理上最严密的局部刚性正则化损失函数。
</details>

---

## 9.4.3 核心数学推导二：时空维度重构与六维张量运算

在深度神经网络中处理四维时空数据时，输入特征张量通常包含六个维度：

$$\mathcal{X} \in \mathbb{R}^{B \times T \times X \times Y \times Z \times C}$$

<div align="center">

<img src="/figures/09-spatial-worlds/latex/04-four-dimensional-worlds/spacetime-axis-refactor.png" alt="六维时空状态在空间卷积时折叠批次与时间，在时间注意力时折叠批次与空间位置" width="86%">

_图 9.4-5：六维时空状态在空间卷积时折叠批次与时间，在时间注意力时折叠批次与空间位置。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

为了在显存有限的硬件上高效训练，系统采用**时空解耦轴重构算子（Spatiotemporal Reshape Operators）**：
1. **空间三维卷积阶段**：将批次维度 $B$ 与时间维度 $T$ 折叠为复合批次 $B \times T$，在三维体素空间 $(X, Y, Z)$ 上执行三维卷积：
   $$\mathcal{X}_{\text{spatial}} = \text{Reshape}(\mathcal{X}, [B \cdot T, X, Y, Z, C])$$
2. **时间序列注意力阶段**：将空间维度 $(X, Y, Z)$ 拍平折叠为复合批次 $B \times (X Y Z)$，沿纯时间轴 $T$ 执行自注意力时序交互：
   $$\mathcal{X}_{\text{temporal}} = \text{Reshape}(\mathcal{X}, [B \cdot (X Y Z), T, C])$$

这种时空轴交叉折叠机制，将四维全连接注意力的 $\mathcal{O}((T X Y Z)^2)$ 天文数字计算复杂度，大幅削减为了 $\mathcal{O}(T \cdot (XYZ)^2 + (XYZ) \cdot T^2)$ 的工程可实现规模！

<details>
<summary><b>深入推导：时空解耦注意力（Time-Space Decoupled Attention）与全维联合注意力的误差界证明（点击展开查看完整推导）</b></summary>

设时空联合注意力核矩阵为 $\mathbf{K}_{\text{joint}} \in \mathbb{R}^{(T S) \times (T S)}$，解耦克罗内克积注意力核为 $\mathbf{K}_{\text{decoupled}} = \mathbf{K}_T \otimes \mathbf{K}_S$。
根据矩阵极值低秩逼近定理，若物理场景满足平稳动力学假设（即空间结构分布与全局时间演化具有局部条件独立性），则两者的核算子差值在上确界范数下满足：
$$\|\mathbf{K}_{\text{joint}} - \mathbf{K}_T \otimes \mathbf{K}_S\|_2 \le \frac{1}{\sqrt{S}} \sum_{k=2}^{\min(T, S)} \sigma_k$$
其中奇异值 $\sigma_k$ 随空间高频衰减极快，证明了时空解耦策略在几乎无精度损失的前提下实现了数十倍的计算加速。
</details>

---

## 9.4.4 纯底层 PyTorch 代码实现：动态四维形变辐射场与时空解耦渲染引擎

下面我们使用纯底层 PyTorch 算子实现一个完整的动态四维形变网络、规范辐射场以及时空张量解耦计算引擎。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformationField(nn.Module):
    """
    时空形变场网络 (Deformation Network)
    输入时空坐标 (x_t, t)，预测空间位移增量 Delta x_t
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # 3D 空间坐标 + 1D 标量时间
        self.net = nn.Sequential(
            nn.Linear(3 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3) # 输出 (dx, dy, dz)
        )
        # 初始化最后一层权重为很小的值，使初始形变接近 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pts_t: torch.Tensor, time_t: torch.Tensor) -> torch.Tensor:
        """
        :param pts_t: (B, N_pts, 3) 空间采样点
        :param time_t: (B, N_pts, 1) 时间戳
        :return: (B, N_pts, 3) 位移场 Delta x
        """
        inputs = torch.cat([pts_t, time_t], dim=-1)
        delta_x = self.net(inputs)
        return delta_x

class CanonicalRadianceField(nn.Module):
    """
    规范空间静态辐射场 (Canonical Radiance Field)
    在规范参考系中查询几何密度 sigma 与 RGB 颜色
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.sigma_head = nn.Linear(hidden_dim, 1)
        self.color_head = nn.Linear(hidden_dim, 3)

    def forward(self, pts_can: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param pts_can: (B, N_pts, 3) 规范坐标点
        :return: (sigmas, colors)
        """
        feat = self.net(pts_can)
        # 密度必须为正，使用 ReLU
        sigmas = F.relu(self.sigma_head(feat)).squeeze(-1) # (B, N_pts)
        # 颜色使用 Sigmoid 约束在 [0, 1] 范围
        colors = torch.sigmoid(self.color_head(feat))      # (B, N_pts, 3)
        return sigmas, colors

class VolumeRenderingIntegrator:
    @staticmethod
    def render_rays(sigmas: torch.Tensor, colors: torch.Tensor, deltas: torch.Tensor):
        densities = sigmas * deltas
        alphas = 1.0 - torch.exp(-densities)
        cum_densities = torch.cumsum(densities, dim=-1)
        cum_densities_shifted = F.pad(cum_densities[..., :-1], (1, 0), value=0.0)
        transmittance = torch.exp(-cum_densities_shifted)
        weights = transmittance * alphas
        rendered_colors = (weights.unsqueeze(-1) * colors).sum(dim=1)
        return rendered_colors, weights

class Dynamic4DRenderer(nn.Module):
    """
    完整的动态四维神经辐射场渲染引擎
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.deform_field = DeformationField(hidden_dim=hidden_dim)
        self.canonical_field = CanonicalRadianceField(hidden_dim=hidden_dim)

    def forward(
        self, pts_t: torch.Tensor, time_t: torch.Tensor, deltas: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param pts_t: (B, N_pts, 3) 动态空间采样点
        :param time_t: (B, N_pts, 1) 时间戳
        :param deltas: (B, N_pts) 采样步长
        :return: (rendered_colors, sigmas, delta_x)
        """
        # 1. 计算时空位移场
        delta_x = self.deform_field(pts_t, time_t)

        # 2. 映射回规范空间坐标
        pts_can = pts_t + delta_x

        # 3. 查询规范空间密度与色彩
        sigmas, colors = self.canonical_field(pts_can)

        # 4. 执行离散体渲染积分
        rendered_colors, _ = VolumeRenderingIntegrator.render_rays(sigmas, colors, deltas)

        return rendered_colors, sigmas, delta_x

# ===================================================================
# 单元测试：时空形变求解与平滑度损失反向传播校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 2
    num_pts = 16

    renderer = Dynamic4DRenderer(hidden_dim=64)
    optimizer = torch.optim.Adam(renderer.parameters(), lr=1e-3)

    dummy_pts = torch.randn(batch_size, num_pts, 3)
    dummy_time = torch.full((batch_size, num_pts, 1), 0.5)
    dummy_deltas = torch.full((batch_size, num_pts), 0.1)

    # 1. 前向推演
    rendered_rgb, sigmas, delta_x = renderer(dummy_pts, dummy_time, dummy_deltas)

    # 2. 计算合成损失与形变正则化损失
    target_rgb = torch.rand(batch_size, 3)
    color_loss = F.mse_loss(rendered_rgb, target_rgb)
    smooth_loss = delta_x.pow(2).mean() # 惩罚过大形变
    total_loss = color_loss + 0.1 * smooth_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(f"[4D NeRF Test] 合成图像 RGB 形状: {rendered_rgb.shape}")
    print(f"[4D NeRF Test] 预测位移张量形状: {delta_x.shape}")
    print(f"[4D NeRF Test] 训练联合损失: {total_loss.item():.4f}")

    assert rendered_rgb.shape == (batch_size, 3), "体渲染输出形状不符！"
    assert delta_x.shape == (batch_size, num_pts, 3), "位移场张量维度不符！"
    assert not torch.isnan(total_loss), "损失出现 NaN！"
    print("✓ 动态四维时空形变场与规范辐射场渲染引擎单测全部通过！")
```

---

## 9.4.5 本节小结

回顾本节内容，我们建立了动态四维时空建模的完整认知脉络：
1. **时空解耦的几何哲学**：将复杂的动态物理演化解耦为静态的“规范空间”与连续变化的“时空形变场 $\Delta \mathbf{x}(\mathbf{x}, t)$”；
2. **场景流与物理正则化**：利用瞬时三维速度场刻画物理因果，结合弹性应变平滑度约束杜绝非物理撕裂；
3. **时空解耦注意力**：通过空间三维与时间一维的轴交叉重构，实现了大规模四维张量的高效轻量化运算。
