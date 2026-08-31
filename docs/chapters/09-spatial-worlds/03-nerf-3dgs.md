# 9.3 神经辐射场（NeRF）与 3D 高斯溅射（3DGS）基础

给定同一场景的多张照片，怎样合成相机从新位置看到的图像？关键是建立一个可查询的三维场景表示，并说明一条相机射线如何累积颜色。本节比较两种表示：NeRF 用神经网络表示连续辐射场，3DGS 用一组显式三维高斯表示场景。

<div align="center">
<img src="/figures/09-spatial-worlds/source/03-nerf-3dgs/nerf-fig1.png" alt="NeRF 用稀疏输入视角拟合连续辐射场，再从未见相机位姿合成新视角。" width="86%">

_图 9.3-1：NeRF 用稀疏输入视角拟合连续辐射场，再从未见相机位姿合成新视角。 出处：Ben Mildenhall et al.，[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)（2020），Figure 1。_
</div>

Mildenhall 等人提出神经辐射场（Neural Radiance Fields, NeRF），用一个以三维位置和观察方向为输入的 MLP 表示体密度与视角相关颜色，再通过体渲染合成新视角 [[Mildenhall et al., 2020]](https://arxiv.org/abs/2003.08934)。原始 NeRF 的网络查询和沿光线密集采样使训练与渲染较慢。Kerbl 等人提出 3D 高斯溅射（3D Gaussian Splatting, 3DGS），用显式三维高斯集合与可微分的基于瓦片的光栅化进行优化，并在论文数据集上报告实时新视角渲染 [[Kerbl et al., 2023]](https://arxiv.org/abs/2308.04079)。

下面从射线方程与透射率出发推导离散体渲染，再实现一个微型 NeRF，最后解释 3D 高斯如何投影到图像平面。

## 光线与体渲染：物理直觉与数学推导

每个像素对应一条从相机光心穿过成像平面的射线。

当我们使用相机拍摄一张照片时，相机传感器上的每一个像素（像素中心），都在空间中对应着一条从相机光心出发，穿过该像素并射向三维场景的射线。我们可以使用高中数学中的参数方程来精确描述这条光线。设相机光心（原点）的三维坐标为 $\mathbf{o} = (o_x, o_y, o_z)^\top$，光线的单位方向向量为 $\mathbf{d} = (d_x, d_y, d_z)^\top$，那么光线上任意一点的三维坐标 $\mathbf{r}(t)$ 可以表示为：

$$ \mathbf{r}(t) = \mathbf{o} + t \mathbf{d} $$

其中，$t \ge 0$ 表示光线在方向 $\mathbf{d}$ 上行进的距离（或时间参量）。

当这条光线穿过充满半透明介质（例如云雾或带有色彩的粒子）的三维空间时，介质会对光线产生两个核心影响：**发光（Emission）**和**吸收（Absorption）**。
令 $\mathbf{c}(\mathbf{r}(t),\mathbf{d})$ 表示位置与视角相关的颜色，$\sigma(\mathbf{r}(t))\ge 0$ 表示单位距离上的衰减率。对很短的距离 $dt$，发生终止的概率近似为 $\sigma(t)dt$。

### 累积透射率与体渲染方程

考虑光线从近端 $t_n$ 穿行到远端 $t_f$。对于光线路径上某一点 $t$，它发出的光想要最终到达相机光心 $\mathbf{o}$，必须穿过区间 $[t_n, t]$。在这一段路径中，光线没有被阻挡的概率被称为累积透射率（Transmittance），记为 $T(t)$。

基于概率论的乘法法则，光线在行进微小距离 $dt$ 后未被阻挡的概率为 $1 - \sigma(t) dt$。因此，累积透射率随距离的变化率可以表示为一个微分方程：

$$ \frac{dT(t)}{dt} = - \sigma(t) T(t) $$

令起始点 $T(t_n)=1$，解这个微分方程得到：

$$ T(t) = \exp \left( - \int_{t_n}^{t} \sigma(s) ds \right) $$

相机传感器在最终接收到的颜色 $C(\mathbf{r})$，是整条光线上所有点发出颜色的积分总和，但每一点的贡献都必须乘上它能够到达相机的概率（即透射率 $T(t)$），以及该点本身的密度 $\sigma(t)$：

$$ C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(t) \mathbf{c}(t, \mathbf{d}) dt $$

这就是 NeRF 使用的发射—吸收体渲染形式。

### 离散化近似推导

数值计算时沿光线采样 $N$ 个点 $t_1,\ldots,t_N$，把积分区间分成小段，第 $i$ 段长度为 $\delta_i=t_{i+1}-t_i$。

在第 $i$ 个区间 $[t_i,t_{i+1}]$ 内，假设密度 $\sigma_i$ 和颜色 $\mathbf c_i$ 保持恒定。该段使光线终止的概率（不透明度）为 $\alpha_i$，剩余透射率为 $1-\alpha_i$：

$$ \alpha_i = 1 - \exp(-\sigma_i \delta_i) $$

连续积分可以近似为：

$$ \hat{C}(\mathbf{r}) = \sum_{i=1}^N T_i \alpha_i \mathbf{c}_i $$

其中，累积透射率 $T_i$ 是在此之前所有区间的透射概率之积：

$$ T_i = \prod_{j=1}^{i-1} (1 - \alpha_j) = \exp \left( - \sum_{j=1}^{i-1} \sigma_j \delta_j \right) $$

<div align="center">
<img src="/figures/09-spatial-worlds/latex/03-nerf-3dgs/front-to-back-transmittance.png" alt="沿射线从近到远，每个样本的颜色权重由此前透射率连乘与本点不透明度共同决定" width="86%">

_图 9.3-2：第 i 个样本只有在前方样本均未遮挡时才可见，因此其颜色权重为此前透射率 T_i 与本点不透明度 α_i 的乘积。本文根据上式绘制。_
</div>

这些算子对有限的 $\sigma_i$ 和 $\mathbf{c}_i$ 可微，因此可以通过反向传播更新生成它们的网络参数。

## 神经辐射场（NeRF）原理

NeRF 使用 MLP 隐式表示连续三维空间中的颜色场和密度场。

<div align="center">
<img src="/figures/09-spatial-worlds/source/03-nerf-3dgs/nerf-fig2b.png" alt="NeRF 的方法图串联相机射线采样、位置与方向编码、MLP 查询以及体渲染积分。" width="86%">

_图 9.3-3：NeRF 的方法图串联相机射线采样、位置与方向编码、MLP 查询以及体渲染积分。 出处：Ben Mildenhall et al.，[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)（2020），Figure 2。_
</div>

### 连续场景的MLP参数化

我们定义一个连续函数 $F_{\Theta}$，其输入为三维空间坐标 $\mathbf{x} = (x, y, z)$ 和视角方向 $\mathbf{d} = (\theta, \phi)$，输出为体密度 $\sigma$ 和RGB颜色 $\mathbf{c}$：

$$ F_{\Theta}: (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma) $$

其中 $\Theta$ 为神经网络的权重。之所以将视角方向 $\mathbf{d}$ 也作为输入，是为了让模型能够学习到非朗伯体（Non-Lambertian）效应，即真实世界中物体表面在不同观察角度下呈现的高光和反射变化。为了保证物理意义，$\sigma$ 只与空间位置有关（无论从哪个方向看，该点的物质密度都是不变的），而颜色 $\mathbf{c}$ 则同时取决于位置和视角。

### 位置编码（Positional Encoding）

如果直接将低维的坐标 $(\mathbf{x}, \mathbf{d})$ 输入给标准的 ReLU MLP，网络会倾向于学习低频的平滑函数，导致渲染出的图像丢失纹理细节，显得模糊不堪。这种现象被称为深度网络的谱偏差（Spectral Bias）。

为了使神经网络能够捕捉高频细节，Mildenhall 等人借鉴了 Transformer 中的思想，引入了位置编码 $\gamma(\cdot)$，将低维输入映射到高维空间：

$$ \gamma(p) = \left( \sin(2^0 \pi p), \cos(2^0 \pi p), \ldots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p) \right) $$

通过这种映射，标量坐标 $p$ 被展开成了 $2L$ 维的向量。在实际操作中，对三维坐标 $\mathbf{x}$ 和方向 $\mathbf{d}$ 分别采用不同长度 $L$ 的位置编码。

### 损失函数与优化

训练时从多视角图像采样像素，根据已知相机参数生成射线，查询 MLP 得到颜色与密度，再用离散体渲染合成预测颜色 $\hat C(\mathbf r)$。原始 NeRF 使用预测颜色和真实像素颜色之间的平方误差：

$$ \mathcal{L} = \sum_{\mathbf{r} \in \mathcal{R}} \left\| \hat{C}(\mathbf{r}) - C(\mathbf{r}) \right\|_2^2 $$

## 微型 NeRF 代码实现

下面实现位置编码、微型 MLP 和离散体渲染。为突出张量流，省略相机射线生成、分层采样与训练循环。

```python
import torch
from torch import nn

def positional_encoding(x, L):
    """
    计算位置编码
    x: 形状为 (..., D) 的输入张量
    L: 频率级数
    """
    out = [x]
    for i in range(L):
        out.append(torch.sin(2.0 ** i * torch.pi * x))
        out.append(torch.cos(2.0 ** i * torch.pi * x))
    return torch.cat(out, dim=-1)

class MicroNeRF(nn.Module):
    def __init__(self, D=8, W=256, L_pos=10, L_dir=4):
        super().__init__()
        self.L_pos = L_pos
        self.L_dir = L_dir

        # 输入维度: 坐标(3)扩展后 + 视角(3)扩展后
        pos_dim = 3 + 3 * 2 * L_pos
        dir_dim = 3 + 3 * 2 * L_dir

        # 共享特征提取网络
        self.pts_linears = nn.ModuleList([
            nn.Linear(
                pos_dim if i == 0 else W + pos_dim if i == 4 else W,
                W
            )
            for i in range(D)
        ])

        # 密度输出层
        self.density_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)

        # 颜色输出层 (结合方向信息)
        self.views_linears = nn.Sequential(
            nn.Linear(W + dir_dim, W // 2),
            nn.ReLU(),
            nn.Linear(W // 2, 3)
        )

    def forward(self, pts, views):
        """
        pts: 采样点坐标, 形状 [batch_size, num_samples, 3]
        views: 视角方向, 形状 [batch_size, num_samples, 3]
        """
        # 1. 计算位置编码
        pts_encoded = positional_encoding(pts, self.L_pos)
        views_encoded = positional_encoding(views, self.L_dir)

        # 2. 通过共享 MLP 提取几何特征
        h = pts_encoded
        for i, l in enumerate(self.pts_linears):
            # 第 5 个线性层读取位置编码跳连
            if i == 4:
                h = torch.cat([pts_encoded, h], dim=-1)
            h = l(h)
            h = torch.relu(h)

        # 3. 输出密度 (通过 Softplus 确保正值)
        density = torch.nn.functional.softplus(self.density_linear(h))

        # 4. 结合视角特征输出颜色 (通过 Sigmoid 限制在 [0,1])
        feature = self.feature_linear(h)
        h = torch.cat([feature, views_encoded], -1)
        rgb = torch.sigmoid(self.views_linears(h))

        return rgb, density

def volume_render(rgb, density, z_vals, ray_dirs):
    """
    离散化体渲染方程的实现
    rgb: [batch_size, num_samples, 3]
    density: [batch_size, num_samples, 1]
    z_vals: [batch_size, num_samples] 每条光线上采样点的距离
    ray_dirs: [batch_size, 3]
    """
    # 计算相邻采样点间的距离 delta
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    # 为最后一个点补充一个较大的距离，并保持设备与数据类型一致
    dists = torch.cat([dists, dists[..., :1].new_full(dists[..., :1].shape, 1e10)], -1)

    # 考虑射线方向对距离的缩放
    dists = dists * torch.norm(ray_dirs[..., None, :], dim=-1)

    # 公式(4): 计算 alpha_i = 1 - exp(-sigma_i * delta_i)
    alpha = 1.0 - torch.exp(-density.squeeze(-1) * dists)

    # 公式(5): 计算累积透射率 T_i
    # 累积连乘: T_i = prod(1 - alpha_{j<i})
    # 为了数值稳定性，我们在累积前向计算中添加极小项
    prefix = alpha.new_ones((*alpha.shape[:-1], 1))
    T = torch.cumprod(torch.cat([prefix, 1.0 - alpha + 1e-10], -1), -1)[..., :-1]

    # 公式(4)最终合并，计算权重 w_i = T_i * alpha_i
    weights = alpha * T

    # 对每条光线上的颜色进行加权求和
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    return rgb_map
```

## 空间的新基底：3D高斯溅射（3DGS）

原始 NeRF 每条射线要查询许多采样点，并让每个点通过 MLP，因此训练和渲染较慢。后续工作可用网格、哈希编码或缓存加速；3DGS 则换成显式基元与光栅化路线。

<div align="center">
<img src="/figures/09-spatial-worlds/source/03-nerf-3dgs/instantngp-fig1.png" alt="Instant-NGP 用多分辨率哈希编码显著缩短 NeRF 等神经图形基元的训练时间，并展示不同训练时长的输出质量。" width="86%">

_图 9.3-4：Instant-NGP 用多分辨率哈希编码显著缩短 NeRF 等神经图形基元的训练时间，并展示不同训练时长的输出质量。 出处：Thomas Müller et al.，[Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://arxiv.org/abs/2201.05989)（2022），Figure 1。_
</div>

**3D 高斯溅射（3D Gaussian Splatting, 3DGS）**用可优化的三维高斯集合表示场景。渲染时把高斯投影到图像平面并做透明度合成，不需要像原始 NeRF 那样逐点查询深层 MLP。

### 显式的三维高斯基元

在 3DGS 中，三维场景由许多高斯基元表示。第 $k$ 个基元包含：

<div align="center">
<img src="/figures/09-spatial-worlds/source/03-nerf-3dgs/3dgs-fig2.png" alt="3DGS 从稀疏 SfM 点初始化三维高斯，交替优化与密度控制，再通过可微光栅化生成图像。" width="86%">

_图 9.3-5：3DGS 从稀疏 SfM 点初始化三维高斯，交替优化与密度控制，再通过可微光栅化生成图像。 出处：Bernhard Kerbl et al.，[3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079)（2023），Figure 2。_
</div>

1. **均值中心** $\boldsymbol{\mu}_k \in \mathbb{R}^3$：高斯体在三维空间中的中心位置。
2. **协方差矩阵** $\boldsymbol{\Sigma}_k \in \mathbb{R}^{3 \times 3}$：决定了高斯椭球的大小和朝向。
3. **不透明度** $\alpha_k \in [0, 1]$：表示该基元的不透明度。
4. **球谐系数（Spherical Harmonics, SH）**：编码视角依赖的颜色 $\mathbf{c}_k$。

对于三维空间中的任意一点 $\mathbf{x}$，第 $k$ 个高斯分布对该点密度的贡献为：

$$ G(\mathbf{x}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) = \exp \left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k) \right) $$

为了保持协方差矩阵有效，3DGS 用缩放矩阵 $\mathbf S$ 和旋转矩阵 $\mathbf R$ 参数化它：
$$ \boldsymbol{\Sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^\top \mathbf{R}^\top $$

### 溅射（Splatting）与可微光栅化

把三维高斯投影到二维像素平面的过程称为**溅射（Splatting）**。在局部线性近似下，三维椭球会在图像上形成二维椭圆。

给定相机变换 $\mathbf W$ 和投影在高斯中心处的雅可比 $\mathbf J$，三维协方差可近似投影为二维协方差：

$$ \boldsymbol{\Sigma}' = \mathbf{J} \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^\top \mathbf{J}^\top $$

投影完成后，对覆盖同一像素的高斯按深度排序并做 Alpha 合成，其形式与离散体渲染相似：

$$ C = \sum_{i \in \mathcal{N}} c_i \alpha'_i \prod_{j=1}^{i-1} (1 - \alpha'_j) $$

这里的 $\mathcal{N}$ 是所有在深度（从近到远）上排序后的、且覆盖当前像素的高斯基元集合。$\alpha'_i$ 是由二维高斯求值并乘以基础不透明度 $\alpha_k$ 得到的像素级不透明度。

3DGS 使用基于图块的筛选、排序和可微光栅化评估这些贡献。原论文在指定数据集、分辨率与硬件上报告实时渲染；实际速度仍取决于高斯数量、图像大小和实现。显式表示减少了 MLP 查询，却要存储、排序并合成大量基元。
