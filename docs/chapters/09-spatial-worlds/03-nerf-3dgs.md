# 神经辐射场（NeRF）与3D高斯溅射（3DGS）基础

如何将真实世界的连续三维空间及其光影表现，转化为计算机能够理解并高效渲染的数学表达？这是计算机图形学和三维视觉领域半个世纪以来的核心命题。传统的三维表示方法（如体素网络、点云和多边形网格）在表达复杂拓扑结构或实现高保真度的新视角合成时，往往会遭遇存储空间爆炸或几何离散化带来的失真。

2020年，Mildenhall等人提出神经辐射场（Neural Radiance Fields, NeRF）[[Mildenhall et al., 2020]](https://arxiv.org/abs/2003.08934)，在三维视觉领域引发了一场深刻的范式革命。NeRF摒弃了显式的离散几何表示，转而将整个连续场景编码为一个隐式的多层感知机（MLP）权重矩阵中。然而，NeRF基于多层感知机的密集光线采样和体渲染计算成本极其高昂。为了打破这一计算瓶颈，Kerbl等人于2023年提出了3D高斯溅射（3D Gaussian Splatting, 3DGS）[[Kerbl et al., 2023]](https://arxiv.org/abs/2308.04079)，将场景重新解构为显式的非结构化三维高斯分布集合，并结合可微光栅化技术，实现了高质量且实时的场景渲染。

在本节中，我们将从最基础的直线方程和光学原理出发，严格推导连续空间中的体渲染方程，进而剖析NeRF的数学机制与代码实现，最后过渡到当前极具统治力的3D高斯溅射架构。

## 光线与体渲染：物理直觉与数学推导

要理解如何合成一个三维场景的新视角图像，我们首先需要回到高中物理中的小孔成像原理与解析几何。

当我们使用相机拍摄一张照片时，相机传感器上的每一个像素（像素中心），都在空间中对应着一条从相机光心出发，穿过该像素并射向三维场景的射线。我们可以使用高中数学中的参数方程来精确描述这条光线。设相机光心（原点）的三维坐标为 $\mathbf{o} = (o_x, o_y, o_z)^\top$，光线的单位方向向量为 $\mathbf{d} = (d_x, d_y, d_z)^\top$，那么光线上任意一点的三维坐标 $\mathbf{r}(t)$ 可以表示为：

$$ \mathbf{r}(t) = \mathbf{o} + t \mathbf{d} $$

其中，$t \ge 0$ 表示光线在方向 $\mathbf{d}$ 上行进的距离（或时间参量）。

当这条光线穿过充满半透明介质（例如云雾或带有色彩的粒子）的三维空间时，介质会对光线产生两个核心影响：**发光（Emission）**和**吸收（Absorption）**。
假设空间中每一点 $\mathbf{r}(t)$ 都会向任意方向发出颜色为 $\mathbf{c}(\mathbf{r}(t), \mathbf{d})$ 的光，同时，空间中存在一定密度的“粒子”，这些粒子会阻挡光线的传播。我们定义体密度（Volume Density）$\sigma(\mathbf{r}(t))$，表示光线在点 $\mathbf{r}(t)$ 处行进微小距离 $dt$ 时，被粒子阻挡的概率。

### 累积透射率与体渲染方程

考虑光线从近端 $t_n$ 穿行到远端 $t_f$。对于光线路径上某一点 $t$，它发出的光想要最终到达相机光心 $\mathbf{o}$，必须穿过区间 $[t_n, t]$。在这一段路径中，光线没有被阻挡的概率被称为累积透射率（Transmittance），记为 $T(t)$。

基于概率论的乘法法则，光线在行进微小距离 $dt$ 后未被阻挡的概率为 $1 - \sigma(t) dt$。因此，累积透射率随距离的变化率可以表示为一个微分方程：

$$ \frac{dT(t)}{dt} = - \sigma(t) T(t) $$

对该公式求解，并假设在起始点 $T(t_n) = 1$，我们可以得到累积透射率的积分形式：

$$ T(t) = \exp \left( - \int_{t_n}^{t} \sigma(s) ds \right) $$

相机传感器在最终接收到的颜色 $C(\mathbf{r})$，是整条光线上所有点发出颜色的积分总和，但每一点的贡献都必须乘上它能够到达相机的概率（即透射率 $T(t)$），以及该点本身的密度 $\sigma(t)$：

$$ C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(t) \mathbf{c}(t, \mathbf{d}) dt $$

该公式就是经典的体渲染方程（Volume Rendering Equation）。它不仅是计算机图形学中渲染半透明材质的基础，更是神经辐射场的理论内核。

### 离散化近似推导

然而，计算机无法直接计算连续的积分。我们需要对其进行离散化近似求解。沿光线方向采样 $N$ 个离散点，对应距离为 $t_1, t_2, \ldots, t_N$。我们将积分区间划分为 $N$ 个微小线段，第 $i$ 段的长度为 $\delta_i = t_{i+1} - t_i$。

在第 $i$ 个微小区间 $[t_i, t_{i+1}]$ 内，假设密度 $\sigma_i$ 和颜色 $\mathbf{c}_i$ 保持恒定。光线穿过该微小区间后的透射概率定义为 $\alpha_i$（也称为不透明度，Opacity）：

$$ \alpha_i = 1 - \exp(-\sigma_i \delta_i) $$

此时，连续的积分该公式可以通过前向差分转化为离散的黎曼和：

$$ \hat{C}(\mathbf{r}) = \sum_{i=1}^N T_i \alpha_i \mathbf{c}_i $$

其中，累积透射率 $T_i$ 是在此之前所有区间的透射概率之积：

$$ T_i = \prod_{j=1}^{i-1} (1 - \alpha_j) = \exp \left( - \sum_{j=1}^{i-1} \sigma_j \delta_j \right) $$

上述离散化公式是严格可微的，这意味着我们可以利用现代深度学习框架的自动求导机制，通过反向传播来优化每个采样点的 $\sigma_i$ 和 $\mathbf{c}_i$。

## 神经辐射场（NeRF）原理

在明确了体渲染的数学模型后，NeRF 的核心思想呼之欲出：**使用多层感知机（MLP）来隐式地表示连续三维空间中的颜色场和密度场**。

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

NeRF的训练过程极致简约：我们从给定的多视角图像数据集中随机采样一组像素，根据相机参数发射射线。通过查询 MLP 得到每条射线上的颜色与密度，使用体渲染该公式计算出该射线的预测颜色 $\hat{C}(\mathbf{r})$。损失函数即为预测颜色与真实像素颜色 $C(\mathbf{r})$ 之间的均方误差（MSE）：

$$ \mathcal{L} = \sum_{\mathbf{r} \in \mathcal{R}} \left\| \hat{C}(\mathbf{r}) - C(\mathbf{r}) \right\|_2^2 $$

## 微型 NeRF 代码实现

(**下面，我们将用代码严谨地复现NeRF的前向计算流程**)，包括射线采样、位置编码和离散体渲染。为了简洁，我们省略了实际复杂的射线生成（即相机参数解析）和分层采样策略。

```{.python .input}
#@tab pytorch
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
        self.pts_linears = nn.ModuleList(
            [nn.Linear(pos_dim, W)] + 
            [nn.Linear(W, W) if i != 4 else nn.Linear(W + pos_dim, W) for i in range(D-1)]
        )
        
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
            h = l(h)
            h = torch.relu(h)
            # 在第 5 层注入残差连接
            if i == 4:
                h = torch.cat([pts_encoded, h], -1)
                
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
    # 为最后一个点补充一个极大的距离
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)
    
    # 考虑射线方向对距离的缩放
    dists = dists * torch.norm(ray_dirs[..., None, :], dim=-1)
    
    # 公式(4): 计算 alpha_i = 1 - exp(-sigma_i * delta_i)
    alpha = 1.0 - torch.exp(-density.squeeze(-1) * dists)
    
    # 公式(5): 计算累积透射率 T_i
    # 累积连乘: T_i = prod(1 - alpha_{j<i})
    # 为了数值稳定性，我们在累积前向计算中添加极小项
    T = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    
    # 公式(4)最终合并，计算权重 w_i = T_i * alpha_i
    weights = alpha * T
    
    # 对每条光线上的颜色进行加权求和
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    return rgb_map
```

## 空间的新基底：3D高斯溅射（3DGS）

NeRF 带来了惊艳的效果，但也暴露出了致命缺陷。一条射线上通常需要密集采样成百上千个点，每一个点都需要通过整个 MLP 进行前向传播。这种基于隐式函数的体渲染方式使得实时渲染（例如达到 $\geq 30$ FPS 的高分辨率输出）几乎成为奢望。

此时，**3D高斯溅射（3D Gaussian Splatting, 3DGS）** 横空出世。它通过将隐式连续场转化为显式的离散三维高斯分布，从而彻底绕过了庞大的 MLP 前向计算。如果说 NeRF 是在使用一个复杂的微积分黑盒来推导空间，那么 3DGS 则是像泼墨画一般，用无数个具有物理属性的椭球（三维高斯）直接堆砌出逼真的世界。

### 显式的三维高斯基元

在 3DGS 中，三维场景被表示为数以百万计的“高斯基元”。每一个第 $k$ 个高斯基元由以下参数精确定义：
1. **均值中心** $\boldsymbol{\mu}_k \in \mathbb{R}^3$：高斯体在三维空间中的中心位置。
2. **协方差矩阵** $\boldsymbol{\Sigma}_k \in \mathbb{R}^{3 \times 3}$：决定了高斯椭球的大小和朝向。
3. **不透明度** $\alpha_k \in [0, 1]$：表示该基元的不透明度。
4. **球谐系数（Spherical Harmonics, SH）**：编码视角依赖的颜色 $\mathbf{c}_k$。

对于三维空间中的任意一点 $\mathbf{x}$，第 $k$ 个高斯分布对该点密度的贡献为：

$$ G(\mathbf{x}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) = \exp \left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k) \right) $$

为了确保协方差矩阵 $\boldsymbol{\Sigma}$ 在优化过程中始终是半正定的，3DGS 采用了精妙的参数化手段。它将协方差分解为缩放矩阵 $\mathbf{S}$ 和旋转矩阵 $\mathbf{R}$：
$$ \boldsymbol{\Sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^\top \mathbf{R}^\top $$

### 溅射（Splatting）与可微光栅化

如何将这几百万个三维高斯“投影”到二维像素平面上？在图形学中，这个过程被称为**溅射（Splatting）**。这也许是整个框架中最需借助直觉的部分：想象你向一堵玻璃墙投掷一个个柔软的水球，水球在撞击玻璃时会压扁成椭圆形的印记，我们将三维的高斯分布投影到二维成像平面，也会得到二维的高斯分布。

在严谨的数学视角下，给定相机的观测矩阵 $\mathbf{W}$ 和射影变换的雅可比矩阵 $\mathbf{J}$，三维空间的协方差 $\boldsymbol{\Sigma}$ 可以近似投影为二维像素平面的协方差 $\boldsymbol{\Sigma}'$：

$$ \boldsymbol{\Sigma}' = \mathbf{J} \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^\top \mathbf{J}^\top $$

投影完成后，在特定的二维像素点处，我们需要对所有覆盖该像素的高斯印记进行颜色合成。基于传统的 Alpha 合成原理，其公式与体渲染极其相似：

$$ C = \sum_{i \in \mathcal{N}} c_i \alpha'_i \prod_{j=1}^{i-1} (1 - \alpha'_j) $$

这里的 $\mathcal{N}$ 是所有在深度（从近到远）上排序后的、且覆盖当前像素的高斯基元集合。$\alpha'_i$ 是由二维高斯求值并乘以基础不透明度 $\alpha_k$ 得到的像素级不透明度。

通过基于平铺（Tile-based）的并行排序算法和高度优化的 CUDA 算子，3DGS 能够以极高的帧率（通常达到实时 100+ FPS）直接评估上述公式。由于没有任何深层神经网络的参与，仅仅依赖矩阵乘法和显式求和，它彻底解放了计算资源。
