# 9.3 神经辐射场 (NeRF) 与三维高斯泼溅 (3DGS)

在前面的章节中，我们学习了如何通过相机几何投影与 BEV 栅格化将多视角图像转化为离散的特征网格。然而，物理世界中的真实物体往往拥有极其细腻的微观几何结构（如毛发、树叶、反光金属与透明玻璃）。如果仅用粗糙的立方体体素来描述世界，分辨率将严重受限，且无法呈现物体随视角变化而产生的逼真高光反光。

2020 年，加州大学伯克利分校的 Ben Mildenhall、Matthew Tancik 等人提出了震撼计算机视觉界的 **神经辐射场（Neural Radiance Fields, NeRF）**。NeRF 放弃了显式的多边形网格或体素，将整个连续三维空间压缩为一个全连接神经网络，开创了隐式可微体渲染的新纪元。

2023 年，INRIA 与蔚蓝海岸大学的 Bernhard Kerbl 等人进一步推出了 **三维高斯泼溅（3D Gaussian Splatting, 3DGS）**。3DGS 结合了显式三维高斯椭球基元与硬件级 GPU 瓦片光栅化，在维持照片级渲染质量的同时，将渲染帧率直接拉升到了惊人的 $100+\text{ FPS}$ 实时交互速度。

<div align="center">

<img src="/figures/09-spatial-worlds/source/03-nerf-3dgs/nerf-fig1.png" alt="NeRF 用稀疏输入视角拟合连续辐射场，再从未见相机位姿合成新视角。" width="86%">

_图 9.3-1：NeRF 用稀疏输入视角拟合连续辐射场，再从未见相机位姿合成新视角。 出处：[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis，Ben Mildenhall et al.，2020](https://arxiv.org/abs/2003.08934)。_

</div>

---

## 9.3.1 物理与光学基石：体积辐射传输与新视角合成演进

要理解辐射场与高斯渲染的数学内核，我们首先需要从经典大气光学中的光线体积传输物理定律讲起。

### 1. 经典物理中的光线衰减与微积分体渲染
想象一束清晨的太阳光穿过充满薄雾的森林：
- 光线在空气中直线传播时，每穿过一小段微元微小的距离 $ds$，就会有一部分光子被空气中的微小水滴粒子吸收或散射（这一性质在物理学中被称为**体积不透明度 / 体积消光系数 $\sigma$**，单位为 $\text{m}^{-1}$）；
- 与此同时，悬浮在空中的发光微粒自身也在向周围辐射出色彩光芒（称为**辐射发光度 $\mathbf{c}$**）。

当一束光线最终穿过薄雾射入我们的眼睛时，视网膜接收到的总颜色，正是**光线上所有微元发射出的光芒，沿着传播路径经历层层衰减后累加积攒的积分总和**！

<div align="center">

<img src="/figures/09-spatial-worlds/latex/03-nerf-3dgs/front-to-back-transmittance.png" alt="沿射线从近到远，每个样本的颜色权重由此前透射率连乘与本点不透明度共同决定" width="86%">

_图 9.3-2：沿射线从近到远，每个样本的颜色权重由此前透射率连乘与本点不透明度共同决定。本文绘制；TikZ/LaTeX 编译。_

</div>

### 2. 从隐式连续函数到显式高斯基元的演进
- **NeRF（连续隐式场）**：用一个神经网络输入空间坐标 $(x, y, z)$ 与观察角度 $(\theta, \phi)$，输出该点的密度 $\sigma$ 与 RGB 颜色 $\mathbf{c}$。渲染时必须沿光线密集采样数百个点执行多次网络推理，计算开销巨大（单帧渲染耗时数秒）；
- **3DGS（显式各向异性高斯球）**：将连续空间离散化为数百万个可自由拉伸、旋转、半透明的**三维高斯椭球（3D Gaussians）**。通过将三维椭球直接“泼溅（Splatting）”投影到二维屏幕上，利用 GPU 硬件并行排序完成极速合成。

<div align="center">

<img src="/figures/09-spatial-worlds/source/03-nerf-3dgs/instantngp-fig1.png" alt="Instant-NGP 用多分辨率哈希编码显著缩短 NeRF 等神经图形基元的训练时间，并展示不同训练时长的输出质量。" width="86%">

_图 9.3-3：Instant-NGP 用多分辨率哈希编码显著缩短 NeRF 等神经图形基元的训练时间，并展示不同训练时长的输出质量。 出处：[Instant Neural Graphics Primitives with a Multiresolution Hash Encoding，Thomas Müller et al.，2022](https://arxiv.org/abs/2201.05989)。_

</div>

---

## 9.3.2 核心数学推导一：NeRF 体积渲染积分与离散数值求积

设从相机光心 $\mathbf{o}$ 出发、沿方向单位向量 $\mathbf{d}$ 发出一条视线射线（Ray）：

$$\mathbf{r}(t) = \mathbf{o} + t \mathbf{d}, \quad t \in [t_n, t_f]$$

<div align="center">

<img src="/figures/09-spatial-worlds/source/03-nerf-3dgs/nerf-fig2b.png" alt="NeRF 的方法图串联相机射线采样、位置与方向编码、MLP 查询以及体渲染积分。" width="86%">

_图 9.3-4：NeRF 的方法图串联相机射线采样、位置与方向编码、MLP 查询以及体渲染积分。 出处：[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis，Ben Mildenhall et al.，2020](https://arxiv.org/abs/2003.08934)。_

</div>

### 1. 连续辐射传输方程
根据比尔-朗伯定律（Beer-Lambert Law），光线从起点 $t_n$ 传播到位置 $t$ 时，未被前方介质阻挡的**累积透射率（Transmittance）**为：

$$T(t) = \exp\left( -\int_{t_n}^t \sigma(\mathbf{r}(s)) ds \right)$$

相机光心最终接收到的期望色彩值 $C(\mathbf{r})$ 为整条射线上颜色的连续线积分：

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d}) dt$$

### 2. 离散数值求积公式（Numerical Quadrature）
由于神经网络无法直接求解连续解析积分，我们在射线区间 $[t_n, t_f]$ 内均匀或分层采样 $N$ 个离散点 $t_1 < t_2 < \dots < t_N$。第 $i$ 个采样区间的步长为 $\delta_i = t_{i+1} - t_i$。

连续积分被离散化为极其优美的初等代数求和公式：

$$\hat{C}(\mathbf{r}) = \sum_{i=1}^N T_i \alpha_i \mathbf{c}_i$$

其中：
- **第 $i$ 点的不透明度（Opacity）**：$\alpha_i = 1 - \exp(-\sigma_i \delta_i) \in [0, 1]$；
- **到达第 $i$ 点的累积透射率**：$T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right) = \prod_{j=1}^{i-1} (1 - \alpha_j)$。

> **初等代数直觉**：
> 观察权重因子 $w_i = T_i \alpha_i$：
> $T_i$ 描述了“光线能够活着穿过前面 $i-1$ 层障碍物的概率”；$\alpha_i$ 描述了“光线恰好在第 $i$ 个采样点被拦截并显色的概率”。
> 这一前向累乘求和过程，完全等价于图形学中经典的前后 Alpha 图像透明度混合！

**手算代入算例**：
设射线上仅采样 2 个点，步长 $\delta_1 = \delta_2 = 1.0\text{ m}$。
- 第 1 点：密度 $\sigma_1 = 0.6931$（$\exp(-0.6931) \approx 0.5$），发射红色 $\mathbf{c}_1 = [1.0, 0.0, 0.0]^\top$；
- 第 2 点：密度 $\sigma_2 = 10.0$（极度不透明，$\exp(-10.0) \approx 0.0$），发射蓝色 $\mathbf{c}_2 = [0.0, 0.0, 1.0]^\top$。

1. 计算第 1 点：
   - 透射率：$T_1 = 1.0$（前面没有任何遮挡）；
   - 不透明度：$\alpha_1 = 1 - \exp(-0.6931 \times 1.0) = 1 - 0.5 = 0.5$；
   - 颜色贡献权重：$w_1 = T_1 \alpha_1 = 1.0 \times 0.5 = 0.5$；
2. 计算第 2 点：
   - 透射率：$T_2 = T_1 (1 - \alpha_1) = 1.0 \times (1 - 0.5) = 0.5$；
   - 不透明度：$\alpha_2 = 1 - \exp(-10.0) \approx 1 - 0 = 1.0$；
   - 颜色贡献权重：$w_2 = T_2 \alpha_2 = 0.5 \times 1.0 = 0.5$；
3. 合成最终像素颜色：
   $$\hat{C} = w_1 \mathbf{c}_1 + w_2 \mathbf{c}_2 = 0.5 \times [1, 0, 0]^\top + 0.5 \times [0, 0, 1]^\top = [0.5, 0.0, 0.5]^\top$$

结果合成出了一半红一半蓝的紫色！整个过程没有任何微积分难题，全部是清晰明了的初等代数代入。

<details>
<summary><b>深入推导：比尔-朗伯微分衰减模型与体积渲染积分变分严格推导（点击展开查看完整推导）</b></summary>

设光线沿射线 $t$ 传播，其光强为 $I(t)$。在微元区间 $[t, t + dt]$ 内，由介质吸收导致的衰减量与当前光强和密度成正比：
$$\frac{dI(t)}{dt} = -\sigma(t) I(t) + \sigma(t) \mathbf{c}(t)$$
这是一阶非齐次线性常微分方程。求解对应齐次方程可得积分因子即累积透射率：
$$T(t) = \exp\left( -\int_{t_n}^t \sigma(s) ds \right)$$
利用常数变易法代入边界条件 $I(t_n) = 0$，对微分方程在 $[t_n, t_f]$ 积分，可严格导出：
$$I(t_f) = \int_{t_n}^{t_f} T(t) \sigma(t) \mathbf{c}(t) dt$$
该公式确立了体渲染模型的物理严密性。
</details>

---

## 9.3.3 核心数学推导二：3DGS 各向异性高斯球与屏幕投影

3DGS 将空间表征为成百上千个三维高斯椭球。

<div align="center">

<img src="/figures/09-spatial-worlds/source/03-nerf-3dgs/3dgs-fig2.png" alt="3DGS 从稀疏 SfM 点初始化三维高斯，交替优化与密度控制，再通过可微光栅化生成图像。" width="86%">

_图 9.3-5：3DGS 从稀疏 SfM 点初始化三维高斯，交替优化与密度控制，再通过可微光栅化生成图像。 出处：[3D Gaussian Splatting for Real-Time Radiance Field Rendering，Bernhard Kerbl et al.，2023](https://arxiv.org/abs/2308.04079)。_

</div>

### 1. 三维高斯概率分布与各向异性协方差矩阵
一个三维高斯椭球以空间均值中心 $\boldsymbol{\mu} \in \mathbb{R}^3$ 与三维协方差矩阵 $\boldsymbol{\Sigma} \in \mathbb{R}^{3 \times 3}$ 定义：

$$G(\mathbf{x}) = \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)$$

为了保证协方差矩阵 $\boldsymbol{\Sigma}$ 在训练优化中严格保持**半正定性（Positive Semi-Definite）**，3DGS 将其显式分解为**三维缩放对角阵 $\mathbf{S} = \text{diag}(s_x, s_y, s_z)$** 与由单位四元数表示的**三维旋转矩阵 $\mathbf{R}$**：

$$\boldsymbol{\Sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^\top \mathbf{R}^\top$$

### 2. 投影到 2D 屏幕上的 2D 协方差矩阵
根据 Zwicker 等人（2001）的 EWA 表面泼溅理论，三维高斯球经透视投影变换 $\mathbf{W}$（外参）与局部射影线性化雅可比矩阵 $\mathbf{J}$ 投影到相机二维像素平面后，其二维协方差矩阵 $\boldsymbol{\Sigma}' \in \mathbb{R}^{2 \times 2}$ 满足严格解析式：

$$\boldsymbol{\Sigma}' = \mathbf{J} \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^\top \mathbf{J}^\top$$

在二维像素平面上，所有落入视锥的高斯椭球被切分为 $16 \times 16$ 像素的小瓦片（Tiles），并按照深度键值进行极速硬件并行基数排序，瞬间完成全图 Alpha 混合渲染！

<details>
<summary><b>深入推导：局部射影投影雅可比矩阵 $\mathbf{J}$ 与 EWA 椭球重采样证明（点击展开查看完整推导）</b></summary>

设相机坐标系下高斯中心为 $\mathbf{t} = (t_x, t_y, t_z)^\top$。
透视投影映射为 $(u, v) = (f_x \frac{t_x}{t_z} + c_x, f_y \frac{t_y}{t_z} + c_y)$。
其对相机坐标 $\mathbf{t}$ 的一阶偏导数雅可比矩阵为：
$$\mathbf{J} = \begin{bmatrix} \frac{\partial u}{\partial t_x} & \frac{\partial u}{\partial t_y} & \frac{\partial u}{\partial t_z} \\ \frac{\partial v}{\partial t_x} & \frac{\partial v}{\partial t_y} & \frac{\partial v}{\partial t_z} \end{bmatrix} = \begin{bmatrix} \frac{f_x}{t_z} & 0 & -\frac{f_x t_x}{t_z^2} \\ 0 & \frac{f_y}{t_z} & -\frac{f_y t_y}{t_z^2} \end{bmatrix}$$
根据多维随机变量线性变换协方差性质 $\text{Cov}(\mathbf{A} \mathbf{X}) = \mathbf{A} \text{Cov}(\mathbf{X}) \mathbf{A}^\top$，二维投影协方差严格等于 $\boldsymbol{\Sigma}' = \mathbf{J} (\mathbf{R}_{\text{cam}} \boldsymbol{\Sigma} \mathbf{R}_{\text{cam}}^\top) \mathbf{J}^\top$。为消除混叠走样，通常在对角线上补偿微小低通滤波核 $\boldsymbol{\Sigma}' \leftarrow \boldsymbol{\Sigma}' + 0.3 \mathbf{I}_{2 \times 2}$。
</details>

---

## 9.3.4 纯底层 PyTorch 代码实现：体渲染数值积分器与 3D 高斯投影引擎

下面我们使用纯底层 PyTorch 算子实现 NeRF 离散体渲染数值求积引擎与 3D 高斯椭球的 2D 投影协方差求解器。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VolumeRenderingIntegrator:
    """
    NeRF 离散体渲染积分求积引擎
    实现射线上离散采样的累积透射率与颜色合成
    """
    @staticmethod
    def render_rays(sigmas: torch.Tensor, colors: torch.Tensor, deltas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param sigmas: (B, N_samples) 采样点体积密度
        :param colors: (B, N_samples, 3) 采样点 RGB 颜色
        :param deltas: (B, N_samples) 相邻采样点步长
        :return: (rendered_colors, accumulated_weights)
        """
        # 1. 计算单点不透明度 alpha_i = 1 - exp(-sigma_i * delta_i)
        densities = sigmas * deltas
        alphas = 1.0 - torch.exp(-densities) # (B, N_samples)

        # 2. 计算累积透射率 T_i = prod(1 - alpha_j)
        # 为防止数值下溢，在 log 空间中累加: T_i = exp(-cumsum(sigma * delta))
        cum_densities = torch.cumsum(densities, dim=-1)
        # 在序列前端补 0: T_1 = exp(0) = 1.0
        cum_densities_shifted = F.pad(cum_densities[..., :-1], (1, 0), value=0.0)
        transmittance = torch.exp(-cum_densities_shifted) # (B, N_samples)

        # 3. 颜色权重 w_i = T_i * alpha_i
        weights = transmittance * alphas # (B, N_samples)

        # 4. 沿射线求和合成色彩: (B, N_samples, 1) * (B, N_samples, 3) -> (B, 3)
        rendered_colors = (weights.unsqueeze(-1) * colors).sum(dim=1)
        return rendered_colors, weights

class Gaussian3DProjector:
    """
    3D 高斯泼溅 (3DGS) 屏幕空间 2D 协方差投影引擎
    """
    @staticmethod
    def project_covariances(
        scales: torch.Tensor,
        rotations: torch.Tensor,
        camera_pts: torch.Tensor,
        f_x: float,
        f_y: float
    ) -> torch.Tensor:
        """
        :param scales: (N, 3) 三维缩放因子 (s_x, s_y, s_z)
        :param rotations: (N, 3, 3) 三维旋转矩阵 R
        :param camera_pts: (N, 3) 相机坐标系下的中心点 (t_x, t_y, t_z)
        :return: (N, 2, 2) 投影到屏幕的 2D 协方差矩阵 Sigma'
        """
        N = scales.size(0)

        # 1. 重构 3D 协方差矩阵 Sigma = R * S * S^T * R^T
        s_mat = torch.diag_embed(scales) # (N, 3, 3)
        m_mat = torch.bmm(rotations, s_mat) # (N, 3, 3)
        sigma_3d = torch.bmm(m_mat, m_mat.transpose(1, 2)) # (N, 3, 3)

        # 2. 计算射影雅可比矩阵 J
        tx = camera_pts[:, 0]
        ty = camera_pts[:, 1]
        tz = camera_pts[:, 2].clamp_min(1e-4)

        j_mat = torch.zeros(N, 2, 3, device=scales.device)
        j_mat[:, 0, 0] = f_x / tz
        j_mat[:, 0, 2] = -(f_x * tx) / (tz ** 2)
        j_mat[:, 1, 1] = f_y / tz
        j_mat[:, 1, 2] = -(f_y * ty) / (tz ** 2)

        # 3. 屏幕投影 Sigma' = J * Sigma_3D * J^T
        sigma_2d = torch.bmm(torch.bmm(j_mat, sigma_3d), j_mat.transpose(1, 2)) # (N, 2, 2)

        # 4. 补偿抗锯齿低通滤波
        sigma_2d[:, 0, 0] += 0.3
        sigma_2d[:, 1, 1] += 0.3
        return sigma_2d

# ===================================================================
# 单元测试与手算算例精确校验
# ===================================================================
if __name__ == "__main__":
    # 1. 测试 NeRF 体渲染积分器 (验证正文中的双点紫色手算算例)
    batch_rays = 1
    num_pts = 2

    test_sigmas = torch.tensor([[0.693147, 10.0]])
    test_colors = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    test_deltas = torch.tensor([[1.0, 1.0]])

    rendered_rgb, pt_weights = VolumeRenderingIntegrator.render_rays(
        test_sigmas, test_colors, test_deltas
    )

    r_val, g_val, b_val = rendered_rgb[0].tolist()
    print(f"[NeRF Test] 渲染色彩输出: ({r_val:.3f}, {g_val:.3f}, {b_val:.3f}), 期望: (0.500, 0.000, 0.500)")
    print(f"[NeRF Test] 采样点权重分量: {pt_weights[0].tolist()}")

    assert abs(r_val - 0.5) < 1e-3 and abs(b_val - 0.5) < 1e-3, "体渲染手算代入校验失败！"

    # 2. 测试 3D 高斯屏幕投影引擎
    num_gaussians = 3
    dummy_scales = torch.tensor([[0.1, 0.1, 0.1]] * num_gaussians)
    dummy_rot = torch.eye(3).unsqueeze(0).expand(num_gaussians, -1, -1)
    dummy_cam_pts = torch.tensor([[0.0, 0.0, 2.0], [1.0, -0.5, 4.0], [-2.0, 1.0, 5.0]])

    sigma_2d_out = Gaussian3DProjector.project_covariances(
        dummy_scales, dummy_rot, dummy_cam_pts, f_x=1000.0, f_y=1000.0
    )

    print(f"[3DGS Test] 2D 屏幕投影协方差形状: {sigma_2d_out.shape}")
    assert sigma_2d_out.shape == (num_gaussians, 2, 2), "2D 协方差矩阵维度不符！"
    assert (sigma_2d_out[:, 0, 0] > 0).all() and (sigma_2d_out[:, 1, 1] > 0).all(), "协方差对角线方差非正！"
    print("✓ NeRF 体积渲染数值求积与 3DGS 屏幕投影引擎单测全部通过！")
```

---

## 9.3.5 本节小结

回顾本节内容，我们建立了三维神经辐射场与高斯渲染的完整物理与数学图谱：
1. **体积辐射传输方程**：光线沿射线的能量积分与累积透射率 $T(t)$ 揭示了透明介质与半透明遮挡的本质；
2. **NeRF 的连续隐式革命**：以神经网络拟合连续场，通过分层离散数值求积实现了照片级新视角合成；
3. **3DGS 的显式实时突破**：利用各向异性三维高斯椭球分解 $\boldsymbol{\Sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^\top \mathbf{R}^\top$ 与屏幕射影雅可比投影，将高保真渲染推向百帧实时新时代。
