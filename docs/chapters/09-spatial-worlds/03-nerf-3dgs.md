# 9.3 神经辐射场 (NeRF) 与三维高斯泼溅 (3DGS)

在前面的章节中，我们学习了如何利用刚体外参和相机内参将三维离散点投影到二维像素，以及如何通过 BEV 将多相机画面提升为平面的二维栅格。

然而，物理世界是一个充满复杂光影变化、半透明烟雾、高光反光以及精细毛发边缘的连续立体介质。如果我们仅仅用粗糙的网格（Meshes）或离散的体素（Voxels）去拼凑三维世界，将不可避免地面临几何分辨率受限、显存爆炸以及无法模拟真实光学材质的困境。

2020 年，Mildenhall 等人提出了 **神经辐射场（Neural Radiance Fields, NeRF）**，开创了用连续神经网络隐式拟合空间几何与光辐射的新时代；2023 年，Kerbl 等人推出了 **三维高斯泼溅（3D Gaussian Splatting, 3DGS）**，以极具代数美感的各向异性高斯椭球显式表征，将真实感渲染直接推向了百帧以上的实时渲染速度。

本节我们将从经典光学传输方程出发，严密推导体积渲染积分与 3D 高斯屏幕投影算法，并使用纯底层 PyTorch 从零手写可微体渲染求积与协方差投影引擎。

<div align="center">

<img src="/figures/09-spatial-worlds/source/03-nerf-3dgs/nerf-fig1.png" alt="NeRF 沿相机光线采样三维点与视角方向，预测密度与颜色，再通过体渲染合成二维图像。" width="86%">

_图 9.3-1：NeRF 沿相机光线采样三维点与视角方向，预测密度与颜色，再通过体渲染合成二维图像。 出处：[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis，Ben Mildenhall et al.，2020](https://arxiv.org/abs/2003.08934)。_

</div>

---

## 9.3.1 物理与光学基石：体积辐射传输方程与比尔-朗伯定律

要理解 NeRF 的数学本质，我们首先必须回到经典光学中描述光穿过发光吸收介质的**体积辐射传输方程（Volume Rendering Equation）**。

### 1. 经典比尔-朗伯定律（Beer-Lambert Law）
想象一束激光穿过弥漫着烟雾或悬浮颗粒的透明玻璃水箱：
- 光线每向前推进微小的距离 $dt$，被颗粒吸收或散射损失的光强与当前光强 $I(t)$ 以及介质在该处的体积密度（消光系数）$\sigma(t)$ 成正比：
  $$\frac{dI(t)}{dt} = -\sigma(t) I(t)$$
- 解该一阶常微分方程，得到光线从起点 $0$ 到达距离 $t$ 时的**累积透射率（Accumulated Transmittance）**：
  $$T(t) = \exp\left( -\int_0^t \sigma(s) ds \right)$$

> **物理直觉**：
> - $T(t) \in [0, 1]$ 描述了“光线在没有撞到任何粒子的情况下，畅通无阻地穿透到深度 $t$ 的生存概率”；
> - 如果前方全部是空旷的空气（$\sigma(s) = 0$），则 $T(t) = e^0 = 1.0$，光线 $100\%$ 穿透；
> - 如果前方有一面坚硬的水泥墙（$\sigma(s) \to \infty$），则 $T(t) = e^{-\infty} \to 0$，光线被完全遮挡截断。

### 2. 连续体积辐射传输积分公式
设一条相机视线（Ray）的参数方程为 $\mathbf{r}(t) = \mathbf{o} + t \mathbf{d}$（其中 $\mathbf{o}$ 为相机光心，$\mathbf{d}$ 为视线单位方向向量）。
空间中任意点 $\mathbf{r}(t)$ 都在自发光，其发射的 RGB 颜色为 $\mathbf{c}(\mathbf{r}(t), \mathbf{d})$，体积密度为 $\sigma(\mathbf{r}(t))$。

相机像素最终接收到的连续合成色彩 $C(\mathbf{r})$，为整条光线上所有微元发射光线在衰减后的累积积分：

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \cdot \sigma(\mathbf{r}(t)) \cdot \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt$$

其中 $[t_n, t_f]$ 为相机的近裁切面与远裁切面。

<div align="center">

<img src="/figures/09-spatial-worlds/latex/03-nerf-3dgs/front-to-back-transmittance.png" alt="沿射线从近到远，每个样本的颜色权重由此前透射率连乘与本点不透明度共同决定" width="86%">

_图 9.3-2：沿射线从近到远，每个样本的颜色权重由此前透射率连乘与本点不透明度共同决定。_

</div>

---

## 9.3.2 核心数学推导一：NeRF 的离散数值正交求积

在计算机中，我们无法直接计算连续无穷微积分。NeRF 采用了**离散数值求积（Numerical Quadrature）**算法。

<div align="center">

<img src="/figures/09-spatial-worlds/source/03-nerf-3dgs/nerf-fig2b.png" alt="NeRF 将位置与视角通过正弦高频编码映射到高维，再由 MLP 预测密度与视角相关的辐射色彩。" width="86%">

_图 9.3-3：NeRF 将位置与视角通过正弦高频编码映射到高维，再由 MLP 预测密度与视角相关的辐射色彩。 出处：[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis，Ben Mildenhall et al.，2020](https://arxiv.org/abs/2003.08934)。_

</div>

<div align="center">

<img src="/figures/09-spatial-worlds/source/03-nerf-3dgs/instantngp-fig1.png" alt="Instant-NGP 使用多分辨率哈希网格替代庞大 MLP，实现秒级高保真辐射场训练。" width="86%">

_图 9.3-4：Instant-NGP 使用多分辨率哈希网格替代庞大 MLP，实现秒级高保真辐射场训练。 出处：[Instant Neural Graphics Primitives with a Multiresolution Hash Encoding，Thomas Müller et al.，2022](https://arxiv.org/abs/2201.05989)。_

</div>

### 1. 离散 Alpha 混合公式
我们将近远平面区间 $[t_n, t_f]$ 等距采样 $N$ 个离散点 $t_1 < t_2 < \dots < t_N$，第 $i$ 个采样区间的微元步长为 $\delta_i = t_{i+1} - t_i$。

对于第 $i$ 个局部采样点：
1. **单点不透明度（Alpha Value）**：
   $$\alpha_i = 1 - \exp(-\sigma_i \delta_i) \in [0, 1]$$
2. **累积透射率（Transmittance）**：前 $i-1$ 个点全部未被完全遮挡的概率：
   $$T_i = \prod_{j=1}^{i-1} (1 - \alpha_j) = \exp\left( -\sum_{j=1}^{i-1} \sigma_j \delta_j \right)$$
3. **复合像素合成色彩**：
   $$\hat{C}(\mathbf{r}) = \sum_{i=1}^N T_i \cdot \alpha_i \cdot \mathbf{c}_i = \sum_{i=1}^N w_i \cdot \mathbf{c}_i, \quad \text{其中 } w_i = T_i \alpha_i$$

**手算代入算例**：
设一条光线上仅有两个采样点，步长均为 $\delta_1 = \delta_2 = 1.0$：
- 第 1 个点：密度 $\sigma_1 = \ln(2) \approx 0.6931$，颜色为纯红色 $\mathbf{c}_1 = [1.0, 0.0, 0.0]^\top$；
- 第 2 个点：密度 $\sigma_2 = 10.0$（坚硬表面），颜色为纯蓝色 $\mathbf{c}_2 = [0.0, 0.0, 1.0]^\top$。

我们来手动计算最终像素颜色：
1. 计算第 1 点：
   $$\alpha_1 = 1 - e^{-0.6931 \times 1.0} = 1 - 0.5 = 0.5$$
   $$T_1 = 1.0 \implies w_1 = T_1 \alpha_1 = 1.0 \times 0.5 = 0.5$$
2. 计算第 2 点：
   $$\alpha_2 = 1 - e^{-10.0 \times 1.0} \approx 1 - 0.0 = 1.0$$
   $$T_2 = 1 - \alpha_1 = 1 - 0.5 = 0.5 \implies w_2 = T_2 \alpha_2 = 0.5 \times 1.0 = 0.5$$
3. 像素合成颜色：
   $$\hat{C} = w_1 \mathbf{c}_1 + w_2 \mathbf{c}_2 = 0.5 \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix} + 0.5 \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0.0 \\ 0.5 \end{bmatrix} \quad (\text{呈现半透明红色与蓝色混合后的紫色！})$$

初等代数的直观代入揭示了体渲染的精髓：前面的半透明红色粒子吸收了一半光强，使后方的蓝色物体只能透出 $50\%$ 的能量！

<details>
<summary><b>深入推导：连续体渲染积分到离散 Alpha 混合阶梯近似的严格收敛性证明（点击展开查看完整推导）</b></summary>

在区间 $[t_i, t_{i+1}]$ 内假设密度 $\sigma(t)$ 与色彩 $\mathbf{c}(t)$ 为常数 $\sigma_i, \mathbf{c}_i$。
单区间积分展开为：
$$\int_{t_i}^{t_{i+1}} T(t) \sigma_i \mathbf{c}_i dt = \mathbf{c}_i \sigma_i \int_{t_i}^{t_{i+1}} \exp\left(-\int_0^t \sigma(s)ds\right) dt = \mathbf{c}_i \sigma_i T_i \int_0^{\delta_i} e^{-\sigma_i s} ds = \mathbf{c}_i T_i (1 - e^{-\sigma_i \delta_i})$$
令 $\alpha_i = 1 - e^{-\sigma_i \delta_i}$，严格证得单区间积分等价于 $T_i \alpha_i \mathbf{c}_i$。当网格划分极大模长 $\max \delta_i \to 0$ 时，黎曼和严格依概率收敛于连续能量积分。
</details>

---

## 9.3.3 核心数学推导二：3D 高斯泼溅 (3DGS) 的协方差投影与 EWA 滤波

NeRF 虽能生成照片级图像，但渲染单个像素需要调用数百次神经网络前向推理，渲染一张 $1080\text{p}$ 图像需要耗费数秒。
2023 年诞生的 **3D 高斯泼溅（3DGS）** 彻底颠覆了隐式体渲染，改用数百万个显式的**三维高斯椭球（3D Gaussians）**表征场景。

<div align="center">

<img src="/figures/09-spatial-worlds/source/03-nerf-3dgs/3dgs-fig2.png" alt="3DGS 将空间表征为离散高斯椭球，投影到二维屏幕后执行基于图块的极速光栅化。" width="86%">

_图 9.3-5：3DGS 将空间表征为离散高斯椭球，投影到二维屏幕后执行基于图块的极速光栅化。 出处：[3D Gaussian Splatting for Real-Time Radiance Field Rendering，Bernhard Kerbl et al.，2023](https://arxiv.org/abs/2308.04079)。_

</div>

### 1. 三维各向异性高斯函数
每一个三维高斯球由其中心位置 $\boldsymbol{\mu} \in \mathbb{R}^3$ 与三维协方差矩阵 $\boldsymbol{\Sigma} \in \mathbb{R}^{3 \times 3}$ 唯一定义：

$$G(\mathbf{x}) = \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)$$

为了保证协方差矩阵 $\boldsymbol{\Sigma}$ 在训练优化中严格保持**半正定性（Positive Semi-Definite）**，3DGS 将其显式分解为**三维缩放对角阵 $\mathbf{S} = \text{diag}(s_x, s_y, s_z)$** 与由单位四元数表示的**三维旋转矩阵 $\mathbf{R}$**：

$$\boldsymbol{\Sigma} = \mathbf{R} \mathbf{S} \mathbf{S}^\top \mathbf{R}^\top$$

### 2. 投影到 2D 屏幕上的 2D 协方差矩阵与手算算例
根据 Zwicker 等人（2001）的 EWA 表面泼溅理论，三维高斯球经透视投影变换 $\mathbf{W}$（外参）与局部射影线性化雅可比矩阵 $\mathbf{J}$ 投影到相机二维像素平面后，其二维协方差矩阵 $\boldsymbol{\Sigma}' \in \mathbb{R}^{2 \times 2}$ 满足严格解析式：

$$\boldsymbol{\Sigma}' = \mathbf{J} \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^\top \mathbf{J}^\top$$

其中局部射影雅可比矩阵为：
$$\mathbf{J} = \begin{bmatrix} \frac{f_x}{t_z} & 0 & -\frac{f_x t_x}{t_z^2} \\ 0 & \frac{f_y}{t_z} & -\frac{f_y t_y}{t_z^2} \end{bmatrix}$$

**手算代入算例**：
设某 3D 高斯球位于相机正前方 $\mathbf{t} = [0, 0, 2.0]^\top$（深度 $t_z = 2.0\text{ m}, t_x = 0, t_y = 0$），旋转为单位阵 $\mathbf{R} = \mathbf{I}$，三维轴半径尺度为 $\mathbf{S} = \text{diag}(0.1, 0.1, 0.1)\text{ m}$，相机焦距 $f_x = f_y = 1000$。

1. **计算三维协方差**：
   $$\boldsymbol{\Sigma} = \mathbf{S} \mathbf{S}^\top = \text{diag}(0.01, 0.01, 0.01)$$
2. **计算雅可比矩阵**：
   $$\mathbf{J} = \begin{bmatrix} \frac{1000}{2} & 0 & 0 \\ 0 & \frac{1000}{2} & 0 \end{bmatrix} = \begin{bmatrix} 500 & 0 & 0 \\ 0 & 500 & 0 \end{bmatrix}$$
3. **计算屏幕 2D 协方差**：
   $$\boldsymbol{\Sigma}' = \mathbf{J} \boldsymbol{\Sigma} \mathbf{J}^\top = \begin{bmatrix} 500^2 \times 0.01 & 0 \\ 0 & 500^2 \times 0.01 \end{bmatrix} = \begin{bmatrix} 2500 & 0 \\ 0 & 2500 \end{bmatrix}$$
4. 屏幕标准差 $\sigma_{\text{screen}} = \sqrt{2500} = 50\text{ 像素}$！
   在相机画面上，这个物理直径 $20\text{ 厘米}$ 的小球在 2 米距离处清晰地投影为一个半径 $50$ 像素的圆形光斑！

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
