# 9.1 相机几何与三维空间投影

在构建空间世界模型（Spatial World Models）与具身智能感知的宏大版图中，我们首先必须攻克一个最基础的立体几何与光学难题：**现实世界中充满深度与体积的三维物体，是如何投射为相机传感器上的一张张二维平面像素画面的？**

无论是自动驾驶汽车依靠环视相机重构周围街道的三维鸟瞰图（BEV），还是机械臂依靠双目立体视觉判断桌面上水杯的抓取点，亦或是神经辐射场（NeRF）与 3D 高斯泼溅（3DGS）从多视角照片重构连续空间，它们的数学底层全部建立在严密的**相机几何与透视投影（Projective Geometry）**理论之上。

本节我们将从墨子小孔成像的经典光学实验出发，层层推演针孔相机模型、齐次坐标、内参矩阵 $\mathbf{K}$ 与外参刚体变换矩阵 $[\mathbf{R} \mid \mathbf{t}]$，并使用纯底层 PyTorch 算子手写可批量加速的三维投影引擎。

## 本章总览

<div align="center">

<img src="/figures/09-spatial-worlds/latex/01-camera-geometry/chapter-overview.png" alt="第 9 章学习路线：从相机几何到四维自动驾驶世界模型" width="100%">

_第 9 章学习路线：从二维成像几何出发，通向三维占用、连续场、四维预测与驾驶规划。_

</div>

<div align="center">

<img src="/figures/09-spatial-worlds/source/01-camera-geometry/dust3r-fig1.png" alt="DUSt3R 从未标定图像直接恢复稠密三维点图与相机关系，展示相机几何最终要支持的真实重建任务。" width="86%">

_图 9.1-1：DUSt3R 从未标定图像直接恢复稠密三维点图与相机关系，展示相机几何最终要支持的真实重建任务。 出处：[DUSt3R: Geometric 3D Vision Made Easy，Shuzhe Wang et al.，2024](https://arxiv.org/abs/2312.14132)。_

</div>

---

## 9.1.1 物理与几何基石：小孔成像与射影几何的千年演进

要理解现代三维视觉的投影方程，我们首先需要回顾人类对光线传播与成像几何的经典认知历程。

### 1. 经典光学起点：小孔成像与相似三角形
早在公元前四世纪，中国古代思想家墨子在《墨经》中就记录了人类历史上最早的**小孔成像**实验：“景倒，在午有端，与景长。说在端。”
- 当光线穿过暗室墙壁上的一个微小针孔时，来自物体顶部的光线沿直线穿过小孔射在暗室底部的墙壁上，来自底部的光线射在顶部，形成一个上下颠倒、左右相反的实像；
- 在初等平面几何中，这一光学现象完美对应着**相似三角形定理**：物体在成像平面上的尺寸，与物体的真实物理尺寸之比，严格等于像距与物距之比。

### 2. 射影几何与透视除法的数学本质
在文艺复兴时期，达芬奇、布鲁内莱斯基等艺术家为了在画布上真实呈现三维透视感，创立了早期透视画法。
19 世纪，数学家彭赛列（Poncelet）等人将透视画法形式化为严密的**射影几何学（Projective Geometry）**。

在三维欧几里得空间中，两条平行线永远不会相交；但在现实世界的铁轨照片中，两条平行的铁轨在视线远方会汇聚于一个点（灭点，Vanishing Point）。
射影几何通过引入**齐次坐标（Homogeneous Coordinates）**，将非线性的透视投影转化为了优雅的线性矩阵乘法，构成了现代计算机视觉不可动摇的数理基石。

<div align="center">

<img src="/figures/09-spatial-worlds/source/01-camera-geometry/mipnerf-fig1.png" alt="Mip-NeRF 对比单条相机射线与具有像素面积的锥台，直观呈现像素在三维空间中对应的采样区域。" width="86%">

_图 9.1-2：Mip-NeRF 对比单条相机射线与具有像素面积的锥台，直观呈现像素在三维空间中对应的采样区域。 出处：[Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields，Jonathan T. Barron et al.，2021](https://arxiv.org/abs/2103.13415)。_

</div>

---

## 9.1.2 核心数学推导一：针孔相机模型与内参矩阵 $\mathbf{K}$

设相机光心（即针孔位置）位于三维空间原点 $O_c$，以向前观察的方向为 $Z_c$ 轴（光轴），水平向右为 $X_c$ 轴，竖直向下为 $Y_c$ 轴，构成标准**相机坐标系**。

假定在距离光心 $f$（焦距）处放置一块垂直于光轴的成像传感器平面。三维空间中任意一个物理点 $P_c = (X_c, Y_c, Z_c)^\top$ 发出的光线经过光心，投射到成像平面上的物理坐标为 $(x, y)$。

根据相似三角形定理：

$$\frac{x}{f} = \frac{X_c}{Z_c} \implies x = f \frac{X_c}{Z_c}$$

$$\frac{y}{f} = \frac{Y_c}{Z_c} \implies y = f \frac{Y_c}{Z_c}$$

> **初等几何直觉**：
> 物体距离相机越远（$Z_c$ 越大），在画面中呈现的尺寸就按 $\frac{1}{Z_c}$ 的比例反比缩小；这就是我们日常生活中“近大远小”的严格数学表达。

<div align="center">

<img src="/figures/09-spatial-worlds/source/01-camera-geometry/deepv2d-fig2.png" alt="DeepV2D 的深度模块把相机模型、重投影代价体与深度更新连接起来，展示投影矩阵如何进入可学习三维视觉系统。" width="86%">

_图 9.1-3：DeepV2D 的深度模块把相机模型、重投影代价体与深度更新连接起来，展示投影矩阵如何进入可学习三维视觉系统。 出处：[DeepV2D: Video to Depth with Differentiable Structure from Motion，Zachary Teed et al.，2020](https://arxiv.org/abs/1812.04605)。_

</div>

### 1. 从连续毫米物理坐标到离散数字像素坐标
成像传感器上的物理尺寸 $(x, y)$ 单位是毫米，而计算机读取的数字图像以“像素（Pixel）”为单位，且像素坐标系原点 $(u, v) = (0, 0)$ 位于图像的左上角。
设 $x$ 方向每毫米包含 $m_x$ 个像素，$y$ 方向每毫米包含 $m_y$ 个像素，光轴与图像平面的交点（主点 Principal Point）在像素坐标系下的坐标为 $(c_x, c_y)$：

$$u = m_x x + c_x = (m_x f) \frac{X_c}{Z_c} + c_x = f_x \frac{X_c}{Z_c} + c_x$$

$$v = m_y y + c_y = (m_y f) \frac{Y_c}{Z_c} + c_y = f_y \frac{Y_c}{Z_c} + c_y$$

其中 $f_x = m_x f, f_y = m_y f$ 分别为以像素为单位的等效水平焦距与垂直焦距。

### 2. 齐次坐标与内参矩阵 $\mathbf{K}$
引入齐次坐标，我们可以将上述带除法的非线性映射，改写为矩阵乘法：

$$Z_c \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = \mathbf{K} P_c$$

其中矩阵 $\mathbf{K} \in \mathbb{R}^{3 \times 3}$ 被称为**相机内参矩阵（Camera Intrinsic Matrix）**。

<details>
<summary><b>深入推导：径向与切向镜头光学畸变（Brown-Conrady 模型）非线性泰勒展开推导（点击展开查看完整推导）</b></summary>

实际光学镜头存在厚度与曲率，使光线偏离理想针孔。设归一化平面坐标为 $x' = X_c/Z_c, y' = Y_c/Z_c$，径向半径平方 $r^2 = x'^2 + y'^2$。
真实发生畸变后的归一化坐标 $(\tilde{x}, \tilde{y})$ 服从 Brown-Conrady 展开：
$$\begin{aligned}
\tilde{x} &= x' \underbrace{(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)}_{\text{径向桶形/枕形畸变}} + \underbrace{[2 p_1 x' y' + p_2 (r^2 + 2 x'^2)]}_{\text{透镜安装非平行切向畸变}} \\
\tilde{y} &= y' (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + [p_1 (r^2 + 2 y'^2) + 2 p_2 x' y']
\end{aligned}$$
最终像素坐标由畸变点乘以焦距并平移主点得到：$u = f_x \tilde{x} + c_x, v = f_y \tilde{y} + c_y$。现代视觉管线通常预先标定 $(k_1, k_2, p_1, p_2)$ 并执行去畸变重映射（Undistortion）。
</details>

---

## 9.1.3 核心数学推导二：外参刚体变换与全局投影模型

在自动驾驶或机器人工作站中，物体通常定义在统一的**世界坐标系（World Frame）**中，而相机会随车身或机械臂运动。

<div align="center">

<img src="/figures/09-spatial-worlds/source/01-camera-geometry/deepv2d-fig3.png" alt="DeepV2D 的运动模块从多帧残差流联合优化相机位姿，展示外参在可微结构恢复中的更新路径。" width="86%">

_图 9.1-4：DeepV2D 的运动模块从多帧残差流联合优化相机位姿，展示外参在可微结构恢复中的更新路径。 出处：[DeepV2D: Video to Depth with Differentiable Structure from Motion，Zachary Teed et al.，2020](https://arxiv.org/abs/1812.04605)。_

</div>

### 1. 世界系向相机系的欧氏刚体变换
设世界坐标系下一点为 $P_w = (X_w, Y_w, Z_w)^\top$。从世界坐标系到相机坐标系的刚体变换由旋转矩阵 $\mathbf{R} \in SO(3)$ 与平移向量 $\mathbf{t} \in \mathbb{R}^3$ 完全确定：

$$P_c = \mathbf{R} P_w + \mathbf{t}$$

利用 $4 \times 4$ 齐次变换矩阵，世界向相机坐标系的变换写为：

$$\begin{bmatrix} P_c \\ 1 \end{bmatrix} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix} \begin{bmatrix} P_w \\ 1 \end{bmatrix} = \mathbf{T}_{cw} \begin{bmatrix} P_w \\ 1 \end{bmatrix}$$

其中 $\mathbf{T}_{cw} = [\mathbf{R} \mid \mathbf{t}]$ 被称为相机的**外参矩阵（Extrinsic Matrix）**。

### 2. 完整的相机投影方程与透视除法
将外参与内参级联，得到将三维世界点直接投影至二维图像齐次坐标的完整方程：

$$Z_c \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{R} \mid \mathbf{t}] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix} = \mathbf{P} \begin{bmatrix} P_w \\ 1 \end{bmatrix}$$

其中 $\mathbf{P} = \mathbf{K} [\mathbf{R} \mid \mathbf{t}] \in \mathbb{R}^{3 \times 4}$ 称为相机的**投影矩阵（Projection Matrix）**。

<div align="center">

<img src="/figures/09-spatial-worlds/latex/01-camera-geometry/perspective-division-scale.png" alt="世界齐次点经外参与内参得到带深度尺度的三向量，再由第三分量完成透视除法" width="86%">

_图 9.1-5：投影矩阵输出带深度尺度的三维齐次向量；除以第三分量 Z_c 完成透视除法，恢复二维像素坐标。_

</div>

**手算代入算例**：
设某相机内参为 $f_x = 1000, f_y = 1000, c_x = 400, c_y = 300$。相机外参无旋转（$\mathbf{R} = \mathbf{I}$），沿 $Z$ 轴向后平移 $\mathbf{t} = [0, 0, -2]^\top$。
世界坐标系中有一个目标点 $P_w = [1.0, 2.0, 10.0]^\top$。

1. 计算相机坐标系下的点坐标：
   $$P_c = \mathbf{R} P_w + \mathbf{t} = [1.0, 2.0, 10.0]^\top + [0, 0, -2]^\top = [1.0, 2.0, 8.0]^\top$$
   （物体位于相机前方，物理深度为 $Z_c = 8.0\text{ 米}$）；
2. 乘以相机内参矩阵：
   $$\begin{bmatrix} \tilde{u} \\ \tilde{v} \\ \tilde{w} \end{bmatrix} = \begin{bmatrix} 1000 & 0 & 400 \\ 0 & 1000 & 300 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1.0 \\ 2.0 \\ 8.0 \end{bmatrix} = \begin{bmatrix} 1000 \times 1.0 + 400 \times 8.0 \\ 1000 \times 2.0 + 300 \times 8.0 \\ 8.0 \end{bmatrix} = \begin{bmatrix} 1000 + 3200 \\ 2000 + 2400 \\ 8.0 \end{bmatrix} = \begin{bmatrix} 4200 \\ 4400 \\ 8.0 \end{bmatrix}$$
3. 执行透视除法（除以第三分量 $Z_c = 8.0$）：
   $$u = \frac{4200}{8.0} = 525\text{ 像素}, \quad v = \frac{4400}{8.0} = 550\text{ 像素}$$

整个推导过程一清二楚：三维点在相机内投影在第 $(525, 550)$ 像素位置！

<details>
<summary><b>深入推导：对极几何（Epipolar Geometry）基础矩阵 $\mathbf{F}$ 与本质矩阵 $\mathbf{E}$ 代数推导（点击展开查看完整推导）</b></summary>

设双目相机左右光心分别为 $O_1, O_2$，相对旋转平移为 $(\mathbf{R}, \mathbf{t})$。空间点在两相机的归一化坐标分别为 $\mathbf{x}_1, \mathbf{x}_2$。
向量 $\mathbf{x}_1, \mathbf{t}, \mathbf{R}^\top \mathbf{x}_2$ 必定共面于对极平面上，其标量三重积为零：
$$\mathbf{x}_2^\top (\mathbf{t} \times \mathbf{R} \mathbf{x}_1) = 0 \implies \mathbf{x}_2^\top [\mathbf{t}]_\times \mathbf{R} \mathbf{x}_1 = 0$$
定义**本质矩阵（Essential Matrix）** $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$，满足 $\mathbf{x}_2^\top \mathbf{E} \mathbf{x}_1 = 0$。
将像素坐标 $\mathbf{p} = \mathbf{K} \mathbf{x} \implies \mathbf{x} = \mathbf{K}^{-1} \mathbf{p}$ 代入：
$$\mathbf{p}_2^\top \underbrace{\mathbf{K}_2^{-\top} \mathbf{E} \mathbf{K}_1^{-1}}_{\mathbf{F}} \mathbf{p}_1 = 0$$
矩阵 $\mathbf{F} \in \mathbb{R}^{3 \times 3}$ 称为**基础矩阵（Fundamental Matrix）**，其秩严格为 2，将左图的一个像素点严格约束在右图的一条**对极线（Epipolar Line）**上，是立体视觉深度重建的代数核心。
</details>

---

## 9.1.4 纯底层 PyTorch 代码实现：批量三维空间投影与视锥裁剪引擎

下面我们使用纯底层 PyTorch 算子实现支持批量并行的三维点投影与视锥有效性裁剪引擎。

```python
import torch

def batch_project_points_3d(
    points_world: torch.Tensor,
    k_mat: torch.Tensor,
    r_mat: torch.Tensor,
    t_vec: torch.Tensor,
    img_size: tuple[int, int] = (800, 600)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    批量将三维世界点投影到二维图像平面，并进行视锥裁剪
    :param points_world: (B, N, 3) 三维世界点坐标
    :param k_mat: (B, 3, 3) 相机内参矩阵
    :param r_mat: (B, 3, 3) 旋转外参矩阵
    :param t_vec: (B, 3) 或 (B, 3, 1) 平移外参向量
    :param img_size: (width, height) 图像分辨率
    :return: (pixels_2d, depths, valid_mask)
    """
    if t_vec.dim() == 2:
        t_vec = t_vec.unsqueeze(-1) # (B, 3, 1)

    # 1. 世界坐标系 -> 相机坐标系: P_c = R * P_w + t
    # points_world: (B, N, 3) -> 转置为 (B, 3, N)
    pts_w_t = points_world.transpose(1, 2)
    pts_c = torch.bmm(r_mat, pts_w_t) + t_vec # (B, 3, N)

    # 提取物理深度 Z_c
    depths = pts_c[:, 2, :] # (B, N)

    # 2. 相机坐标系 -> 齐次像素坐标: P_img = K * P_c
    pts_img_homo = torch.bmm(k_mat, pts_c) # (B, 3, N)

    # 3. 透视除法: u = X_img / Z_c, v = Y_img / Z_c
    safe_depths = torch.clamp_min(depths, 1e-5).unsqueeze(1) # (B, 1, N)
    pixels_uv = pts_img_homo[:, :2, :] / safe_depths        # (B, 2, N)
    pixels_2d = pixels_uv.transpose(1, 2)                   # (B, N, 2)

    # 4. 视锥有效性裁剪 (深度必须为正，且落在图像分辨率内)
    w_limit, h_limit = img_size
    valid_depth = depths > 0.1
    valid_u = (pixels_2d[..., 0] >= 0) & (pixels_2d[..., 0] < w_limit)
    valid_v = (pixels_2d[..., 1] >= 0) & (pixels_2d[..., 1] < h_limit)
    valid_mask = valid_depth & valid_u & valid_v # (B, N)

    return pixels_2d, depths, valid_mask

# ===================================================================
# 单元测试与手算算例精确校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 1
    num_points = 3
    img_w, img_h = 800, 600

    # 构造测试内参
    k = torch.tensor([[
        [1000.0, 0.0, 400.0],
        [0.0, 1000.0, 300.0],
        [0.0, 0.0, 1.0]
    ]], dtype=torch.float32)

    # 构造外参：无旋转，Z 轴后退 2 米
    r = torch.eye(3).unsqueeze(0)
    t = torch.tensor([[0.0, 0.0, -2.0]])

    # 构造 3 个世界点：
    # 点 1: [1.0, 2.0, 10.0] -> 对应正文手算算例，应为 (525, 550)
    # 点 2: [0.0, 0.0, 12.0] -> 位于主光轴上，应为 (400, 300)
    # 点 3: [100.0, 100.0, 5.0] -> 超出视野边界，valid 应为 False
    pts_w = torch.tensor([[
        [1.0, 2.0, 10.0],
        [0.0, 0.0, 12.0],
        [100.0, 100.0, 5.0]
    ]], dtype=torch.float32)

    pixels, depths, valid = batch_project_points_3d(pts_w, k, r, t, img_size=(img_w, img_h))

    p1_u, p1_v = pixels[0, 0].tolist()
    p2_u, p2_v = pixels[0, 1].tolist()

    print(f"[Projection Test] 点 1 投影像素: ({p1_u:.1f}, {p1_v:.1f}), 期望: (525.0, 550.0)")
    print(f"[Projection Test] 点 2 投影像素: ({p2_u:.1f}, {p2_v:.1f}), 期望: (400.0, 300.0)")
    print(f"[Projection Test] 视锥有效掩码: {valid[0].tolist()}")

    assert abs(p1_u - 525.0) < 1e-4 and abs(p1_v - 550.0) < 1e-4, "点 1 手算算例校验失败！"
    assert abs(p2_u - 400.0) < 1e-4 and abs(p2_v - 300.0) < 1e-4, "点 2 主点投影校验失败！"
    assert valid[0, 0].item() is True and valid[0, 2].item() is False, "视锥裁剪掩码逻辑错误！"
    print("✓ 相机几何投影与批量视锥裁剪引擎单测全部通过！")
```

---

## 9.1.5 本节小结

回顾本节内容，我们建立了三维空间向二维图像投射的严密数学图谱：
1. **相似三角形与针孔模型**：透视投影以物理深度 $Z_c$ 反比缩放物体尺寸，奠定了“近大远小”的光学本质；
2. **内参矩阵 $\mathbf{K}$**：封装了焦距与主点偏移，将毫米物理尺寸转换为数字像素网格；
3. **外参刚体变换 $[\mathbf{R} \mid \mathbf{t}]$**：通过欧氏刚体变换连接世界坐标系与相机视线，齐次矩阵乘法与透视除法完成了三维到二维维度的投影压缩。
