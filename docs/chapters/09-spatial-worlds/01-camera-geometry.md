# 9.1 相机几何与三维视觉基础

> **本章导读**
>
> **讲什么：** 本章让世界模型从平面画面进入三维空间，并进一步加入时间形成四维场景。我们会从相机投影出发，构造鸟瞰图和三维占据表示，比较 NeRF 与 3D 高斯等静态重建方法，再预测空间随时间和车辆动作如何变化。
>
> **为什么二维画面不足以支持空间决策：** 图像中相邻的两个像素，可能来自现实中相距很远的物体；被卡车遮住的行人也不会因为画面里看不见就消失。自动驾驶和机器人需要的是距离、可通行区域、遮挡后的占据状态及其未来变化，而这些信息不能从单张二维投影中直接读出。
>
> **故事线：** `理解三维点如何投影成像素 → 把多视角特征提升到 BEV 与占据网格 → 重建静态三维场景 → 加入时间得到 4D 世界 → 用动作条件预测驾驶场景的未来`

把同一个路标放在相机前 2 米和 20 米处，它在图像中的像素高度会明显不同。要从像素反推距离，先要弄清三维点如何投影到图像平面。相机几何给出了坐标系、焦距、像素位置与深度之间的关系。

<div align="center">
<img src="/figures/09-spatial-worlds/source/01-camera-geometry/dust3r-fig1.png" alt="DUSt3R 从未标定图像直接恢复稠密三维点图与相机关系，展示相机几何最终要支持的真实重建任务。" width="86%">

_图 9.1-1：DUSt3R 从未标定图像直接恢复稠密三维点图与相机关系，展示相机几何最终要支持的真实重建任务。 出处：Shuzhe Wang et al.，[DUSt3R: Geometric 3D Vision Made Easy](https://arxiv.org/abs/2312.14132)（2024），Figure 1。_
</div>

Hartley 与 Zisserman 的《Multiple View Geometry in Computer Vision》系统整理了射影几何、相机模型和多视图约束 [[Hartley & Zisserman, 2003]](https://www.cambridge.org/core/books/multiple-view-geometry-in-computer-vision/0B6F289C78B2B23F596CAA76D3D43F7A)。本节只使用其中最基础的针孔模型，并暂时忽略镜头畸变。

下面依次推导透视投影、齐次坐标、内参与外参，再用 PyTorch 投影一组三维点。

## 针孔相机模型

在探讨复杂的现代镜头之前，我们先回到最简单的成像装置——小孔成像（Camera Obscura）。根据几何光学的基本原理，光在均匀介质中沿直线传播。当三维空间中的光线穿过一个极小的孔（针孔）时，会在孔后方的屏幕上形成一个倒立的实像。

为了方便数学分析，在计算机视觉中，我们通常将成像平面（Image Plane）对称地移动到针孔的前方。这样，成像就是正立的，且不改变投影的几何比例。

假设相机的投影中心（即针孔）位于三维坐标系的原点 $O_c = (0, 0, 0)$，相机的光轴（Optical Axis）沿着 $Z$ 轴的正方向。成像平面位于 $Z = f$ 处，这里的 $f$ 物理上代表相机的焦距（Focal Length）。

现在，考虑三维空间中的一个点 $P$。在相机坐标系下，它的坐标可以表示为 $P = (X, Y, Z)$。这条从原点 $O_c$ 出发、经过点 $P$ 的光线，会与成像平面相交于点 $p$。我们记点 $p$ 在成像平面上的物理坐标为 $(x, y, f)$。由于它在平面上，其 $Z$ 坐标必然为 $f$。

从侧面（例如 $Y-Z$ 平面或 $X-Z$ 平面）观察这个投影过程，我们可以清晰地看到两个相似三角形。以 $X-Z$ 平面为例，原点 $O_c$、点 $(X, 0, Z)$ 以及它们在 $Z$ 轴上的投影点构成了一个大三角形；而原点 $O_c$、点 $(x, 0, f)$ 及其在 $Z$ 轴上的投影点构成了一个小三角形。

<div align="center">
<img src="/figures/09-spatial-worlds/source/01-camera-geometry/mipnerf-fig1.png" alt="Mip-NeRF 对比单条相机射线与具有像素面积的锥台，直观呈现像素在三维空间中对应的采样区域。" width="86%">

_图 9.1-2：Mip-NeRF 对比单条相机射线与具有像素面积的锥台，直观呈现像素在三维空间中对应的采样区域。 出处：Jonathan T. Barron et al.，[Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields](https://arxiv.org/abs/2103.13415)（2021），Figure 1。_
</div>

根据相似三角形的比例关系，对应边的比值相等，我们立刻可以写出如下等式：

$$ \frac{x}{f} = \frac{X}{Z} $$

$$ \frac{y}{f} = \frac{Y}{Z} $$

通过简单的代数变形，我们可以得到投影点 $p$ 的物理坐标 $(x, y)$ 与空间点 $P$ 的坐标 $(X, Y, Z)$ 之间的关系：

$$ x = f \frac{X}{Z} $$

$$ y = f \frac{Y}{Z} $$

这就是针孔透视投影。对于相同大小的物体，深度 $Z$ 增加时，其投影尺寸按 $1/Z$ 缩小，这就是“近大远小”。

## 齐次坐标系与射影几何

由于要除以深度 $Z$，透视投影在普通欧几里得坐标中不是线性变换，不能只用一次固定矩阵乘法直接得到 $(x,y)$。

为了解决这个问题，我们需要引入射影几何中的一个伟大发明：**齐次坐标（Homogeneous Coordinates）**。

> 我们可以将齐次坐标理解为在现有的 $N$ 维空间中增加了一个“影子”维度。想象在二维平面上平移一个物体，这需要加法；但如果我们把这个二维平面看作是三维空间中高度为 $W=1$ 的一层纸，我们就可以通过在三维空间中对这层纸进行切变（Shear）操作（这是纯粹的线性矩阵乘法），来等价实现二维平面上的平移。当操作完成后，我们只需要除以这个额外维度的高度 $W$，就能重新把结果“投影”回原来的坐标系。

具体而言，对于一个二维向量 $(x, y)^\top$，其齐次坐标表示为增加了一个分量 $1$ 的三维向量 $(x, y, 1)^\top$。同样地，三维向量 $(X, Y, Z)^\top$ 的齐次坐标为四维向量 $(X, Y, Z, 1)^\top$。

最关键的规则是：在齐次坐标系下，任何向量乘以一个非零标量 $w$，仍然表示同一个几何点。即：
$$ (wx, wy, w)^\top \equiv (x, y, 1)^\top $$
要将齐次坐标转换为普通的非齐次坐标，只需将所有分量除以最后一个分量即可。

用齐次坐标可以把透视除法推迟到最后一步。考虑三维点 $(X,Y,Z)^\top$ 和图像平面点 $(x,y,1)^\top$，构造映射：

$$
\begin{bmatrix}
Z x \\
Z y \\
Z
\end{bmatrix}
=
\begin{bmatrix}
f X \\
f Y \\
Z
\end{bmatrix}
=
\begin{bmatrix}
f & 0 & 0 & 0 \\
0 & f & 0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
X \\
Y \\
Z \\
1
\end{bmatrix}
$$

向量 $(Zx,Zy,Z)^\top$ 与 $(x,y,1)^\top$ 表示同一个齐次点。矩阵乘法得到齐次坐标后，再除以第三个分量恢复普通像素坐标；非线性的透视除法并没有消失，只是被放到最后执行。

## 相机内参矩阵 (Intrinsic Matrix)

到目前为止，我们计算出的 $(x, y)$ 是在相机物理成像平面上的坐标，单位通常是米或毫米。但在真实的计算机中，图像是由一个个离散的像素（Pixel）组成的，像素坐标系的原点通常位于图像的左上角，向右为 $u$ 轴正方向，向下为 $v$ 轴正方向。

因此，我们需要将物理坐标 $(x, y)$ 转换为像素坐标 $(u, v)$。这个转换主要涉及两个步骤：

1. **缩放（Scaling）**：将物理长度单位转换为像素个数。
2. **平移（Translation）**：将坐标原点从图像中心移动到左上角。

假设在 $x$ 方向上每米包含 $m_x$ 个像素，在 $y$ 方向上每米包含 $m_y$ 个像素。那么，物理坐标 $x$ 和 $y$ 对应的像素个数分别为 $m_x x$ 和 $m_y y$。
此外，由于光轴通常穿过图像的中心（称为主点，Principal Point），而像素坐标系的原点在左上角，我们需要加上主点在像素坐标系下的偏移量 $(c_x, c_y)$。

将这些步骤结合起来，我们可以得到：

$$ u = m_x x + c_x $$

$$ v = m_y y + c_y $$

将这两个公式代入上述方程，得到：

$$ u = m_x f \frac{X}{Z} + c_x = f_x \frac{X}{Z} + c_x $$

$$ v = m_y f \frac{Y}{Z} + c_y = f_y \frac{Y}{Z} + c_y $$

这里，我们定义 $f_x = m_x f$ 和 $f_y = m_y f$ 为相机在 $u$ 和 $v$ 方向上的有效焦距（以像素为单位）。通常对于正方形像素，$f_x$ 和 $f_y$ 是非常接近的。

利用齐次坐标，可以把上述映射写成矩阵形式：

$$
\begin{bmatrix}
Z u \\
Z v \\
Z
\end{bmatrix}
=
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
X \\
Y \\
Z
\end{bmatrix}
$$

中间的 $3\times3$ 矩阵称为**相机内参矩阵（Camera Intrinsic Matrix）**，记作 $\mathbf K$。

<div align="center">
<img src="/figures/09-spatial-worlds/source/01-camera-geometry/deepv2d-fig2.png" alt="DeepV2D 的深度模块把相机模型、重投影代价体与深度更新连接起来，展示投影矩阵如何进入可学习三维视觉系统。" width="86%">

_图 9.1-3：DeepV2D 的深度模块把相机模型、重投影代价体与深度更新连接起来，展示投影矩阵如何进入可学习三维视觉系统。 出处：Zachary Teed；Jia Deng，[DeepV2D: Video to Depth with Differentiable Structure from Motion](https://arxiv.org/abs/1812.04605)（2020），Figure 2。_
</div>

$$
\mathbf{K} = \begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
$$

内参描述焦距、主点和像素尺度。在固定成像设置下可视为常数；图像缩放、裁剪、数字防抖或变焦后，应同步更新相应内参。

## 相机外参矩阵 (Extrinsic Matrix)

我们在前文中探讨的投影过程，都是建立在三维点 $P$ 是以**相机自身坐标系**（即以针孔为原点，光轴为 $Z$ 轴）来描述的前提下的。然而，在自动驾驶或三维重建中，我们通常需要在一个统一的**世界坐标系（World Coordinate System）**中描述所有物体。

这就要求我们在进行透视投影之前，先将世界坐标系中的三维点 $P_w = (X_w, Y_w, Z_w)^\top$ 变换到相机坐标系下的点 $P_c = (X_c, Y_c, Z_c)^\top$。

根据欧几里得几何理论，两个三维直角坐标系之间的变换可以通过一次刚体变换（Rigid Body Transformation）来描述，即一个旋转（Rotation）和一个平移（Translation）。

<div align="center">
<img src="/figures/09-spatial-worlds/source/01-camera-geometry/deepv2d-fig3.png" alt="DeepV2D 的运动模块从多帧残差流联合优化相机位姿，展示外参在可微结构恢复中的更新路径。" width="86%">

_图 9.1-4：DeepV2D 的运动模块从多帧残差流联合优化相机位姿，展示外参在可微结构恢复中的更新路径。 出处：Zachary Teed；Jia Deng，[DeepV2D: Video to Depth with Differentiable Structure from Motion](https://arxiv.org/abs/1812.04605)（2020），Figure 3。_
</div>

设世界坐标系到相机坐标系的旋转矩阵为 $\mathbf{R} \in SO(3)$，平移向量为 $\mathbf{t} \in \mathbb{R}^3$。那么两个坐标系下同一个点的坐标关系为：

$$ P_c = \mathbf{R} P_w + \mathbf{t} $$

平移在三维欧几里得坐标中不是线性算子。把 $P_w$ 和 $P_c$ 扩展成四维齐次向量后，旋转和平移可以统一写成 $4\times4$ 矩阵乘法：

$$
\begin{bmatrix}
P_c \\
1
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{R} & \mathbf{t} \\
\mathbf{0}^\top & 1
\end{bmatrix}
\begin{bmatrix}
P_w \\
1
\end{bmatrix}
$$

这里的 $\begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}$ 称为相机的**外参矩阵（Extrinsic Matrix）**。本节约定它把世界坐标变换到相机坐标；相机位姿本身是这个变换的逆。相机移动时外参会改变，而内参在固定成像设置下通常保持不变。

## 完整的投影模型

给定世界坐标中的三维点 $P_w$，先用外参变换到相机坐标，再用内参映射到像素齐次坐标。

结合这两个公式，我们得到完整的相机投影方程：

$$
Z_c \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
=
\mathbf{K} \begin{bmatrix} \mathbf{R} & \mathbf{t} \end{bmatrix}
\begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}
$$

<div align="center">
<img src="/figures/09-spatial-worlds/latex/01-camera-geometry/perspective-division-scale.png" alt="世界齐次点经外参与内参得到带深度尺度的三向量，再由第三分量完成透视除法" width="86%">

_图 9.1-5：投影矩阵先输出带共同尺度的齐次坐标；只有除以第三分量 Z_c，前两项才恢复为像素坐标。本文根据上式绘制。_
</div>

其中 $\mathbf P=\mathbf K[\mathbf R\mid\mathbf t]$ 是 $3\times4$ 相机投影矩阵。它描述理想针孔成像；真实镜头的径向、切向畸变还需要额外模型。

## 代码实现

下面用张量一次投影 $N$ 个三维点，并同时返回点是否位于相机前方。

```python
import torch

def project_points(points_3d, K, R, t):
    """
    将三维世界坐标点投影到二维图像平面。

    参数:
    points_3d: 形状为 (N, 3) 的三维点坐标张量 (X_w, Y_w, Z_w)
    K: 形状为 (3, 3) 的相机内参矩阵
    R: 形状为 (3, 3) 的旋转矩阵 (外参)
    t: 形状为 (3, 1) 的平移向量 (外参)

    返回:
    pixels_2d: 形状为 (N, 2) 的二维像素坐标 (u, v)
    valid: 形状为 (N,) 的布尔掩码，表示深度为正
    """
    # 步骤1：应用外参，将世界坐标系转换到相机坐标系
    # P_c = R * P_w + t
    # 注意维度对齐：points_3d (N, 3)，R^T (3, 3) -> (N, 3)
    points_c = torch.matmul(points_3d, R.T) + t.T

    # 步骤2：应用内参，进行透视投影
    # P_img = K * P_c
    points_img = torch.matmul(points_c, K.T)

    # 步骤3：透视除法 (Perspective Division)，从齐次坐标恢复到非齐次二维坐标
    depth = points_c[:, 2:3]
    valid = depth[:, 0] > 1e-6
    safe_depth = depth.clamp_min(1e-6)
    pixels_2d = points_img[:, :2] / safe_depth

    return pixels_2d, valid

# 模拟数据
# 定义相机内参矩阵，假设图像尺寸为 800x600，焦距约为 1000 像素
K = torch.tensor([
    [1000.0, 0.0, 400.0],
    [0.0, 1000.0, 300.0],
    [0.0, 0.0, 1.0]
], dtype=torch.float32)

# 定义无旋转的单位矩阵，平移为 Z 轴方向向后移动 5 个单位
R = torch.eye(3, dtype=torch.float32)
t = torch.tensor([[0.0], [0.0], [-5.0]], dtype=torch.float32)

# 随机生成 5 个三维世界点
points_w = torch.tensor([
    [0.0, 0.0, 10.0],
    [1.0, 2.0, 15.0],
    [-2.0, -1.0, 20.0],
    [3.0, -2.0, 12.0],
    [-1.0, 3.0, 8.0]
], dtype=torch.float32)

# 计算投影结果
uv_coords, valid = project_points(points_w, K, R, t)
print(uv_coords)
print("位于相机前方:", valid)
```

代码中的 `R.T` 与 `t.T` 是为了让行向量批次满足 $P_c=P_wR^\top+t^\top$。实际使用时还应检查像素是否落在图像边界内，并应用相机标定得到的畸变参数。

## 小结

- **针孔相机模型**通过简单的相似三角形，描述了三维世界投射到二维平面的几何光学规律。
- 齐次坐标把透视除法推迟到矩阵乘法之后，最终仍要除以深度分量。
- **相机内参矩阵 $\mathbf{K}$** 描述焦距、主点和像素尺度，也会随图像缩放、裁剪或变焦改变。
- 相机外参包含了旋转矩阵 $\mathbf{R}$ 和平移向量 $\mathbf{t}$，描述了相机在三维世界中的位置与朝向。
- **相机投影矩阵**是针孔模型下内参与外参的乘积；镜头畸变需要单独处理。
