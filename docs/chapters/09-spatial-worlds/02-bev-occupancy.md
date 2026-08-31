# 鸟瞰图（BEV）与占据网格（Occupancy Grid）

在现代自动驾驶与具身智能领域，我们面临着一个本质的几何认知难题：智能体通过摄像头获取的图像，是三维物理世界在二维像素平面上的投影；然而，要在世界中安全地导航、规划与避障，智能体必须在三维真实空间中进行推理。长久以来，计算机视觉界主要在图像视图（Image View，或称透视图Perspective View）中解决目标检测与语义分割问题。然而，这种二维表征存在着严重的遮挡效应、尺度随距离剧烈变化等缺陷，且无法直接被下游的路径规划器与控制算法所使用。

为了弥合二维感知与三维规划之间的鸿沟，鸟瞰图（Bird's-Eye View, BEV）与占据网格（Occupancy Grid）表征应运而生。BEV将感知结果从各个摄像头的透视空间统一转换到自车（Ego Vehicle）正上方的俯视平面，而占据网格进一步保留了高度信息，将三维空间离散化为体素（Voxel）。这种基于空间世界模型（Spatial World Models）的架构能够天然地融合多传感器数据，使得智能体对空间中“空闲区域”（Free Space）的理解达到前所未有的精度。

在本节中，我们将首先回溯到这门技术的几何基石。正是基于这些严谨的几何与矩阵代数知识，诸如 [[Philion & Fidler, 2020]](https://arxiv.org/abs/2008.05711) 提出的 Lift, Splat, Shoot (LSS) 范式，以及随后 [[Li et al., 2022]](https://arxiv.org/abs/2203.17270) 提出的 BEVFormer 才得以建立起纯视觉 3D 空间感知的坚实理论体系。我们将从最基础的透视几何出发，一步步推导如何让神经网络学会在脑海中重构三维世界。

## 相机成像背后的几何投影基础

为了理解神经网络如何将二维像素“拉升”（Lift）回三维空间，我们必须首先清楚三维世界是如何被“压缩”成二维图像的。这是探讨所有空间表征模型的起点。

我们可以将摄像机的成像过程视为光线沿直线传播这一物理现象在几何学上的自然推演。假设我们使用的是一个理想的针孔相机（Pinhole Camera）。在最基础的高中几何中我们学过相似三角形：当三维空间中的物体反射的光线穿过一个小孔，会在后方的感光屏上留下倒立的像。为了数学表达上的直观与便捷，在多视图几何中，我们通常将虚拟的成像平面置于相机光心（小孔）的前方，从而得到一个正立的像。

整个投影过程涉及多个坐标系之间的严格仿射变换与透视除法。设三维物理世界中有一个点，其在世界坐标系下的齐次坐标表示为 $\mathbf{P}_w = [X_w, Y_w, Z_w, 1]^\top$。

首先，我们需要将这个点转换到相机的自身参照系中。引入相机外参矩阵（Extrinsic Matrix）$\mathbf{T} \in \mathbb{R}^{4 \times 4}$，该矩阵包含了相机的旋转矩阵 $\mathbf{R} \in \mathbb{R}^{3 \times 3}$ 和平移向量 $\mathbf{t} \in \mathbb{R}^{3 \times 1}$。该点在相机坐标系下的坐标 $\mathbf{P}_c = [X_c, Y_c, Z_c, 1]^\top$ 可以通过如下矩阵乘法获得：

$$
\mathbf{P}_c = \mathbf{T} \mathbf{P}_w = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0} & 1 \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}
$$

接下来，我们需要将相机坐标系下的三维点 $\mathbf{P}_c$ 投影到位于 $Z_c = f$ 处的二维图像物理平面上，其中 $f$ 为相机的焦距。根据相似三角形定理，投影点在物理图像平面上的坐标 $(x, y)$ 必然满足以下等比例关系：

$$
\frac{x}{f} = \frac{X_c}{Z_c}, \quad \frac{y}{f} = \frac{Y_c}{Z_c}
$$

最后，物理平面上的连续坐标需要被采样为感光元件上的离散像素坐标 $(u, v)$。考虑到感光元件在横纵方向上的物理尺寸比例，以及图像坐标原点往往在左上角的设定，我们引入相机内参矩阵（Intrinsic Matrix）$\mathbf{K} \in \mathbb{R}^{3 \times 3}$。结合该公式，完整的从相机坐标到像素坐标的投影可以被写成如下的严格矢量化形式：

$$
Z_c \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = \mathbf{K} \mathbf{P}_c^{(1:3)}
$$

在该公式中，等式左侧的标量 $Z_c$ 起到了至关重要的作用——透视除法（Perspective Division）。在实际计算中，我们通过矩阵相乘后，必须将得到的前两项除以第三项（即深度 $Z_c$），才能得到真实的二维像素坐标 $(u, v)$。正是这个除法操作，导致了深度信息的不可逆丢失。

## 逆投影悖论与 Lift, Splat, Shoot 范式

当智能体试图从二维图像逆推三维世界时，面临着一个病态（Ill-posed）的数学问题。观察该公式我们可以发现，给定图像上的一个确定的像素坐标 $(u, v)$，以及已知的内参矩阵 $\mathbf{K}$，我们能够求得的仅仅是该点在三维空间中所在的一条射线的方向向量：

$$
\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = Z_c \mathbf{K}^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

该公式中，深度标量 $Z_c$ 成为了唯一的自由度。换言之，二维像素完全缺乏深度的绝对数值，导致沿着这条射线上的任何一个三维点，都能在投影后严丝合缝地重叠在同一像素上。这就是困扰纯视觉 3D 目标检测多年的核心瓶颈。

传统的方法试图通过深度估算网络，让模型回归出一个确定性的深度值 $\hat{Z}_c$。然而，由于遮挡和纹理缺失，单目深度估计往往存在极大的不确定性。一旦回归的深度出现偏差，三维空间中的特征投影就会彻底错位。

Lift, Splat, Shoot（LSS）为每个图像位置预测离散深度分布，并把图像特征沿视锥提升到三维后汇聚到 BEV 网格 [[Philion & Fidler, 2020]](https://arxiv.org/abs/2008.05711)。深度分布保留了多个可能深度，但并不能从根本上消除深度估计误差。

::: info 说明
我们可以将“提升（Lift）”操作想象为一座灯塔向夜空中发射光束的过程。图像平面上的每一个像素都是一个微小的光源，像素的语义特征决定了这束光的“颜色”；而深度概率分布则决定了这束光在距离灯塔不同深度的夜空中，哪里显得最为明亮。通过这种方式，原本扁平的二维特征图在三维空间中弥散开来，化作一片充满概率的三维特征光晕（视锥）。这是理解连续物理映射与离散张量操作之间的绝佳桥梁。
:::

具体而言，我们预先定义一个包含 $D$ 个离散深度面（例如每隔 0.5 米取一个值）的深度集合 $\mathcal{D} = \{d_1, d_2, \dots, d_D\}$。给定从图像主干网络（Backbone）中提取的像素特征向量 $\mathbf{c} \in \mathbb{R}^C$，以及通过一个平行分支预测出的对应于该像素的深度概率分布 $\mathbf{p} \in \mathbb{R}^D$（满足 $\sum_{i=1}^D p_i = 1$），我们在三维空间距离光心 $d_i$ 处的特征表示 $\mathbf{f}_{d_i}$ 可以被严谨地定义为特征向量与概率标量的乘积：

$$
\mathbf{f}_{d_i} = p_i \cdot \mathbf{c}
$$

如果我们用张量的视角来审视，(**这实际上是一个跨越维度的外积操作**)。设输入的特征图张量维度为 $C \times H \times W$，深度分布张量的维度为 $D \times H \times W$，通过张量的广播机制相乘，我们将得到一个维度为 $C \times D \times H \times W$ 的视锥（Frustum）特征张量。这个四维张量完美地保留了每一个像素特征在 $D$ 个不同深度面上的可能性。

得到了视锥特征后，接下来的操作被称为“Splat”。借助该公式和外参矩阵的逆矩阵 $\mathbf{T}^{-1}$，我们可以精确地计算出这 $D \times H \times W$ 个网格点在世界坐标系下的真实物理三维坐标 $(X_w, Y_w, Z_w)$。随后，我们预先在自车正上方的空间中定义一个标准的鸟瞰图 BEV 网格（例如 $X$ 轴和 $Y$ 轴上每隔 0.2 米划分一个格子）。由于每一个视锥点都被映射到了绝对物理空间中，我们将它们落入同一个 BEV 格子的所有特征向量执行求和（Sum Pooling）或最大化（Max Pooling）操作，从而将三维的视锥张量“压扁”回一个 $C \times H_{bev} \times W_{bev}$ 的 BEV 俯视特征图。

## 空间交叉注意力机制：BEVFormer 范式

LSS 采用“先提升到视锥、再聚合到 BEV”的 2D 到 3D 路线。视锥张量的大小随图像分辨率、深度分桶数和特征维度的乘积增长，而不是指数增长；多个相机特征在 BEV 中通过池化聚合，也会限制自适应融合能力。

BEVFormer 从 BEV 查询出发，把参考点投影到多相机图像并用空间交叉注意力采样特征 [[Li et al., 2022]](https://arxiv.org/abs/2203.17270)。这种“由 BEV 查询图像”的方向区别于 LSS 的视锥提升，并影响了许多后续方法；原论文并不能支持“绝大多数端到端自动驾驶模型都以它为基座”的绝对结论。

在 BEVFormer 中，我们首先在二维 BEV 空间中初始化一组可学习的网格查询参数（Grid Queries）$\mathbf{Q} \in \mathbb{R}^{H_{bev} \times W_{bev} \times C}$。每一个查询向量 $\mathbf{q}_p$ 唯一对应物理世界中的一根自车周围的垂直柱子（Pillar）。

为了让查询向量能够看到周围的环境，模型执行了一种被称为空间交叉注意力（Spatial Cross-Attention, SCA）的精密机制：

1. 对于每一个查询点（对应柱子的中心 $(X_w, Y_w)$），我们沿着高度 $Z_w$ 方向均匀采样一系列三维参考点 $\mathbf{P}_{ref} = [X_w, Y_w, Z_i]^\top$。
2. 我们利用这两个公式，将这些三维参考点严格地投影到所有相机的二维图像面上，获得对应的二维像素参考点坐标集 $\mathbf{p}_{ref, 2D}$。
3. (**查询这些二维坐标点周围的图像特征**) 并非使用传统的稠密注意力，而是引入了可变形注意力机制（Deformable Attention），只在被投影位置的局部极小邻域内采样并加权求和，从而极大地削减了计算开销：

$$
\text{SCA}(\mathbf{q}_p, \mathbf{F}) = \sum_{i=1}^{N_{ref}} \sum_{v \in \mathcal{V}_{hit}} \text{DeformAttn}(\mathbf{q}_p, \mathbf{p}_{ref, 2D}^{(i, v)}, \mathbf{F}_v)
$$

在这个公式中，$\mathcal{V}_{hit}$ 表示该 3D 参考点能够成功投影进其视野的有效相机视图集合，$\mathbf{F}_v$ 是第 $v$ 个相机的图像特征图。由此可见，BEVFormer 优雅地绕开了显式深度的估算，让网络在注意力权重的学习过程中，隐式地完成了从三维到二维的特征寻址任务。

## 迈向真三维：占据网格（Occupancy Grid）的崛起

鸟瞰图（BEV）表征在大多数二维地面导航任务中表现卓越，但它在本质上是对物理世界 $Z$ 轴（高度）的一次暴力压缩。设想一辆自动驾驶汽车正在接近一辆拖车或者一块悬空的广告牌：在 BEV 平面中，这些悬空物体的特征会被投影在地面上，系统极易将其误判为不可通行的路障。反之，那些不规则形状的灌木或是底盘较高的特种车辆，也难以用传统的 3D 边界框（Bounding Box）进行精确建模。

占据网格（Occupancy Grid）网络由此应运而生。它是 BEV 的高维拓展，将自车周围的三维空间精细地划分为大小一致的三维体素（Voxel）。我们的目标不再是预测 2D 的俯视图特征，而是将特征空间扩展为 $C \times X_{dim} \times Y_{dim} \times Z_{dim}$ 的真实三维张量。

网络需要对空间中的每一个体素执行分类任务，判断其处于三种状态之一：

1. **空闲（Free）**：代表可行驶空间。
2. **占据且无特定语义（Occupied-Unknown）**：这一项是 Occupancy 表征的杀手锏。它允许模型感知到前方存在障碍物（例如翻倒的奇形怪状的垃圾桶），即便该物体不属于任何预定义的训练类别（白名单之外的通用障碍物）。
3. **占据且具备语义（Occupied-Semantic）**：该体素被特定的物体（如汽车、行人、路牌）占据。

建立这种表征最大的挑战在于训练数据的真值（Ground Truth）获取。由于单帧激光雷达（LiDAR）产生的点云非常稀疏，无法直接作为稠密体素的监督信号。学术界（如 SurroundOcc 或 Occ3D 等工作）通常采用多帧点云拼接的方法。通过对一段较长轨迹上的 LiDAR 数据进行精确的姿态对齐与融合，构建出极度稠密的局部场景。随后，采用基于射线追踪（Ray-casting）的算法：从传感器原点发出的光线，沿途穿越的体素标记为“空闲”，光线击中的末端点标记为“占据”，未被光线探索到的背光面则标记为“未知”，从而构建出严谨的三维占据监督信号。

## 代码实现：构建简易的视锥生成与特征提升

为了将抽象的几何理论具象化，我们使用 PyTorch 实现 LSS 范式中最核心的“视锥生成”与“外积提升”模块。以下代码严密追踪了张量维度的每一次变换，确保符合该公式描述的逻辑。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FrustumGenerator(nn.Module):
    def __init__(self, depth_bound, depth_interval, downsample_factor, img_size):
        """
        初始化视锥生成器。
        参数：
            depth_bound (tuple): (深度最小值, 深度最大值)，单位为米。
            depth_interval (float): 深度的离散步长。
            downsample_factor (int): 图像特征图相对于原图的下采样倍率（如16）。
            img_size (tuple): 原始图像尺寸 (H, W)。
        """
        super().__init__()
        self.downsample_factor = downsample_factor
        self.feat_size = (img_size[0] // downsample_factor, img_size[1] // downsample_factor)

        # [构建离散深度的张量]
        self.depths = torch.arange(
            depth_bound[0], depth_bound[1], depth_interval, dtype=torch.float32
        )
        self.D = self.depths.shape[0]

    def create_frustum(self):
        """
        根据特征图尺寸和深度范围，构建相机坐标系下的基础视锥。
        返回张量形状：(D, H_f, W_f, 3)，其中最后一维是 (u, v, d) 坐标。
        """
        H_f, W_f = self.feat_size

        # 构建像素坐标网格
        xs = torch.linspace(0, W_f - 1, W_f, dtype=torch.float32)
        ys = torch.linspace(0, H_f - 1, H_f, dtype=torch.float32)
        # 注意 meshgrid 的索引顺序
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing='ij')

        # 将坐标重新缩放回原始图像尺寸
        x_grid = x_grid * self.downsample_factor
        y_grid = y_grid * self.downsample_factor

        # 扩展出深度维度：我们为每一个深度面复制一份二维网格
        x_frustum = x_grid.unsqueeze(0).repeat(self.D, 1, 1) # (D, H_f, W_f)
        y_frustum = y_grid.unsqueeze(0).repeat(self.D, 1, 1) # (D, H_f, W_f)

        # 扩展深度张量，匹配空间维度
        d_frustum = self.depths.view(-1, 1, 1).repeat(1, H_f, W_f) # (D, H_f, W_f)

        # 将三者拼接，构成视锥点的 (u, v, d) 表示
        frustum = torch.stack((x_frustum, y_frustum, d_frustum), dim=-1)
        return frustum

class LSSFeatureLifter(nn.Module):
    def __init__(self, in_channels, out_channels, num_depth_bins):
        super().__init__()
        self.num_depth_bins = num_depth_bins
        # 我们使用一个简单的卷积层同时预测图像上下文特征与深度概率
        self.cam_encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels + num_depth_bins, kernel_size=1)
        )

    def forward(self, img_feat):
        """
        执行 Lift 操作：将图像特征通过深度概率分布扩展至三维视锥。
        参数：
            img_feat: 图像主干网络输出特征 (B, C_in, H, W)
        """
        B, _, H, W = img_feat.shape
        # [通过 1x1 卷积预测深度与上下文特征]
        cam_out = self.cam_encoder(img_feat)

        # 沿通道维度切分，分离深度分支与上下文分支
        depth_logits = cam_out[:, :self.num_depth_bins, :, :]
        context_feat = cam_out[:, self.num_depth_bins:, :, :]

        # [计算离散深度层面上的概率分布]
        # 使用 softmax 确保概率和为 1，形状为 (B, D, H, W)
        depth_prob = F.softmax(depth_logits, dim=1)

        # [张量广播与外积]
        # context_feat: (B, C, 1, H, W)
        # depth_prob:   (B, 1, D, H, W)
        # 相乘后的 frustum_feat 形状为 (B, C, D, H, W)
        context_feat = context_feat.unsqueeze(2)
        depth_prob = depth_prob.unsqueeze(1)

        frustum_feat = context_feat * depth_prob
        return frustum_feat
```

这段代码精确地展示了连续物理世界在离散计算体系下的映射。视锥张量构建完成后，便可以使用相机内参矩阵将其投影回世界物理坐标系，为后续的体素池化（Voxel Pooling）奠定基础。

## 小结

- 从二维图像跨越到三维空间表征，其数学本质是**逆投影过程中的深度信息缺失**问题。
- **Lift-Splat-Shoot 范式**通过预测连续概率分布的方法，巧妙地避免了单一深度估算的弊端，通过特征外积生成了充满三维空间的视锥表示。
- **BEVFormer** 通过显式构建 BEV 网格，并利用 3D 参考点进行可变形的交叉注意力采样，大幅优化了特征聚合的效率与适应性。
- 占据网格网络将空间进一步提升至真正的三维（Voxel），不仅能够更细致地描绘复杂的空间几何轮廓，更为检测未定义的异常障碍物提供了可行的基础框架。
