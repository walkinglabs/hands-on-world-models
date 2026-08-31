# 9.2 鸟瞰图（BEV）与占据网格（Occupancy Grid）

前视相机里，远处车道很窄，近处车道很宽；换到俯视坐标后，同样一米在不同位置可以对应相同网格尺度。鸟瞰图（Bird's-Eye View, BEV）把多相机特征放到统一地面坐标，三维占据网格则继续保留高度，用于描述空间中哪些体素空闲、被占据或属于某个语义类别。

<div align="center">
<img src="/figures/09-spatial-worlds/source/02-bev-occupancy/lss-fig1.png" alt="Lift-Splat-Shoot 把环视相机图像变成统一鸟瞰表示，显示 BEV 如何服务车辆周围的空间推理。" width="86%">

_图 9.2-1：Lift-Splat-Shoot 把环视相机图像变成统一鸟瞰表示，显示 BEV 如何服务车辆周围的空间推理。 出处：Jonah Philion；Sanja Fidler，[Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D](https://arxiv.org/abs/2008.05711)（2020），Figure 1。_
</div>

BEV 与占据都需要显式处理相机标定、深度歧义和遮挡。统一坐标便于融合传感器和连接下游模块，但精度仍取决于数据、分辨率、标定和监督质量。

LSS 预测像素的离散深度分布，再把图像特征提升并汇聚到 BEV [[Philion & Fidler, 2020]](https://arxiv.org/abs/2008.05711)；BEVFormer 则从 BEV 查询出发，到多相机特征中采样 [[Li et al., 2022]](https://arxiv.org/abs/2203.17270)。下面先解释它们共享的投影几何，再比较两种特征聚合方向。

## 相机成像背后的几何投影基础

二维到三维的困难来自投影的多对一：一条射线上的许多三维点会落到同一个像素。

我们可以将摄像机的成像过程视为光线沿直线传播这一物理现象在几何学上的自然推演。假设我们使用的是一个理想的针孔相机（Pinhole Camera）。在最基础的高中几何中我们学过相似三角形：当三维空间中的物体反射的光线穿过一个小孔，会在后方的感光屏上留下倒立的像。为了数学表达上的直观与便捷，在多视图几何中，我们通常将虚拟的成像平面置于相机光心（小孔）的前方，从而得到一个正立的像。

设三维点在世界坐标系下的齐次坐标为 $\mathbf{P}_w=[X_w,Y_w,Z_w,1]^\top$。先用外参变换到相机坐标，再做透视除法。

首先，我们需要将这个点转换到相机的自身参照系中。引入相机外参矩阵（Extrinsic Matrix）$\mathbf{T} \in \mathbb{R}^{4 \times 4}$，该矩阵包含了相机的旋转矩阵 $\mathbf{R} \in \mathbb{R}^{3 \times 3}$ 和平移向量 $\mathbf{t} \in \mathbb{R}^{3 \times 1}$。该点在相机坐标系下的坐标 $\mathbf{P}_c = [X_c, Y_c, Z_c, 1]^\top$ 可以通过如下矩阵乘法获得：

$$
\mathbf{P}_c = \mathbf{T} \mathbf{P}_w = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0} & 1 \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}
$$

接下来，我们需要将相机坐标系下的三维点 $\mathbf{P}_c$ 投影到位于 $Z_c = f$ 处的二维图像物理平面上，其中 $f$ 为相机的焦距。根据相似三角形定理，投影点在物理图像平面上的坐标 $(x, y)$ 必然满足以下等比例关系：

$$
\frac{x}{f} = \frac{X_c}{Z_c}, \quad \frac{y}{f} = \frac{Y_c}{Z_c}
$$

最后，物理平面坐标要换算为像素坐标 $(u,v)$。相机内参矩阵 $\mathbf K\in\mathbb R^{3\times3}$ 编码焦距、主点和像素尺度，投影可写为：

$$
Z_c \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = \mathbf{K} \mathbf{P}_c^{(1:3)}
$$

矩阵乘法后要把前两项除以第三项 $Z_c$ 才得到像素坐标 $(u,v)$。单张图像中的一个像素没有携带唯一深度；多视角、运动、双目或学习先验可以进一步约束它。

## 逆投影悖论与 Lift, Splat, Shoot 范式

从单个二维像素逆推三维位置是欠定问题。给定像素 $(u,v)$ 和内参 $\mathbf K$，只能求出一条射线的方向：

$$
\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = Z_c \mathbf{K}^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

<div align="center">
<img src="/figures/09-spatial-worlds/latex/02-bev-occupancy/pixel-ray-depth-family.png" alt="同一像素经内参逆变换只确定射线方向，不同正深度对应射线上不同三维点" width="86%">

_图 9.2-2：像素和内参固定的是射线方向；深度 Z_c 仍是自由变量，因此射线上多个三维点会投影到同一个像素。本文根据上式绘制。_
</div>

深度 $Z_c$ 是这条射线上的自由度。仅凭像素 $(u,v)$，射线上的不同三维点具有相同投影；LSS 用预测的深度分布表达这种不确定性。

可以让网络回归单一深度 $\hat Z_c$，但遮挡、弱纹理和尺度歧义会造成误差。深度误差随后会转化为三维位置误差。

Lift, Splat, Shoot（LSS）为每个图像位置预测离散深度分布，并把图像特征沿视锥提升到三维后汇聚到 BEV 网格 [[Philion & Fidler, 2020]](https://arxiv.org/abs/2008.05711)。深度分布保留了多个可能深度，但并不能从根本上消除深度估计误差。

<div align="center">
<img src="/figures/09-spatial-worlds/source/02-bev-occupancy/lss-fig4.png" alt="LSS 将图像特征沿离散深度提升到视锥体，再汇聚到鸟瞰网格并输出规划相关预测。" width="86%">

_图 9.2-3：LSS 将图像特征沿离散深度提升到视锥体，再汇聚到鸟瞰网格并输出规划相关预测。 出处：Jonah Philion；Sanja Fidler，[Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D](https://arxiv.org/abs/2008.05711)（2020），Figure 4。_
</div>

具体而言，我们预先定义一个包含 $D$ 个离散深度面（例如每隔 0.5 米取一个值）的深度集合 $\mathcal{D} = \{d_1, d_2, \dots, d_D\}$。给定从图像主干网络（Backbone）中提取的像素特征向量 $\mathbf{c} \in \mathbb{R}^C$，以及通过一个平行分支预测出的对应于该像素的深度概率分布 $\mathbf{p} \in \mathbb{R}^D$（满足 $\sum_{i=1}^D p_i = 1$），我们在三维空间距离光心 $d_i$ 处的特征表示 $\mathbf{f}_{d_i}$ 可以被严谨地定义为特征向量与概率标量的乘积：

$$
\mathbf{f}_{d_i} = p_i \cdot \mathbf{c}
$$

设图像特征为 $C\times H\times W$，深度分布为 $D\times H\times W$。广播相乘后得到 $C\times D\times H\times W$ 的视锥特征；同一个像素特征按预测概率分配到 $D$ 个深度区间。

“Splat”把视锥采样点依据内外参转换到自车坐标，再汇聚到预定义的 BEV 网格。落入同一格子的特征可以求和或池化，得到 $C\times H_{bev}\times W_{bev}$ 的俯视特征。坐标只在标定准确时成立，离散网格还会引入量化误差。

## 空间交叉注意力机制：BEVFormer 范式

LSS 采用“先提升到视锥、再聚合到 BEV”的 2D 到 3D 路线。视锥张量的大小随图像分辨率、深度分桶数和特征维度的乘积增长，而不是指数增长；多个相机特征在 BEV 中通过池化聚合，也会限制自适应融合能力。

BEVFormer 从 BEV 查询出发，把参考点投影到多相机图像并用空间交叉注意力采样特征 [[Li et al., 2022]](https://arxiv.org/abs/2203.17270)。这种“由 BEV 查询图像”的方向不同于 LSS 的视锥提升；这里只比较两者的特征流向，不把它概括为所有端到端驾驶模型的共同基座。

<div align="center">
<img src="/figures/09-spatial-worlds/source/02-bev-occupancy/bevformer-fig2.png" alt="BEVFormer 用空间交叉注意力从多相机特征更新 BEV 查询，并用时间自注意力融合历史 BEV。" width="86%">

_图 9.2-4：BEVFormer 用空间交叉注意力从多相机特征更新 BEV 查询，并用时间自注意力融合历史 BEV。 出处：Zhiqi Li et al.，[BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers](https://arxiv.org/abs/2203.17270)（2022），Figure 2。_
</div>

在 BEVFormer 中，我们首先在二维 BEV 空间中初始化一组可学习的网格查询参数（Grid Queries）$\mathbf{Q} \in \mathbb{R}^{H_{bev} \times W_{bev} \times C}$。每一个查询向量 $\mathbf{q}_p$ 唯一对应物理世界中的一根自车周围的垂直柱子（Pillar）。

为了让查询读取图像特征，模型使用空间交叉注意力（Spatial Cross-Attention, SCA）：

1. 对于每一个查询点（对应柱子的中心 $(X_w, Y_w)$），我们沿着高度 $Z_w$ 方向均匀采样一系列三维参考点 $\mathbf{P}_{ref} = [X_w, Y_w, Z_i]^\top$。
2. 利用相机内外参，把三维参考点投影到各相机图像，得到二维参考坐标集 $\mathbf{p}_{ref,2D}$。
3. 使用可变形注意力在投影位置附近采样少量图像特征并加权求和：

$$
\text{SCA}(\mathbf{q}_p, \mathbf{F}) = \sum_{i=1}^{N_{ref}} \sum_{v \in \mathcal{V}_{hit}} \text{DeformAttn}(\mathbf{q}_p, \mathbf{p}_{ref, 2D}^{(i, v)}, \mathbf{F}_v)
$$

其中 $\mathcal{V}_{hit}$ 表示参考点具有正深度且投影位于画面内的相机集合，$\mathbf F_v$ 是第 $v$ 个相机的特征图。BEVFormer 不为每个像素显式预测深度分布，而是从预设的三维参考点回到图像采样。

## 迈向真三维：占据网格（Occupancy Grid）的崛起

二维 BEV 往往沿高度聚合特征，因此难以区分地面障碍、悬空结构和可从下方通过的空间。三维占据网格把高度轴保留下来，也能表示不适合用边界框描述的形状。

<div align="center">
<img src="/figures/09-spatial-worlds/source/02-bev-occupancy/tpvformer-fig3.png" alt="TPVFormer 并列比较体素、BEV 与三正交平面表示，说明保留高度信息时的空间表征取舍。" width="86%">

_图 9.2-5：TPVFormer 并列比较体素、BEV 与三正交平面表示，说明保留高度信息时的空间表征取舍。 出处：Yuanhui Huang et al.，[TPVFormer: Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction](https://arxiv.org/abs/2302.07817)（2023），Figure 3。_
</div>

占据网格把自车周围空间划分为体素（Voxel），输出可组织为 $C\times X_{dim}\times Y_{dim}\times Z_{dim}$ 的三维张量。

一种标签设计把每个体素分为：

1. **空闲（Free）**：代表可行驶空间。
2. **占据但语义未知（Occupied-Unknown）**：几何上判断有物体，但不把它归入已知类别。是否能识别训练分布外障碍物仍需专门评测。
3. **占据且具备语义（Occupied-Semantic）**：该体素被特定的物体（如汽车、行人、路牌）占据。

占据真值也很难获得。常见做法是依据自车位姿对齐多帧激光雷达点云，再用射线遍历标记可观测空闲区域与击中位置。动态物体需要单独补偿运动，否则多帧融合会留下拖影；未被射线覆盖的体素应保留为未知，而不是当作空闲。

## 代码实现：构建简易的视锥生成与特征提升

下面实现视锥坐标和特征提升。代码尚未包含从相机坐标到自车坐标的转换，也没有实现体素池化。

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
        self.register_buffer(
            "depths",
            torch.arange(
                depth_bound[0], depth_bound[1], depth_interval,
                dtype=torch.float32
            )
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
        x_frustum = x_grid.unsqueeze(0).expand(self.D, -1, -1)
        y_frustum = y_grid.unsqueeze(0).expand(self.D, -1, -1)

        # 扩展深度张量，匹配空间维度
        d_frustum = self.depths.view(-1, 1, 1).expand(-1, H_f, W_f)

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

输出 `frustum_feat` 仍位于相机视锥索引中。下一步要用内外参把每个 $(u,v,d)$ 反投影到自车坐标，再按 BEV 网格索引做体素池化。

## 小结

- 从二维图像跨越到三维空间表征，其数学本质是**逆投影过程中的深度信息缺失**问题。
- **Lift-Splat-Shoot**预测离散深度分布，用外积生成视锥特征，再汇聚到 BEV。
- **BEVFormer**从 BEV 查询出发，把参考点投影回多相机图像并采样特征。
- 三维占据保留高度信息，但同时增加显存、标注和遮挡推断的难度。
