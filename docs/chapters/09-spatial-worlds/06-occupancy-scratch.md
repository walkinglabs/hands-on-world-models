# 9.6 三维占据网格预测（Occupancy Prediction）的从零开始实现

在进入三维占据预测之前，我们先看 BEV 表示。BEVFormer 用网格状 BEV 查询与多相机图像交互，并结合历史 BEV 特征，支持三维目标检测和地图分割 [[Li et al., 2022]](https://arxiv.org/abs/2203.17270)。原论文没有评测轨迹规划，因此这里不把“简化下游规划”写成由该引用直接证明的实验结论。

<div align="center">
<img src="/figures/09-spatial-worlds/source/06-occupancy-scratch/monoscene-fig1.png" alt="MonoScene 从单张道路图像恢复可见与遮挡区域的稠密三维语义占据，展示占据预测的直接输出。" width="86%">

_图 9.6-1：MonoScene 从单张道路图像恢复可见与遮挡区域的稠密三维语义占据，展示占据预测的直接输出。 出处：Anh-Quan Cao；Raoul de Charette，[MonoScene: Monocular 3D Semantic Scene Completion](https://arxiv.org/abs/2112.00726)（2022），Figure 1。_
</div>

二维 BEV 特征通常沿高度聚合信息，而三维占据表示把车辆周围空间离散为体素，并预测占用或语义类别。SurroundOcc 研究多相机图像到稠密三维占用的预测 [[Wei et al., 2023]](https://arxiv.org/abs/2303.09551)，OpenOccupancy 提供占用感知基准及多模态基线 [[Xiaofeng Wang et al., 2023b]](https://arxiv.org/abs/2303.03991)，TPVFormer 则用三视图表示进行三维语义占用预测 [[Huang et al., 2023]](https://arxiv.org/abs/2302.07817)。三篇引用分别对应方法或基准贡献，而不是共同证明一个笼统的“奠定基础”判断。

本节从体素离散化和占据概率出发，实现可学习体素查询、基于投影坐标的图像特征采样，以及三维分类头。代码展示核心张量接口，不复现任何一篇论文的完整训练系统。

## 空间几何离散化与占据概率分布

连续空间要先离散成有限网格。每个体素覆盖一小块三维区域，分辨率越高，几何边界越细，但存储和计算量也越大。

### 三维体素网格的定义

假设我们关注车辆周围的一个长方体物理空间。我们定义该空间在世界坐标系下的范围为 $[X_{\min}, X_{\max}] \times [Y_{\min}, Y_{\max}] \times [Z_{\min}, Z_{\max}]$。如果我们在三个维度上的空间分辨率（即每个微小正方体的边长）分别为 $r_x, r_y, r_z$，那么该空间可以被划分为一个三维网格，其网格的维度 $(W, H, D)$ 可以表示为：

$$W = \left\lceil\frac{X_{\max} - X_{\min}}{r_x}\right\rceil, \quad H = \left\lceil\frac{Y_{\max} - Y_{\min}}{r_y}\right\rceil, \quad D = \left\lceil\frac{Z_{\max} - Z_{\min}}{r_z}\right\rceil$$

在这个网格系统中，每一个离散的三维坐标索引 $(i, j, k)$（其中 $0 \le i < W, 0 \le j < H, 0 \le k < D$）都唯一对应物理空间中的一个体素微元。

### 占据状态的概率建模

对于网格中的任意体素 $v_{i,j,k}$，它的物理状态最基础的描述是“是否被占据”（Occupied or Free）。这是一个典型的伯努利试验（Bernoulli Trial）。我们设随机变量 $O_{i,j,k} \in \{0, 1\}$ 表示该状态：

$$P(O_{i,j,k} = 1) = p_{i,j,k}$$

当我们进一步考虑该空间的语义类别（例如：车辆、行人、道路、建筑物等）时，状态空间将扩展为多项分布（Multinomial Distribution）。假设共有 $C$ 个语义类别，并额外增加一个表示“游离态（Free）”的类别，总类别数为 $K = C + 1$。则我们希望模型预测的是条件概率分布：

$$\hat{y}_{i,j,k}^{(c)} = P(S_{i,j,k} = c \mid \mathbf{I})$$

其中 $\mathbf{I}$ 表示输入的多视角环视图像序列，$\hat{y}_{i,j,k}^{(c)}$ 表示体素属于第 $c$ 类的概率。

## 2D 到 3D 空间的特征提升（2D-to-3D Lift）

输入是二维透视图像，输出却是三维网格。一个像素只确定从相机出发的一条射线，不能单独确定深度；模型必须利用深度分布、多视角对应或三维查询来消除这部分歧义。

<div align="center">
<img src="/figures/09-spatial-worlds/source/06-occupancy-scratch/monoscene-fig3.png" alt="MonoScene 的 FLoSP 把二维多尺度特征沿相机视线投影到三维体素查询，处理 2D 到 3D 的特征提升。" width="86%">

_图 9.6-2：MonoScene 的 FLoSP 把二维多尺度特征沿相机视线投影到三维体素查询，处理 2D 到 3D 的特征提升。 出处：Anh-Quan Cao；Raoul de Charette，[MonoScene: Monocular 3D Semantic Scene Completion](https://arxiv.org/abs/2112.00726)（2022），Figure 3。_
</div>

### 深度分布估计

经典的 LSS (Lift, Splat, Shoot) [[Philion & Fidler, 2020]](https://arxiv.org/abs/2008.05711) 方法通过显式地预测像素的深度分布来解决这一问题。对于图像上提取的某一个特征像素点及其特征向量 $\mathbf{f} \in \mathbb{R}^F$，我们不预测一个确定性的单一深度值，而是预测它在预定义的离散深度区间 $[d_1, d_2, \ldots, d_D]$ 上的概率分布 $\mathbf{p} \in \mathbb{R}^D$。

通过特征向量 $\mathbf{f}$ 与概率分布 $\mathbf{p}$ 的外积，我们得到该像素对应的视锥体（Frustum）特征 $\mathbf{F} \in \mathbb{R}^{D \times F}$：

$$\mathbf{F} = \mathbf{p} \otimes \mathbf{f} \implies F_{d,c} = p_d f_c$$

<div align="center">
<img src="/figures/09-spatial-worlds/latex/06-occupancy-scratch/depth-feature-outer-product.png" alt="深度概率向量与像素特征向量做外积，每个深度概率缩放整条通道特征" width="86%">

_图 9.6-3：Lift 不是把深度概率与特征做点积，而是让每个 p_d 缩放整条 C 维特征，形成 D×C 的深度—通道切片。本文根据上式绘制。_
</div>

这意味着，二维像素的特征被沿着相机光心出发的射线，按照深度概率权重播撒到了三维空间中。

### 初始化三维体素查询特征

在真实工程中，如 SurroundOcc 项目，通常会显式地在三维空间中初始化一套可学习的体素特征参数（Voxel Queries） $\mathbf{Q} \in \mathbb{R}^{W \times H \times D \times C_{emb}}$。

<div align="center">
<img src="/figures/09-spatial-worlds/source/06-occupancy-scratch/surroundocc-fig3.png" alt="SurroundOcc 对比三维体素查询与 BEV 查询的跨视图注意力，说明高度维度何时被显式保留。" width="86%">

_图 9.6-4：SurroundOcc 对比三维体素查询与 BEV 查询的跨视图注意力，说明高度维度何时被显式保留。 出处：Yi Wei et al.，[SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving](https://arxiv.org/abs/2303.09551)（2023），Figure 3。_
</div>

(**我们先定义三维体素查询的初始化代码**)：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelQueryGenerator(nn.Module):
    def __init__(self, grid_size, embed_dims):
        super().__init__()
        # grid_size = [W, H, D]
        self.W, self.H, self.D = grid_size
        self.embed_dims = embed_dims

        # 显式初始化三维体素查询参数
        # 维度: (1, C, D, H, W)，遵循 PyTorch 常规的 3D 卷积排列方式
        self.voxel_queries = nn.Parameter(
            torch.zeros(1, embed_dims, self.D, self.H, self.W)
        )
        nn.init.normal_(self.voxel_queries, mean=0.0, std=0.02)

    def forward(self, batch_size):
        # expand 不复制底层参数存储
        return self.voxel_queries.expand(batch_size, -1, -1, -1, -1)

# 测试初始化
grid_size = [100, 100, 8]  # X=100, Y=100, Z=8 的体素网格
embed_dims = 64
query_gen = VoxelQueryGenerator(grid_size, embed_dims)
queries = query_gen(batch_size=2)
print("Voxel queries shape:", queries.shape)  # [2, 64, 8, 100, 100]
```

## 空间交叉注意力与特征聚合

拿到三维体素查询 $\mathbf{Q}$ 后，需要用二维图像特征更新它们。BEVFormer 从 BEV 查询的参考点出发，在多相机特征上做空间交叉注意力 [[Li et al., 2022]](https://arxiv.org/abs/2203.17270)。将相同接口推广到三维查询时，每个体素中心也要先投影到各相机。

<div align="center">
<img src="/figures/09-spatial-worlds/source/06-occupancy-scratch/occformer-fig1.png" alt="OccFormer 在局部与全局路径中更新三维体素特征，再解码为稠密语义占据。" width="86%">

_图 9.6-5：OccFormer 在局部与全局路径中更新三维体素特征，再解码为稠密语义占据。 出处：Yunpeng Zhang et al.，[OccFormer: Dual-path Transformer for Vision-based 3D Semantic Occupancy Prediction](https://arxiv.org/abs/2304.05316)（2023），Figure 1。_
</div>

对于任意一个三维体素中心点 $(x, y, z)$，我们可以通过相机的内外参矩阵 $\mathbf{P} \in \mathbb{R}^{3 \times 4}$ 将其投影到第 $i$ 个相机的图像平面像素坐标 $(u_i, v_i)$ 上：

$$d [u_i, v_i, 1]^T = \mathbf{P} \cdot [x, y, z, 1]^T$$

其中 $d$ 是该三维点在相机坐标系下的深度值。

只有深度 $d>0$ 且投影点位于图像边界内时，才具备采样条件；这仍不表示体素没有被其他物体遮挡。在这些有效位置上，可以使用双线性采样或可变形注意力读取图像特征：

$$\mathbf{Q}_{x,y,z}' = \mathbf{Q}_{x,y,z} + \sum_{i \in \mathcal{V}_{x,y,z}} \text{DeformAttn}(\mathbf{Q}_{x,y,z}, \mathbf{F}_i, \mathbf{P}(x,y,z))$$

其中 $\mathcal{V}_{x,y,z}$ 表示能观测到该体素的相机集合，$\mathbf{F}_i$ 是第 $i$ 视角的图像特征。

### 开源项目的架构演进

不同方法采用了不同的三维表示：

1. **SurroundOcc**：采用多尺度 3D 卷积网络来逐步上采样和精细化 3D Voxel 特征。为了显存可控，其通常采用相对较粗的初始网格分辨率。
2. **TPVFormer**：用顶视图、侧视图和正视图三个正交平面隐式表达三维空间。仅比较表示元素数量时，$WHD$ 变为 $WH+HD+WD$；完整计算量还取决于注意力和解码器。
3. **OpenOccupancy**：提供统一的占据数据处理与评测基准，并比较相机、激光雷达及融合基线，重点是开放基准而不是某一种固定网络结构。

下面假定相机投影已经生成归一化采样坐标，用 `grid_sample` 真正读取多相机特征。几何矩阵到采样坐标的转换可复用 9.1 节的投影函数。

```python
class SimpleSpatialCrossAttention(nn.Module):
    def __init__(self, embed_dims):
        super().__init__()
        self.embed_dims = embed_dims
        # 为了演示，我们使用简单的线性映射代替复杂的 Deformable Attention
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, voxel_queries, image_features, sampling_grids, visibility):
        """
        voxel_queries: [B, C, D, H, W]
        image_features: [B, N_cams, C, H_img, W_img]
        sampling_grids: [B, N_cams, Q, 1, 2]，坐标已归一化到 [-1, 1]
        visibility: [B, N_cams, Q]，表示正深度且位于画面内
        """
        B, C, D, H, W = voxel_queries.shape
        # 将体素展平为序列 [B, D*H*W, C]
        Q = D * H * W
        queries_flat = voxel_queries.reshape(B, C, Q).permute(0, 2, 1)

        B_img, N, C_img, H_img, W_img = image_features.shape
        assert (B_img, C_img) == (B, C)
        features = image_features.reshape(B * N, C, H_img, W_img)
        grids = sampling_grids.reshape(B * N, Q, 1, 2)
        sampled = F.grid_sample(
            features, grids, mode="bilinear", padding_mode="zeros",
            align_corners=False
        )
        sampled = sampled.squeeze(-1).reshape(B, N, C, Q).permute(0, 1, 3, 2)

        weights = visibility.to(sampled.dtype).unsqueeze(-1)
        sampled_context = (sampled * weights).sum(dim=1)
        sampled_context = sampled_context / weights.sum(dim=1).clamp_min(1.0)

        # 简单的线性融合
        updated_queries = queries_flat + self.output_proj(F.relu(self.value_proj(sampled_context)))

        # 恢复 3D 形状
        voxel_queries = updated_queries.permute(0, 2, 1).reshape(B, C, D, H, W)
        return voxel_queries
```

## 占据分类头与损失函数设计

经过多层 3D 卷积或 Transformer 的处理，我们最终得到了高分辨率的稠密三维特征图 $\mathbf{V}_{out} \in \mathbb{R}^{B \times C_{emb} \times D \times H \times W}$。

最后一步是施加一个分类头，将其映射到 $K$ 个语义类别的概率对数（Logits）。

(**定义占据网格的三维卷积预测头**)：

```python
class OccupancyHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # 使用 3D 卷积进行局部特征融合并输出分类预测
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # 输入维度: [B, C, D, H, W]
        # 输出维度: [B, num_classes, D, H, W]
        logits = self.conv(x)
        return logits

# 实例化预测头，假设16个语义类别 + 1个空闲类 = 17类
occ_head = OccupancyHead(in_channels=64, num_classes=17)
final_logits = occ_head(queries)
print("Final logits shape:", final_logits.shape)  # [2, 17, 8, 100, 100]
```

### 类别不平衡的三维空间损失

在常见的驾驶体素范围中，空闲体素通常多于占据体素，小物体类别的样本又更少。若直接对所有体素等权平均，损失可能主要反映空闲区域。

可以选用类别加权交叉熵、Focal Loss 或针对几何的辅助损失。令 $p_t$ 为模型分配给真实类别的概率，Focal Loss 写为：

$$L_{focal} = -\alpha_t(1-p_t)^\gamma\log p_t$$

其中 $\alpha_t$ 是类别权重，$\gamma$ 控制对易分样本的降权程度。它们是需要验证的超参数，并非所有数据集都必须采用同一设置。

相机还会受到遮挡。训练标签若包含被遮挡区域，模型实际同时在做可见表面识别和场景补全；评测时应区分可见、遮挡与未知体素，避免把没有观测证据的区域误当成确定空闲。

## 小结

- **三维占据网格预测**把多视角观测映射为体素级占用或语义分类，保留高度信息，但仍受分辨率、遮挡和标注误差限制。
- 核心操作包含了三维体素查询的初始化，以及基于相机内外参投影的**空间交叉注意力特征采样**。
- 稠密三维网格的显存会随 $WHD$ 增长；正交平面、稀疏体素和多尺度结构提供不同的折中。
- 类别不平衡可以用加权交叉熵或 Focal Loss 处理，但应同时报告各类别和可见性分组指标。

## 练习

1. 如果感知范围是前后各 50 米、左右各 50 米，高度从 -5 米到 3 米，网格分辨率为 0.5 米，计算 $W,H,D$，并估算一个 64 通道 FP32 特征体的显存。
   - **提示**：先计算三个维度的网格数量，然后相乘得到总的体素个数。结合每个浮点数 4 字节（Bytes）计算总内存。
2. 尝试修改 `VoxelQueryGenerator`，不再使用 `nn.Parameter` 初始化全局查询向量，而是用 `torch.meshgrid` 生成网格的三维绝对坐标，并对其应用一层多层感知机（MLP）和正弦位置编码（Sine Positional Encoding）。这会对模型的收敛速度产生什么影响？
   - **提示**：查阅 Transformer 的位置编码机制，思考显式提供空间坐标是否能帮助模型更快学习到 2D-to-3D 的映射几何关系。
3. 仔细思考 TPVFormer 将 $V \in \mathbb{R}^{W \times H \times D}$ 降维到三个二维张量的思路。这种隐式表示在表达一个中空的管状物体（如隧道）时，会出现哪些信息模糊或混叠？
