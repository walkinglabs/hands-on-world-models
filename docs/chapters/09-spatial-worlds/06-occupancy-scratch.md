# 9.6 三维占据网格预测（Occupancy Prediction）的从零开始实现

在进入三维占据预测之前，我们先看 BEV 表示。BEVFormer 用网格状 BEV 查询与多相机图像交互，并结合历史 BEV 特征，支持三维目标检测和地图分割 [[Li et al., 2022]](https://arxiv.org/abs/2203.17270)。原论文没有评测轨迹规划，因此这里不把“简化下游规划”写成由该引用直接证明的实验结论。

二维 BEV 特征通常沿高度聚合信息，而三维占据表示把车辆周围空间离散为体素，并预测占用或语义类别。SurroundOcc 研究多相机图像到稠密三维占用的预测 [[Wei et al., 2023]](https://arxiv.org/abs/2303.09551)，OpenOccupancy 提供占用感知基准及多模态基线 [[Xiaofeng Wang et al., 2023b]](https://arxiv.org/abs/2303.03991)，TPVFormer 则用三视图表示进行三维语义占用预测 [[Huang et al., 2023]](https://arxiv.org/abs/2302.07817)。三篇引用分别对应方法或基准贡献，而不是共同证明一个笼统的“奠定基础”判断。

在本节中，我们将从最基础的几何投影与概率论出发，逐步推导并从零开始实现一个标准的三维占据网格预测模型。同时，我们也会深入剖析这些流行开源项目在真实自动驾驶系统中的落地工程经验。

## 空间几何离散化与占据概率分布

为了让计算机能够处理连续的三维物理空间，我们首先需要对其进行离散化。这与我们在高中物理中计算物体体积时，将不规则物体切分为微小正方体的微元法思想是完全一致的。

### 三维体素网格的定义

假设我们关注车辆周围的一个长方体物理空间。我们定义该空间在世界坐标系下的范围为 $[X_{\min}, X_{\max}] \times [Y_{\min}, Y_{\max}] \times [Z_{\min}, Z_{\max}]$。如果我们在三个维度上的空间分辨率（即每个微小正方体的边长）分别为 $r_x, r_y, r_z$，那么该空间可以被划分为一个三维网格，其网格的维度 $(W, H, D)$ 可以表示为：

$$W = \frac{X_{\max} - X_{\min}}{r_x}, \quad H = \frac{Y_{\max} - Y_{\min}}{r_y}, \quad D = \frac{Z_{\max} - Z_{\min}}{r_z}$$

在这个网格系统中，每一个离散的三维坐标索引 $(i, j, k)$（其中 $0 \le i < W, 0 \le j < H, 0 \le k < D$）都唯一对应物理空间中的一个体素微元。

### 占据状态的概率建模

对于网格中的任意体素 $v_{i,j,k}$，它的物理状态最基础的描述是“是否被占据”（Occupied or Free）。这是一个典型的伯努利试验（Bernoulli Trial）。我们设随机变量 $O_{i,j,k} \in \{0, 1\}$ 表示该状态：

$$P(O_{i,j,k} = 1) = p_{i,j,k}$$

当我们进一步考虑该空间的语义类别（例如：车辆、行人、道路、建筑物等）时，状态空间将扩展为多项分布（Multinomial Distribution）。假设共有 $C$ 个语义类别，并额外增加一个表示“游离态（Free）”的类别，总类别数为 $K = C + 1$。则我们希望模型预测的是条件概率分布：

$$\hat{y}_{i,j,k}^{(c)} = P(S_{i,j,k} = c \mid \mathbf{I})$$

其中 $\mathbf{I}$ 表示输入的多视角环视图像序列，$\hat{y}_{i,j,k}^{(c)}$ 表示体素属于第 $c$ 类的概率。

## 2D 到 3D 空间的特征提升（2D-to-3D Lift）

占据网格预测的核心挑战在于：输入数据是二维的透视图像，而输出目标是三维的空间网格。我们需要建立一种严谨的数学映射，将二维图像特征“拉升”（Lift）到三维空间中。

> 唯一的例外：我们可以借用射影几何中的光线追踪（Ray Casting）思想来理解这个过程。想象在黑夜中，多个相机的像素就像是一束束向外发射的手电筒光束。每一束光在穿过三维空间时，会在沿途的每一个体素上留下一定的“照亮概率”（深度分布）。当所有相机发出的光束在三维空间中交汇重叠时，物理实体所在的位置就会被多束概率高光所点亮，从而在黑暗的三维网格中“显影”出物体的真实轮廓。

### 深度分布估计

经典的 LSS (Lift, Splat, Shoot) [[Philion & Fidler, 2020]](https://arxiv.org/abs/2008.05711) 方法通过显式地预测像素的深度分布来解决这一问题。对于图像上提取的某一个特征像素点及其特征向量 $\mathbf{f} \in \mathbb{R}^F$，我们不预测一个确定性的单一深度值，而是预测它在预定义的离散深度区间 $[d_1, d_2, \ldots, d_D]$ 上的概率分布 $\mathbf{p} \in \mathbb{R}^D$。

通过特征向量 $\mathbf{f}$ 与概率分布 $\mathbf{p}$ 的外积，我们得到该像素对应的视锥体（Frustum）特征 $\mathbf{F} \in \mathbb{R}^{D \times F}$：

$$\mathbf{F} = \mathbf{p} \otimes \mathbf{f} \implies F_{d, f} = p_d \cdot f_f$$

这意味着，二维像素的特征被沿着相机光心出发的射线，按照深度概率权重播撒到了三维空间中。

### 初始化三维体素查询特征

在真实工程中，如 SurroundOcc 项目，通常会显式地在三维空间中初始化一套可学习的体素特征参数（Voxel Queries） $\mathbf{Q} \in \mathbb{R}^{W \times H \times D \times C_{emb}}$。

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
        # 初始化权重
        nn.init.normal_(self.voxel_queries, mean=0., std=1.)

    def forward(self, batch_size):
        # 将查询参数扩展至当前批次大小
        return self.voxel_queries.repeat(batch_size, 1, 1, 1, 1)

# 测试初始化
grid_size = [100, 100, 8] # X=100, Y=100, Z=8 的体素网格
embed_dims = 64
query_gen = VoxelQueryGenerator(grid_size, embed_dims)
queries = query_gen(batch_size=2)
print("Voxel queries shape:", queries.shape) # 预期: [2, 64, 8, 100, 100]
```

## 空间交叉注意力与特征聚合

拿到三维体素查询 $\mathbf{Q}$ 后，我们需要用二维图像特征来更新它们。这里我们引入 BEVFormer [[Li et al., 2022]](https://arxiv.org/abs/2203.17270) 中首创的空间交叉注意力机制（Spatial Cross-Attention, SCA），并将其扩展到三维。

对于任意一个三维体素中心点 $(x, y, z)$，我们可以通过相机的内外参矩阵 $\mathbf{P} \in \mathbb{R}^{3 \times 4}$ 将其投影到第 $i$ 个相机的图像平面像素坐标 $(u_i, v_i)$ 上：

$$d [u_i, v_i, 1]^T = \mathbf{P} \cdot [x, y, z, 1]^T$$

其中 $d$ 是该三维点在相机坐标系下的深度值。

如果投影点 $(u_i, v_i)$ 落在了图像边界内，这就意味着该相机能够“看到”这个体素。在此基础上，我们可以使用可变形注意力机制（Deformable Attention）在图像特征图上采样局部特征，来更新体素查询：

$$\mathbf{Q}_{x,y,z}' = \mathbf{Q}_{x,y,z} + \sum_{i \in \mathcal{V}_{x,y,z}} \text{DeformAttn}(\mathbf{Q}_{x,y,z}, \mathbf{F}_i, \mathbf{P}(x,y,z))$$

其中 $\mathcal{V}_{x,y,z}$ 表示能观测到该体素的相机集合，$\mathbf{F}_i$ 是第 $i$ 视角的图像特征。

### 开源项目的架构演进

在这一步的工程落地上，不同开源项目采取了截然不同的优化策略：

1. **SurroundOcc**：采用多尺度 3D 卷积网络来逐步上采样和精细化 3D Voxel 特征。为了显存可控，其通常采用相对较粗的初始网格分辨率。
2. **TPVFormer**：敏锐地意识到稠密 $W \times H \times D$ 体素带来的 $O(W \cdot H \cdot D)$ 显存爆炸问题。它极其聪明地将三维空间投影到三个正交的平面（顶视图、侧视图、正视图），即 Tri-Perspective View。通过三个面积为 $W \times H$、$H \times D$、$W \times D$ 的 2D 平面特征来隐式表达 3D 空间，将复杂度降维到了 $O(W \cdot H + H \cdot D + W \cdot D)$，极大地降低了计算开销。
3. **OpenOccupancy**：提出了一个自适应分辨率的框架，结合了密集 3D 卷积和级联的稀疏注意力机制。

(**我们利用简化的几何投影机制来实现 3D 到 2D 的特征采样更新**)：

```python
class SimpleSpatialCrossAttention(nn.Module):
    def __init__(self, embed_dims):
        super().__init__()
        self.embed_dims = embed_dims
        # 为了演示，我们使用简单的线性映射代替复杂的 Deformable Attention
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

    def forward(self, voxel_queries, image_features, proj_mats):
        """
        voxel_queries: [B, C, D, H, W]
        image_features: [B, N_cams, C, H_img, W_img]
        proj_mats: 投影矩阵等几何信息 (此处简化)
        """
        B, C, D, H, W = voxel_queries.shape
        # 将体素展平为序列 [B, D*H*W, C]
        queries_flat = voxel_queries.view(B, C, -1).permute(0, 2, 1)

        # [演示简化逻辑] 假设我们已经通过公式 (9.6.5) 找到了每个体素在图像上的对应特征
        # 在实际工程中，这里会调用 grid_sample 或自定义 CUDA 算子进行高效采样
        # 我们随机生成一个模拟采样后的上下文特征
        sampled_context = torch.randn_like(queries_flat)

        # 简单的线性融合
        updated_queries = queries_flat + self.output_proj(F.relu(self.value_proj(sampled_context)))

        # 恢复 3D 形状
        voxel_queries = updated_queries.permute(0, 2, 1).view(B, C, D, H, W)
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
print("Final logits shape:", final_logits.shape) # [2, 17, 8, 100, 100]
```

### 极度不平衡的三维空间损失

在真实的物理空间中，“空闲（Free）”体素往往占据了 90% 以上的空间。如果直接使用标准的交叉熵损失（Cross Entropy Loss），网络将被大量的空闲体素所主导，导致难以正确预测稀有物体（如行人、自行车）。

因此，工程上强制要求使用加权焦点损失（Weighted Focal Loss）。对于类别 $c$ 的体素标签 $y$ 和预测概率 $\hat{y}$，Focal Loss 定义为：

$$L_{focal} = - \alpha_c (1 - \hat{y})^\gamma y \log(\hat{y})$$

其中 $\alpha_c$ 是针对不同类别的频率所设置的静态权重（频率越低，权重越高），$\gamma$ 是聚焦参数（通常取 2.0），用于降低易分样本（高置信度的 Free 体素）的梯度贡献。

此外，由于相机只能观测到物体面向镜头的表面，物体背面的体素不可见。这就要求模型具备“场景补全（Scene Completion）”的推理能力。在 OpenOccupancy 中，研究者利用几何一致性损失，鼓励模型依据可见的局部几何特征推断出物体完整的 3D 体积。

## 小结

- **三维占据网格预测**旨在将 2D 环视图像直接映射为 3D 体素级的语义分类，彻底解决了 2D Bounding Box 或 2D BEV 带来的空间信息丢失问题。
- 核心操作包含了三维体素查询的初始化，以及基于相机内外参投影的**空间交叉注意力特征采样**。
- 在真实世界的工程落地中，3D 高分辨率网格带来的显存开销是致命瓶颈。TPVFormer 的正交平面降维思想，以及 SurroundOcc 的多尺度层级上采样，都是极为经典的工程优化策略。
- 针对 3D 空间中极端的“空闲-占据”类别不平衡，**Focal Loss** 及其变体是不可或缺的优化手段。

## 练习

1. 在该公式中，如果感知范围是前后各 50 米，左右各 50 米，上下从 -5 米到 3 米。网格分辨率为 0.5 米。请计算 $W, H, D$ 的精确数值，并估算在 FP32 精度下，存储该 64 维隐藏特征图需要多少显存？
   - **提示**：先计算三个维度的网格数量，然后相乘得到总的体素个数。结合每个浮点数 4 字节（Bytes）计算总内存。
2. 尝试修改 `VoxelQueryGenerator`，不再使用 `nn.Parameter` 初始化全局查询向量，而是用 `torch.meshgrid` 生成网格的三维绝对坐标，并对其应用一层多层感知机（MLP）和正弦位置编码（Sine Positional Encoding）。这会对模型的收敛速度产生什么影响？
   - **提示**：查阅 Transformer 的位置编码机制，思考显式提供空间坐标是否能帮助模型更快学习到 2D-to-3D 的映射几何关系。
3. 仔细思考 TPVFormer 将 $V \in \mathbb{R}^{W \times H \times D}$ 降维到三个二维张量的思路。这种隐式表示在表达一个中空的管状物体（如隧道）时，会出现哪些信息模糊或混叠？
