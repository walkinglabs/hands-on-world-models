# 三维占据网格预测（Occupancy Prediction）的从零开始实现

在前面的章节中，我们深入探讨了如何将多视角的二维图像特征投影到俯视图（Bird's Eye View, BEV）空间，从而实现对周围环境的二维感知。然而，真实的物理世界是三维的。在自动驾驶和机器人导航等复杂场景中，仅仅知道“地面上哪里有物体”是远远不够的。例如，立交桥的悬挑结构、路边伸出的树枝、或者是形状不规则的大型工程车辆，它们在二维 BEV 投影下往往会丢失关键的几何信息，甚至导致危险的碰撞误判。

为了突破这一瓶颈，研究人员开始将目光投向**三维占据网格预测**（3D Occupancy Prediction）。这一任务要求模型不仅要识别出物体的语义类别，还要精确预测它们在三维空间中占据的具体体积。早期的三维场景理解高度依赖于昂贵的激光雷达（LiDAR）点云数据，但点云本身具有稀疏性，且随着距离的增加，数据密度急剧下降。近年来，以纯视觉（Vision-only）驱动的三维占据网络逐渐成为学术界的主流 [Huang et al., 2023] [Wei et al., 2023] [Wang et al., 2023]。这类方法试图直接从多张二维相机图像中，恢复出致密的三维体素（Voxel）表达。

在本节中，我们将从最基础的几何投影原理出发，一步步推导并实现一个极简但完整的三维占据网格预测网络。

## 三维体素空间的几何定义

要预测三维空间中的占据状态，我们首先需要对连续的物理世界进行离散化。最自然的方式是使用**体素**（Voxel），即三维空间中的像素。

假设我们需要感知的物理空间范围为 $X \times Y \times Z$ 米。为了让计算机能够处理，我们沿着长、宽、高三个维度，按照一定的分辨率 $\Delta x, \Delta y, \Delta z$ 将其切割成均匀的小方块。

令空间范围的边界为 $[x_{\min}, x_{\max}]$，$[y_{\min}, y_{\max}]$，$[z_{\min}, z_{\max}]$。我们可以计算出每个维度上体素的数量：

$$W = \frac{x_{\max} - x_{\min}}{\Delta x}, \quad H = \frac{y_{\max} - y_{\min}}{\Delta y}, \quad D = \frac{z_{\max} - z_{\min}}{\Delta z}$$

这样，我们将原本连续的三维坐标 $(x, y, z)$ 映射到了一个离散的三维张量网格索引 $(u, v, w)$ 中。对于任意一个体素网格中的索引 $(u, v, w)$，它所代表的三维物理空间中心坐标可以通过以下线性变换得出：

$$x_c = x_{\min} + (u + 0.5) \cdot \Delta x$$
$$y_c = y_{\min} + (v + 0.5) \cdot \Delta y$$
$$z_c = z_{\min} + (w + 0.5) \cdot \Delta z$$

**(**我们首先定义三维体素网格的基本参数，并生成其在物理空间中的三维坐标网格。**)**

```{.python .input}
#@tab pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_3d_grid(x_bound, y_bound, z_bound):
    """
    生成三维体素网格坐标
    边界格式为: [min, max, resolution]
    """
    # 计算各个维度上的体素数量
    nx = int((x_bound[1] - x_bound[0]) / x_bound[2])
    ny = int((y_bound[1] - y_bound[0]) / y_bound[2])
    nz = int((z_bound[1] - z_bound[0]) / z_bound[2])
    
    # 生成一维坐标序列，加上 0.5 取得体素中心
    x_coords = torch.arange(nx, dtype=torch.float32) * x_bound[2] + x_bound[0] + x_bound[2] / 2
    y_coords = torch.arange(ny, dtype=torch.float32) * y_bound[2] + y_bound[0] + y_bound[2] / 2
    z_coords = torch.arange(nz, dtype=torch.float32) * z_bound[2] + z_bound[0] + z_bound[2] / 2
    
    # 生成三维网格，形状为 (nx, ny, nz)
    grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    
    # 将坐标拼接，形状变为 (nx, ny, nz, 3)
    grid_3d = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    return grid_3d

# 定义一个 10m x 10m x 4m 的极小空间，分辨率为 1m，仅作演示
x_bound = [-5.0, 5.0, 1.0]
y_bound = [-5.0, 5.0, 1.0]
z_bound = [-2.0, 2.0, 1.0]

voxel_grid = generate_3d_grid(x_bound, y_bound, z_bound)
print("体素网格形状:", voxel_grid.shape) # 预期: (10, 10, 4, 3)
```

## 二维到三维的特征投影（2D-to-3D Lifting）

我们已经定义了三维体素网格。现在的核心难点在于：如何将二维图像提取到的特征，填充到这个三维的网格中？

在二维图像中，一个像素点的值受到一整条光线（Ray）上所有物理空间点的影响。由于缺失深度信息，我们无法直接确定图像上的某个像素究竟对应三维空间中的哪一个确切的点。一种经典且直观的策略是**基于几何投影的查询机制**（Geometry-guided Query），类似于 [Li et al., 2022a] 在 BEVFormer 中提出的空间交叉注意力机制（Spatial Cross-Attention）向三维空间的自然扩展。

其核心思想非常朴素：对于三维网格中的每一个体素，我们主动去“询问”它应该包含什么特征。

### 相机投影原理与几何变换

回顾高中的几何光学知识，相机的成像过程本质上是一个透视投影。假设我们在自车坐标系下有一个三维点 $P_{\text{ego}} = (X, Y, Z)$，我们首先需要将其转换到相机坐标系下 $P_{\text{cam}} = (X_c, Y_c, Z_c)$。这一步通过外参矩阵（Extrinsics）完成。

由于平移和旋转操作无法用简单的三维线性矩阵直接叠加表示，我们引入齐次坐标（Homogeneous Coordinates），将三维点扩充为四维向量 $\tilde{P}_{\text{ego}} = (X, Y, Z, 1)^T$。外参矩阵 $E \in \mathbb{R}^{4 \times 4}$ 描述了相机相对于自车的旋转矩阵 $R \in \mathbb{R}^{3 \times 3}$ 和平移向量 $\mathbf{t} \in \mathbb{R}^{3}$：

$$
\begin{bmatrix}
X_c \\ Y_c \\ Z_c \\ 1
\end{bmatrix}
= E \tilde{P}_{\text{ego}} =
\begin{bmatrix}
R & \mathbf{t} \\
\mathbf{0}^T & 1
\end{bmatrix}
\begin{bmatrix}
X \\ Y \\ Z \\ 1
\end{bmatrix}
$$

接着，我们将相机坐标系下的三维点投影到二维像素平面上。这一步通过相机内参矩阵（Intrinsics）$K \in \mathbb{R}^{3 \times 3}$ 完成。内参矩阵包含了焦距 $f_x, f_y$ 和光心偏移 $c_x, c_y$。投影公式为：

$$
Z_c \begin{bmatrix}
u \\ v \\ 1
\end{bmatrix}
= K \begin{bmatrix}
X_c \\ Y_c \\ Z_c
\end{bmatrix} =
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
X_c \\ Y_c \\ Z_c
\end{bmatrix}
$$

其中 $(u, v)$ 就是该三维点在图像上的像素坐标。需要注意的是，由于除以了深度 $Z_c$，这是一个非线性的归一化过程。

### 基于网格采样的特征聚合

当我们知道了一个体素在不同相机图像上的像素位置 $(u, v)$ 后，我们就可以直接使用双线性插值（Bilinear Interpolation）从对应的图像特征图上采样特征向量。由于同一体素可能被多个相机观察到，我们需要将来自不同视图的特征进行融合（例如简单的求均值或拼接后经过感知机处理）。

**(**我们现在实现这个从三维体素向二维图像反投影并采样特征的模块。**)**

```{.python .input}
#@tab pytorch
class VolumeFeatureLifting(nn.Module):
    def __init__(self, x_bound, y_bound, z_bound):
        super().__init__()
        # 生成体素中心的三维坐标，注册为不可训练的 buffer
        grid_3d = generate_3d_grid(x_bound, y_bound, z_bound)
        # 将网格展平为 (N, 3)，方便批量处理，N = nx * ny * nz
        self.register_buffer('grid_3d', grid_3d.reshape(-1, 3))
        
        self.nx = int((x_bound[1] - x_bound[0]) / x_bound[2])
        self.ny = int((y_bound[1] - y_bound[0]) / y_bound[2])
        self.nz = int((z_bound[1] - z_bound[0]) / z_bound[2])

    def forward(self, img_features, extrinsics, intrinsics):
        """
        img_features: (B, N_cam, C, H, W) 图像特征序列
        extrinsics: (B, N_cam, 4, 4) 从自车到相机的外参变换矩阵
        intrinsics: (B, N_cam, 3, 3) 相机内参矩阵
        """
        B, N_cam, C, H, W = img_features.shape
        N_voxels = self.grid_3d.shape[0]
        
        # 将三维网格扩展至具有 Batch 维度: (B, N_voxels, 3)
        pts_3d = self.grid_3d.unsqueeze(0).expand(B, -1, -1)
        
        # 补充齐次坐标位: (B, N_voxels, 4)
        ones = torch.ones((B, N_voxels, 1), device=pts_3d.device)
        pts_3d_homo = torch.cat([pts_3d, ones], dim=-1)
        
        # 准备输出容器，初始化为全零: (B, C, Nx, Ny, Nz)
        volume_features = torch.zeros(B, C, self.nx, self.ny, self.nz, device=img_features.device)
        # 记录每个体素被多少个相机看到，用于求均值
        hit_counts = torch.zeros(B, 1, self.nx, self.ny, self.nz, device=img_features.device)
        
        # 遍历每个相机视角
        for cam_idx in range(N_cam):
            extrinsic = extrinsics[:, cam_idx, :, :] # (B, 4, 4)
            intrinsic = intrinsics[:, cam_idx, :, :] # (B, 3, 3)
            feat = img_features[:, cam_idx, :, :, :] # (B, C, H, W)
            
            # 1. 自车坐标系转换至相机坐标系
            # pts_3d_homo: (B, N_voxels, 4) -> (B, 4, N_voxels)
            pts_cam_homo = torch.bmm(extrinsic, pts_3d_homo.transpose(1, 2))
            # 提取前三个坐标 (X_c, Y_c, Z_c)
            pts_cam = pts_cam_homo[:, :3, :] # (B, 3, N_voxels)
            
            # 过滤掉相机背后的点 (深度 Z_c < 0)
            depth = pts_cam[:, 2:3, :] # (B, 1, N_voxels)
            valid_depth_mask = depth > 1e-5
            
            # 2. 相机坐标系投影至图像像素坐标平面
            # intrinsic: (B, 3, 3), pts_cam: (B, 3, N_voxels)
            uv_homo = torch.bmm(intrinsic, pts_cam) # (B, 3, N_voxels)
            # 归一化深度
            uv = uv_homo[:, :2, :] / (depth + 1e-5) # (B, 2, N_voxels)
            u, v = uv[:, 0, :], uv[:, 1, :]
            
            # 检查像素坐标是否在图像视野范围内
            valid_u = (u >= 0) & (u < W)
            valid_v = (v >= 0) & (v < H)
            valid_mask = valid_depth_mask.squeeze(1) & valid_u & valid_v # (B, N_voxels)
            
            # 为了使用 F.grid_sample，我们需要将 (u, v) 归一化到 [-1, 1] 区间
            # grid_sample 接受的坐标形状必须为 (B, H_out, W_out, 2)
            u_norm = (u / (W - 1)) * 2.0 - 1.0
            v_norm = (v / (H - 1)) * 2.0 - 1.0
            grid = torch.stack([u_norm, v_norm], dim=-1) # (B, N_voxels, 2)
            grid = grid.unsqueeze(1) # (B, 1, N_voxels, 2)，将 N_voxels 视为 W_out 维度
            
            # 3. 双线性插值采样特征
            # sampled_feat: (B, C, 1, N_voxels)
            sampled_feat = F.grid_sample(feat, grid, align_corners=True, padding_mode='zeros')
            sampled_feat = sampled_feat.squeeze(2) # (B, C, N_voxels)
            
            # 应用掩码，只保留有效的采样特征
            valid_mask_ext = valid_mask.unsqueeze(1).float() # (B, 1, N_voxels)
            sampled_feat = sampled_feat * valid_mask_ext
            
            # 将展平的特征重新变回三维体素的形状
            sampled_feat_3d = sampled_feat.reshape(B, C, self.nx, self.ny, self.nz)
            valid_mask_3d = valid_mask_ext.reshape(B, 1, self.nx, self.ny, self.nz)
            
            # 累加多视角的特征
            volume_features += sampled_feat_3d
            hit_counts += valid_mask_3d
            
        # 求特征均值融合
        hit_counts_safe = torch.clamp(hit_counts, min=1.0)
        volume_features = volume_features / hit_counts_safe
        
        return volume_features
```

> [!NOTE]
> 在真实的复杂模型中，上述特征融合过程往往还会结合变形注意力机制（Deformable Attention），让体素不仅采样投影中心点的特征，还能自适应地学习周边区域的关键特征，从而克服由深度估计不准带来的投影偏差。

## 三维特征聚合与语义占据预测网络

在获取了初步的、填充好的三维特征空间后，我们得到了一个张量，其形状为 `(B, C, X, Y, Z)`。这是一个不折不扣的四维时空张量（若将批次 $B$ 忽略，仅看特征通道和三个空间维度）。

这与我们在图像处理中遇到的二维特征图 `(B, C, H, W)` 极为相似。因此，自然的做法是采用三维卷积（3D Convolution）来进一步对空间信息进行上下文聚合。正如二维卷积能够捕捉图像的边缘和纹理，三维卷积能够捕捉三维空间中的几何表面特征和物体形状。

然而，三维卷积的计算代价和显存占用是极其昂贵的。其参数量和计算量随着体素分辨率的增加呈立方级增长。为了在保证性能的同时控制计算成本，我们通常设计一个轻量级的三维编码器-解码器结构，或者使用具有残差连接的简单 3D 卷积堆叠。

最后，在这个聚合后的三维特征张量上，我们使用一个 $1 \times 1 \times 1$ 的 3D 卷积层作为输出分类头，将特征通道维度压缩为语义类别的数量 $N_{classes}$。这样，网络为三维网格中的每一个体素输出一个概率分布，指示该空间是被空气占据（空闲），还是被某类特定物体占据。

**(**我们将三维特征提取与预测头整合成一个完整的纯视觉占据网格预测网络。**)**

```{.python .input}
#@tab pytorch
class OccupancyPredictionNetwork(nn.Module):
    def __init__(self, in_channels, num_classes, x_bound, y_bound, z_bound):
        super().__init__()
        # 1. 2D到3D提升模块
        self.lifting_module = VolumeFeatureLifting(x_bound, y_bound, z_bound)
        
        # 2. 三维特征聚合网络 (简单的 3D 残差块)
        hidden_dim = 64
        self.conv3d_1 = nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(hidden_dim)
        
        self.conv3d_2 = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(hidden_dim)
        
        # 3. 占据分类头 (预测每一个体素的类别，包含 'free' 类)
        self.head = nn.Conv3d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, img_features, extrinsics, intrinsics):
        # 得到形状为 (B, C, Nx, Ny, Nz) 的初始三维特征
        vol_feat = self.lifting_module(img_features, extrinsics, intrinsics)
        
        # 3D 卷积聚合上下文特征
        x = F.relu(self.bn1(self.conv3d_1(vol_feat)))
        x_res = F.relu(self.bn2(self.conv3d_2(x)))
        x = x + x_res # 残差连接
        
        # 输出分类逻辑回归值 (Logits)，形状为 (B, num_classes, Nx, Ny, Nz)
        logits = self.head(x)
        return logits
```

## 损失函数：处理极度不平衡的三维空间

有了网络的输出逻辑回归值 $y \in \mathbb{R}^{C \times W \times H \times D}$ 和真实的占据网格标签 $\hat{y} \in \{0, 1, \dots, C-1\}^{W \times H \times D}$，其中 $C$ 代表类别数，类别 0 通常保留给“空闲空间（Free Space）”。

乍看之下，我们可以简单地将多维张量展平，并在所有体素上计算标准的多类别交叉熵损失（Cross-Entropy Loss）：

$$L_{\text{CE}} = - \frac{1}{W H D} \sum_{u=1}^{W} \sum_{v=1}^{H} \sum_{w=1}^{D} \log p(\hat{y}_{u,v,w} \mid x)$$

其中 $p$ 是经过 Softmax 函数后的模型预测概率分布。

然而，在真实世界中，三维空间呈现出**极端的类别不平衡（Class Imbalance）**。绝大部分空间都是空气（Free），而只有少部分体素被真实的物理表面占据。不仅如此，不同类别的物体体积差异巨大，例如一辆卡车所占据的体素数量可能是一辆自行车的数百倍。

如果在训练时不加干预，模型会倾向于预测所有体素都是“空闲”，从而轻易地获得非常高的准确率，但这在自动驾驶中是致命的。为了解决这一问题，我们通常采用加权交叉熵或引入**焦点损失（Focal Loss）** [Lin et al., 2017]。焦点损失通过降低易分类样本（大部分背景和简单类别）的权重，迫使模型将更多的注意力集中在困难且稀有的占据前景上。

> [!TIP]
> **深层几何直觉：**
> 预测三维空间的占据属性，本质上等同于在一片汪洋大海中寻找孤岛。为了避免网络彻底被“水”（背景）淹没，我们可以采用一种策略：**仅针对物体表面及附近的体素计算梯度**，或者给予“空闲”类别极低的损失权重。这就是为何诸如 OHEM（Online Hard Example Mining）或 Focal Loss 在三维场景感知任务中不可或缺的原因。

**(**我们构建一个带有类别权重的 3D 占据网格交叉熵损失函数。**)**

```{.python .input}
#@tab pytorch
def occupancy_loss(logits, targets, class_weights=None):
    """
    计算三维占据网格的损失
    logits: (B, num_classes, Nx, Ny, Nz)
    targets: (B, Nx, Ny, Nz) 长整型，值为 [0, num_classes-1]
    class_weights: (num_classes,) 用于应对类别不平衡的张量
    """
    # CrossEntropyLoss 原生支持 N 维输入
    # 要求预测值的 C 位于维度 1，target 不包含通道维度，这恰好满足我们的设计
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    loss = criterion(logits, targets)
    return loss

# 演示损失计算
B, num_classes, Nx, Ny, Nz = 2, 12, 10, 10, 4
# 随机生成网络输出
mock_logits = torch.randn(B, num_classes, Nx, Ny, Nz)
# 随机生成真实标签 (在大多数场景下，0 代表 Free，应当占绝大比例)
mock_targets = torch.randint(0, num_classes, (B, Nx, Ny, Nz))
# 大量填充 0 模拟真实稀疏性
mock_targets[mock_targets > 3] = 0 

# 为类别赋予权重，假设 0 是 free_space，权重最低；其余障碍物类别权重较高
weights = torch.ones(num_classes)
weights[0] = 0.1
weights[1:] = 2.0

loss = occupancy_loss(mock_logits, mock_targets, class_weights=weights)
print(f"训练步骤的体素分类损失: {loss.item():.4f}")
```

## 小结

在本节中，我们从物理和几何的源头探讨了如何赋予深度学习模型重构三维空间的能力：
1. 我们建立了一个**离散化的三维体素网格**来映射真实世界的连续空间。
2. 通过深入推导相机外参和内参的作用，我们实现了**基于几何反向投影的三维特征采样**，建立了二维图像与三维空间之间的桥梁。
3. 利用三维卷积聚合空间上下文，我们设计了一个端到端的三维特征感知与预测网络。
4. 我们剖析了空间极度稀疏性和类别不平衡所带来的挑战，并引入了加权损失来引导模型的优化方向。

这套基础架构不仅是当代纯视觉三维占据预测的核心基石，也为理解复杂的三维场景生成和重建技术铺平了道路。
