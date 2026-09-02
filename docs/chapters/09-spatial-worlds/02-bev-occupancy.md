# 9.2 鸟瞰图 (BEV) 与三维占据网格 (Occupancy)

在上一节中，我们建立了从三维物理世界向二维相机图像进行透视投影的前向几何模型。然而，在自动驾驶车辆行驶或移动机器人导航时，算法面临的却是一个截然相反、且在数学上极其病态的逆向问题：**如何从车身搭载的多个二维透视相机画面中，逆向重构出以车辆为中心的统一三维全局物理空间？**

如果直接在二维相机图像上做目标检测，不同相机之间视场存在大量盲区与重叠，且二维检测框无法直接获知障碍物的真实三维物理距离；若前方有一辆大货车，它在相机图像中会遮挡住后方的行人和小汽车。

为了克服单视角透视畸变与遮挡难题，现代空间感知体系开创了 **鸟瞰图（Bird's Eye View, BEV）** 与 **三维语义占据网格（3D Occupancy Grids）** 范式——将多相机多视角的透视特征在空间中“提升”并“缝合”为一个俯视的全局公制坐标系地图。

<div align="center">

<img src="/figures/09-spatial-worlds/source/02-bev-occupancy/lss-fig1.png" alt="Lift-Splat-Shoot 把环视相机图像变成统一鸟瞰表示，显示 BEV 如何服务车辆周围的空间推理。" width="86%">

_图 9.2-1：Lift-Splat-Shoot 把环视相机图像变成统一鸟瞰表示，显示 BEV 如何服务车辆周围的空间推理。 出处：[Lift, Splat, Shoot: Encoding Images from Arbitrary Cameras to Polytope Inputs for Planning，Jonah Philion et al.，2020](https://arxiv.org/abs/2008.05711)。_

</div>

---

## 9.2.1 物理与几何基石：透视遮挡与自顶向下的上帝视角

要理解 BEV 的诞生，我们首先需要从人类驾驶员在空间中的宏观心智模型讲起。

### 1. 人类空间认知的“俯视心智地图”
当你在一个车流密集的十字路口准备左转时，尽管你的眼睛只能看到挡风玻璃前方和左右后视镜中片段式的倾斜透视画面，但你的大脑小脑却在内心构建了一张**以上帝视角俯视的二维动态网格地图**：
- “我的车位于原点，左前方 15 米有一辆直行客车，右侧 3 米有一辆电动自行车，左转弯道半径为 20 米”。

这种自顶向下、无透视畸变的俯视平面，正是下游运动规划器（Motion Planner）计算防碰撞安全轨迹的最佳几何空间。

### 2. 单目深度的内在多义性（Ray Ambiguity）
在几何学中，当空间点 $P$ 投影到像素 $(u, v)$ 时，根据投影公式 $P_c = Z_c \mathbf{K}^{-1} [u, v, 1]^\top$，由于单张图像丢失了深度标量 $Z_c$，**单个像素在三维空间中并不对应一个点，而是对应从相机光心出发穿过该像素的一整条无限延展的射线（Ray）**！

如果无法确定物体在射线上的确切深度，该像素上的特征究竟应该放置在距离相机 $2\text{ 米}$ 处，还是 $20\text{ 米}$ 处？这一“射线多义性”是 2D 向 3D 转换的核心数学挑战。

<div align="center">

<img src="/figures/09-spatial-worlds/latex/02-bev-occupancy/pixel-ray-depth-family.png" alt="同一像素经内参逆变换只确定射线方向，不同正深度对应射线上不同三维点" width="86%">

_图 9.2-2：同一像素经内参逆变换只确定射线方向，不同正深度对应射线上不同三维点。本文绘制；TikZ/LaTeX 编译。_

</div>

---

## 9.2.2 核心数学推导一：Lift-Splat-Shoot (LSS) 的外积特征提升

2020 年，Jonah Philion 与 Sanja Fidler 提出了开创性的 **Lift-Splat-Shoot (LSS)** 架构，以极具代数美感的方式解决了射线多义性问题。

<div align="center">

<img src="/figures/09-spatial-worlds/source/02-bev-occupancy/lss-fig4.png" alt="LSS 将图像特征沿离散深度提升到视锥体，再汇聚到鸟瞰网格并输出规划相关预测。" width="86%">

_图 9.2-3：LSS 将图像特征沿离散深度提升到视锥体，再汇聚到鸟瞰网格并输出规划相关预测。 出处：[Lift, Splat, Shoot: Encoding Images from Arbitrary Cameras to Polytope Inputs for Planning，Jonah Philion et al.，2020](https://arxiv.org/abs/2008.05711)。_

</div>

### 1. 离散深度概率分布与外积提升（Lift）
LSS 不去强行预测一个单一确定性的深度数值（因为单目深度预测极易在弱纹理区域出错），而是将连续深度范围 $[d_{\min}, d_{\max}]$ 等距离划分为 $D$ 个离散深度分桶（例如 $D = 60$，每个桶跨度 $1.0\text{ 米}$）。

对于图像特征图上的某个像素 $(u, v)$，网络同时预测两个向量：
1. **语义特征向量**：$\mathbf{c} \in \mathbb{R}^C$（描述该像素“是什么物体”，维度为 $C$）；
2. **深度离散概率分布**：$\mathbf{p} = [p_1, p_2, \dots, p_D]^\top \in \mathbb{R}^D$（经过 Softmax 归一化，$\sum_{d=1}^D p_d = 1$，描述该物体落在第 $d$ 个深度桶的概率）。

LSS 通过**向量外积（Outer Product）**，将这两个向量相乘，生成沿射线延展的 3D 视锥体特征点云（Frustum Features）：

$$\mathbf{F}_{(u, v)} = \mathbf{p} \otimes \mathbf{c} = \mathbf{p} \mathbf{c}^\top \in \mathbb{R}^{D \times C}$$

对于第 $d$ 个深度切片，其特征强度为 $p_d \cdot \mathbf{c}$。若网络极度确信物体在第 10 米处（$p_{10} = 0.9$），则第 10 米处的特征强度极大，其余距离处的特征强度自动衰减为 0！

**手算代入算例**：
设某像素特征向量为 $\mathbf{c} = [2.0, -1.0]^\top$（维度 $C = 2$），离散深度分桶仅设 2 个桶，网络预测的概率分布为 $\mathbf{p} = [0.8, 0.2]^\top$（第一桶概率 $80\%$，第二桶概率 $20\%$）。

计算外积特征矩阵：
$$\mathbf{F} = \mathbf{p} \mathbf{c}^\top = \begin{bmatrix} 0.8 \\ 0.2 \end{bmatrix} \begin{bmatrix} 2.0 & -1.0 \end{bmatrix} = \begin{bmatrix} 0.8 \times 2.0 & 0.8 \times (-1.0) \\ 0.2 \times 2.0 & 0.2 \times (-1.0) \end{bmatrix} = \begin{bmatrix} 1.6 & -0.8 \\ 0.4 & -0.2 \end{bmatrix}$$

这一外积算子极为巧妙：无需任何三维标注，网络在无监督反向传播下就能自主学会沿射线分配深度概率！

### 2. 空间飞溅汇聚（Splat / Voxel Pooling）
根据相机的内参逆矩阵 $\mathbf{K}^{-1}$ 与外参矩阵 $[\mathbf{R} \mid \mathbf{t}]$，视锥体内的每一个三维特征点 $(u, v, d)$ 都可以被刚体变换精确映射到以自车为中心的世界三维物理坐标 $(X, Y, Z)$：

$$P_{\text{world}} = \mathbf{R}^\top \left( d \cdot \mathbf{K}^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} - \mathbf{t} \right)$$

随后，空间被划分为规则的 BEV 网格柱（Pillars，每个柱子平面尺寸例如 $0.5\text{ m} \times 0.5\text{ m}$）。所有落入同一个柱状体内的三维特征点被累加求和（Voxel Pooling），直接拍平生成二维全局 BEV 特征图！

<details>
<summary><b>深入推导：基于前缀和（Cumsum Trick）的极速 GPU 体素池化并行算子数理分析（点击展开查看完整推导）</b></summary>

在处理多相机视锥时，生成的特征点云规模高达数百万人次。直接在 GPU 上执行原子加法（Atomic Add）会面临严重的线程写冲突与内存带宽瓶颈。
LSS 提出了著名的 **前缀和技巧（Cumsum Trick）**：
1. 首先计算每个三维点所属的体素网格 ID，并对 ID 进行并行基数排序（Radix Sort）；
2. 对排序后的特征数组沿通道执行前缀和扫描：$S_k = \sum_{j=1}^k \mathbf{f}_j$；
3. 任意体素网格区间 $[L, R]$ 内部的所有特征和，可由区间端点差分在 $\mathcal{O}(1)$ 常数时间内瞬时计算：
   $$\mathbf{F}_{\text{voxel}} = S_R - S_{L-1}$$
该算法彻底消除了写锁争用，使池化速度提升了两个数量级。
</details>

---

## 9.2.3 核心数学推导二：BEVFormer 的可变形交叉注意力

LSS 采用的是“自底向上（Bottom-Up）”的特征显式提升路径；而在 2022 年，中国科学院与商汤科技联合提出了 **BEVFormer**，开创了“自顶向下（Top-Down）”的查询采样范式。

<div align="center">

<img src="/figures/09-spatial-worlds/source/02-bev-occupancy/bevformer-fig2.png" alt="BEVFormer 用空间交叉注意力从多相机特征更新 BEV 查询，并用时间自注意力融合历史 BEV。" width="86%">

_图 9.2-4：BEVFormer 用空间交叉注意力从多相机特征更新 BEV 查询，并用时间自注意力融合历史 BEV。 出处：[BEVFormer: Learning Bird's-Eye-View Representation with Spatiotemporal Transformers，Zhiqi Li et al.，2022](https://arxiv.org/abs/2203.17270)。_

</div>

<div align="center">

<img src="/figures/09-spatial-worlds/source/02-bev-occupancy/tpvformer-fig3.png" alt="TPVFormer 并列比较体素、BEV 与三正交平面表示，说明保留高度信息时的空间表征取舍。" width="86%">

_图 9.2-5：TPVFormer 并列比较体素、BEV 与三正交平面表示，说明保留高度信息时的空间表征取舍。 出处：[Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction，Yuanhui Huang et al.，2023](https://arxiv.org/abs/2302.07817)。_

</div>

### 1. 空间可变形交叉注意力（Spatial Cross-Attention）
在 BEVFormer 中，系统在地面网格上预设一系列可学习的 **BEV 查询词元（BEV Queries）** $\mathbf{Q} \in \mathbb{R}^{H_{\text{bev}} \times W_{\text{bev}} \times C}$。
对于网格上的某个平面位置 $(x, y)$：
1. 沿高度方向抬起 $N_{\text{pts}}$ 个空间高度锚点：$\{(x, y, z_1), (x, y, z_2), \dots, (x, y, z_{N_{\text{pts}}})\}$；
2. 利用上一节学过的投影矩阵 $\mathbf{P}_i = \mathbf{K}_i [\mathbf{R}_i \mid \mathbf{t}_i]$，将这些三维锚点投影回各个相机的二维像素平面；
3. 利用**可变形注意力（Deformable Attention）**，仅在投影像素周围采样极少数关键特征点进行双线性插值并加权汇聚。

这种自顶向下的查询机制，完全绕开了计算昂贵的稠密视锥体云构建，实现了跨相机视野的自然缝合与远距离物体的精准感知。

<details>
<summary><b>深入推导：空间可变形注意力双线性插值采样与全微分几何反向传播（点击展开查看完整推导）</b></summary>

设投影二维采样点为 $\mathbf{p} = (u, v)$，其在特征图 $\mathbf{X}$ 上的双线性插值响应为：
$$\text{Sample}(\mathbf{X}, \mathbf{p}) = \sum_{i, j} \max(0, 1 - |u - i|) \max(0, 1 - |v - j|) \mathbf{X}[i, j]$$
对于 BEV 查询 $\mathbf{q}$，可变形注意力输出为 $M$ 个注意力头的加权和：
$$\text{DeformAttn}(\mathbf{q}, \mathbf{p}, \mathbf{X}) = \sum_{m=1}^M \mathbf{W}_m \left[ \sum_{k=1}^K A_{m, k} \cdot \text{Sample}(\mathbf{X}, \mathbf{p} + \Delta \mathbf{p}_{m, k}) \right]$$
采样偏移量 $\Delta \mathbf{p}$ 与权重 $A$ 全部由查询向量 $\mathbf{q}$ 经线性层动态生成，对采样坐标 $\mathbf{p}$ 的全导数 $\frac{\partial \text{Sample}}{\partial \mathbf{p}}$ 处处连续可导，保证了端到端几何感知的顺畅梯度反传。
</details>

---

## 9.2.4 纯底层 PyTorch 代码实现：从零构建 LSS 外积特征提升与体素池化

下面我们使用纯底层 PyTorch 算子手写实现 LSS 的核心模块：包括离散深度概率外积提升（Lift）与基于几何映射的 BEV 栅格汇聚引擎。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSSFrustumLifter(nn.Module):
    """
    LSS 视锥特征提升层 (Lift Layer)
    将二维图像特征与离散深度概率分布做外积，生成三维视锥点云特征
    """
    def __init__(self, in_channels: int = 64, out_channels: int = 64, num_depth_bins: int = 30):
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.out_channels = out_channels

        # 深度与特征联合预测头
        self.conv = nn.Conv2d(
            in_channels, num_depth_bins + out_channels, kernel_size=1, bias=True
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: (B, in_channels, H, W) 2D 图像特征
        :return: frustum_features: (B, num_depth_bins, H, W, out_channels), depth_probs: (B, num_depth_bins, H, W)
        """
        B, _, H, W = x.shape
        out = self.conv(x) # (B, D + C, H, W)

        # 1. 拆分深度分布与语义特征
        depth_logits = out[:, :self.num_depth_bins, :, :]
        features = out[:, self.num_depth_bins:, :, :] # (B, C, H, W)

        # 2. 深度方向 Softmax 归一化
        depth_probs = F.softmax(depth_logits, dim=1) # (B, D, H, W)

        # 3. 外积提升: depth_probs (B, D, H, W, 1) * features (B, 1, H, W, C)
        depth_expanded = depth_probs.unsqueeze(-1)                    # (B, D, H, W, 1)
        feat_expanded = features.permute(0, 2, 3, 1).unsqueeze(1)    # (B, 1, H, W, C)

        frustum_features = depth_expanded * feat_expanded             # (B, D, H, W, C)
        return frustum_features, depth_probs

class SimpleBEVPooler:
    """
    简化版 BEV 栅格池化聚合器 (Splat Layer)
    将视锥三维点坐标映射到 BEV 网格并完成柱状体积分
    """
    def __init__(self, bev_size: tuple[int, int] = (50, 50), x_range: tuple = (-25, 25), y_range: tuple = (0, 50)):
        self.bev_h, self.bev_w = bev_size
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range

    def pool(self, frustum_features: torch.Tensor, world_coords: torch.Tensor) -> torch.Tensor:
        """
        :param frustum_features: (B, D, H, W, C)
        :param world_coords: (B, D, H, W, 3) 对应的世界/自车坐标 (X, Y, Z)
        :return: (B, C, bev_h, bev_w) 拍平后的全局 BEV 特征图
        """
        B, D, H, W, C = frustum_features.shape
        flat_feats = frustum_features.reshape(B, -1, C)     # (B, N_pts, C)
        flat_coords = world_coords.reshape(B, -1, 3)        # (B, N_pts, 3)

        bev_map = torch.zeros(B, C, self.bev_h, self.bev_w, device=frustum_features.device)

        # 将连续物理坐标转换为离散 BEV 网格索引
        x_pts = flat_coords[..., 0] # (B, N_pts)
        y_pts = flat_coords[..., 1] # (B, N_pts)

        x_idx = ((x_pts - self.x_min) / (self.x_max - self.x_min) * self.bev_w).long()
        y_idx = ((y_pts - self.y_min) / (self.y_max - self.y_min) * self.bev_h).long()

        # 边界有效性过滤
        valid = (x_idx >= 0) & (x_idx < self.bev_w) & (y_idx >= 0) & (y_idx < self.bev_h)

        for b in range(B):
            v_mask = valid[b]
            b_x = x_idx[b, v_mask]
            b_y = y_idx[b, v_mask]
            b_feats = flat_feats[b, v_mask] # (N_valid, C)

            # 使用 index_put_ 进行体素累加
            bev_map[b].index_put_((slice(None), b_y, b_x), b_feats.T, accumulate=True)

        return bev_map

# ===================================================================
# 单元测试与外积特征守恒校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 2
    in_c = 32
    out_c = 16
    img_h, img_w = 16, 16
    num_depths = 20

    # 1. 实例化提升网络与池化器
    lifter = LSSFrustumLifter(in_channels=in_c, out_channels=out_c, num_depth_bins=num_depths)
    lifter.eval()
    pooler = SimpleBEVPooler(bev_size=(32, 32), x_range=(-16, 16), y_range=(0, 32))

    dummy_2d_feat = torch.randn(batch_size, in_c, img_h, img_w)

    with torch.no_grad():
        frustum_feats, d_probs = lifter(dummy_2d_feat)

    print(f"[LSS Test] 输入 2D 特征图形状: {dummy_2d_feat.shape}")
    print(f"[LSS Test] 提升后 3D 视锥特征形状: {frustum_feats.shape}")
    print(f"[LSS Test] 深度概率单像素求和: {d_probs[0, :, 5, 5].sum().item():.4f}")

    assert frustum_feats.shape == (batch_size, num_depths, img_h, img_w, out_c), "视锥特征张量形状不符！"
    assert abs(d_probs[0, :, 5, 5].sum().item() - 1.0) < 1e-4, "深度分布 Softmax 未归一化！"

    # 2. 模拟生成空间世界坐标并测试 BEV 池化
    dummy_world_xyz = torch.randn(batch_size, num_depths, img_h, img_w, 3) * 10.0
    dummy_world_xyz[..., 1] = dummy_world_xyz[..., 1].abs() # 确保 y 位于前方

    bev_out = pooler.pool(frustum_feats, dummy_world_xyz)
    print(f"[BEV Test] 生成全局 BEV 特征图形状: {bev_out.shape}")

    assert bev_out.shape == (batch_size, out_c, 32, 32), "BEV 网格输出维度不符！"
    print("✓ LSS 视锥外积提升与 BEV 栅格池化聚合单测全部通过！")
```

---

## 9.2.5 本节小结

回顾本节内容，我们建立了多视角透视向全局统一空间投影的严密理论脉络：
1. **射线多义性破局**：单目像素沿视线具有深度不确定性，LSS 通过预测离散深度概率分布化解了多义性；
2. **外积特征提升（Lift）**：利用语义特征与深度分布的外积张量 $\mathbf{p} \otimes \mathbf{c}$，自适应地将 2D 信息铺设到 3D 视锥空间中；
3. **空间池化与 BEV 转换**：通过外参刚体几何逆变换与柱状体体积积分，多相机视场被无缝缝合为一个上帝视角的全局公制特征地图。
