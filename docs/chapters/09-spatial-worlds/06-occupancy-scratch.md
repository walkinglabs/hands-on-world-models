# 9.6 从零实现三维占据网格预测 (Occupancy from Scratch)

在自动驾驶感知技术的演进历程中，早期的算法几乎全部聚焦于**三维三维边界框（3D Bounding Boxes）**的回归。算法假定街道上的所有交通参与者都可以被简化为一个长方体盒子（如轿车、货车、行人、骑行者）。

然而，当自动驾驶车辆驶入复杂的非结构化道路时，这一简单的长方体假设却屡屡引发严重的感知漏检：
- 道路施工区域随意堆放的不规则沙石堆、散落一地的施工警示水马；
- 翻倒在高速公路中央的异形货车与散落的货物；
- 倾斜下垂伸入车道内部的树木枝叶。

这些“无法用标准长方体框定”的通用任意障碍物（General Obstacles），构成了自动驾驶感知中最致命的盲区。

为了实现对物理世界几何结构的终极刻画，**三维语义占据网格预测（3D Semantic Occupancy Prediction）**应运而生。它将空间切割为数百万个细小的微观体素立方体，直接判定每一个体素的空间占有状态。

本节我们将从零推导 2D 到 3D 的特征投影机制（FLoSP）、样本极度不平衡下的 Focal 损失，并使用纯底层 PyTorch 从零手写一个完整的三维占据预测网络。

<div align="center">

<img src="/figures/09-spatial-worlds/source/06-occupancy-scratch/monoscene-fig1.png" alt="MonoScene 从单张道路图像恢复可见与遮挡区域的稠密三维语义占据，展示占据预测的直接输出。" width="86%">

_图 9.6-1：MonoScene 从单张道路图像恢复可见与遮挡区域的稠密三维语义占据，展示占据预测的直接输出。 出处：[MonoScene: Monocular 3D Semantic Scene Completion，Anh-Quan Cao et al.，2022](https://arxiv.org/abs/2112.00726)。_

</div>

---

## 9.6.1 物理与几何基石：从离散边界框到连续体素场

要理解三维占据网格的优越性，我们首先需要从离散几何学对三维空间的表征范式讲起。

### 1. 乐高积木式空间离散化
想象我们用无数个极小的正方体乐高积木去拼搭整个街道：
- 设自车周围的三维物理空间范围为 $X \in [-40\text{ m}, 40\text{ m}], Y \in [-40\text{ m}, 40\text{ m}], Z \in [-1\text{ m}, 5.4\text{ m}]$；
- 我们将每个体素（Voxel）的物理分辨率设为 $\Delta x = \Delta y = \Delta z = 0.4\text{ 米}$；
- 整个物理空间被离散化为一个分辨率为 $200 \times 200 \times 16 = 640,000$ 个体素的三维立方体网格 $\mathcal{V}$。

对于每一个体素坐标 $(i, j, k)$，网络的任务是输出一个类别概率分布：该立方体是空旷的空气（Free）、水泥路面、坚硬建筑、还是行驶中的车辆。

### 2. 空间空旷性的“稀疏性诅咒”
在真实的道路环境中，整个三维立体空间有超过 **$95\%$ 以上的体积都充满了空旷的空气**，真正被实体物体占据的体素不足 $5\%$。
这种极度的空间稀疏性带来了一个核心挑战：在计算时如何避免对无用的空气体素做无效计算，以及在训练时如何防止海量的空气样本压垮微弱的障碍物梯度。

<div align="center">

<img src="/figures/09-spatial-worlds/source/06-occupancy-scratch/monoscene-fig3.png" alt="MonoScene 的 FLoSP 把二维多尺度特征沿相机视线投影到三维体素查询，处理 2D 到 3D 的特征提升。" width="86%">

_图 9.6-2：MonoScene 的 FLoSP 把二维多尺度特征沿相机视线投影到三维体素查询，处理 2D 到 3D 的特征提升。 出处：[MonoScene: Monocular 3D Semantic Scene Completion，Anh-Quan Cao et al.，2022](https://arxiv.org/abs/2112.00726)。_

</div>

---

## 9.6.2 核心数学推导一：特征沿光线采样（FLoSP）与双线性插值

如何从二维图像特征图生成初始的三维体素特征？MonoScene 提出了**视线特征采样（Feature-Line of Sight Projection, FLoSP）**算法。

<div align="center">

<img src="/figures/09-spatial-worlds/latex/06-occupancy-scratch/depth-feature-outer-product.png" alt="深度概率向量与像素特征向量做外积，每个深度概率缩放整条通道特征" width="86%">

_图 9.6-3：深度概率向量与像素特征向量做外积，每个深度概率缩放整条通道特征。_

</div>

### 1. 三维体素中心反投影
对于三维体素网格中的每一个体素中心点 $P_{\text{voxel}} = (X_w, Y_w, Z_w)^\top$：
利用相机的内参矩阵 $\mathbf{K}$ 与外参刚体变换矩阵 $[\mathbf{R} \mid \mathbf{t}]$，计算该体素中心在二维相机图像上的投影像素位置 $(u, v)$：

$$Z_c \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} [\mathbf{R} \mid \mathbf{t}] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

$$u = \frac{X_{\text{img}}}{Z_c}, \quad v = \frac{Y_{\text{img}}}{Z_c}$$

### 2. 双线性插值采样（Bilinear Interpolation）
由于投影出的 $(u, v)$ 通常是浮点数坐标，我们利用该点周围的 4 个整型像素网格点，执行双线性插值提取二维特征向量 $\mathbf{f}_{2D} \in \mathbb{R}^C$：

$$\mathbf{V}(i, j, k) = \sum_{x \in \{\lfloor u \rfloor, \lceil u \rceil\}} \sum_{y \in \{\lfloor v \rfloor, \lceil v \rceil\}} (1 - |u - x|) (1 - |v - y|) \mathbf{F}_{2D}[y, x]$$

**手算代入算例**：
设某体素投影得到的像素坐标为 $(u, v) = (10.2, 20.6)$。
周围 4 个最近邻像素的特征标量分别为：
- 左上 $(10, 20)$ 特征值为 $1.0$；权重为 $(1 - 0.2) \times (1 - 0.6) = 0.8 \times 0.4 = 0.32$；
- 右上 $(11, 20)$ 特征值为 $2.0$；权重为 $(0.2) \times (1 - 0.6) = 0.2 \times 0.4 = 0.08$；
- 左下 $(10, 21)$ 特征值为 $3.0$；权重为 $(1 - 0.2) \times (0.6) = 0.8 \times 0.6 = 0.48$；
- 右下 $(11, 21)$ 特征值为 $4.0$；权重为 $(0.2) \times (0.6) = 0.2 \times 0.6 = 0.12$。

我们计算插值后的体素特征值：
$$V = 0.32 \times 1.0 + 0.08 \times 2.0 + 0.48 \times 3.0 + 0.12 \times 4.0 = 0.32 + 0.16 + 1.44 + 0.48 = 2.40$$

四项权重之和恰好为 $0.32 + 0.08 + 0.48 + 0.12 = 1.0$！初等代数的代入过程极其严谨，为每个 3D 体素赋予了高保真度的视觉色彩与纹理特征。

<details>
<summary><b>深入推导：三维体素多尺度反投影（FLoSP）插值核函数的全微分求导推导（点击展开查看完整推导）</b></summary>

设插值核函数为 $K(u, x) = \max(0, 1 - |u - x|)$。
体素特征对 2D 卷积特征图 $\mathbf{F}[y, x]$ 的偏导数为双线性权重乘积：
$$\frac{\partial \mathbf{V}(i, j, k)}{\partial \mathbf{F}[y, x]} = K(u, x) K(v, y)$$
当体素在三维物理空间移动微小位移 $d\mathbf{X}$ 时，投影坐标产生微小位移 $d\mathbf{p} = \mathbf{J}_{\text{proj}} d\mathbf{X}$。
由链式法则可得体素特征对三维物理坐标的微分梯度为：
$$\nabla_{\mathbf{X}} \mathbf{V} = \mathbf{J}_{\text{proj}}^\top \begin{bmatrix} \sum_{x, y} \frac{\partial K(u, x)}{\partial u} K(v, y) \mathbf{F}[y, x] \\ \sum_{x, y} K(u, x) \frac{\partial K(v, y)}{\partial v} \mathbf{F}[y, x] \end{bmatrix}$$
该连续可微性保证了 3D 几何特征可以通过反向传播直接优化底层的 2D 视觉主干网络。
</details>

---

## 9.6.3 核心数学推导二：极度类别不平衡下的 Focal 损失与手算对比

由于空旷空气体素占据了 $95\%$ 以上的体积，若直接使用标准交叉熵损失，网络会迅速学会“将所有体素全部预测为空气”的躺平作弊策略。

<div align="center">

<img src="/figures/09-spatial-worlds/source/06-occupancy-scratch/surroundocc-fig3.png" alt="SurroundOcc 对比三维体素查询与 BEV 查询的跨视图注意力，说明高度维度何时被显式保留。" width="86%">

_图 9.6-4：SurroundOcc 对比三维体素查询与 BEV 查询的跨视图注意力，说明高度维度何时被显式保留。 出处：[SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving，Yi Wei et al.，2023](https://arxiv.org/abs/2303.09551)。_

</div>

<div align="center">

<img src="/figures/09-spatial-worlds/source/06-occupancy-scratch/occformer-fig1.png" alt="OccFormer 在局部与全局路径中更新三维体素特征，再解码为稠密语义占据。" width="86%">

_图 9.6-5：OccFormer 在局部与全局路径中更新三维体素特征，再解码为稠密语义占据。 出处：[OccFormer: Dual-path Transformer for 3D Semantic Occupancy Prediction，Xiaofeng Wang et al.，2023](https://arxiv.org/abs/2304.05316)。_

</div>

为了解决该痛点，系统采用了**三维 Focal 损失（Focal Loss）**：

$$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

其中：
- $p_t \in [0, 1]$ 为模型对真实类别的预测概率；
- $\gamma \ge 0$ 称为**聚焦参数（Focusing Parameter）**，通常取 $\gamma = 2.0$；
- $\alpha_t \in [0, 1]$ 为类别平衡权重（对于稀有障碍物可设为 $0.75$，空气设为 $0.25$）。

### 极度不平衡下的精确手算对比算例
假设 $\gamma = 2.0, \alpha = 1.0$：
1. **简单负样本（易判空气）**：真实为空气（$y=0$），网络预测其为空气的概率高达 $p_t = 0.99$：
   - 传统交叉熵损失：$\text{CE} = -\ln(0.99) \approx 0.01005$；
   - Focal 损失调制因子：$(1 - 0.99)^2 = 0.01^2 = 0.0001$；
   - 最终 Focal 损失：$\mathcal{L}_{\text{Focal}} = 0.0001 \times 0.01005 \approx 1.0 \times 10^{-6}$；
2. **困难正样本（罕见障碍物）**：真实为异形障碍物（$y=1$），网络信心不足，预测概率仅为 $p_t = 0.20$：
   - 传统交叉熵损失：$\text{CE} = -\ln(0.20) \approx 1.6094$；
   - Focal 损失调制因子：$(1 - 0.20)^2 = 0.8^2 = 0.64$；
   - 最终 Focal 损失：$\mathcal{L}_{\text{Focal}} = 0.64 \times 1.6094 \approx 1.030$。

对比两者的损失贡献比率：
$$\frac{\mathcal{L}_{\text{hard}}}{\mathcal{L}_{\text{easy}}} = \frac{1.030}{1.0 \times 10^{-6}} \approx 1,030,000 \text{ 倍！}$$

这一百万倍的损失差距让网络瞬间无视了数百万个简单的空气体素，把全部反向传播的梯度能量集中攻坚长尾罕见的致命障碍物！

<details>
<summary><b>深入推导：Lovász-Softmax 损失对三维交并比（mIoU）次模扩展的数学证明（点击展开查看完整推导）</b></summary>

在语义占据评估中，最终指标为平均交并比（mIoU $\text{IoU} = \frac{|A \cap B|}{|A \cup B|}$）。
然而 IoU 是离散计数阶跃函数，不可直接求导。
根据次模函数分析（Submodular Analysis），集合误差函数 $\Delta(m) = 1 - \text{IoU}(m)$ 的 Lovász 扩展（Lovász Extension）给出了在连续概率向量上的凸包松弛：
$$\mathcal{L}_{\text{Lovasz}}(m) = \sum_{i=1}^N m_{\pi(i)} \left( \Delta(\{\pi(1), \dots, \pi(i)\}) - \Delta(\{\pi(1), \dots, \pi(i-1)\}) \right)$$
其中 $\pi$ 为误差向量按降序排列的排列置换。Lovász 扩展损失在理论上被证明是离散 Jaccard 损失的最佳凸上界，在反向传播中直接最大化三维体素占据的 IoU 指标。
</details>

---

## 9.6.4 纯底层 PyTorch 代码实现：从零搭建三维占据网格预测网络

下面我们使用纯底层 PyTorch 算子实现一个完整的 2D 到 3D 占据网格预测网络，包括双线性体素反投影采样、三维轻量卷积解码器与 Focal 损失计算。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleOccupancyNetwork(nn.Module):
    """
    轻量级三维语义占据网格预测网络
    """
    def __init__(self, in_channels: int = 32, num_classes: int = 4, voxel_res: tuple = (16, 16, 8)):
        super().__init__()
        self.voxel_d, self.voxel_h, self.voxel_w = voxel_res
        self.num_classes = num_classes

        # 3D 卷积编码器-解码器
        self.conv3d_block = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, num_classes, kernel_size=1) # 输出各类别 Logits
        )

    def sample_2d_to_3d(self, img_feat: torch.Tensor, grid_norm: torch.Tensor) -> torch.Tensor:
        """
        利用 grid_sample 将 2D 图像特征双线性插值采样到 3D 体素网格中
        :param img_feat: (B, C, H, W) 2D 图像特征
        :param grid_norm: (B, D, H_v, W_v, 2) 归一化在 [-1, 1] 的 2D 投影采样坐标
        :return: (B, C, D, H_v, W_v) 3D 体素初始特征
        """
        B, C, H, W = img_feat.shape
        # grid_sample 要求 4D 或 5D 输入，此处将 D 展平到批次执行采样
        B, D_v, H_v, W_v, _ = grid_norm.shape
        flat_grid = grid_norm.view(B, D_v * H_v, W_v, 2)

        # 双线性采样
        sampled_2d = F.grid_sample(
            img_feat, flat_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ) # (B, C, D*H, W)

        sampled_3d = sampled_2d.view(B, C, D_v, H_v, W_v)
        return sampled_3d

    def forward(self, img_feat: torch.Tensor, grid_norm: torch.Tensor) -> torch.Tensor:
        """
        前向计算
        :return: (B, num_classes, D, H_v, W_v) 体素类别 Logits
        """
        init_voxel_feat = self.sample_2d_to_3d(img_feat, grid_norm)
        occupancy_logits = self.conv3d_block(init_voxel_feat)
        return occupancy_logits

def focal_loss_3d(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """
    3D 体素 Focal 损失函数
    :param logits: (B, num_classes, D, H, W)
    :param targets: (B, D, H, W) 整数类别标签
    :return: 标量损失
    """
    ce_loss = F.cross_entropy(logits, targets, reduction="none") # (B, D, H, W)
    p_t = torch.exp(-ce_loss) # 获取真实类别的预测概率
    focal_weight = (1.0 - p_t) ** gamma
    return (focal_weight * ce_loss).mean()

# ===================================================================
# 单元测试与 3D 卷积前向推理校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 2
    in_channels = 16
    num_classes = 4 # 0: 空气, 1: 道路, 2: 车辆, 3: 障碍物
    img_h, img_w = 32, 32
    voxel_dim = (8, 16, 16) # (D, H, W)

    model = SimpleOccupancyNetwork(
        in_channels=in_channels, num_classes=num_classes, voxel_res=voxel_dim
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dummy_2d_feat = torch.randn(batch_size, in_channels, img_h, img_w)
    # 构造归一化的 2D 采样网格 [-1, 1]
    dummy_grid = torch.rand(batch_size, voxel_dim[0], voxel_dim[1], voxel_dim[2], 2) * 2.0 - 1.0
    # 模拟真实 3D 标签 (绝大多数为 0: 空气)
    dummy_targets = torch.zeros(batch_size, voxel_dim[0], voxel_dim[1], voxel_dim[2], dtype=torch.long)
    dummy_targets[:, 2:4, 5:8, 5:8] = 2 # 局部的车辆体素

    # 1. 前向推理
    logits = model(dummy_2d_feat, dummy_grid)
    loss = focal_loss_3d(logits, dummy_targets, gamma=2.0)

    # 2. 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"[Occupancy Test] 3D 体素预测输出形状: {logits.shape}")
    print(f"[Occupancy Test] 3D Focal 训练损失: {loss.item():.4f}")

    assert logits.shape == (batch_size, num_classes, *voxel_dim), "3D 体素输出张量形状不符！"
    assert not torch.isnan(loss), "训练损失计算出现 NaN！"
    print("✓ 从零实现三维占据网格预测网络与 3D Focal 损失单测全部通过！")
```

---

## 9.6.5 本节小结

回顾本节内容，我们建立了三维语义占据网格预测的完整技术图谱：
1. **几何范式跃迁**：从简化的 3D 长方体检测框走向稠密三维体素网格，彻底消除了未知异形障碍物的漏检盲区；
2. **2D 到 3D 的视线反投影**：利用双线性插值采样将二维高维图像特征无缝铺设至三维体素网格中；
3. **稀疏不平衡优化**：通过 Focal 损失动态抑制占据 $95\%$ 以上体积的简单空气背景，使模型集中算力攻坚长尾障碍物。
