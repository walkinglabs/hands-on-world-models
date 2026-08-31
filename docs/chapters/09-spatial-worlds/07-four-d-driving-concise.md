# 9.7 4D 自动驾驶预测模型的简洁实现

一辆车从路边驶入主路时，当前占据网格只能说明“它此刻在哪里”，不能说明下一秒会占据哪片空间。把三维空间 $(X,Y,Z)$ 与时间 $T$ 一起建模，便得到本节所说的 4D 表示。它关注的不是单帧重建，而是空间状态如何随时间和自车动作变化。

<div align="center">
<img src="/figures/09-spatial-worlds/source/07-four-d-driving-concise/driveoccworld-fig4.png" alt="Drive-OccWorld 展示多步三维占据与场景流预测，直观看到车辆周围空间如何随时间演化。" width="86%">

_图 9.7-1：Drive-OccWorld 展示多步三维占据与场景流预测，直观看到车辆周围空间如何随时间演化。 出处：Yu Yang et al.，[Drive-OccWorld: Learning 3D Occupancy Forecasting from Monocular Multi-Camera Videos](https://arxiv.org/abs/2408.14197)（2024），Figure 4。_
</div>

UniAD 把检测、跟踪、地图、运动预测与规划统一到基于查询的 Transformer 框架中，并在 BEV 表征上完成多任务交互 [[Yihan Hu et al., 2023]](https://arxiv.org/abs/2212.10156)；它不是“简单的循环神经网络展开”。二维 BEV 的柱状表示会压缩高度信息，而 OccWorld 等工作进一步使用三维占用表示预测未来场景演化 [[Zheng et al., 2023]](https://arxiv.org/abs/2311.16038)。

<div align="center">
<img src="/figures/09-spatial-worlds/source/07-four-d-driving-concise/uniad-fig2.png" alt="UniAD 让跟踪、地图、运动预测、占据与规划查询在统一 BEV 管线中交互。" width="86%">

_图 9.7-2：UniAD 让跟踪、地图、运动预测、占据与规划查询在统一 BEV 管线中交互。 出处：Yihan Hu et al.，[Planning-oriented Autonomous Driving](https://arxiv.org/abs/2212.10156)（2023），Figure 2。_
</div>

本节从匀速运动的状态更新出发，把状态推广到三维特征体，再用一个动作条件 3D 卷积网络预测下一时刻的占据概率。这个实现用于说明张量接口，不等同于 UniAD 或 OccWorld 的完整复现。

## 从运动学方程到 4D 状态转移

先看匀速直线运动。若汽车在时刻 $t$ 的位置为 $x_t$，速度在 $\Delta t$ 内近似不变，则下一时刻的位置为：

$$x_{t+1} = x_t + v_t \Delta t$$

其中，$v_t\Delta t$ 是状态增量。把标量状态推广到驾驶场景，可令 $t$ 时刻的三维空间特征为 $\mathbf{S}_t \in \mathbb{R}^{C \times Z \times H \times W}$：$C$ 是通道数，$Z,H,W$ 是体素网格的三个空间维度。

场景变化既来自其他车辆和行人的运动，也受到自车转向与加速的影响。引入自车动作 $\mathbf{a}_t \in \mathbb{R}^{D_a}$，用神经网络参数化状态转移：

<div align="center">
<img src="/figures/09-spatial-worlds/source/07-four-d-driving-concise/fiery-fig2.png" alt="FIERY 从环视视频构建时序 BEV 潜变量，并预测未来实例分割与运动。" width="86%">

_图 9.7-3：FIERY 从环视视频构建时序 BEV 潜变量，并预测未来实例分割与运动。 出处：Anthony Hu et al.，[FIERY: Future Instance Prediction in Bird’s-Eye View from Surround Monocular Cameras](https://arxiv.org/abs/2104.10490)（2021），Figure 2。_
</div>

$$\mathbf{S}_{t+1} = \mathcal{F}_\theta(\mathbf{S}_t, \mathbf{a}_t)$$

$\mathcal{F}_\theta$ 直接预测离散时间的状态变化，不一定对应某个已知微分方程的数值积分。3D 卷积适合建模局部空间邻域，循环、注意力或多步卷积则负责利用时间上下文。是否具有几何和动力学一致性，需要由训练目标与评测验证。

## 4D 占位网格预测的数学推导

为了得到可检查的空间输出，可以把 $\mathbf{S}_{t+1}$ 解码为体素占据概率。对每个坐标索引 $(z,y,x)$，模型输出下一时刻被物体占据的概率 $p(O_{z,y,x}^{t+1}=1)$。

对于单个体素，这是一个典型的二分类问题。其伯努利分布下的交叉熵损失函数（Cross-Entropy Loss）为：

$$l(z,y,x) = - \left[ o_{z,y,x} \log p_{z,y,x} + (1 - o_{z,y,x}) \log (1 - p_{z,y,x}) \right]$$

其中 $o_{z,y,x} \in \{0, 1\}$ 是该位置是否被占据的真实标签（Ground Truth），$p_{z,y,x}$ 是模型预测的概率标量。

设批次大小为 $B$，预测张量为 $\mathbf{\hat{O}} \in (0,1)^{B \times Z \times H \times W}$，标签为 $\mathbf{O} \in \{0,1\}^{B \times Z \times H \times W}$。对所有体素取平均，得到：

$$\mathcal{L}_{occ} = -\frac{1}{BZHW} \sum_{b,z,y,x} \left[ O_{bzyx}\log \hat O_{bzyx} + (1-O_{bzyx})\log(1-\hat O_{bzyx}) \right]$$

<div align="center">
<img src="/figures/09-spatial-worlds/latex/07-four-d-driving-concise/occupancy-bce-four-axis-mean.png" alt="每个批次和三维体素先计算 BCE，再沿批次、高度和两个平面轴求和并归一化为标量" width="86%">

_图 9.7-4：每个 b,z,y,x 位置先产生一个标量 BCE；四重求和消去全部索引，再除以体素总数得到标量均值。本文根据上式绘制。_
</div>

最小化这个损失会提高体素分类准确度，但它本身不保证长期动力学一致，也没有处理空闲体素远多于占据体素的类别不平衡。

## 核心模型代码实现

下面实现一个简化网络：3D 卷积编码当前特征体，把动作向量广播到每个体素，再预测下一状态和占据概率。

```python
import torch
from torch import nn

class Conv3DBlock(nn.Module):
    """基础的 3D 卷积块用于空间特征提取"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 形状: (B, C, Z, H, W)
        return self.relu(self.norm(self.conv(x)))

class Simple4DPredictor(nn.Module):
    """简洁的 4D 世界模型预测器"""
    def __init__(self, hidden_dim=64, action_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 空间特征处理
        self.spatial_encoder = Conv3DBlock(hidden_dim, hidden_dim)
        # 将动作映射到空间特征通道
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU()
        )
        # 用 3D 卷积模拟局部状态传播
        self.transition_net = nn.Sequential(
            Conv3DBlock(hidden_dim * 2, hidden_dim),
            Conv3DBlock(hidden_dim, hidden_dim)
        )
        # 占位网格解码器
        self.occupancy_decoder = nn.Conv3d(hidden_dim, 1, kernel_size=1)

    def forward(self, state_t, action_t):
        """
        state_t: 当前时刻三维状态 (B, C, Z, H, W)
        action_t: 当前自车动作 (B, action_dim) 包含转向和加速度
        """
        B, C, Z, H, W = state_t.shape

        # 1. 编码空间几何
        spatial_features = self.spatial_encoder(state_t)

        # 2. 处理并广播动作特征到完整的三维空间维度
        act_feat = self.action_mlp(action_t)  # (B, C)
        act_feat_broadcast = act_feat.view(B, C, 1, 1, 1).expand(B, C, Z, H, W)

        # 3. 状态转移算子 F_theta: 融合当前状态与动作
        fused_state = torch.cat([spatial_features, act_feat_broadcast], dim=1)
        state_next = self.transition_net(fused_state)  # (B, C, Z, H, W)

        # 4. 解码为概率分布 O_hat
        occupancy_logits = self.occupancy_decoder(state_next)
        occupancy_prob = torch.sigmoid(occupancy_logits)

        return state_next, occupancy_prob
```

动作特征被广播到所有体素，因此每个空间位置都能读取同一个全局控制条件。网络仍需从数据中学习动作如何影响不同位置；广播操作本身并不实施自车坐标变换，也不保证状态平滑。

## 从教学模型到可评测系统

上面的网络只接收一个已经对齐的三维状态，实际系统还要解决多相机特征提升、遮挡区域监督、自车坐标变换和多步误差累积。评测时至少应分开观察三类指标：

<div align="center">
<img src="/figures/09-spatial-worlds/source/07-four-d-driving-concise/occworld-fig2.png" alt="OccWorld 先编码多相机观测为三维占据状态，再在潜在空间中预测未来并解码多步占据。" width="86%">

_图 9.7-5：OccWorld 先编码多相机观测为三维占据状态，再在潜在空间中预测未来并解码多步占据。 出处：Wenzhao Zheng et al.，[OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving](https://arxiv.org/abs/2311.16038)（2023），Figure 2。_
</div>

- **单步占据质量**：比较每个语义类别的 IoU，而不是只看被空闲体素主导的总体准确率。
- **多步预测稳定性**：按预测时距分别报告误差，检查远期结果是否逐渐模糊或静止。
- **规划相关性**：把预测送入固定规划器，观察碰撞率、可行驶区域违规和舒适性指标；视觉上合理并不等同于驾驶上安全。

部署预算也应在具体硬件上测量。体素分辨率、时间长度和通道数都会直接改变显存与延迟；量化是否可用，则取决于目标设备、算子支持和精度回退结果。这里不假定某个通用大语言模型运行时能够直接加速 3D 占据网络。

## 练习

1. 若自车状态还受到可观测扰动 $\mathbf{d}_t$ 影响，把它加入状态转移函数，并写出相应张量形状。
2. 当前实现只广播动作。尝试先依据自车位姿变化对 `spatial_features` 做三维重采样，再预测其他物体的残余运动。
3. 分别把 $Z,H,W$ 扩大一倍，估算特征体的显存和 3D 卷积计算量如何变化。
