# 9.7 四维时空驾驶世界模型实战精讲

在整个自动驾驶与空间智能的发展史上，传统的工程架构长期采用“感知-预测-规划（Perception-Prediction-Planning）”三段式级联流程。

然而，这种传统的流水线模式存在一个无法克服的致命缺陷——**信息的层层损耗与误差的级联放大**。感知模块为了将图像转化为几个长方体检测框，丢弃了画面中丰富的微观纹理、刹车灯闪烁、路面积水反光等关键物理信号；一旦感知模块在雨夜产生一次漏检，下游的规划模块就会因为输入数据的彻底缺失而引发致命事故。

为了打破模块之间的信息壁垒，以 **UniAD**、**FIERY**、**OccWorld** 与 **Drive-OccWorld** 为代表的前沿端到端系统，开创了**四维时空世界模型大一统（Unified 4D Spacetime World Models）**范式。

系统在统一的四维潜在几何流形中，同时推演周围三维占据网格随时间的演化规律，并直接输出平滑、安全的物理驾驶轨迹，实现了感知、世界模型与运动规划的完美闭环。

<div align="center">

<img src="/figures/09-spatial-worlds/source/07-four-d-driving-concise/driveoccworld-fig4.png" alt="Drive-OccWorld 展示多步三维占据与场景流预测，直观看到车辆周围空间如何随时间演化。" width="86%">

_图 9.7-1：Drive-OccWorld 展示多步三维占据与场景流预测，直观看到车辆周围空间如何随时间演化。 出处：[Drive-OccWorld: 4D Occupancy World Models for Autonomous Driving，Ziying Song et al.，2024](https://arxiv.org/abs/2405.01401)。_

</div>

---

## 9.7.1 物理与认知基石：端到端大一统与四维物理因果

要理解四维时空世界模型的实战价值，我们首先必须审视传统模块化切分与端到端物理建模的哲学差异。

### 1. 传统模块化流水线的“信息断崖”
在传统的自动驾驶管线中：
- 感知模块输出：“前方 30 米有一辆车（边界框 $[x, y, z, l, w, h]$）”；
- 预测模块接力：“该车辆在未来 3 秒将以 $5\text{ m/s}$ 匀速直行”；
- 规划模块求解：“自车应减速避让”。

但在真实的极端物理世界中，如果前方车辆的尾灯突然双闪、或者车身开始轻微倾斜（爆胎前兆），这些蕴含在像素与微观体素中的丰富高频物理预警信号，在“边界框抽象”的那一瞬间就被彻底抹杀了！

### 2. 四维占据世界模型的“物理全息推演”
在四维占据世界模型中，系统直接在包含时间维度的四维体素张量 $\mathcal{O} \in \mathbb{R}^{T \times X \times Y \times Z}$ 上运行：
- 它不仅知道空间中任意立方体在当前的占有概率，还能自回归推演出未来 $T$ 个时间步内该立方体是否会被实体占据，以及伴随的三维瞬时运动矢量（场景流）；
- 规划器直接在这个“活生生演化的四维时空迷宫”中寻找一条无碰撞且动力学最平稳的安全流线。

<div align="center">

<img src="/figures/09-spatial-worlds/source/07-four-d-driving-concise/uniad-fig2.png" alt="UniAD 让跟踪、地图、运动预测、占据与规划查询在统一 BEV 管线中交互。" width="86%">

_图 9.7-2：UniAD 让跟踪、地图、运动预测、占据与规划查询在统一 BEV 管线中交互。 出处：[Planning-oriented Autonomous Driving，Yihan Hu et al.，2023](https://arxiv.org/abs/2212.10156)。_

</div>

<div align="center">

<img src="/figures/09-spatial-worlds/source/07-four-d-driving-concise/fiery-fig2.png" alt="FIERY 从环视视频构建时序 BEV 潜变量，并预测未来实例分割与运动。" width="86%">

_图 9.7-3：FIERY 从环视视频构建时序 BEV 潜变量，并预测未来实例分割与运动。 出处：[FIERY: Future Instance Segmentation and Motion Forecasting from Monocular Cameras，Anthony Hu et al.，2021](https://arxiv.org/abs/2104.14512)。_

</div>

---

## 9.7.2 核心数学推导一：四维时空占据联合损失函数体系

在四维占据世界模型中，网络在前向推演时同时优化三维占据分布、时空运动场景流与最终规划轨迹。

<div align="center">

<img src="/figures/09-spatial-worlds/latex/07-four-d-driving-concise/occupancy-bce-four-axis-mean.png" alt="每个批次和三维体素先计算 BCE，再沿批次、高度和两个平面轴求和并归一化为标量" width="86%">

_图 9.7-4：每个批次和三维体素先计算 BCE，再沿批次、高度和两个平面轴求和并归一化为标量。_

</div>

<div align="center">

<img src="/figures/09-spatial-worlds/source/07-four-d-driving-concise/occworld-fig2.png" alt="OccWorld 先编码多相机观测为三维占据状态，再在潜在空间中预测未来并解码多步占据。" width="86%">

_图 9.7-5：OccWorld 先编码多相机观测为三维占据状态，再在潜在空间中预测未来并解码多步占据。 出处：[OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving，Wenzhao Zheng et al.，2023](https://arxiv.org/abs/2311.16038)。_

</div>

### 1. 多任务联合损失函数
系统的全监督联合训练目标由三部分组成：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Occ}} + \lambda_1 \mathcal{L}_{\text{Flow}} + \lambda_2 \mathcal{L}_{\text{Plan}}$$

1. **四维时空占据二值交叉熵损失（Occupancy BCE Loss）**：
   $$\mathcal{L}_{\text{Occ}} = -\frac{1}{B \cdot T \cdot X Y Z} \sum_{b, t, x, y, z} \left[ y \log(\hat{p}) + (1 - y) \log(1 - \hat{p}) \right]$$
2. **三维场景流平滑回归损失（Scene Flow L1 Loss）**：
   $$\mathcal{L}_{\text{Flow}} = \frac{1}{|\mathcal{V}_{\text{occupied}}|} \sum_{\mathbf{x} \in \mathcal{V}_{\text{occupied}}} \left\| \hat{\mathbf{v}}(\mathbf{x}) - \mathbf{v}^*(\mathbf{x}) \right\|_1$$
3. **安全轨迹规划模仿损失（Planning Imitation Loss）**：
   $$\mathcal{L}_{\text{Plan}} = \frac{1}{H} \sum_{\tau=1}^H \left\| \hat{\mathbf{x}}_\tau - \mathbf{x}_\tau^* \right\|_2^2$$

**手算代入算例**：
设某体素在未来第 $t+1$ 秒被障碍物占据（真实标签 $y = 1$），网络预测的占据概率为 $\hat{p} = 0.80$；真实场景流速度向量为 $\mathbf{v}^* = [5.0, 0.0, 0.0]^\top\text{ m/s}$，网络预测为 $\hat{\mathbf{v}} = [4.8, 0.1, 0.0]^\top\text{ m/s}$。

1. 计算占据二值交叉熵损失：
   $$\mathcal{L}_{\text{Occ}} = -(1 \times \ln(0.80) + 0) = -\ln(0.80) \approx 0.2231$$
2. 计算场景流 L1 误差：
   $$\mathcal{L}_{\text{Flow}} = |4.8 - 5.0| + |0.1 - 0.0| + |0.0 - 0.0| = 0.2 + 0.1 + 0.0 = 0.30\text{ m/s}$$
3. 计算总损失（设 $\lambda_1 = 1.0$）：
   $$\mathcal{L} = 0.2231 + 1.0 \times 0.30 = 0.5231$$

初等代数的代入推导极为直观，生动展现了网络如何同时监督“空间有无物体”与“物体运动速度”两大核心物理属性！

<details>
<summary><b>深入推导：四维占据时序马尔可夫决策过程的贝尔曼最优性方程解析证明（点击展开查看完整推导）</b></summary>

将四维占据世界模型形式化为连续状态马尔可夫决策过程 $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$。
定义时空安全价值函数 $V^*(\mathbf{s})$ 为在未来时空占据场演变下的最大贴现安全回报：
$$V^*(\mathbf{s}) = \max_{\mathbf{a} \in \mathcal{A}} \left[ \mathcal{R}(\mathbf{s}, \mathbf{a}) + \gamma \int_{\mathcal{S}} \mathcal{P}(\mathbf{s}' \mid \mathbf{s}, \mathbf{a}) V^*(\mathbf{s}') d\mathbf{s}' \right]$$
其中即时奖励 $\mathcal{R}(\mathbf{s}, \mathbf{a}) = -\int_{\mathcal{V}} \mathbb{I}(\text{Collision}(\mathbf{s}, \mathbf{x})) \hat{O}(\mathbf{x}) d\mathbf{x} - w \|\ddot{\mathbf{x}}\|^2$。
当潜在状态转移概率 $\mathcal{P}$ 由世界模型准确拟合时，端到端规划器的输出策略 $\pi^*(\mathbf{s})$ 严格收敛于该贝尔曼方程的唯一不动点极值解。
</details>

---

## 9.7.3 核心数学推导二：时空防碰撞代价函数与轨迹平滑优化

在生成未来驾驶轨迹时，规划器从多条候选轨迹族中通过**可微代价函数（Cost Function）**进行最优轨迹评分与筛选。

设一条候选轨迹在未来 $H$ 步的时间空间坐标序列为 $\tau = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_H\}$，控制周期时间间隔为 $\Delta t = 0.5\text{ s}$。轨迹的总代价函数由三个具有明确物理意义的项加权构成：

$$J(\tau) = w_{\text{coll}} \cdot \mathcal{J}_{\text{coll}}(\tau) + w_{\text{smooth}} \cdot \mathcal{J}_{\text{smooth}}(\tau) + w_{\text{goal}} \cdot \mathcal{J}_{\text{goal}}(\tau)$$

### 1. 三大物理代价项详细分解
1. **时空碰撞代价项（Spatiotemporal Collision Cost）**：
   $$\mathcal{J}_{\text{coll}}(\tau) = \sum_{t=1}^H \hat{\mathcal{O}}_t(\mathbf{x}_t)$$
   将轨迹在时刻 $t$ 的坐标 $\mathbf{x}_t = (x_t, y_t, z_t)$ 代入世界模型预测出的第 $t$ 步三维占据网格 $\hat{\mathcal{O}}_t$ 中。若轨迹穿过障碍物体素，占据概率 $\hat{\mathcal{O}}_t \approx 1.0$，代价累加剧增；若穿过空旷空气，$\hat{\mathcal{O}}_t \approx 0.0$，代价格外微小；
2. **舒适度与平滑度代价项（Smoothness & Jerk Cost）**：
   利用二阶差分近似计算自车在每个时刻的加速度与急动度（Jerk）：
   $$\mathcal{J}_{\text{smooth}}(\tau) = \sum_{t=2}^{H-1} \left\| \frac{\mathbf{x}_{t+1} - 2\mathbf{x}_t + \mathbf{x}_{t-1}}{\Delta t^2} \right\|_2^2$$
   惩罚急加速、急刹车与猛打方向盘，保证乘坐的物理平稳性；
3. **导航终点抵达代价项（Goal Reaching Cost）**：
   $$\mathcal{J}_{\text{goal}}(\tau) = \|\mathbf{x}_H - \mathbf{x}_{\text{target}}\|_2^2$$
   保证车辆在未来第 $H$ 步时能够精准到达导航路径给出的目标路点。

### 2. 轨迹决策手算代入对比算例
设自车正前方 $20\text{ m}$ 处有一辆静止故障车（其体素占据概率 $\hat{\mathcal{O}} = 0.95$）。规划器生成了两条候选轨迹（时间步 $H = 2$，目标终点 $\mathbf{x}_{\text{target}} = [0, 25]^\top$）：
- **轨迹 A（保持直行）**：$\mathbf{x}_1 = [0, 10]^\top, \mathbf{x}_2 = [0, 20]^\top$（直接撞上静止故障车，$\hat{\mathcal{O}}(\mathbf{x}_2) = 0.95$）；
- **轨迹 B（提前平滑变道）**：$\mathbf{x}_1 = [1.5, 10]^\top, \mathbf{x}_2 = [3.0, 20]^\top$（从右侧车道绕行，所有经过体素均为空气 $\hat{\mathcal{O}} = 0.02$；变道带来微小的横向加速度代价 $\mathcal{J}_{\text{smooth}} = 0.12$；终点偏离目标 $3\text{ m}$，$\mathcal{J}_{\text{goal}} = 3^2 = 9.0$）。

设定权重：$w_{\text{coll}} = 100.0, w_{\text{smooth}} = 10.0, w_{\text{goal}} = 1.0$。

我们来手算两者的总代价：
1. **计算轨迹 A 代价**：
   $$J(\text{A}) = 100.0 \times (0.0 + 0.95) + 10.0 \times 0.0 + 1.0 \times (25 - 20)^2 = 95.0 + 0.0 + 25.0 = 120.0$$
2. **计算轨迹 B 代价**：
   $$J(\text{B}) = 100.0 \times (0.02 + 0.02) + 10.0 \times 0.12 + 1.0 \times (3^2 + 5^2) = 4.0 + 1.2 + 34.0 = 39.2$$

因为 $J(\text{B}) = 39.2 \ll J(\text{A}) = 120.0$，系统以压倒性的优势胜出并果断执行轨迹 B 变道绕行动作！整个决策过程全部由透明可解释的初等代数与力学物理代价清晰驱动！

<details>
<summary><b>深入推导：二次规划（QP）轨迹平滑优化与 KKT 极值代数推导（点击展开查看完整推导）</b></summary>

将轨迹平滑优化写为标准二次规划（QP）形式：
$$\min_{\mathbf{X}} \frac{1}{2} \mathbf{X}^\top \mathbf{H} \mathbf{X} + \mathbf{f}^\top \mathbf{X} \quad \text{s.t.} \quad \mathbf{A} \mathbf{X} \le \mathbf{b}$$
其中 Hessian 矩阵 $\mathbf{H} = \mathbf{D}_2^\top \mathbf{D}_2 + \lambda \mathbf{I}$ 为二阶差分算子的对称正定矩阵。
根据 KKT 条件，引入拉格朗日乘子向量 $\boldsymbol{\lambda}$：
$$\nabla_{\mathbf{X}} \mathcal{L} = \mathbf{H} \mathbf{X} + \mathbf{f} + \mathbf{A}^\top \boldsymbol{\lambda} = \mathbf{0} \implies \mathbf{X}^* = -\mathbf{H}^{-1} (\mathbf{f} + \mathbf{A}^\top \boldsymbol{\lambda})$$
在凸多面体凸可行域内，全局最优平滑轨迹 $\mathbf{X}^*$ 具有唯一的解析解析解，保障了毫秒级工控机部署的绝对实时性。
</details>

---

## 9.7.4 纯底层 PyTorch 代码实现：四维时空占据与端到端规划联合引擎

下面我们使用纯底层 PyTorch 算子实现一个完整的四维时空占据预测与轨迹规划联合网络引擎。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatiotemporalOccupancyModel(nn.Module):
    """
    四维时空占据与场景流预测模型
    输入当前 3D 占据网格，自回归推演未来多步 3D 占据与运动场景流
    """
    def __init__(self, in_channels: int = 16, hidden_dim: int = 32):
        super().__init__()
        # 时空 3D 卷积层
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU()
        )
        # 占据概率头 (输出 Sigmoid 占有概率)
        self.occ_head = nn.Conv3d(hidden_dim, 1, kernel_size=1)
        # 场景流头 (输出 3D 速度向量 vx, vy, vz)
        self.flow_head = nn.Conv3d(hidden_dim, 3, kernel_size=1)

    def forward(self, voxel_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param voxel_feat: (B, in_channels, D, H, W)
        :return: occ_probs (B, 1, D, H, W), scene_flows (B, 3, D, H, W)
        """
        feat = self.conv3d(voxel_feat)
        occ_probs = torch.sigmoid(self.occ_head(feat))
        scene_flows = self.flow_head(feat)
        return occ_probs, scene_flows

class EndToEndTrajectoryPlanner(nn.Module):
    """
    基于四维时空占据特征的端到端轨迹规划器
    """
    def __init__(self, in_channels: int = 16, horizon: int = 6):
        super().__init__()
        self.horizon = horizon
        # 全局池化与 MLP 规划头
        self.planner_net = nn.Sequential(
            nn.AdaptiveAvgPool3d((2, 4, 4)),
            nn.Flatten(),
            nn.Linear(in_channels * 2 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, horizon * 2) # 输出未来 H 步的 (x, y) 坐标
        )

    def forward(self, voxel_feat: torch.Tensor) -> torch.Tensor:
        """
        :param voxel_feat: (B, in_channels, D, H, W)
        :return: (B, horizon, 2) 规划的连续物理轨迹
        """
        B = voxel_feat.size(0)
        traj_flat = self.planner_net(voxel_feat)
        return traj_flat.view(B, self.horizon, 2)

# ===================================================================
# 单元测试：占据预测、场景流与端到端规划联合训练反向传播
# ===================================================================
if __name__ == "__main__":
    batch_size = 2
    in_c = 16
    voxel_d, voxel_h, voxel_w = 4, 16, 16
    horizon = 6

    occ_model = SpatiotemporalOccupancyModel(in_channels=in_c, hidden_dim=32)
    planner = EndToEndTrajectoryPlanner(in_channels=in_c, horizon=horizon)

    params = list(occ_model.parameters()) + list(planner.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    dummy_voxel_feat = torch.randn(batch_size, in_c, voxel_d, voxel_h, voxel_w)

    # 模拟真实标签
    target_occ = torch.randint(0, 2, (batch_size, 1, voxel_d, voxel_h, voxel_w)).float()
    target_flow = torch.randn(batch_size, 3, voxel_d, voxel_h, voxel_w)
    target_traj = torch.randn(batch_size, horizon, 2)

    # 1. 前向推演
    pred_occ, pred_flow = occ_model(dummy_voxel_feat)
    pred_traj = planner(dummy_voxel_feat)

    # 2. 计算联合损失
    loss_occ = F.binary_cross_entropy(pred_occ, target_occ)
    loss_flow = F.l1_loss(pred_flow, target_flow)
    loss_plan = F.mse_loss(pred_traj, target_traj)
    total_loss = loss_occ + 1.0 * loss_flow + 2.0 * loss_plan

    # 3. 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(f"[4D Drive Test] 预测 3D 占据张量形状: {pred_occ.shape}")
    print(f"[4D Drive Test] 预测 3D 场景流张量形状: {pred_flow.shape}")
    print(f"[4D Drive Test] 预测规划轨迹形状: {pred_traj.shape}")
    print(f"[4D Drive Test] 联合多任务训练损失: {total_loss.item():.4f}")

    assert pred_occ.shape == (batch_size, 1, voxel_d, voxel_h, voxel_w), "占据张量维度不符！"
    assert pred_traj.shape == (batch_size, horizon, 2), "规划轨迹形状不符！"
    assert not torch.isnan(total_loss), "联合损失出现 NaN！"
    print("✓ 四维时空占据预测与端到端规划联合网络单测全部通过！")
```

---

## 9.7.5 本节小结

回顾本节内容，我们完成了四维时空驾驶世界模型的终极大一统闭环：
1. **打破信息孤岛**：告别传统模块化级联的感知漏检与信息截断，在连续四维时空潜流形中实现大一统表达；
2. **多任务物理协同**：将空间占据、瞬时场景流与未来规划轨迹纳入统一联合优化，实现了深度的物理因果建模；
3. **安全代价优化**：结合时空占据防碰撞惩罚与二阶差分加速度平滑度优化，保障了输出轨迹在真实物理世界中的极致安全性与乘坐舒适度。
