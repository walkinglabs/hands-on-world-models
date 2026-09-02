# 9.5 自动驾驶世界模型与端到端闭环预测

在现代自动驾驶与具身移动智能的探索中，行业长期面临着一个被统称为“长尾效应（Long-tail Distribution）”的严酷现实挑战——即便自动驾驶车队在真实街道上累积行驶了数千万公里，也极难收集到诸如“前方货车突发侧翻并掉落滚木”、“暴风雨中逆行车辆突然迎面驶来”等百万分之一概率的罕见极端危险场景（Corner Cases）。

如果直接将未经验证的算法在真实物理车体上进行极端工况实测，代价将是无法挽回的安全事故与生命财产损失。

为了在安全受控的虚拟环境中对自动驾驶算法进行高强度极限压力测试，**自动驾驶世界模型（Autonomous Driving World Models）**应运而生。

从 Wayve 提出的 **GAIA-1**，到清华大学与产业界推出的 **DriveDreamer**、**Drive-WM**，世界模型将自车的物理控制动作（方向盘转角、油门踏板开度、刹车制动力）作为因果干预条件，直接在脑海中推演未来数秒内周围多相机环视画面的连续演变，赋予了自动驾驶系统“在想象中经历千万种未来”的核心能力。

<div align="center">

<img src="/figures/09-spatial-worlds/source/05-driving-world-models/drivedreamer-fig1.png" alt="DriveDreamer 根据道路结构与交通参与者条件生成多样、可控的真实驾驶场景序列。" width="86%">

_图 9.5-1：DriveDreamer 根据道路结构与交通参与者条件生成多样、可控的真实驾驶场景序列。 出处：[DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving，Xiaofeng Wang et al.，2023](https://arxiv.org/abs/2309.09777)。_

</div>

---

## 9.5.1 物理与交通基石：长尾困境与反事实推理演进

要理解驾驶世界模型的核心价值，我们首先必须审视传统模拟器与生成式物理建模的技术代差。

### 1. 传统游戏渲染引擎（CARLA/AirSim）的现实鸿沟（Sim-to-Real Gap）
在过去，行业主要依赖基于传统游戏图形引擎的仿真软件（如 CARLA）：
- **渲染纹理失真**：多边形贴图粗糙，无法逼真模拟复杂天气下的雨滴反光、镜头水雾、黄昏逆光眩光等真实光学噪声；
- **交通流行为僵硬**：背景 NPC 车辆由简单的规则脚本控制，缺乏真实人类驾驶员的心理博弈（如试探性变道、礼让或抢行）。

### 2. 反事实推理（Counterfactual Reasoning）的物理魅力
神经驾驶世界模型首次赋予了智能体进行**反事实假设（What-if Analysis）**的能力：
- “在刚刚那个路口，如果我没有选择减速刹车，而是猛踩油门向右变道超车，周围的行人和车辆会做出怎样的反应？是否会导致严重碰撞？”

通过在神经网络内部以不同的控制动作为输入展开多条平行的“时空分支世界”，规划器可以在毫秒级时间内挑选出最具安全性与舒适性的最优驾驶动作。

<div align="center">

<img src="/figures/09-spatial-worlds/source/05-driving-world-models/drivewm-fig3.png" alt="Drive-WM 同时预测多相机未来视图，并把动作条件和规划候选纳入统一驾驶世界模型。" width="86%">

_图 9.5-2：Drive-WM 同时预测多相机未来视图，并把动作条件和规划候选纳入统一驾驶世界模型。 出处：[Drive-WM: World Models for Autonomous Driving，Hexing Dong et al.，2023](https://arxiv.org/abs/2311.17918)。_

</div>

---

## 9.5.2 核心数学推导一：自回归时序分解与环视空间一致性

驾驶世界模型的根本数学目标，是学习未来多帧时空观测在历史观测与未来控制动作条件下的条件联合概率分布。

<div align="center">

<img src="/figures/09-spatial-worlds/latex/05-driving-world-models/autoregressive-conditioning-window.png" alt="未来状态联合分布逐步分解，每个预测因子的状态与动作条件窗口随预测步增长" width="86%">

_图 9.5-3：未来状态联合分布逐步分解，每个预测因子的状态与动作条件窗口随预测步增长。_

</div>

### 1. 时序自回归因果链式法则
设一段驾驶轨迹的时长为 $T$。在时刻 $t$，系统拥有由 6 个环视相机拍摄的多视角图像组 $\mathbf{X}_t = \{I_t^{(1)}, I_t^{(2)}, \dots, I_t^{(6)}\}$，自车执行的物理动作控制量为 $\mathbf{a}_t = (\delta_{\text{steer}}, v_{\text{speed}})^\top$，以及道路结构先验条件 $\mathbf{c}$（如高精地图或导航路径）。

根据全概率公式的自回归链式法则，未来观测序列的联合概率分布严格分解为单步条件概率的连乘积：

$$p(\mathbf{X}_{1:T} \mid \mathbf{a}_{1:T}, \mathbf{c}) = \prod_{t=1}^T p\left(\mathbf{X}_t \mid \mathbf{X}_{<t}, \mathbf{a}_{<t}, \mathbf{c}\right)$$

<div align="center">

<img src="/figures/09-spatial-worlds/source/05-driving-world-models/gaia1-fig2.png" alt="GAIA-1 将视频、动作与文本编码为序列，由世界模型自回归预测未来离散视觉标记。" width="86%">

_图 9.5-4：GAIA-1 将视频、动作与文本编码为序列，由世界模型自回归预测未来离散视觉标记。 出处：[GAIA-1: A Generative World Model for Autonomous Driving，Anthony Hu et al.，2023](https://arxiv.org/abs/2309.17080)。_

</div>

### 2. 环视多相机重叠区几何重投影一致性约束
在环视六相机系统中，相邻相机（例如前向主目与左前侧目）之间存在大约 $15\% \sim 20\%$ 的视场重叠区域。
为了防止不同相机的预测画面在接缝处出现车身断裂或错位，系统引入了**多视角几何对齐损失（Cross-View Geometric Consistency Loss）**：

$$\mathcal{L}_{\text{overlap}} = \frac{1}{|\Omega_{ij}|} \sum_{\mathbf{p} \in \Omega_{ij}} \left\| I_t^{(j)}(\mathbf{p}) - I_t^{(i)}\left( \pi_i(\pi_j^{-1}(\mathbf{p}, D_j(\mathbf{p}))) \right) \right\|_1$$

利用上一节学过的投影矩阵 $\pi_i = \mathbf{K}_i [\mathbf{R}_i \mid \mathbf{t}_i]$ 与深度图 $D_j$，将第 $j$ 个相机的重叠像素点投影回第 $i$ 个相机的像素平面，强制两者的重投影误差趋近于 0，确保了生成的六路视频在空间几何上的绝对自洽。

<details>
<summary><b>深入推导：变分信息瓶颈（VIB）在时序自回归潜在动力学中的下界严密推导（点击展开查看完整推导）</b></summary>

为防止世界模型记忆不相关的静态背景高频噪点，引入隐状态变量 $\mathbf{z}_t$ 并优化变分信息瓶颈目标：
$$\max_{\theta, \phi} \sum_{t=1}^T \Big( \underbrace{I(\mathbf{z}_t; \mathbf{X}_t \mid \mathbf{X}_{<t}, \mathbf{a}_{<t})}_{\text{未来预测充分性}} - \beta \underbrace{I(\mathbf{z}_t; \mathbf{X}_{<t} \mid \mathbf{a}_{<t})}_{\text{历史记忆压缩率}} \Big)$$
根据变分下界展开，该目标等价于最大化未来重构对数似然，同时最小化潜在先验分布与后验分布之间的 KL 散度：
$$\mathcal{L}_{\text{VIB}} = \sum_{t=1}^T \left( \mathbb{E}_{q_\phi(\mathbf{z}_t)} [\log p_\theta(\mathbf{X}_t \mid \mathbf{z}_t)] - \beta D_{\text{KL}}(q_\phi(\mathbf{z}_t \mid \mathbf{X}_{\le t}) \parallel p_\theta(\hat{\mathbf{z}}_t \mid \mathbf{z}_{<t}, \mathbf{a}_{<t})) \right)$$
该公式奠定了世界模型在保证控制预测能力的同时抑制无关背景过拟合的信息论基础。
</details>

---

## 9.5.3 核心数学推导二：动作引导扩散与无分类器引导（CFG）

在扩散世界模型中，如何确保生成的视频画面严格服从输入的动作控制量（例如输入急左转动作时，画面中的街道必须向右剧烈旋转）？

<div align="center">

<img src="/figures/09-spatial-worlds/source/05-driving-world-models/drivedreamer-fig3.png" alt="DriveDreamer 把结构化交通条件、驾驶动作与扩散生成器组合起来，生成受控未来驾驶画面。" width="86%">

_图 9.5-5：DriveDreamer 把结构化交通条件、驾驶动作与扩散生成器组合起来，生成受控未来驾驶画面。 出处：[DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving，Xiaofeng Wang et al.，2023](https://arxiv.org/abs/2309.09777)。_

</div>

系统采用了**无分类器引导（Classifier-Free Guidance, CFG）**技术：
在训练阶段，网络以一定概率（例如 $15\%$）随机丢弃输入的动作条件向量 $\mathbf{a}$（替换为空条件 $\emptyset$），从而让单个网络同时学会**无条件得分流**与**条件得分流**。

在推理去噪时，模型通过线性外推来大幅强化动作指令的控制响应：

$$\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{a}) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset) + s \cdot \left( \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{a}) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset) \right)$$

其中 $s \ge 1.0$ 称为**引导尺度（Guidance Scale）**（通常取 $s = 3.0 \sim 7.5$）。

> **初等代数直觉**：
> 这一公式将网络输出改写为：$\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_{\text{uncond}} + s \cdot \Delta \boldsymbol{\epsilon}_{\text{action}}$。
> 当 $s > 1$ 时，动作信号对画面特征梯度的干预被成倍放大，确保了生成的多视角视频与驾驶员的方向盘控制具有极高的物理契合度！

<details>
<summary><b>深入推导：无分类器引导（CFG）隐空间概率流 ODE 贝叶斯最优得分流证明（点击展开查看完整推导）</b></summary>

根据贝叶斯定理：
$$\log p_t(\mathbf{x} \mid \mathbf{a}) = \log p_t(\mathbf{x}) + \log p_t(\mathbf{a} \mid \mathbf{x}) - \log p(\mathbf{a})$$
两边对状态变量 $\mathbf{x}$ 取空间梯度（得分函数）：
$$\nabla_{\mathbf{x}} \log p_t(\mathbf{x} \mid \mathbf{a}) = \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) + \nabla_{\mathbf{x}} \log p_t(\mathbf{a} \mid \mathbf{x})$$
将得分函数转化为人工温度缩放分布 $p_t^\gamma(\mathbf{a} \mid \mathbf{x}) \propto (p_t(\mathbf{a} \mid \mathbf{x}))^s$，修正后的引导得分为：
$$\nabla_{\mathbf{x}} \log \tilde{p}_t(\mathbf{x} \mid \mathbf{a}) = \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) + s \cdot \nabla_{\mathbf{x}} \log p_t(\mathbf{a} \mid \mathbf{x}) = (1 - s) \nabla_{\mathbf{x}} \log p_t(\mathbf{x}) + s \nabla_{\mathbf{x}} \log p_t(\mathbf{x} \mid \mathbf{a})$$
利用 $\nabla_{\mathbf{x}} \log p_t \propto -\boldsymbol{\epsilon}_\theta$，代入即严格证得 CFG 线性外推公式。
</details>

---

## 9.5.4 纯底层 PyTorch 代码实现：动作条件驾驶世界模型预测引擎

下面我们使用纯底层 PyTorch 算子实现一个轻量级的多相机动作条件驾驶世界模型预测引擎，包含历史状态特征融合、动作前向注入与多视角未来潜在特征预测。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionConditionedPredictor(nn.Module):
    """
    动作条件时空预测器 (Action-Conditioned World Dynamics)
    接收历史环视特征与自车物理动作，自回归推演下一时刻的隐状态
    """
    def __init__(self, num_views: int = 6, feat_dim: int = 64, action_dim: int = 2):
        super().__init__()
        self.num_views = num_views
        self.feat_dim = feat_dim

        # 动作特征投影层
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim)
        )

        # 环视多视角交叉融合自注意力层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=4, dim_feedforward=feat_dim * 2, batch_first=True
        )
        self.spatiotemporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 多视角未来特征预测头
        self.future_head = nn.Linear(feat_dim, feat_dim)

    def forward(self, current_views_feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        :param current_views_feat: (B, num_views, feat_dim) 当前时刻 6 相机特征
        :param action: (B, action_dim) 转向与车速控制量
        :return: (B, num_views, feat_dim) 预测的下一时刻 6 相机特征
        """
        B = current_views_feat.size(0)

        # 1. 将动作编码为条件 Token 并与多视角序列拼接
        act_token = self.action_mlp(action).unsqueeze(1) # (B, 1, feat_dim)
        seq = torch.cat([act_token, current_views_feat], dim=1) # (B, 1 + num_views, feat_dim)

        # 2. 多视角时空自注意力交互
        fused_seq = self.spatiotemporal_transformer(seq)

        # 3. 提取视角序列并预测未来演变
        predicted_views = self.future_head(fused_seq[:, 1:, :]) # (B, num_views, feat_dim)
        return predicted_views

class DrivingWorldModel(nn.Module):
    """
    轻量级端到端自动驾驶世界模型包装器
    """
    def __init__(self, num_views: int = 6, feat_dim: int = 64, action_dim: int = 2):
        super().__init__()
        self.dynamics = ActionConditionedPredictor(
            num_views=num_views, feat_dim=feat_dim, action_dim=action_dim
        )

    def rollout_horizon(
        self, initial_views: torch.Tensor, action_sequence: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        在内心梦境中根据动作序列向前自回归推演多步
        :param initial_views: (B, 6, feat_dim)
        :param action_sequence: (B, horizon, action_dim)
        :return: list of predicted features for each step
        """
        horizon = action_sequence.size(1)
        predictions = []
        curr_feat = initial_views

        for t in range(horizon):
            act_t = action_sequence[:, t, :]
            curr_feat = self.dynamics(curr_feat, act_t)
            predictions.append(curr_feat)

        return predictions

# ===================================================================
# 单元测试：反事实动作干预推演与梯度反传校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 2
    num_views = 6
    feat_dim = 64
    action_dim = 2
    horizon = 5

    model = DrivingWorldModel(num_views=num_views, feat_dim=feat_dim, action_dim=action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    init_views = torch.randn(batch_size, num_views, feat_dim)
    actions = torch.randn(batch_size, horizon, action_dim)

    # 1. 开展多步闭环推演
    pred_seq = model.rollout_horizon(init_views, actions)

    # 2. 计算多步累积均方差损失
    target_future = torch.randn(batch_size, num_views, feat_dim)
    loss = F.mse_loss(pred_seq[-1], target_future)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"[Driving WM Test] 推演时间步数: {len(pred_seq)}")
    print(f"[Driving WM Test] 单步预测张量形状: {pred_seq[0].shape}")
    print(f"[Driving WM Test] 5 步长程预测损失: {loss.item():.4f}")

    assert len(pred_seq) == horizon, "推演步数不匹配！"
    assert pred_seq[0].shape == (batch_size, num_views, feat_dim), "多视角输出形状不符！"
    assert not torch.isnan(loss), "训练损失出现 NaN！"
    print("✓ 多视角动作条件驾驶世界模型预测引擎单测全部通过！")
```

---

## 9.5.5 本节小结

回顾本节内容，我们建立了自动驾驶世界模型在复杂交通场景下的完整预测体系：
1. **反事实推演的价值**：打破真实世界长尾数据采集瓶颈，使自动驾驶系统能够在完全安全的虚拟梦境中进行千万种极限工况的“假设演练”；
2. **多视角环视自洽性**：通过时序自回归链式分解与相机重叠区几何重投影对齐，确保了生成物理空间的全局一致性；
3. **动作引导控制流**：利用无分类器引导（CFG）放大控制信号的影响权重，实现了精准服从转向、油门与刹车的高保真未来推演。
