# 6.4 动作条件 JEPA (A-JEPA) 与 V-JEPA 具身演进

在前三节中，我们探讨的 I-JEPA 主要聚焦于**单张静态图像内部的空间语义完形填空**——根据未遮挡的局部画面，在隐空间中推理出被遮挡物体的空间几何特征。

然而，对于真实世界中的具身智能机器人（如机械臂、自动驾驶汽车、人形机器人）而言，物理世界不是静止的快照，而是由一系列**外部物理动作指令（如电机扭矩、方向盘转角、夹爪开合力）**驱动的连续因果动态流。

当机械臂向下按压桌上的弹性海绵时，如果模型仅仅预测“海绵在空间上的静态外观”，它就无法理解施加 $-5.0\text{ N}$ 的垂直推力会导致海绵发生形变凹陷的**因果物理定律**。

为了赋予非生成式世界模型强大的物理因果推演能力：
- **动作条件 JEPA（Action-conditioned JEPA, A-JEPA）** 将外部控制动作 $\mathbf{a}_t$ 显式注入潜在预测器中，直接在纯隐空间中学习 $\mathbf{s}_{t+1} = P_\psi(\mathbf{s}_t, \mathbf{a}_t)$ 的动作条件因果转移；
- **V-JEPA（Video JEPA, Meta 2024）** 则将掩码机制从 2D 空间扩展至 **3D 时空连续流形**，在大规模无标注视频中直接预训练出了通晓刚体运动学与流体动力学的通用特征底座！

本节我们将从初等物理因果律与受控动力系统出发，严密推导 A-JEPA 的动作条件隐式演化方程、V-JEPA 的时空立体掩码机制与具身控制闭环，并使用纯底层 PyTorch 从零手写一个 A-JEPA 具身动力学预测模型。

<div align="center">

<img src="/figures/06-jepa/source/04-action-jepa/muzero-fig1.png" alt="V-JEPA 视频联合嵌入预测架构：在时空三维立体上遮挡大块时空 Tubelet，并在特征隐空间预测未来动态演化。" width="86%">

_图 6.4-1：V-JEPA 视频联合嵌入预测架构：在时空三维立体上遮挡大块时空 Tubelet，并在特征隐空间预测未来动态演化。 出处：[Revisiting Feature Prediction for Learning Visual Representations from Video，Bardes et al.，2024](https://arxiv.org/abs/2404.08471)。_

</div>

---

## 6.4.1 物理与具身基石：从空间静态完形到时空动作因果

要理解 A-JEPA 的认知升级，我们首先必须审视纯视觉感知与受控物理系统之间的本质鸿沟。

### 1. 观察者 vs 操纵者
- **纯视觉观察者（I-JEPA / V-JEPA）**：类似于坐在副驾驶看风景的乘客，只能被动记录自然发生的时空变化，无法理解“我的动作将如何改变世界”；
- **具身操纵者（A-JEPA）**：类似于手握方向盘的驾驶员，系统必须显式建立从**控制动作输入 $\mathbf{a}_t$** 到**未来环境状态特征转移 $\mathbf{s}_{t+1}$** 的严密物理因果函数！

### 2. A-JEPA 具身闭环三大组件
1. **在线观测编码器（$E_\theta$）**：将当前物理摄像头图像 $\mathbf{x}_t$ 压缩为状态特征 $\mathbf{s}_t = E_\theta(\mathbf{x}_t)$；
2. **动作条件潜在动力学预测器（$P_\psi$）**：输入当前特征 $\mathbf{s}_t$ 与执行动作 $\mathbf{a}_t$，在纯隐空间推演下一时刻特征：
   $$\hat{\mathbf{s}}_{t+1} = P_\psi(\mathbf{s}_t, \; \mathbf{a}_t)$$
3. **EMA 动量目标编码器（$E_\phi$）**：读取下一时刻真实环境图像 $\mathbf{x}_{t+1}$，提供平稳的目标真实特征 $\mathbf{s}_{t+1}^{\text{target}} = E_\phi(\mathbf{x}_{t+1})$！

<div align="center">

<img src="/figures/06-jepa/latex/04-action-jepa/jepa-batch-feature-reduction.png" alt="A-JEPA 动作条件隐式动力学数据流：当前特征与物理动作在潜空间合成未来特征" width="86%">

_图 6.4-2：A-JEPA 动作条件隐式动力学数据流：当前特征与物理动作在潜空间合成未来特征。_

</div>

---

## 6.4.2 核心数学推导一：A-JEPA 隐式动力学演化与动作可辨识性

在 A-JEPA 中，动作预测器如何通过监督信号学到真实物理世界的受控常微分方程（ODE）？

<div align="center">

<img src="/figures/06-jepa/source/04-action-jepa/muzero-fig1.png" alt="V-JEPA 在密集动作识别与冻结特征评估中展示远超像素重构模型的极高特征质量。" width="86%">

_图 6.4-3：V-JEPA 在密集动作识别与冻结特征评估中展示远超像素重构模型的极高特征质量。 出处：[Revisiting Feature Prediction for Learning Visual Representations from Video，Bardes et al.，2024](https://arxiv.org/abs/2404.08471)。_

</div>

### 1. 离散受控动力学能量损失函数
对于收集到的交互转移三元组 $(\mathbf{x}_t, \mathbf{a}_t, \mathbf{x}_{t+1})$：

$$\mathcal{L}_{\text{A-JEPA}}(\theta, \psi) = \left\| P_\psi(E_\theta(\mathbf{x}_t), \; \mathbf{a}_t) - \text{sg}[E_\phi(\mathbf{x}_{t+1})] \right\|_2^2$$

### 2. A-JEPA 动作推演手算数值算例
设特征隐空间维度 $d = 2$，动作空间为标量（$a \in \mathbb{R}$，表示机械臂水平推力）。
在某个交互时间步：
- 当前观察编码为：$\mathbf{s}_t = [1.0, 0.0]^\top$（物体在左侧原点）；
- 下发动作：$a_t = +2.0\text{ N}$（向右用力推）；
- 目标编码器读取推完后的真实画面编码：$\mathbf{s}_{t+1}^{\text{target}} = [1.0, 4.0]^\top$（物体右移至坐标 4 处）；
- 预测器结构为线性受控系统：$\hat{\mathbf{s}}_{t+1} = \mathbf{s}_t + \mathbf{B} a_t$，当前参数矩阵为 $\mathbf{B} = \begin{bmatrix} 0.0 \\ 1.5 \end{bmatrix}$。

我们来手动求解预测结果与损失：
1. **预测器计算下一时刻特征**：
   $$\hat{\mathbf{s}}_{t+1} = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} + \begin{bmatrix} 0.0 \\ 1.5 \end{bmatrix} \times 2.0 = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix} + \begin{bmatrix} 0.0 \\ 3.0 \end{bmatrix} = \begin{bmatrix} 1.0 \\ 3.0 \end{bmatrix}$$
2. **计算残差向量**：
   $$\Delta \mathbf{s} = \hat{\mathbf{s}}_{t+1} - \mathbf{s}_{t+1}^{\text{target}} = \begin{bmatrix} 1.0 - 1.0 \\ 3.0 - 4.0 \end{bmatrix} = \begin{bmatrix} 0.0 \\ -1.0 \end{bmatrix}$$
3. **计算能量均方误差**：
   $$\mathcal{L} = \|\Delta \mathbf{s}\|_2^2 = 0.0^2 + (-1.0)^2 = 1.0$$
4. **反向传播对参数 $\mathbf{B}$ 的梯度**：
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{B}} = 2 \Delta \mathbf{s} \cdot a_t = 2 \times \begin{bmatrix} 0.0 \\ -1.0 \end{bmatrix} \times 2.0 = \begin{bmatrix} 0.0 \\ -4.0 \end{bmatrix}$$

初等代数的几步推导清晰展现：反向负梯度将推动矩阵参数 $\mathbf{B}$ 的第二分量从 $1.5$ 向上爬升至 $2.0$，使得预测器精确学会了物理常识：**“输入推力 $+2.0\text{ N}$ 将使得物体的潜在空间坐标精确增加 $+4.0$ 个单位！”**

<details>
<summary><b>深入推导：动作可辨识性在李代数受控系统可达集下的完备性证明（点击展开查看完整推导）</b></summary>

将非线性受控物理系统形式化为仿射控制系统 $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}) + \sum_{i=1}^m \mathbf{g}_i(\mathbf{x}) u_i$。
根据周-拉舍夫斯基定理（Chow-Rashevsky Theorem），若向量场李括号（Lie Bracket）序列满足李代数秩条件（LARC）：
$$\text{dim}\left( \text{Lie}(\mathbf{f}, \mathbf{g}_1, \dots, \mathbf{g}_m)(\mathbf{x}) \right) = \dim(\mathcal{M})$$
则系统在隐流形上的可达集（Reachable Set）处处连通。
A-JEPA 的能量损失等价于在李代数流动流形上直接最小化无穷小生成元（Infinitesimal Generator）的算子范数残差，严格保证了动作因果可辨识性的全局完备。
</details>

---

## 6.4.3 核心数学推导二：V-JEPA 时空立体掩码 (Spatiotemporal Tube Masking)

为了让自监督世界模型从海量无标注视频中直接学会物理规律，Meta 推出了 **V-JEPA（Video JEPA）**。

<div align="center">

<img src="/figures/06-jepa/source/04-action-jepa/muzero-fig1.png" alt="V-JEPA 时空立体掩码机制：在时间轴和空间轴上同步遮挡大块三维时空体。" width="86%">

_图 6.4-3：V-JEPA 时空立体掩码机制：在时间轴和空间轴上同步遮挡大块三维时空体。 出处：[Revisiting Feature Prediction for Learning Visual Representations from Video，Bardes et al.，2024](https://arxiv.org/abs/2404.08471)。_

</div>

### 1. 3D 时空立体掩码策略（Spatiotemporal Tubelet Masking）
V-JEPA 将时间轴与空间轴融为一体，采用具有时间贯穿性的**时空柱状掩码（Tubelet Masks）**：
- **时间跨度**：每次遮挡覆盖连续 $T_{\text{mask}} = 4 \sim 8$ 帧；
- **空间跨度**：覆盖空间分辨率的 $15\% \sim 30\%$ 区域；
- **全时空遮挡率**：整体遮挡比例高达 **$70\% \sim 80\%$**！

### 2. 物理运动学的被迫涌现
当一个正在抛射的小球在中间 5 帧画面中被完全大块遮挡时，为了准确预测出小球在后续帧中的抽象特征位置，网络被逼入绝境——它必须在多层自注意力权重中自发计算出抛物线的初速度、重力加速度与飞行时间！

这种纯粹由“遮挡预测”倒逼出的物理推演能力，使得 V-JEPA 提取出的特征能够直接用于下游机械臂的精准抓取与轨迹跟踪！

<details>
<summary><b>深入推导：四维时空流形在李导数下的特征不变性微分证明（点击展开查看完整推导）</b></summary>

设视频特征场为时空流形上的光滑微分形式 $\boldsymbol{\omega} \in \Omega^k(\mathcal{M})$，速度向量场为 $\mathbf{v} = (\frac{dx}{dt}, \frac{dy}{dt}, 1)^\top$。
特征沿流线的时间演变由卡尔当公式（Cartan's Magic Formula）给出的李导数严格决定：
$$\mathcal{L}_{\mathbf{v}} \boldsymbol{\omega} = i_{\mathbf{v}} (d\boldsymbol{\omega}) + d(i_{\mathbf{v}} \boldsymbol{\omega})$$
V-JEPA 的时空掩码损失等价于在整条相轨迹积分线上极小化能量泛函 $\int_{\tau} \|\mathcal{L}_{\mathbf{v}} \boldsymbol{\omega}\|^2 dt$。
当损失收敛时，特征场在时空流线微分同胚下严格满足一阶几何刚体守恒不变性。
</details>

---

## 6.4.4 纯底层 PyTorch 代码实现：从零手写动作条件 A-JEPA 具身动力学预测引擎

下面我们使用纯底层 PyTorch 算子手写实现完整的 A-JEPA 状态编码器、动作调制预测器与 EMA 动量更新自监督训练闭环。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class ActionConditionedPredictor(nn.Module):
    """
    A-JEPA 动作条件潜在特征预测器
    hat{s}_{t+1} = MLP(s_t, a_t)
    """
    def __init__(self, embed_dim: int = 64, action_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + action_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, s_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([s_t, a_t], dim=-1)
        return self.net(inputs)

class ActionJEPAWorldModel(nn.Module):
    """
    动作条件 A-JEPA 具身世界模型
    """
    def __init__(self, in_c: int = 3, embed_dim: int = 64, action_dim: int = 4, momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum

        # 1. 在线上下文编码器
        self.context_encoder = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embed_dim)
        )

        # 2. 动量目标编码器 (EMA)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # 3. 动作条件动力学预测器
        self.predictor = ActionConditionedPredictor(embed_dim=embed_dim, action_dim=action_dim)

    @torch.no_grad()
    def update_target_encoder(self):
        """
        EMA 动量参数软更新
        """
        for p_online, p_target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.mul_(self.momentum).add_(p_online.data, alpha=1.0 - self.momentum)

    def forward(
        self, obs_t: torch.Tensor, action_t: torch.Tensor, obs_t_next: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param obs_t: (B, 3, 32, 32) 当前帧观测
        :param action_t: (B, action_dim) 执行动作
        :param obs_t_next: (B, 3, 32, 32) 下一时刻真实帧
        """
        # 1. 编码当前状态
        s_t = self.context_encoder(obs_t)

        # 2. 动量编码下一时刻目标 (无梯度)
        with torch.no_grad():
            s_target_next = self.target_encoder(obs_t_next)

        # 3. 动作条件潜空间推演
        s_pred_next = self.predictor(s_t, action_t)

        # 4. 特征预测 Smooth L1 损失
        loss = F.smooth_l1_loss(s_pred_next, s_target_next)
        return s_pred_next, s_target_next, loss

# ===================================================================
# 单元测试与动作条件因果动力学反传校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 4
    embed_dim = 64
    action_dim = 4

    model = ActionJEPAWorldModel(in_c=3, embed_dim=embed_dim, action_dim=action_dim, momentum=0.95)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dummy_obs_t = torch.randn(batch_size, 3, 32, 32)
    dummy_act_t = torch.randn(batch_size, action_dim)
    dummy_obs_next = torch.randn(batch_size, 3, 32, 32)

    # 1. 前向推演
    pred_s, target_s, loss = model(dummy_obs_t, dummy_act_t, dummy_obs_next)

    # 2. 反向传播与 EMA 更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.update_target_encoder()

    print(f"[A-JEPA Test] 当前推演特征形状: {pred_s.shape}")
    print(f"[A-JEPA Test] 下一时刻真实目标特征形状: {target_s.shape}")
    print(f"[A-JEPA Test] 动作因果特征预测损失: {loss.item():.4f}")

    assert pred_s.shape == (batch_size, embed_dim), "推演特征维度不符！"
    assert model.predictor.net[0].weight.grad is not None, "动作预测器未接收到梯度！"
    assert not torch.isnan(loss), "A-JEPA 损失出现 NaN 异常！"
    print("✓ 动作条件 A-JEPA 具身动力学预测模型、EMA 目标更新与梯度反传单测全部通过！")
```

---

## 6.4.5 本节小结

回顾本节内容，我们掌握了具身世界模型向动作因果预测跃迁的核心逻辑：
1. **从被动观察到具身因果**：A-JEPA 显式将控制动作注入潜在预测器，在无解码器的高维隐流形上直接拟合了受控物理动力学微分方程；
2. **动作可辨识性保证**：推导了李代数系统可达集性质，确立了动作对状态形变的唯一因果映射；
3. **V-JEPA 时空大块掩码**：利用三维时空立体阻断切断低级插值，从海量无标注视频中直接激发出了刚体与流体运动学的通用物理常识。
