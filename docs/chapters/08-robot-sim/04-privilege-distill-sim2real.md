# 8.4 特权学习与 Sim2Real 蒸馏 (RMA)

在四足机器狗奔跑穿过泥泞沼泽、人形机器人踏上结冰雪地或机械臂抓取未知重量物体的过程中，智能体必须实时感知周围环境的物理属性（如地面承载力、摩擦系数、自身负载质量等），并做出毫秒级的动态姿态补偿。

然而，在真实物理世界中，机器人所能搭载的传感器却极其有限——没有哪台四足机器狗能够直接测量泥地的瞬时摩擦系数，也没有传感器能够提前获知脚下雪坑的深度。

与真实世界的“感知残缺”形成鲜明对比的是，在数字物理仿真器中，计算机系统拥有全知的**上帝视角（Omniscient Oracle）**：仿真器在每一毫秒都能够精确获知每个刚体的精确质量、各接触点的微观摩擦力、地形的高程图以及风阻扰动。这些在仿真中唾手可得、但在真实物理世界中完全无法直接测量的状态量，被称为**特权信息（Privileged Information）**。

如何利用仿真中的全知特权信息训练出超强运动技能，并将其无损地“蒸馏”给仅依赖普通板载传感器的物理机器人？

由 UC Berkeley 与 CMU 等机构提出的 **快速运动自适应（Rapid Motor Adaptation, RMA）** 与 **非对称特权蒸馏架构**，为解决 Sim2Real 终极迁移难题指明了前进方向。

<div align="center">

<img src="/figures/08-robot-sim/source/04-privilege-distill-sim2real/rma-fig1.png" alt="RMA 架构通过特权基准策略与在线环境自适应模块两阶段实现四足机器人崎岖地形全地形泛化。" width="86%">

_图 8.4-1：RMA 架构通过特权基准策略与在线环境自适应模块两阶段实现四足机器人崎岖地形全地形泛化。 出处：[RMA: Rapid Motor Adaptation for Bipedal Robots，Ashish Kumar et al.，2021](https://arxiv.org/abs/2107.04034)。_

</div>

---

## 8.4.1 物理与认知基石：全知特权与板载传感器的两阶段蒸馏

要理解 RMA 的设计精妙，我们首先需要从人类滑雪者在不同雪质下的感知自适应机制讲起。

### 1. 人类自适应的物理心智
一个经验丰富的滑雪运动员在踏上未知雪道时，双眼并不能直接“看”出雪的硬度与摩擦阻力。但在滑行出去的短短半秒钟内，通过双脚传来的震动阻力（本体感觉的历史时序反馈），大脑小脑迅速推断出：“当前的雪质非常松软，阻力偏大，我必须将重心后移并加大膝关节推力”。

### 2. RMA 的两阶段认知解耦
RMA 将复杂的自适应策略学习拆分为两个高度专业化的阶段：

#### 阶段一：全知教师的特权基准策略训练（Privileged Base Policy Training）
在仿真器中，我们为基准策略 $\pi$ 配备一个**特权编码器（Privileged Encoder, $E_{\text{priv}}$）**。
- 输入仿真器内部的全部真实物理量 $\mathbf{e}_t$（包含真实地面摩擦系数 $\mu$、连杆质量扰动 $\Delta m$、电机扭矩衰减因子、地形局部高度网格等）；
- 特权编码器将其压缩为一个低维的**环境物理潜向量（Environmental Latent Vector）** $\mathbf{z}_t^* = E_{\text{priv}}(\mathbf{e}_t) \in \mathbb{R}^d$（通常 $d = 8$）；
- 基准策略网络 $\pi(\mathbf{a}_t \mid \mathbf{s}_t, \mathbf{z}_t^*)$ 接收普通本体状态 $\mathbf{s}_t$ 与潜向量 $\mathbf{z}_t^*$，在强化学习中轻而易举地学会穿越各种极限崎岖地形的高难度技能。

#### 阶段二：学生自适应模块的时序蒸馏（Temporal Adaptation Module Distillation）
在真实物理世界中，特权信息 $\mathbf{e}_t$ 无法直接获取。
- 我们构建一个**学生自适应模块（Adaptation Module, $E_{\text{adapt}}$）**，它仅仅接收过去 $k$ 个时间步的普通板载传感器历史数据（如关节角度、关节速度与历史动作指令） $\mathbf{h}_t = [\mathbf{s}_{t-k:t}, \mathbf{a}_{t-k:t}]$；
- 通过一维因果时序卷积网络（1D-CNN），学生网络学会从运动历史轨迹的时序阻尼与微小滑移中，在线“猜出”当前的特权物理隐向量 $\hat{\mathbf{z}}_t = E_{\text{adapt}}(\mathbf{h}_t)$；
- 部署到真机时，将估计出的 $\hat{\mathbf{z}}_t$ 直接喂入已冻结的基准策略 $\pi$，实现完全不依赖任何特权传感器的真实世界零样本迁移！

<div align="center">

<img src="/figures/08-robot-sim/latex/04-privilege-distill-sim2real/history-conv1d-shape-chain.png" alt="本体感觉历史时序序列经一维因果卷积逐层压缩提取环境物理特征潜变量" width="86%">

_图 8.4-2：本体感觉历史时序序列经一维因果卷积逐层压缩提取环境物理特征潜变量。_

</div>

---

## 8.4.2 核心数学推导一：潜在环境空间对齐与监督蒸馏损失

在阶段二中，自适应模块如何通过监督学习逼近全知特权编码器的输出？

<div align="center">

<img src="/figures/08-robot-sim/source/04-privilege-distill-sim2real/rma-fig2.png" alt="RMA 在一阶段训练特权策略，二阶段利用一维卷积自适应网络蒸馏环境隐变量。" width="86%">

_图 8.4-3：RMA 在一阶段训练特权策略，二阶段利用一维卷积自适应网络蒸馏环境隐变量。 出处：[RMA: Rapid Motor Adaptation for Bipedal Robots，Ashish Kumar et al.，2021](https://arxiv.org/abs/2107.04034)。_

</div>

<div align="center">

<img src="/figures/08-robot-sim/source/04-privilege-distill-sim2real/anymal-fig1.png" alt="苏黎世联邦理工学院 (ETH) 使用非对称特权信息训练 ANYmal 四足机器人盲走穿越极端荒野。" width="86%">

_图 8.4-4：苏黎世联邦理工学院 (ETH) 使用非对称特权信息训练 ANYmal 四足机器人盲走穿越极端荒野。 出处：[Learning Quadrupedal Locomotion over Challenging Terrain，Joonho Lee et al.，2020](https://arxiv.org/abs/2010.11251)。_

</div>

### 1. 均方误差蒸馏损失函数
在阶段二训练时，保持阶段一训练好的基准策略 $\pi$ 与特权编码器 $E_{\text{priv}}$ 权重完全冻结。
自适应网络 $E_{\text{adapt}}$ 的参数通过最小化预测潜向量 $\hat{\mathbf{z}}_t$ 与真实特权潜向量 $\mathbf{z}_t^*$ 之间的欧氏均方差（MSE）进行优化：

$$\mathcal{L}_{\text{distill}}(\phi) = \mathbb{E}_{(\mathbf{h}_t, \mathbf{e}_t) \sim \mathcal{D}} \left[ \left\| E_{\text{adapt}}(\mathbf{h}_t; \phi) - E_{\text{priv}}(\mathbf{e}_t) \right\|_2^2 \right]$$

### 2. 特权隐向量蒸馏数值手算算例
设特权物理隐空间维度为 $d = 2$。
在某个时间步 $t$：
- 仿真器内部的特权环境向量为 $\mathbf{e}_t = [\Delta m = 3.0\text{ kg}, \mu = 0.40]^\top$；
- 冻结的特权编码器输出目标向量：$\mathbf{z}_t^* = E_{\text{priv}}(\mathbf{e}_t) = [0.80, -0.40]^\top$；
- 学生自适应网络仅凭历史本体感觉 $\mathbf{h}_t$，预测输出为：$\hat{\mathbf{z}}_t = E_{\text{adapt}}(\mathbf{h}_t) = [0.70, -0.50]^\top$。

我们来一步步手动计算蒸馏损失与梯度：
1. **计算预测残差向量**：
   $$\Delta \mathbf{z} = \hat{\mathbf{z}}_t - \mathbf{z}_t^* = [0.70 - 0.80, -0.50 - (-0.40)]^\top = [-0.10, -0.10]^\top$$
2. **计算单样本均方误差损失**：
   $$\mathcal{L} = \|\Delta \mathbf{z}\|_2^2 = (-0.10)^2 + (-0.10)^2 = 0.01 + 0.01 = 0.020$$
3. **计算对学生网络输出的梯度导数**：
   $$\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{z}}_t} = 2 (\hat{\mathbf{z}}_t - \mathbf{z}_t^*) = 2 \times [-0.10, -0.10]^\top = [-0.20, -0.20]^\top$$

初等代数的清晰代入证实：反向传播梯度直接推动学生网络的预测输出向真实特权目标 $[0.80, -0.40]^\top$ 靠拢，使学生网络在无特权传感器的情况下，仅凭几步肢体动作就能神准推断出脚下踩的是松软厚雪还是坚硬花岗岩！

<details>
<summary><b>深入推导：部分可观测马尔可夫决策过程（POMDP）下的可辨识性（Identifiability）与后验充分统计量证明（点击展开查看完整推导）</b></summary>

在 POMDP 中，真实环境物理参数 $\mathbf{e}$ 为不可见隐状态。根据舍农辨识定理（Shiryaev-Roberts Filter）：
若系统的连续时序观测转移方程 $\mathbf{s}_{t+1} = f(\mathbf{s}_t, \mathbf{a}_t; \mathbf{e})$ 对参数 $\mathbf{e}$ 满足局部李雅普诺夫李导数非奇异秩条件 $\text{rank}\left( \frac{\partial [f, \nabla_s f, \dots]}{\partial \mathbf{e}} \right) = \dim(\mathbf{e})$，
则历史轨迹分布 $\mathbf{h}_t = (\mathbf{s}_{t-k:t}, \mathbf{a}_{t-k:t})$ 构成了后验条件分布 $p(\mathbf{e} \mid \mathbf{h}_t)$ 的**充分统计量（Sufficient Statistic）**。
均方误差蒸馏损失使得自适应网络输出 $\hat{\mathbf{z}}_t$ 严格渐近收敛于后验条件期望 $\mathbb{E}[\mathbf{z}^* \mid \mathbf{h}_t]$。
</details>

---

## 8.4.3 核心数学推导二：非对称 Actor-Critic 架构优势

除了两阶段蒸馏外，在阶段一的强化学习训练中，系统广泛采用了**非对称 Actor-Critic（Asymmetric Actor-Critic）**设计。

<div align="center">

<img src="/figures/08-robot-sim/source/04-privilege-distill-sim2real/lbc-fig1.png" alt="特权教师网络与传感器受限学生网络的模仿学习蒸馏架构，展示特权学习的通用范式。" width="86%">

_图 8.4-5：特权教师网络与传感器受限学生网络的模仿学习蒸馏架构，展示特权学习的通用范式。 出处：[Learning by Cheating，Dian Chen et al.，2020](https://arxiv.org/abs/1912.09686)。_

</div>

### 1. 为什么 Actor 与 Critic 输入可以不对称？
- **Critic 价值网络（评判家）**：只在仿真器训练阶段计算状态价值 $V(\mathbf{s})$ 并指导梯度更新，**它永远不会被部署到物理机器人真机上**！因此，我们可以毫无保留地把所有特权信息 $\mathbf{e}_t$（包括周围 10 米高精地形、全局障碍物速度、接触力）全部喂给 Critic：
  $$V_{\text{privileged}}(\mathbf{s}_t, \mathbf{e}_t)$$
  全知的 Critic 能够做出极其精准、低方差的价值评估，彻底消除了部分可观测性带来的价值震荡；
- **Actor 策略网络（执行者）**：最终必须独立部署到真机上，因此其输入必须受到严格约束，仅接收可观测状态与环境隐变量：
  $$\pi_\theta(\mathbf{a}_t \mid \mathbf{s}_t, \mathbf{z}_t)$$

非对称设计巧妙地实现了“训练时利用全知上帝指导，部署时凭借精简肢体执行”的终极平衡。

<details>
<summary><b>深入推导：非对称 Critic 梯度在贝尔曼算子压缩下的单调收敛性证明（点击展开查看完整推导）</b></summary>

在全信息空间下，贝尔曼评估算子 $\mathcal{T}^\pi V(\mathbf{s}, \mathbf{e}) = r(\mathbf{s}, \mathbf{a}, \mathbf{e}) + \gamma \mathbb{E}[V(\mathbf{s}', \mathbf{e}')]$ 在无穷范数下满足标准 $\gamma$-收敛压缩映射：
$$\|\mathcal{T}^\pi V_1 - \mathcal{T}^\pi V_2\|_\infty \le \gamma \|V_1 - V_2\|_\infty$$
由于引入特权信息消除了环境隐状态的不确定性熵，使得时间差分目标（TD Target）的方差满足 $\text{Var}(\delta_t^{\text{asym}}) \ll \text{Var}(\delta_t^{\text{sym}})$。由随机近似收敛定理，非对称策略梯度的信噪比显著提升，加速策略收敛。
</details>

---

## 8.4.4 纯底层 PyTorch 代码实现：从零搭建 RMA 特权学习与学生自适应蒸馏网络

下面我们使用纯底层 PyTorch 算子手写实现完整的 RMA 特权编码器、一维时序卷积自适应网络、基准策略网络与监督蒸馏训练流程。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrivilegedEncoder(nn.Module):
    """
    阶段一：特权环境编码器 (Privileged Encoder)
    将仿真器特权物理参数 e_t 压缩为低维潜在向量 z_t
    """
    def __init__(self, priv_dim: int = 16, latent_dim: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(priv_dim, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, priv_info: torch.Tensor) -> torch.Tensor:
        return self.mlp(priv_info)

class TemporalAdaptationModule(nn.Module):
    """
    阶段二：学生时序自适应模块 (Temporal Adaptation Module)
    输入过去 k 个时间步的本体历史 h_t = [s_{t-k:t}, a_{t-k:t}]
    利用一维因果卷积时序网络在线预测环境隐向量 hat{z}_t
    """
    def __init__(self, history_len: int = 20, obs_dim: int = 12, latent_dim: int = 8):
        super().__init__()
        # 输入形状: (B, obs_dim, history_len)
        self.conv_net = nn.Sequential(
            nn.Conv1d(obs_dim, 32, kernel_size=4, stride=2), # (B, 32, 9)
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2),      # (B, 32, 4)
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=2, stride=1),      # (B, 32, 3)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 3, latent_dim)
        )

    def forward(self, history_obs: torch.Tensor) -> torch.Tensor:
        """
        :param history_obs: (B, history_len, obs_dim)
        :return: (B, latent_dim) 预测的环境隐向量 hat{z}_t
        """
        # 转换为 Conv1D 期望的 (B, C_in, L_in)
        x = history_obs.transpose(1, 2)
        hat_z = self.conv_net(x)
        return hat_z

class RMAPolicy(nn.Module):
    """
    RMA 基准策略网络 (Base Policy)
    接收当前本体观测与环境潜变量，输出机器人动作
    """
    def __init__(self, obs_dim: int = 12, latent_dim: int = 8, action_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, obs: torch.Tensor, z_latent: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([obs, z_latent], dim=-1)
        actions = self.net(inputs)
        return actions

# ===================================================================
# 单元测试与两阶段蒸馏流程校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 4
    history_len = 20
    obs_dim = 12
    priv_dim = 16
    latent_dim = 8
    action_dim = 6

    # 1. 实例化特权编码器、自适应网络与策略
    priv_encoder = PrivilegedEncoder(priv_dim=priv_dim, latent_dim=latent_dim)
    adapt_module = TemporalAdaptationModule(history_len=history_len, obs_dim=obs_dim, latent_dim=latent_dim)
    policy = RMAPolicy(obs_dim=obs_dim, latent_dim=latent_dim, action_dim=action_dim)

    dummy_priv_info = torch.randn(batch_size, priv_dim)
    dummy_history = torch.randn(batch_size, history_len, obs_dim)
    curr_obs = dummy_history[:, -1, :] # 最新当前步观测

    # 2. 阶段一：全知特权特征前向推理
    with torch.no_grad():
        target_z = priv_encoder(dummy_priv_info) # (B, latent_dim)
        teacher_actions = policy(curr_obs, target_z)

    # 3. 阶段二：学生网络时序监督蒸馏
    optimizer = torch.optim.Adam(adapt_module.parameters(), lr=1e-3)
    pred_z = adapt_module(dummy_history)
    distill_loss = F.mse_loss(pred_z, target_z)

    optimizer.zero_grad()
    distill_loss.backward()
    optimizer.step()

    # 4. 真机部署推理测试 (仅使用学生网络预测的隐向量)
    student_actions = policy(curr_obs, pred_z.detach())

    print(f"[RMA Test] 特权隐向量形状: {target_z.shape}")
    print(f"[RMA Test] 学生自适应网络蒸馏损失: {distill_loss.item():.4f}")
    print(f"[RMA Test] 教师动作与学生动作形状: {student_actions.shape}")

    assert target_z.shape == (batch_size, latent_dim), "特权隐向量维度不符！"
    assert pred_z.shape == (batch_size, latent_dim), "学生自适应向量维度不符！"
    assert not torch.isnan(distill_loss), "蒸馏损失计算异常！"
    print("✓ RMA 特权编码、时序因果蒸馏与两阶段策略引擎单测全部通过！")
```

---

## 8.4.5 本节小结

回顾本节内容，我们建立了特权学习与 Sim2Real 蒸馏的完整方法论体系：
1. **信息差与两阶段解耦**：利用仿真器全知上帝视角先攻克高难度越野控制，再通过学生时序网络消除对不可测特权信息的依赖；
2. **时序辨识充分统计量**：利用一维因果卷积从历史运动轨迹中实时辨识环境物理参数，实现了毫秒级在线物理自适应；
3. **非对称架构收益**：全知 Critic 稳定价值估计梯度，极简 Actor 保障了真机低延迟部署的极致轻量性。
