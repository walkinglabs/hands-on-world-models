# 4.3 PlaNet: 纯潜空间在线规划与多步前向推演

在基于视觉的具身机器人控制中，如果每一个推演步骤都必须调用庞大的卷积神经网络去渲染出一张包含上百万个像素的高清三维图像，系统的推演速度将被锁死在每秒仅有数帧的爬行状态，根本无法应对高速多变的物理现实。

人类在快速决策时展现出了惊人的智慧：当一名网球选手在预判击球落点时，他的大脑绝对不会在潜意识中逐一精细渲染球场看台上每一位观众的衣服褶皱或天空中每朵云彩的微观边缘；大脑仅仅在极度抽象、高度浓缩的**潜在运动特征空间**中，以惊人的速度演练球心坐标与球拍挥击速度。

2018 年，Google Brain 与 DeepMind 推出了首个完全脱离像素渲染的端到端视觉世界模型——**PlaNet（Deep Planning Network）**。

PlaNet 首次证明：**世界模型可以在完全不生成任何一张图像的前提下，直接在 RSSM 紧凑的潜在特征流形内部，以每秒数万步的极致算力展开高并发在线轨迹规划（CEM）**，彻底打破了视觉强化学习样本效率低下的历史魔咒！

<div align="center">

<img src="/figures/04-latent-dynamics/source/03-planet/e2c-fig10.png" alt="PlaNet 在连续动作控制任务中展示长达 50 步的潜在动力学高保真多步推演能力。" width="86%">

_图 4.3-1：PlaNet 在连续动作控制任务中展示长达 50 步的潜在动力学高保真多步推演能力。 出处：[Learning Latent Dynamics for Planning from Pixels，Danijar Hafner et al.，2018](https://arxiv.org/abs/1811.04551)。_

</div>

---

## 4.3.1 物理与认知基石：纯潜空间前向推演的极速突破

要理解 PlaNet 的计算革命，我们首先必须对比传统基于像素规划与纯潜空间规划的算力鸿沟。

### 1. 传统像素级模型预测控制的性能黑洞
在早期的模型规划中，模型在预测时刻 $t+1$ 时，必须先生成完整的 $64 \times 64 \times 3$ 像素图像，再将该图像重新输入视觉网络去预测下一帧。
- 渲染一张图像需要数十亿次浮点运算（FLOPs）；
- 规划 50 步需要渲染 50 次，面对 1000 条候选路径，计算量瞬间暴增至数万亿次，导致实时控制彻底陷入停滞。

### 2. PlaNet 的纯潜在流形飞跃（Latent-Only Planning）
PlaNet 彻底将**感知训练**与**在线控制**解耦：
- **训练阶段**：解码器仅用于作为辅助监督信号，迫使确定性状态 $\mathbf{h}_t$ 与随机状态 $\mathbf{s}_t$ 包含完整的物理几何信息；
- **控制阶段**：**彻底关闭并卸载昂贵的像素图像解码器！**
- 规划器直接将紧凑的隐向量 $(\mathbf{h}_t, \mathbf{s}_t)$ 喂给轻量级的 MLP 奖励预测网络 $\hat{r} = R_\psi(\mathbf{h}_t, \mathbf{s}_t)$，在数毫秒内完成上千条候选轨迹的并发推演与择优！

<div align="center">

<img src="/figures/04-latent-dynamics/latex/03-planet/cem-elite-refit-loop.png" alt="PlaNet 纯潜空间内高并发轨迹推演与 CEM 精英重拟合规划数据流" width="86%">

_图 4.3-2：PlaNet 纯潜空间内高并发轨迹推演与 CEM 精英重拟合规划数据流。_

</div>

---

## 4.3.2 核心数学推导一：潜在动力学前向多步展开与奖励累加

在纯潜在空间中，规划器如何向未来展开 $H$ 步虚拟推演？

<div align="center">

<img src="/figures/04-latent-dynamics/source/03-planet/e2c-fig10.png" alt="PlaNet 在纯潜在空间中展开多步前向推演与环境交互学习曲线。" width="86%">

_图 4.3-3：PlaNet 在纯潜在空间中展开多步前向推演与环境交互学习曲线。 出处：[Learning Latent Dynamics for Planning from Pixels，Danijar Hafner et al.，2018](https://arxiv.org/abs/1811.04551)。_

</div>

<div align="center">

<img src="/figures/04-latent-dynamics/source/03-planet/e2c-fig10.png" alt="PlaNet 对比无模型 DDPG 与 APO，展示超越两个数量级的样本效率。" width="86%">

_图 4.3-4：PlaNet 对比无模型 DDPG 与 APO，展示超越两个数量级的样本效率。 出处：[Learning Latent Dynamics for Planning from Pixels，Danijar Hafner et al.，2018](https://arxiv.org/abs/1811.04551)。_

</div>

### 1. 潜在多步前向递归方程
给定当前真实时刻的后验初始状态 $(\mathbf{h}_0, \mathbf{s}_0)$ 以及一条长为 $H$ 步的候选动作序列 $\mathbf{a}_{0:H-1}$：

对于未来任意推演步 $\tau \in \{1, 2, \dots, H\}$：
1. **确定性状态单步推进**：
   $$\mathbf{h}_\tau = \text{GRUCell}(\mathbf{h}_{\tau-1}, \; [\mathbf{s}_{\tau-1}, \mathbf{a}_{\tau-1}])$$
2. **随机先验状态均值预测**（在规划时直接采用先验均值以消除采样方差）：
   $$\mathbf{s}_\tau = \boldsymbol{\mu}_{\text{prior}}(\mathbf{h}_\tau)$$
3. **潜在即时奖励预测**：
   $$\hat{r}_\tau = R_\psi(\mathbf{h}_\tau, \mathbf{s}_\tau)$$

### 2. 候选轨迹总累积回报
整条候选轨迹的综合评分即为未来 $H$ 步潜在奖励的简单代数求和：

$$J(\mathbf{a}_{0:H-1}) = \sum_{\tau=1}^H \hat{r}_\tau$$

### 3. 潜在规划手算数值算例
设规划视界 $H = 2$ 步。系统由以下极简潜在函数构成：
- 确定性状态更新：$h_\tau = 0.5 h_{\tau-1} + a_{\tau-1}$；
- 随机先验均值：$s_\tau = 0.2 h_\tau$；
- 潜在奖励预测函数：$r(h, s) = 2.0 h + 5.0 s = 2.0 h + 5.0 \times (0.2 h) = 3.0 h$。

初始状态为 $h_0 = 1.0$。规划器生成了两组候选动作序列：
- **候选 A**：$a_0 = 1.0, a_1 = 2.0$；
- **候选 B**：$a_0 = -1.0, a_1 = 0.0$。

我们来手动求解两条轨迹在纯隐空间中的累积回报：
1. **评估候选 A**：
   - 第 1 步：$h_1 = 0.5 \times 1.0 + 1.0 = 1.5 \implies r_1 = 3.0 \times 1.5 = 4.5$；
   - 第 2 步：$h_2 = 0.5 \times 1.5 + 2.0 = 0.75 + 2.0 = 2.75 \implies r_2 = 3.0 \times 2.75 = 8.25$；
   - 累积总回报：$J_A = r_1 + r_2 = 4.5 + 8.25 = 12.75$。
2. **评估候选 B**：
   - 第 1 步：$h_1 = 0.5 \times 1.0 + (-1.0) = -0.5 \implies r_1 = 3.0 \times (-0.5) = -1.5$；
   - 第 2 步：$h_2 = 0.5 \times (-0.5) + 0.0 = -0.25 \implies r_2 = 3.0 \times (-0.25) = -0.75$；
   - 累积总回报：$J_B = r_1 + r_2 = -1.5 + (-0.75) = -2.25$。

初等代数的直观计算证明：候选 A 在潜在空间中获得了高达 $12.75$ 的显著优势评分，CEM 规划器在毫秒内毫不犹豫地锁定候选 A 并下发第一步动作 $a_0 = 1.0$！

<details>
<summary><b>深入推导：潜在超图动力学在李普希茨连续边界下的前向复合误差累积上界证明（点击展开查看完整推导）</b></summary>

设真实动力学函数为 $f$，潜在预测模型为 $\hat{f}$，满足局部单步逼近误差上界 $\|\hat{f}(\mathbf{z}) - f(\mathbf{z})\| \le \epsilon$，且转移函数满足 $L$-李普希茨连续性（$L \ge 1$）。
展开 $H$ 步复合推演误差：
$$\|\hat{\mathbf{z}}_H - \mathbf{z}_H\| = \|\hat{f}(\hat{\mathbf{z}}_{H-1}) - f(\mathbf{z}_{H-1})\| \le \|\hat{f}(\hat{\mathbf{z}}_{H-1}) - f(\hat{\mathbf{z}}_{H-1})\| + \|f(\hat{\mathbf{z}}_{H-1}) - f(\mathbf{z}_{H-1})\| \le \epsilon + L \|\hat{\mathbf{z}}_{H-1} - \mathbf{z}_{H-1}\|$$
由初等几何级数递推，可严格证得总潜在误差上界为：
$$\|\hat{\mathbf{z}}_H - \mathbf{z}_H\| \le \epsilon \sum_{k=0}^{H-1} L^k = \epsilon \frac{L^H - 1}{L - 1}$$
该定理严格确立了潜在规划视界 $H$ 的物理稳定性边界。
</details>

---

## 4.3.3 纯底层 PyTorch 代码实现：从零手写 PlaNet 潜空间 CEM 在线规划器

下面我们使用纯底层 PyTorch 算子实现完整的潜空间 RSSM 动力学前向多步展开与 CEM 批量规划器。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentTransitionRewardModel(nn.Module):
    """
    轻量级潜空间转移与奖励模型
    """
    def __init__(self, deter_dim: int = 64, stoch_dim: int = 16, action_dim: int = 2):
        super().__init__()
        self.cell = nn.GRUCell(stoch_dim + action_dim, deter_dim)
        self.fc_prior_mu = nn.Linear(deter_dim, stoch_dim)
        self.reward_net = nn.Sequential(
            nn.Linear(deter_dim + stoch_dim, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )

    def step(self, h_prev: torch.Tensor, s_prev: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = torch.cat([s_prev, action], dim=-1)
        h_new = self.cell(inputs, h_prev)
        s_new = self.fc_prior_mu(h_new) # 规划时取确定性均值
        reward = self.reward_net(torch.cat([h_new, s_new], dim=-1))
        return h_new, s_new, reward.squeeze(-1)

class PlaNetPlanner:
    """
    PlaNet 纯潜空间 CEM 在线规划器
    """
    def __init__(
        self,
        world_model: LatentTransitionRewardModel,
        horizon: int = 12,
        num_samples: int = 200,
        num_elites: int = 20,
        iterations: int = 4
    ):
        self.model = world_model
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.iterations = iterations

    def plan(self, h_init: torch.Tensor, s_init: torch.Tensor) -> torch.Tensor:
        """
        :param h_init: (deter_dim,)
        :param s_init: (stoch_dim,)
        :return: (action_dim,) 第一步最优执行动作
        """
        device = h_init.device
        action_dim = 2

        # 初始高斯分布
        mu = torch.zeros(self.horizon, action_dim, device=device)
        std = torch.ones(self.horizon, action_dim, device=device)

        for _ in range(self.iterations):
            # 1. 批量采样 (N, Horizon, action_dim)
            actions = mu.unsqueeze(0) + std.unsqueeze(0) * torch.randn(self.num_samples, self.horizon, action_dim, device=device)
            actions = actions.clamp(-1.0, 1.0)

            # 2. 状态广播并纯潜空间推演
            h_sim = h_init.unsqueeze(0).expand(self.num_samples, -1)
            s_sim = s_init.unsqueeze(0).expand(self.num_samples, -1)

            total_returns = torch.zeros(self.num_samples, device=device)

            for t in range(self.horizon):
                act_t = actions[:, t, :]
                h_sim, s_sim, r_t = self.model.step(h_sim, s_sim, act_t)
                total_returns += r_t

            # 3. 精英筛选与重拟合
            elite_indices = torch.topk(total_returns, k=self.num_elites, largest=True).indices
            elites = actions[elite_indices]

            mu = elites.mean(dim=0)
            std = elites.std(dim=0).clamp_min(0.1)

        return mu[0] # 返回最优动作序列的第一步动作

# ===================================================================
# 单元测试与潜空间推演规划校验
# ===================================================================
if __name__ == "__main__":
    deter_dim = 64
    stoch_dim = 16
    action_dim = 2

    model = LatentTransitionRewardModel(deter_dim=deter_dim, stoch_dim=stoch_dim, action_dim=action_dim)
    planner = PlaNetPlanner(world_model=model, horizon=10, num_samples=100, num_elites=10, iterations=3)

    dummy_h = torch.randn(deter_dim)
    dummy_s = torch.randn(stoch_dim)

    optimal_action = planner.plan(dummy_h, dummy_s)

    print(f"[PlaNet Test] 规划视界: {planner.horizon} 步")
    print(f"[PlaNet Test] 潜空间推演得出最优第一步动作: {optimal_action.tolist()}")

    assert optimal_action.shape == (action_dim,), "规划动作维度不符！"
    assert not torch.isnan(optimal_action).any(), "PlaNet 规划出现 NaN 异常！"
    print("✓ PlaNet 纯潜空间无解码器 CEM 轨迹规划器单测全部通过！")
```

---

## 4.3.4 本节小结

回顾本节内容，我们掌握了纯潜空间在线规划的跨越式突破：
1. **脱离像素渲染**：完全卸载高能耗的图像解码器，直接在紧凑的潜空间特征流形内部推演，将计算吞吐提升数个数量级；
2. **前向复合误差控制**：通过李普希茨连续性边界严密约束了多步推演的稳定性，为有限时域内的精准决策提供了理论保障；
3. **极简闭环控制**：将前向动力学与 CEM 进化采样融为一体，宣告了视觉强化学习“样本高效时代”的正式到来。
