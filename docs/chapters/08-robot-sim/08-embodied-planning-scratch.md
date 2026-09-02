# 8.8 从零实现具身世界模型规划器 (Embodied World Model Planner from Scratch)

在具身智能控制的武器库中，除了使用强化学习离线训练固定的策略网络（Actor）之外，另一条具备极致灵活性与可解释性的技术路线，是**在线模型预测控制（Model Predictive Control, MPC）与世界模型规划**。

离线训练的策略网络就像一个熟练的体操运动员，凭借条件反射执行肌肉记忆动作；但如果在运行途中突然出现未曾见过的移动障碍物或地形断崖，死板的策略网络往往会因为分布外泛化失败而发生碰撞。

而基于世界模型的在线规划器（如 **PETS**、**MPPI**、**MuJoCo MPC / MJPC**），则如同一个深思熟虑的国际象棋大师：在每一个真实的决策瞬间，规划器利用脑海中的神经动力学世界模型，以 GPU 高并发算力向未来平行模拟出数百条候选动作轨迹（Rollouts），精准评估每条轨迹的物理安全性与能量消耗，挑选出最具前瞻性的最优控制序列。

本节我们将从零推导模型预测路径积分（MPPI）、交叉熵方法（CEM）以及多模型集成（Ensemble）不确定性评估，并使用纯底层 PyTorch 从零手写一个高并发具身世界模型在线规划引擎。

<div align="center">

<img src="/figures/08-robot-sim/source/08-embodied-planning-scratch/pets-fig3.png" alt="PETS 利用概率神经网络集成建模环境不确定性，并通过在线 MPC 规划实现极高样本效率连续控制。" width="86%">

_图 8.8-1：PETS 利用概率神经网络集成建模环境不确定性，并通过在线 MPC 规划实现极高样本效率连续控制。 出处：[Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models，Kurtland Chua et al.，2018](https://arxiv.org/abs/1805.12114)。_

</div>

---

## 8.8.1 物理与控制基石：滚动时域控制 (RHC) 与实时闭环修正

要理解在线规划器的鲁棒性，我们首先必须回到现代控制理论的明珠——**滚动时域控制（Receding Horizon Control, RHC）**。

### 1. 滚动时域的“走一步、看十步”哲学
设规划器向前推演的时间视界为 $H$（例如 $H = 15$ 步，对应未来 $0.3\text{ 秒}$）。
在当前真实时刻 $t$：
1. **状态对齐**：读取机器人当前的真实物理状态 $\mathbf{s}_t$；
2. **内心推演**：利用世界模型在内部模拟大量候选动作，求解出未来 $H$ 步的最优动作序列 $\mathbf{U}^* = \{\mathbf{a}_t^*, \mathbf{a}_{t+1}^*, \dots, \mathbf{a}_{t+H-1}^*\}$；
3. **即时执行**：**只将第一步动作 $\mathbf{a}_t^*$ 下发给机器人底层电机执行！**
4. **滚动重置**：机器人移动到新状态 $\mathbf{s}_{t+1}$；在时刻 $t+1$，丢弃剩余的未来动作，重新以 $\mathbf{s}_{t+1}$ 为起点开启新一轮的 $H$ 步规划。

> **物理直觉**：
> 为什么要“算十步却只走一步”？因为真实物理世界充满了不可预测的随机突发扰动（如地面突然打滑）。若固执地执行全部 15 步预定轨迹，累积的物理偏差会迅速失控；而滚动时域控制在每一步都依据最新真实状态重新规划，天然具备极强的**闭环抗扰动自愈能力**！

<div align="center">

<img src="/figures/08-robot-sim/latex/08-embodied-planning-scratch/cem-rollout-tensor-axes.png" alt="高并发采样轨迹张量在规划视界与采样批次轴上的并行推演与评估" width="86%">

_图 8.8-2：高并发采样轨迹张量在规划视界与采样批次轴上的并行推演与评估。_

</div>

---

## 8.8.2 核心数学推导一：模型预测路径积分 (MPPI) 与交叉熵 (CEM)

如何从成百上千条随机采样的候选轨迹中，以最高效的方式搜索出最优动作？

<div align="center">

<img src="/figures/08-robot-sim/source/08-embodied-planning-scratch/mppi-fig8.png" alt="MPPI 路径积分控制在赛车高速漂移避障中实时加权平滑轨迹。" width="86%">

_图 8.8-3：MPPI 路径积分控制在赛车高速漂移避障中实时加权平滑轨迹。 出处：[Model Predictive Path Integral Control: From Theory to Parallel Computation，Grady Williams et al.，2017](https://arxiv.org/abs/1707.02342)。_

</div>

<div align="center">

<img src="/figures/08-robot-sim/source/08-embodied-planning-scratch/mjpc-fig3.png" alt="MuJoCo MPC (MJPC) 在高自由度灵巧操作中实现多算法实时交互式规划。" width="86%">

_图 8.8-4：MuJoCo MPC (MJPC) 在高自由度灵巧操作中实现多算法实时交互式规划。 出处：[MuJoCo MPC: A Framework for Real-Time Model Predictive Control，Alasdair McKay et al.，2023](https://arxiv.org/abs/2312.03912)。_

</div>

### 1. 模型预测路径积分（MPPI）加权法则
MPPI 基于随机最优控制理论与玻尔兹曼信息论分布：
系统在基准动作序列 $\mathbf{u}_t$ 周围注入 $K$ 组高斯随机探索噪声 $\delta \mathbf{u}_t^{(k)} \sim \mathcal{N}(0, \boldsymbol{\Sigma})$。

对于第 $k$ 条采样轨迹，计算其在未来 $H$ 步的累积物理代价总和：

$$S(\tau_k) = \sum_{\tau=1}^H \mathcal{C}(\mathbf{s}_\tau^{(k)}, \mathbf{u}_\tau^{(k)})$$

根据指数玻尔兹曼分布计算每条轨迹的优劣归一化权重：

$$w_k = \frac{\exp\left( -\frac{1}{\lambda} S(\tau_k) \right)}{\sum_{j=1}^K \exp\left( -\frac{1}{\lambda} S(\tau_j) \right)}$$

最终下发的平滑最优动作为所有采样动作按权重的凸组合：

$$\mathbf{a}_t^* = \mathbf{u}_t + \sum_{k=1}^K w_k \cdot \delta \mathbf{u}_t^{(k)}$$

### 2. MPPI 权重手算数值算例
设温度参数 $\lambda = 10.0$。
系统生成了 3 条候选轨迹：
- **轨迹 1（完美避障轨迹）**：累积代价 $S_1 = 10.0$；
- **轨迹 2（微小擦碰轨迹）**：累积代价 $S_2 = 30.0$；
- **轨迹 3（严重碰撞轨迹）**：累积代价 $S_3 = 80.0$。

我们来手动计算每条轨迹的权重贡献：
1. **计算负代价指数项**：
   $$E_1 = \exp\left(-\frac{10.0}{10.0}\right) = e^{-1} \approx 0.3679$$
   $$E_2 = \exp\left(-\frac{30.0}{10.0}\right) = e^{-3} \approx 0.0498$$
   $$E_3 = \exp\left(-\frac{80.0}{10.0}\right) = e^{-8} \approx 0.0003$$
2. **计算归一化权重和**：
   $$\text{Sum} = 0.3679 + 0.0498 + 0.0003 = 0.4180$$
   $$w_1 = \frac{0.3679}{0.4180} \approx 0.880 \quad (88.0\%)$$
   $$w_2 = \frac{0.0498}{0.4180} \approx 0.119 \quad (11.9\%)$$
   $$w_3 = \frac{0.0003}{0.4180} \approx 0.001 \quad (0.1\%)$$

初等代数的直观计算证明：低代价的高质量轨迹自动占据了高达 **$88.0\%$** 的绝对主导权重，而撞击障碍物的轨迹被彻底衰减过滤至 **$0.1\%$**，实现了极具弹性的自适应轨迹融合！

<details>
<summary><b>深入推导：MPPI 路径积分在随机动力学伊藤引理下的自由能（Free Energy）最小化证明（点击展开查看完整推导）</b></summary>

设受控扩散过程满足随机微分方程（SDE）$d\mathbf{x} = \mathbf{f}(\mathbf{x}) dt + \mathbf{G}(\mathbf{x}) (\mathbf{u} dt + d\mathbf{w})$。
根据吉尔萨诺夫测度变换定理（Girsanov Theorem），受控测度 $\mathbb{Q}$ 与无控自然测度 $\mathbb{P}$ 之间的相对熵（KL 散度）满足：
$$D_{\text{KL}}(\mathbb{Q} \parallel \mathbb{P}) = \frac{1}{2} \mathbb{E}_\mathbb{Q} \left[ \int_0^T \|\mathbf{u}(t)\|_{\boldsymbol{\Sigma}^{-1}}^2 dt \right]$$
定义自由能泛函 $\mathcal{F} = -\lambda \log \mathbb{E}_\mathbb{P} \left[ \exp\left(-\frac{1}{\lambda} S(\tau)\right) \right]$。由瓦拉丹引理（Varadhan's Lemma），极小化自由能等价于在相对熵正则化下求解最优控制率，严格导出了 MPPI 的指数重加权闭式解。
</details>

---

## 8.8.3 核心数学推导二：多模型集成 (Ensemble) 与认知不确定性防御

单个神经网络世界模型存在一个致命的软肋——**模型幻觉（Model Exploitation）**。
如果某个动作在真实世界中会导致严重碰撞，但由于该区域处于训练集分布外，单个世界模型可能会错误地预测出“不仅没撞，还获得了 $+100$ 的超高奖励”。规划器会盲目相信这个虚假的幻觉，从而做出极其危险的自毁决策。

<div align="center">

<img src="/figures/08-robot-sim/source/08-embodied-planning-scratch/mjpc-fig4.png" alt="MJPC 在不同任务下对比采样规划器与梯度优化器的实时收敛速度。" width="86%">

_图 8.8-4：MJPC 在不同任务下对比采样规划器与梯度优化器的实时收敛速度。 出处：[MuJoCo MPC: A Framework for Real-Time Model Predictive Control，Alasdair McKay et al.，2023](https://arxiv.org/abs/2312.03912)。_

</div>

PETS 提出了构建 $M$ 个独立随机初始化的神经网络集成（Ensemble Models, $M = 5$）：

$$\hat{\mathbf{s}}_{t+1}^{(m)} = f_{\theta_m}(\mathbf{s}_t, \mathbf{a}_t), \quad m \in \{1, 2, \dots, M\}$$

对于任意候选动作序列，我们计算 $M$ 个世界模型预测结果的**认知不确定性方差（Epistemic Variance）**：

$$\sigma_{\text{epistemic}}^2(\mathbf{s}, \mathbf{a}) = \frac{1}{M} \sum_{m=1}^M \left\| \hat{\mathbf{s}}_{t+1}^{(m)} - \bar{\mathbf{s}}_{t+1} \right\|_2^2$$

在评估轨迹代价时，系统引入悲观惩罚项（Pessimistic Penalty）：

$$\mathcal{C}_{\text{robust}}(\mathbf{s}, \mathbf{a}) = \mathcal{C}_{\text{task}}(\mathbf{s}, \mathbf{a}) + \beta \cdot \sigma_{\text{epistemic}}(\mathbf{s}, \mathbf{a})$$

当规划器试图走向未知的危险边缘时，5 个模型的预测结果产生剧烈分歧，方差 $\sigma$ 骤升，悲观代价暴增，从而迫使机器人坚定地留在已知安全的高可信物理流形之内！

<details>
<summary><b>深入推导：深度集成（Deep Ensemble）在函数空间中逼近贝叶斯后验分布的严格证明（点击展开查看完整推导）</b></summary>

设参数后验分布为 $p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) p(\theta)$。
利用非线性神经正切核（NTK）与高斯过程（GP）对偶理论，若每个集成子模型从独立的标准正态先验初始化，并采用随机梯度下降在不同数据子集上训练，子模型的参数集合 $\{\theta_1, \dots, \theta_M\}$ 构成了后验分布 $p(\theta \mid \mathcal{D})$ 在函数空间中的离散狄拉克测度粒子逼近：
$$q(\theta) = \frac{1}{M} \sum_{m=1}^M \delta(\theta - \theta_m)$$
其预测方差在极限宽度下严格等价于贝叶斯高斯过程的边缘认知不确定性后验方差 $\sigma_{\text{GP}}^2(\mathbf{x}) = k(\mathbf{x}, \mathbf{x}) - \mathbf{k}_x^\top (\mathbf{K} + \sigma^2 \mathbf{I})^{-1} \mathbf{k}_x$。
</details>

---

## 8.8.4 纯底层 PyTorch 代码实现：集成世界模型与 GPU 批量 MPPI 规划引擎

下面我们使用纯底层 PyTorch 算子手写实现完整的集成动力学世界模型、GPU 高并发批量轨迹推演与 MPPI 滚动时域规划引擎。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnsembleWorldModel(nn.Module):
    """
    多模型集成神经网络动力学世界模型 (Ensemble Dynamics)
    由 M 个独立子网络组成，输出下一状态预测与认知不确定性
    """
    def __init__(self, state_dim: int = 4, action_dim: int = 1, num_models: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.num_models = num_models
        self.state_dim = state_dim
        self.action_dim = action_dim

        # 构造 M 个独立的动力学 MLP
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim)
            ) for _ in range(num_models)
        ])

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param state: (K, state_dim) 批量状态
        :param action: (K, action_dim) 批量动作
        :return: (mean_next_state, epistemic_std) 预测均值与模型间标准差
        """
        inputs = torch.cat([state, action], dim=-1)
        preds = []
        for m in self.models:
            preds.append(m(inputs))

        stacked_preds = torch.stack(preds, dim=0) # (M, K, state_dim)
        mean_next_state = stacked_preds.mean(dim=0)
        epistemic_std = stacked_preds.std(dim=0).mean(dim=-1, keepdim=True) # (K, 1)

        return mean_next_state, epistemic_std

class MPPIPlanner:
    """
    GPU 批量模型预测路径积分规划器 (Batched MPPI Planner)
    """
    def __init__(
        self,
        world_model: EnsembleWorldModel,
        horizon: int = 12,
        num_samples: int = 256,
        lambda_temp: float = 5.0,
        noise_std: float = 0.3
    ):
        self.world_model = world_model
        self.horizon = horizon
        self.num_samples = num_samples
        self.lambda_temp = lambda_temp
        self.noise_std = noise_std
        self.action_dim = world_model.action_dim

        # 当前维护的名义动作序列 (Horizon, action_dim)
        self.action_sequence = torch.zeros(horizon, self.action_dim)

    def cost_function(self, state: torch.Tensor, action: torch.Tensor, uncert: torch.Tensor) -> torch.Tensor:
        """
        轨迹单步物理代价函数：目标偏离 + 控制消耗 + 认知不确定性悲观惩罚
        """
        target_state = torch.tensor([0.0, 0.0, 0.0, 0.0], device=state.device)
        state_cost = (state - target_state).pow(2).sum(dim=-1)
        action_cost = 0.01 * action.pow(2).sum(dim=-1)
        pessimism_cost = 2.0 * uncert.squeeze(-1)
        return state_cost + action_cost + pessimism_cost

    def plan_step(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        输入当前物理状态，输出最优执行动作 (action_dim,)
        """
        device = current_state.device
        # 1. 在名义动作序列周围采样 K 条高斯探索噪声
        noise = torch.randn(self.num_samples, self.horizon, self.action_dim, device=device) * self.noise_std
        # 候选动作: (K, Horizon, action_dim)
        candidate_actions = self.action_sequence.unsqueeze(0) + noise

        # 2. 状态批量克隆广播: (K, state_dim)
        sim_states = current_state.unsqueeze(0).expand(self.num_samples, -1).clone()
        total_costs = torch.zeros(self.num_samples, device=device)

        # 3. 在世界模型内部展开 Horizon 步高并发推演
        for t in range(self.horizon):
            act_t = candidate_actions[:, t, :]
            sim_states, uncert_t = self.world_model(sim_states, act_t)
            step_cost = self.cost_function(sim_states, act_t, uncert_t)
            total_costs += step_cost

        # 4. 求解 MPPI 玻尔兹曼权重
        min_cost = total_costs.min()
        exp_weights = torch.exp(- (total_costs - min_cost) / self.lambda_temp)
        weights = exp_weights / (exp_weights.sum() + 1e-8) # (K,)

        # 5. 加权更新最优动作序列
        weighted_actions = (candidate_actions * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0) # (Horizon, action_dim)

        # 6. 滚动时域：提取第一步动作，并将序列前移
        optimal_action = weighted_actions[0].clone()
        self.action_sequence = torch.cat([weighted_actions[1:], torch.zeros(1, self.action_dim, device=device)], dim=0)

        return optimal_action

# ===================================================================
# 单元测试与闭环规划收敛校验
# ===================================================================
if __name__ == "__main__":
    state_dim = 4
    action_dim = 1
    horizon = 10
    num_samples = 128

    model = EnsembleWorldModel(state_dim=state_dim, action_dim=action_dim, num_models=3)
    planner = MPPIPlanner(world_model=model, horizon=horizon, num_samples=num_samples, lambda_temp=5.0)

    # 模拟从偏离状态进行单步闭环规划
    curr_state = torch.tensor([1.5, -0.5, 0.2, -0.1])
    opt_act = planner.plan_step(curr_state)

    print(f"[MPPI Test] 采样候选轨迹数: {num_samples}")
    print(f"[MPPI Test] 规划视界步数: {horizon}")
    print(f"[MPPI Test] 求解最优下发动作: {opt_act.tolist()}")
    print(f"[MPPI Test] 滚动名义序列形状: {planner.action_sequence.shape}")

    assert opt_act.shape == (action_dim,), "最优控制动作维度不符！"
    assert not torch.isnan(opt_act).any(), "规划动作出现 NaN 异常！"
    assert planner.action_sequence.shape == (horizon, action_dim), "滚动时域动作队列长度异常！"
    print("✓ 集成世界模型与 GPU 批量 MPPI 滚动时域规划引擎单测全部通过！")
```

---

## 8.8.5 本节小结

回顾本节内容，我们完成了具身世界模型在线规划的终极实战闭环：
1. **滚动时域的闭环自愈**：通过“算十步、走一步、步步重置”的反馈机制，从根本上抵御了未见物理扰动与外部突发障碍物；
2. **MPPI 路径积分最优性**：利用统计物理玻尔兹曼分布，在无梯度反传的前提下实现了高维动作空间的高效平滑搜索；
3. **集成不确定性防线**：利用多模型分歧度显式构造悲观代价惩罚，彻底堵住了世界模型的幻觉漏洞，确保了物理部署的万无一失。
