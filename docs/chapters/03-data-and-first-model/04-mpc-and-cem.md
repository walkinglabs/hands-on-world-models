# 3.4 模型预测控制与交叉熵方法 (MPC & CEM)

在赋予机器人自主感知与行动能力的工程实践中，除了使用强化学习训练固化的策略网络外，另一条占据现代工业与机器人学半壁江山的技术路径是**基于模型的在线规划与模型预测控制（Model Predictive Control, MPC）**。

与无模型强化学习在面对未知突发工况时的“黑盒猜测”不同，MPC 表现得如同一位严谨的物理学家与棋手：在每一个真实的决策瞬间，智能体利用已知的动力学世界模型，在脑海中向未来虚拟推演数十条乃至数百条候选动作轨迹，精确计算每一条路径的碰撞风险与能耗代价，挑选出最优动作指令。

而在面对高度非线性、不可求导或充满物理接触碰撞的复杂系统时，基于梯度的优化器往往会陷入严重的局部极小值；此时，源自统计物理与进化计算的**交叉熵方法（Cross-Entropy Method, CEM）**展现出了惊人的鲁棒性——通过在高斯概率分布中采样、挑选表现最优的“精英样本”并动态重拟合分布参数，CEM 能够在极其粗糙非凸的代价曲面上快速锁定全局最优轨迹。

本节我们将从初等样本均值方差与极值搜索出发，严密推导 CEM 精英重拟合公式与 MPC 滚动时域闭环自愈机理，并使用纯底层 PyTorch 从零手写一个 GPU 批量 CEM-MPC 在线规划引擎。

<div align="center">

<img src="/figures/03-data-and-first-model/source/04-mpc-and-cem/pets-fig1.png" alt="PETS 算法结合概率动力学神经网络集成与 CEM 在线规划实现高效鲁棒控制。" width="86%">

_图 3.4-1：PETS 算法结合概率动力学神经网络集成与 CEM 在线规划实现高效鲁棒控制。 出处：[Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models，Kurtland Chua et al.，2018](https://arxiv.org/abs/1805.12114)。_

</div>

---

## 3.4.1 物理与控制基石：滚动时域控制与开环推演的闭环自愈

要理解 MPC 的控制哲学，我们首先必须审视开环前向模拟与闭环反馈执行的巧妙结合。

### 1. 滚动时域控制（Receding Horizon Control, RHC）
设规划视界长度为 $H$（如 $H = 10$ 步）。
在当前时刻 $t$：
1. **状态对齐**：测量机器人当前真实物理状态 $\mathbf{s}_t$；
2. **未来规划**：利用动力学模型 $\hat{\mathbf{s}}_{k+1} = f(\hat{\mathbf{s}}_k, \mathbf{a}_k)$，求解未来 $H$ 步的最优动作序列 $\mathbf{U}^* = (\mathbf{a}_t^*, \mathbf{a}_{t+1}^*, \dots, \mathbf{a}_{t+H-1}^*)$；
3. **即时执行**：**仅将第一步动作 $\mathbf{a}_t^*$ 下发给电机执行**；
4. **滚动迭代**：在下一时刻 $t+1$，环境转移到真实状态 $\mathbf{s}_{t+1}$，丢弃上一轮剩余计划，以 $\mathbf{s}_{t+1}$ 为全新起点重新求解未来 $H$ 步的最优动作。

> **初等物理直觉**：
> 为什么要“算十步却只走一步”？因为任何物理模型都不可能百分之百完美（存在风阻、摩擦扰动）。如果固执地把 10 步动作全部执行完毕，初始微小的模型误差会在第 10 步累积放大为剧烈的偏航；而每走一步就用真实状态重新校准，使得系统天然具备了极强的**抗扰动自愈韧性**！

<div align="center">

<img src="/figures/03-data-and-first-model/latex/04-mpc-and-cem/cem-elite-refit.png" alt="交叉熵方法 (CEM) 多轮迭代演化：高斯采样、精英筛选与分布参数收敛" width="86%">

_图 3.4-2：交叉熵方法 (CEM) 多轮迭代演化：高斯采样、精英筛选与分布参数收敛。_

</div>

---

## 3.4.2 核心数学推导一：交叉熵方法 (CEM) 的高斯精英重拟合

在规划未来 $H$ 步的动作序列 $\mathbf{A} \in \mathbb{R}^{H \times d_a}$ 时，整个轨迹的物理代价函数可能高度非凸、不连续且包含悬崖断壁。CEM 通过概率分布的演化来寻找最优参数。

<div align="center">

<img src="/figures/03-data-and-first-model/source/04-mpc-and-cem/planet-fig1.png" alt="PlaNet 世界模型仅通过潜在动力学模型与 CEM 在线规划成功完成多种视觉控制任务。" width="86%">

_图 3.4-3：PlaNet 世界模型仅通过潜在动力学模型与 CEM 在线规划成功完成多种视觉控制任务。 出处：[Learning Latent Dynamics for Planning from Pixels，Danijar Hafner et al.，2018](https://arxiv.org/abs/1811.04551)。_

</div>

### 1. 三步严密 CEM 优化循环流程
系统在维度为 $H \times d_a$ 的动作空间上维护一个高斯分布 $\mathcal{N}(\boldsymbol{\mu}^{(k)}, \text{diag}(\boldsymbol{\sigma}^{2, (k)}))$。

#### 步骤一：批量高斯候选采样（Candidate Sampling）
在第 $k$ 轮迭代中，从当前高斯分布中独立采样 $N$ 条候选动作序列（例如 $N = 256$）：

$$\mathbf{A}_i \sim \mathcal{N}(\boldsymbol{\mu}^{(k)}, \; \text{diag}(\boldsymbol{\sigma}^{2, (k)})), \quad i \in \{1, 2, \dots, N\}$$

#### 步骤二：前向轨迹推演与代价评估（Cost Evaluation）
利用动力学模型计算每条候选动作序列的未来累积物理代价：

$$J(\mathbf{A}_i) = \sum_{t=1}^H \mathcal{C}(\hat{\mathbf{s}}_t^{(i)}, \mathbf{a}_t^{(i)})$$

#### 步骤三：精英筛选与参数极大似然重拟合（Elite Refitting）
将 $N$ 个候选按代价升序排序，挑选出表现最优的 Top-$K_e$ 个样本（精英集合 $\mathcal{E}$，通常取前 $10\%$，即 $K_e = 0.1 N$）。
利用初等样本统计公式更新高斯分布的均值与方差：

$$\boldsymbol{\mu}^{(k+1)} = \frac{1}{K_e} \sum_{i \in \mathcal{E}} \mathbf{A}_i$$

$$\boldsymbol{\sigma}^{2, (k+1)} = \frac{1}{K_e} \sum_{i \in \mathcal{E}} (\mathbf{A}_i - \boldsymbol{\mu}^{(k+1)})^2 + \epsilon$$

引入指数动量平滑（防止早熟收敛陷入方差塌陷）：

$$\boldsymbol{\mu}_{\text{final}} = \alpha \boldsymbol{\mu}^{(k+1)} + (1 - \alpha) \boldsymbol{\mu}^{(k)}, \quad \boldsymbol{\sigma}_{\text{final}} = \alpha \boldsymbol{\sigma}^{(k+1)} + (1 - \alpha) \boldsymbol{\sigma}^{(k)}$$

### 2. CEM 精英重拟合手算数值算例
设规划标量动作 $a$（维度为 1）。当前高斯分布为 $\mu^{(0)} = 0.0, \sigma^{(0)} = 5.0$。
采样了 $N = 4$ 个动作候选：$a = [1.0, 3.0, 5.0, 9.0]$。
经过动力学评估后各动作的代价分别为：
$$J(1.0) = 20.0, \quad J(3.0) = 5.0, \quad J(5.0) = 10.0, \quad J(9.0) = 80.0$$

我们来手动求解下一轮高斯分布参数（取精英数量 $K_e = 2$）：
1. **筛选 Top-2 极小代价精英**：
   - 最小代价 $J(3.0) = 5.0 \implies a_1^* = 3.0$；
   - 次小代价 $J(5.0) = 10.0 \implies a_2^* = 5.0$；
   - 精英集合为 $\mathcal{E} = \{3.0, 5.0\}$。
2. **计算新均值 $\mu^{(1)}$**：
   $$\mu^{(1)} = \frac{3.0 + 5.0}{2} = 4.0$$
3. **计算新方差 $\sigma^{2, (1)}$**：
   $$\sigma^{2, (1)} = \frac{(3.0 - 4.0)^2 + (5.0 - 4.0)^2}{2} = \frac{(-1.0)^2 + (1.0)^2}{2} = \frac{1.0 + 1.0}{2} = 1.0$$
   $$\sigma^{(1)} = \sqrt{1.0} = 1.0$$

初等代数的几步极简运算生动展现了 CEM 的收敛魅力：高斯分布的中心从原本漫无目的的 $0.0$ 迅速聚焦迁移至高价值区域 $4.0$，同时搜索标准差从大范围漫游的 $5.0$ 自动收敛压缩至 $1.0$！

<details>
<summary><b>深入推导：交叉熵方法在 Kullback-Leibler 散度极小化下的理论渐近收敛性证明（点击展开查看完整推导）</b></summary>

设目标优化分布为示性玻尔兹曼测度 $p^*(\mathbf{x}) \propto \mathbb{I}(J(\mathbf{x}) \le \gamma)$。
寻找参数化分布族 $q(\mathbf{x}; \mathbf{v})$ 使得其与最优分布的 KL 散度极小：
$$\min_{\mathbf{v}} D_{\text{KL}}(p^* \parallel q(\cdot; \mathbf{v})) \iff \max_{\mathbf{v}} \int p^*(\mathbf{x}) \log q(\mathbf{x}; \mathbf{v}) d\mathbf{x} = \max_{\mathbf{v}} \mathbb{E}_{\mathbf{x} \sim q(\cdot; \mathbf{u})} \left[ \frac{\mathbb{I}(J(\mathbf{x}) \le \gamma)}{q(\mathbf{x}; \mathbf{u})} \log q(\mathbf{x}; \mathbf{v}) \right]$$
对高斯分布参数 $\boldsymbol{\mu}, \boldsymbol{\Sigma}$ 分别求偏导并置零，微分方程的唯一鞍点解严格等价于精英子集的样本均值与协方差矩阵。
</details>

---

## 3.4.3 核心数学推导二：MPC 滚动时域终端约束与渐近稳定性

在 MPC 长期滚动推演中，如果规划视界 $H$ 过短，算法容易出现“鼠目寸光”导致的死锁或翻车。

<div align="center">

<img src="/figures/03-data-and-first-model/latex/04-mpc-and-cem/cem-elite-refit.png" alt="滚动时域控制时间轴：开环前向规划视界展开与闭环单步执行滚动更新" width="86%">

_图 3.4-4：滚动时域控制时间轴：开环前向规划视界展开与闭环单步执行滚动更新。_

</div>

为了保证闭环系统的全局稳定性，现代非线性 MPC 通常在第 $H$ 步引入**控制李雅普诺夫终端惩罚项（Terminal Cost $V_f(\mathbf{s}_H)$）**与**终端不变集约束（Terminal Invariant Set $\mathbb{X}_f$）**：

$$J_{\text{MPC}} = \sum_{t=0}^{H-1} \mathcal{C}(\mathbf{s}_t, \mathbf{a}_t) + V_f(\mathbf{s}_H), \quad \text{s.t. } \mathbf{s}_H \in \mathbb{X}_f$$

终端惩罚项在数学上等价于对未来无穷时域剩余价值的保守悲观代理估计，确保了机器人在有限视界内的每一步动作都能推动物理系统稳定渐近收敛至平衡点。

<details>
<summary><b>深入推导：非线性受控系统在李雅普诺夫终端惩罚项下的 MPC 渐近稳定性充要条件证明（点击展开查看完整推导）</b></summary>

设最优化目标在时刻 $t$ 的最优代价为 $V^*(\mathbf{s}_t)$。
若终端代价满足李雅普诺夫衰减条件 $\min_{\mathbf{a}} [V_f(f(\mathbf{s}, \mathbf{a})) - V_f(\mathbf{s}) + \mathcal{C}(\mathbf{s}, \mathbf{a})] \le 0, \forall \mathbf{s} \in \mathbb{X}_f$。
在时刻 $t+1$，构造次优动作序列 $\tilde{\mathbf{U}}_{t+1} = \{\mathbf{a}_{t+1}^*, \dots, \mathbf{a}_{t+H-1}^*, \kappa_f(\mathbf{s}_{t+H}^*)\}$。
由最优性定义可知：
$$V^*(\mathbf{s}_{t+1}) - V^*(\mathbf{s}_t) \le J(\tilde{\mathbf{U}}_{t+1}) - V^*(\mathbf{s}_t) \le -\mathcal{C}(\mathbf{s}_t, \mathbf{a}_t^*) < 0$$
函数 $V^*(\mathbf{s})$ 构成了闭环系统的严格控制李雅普诺夫函数（CLF），根据 LaSalle 不变原理，系统状态渐近收敛于原点。
</details>

---

## 3.4.4 纯底层 PyTorch 代码实现：从零手写 GPU 批量 CEM-MPC 在线规划引擎

下面我们使用纯底层 PyTorch 算子手写实现一个支持高并发张量并行模拟的 CEM 优化器与 MPC 在线闭环控制器。

```python
import torch
import torch.nn as nn

class CartPoleDynamics(nn.Module):
    """
    倒立摆动力学模型 (CartPole Dynamics)
    s = [x, x_dot, theta, theta_dot]
    """
    def __init__(self, dt: float = 0.05, g: float = 9.81, mc: float = 1.0, mp: float = 0.1, l: float = 0.5):
        super().__init__()
        self.dt = dt
        self.g = g
        self.mc = mc
        self.mp = mp
        self.l = l

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        :param state: (N, 4) 批量状态
        :param action: (N, 1) 批量水平推力
        :return: (N, 4) 下一状态
        """
        x, x_dot, theta, theta_dot = state.unbind(dim=-1)
        force = action.squeeze(-1).clamp(-10.0, 10.0)

        cos_th = torch.cos(theta)
        sin_th = torch.sin(theta)
        total_m = self.mc + self.mp
        temp = (force + self.mp * self.l * theta_dot.pow(2) * sin_th) / total_m

        theta_acc = (self.g * sin_th - cos_th * temp) / (self.l * (4.0/3.0 - self.mp * cos_th.pow(2) / total_m))
        x_acc = temp - self.mp * self.l * theta_acc * cos_th / total_m

        next_x = x + self.dt * x_dot
        next_x_dot = x_dot + self.dt * x_acc
        next_theta = theta + self.dt * theta_dot
        next_theta_dot = theta_dot + self.dt * theta_acc

        return torch.stack([next_x, next_x_dot, next_theta, next_theta_dot], dim=-1)

class CEMPlanner:
    """
    GPU 批量交叉熵在线规划器 (CEM Planner)
    """
    def __init__(self, dynamics: CartPoleDynamics, horizon: int = 15, num_samples: int = 256, num_elites: int = 25, iterations: int = 5):
        self.dynamics = dynamics
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.iterations = iterations

    def cost_fn(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        物理代价函数：直立惩罚 + 水平位置偏离 + 力矩消耗
        """
        x = states[:, :, 0]
        theta = states[:, :, 2]
        cost = 10.0 * theta.pow(2) + 1.0 * x.pow(2) + 0.01 * actions.squeeze(-1).pow(2)
        return cost.sum(dim=-1) # (N,)

    def plan(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        :param current_state: (4,)
        :return: (horizon, 1) 最优动作序列
        """
        device = current_state.device
        # 初始高斯分布: 均值 0，标准差 2.0
        mu = torch.zeros(self.horizon, 1, device=device)
        std = torch.full((self.horizon, 1), 2.0, device=device)

        for _ in range(self.iterations):
            # 1. 批量采样 (num_samples, horizon, 1)
            actions = mu.unsqueeze(0) + std.unsqueeze(0) * torch.randn(self.num_samples, self.horizon, 1, device=device)

            # 2. 状态并行前向展开
            sim_state = current_state.unsqueeze(0).expand(self.num_samples, -1)
            state_traj = []
            for t in range(self.horizon):
                sim_state = self.dynamics(sim_state, actions[:, t, :])
                state_traj.append(sim_state)

            stacked_states = torch.stack(state_traj, dim=1) # (num_samples, horizon, 4)

            # 3. 评估轨迹代价
            costs = self.cost_fn(stacked_states, actions)

            # 4. 精英筛选与高斯重拟合
            elite_indices = torch.topk(costs, k=self.num_elites, largest=False).indices
            elites = actions[elite_indices] # (num_elites, horizon, 1)

            mu = elites.mean(dim=0)
            std = elites.std(dim=0).clamp_min(0.1)

        return mu # 返回最终收敛的最优动作序列均值

# ===================================================================
# 单元测试与闭环推演校验
# ===================================================================
if __name__ == "__main__":
    dynamics = CartPoleDynamics()
    planner = CEMPlanner(dynamics=dynamics, horizon=10, num_samples=128, num_elites=16, iterations=4)

    # 初始倾斜状态: [x=0, v=0, theta=0.2 rad, omega=0]
    initial_state = torch.tensor([0.0, 0.0, 0.2, 0.0])

    optimal_actions = planner.plan(initial_state)
    first_action = optimal_actions[0]

    print(f"[CEM Test] 规划视界步数: {planner.horizon}")
    print(f"[CEM Test] 第一步下发控制推力: {first_action.item():.4f} N")
    print(f"[CEM Test] 最优动作序列形状: {optimal_actions.shape}")

    assert optimal_actions.shape == (10, 1), "最优动作序列维度不符！"
    assert not torch.isnan(optimal_actions).any(), "CEM 优化出现 NaN 异常！"
    print("✓ 倒立摆动力学模型与 GPU 批量 CEM 在线规划引擎单测全部通过！")
```

---

## 3.4.5 本节小结

回顾本节内容，我们建立了基于模型的在线规划核心知识图谱：
1. **滚动时域的闭环自愈**：通过前向规划多步但仅执行第一步的反馈机制，从根本上抵御了动力学建模误差与外部扰动；
2. **CEM 进化采样的强鲁棒性**：通过高斯精英重拟合，在无需任何梯度反传的前提下征服了高度非凸的物理代价曲面；
3. **世界模型与 MPC 的完美结合**：世界模型负责高保真预测未来，MPC 负责高效求解动作，为后续大章节的梦境强化学习与端到端具身规划提供了坚固的决策底座。
