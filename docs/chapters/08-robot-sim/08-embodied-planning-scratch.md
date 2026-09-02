# 8.8 具身规划（Embodied Planning）的从零开始实现

## 历史脉络与学术背景

设想一个小车需要绕过障碍物，到达两米外的目标点。控制器可以先用模型预测几组“向左、向右、加速”的动作序列，比较它们的预计代价，再执行其中最好序列的第一个动作。执行后，小车根据新观测重新规划。这个“预测—选择—执行一步—再次预测”的循环，就是本节要实现的模型预测控制。

<div align="center">
<img src="/figures/08-robot-sim/source/08-embodied-planning-scratch/mjpc-fig4.png" alt="MuJoCo MPC 在多种机器人与接触任务上实时合成行为，展示在线模型式规划的应用范围。" width="86%">

_图 8.8-1：MuJoCo MPC 在多种机器人与接触任务上实时合成行为，展示在线模型式规划的应用范围。 出处：Taylor Howell et al.，[Predictive Sampling: Real-time Behaviour Synthesis with MuJoCo](https://arxiv.org/abs/2212.00541)（2022），Figure 4。_
</div>

模型预测控制（Model Predictive Control, MPC）在每个控制时刻基于系统模型求解有限时域优化问题，执行当前动作后再用新状态重新求解 [[Camacho & Bordons, 1999]](https://link.springer.com/book/10.1007/978-1-4471-3398-8)。模型不必“精确刻画全部现实”，但其误差会影响滚动预测与动作选择。

深度世界模型与经典 MPC 提供了两种不同的模型式决策路线。Dreamer 从像素观测学习潜在动力学，并在想象轨迹上训练 actor 与 critic [[Hafner et al., 2020]](https://arxiv.org/abs/1912.01603); [[Hafner et al., 2023]](https://arxiv.org/abs/2301.04104)；部署时由策略直接输出动作，因此它不是逐步在线搜索的 MPC。MuJoCo MPC（MJPC）则把 MuJoCo 动力学与预测采样、梯度下降、iLQG 等在线规划器结合 [[Howell et al., 2022]](https://arxiv.org/abs/2212.00541)。本节比较的是“潜在想象后学习策略”与“已知物理模型上的在线优化”，而不是把两者都称为同一种规划器。

<div align="center">
<img src="/figures/08-robot-sim/source/08-embodied-planning-scratch/pets-fig3.png" alt="PETS 用概率集成动力学与轨迹优化在少量真实试验中学习控制，提供学习模型式规划的实证对照。" width="86%">

_图 8.8-2：PETS 用概率集成动力学与轨迹优化在少量真实试验中学习控制，提供学习模型式规划的实证对照。 出处：Kurtland Chua et al.，[Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/abs/1805.12114)（2018），Figure 3。_
</div>

下面先写出有限时域优化问题，再用交叉熵方法（Cross-Entropy Method, CEM）搜索动作序列，最后在 PyTorch 中实现批量轨迹推演。

## 具身规划的物理与数学映射

规划需要三个基本元素：描述当前位置与速度的状态、能够施加的动作，以及预测动作后果的动力学模型。

把智能体在时间步 $t$ 的状态记为 $\mathbf{s}_t \in \mathbb{R}^n$，其中可以包含关节角度、角速度和环境信息。连续动作向量（例如电机扭矩）记为 $\mathbf{a}_t \in \mathbb{R}^m$。世界模型可以抽象为非线性状态转移函数 $f$：

$$ \mathbf{s}_{t+1} = f(\mathbf{s}_t, \mathbf{a}_t) $$

为了量化评价某一特定状态和动作组合的优劣，我们定义一个标量函数，即代价函数（Cost Function） $C(\mathbf{s}, \mathbf{a})$。它的物理意义在于衡量当前系统偏离期望目标（例如球偏离篮筐中心）的程度以及系统做功的能耗。

规划器的目标是寻找一段未来动作序列 $\mathbf{A} = [\mathbf{a}_t, \mathbf{a}_{t+1}, \dots, \mathbf{a}_{t+H-1}]$，使未来 $H$ 个预测时间步内的累积代价尽可能小：

$$ \min_{\mathbf{A}} \sum_{k=0}^{H-1} C(\mathbf{s}_{t+k}, \mathbf{a}_{t+k}) $$

每一步状态都由动力学模型约束。在模型预测控制（MPC）中，控制器会求解长度为 $H$ 的动作序列，但**只执行第一个动作 $\mathbf{a}_t$**。到 $t+1$ 时获得新状态，再重新求解。滚动视界能在下一轮利用观测修正预测误差，但修正能力仍取决于模型质量、规划时域和计算预算。

## 交叉熵方法（CEM）推导与参数迭代

当动力学或代价函数非凸、不可微时，可以用采样方法直接比较候选动作序列。CEM 是其中一种常见的零阶优化方法。

<div align="center">
<img src="/figures/08-robot-sim/source/08-embodied-planning-scratch/mppi-fig8.png" alt="信息论 MPC 的真实车辆时序图展示采样式控制器如何在滚动时域中完成高速转弯。" width="86%">

_图 8.8-3：信息论 MPC 的真实车辆时序图展示采样式控制器如何在滚动时域中完成高速转弯。 出处：Grady Williams et al.，[Information Theoretic MPC for Model-Based Reinforcement Learning](https://arxiv.org/abs/1707.02342)（2017），Figure 8。_
</div>

假设候选动作序列来自一个参数化高斯分布，其均值为 $\boldsymbol{\mu}$，协方差为 $\boldsymbol{\Sigma}$。为了简化实现，下面的代码只保存各维方差，相当于使用对角协方差。

首先，我们在每一次优化迭代开始时，从当前分布中独立同分布地采样出 $N$ 条候选的动作序列轨迹：

$$ \mathbf{A}^{(i)} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), \quad i \in \{1, 2, \dots, N\} $$

接着，针对每一条候选轨迹 $\mathbf{A}^{(i)}$，从初始状态 $\mathbf{s}_t$ 出发，用世界模型展开未来状态，并计算累积代价 $J^{(i)}$：

$$ J^{(i)} = \sum_{k=0}^{H-1} C(\mathbf{s}_{t+k}^{(i)}, \mathbf{a}_{t+k}^{(i)}) $$

<div align="center">
<img src="/figures/08-robot-sim/latex/08-embodied-planning-scratch/cem-rollout-tensor-axes.png" alt="CEM 候选动作张量逐时间切片并沿预测视界累加代价，同时始终保留候选样本轴" width="86%">

_图 8.8-4：每次 rollout 只切出一个时间片，但 N 条候选始终并行保留；沿 H 累加后得到的仍是长度 N 的轨迹代价向量。_
</div>

随后，将这 $N$ 条轨迹按照累积代价 $J^{(i)}$ 从小到大排序，只保留代价最小的前 $K$ 条轨迹（$K < N$）。它们称为精英样本（Elite Samples）。

最后，用 $K$ 个精英样本的经验均值和方差重新估计 $\boldsymbol{\mu}'$ 与 $\boldsymbol{\Sigma}'$，并把它们作为下一轮采样分布。重复这一过程后，采样会逐渐集中到当前找到的低代价区域。CEM 不需要动力学梯度，但仍可能过早集中在局部解附近，因此通常要设置方差下限、平滑更新或多组初始分布。

::: info 说明
从实现角度看，CEM 的更新就是对精英样本重新做一次高斯分布的极大似然估计。硬截断便于实现，但会丢弃非精英样本中的信息。
:::

## 现代开源项目的工程视角：MJPC 与 DreamerV3

在实现代码前，先区分两类容易混淆的模型式决策方法。

**Google DeepMind 的 MuJoCo MPC (MJPC)**
MuJoCo MPC（MJPC）是一个基于 MuJoCo 的实时、交互式模型预测控制框架 [[Howell et al., 2022]](https://arxiv.org/abs/2212.00541)。公开论文描述了预测采样、梯度下降与迭代线性二次高斯（iLQG）等规划器，并用 C++ 多线程并行评估轨迹。它强调实时控制与可交互任务设计，但原文没有把系统概括为“GPU 上的千赫兹 MPPI/CEM 框架”；具体控制频率取决于模型、规划器、时域和硬件。

<div align="center">
<img src="/figures/08-robot-sim/source/08-embodied-planning-scratch/mjpc-fig3.png" alt="MJPC 的交互界面同时展示模型、代价、预测轨迹和控制信号，说明在线规划器怎样进入控制循环。" width="86%">

_图 8.8-5：MJPC 的交互界面同时展示模型、代价、预测轨迹和控制信号，说明在线规划器怎样进入控制循环。 出处：Taylor Howell et al.，[Predictive Sampling: Real-time Behaviour Synthesis with MuJoCo](https://arxiv.org/abs/2212.00541)（2022），Figure 3。_
</div>

**DreamerV3 的潜在空间规划**
DreamerV3 则从观测中学习潜在状态模型，并在潜在想象轨迹上训练 actor 与 critic [[Hafner et al., 2023]](https://arxiv.org/abs/2301.04104)。它在部署时由 actor 直接输出动作，不像 MJPC 那样在每一步在线采样候选轨迹。因此，两者的区别不仅是“物理状态与潜在状态”，还包括在线 MPC 与离线想象训练后策略执行这两种决策方式。

## 具身规划的从零开始实现

带着对经典理论和现代工程架构的理解，我们将利用 PyTorch 编写一个能够利用 GPU 进行张量级并行推演的模型预测控制器。我们首先构建一个模拟的非线性世界模型。

```python
import torch
from torch import nn

class SimulatedWorldModel(nn.Module):
    """
    一个简化的非线性世界模型（用于替代复杂的物理引擎或深度RNN）。
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # 单层网络仅用于演示张量形状，不代表足够准确的实机模型
        self.dynamics = nn.Linear(state_dim + action_dim, state_dim)

    def forward(self, state, action):
        """
        前向传播以计算下一个时间步的状态。
        :param state: 形状为 (batch_size, state_dim) 的张量
        :param action: 形状为 (batch_size, action_dim) 的张量
        :return: 形状为 (batch_size, state_dim) 的下一个状态
        """
        x = torch.cat([state, action], dim=-1)
        next_state = torch.relu(self.dynamics(x))
        return next_state

def cost_function(state, action, target_state):
    """
    代价函数：计算当前状态与目标状态之间的欧式距离平方，并施加对控制能耗的二次惩罚。
    """
    # 状态的 L2 距离，衡量任务完成度
    state_cost = torch.sum((state - target_state) ** 2, dim=-1)
    # 动作的 L2 范数，用于限制控制能耗
    action_cost = 0.1 * torch.sum(action ** 2, dim=-1)
    return state_cost + action_cost
```

接下来，我们编写最核心的 CEM 规划器。在以下的实现中，请务必仔细体会我们是如何在初始状态张量的批次维度（Batch Dimension）上展开 $N$ 条轨迹的，这正是避免显式循环、实现底层硬件加速的工程秘诀。

```python
class CEMPlanner:
    def __init__(self, world_model, target_state, horizon, num_samples,
                 num_elites, iterations, action_low=-1.0, action_high=1.0):
        """
        基于交叉熵方法（CEM）的模型预测控制器。
        :param world_model: 用于预测未来状态流转的世界模型
        :param target_state: 期望达到的目标物理状态
        :param horizon: 预测视界 H
        :param num_samples: 每次迭代采样的轨迹总批次数 N
        :param num_elites: 经过代价排序后选出的精英样本数量 K
        :param iterations: 分布拟合优化的总迭代次数
        """
        self.model = world_model
        self.target_state = target_state
        self.H = horizon
        self.N = num_samples
        self.K = num_elites
        self.iters = iterations
        self.action_dim = world_model.action_dim
        self.action_low = action_low
        self.action_high = action_high

    def plan(self, current_state):
        """
        给定当前状态，返回 CEM 动作序列的第一个动作。
        """
        device, dtype = current_state.device, current_state.dtype
        target_state = self.target_state.to(device=device, dtype=dtype)
        action_mean = torch.zeros(
            self.H, self.action_dim, device=device, dtype=dtype
        )
        action_std = torch.ones_like(action_mean)

        for i in range(self.iters):
            # 张量形状：(N, H, action_dim)
            noise = torch.randn(
                self.N, self.H, self.action_dim, device=device, dtype=dtype
            )
            actions = action_mean.unsqueeze(0) + action_std.unsqueeze(0) * noise
            actions = actions.clamp(self.action_low, self.action_high)

            # 把当前状态复制 N 份，以并行推演候选轨迹
            # 张量形状：(N, state_dim)
            states = current_state.unsqueeze(0).repeat(self.N, 1)
            costs = torch.zeros(self.N, device=device, dtype=dtype)

            # 步骤2：在 H 步内滚动世界模型并累加每条轨迹的代价
            for t in range(self.H):
                # 提取出时间步 t 时，所有 N 条轨迹对应的瞬时动作
                # 形状：(N, action_dim)
                a_t = actions[:, t, :]
                states = self.model(states, a_t)
                costs += cost_function(states, a_t, target_state)

            # 步骤3：选出代价最低的 K 个精英样本
            elite_indices = torch.topk(costs, self.K, largest=False).indices
            elite_actions = actions[elite_indices]  # (K, H, action_dim)

            # 步骤4：用精英样本的经验统计量更新高斯分布
            action_mean = elite_actions.mean(dim=0)
            # 设置方差下限，保留少量探索
            action_std = elite_actions.std(dim=0, unbiased=False).clamp_min(1e-3)

        # MPC 只执行规划序列中的第一个动作
        return action_mean[0]

# 实例化演示模型与规划器
state_dim = 4
action_dim = 2
model = SimulatedWorldModel(state_dim, action_dim)
target = torch.ones(state_dim)

# 设置采样规模 N=1000, 选拔标准 K=100
planner = CEMPlanner(world_model=model, target_state=target,
                     horizon=15, num_samples=1000, num_elites=100, iterations=5)

# 这里只检查张量形状；随机初始化的模型尚不具备真实控制意义
initial_state = torch.rand(state_dim)
best_next_action = planner.plan(initial_state)
print("CEM 返回的首步动作:", best_next_action)
```

## 小结

- MPC 在每个时刻优化有限时域动作序列，只执行首个动作后便重新规划。
- CEM 通过“采样—筛选精英—重估分布”搜索动作序列，不要求动力学可微。
- 批量轨迹推演可以提高候选序列评估效率，但控制效果仍取决于世界模型和代价函数。
- Dreamer 的部署策略与 MJPC 的在线规划属于两种不同的模型式决策路线。

## 练习

1. 当前 `CEMPlanner` 每次都把 `action_mean` 初始化为 0。实现热启动：把上一时刻的动作均值向前移动一格，并为末尾补零。
2. 当经验方差快速接近零时，采样器会失去探索能力。尝试用指数滑动平均更新均值和方差，并比较不同平滑系数。
3. MuJoCo MPC 同时提供采样式、梯度式与 iLQG 规划器 [[Howell et al., 2022]](https://arxiv.org/abs/2212.00541)。如果 `SimulatedWorldModel` 的转移函数可微且在工作区域内足够平滑，可以用 PyTorch 自动微分直接优化连续动作轨迹，再与 CEM 比较计算耗时、局部最优敏感性与收敛稳定性。
