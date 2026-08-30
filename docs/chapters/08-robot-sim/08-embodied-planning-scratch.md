# 具身规划（Embodied Planning）的从零开始实现
:label:`sec_embodied_planning_scratch`

## 历史脉络与学术背景

在深度学习与强化学习交汇的早期，智能体的决策主要依赖于无模型的（Model-Free）策略梯度或价值迭代方法。然而，当我们试图将这些经典算法部署到物理世界中的真实机器人（即具身智能体，Embodied Agent）上时，不可避免地遭遇了严峻的现实壁垒：物理世界的数据采集不仅时间成本极其高昂，而且系统在试错过程中极易发生不可逆的硬件损坏。

为了从根本上规避这些致命缺陷，研究者们转向了基于模型的（Model-Based）规划方法。追本溯源，这一思想滥觞于经典控制理论中的模型预测控制（Model Predictive Control, MPC）[Camacho & Bordons, 1999]。MPC 的核心哲学在于：通过内部构建对周围环境动态演化的精确刻画，在脑海中预演未来可能的多种轨迹，并据此选出当下最优的控制指令。

进入深度学习时代后，得益于神经网络在表征学习上的突破，我们得以构建处理极高维观测的深度世界模型。在此背景下，具身规划技术迎来了两次工程革命。其一，以 Dreamer 系列 [Hafner et al., 2019, 2023] 为代表的算法，成功将原本依赖低维物理状态方程的规划，升维至处理高维像素输入；其二，以 Google DeepMind 推出的 MuJoCo MPC (MJPC) [Howell et al., 2022] 为代表的开源系统，将并行采样规划器与高精度的接触力学模拟器深度绑定，标志着具身规划已从理论上的马尔可夫决策过程（MDP）跨越到了毫秒级实时计算的物理工程实践中。

本节我们将从最为基础的高中运动学知识起步，严格推导具身规划的核心数学机制，并结合现代开源项目的工程理念，在 PyTorch 中从零开始构建一个具备高度张量化特征的模型预测控制器。

## 具身规划的物理与数学映射

为了剥离深度学习概念的表面迷雾，我们首先将具身规划降维映射到经典的高中物理场景中。

想象在物理课上研究抛体运动时，当你试图将一枚篮球精准抛入远处的篮筐。你的大脑实际上在无意识中执行了一套极度复杂的“具身规划”流程。你并没有随机挥动手臂，而是基于对重力加速度和空气阻力（即世界运行的物理规律）的内化理解，在脑海中虚拟出了一条抛物线（轨迹推演），并评估这条抛物线最终距离篮筐中心的偏差有多大（代价评估）。

在严格的数学表述下，我们将智能体在时间步 $t$ 的状态记为 $\mathbf{s}_t \in \mathbb{R}^n$，它包含但不限于机器人的关节角度、角速度以及环境信息。智能体能够施加的连续动作向量（例如电机扭矩）记为 $\mathbf{a}_t \in \mathbb{R}^m$。世界模型（大脑中的物理规律）可以被抽象为一个非线性的动力学状态转移函数 $f$：

$$ \mathbf{s}_{t+1} = f(\mathbf{s}_t, \mathbf{a}_t) $$
:eqlabel:`eq_dynamics_f`

为了量化评价某一特定状态和动作组合的优劣，我们定义一个标量函数，即代价函数（Cost Function） $C(\mathbf{s}, \mathbf{a})$。它的物理意义在于衡量当前系统偏离期望目标（例如球偏离篮筐中心）的程度以及系统做功的能耗。

我们的终极目标是，寻找一段未来连续的动作序列 $\mathbf{A} = [\mathbf{a}_t, \mathbf{a}_{t+1}, \dots, \mathbf{a}_{t+H-1}]$，使得在未来 $H$ 个预测时间步（即预测视界，Prediction Horizon）内的累积代价达到极小值。我们将其严谨地表述为如下的带有等式约束的离散时间最优化问题：

$$ \min_{\mathbf{A}} \sum_{k=0}^{H-1} C(\mathbf{s}_{t+k}, \mathbf{a}_{t+k}) $$
:eqlabel:`eq_mpc_objective`

在这个演化过程中，每一步的状态流转都必须被世界模型 :eqref:`eq_dynamics_f` 严格约束。在模型预测控制（MPC）的经典范式中，我们会在每个离散时间 $t$ 求解出这段长度为 $H$ 的最优动作序列，但**仅仅执行该序列中的第一个动作 $\mathbf{a}_t$**。当时间步推进到 $t+1$ 观测到真实的新状态后，再次启动新一轮的 $H$ 步预测与优化。这种“滚动视界”（Receding Horizon）的控制机制，赋予了系统极强的抗扰动和纠错能力。

## 交叉熵方法（CEM）推导与参数迭代

求解目标函数 :eqref:`eq_mpc_objective` 面临着巨大的挑战，因为复杂的动力学函数 $f$ 往往会导致整个代价流形呈现出极度非凸的几何特征，常规的基于梯度的优化方法极易陷入局部极小值。为了突破这一瓶颈，现代工程中（包括很多早期版本的世界模型架构）广泛采用了一种基于零阶优化的概率方法：交叉熵方法（Cross-Entropy Method, CEM）。

让我们利用统计学知识严密推导 CEM 的迭代机制。假设最优的动作序列是由某种未知的概率分布生成的。作为先验，我们假定动作序列服从参数化的多元高斯分布，其均值向量为 $\boldsymbol{\mu}$，协方差矩阵为 $\boldsymbol{\Sigma}$。

首先，我们在每一次优化迭代开始时，从当前分布中独立同分布地采样出 $N$ 条候选的动作序列轨迹：

$$ \mathbf{A}^{(i)} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), \quad i \in \{1, 2, \dots, N\} $$
:eqlabel:`eq_cem_sample`

接着，针对每一条候选轨迹 $\mathbf{A}^{(i)}$，我们利用初始状态 $\mathbf{s}_t$ 和世界模型 :eqref:`eq_dynamics_f` 迭代展开未来的状态轨迹，并根据 :eqref:`eq_mpc_objective` 计算其对应的累积代价 $J^{(i)}$：

$$ J^{(i)} = \sum_{k=0}^{H-1} C(\mathbf{s}_{t+k}^{(i)}, \mathbf{a}_{t+k}^{(i)}) $$
:eqlabel:`eq_cem_cost`

随后，我们将这 $N$ 条轨迹按照其累积代价 $J^{(i)}$ 从小到大进行严格的升序排列。为了提炼出蕴含最优解信息的样本，我们施加一个硬性截断，仅保留代价最小的前 $K$ 条轨迹（其中 $K < N$），这部分轨迹在文献中被称为精英样本（Elite Samples）。

最后，CEM 最核心的一步在于**分布重塑**。我们摒弃原有的参数，利用这 $K$ 个精英样本在各个维度的经验均值和经验方差，重新极大似然估计（MLE）出新的 $\boldsymbol{\mu}'$ 和 $\boldsymbol{\Sigma}'$，并将其作为下一轮迭代的采样分布。这种从发散采样到向高回报区域坍缩的自适应过程，正是其能够在高维连续动作空间中规避局部最优的关键数学机理。

> [!NOTE]
> 交叉熵方法（CEM）本质上可以视作对变分推断中重要性采样的一种极端近似，它用直接的硬截断代替了平滑的似然加权，这种妥协换取了在实际非平稳环境中的鲁棒性。

## 现代开源项目的工程视角：MJPC 与 DreamerV3

在从零实现代码之前，审视当前最先进的开源项目是如何将上述数学模型进行工程化落地的，显得尤为重要。

**Google DeepMind 的 MuJoCo MPC (MJPC)** 
MJPC [Howell et al., 2022] 致力于为真实物理机器人提供千赫兹（kHz）级别的实时 MPC 控制能力。为了打破计算瓶颈，MJPC 在架构设计上极其强调并行张量化。它放弃了传统依赖 Python for 循环评估代价的做法，将动力学展开 :eqref:`eq_dynamics_f` 和代价计算 :eqref:`eq_cem_cost` 深度编译至底层 C++ 并在多线程甚至 GPU 上并行调度。此外，为了克服原生 CEM 在高维空间探索效率低下的问题，MJPC 广泛引入了模型预测路径积分（MPPI）以及 iLQG（基于梯度的迭代线性二次高斯优化）等算法，通过融合代价地形的局部曲率信息（如海森矩阵的近似）来引导采样方向，这代表了当前具身控制的最前沿工程范式。

**DreamerV3 的潜在空间规划**
与直接依赖物理引擎进行状态展开的 MJPC 截然不同，DreamerV3 [Hafner et al., 2023] 重点攻克了直接从高维像素（如摄像头画面）进行规划的难题。其最惊艳的工程设计在于：它严厉禁止使用极其耗时的原图进行滚动预测。相反，系统预先训练了一个庞大且高度复杂的循环神经网络（RNN）和自编码器（Autoencoder），将高维像素强制压缩至一个极低维度的隐状态（Latent State）向量中。因此，在 Dreamer 架构下，公式 :eqref:`eq_dynamics_f` 中的 $\mathbf{s}_t$ 不再是物理意义上的关节坐标，而纯粹是由神经网络推断出的抽象代数向量。所有的采样、状态推演与代价计算，全部在内存中的张量隐空间（Latent Space）内以光速流转，彻底摆脱了物理渲染引擎的拖累。

## 具身规划的从零开始实现

带着对经典理论和现代工程架构的理解，我们将利用 PyTorch 编写一个能够利用 GPU 进行张量级并行推演的模型预测控制器。我们首先构建一个模拟的非线性世界模型。

```{.python .input}
#@tab pytorch
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
        # 在这里我们采用单层神经网络来近似一个未知的物理动力学方程
        self.dynamics = nn.Linear(state_dim + action_dim, state_dim)
        
    def forward(self, state, action):
        """
        前向传播以计算下一个时间步的状态。
        :param state: 形状为 (batch_size, state_dim) 的张量
        :param action: 形状为 (batch_size, action_dim) 的张量
        :return: 形状为 (batch_size, state_dim) 的下一个状态
        """
        # (**沿着最后一个维度拼接当前状态与动作张量，并通过非线性激活函数以近似真实物理世界的复杂耦合转移**)
        x = torch.cat([state, action], dim=-1)
        next_state = torch.relu(self.dynamics(x))
        return next_state

def cost_function(state, action, target_state):
    """
    代价函数：计算当前状态与目标状态之间的欧式距离平方，并施加对控制能耗的二次惩罚。
    """
    # 状态的 L2 距离，衡量任务完成度
    state_cost = torch.sum((state - target_state) ** 2, dim=-1)
    # 动作的 L2 范数，施加正则化惩罚以避免极其暴力的控制力矩
    action_cost = 0.1 * torch.sum(action ** 2, dim=-1)
    return state_cost + action_cost
```

接下来，我们编写最核心的 CEM 规划器。在以下的实现中，请务必仔细体会我们是如何在初始状态张量的批次维度（Batch Dimension）上展开 $N$ 条轨迹的，这正是避免显式循环、实现底层硬件加速的工程秘诀。

```{.python .input}
#@tab pytorch
class CEMPlanner:
    def __init__(self, world_model, target_state, horizon, num_samples, num_elites, iterations):
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

    def plan(self, current_state):
        """
        给定当前系统的真实物理状态，在脑海中规划并返回最核心的最优立即执行动作。
        """
        # 初始状态下，动作分布为标准正态分布，即没有任何先验偏好
        action_mean = torch.zeros(self.H, self.action_dim)
        action_std = torch.ones(self.H, self.action_dim)
        
        for i in range(self.iters):
            # (**步骤1：利用重参数化技巧，从当前优化得到的高斯分布中，并行且独立同分布地采样出 N 条连续的动作序列**)
            # 张量形状：(N, H, action_dim)
            actions = action_mean.unsqueeze(0) + action_std.unsqueeze(0) * torch.randn(self.N, self.H, self.action_dim)
            
            # 为了进行并行推演，我们需要将标量级别的当前状态复制扩张 N 份
            # 张量形状：(N, state_dim)
            states = current_state.unsqueeze(0).repeat(self.N, 1)
            costs = torch.zeros(self.N)
            
            # (**步骤2：利用已知的世界模型在预测视界 H 内进行自回归滚动推断，累加得到每条轨迹的整体代价**)
            for t in range(self.H):
                # 提取出时间步 t 时，所有 N 条轨迹对应的瞬时动作
                # 形状：(N, action_dim)
                a_t = actions[:, t, :]
                states = self.model(states, a_t)
                costs += cost_function(states, a_t, self.target_state)
            
            # (**步骤3：计算代价序列并执行硬截断，严格筛选出代价最低的 K 个精英样本**)
            _, elite_indices = torch.sort(costs)
            elite_indices = elite_indices[:self.K]
            elite_actions = actions[elite_indices] # 精英动作，形状：(K, H, action_dim)
            
            # (**步骤4：摒弃旧分布，利用精英样本的经验统计量极大似然重构新的多元高斯分布参数**)
            action_mean = elite_actions.mean(dim=0)
            # 添加微小的 epsilon 偏置项以防止系统在多轮迭代后方差彻底坍缩
            action_std = elite_actions.std(dim=0) + 1e-5 
            
        # 依据 MPC 的控制逻辑：仅仅截取规划得到的最优长期序列中的第一个时间步的动作返回
        return action_mean[0]

# 实例化模拟的物理空间环境与对应的求解器进行测试验证
state_dim = 4
action_dim = 2
model = SimulatedWorldModel(state_dim, action_dim)
target = torch.ones(state_dim)

# 设置采样规模 N=1000, 选拔标准 K=100
planner = CEMPlanner(world_model=model, target_state=target, 
                     horizon=15, num_samples=1000, num_elites=100, iterations=5)

# 假设具身智能体现正处于一个纯随机的初始状态
initial_state = torch.rand(state_dim)
best_next_action = planner.plan(initial_state)
print("经历 CEM 多轮迭代收敛后计算出的最优首步执行动作:", best_next_action)
```

## 小结

在本节的探讨中，我们系统性地追溯了具身规划的学术历史，并将高度复杂的模型预测控制思想平缓地降维至经典高中抛体运动的物理直觉层面上。我们利用统计学严密推导了交叉熵方法（CEM）的参数演进法则，剖析了以 MuJoCo MPC 为代表的物理空间规划框架和以 DreamerV3 为代表的隐空间（Latent Space）预测机制的工程落差与一致性。最终，我们从底层纯粹依赖张量计算重写了一套支持大规模并行推断的 MPC 规划器。这不仅解构了具身控制算法的最深层积木，更为后续引入离线强化学习或残差控制奠定了不可或缺的数学与代码基石。

## 练习

1. 在当前 `CEMPlanner` 的实现逻辑中，我们针对每个新的时刻规划时，都粗暴地将 `action_mean` 初始化为绝对的 0。请查阅现代控制论文献（如关于热启动 Warm Start 的讨论），思考应当如何在时间流逝的前提下，巧妙利用上一时刻计算得到的规划结果来对当前时刻的 `action_mean` 提供具有物理连续性的先验初始化？
2. （**提示：考虑迭代后期高斯分布的方差剧烈变化**）随着 CEM 迭代深度的增加，经验方差 `action_std` 将会以极快的速度收敛趋近于零。如果在我们面临的是具有极度非平稳噪声的真实物理实验台，这种过早收敛（Premature Convergence）会引发哪些系统性崩溃？你如何在现有代码的统计重构环节，通过引入指数滑动平均（EMA）平滑系数来抑制这种分布坍塌现象？
3. 根据 [Howell et al., 2022] 的论述，MuJoCo MPC 底层并非完全依赖这种零阶黑盒采样，而是融入了基于动力学梯度的 iLQG 方法。如果在我们的系统中，确信 `SimulatedWorldModel` 所具备的物理转移函数是绝对可微且平滑的，尝试摒弃 CEM 的概率采样框架，直接利用 PyTorch 的反向传播引擎（如 `torch.autograd`）针对目标函数 `cost_function` 对整条连续轨迹进行端到端的梯度下降优化，并对比两种范式在计算耗时与收敛稳定性上的差异。

:begin_tab:pytorch
[讨论](https://discuss.d2l.ai/t/1234)
:end_tab:
