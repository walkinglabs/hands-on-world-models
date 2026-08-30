# 具身规划的从零开始实现

在探讨深度学习与机器人学的交叉领域时，让智能体在复杂的物理世界中进行自主决策与行动，始终是该领域的核心命题。早期的研究主要依赖于经典的运动规划算法（例如A*搜索与快速随机搜索树RRT），或是基于严密解析数学模型的模型预测控制（Model Predictive Control, MPC） `[Camacho & Alba, 2013]`。然而，当机器人走出结构化的实验室，面对充满未知与噪声的开放世界时，构建完美且精确的物理动力学方程往往变得极其困难，甚至是不可能的。

近年来，随着深度学习的发展与“世界模型”（World Models）概念的提出，具身规划（Embodied Planning）的范式发生了深刻的变革。研究者们不再依赖人工预定义的刚体动力学规则，而是让智能体通过大量的数据交互，在神经网络内部学习出一个可微的环境动力学模型（Dynamics Model）。在这个习得的“隐式世界”中进行“想象”与规划，成为了诸如 PETS `[Chua et al., 2018]`、Dreamer `[Hafner et al., 2019]`，乃至近年来基于视觉-语言模型的具身控制（如 RT-1 `[Brohan et al., 2022]`）等经典工作的主流路径。

在本节中，我们将剥离掉复杂的视觉编码器和庞大的语言模型，回归具身规划最纯粹的数学与算法本质。我们将从高中物理中最基础的运动学出发，严密推导并从零开始实现一个基于世界模型和交叉熵方法（Cross-Entropy Method, CEM `[Rubinstein, 1997]`）的具身规划器。

## 场景构建与一维物理映射

为了透彻理解具身规划的本质，我们绝对不能一开始就陷入高维张量的泥潭。让我们将复杂的机器人操作降维，回到高中物理课本中最经典的一维运动学场景。

假设我们控制着一台只能在笔直的一维轨道上移动的微型机器人。在任意时刻 $t$，机器人的状态可以由两个标量完全描述：它在轨道上的位置 $x_t$ 和当前的瞬时速度 $v_t$。机器人唯一能够做出的动作（Action），就是通过电机施加一个持续一小段时间 $\Delta t$ 的加速度 $a_t$。

根据最基础的匀加速直线运动规律，如果我们在时间间隔 $\Delta t$ 内施加恒定的加速度 $a_t$，机器人在下一时刻 $t+1$ 的速度和位置将遵循以下差分方程：

$$v_{t+1} = v_t + a_t \Delta t$$
$$x_{t+1} = x_t + v_t \Delta t + \frac{1}{2} a_t (\Delta t)^2$$

为了数学上的简洁与后续向量化的需要，我们通常会假设时间步长 $\Delta t$ 极小，从而忽略掉二次项 $\frac{1}{2} a_t (\Delta t)^2$。现在，我们定义系统的状态向量为 $\mathbf{s}_t = \begin{bmatrix} x_t \\ v_t \end{bmatrix}$，动作向量为 $\mathbf{u}_t = \begin{bmatrix} a_t \end{bmatrix}$。那么，上述标量方程可以顺理成章地重写为严谨的矩阵乘法形式：

$$\begin{bmatrix} x_{t+1} \\ v_{t+1} \end{bmatrix} = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x_t \\ v_t \end{bmatrix} + \begin{bmatrix} 0 \\ \Delta t \end{bmatrix} a_t$$

更一般地，对于任何线性时不变系统，我们可以将其抽象为状态转移方程：

$$\mathbf{s}_{t+1} = \mathbf{A} \mathbf{s}_t + \mathbf{B} \mathbf{u}_t$$

## 泛化至非线性环境动力学模型

上述方程 :eqref:eq_linear_dynamics 极其优美，但遗憾的是，真实的具身环境（例如机械臂的摩擦力、流体阻力、柔性形变）几乎全是非线性的。因此，具身规划的核心突破，就是**使用深度神经网络来近似并替代这个转移矩阵**。

我们将真实世界中复杂的状态转移过程定义为一个未知的非线性函数 $\mathcal{F}$。通过收集大量机器人在环境中的历史转移数据 $\{(\mathbf{s}_t, \mathbf{u}_t, \mathbf{s}_{t+1})\}$，我们可以训练一个参数为 $\theta$ 的神经网络 $f_\theta$：

$$\mathbf{s}_{t+1} = f_\theta(\mathbf{s}_t, \mathbf{u}_t)$$

这里的 $f_\theta$ 就是我们常说的**前向动力学模型**（Forward Dynamics Model），也是世界模型中最核心的组成部分。它赋予了智能体“预测未来”的能力：给定当前状态和我们即将采取的动作，模型能够推演出下一步的状态。

## 规划目标的数学表达

拥有了预测未来的能力后，我们需要一种数学语言来描述“任务目标”。在具身规划中，目标通常被形式化为一个**代价函数**（Cost Function） $c(\mathbf{s}_t, \mathbf{u}_t)$。代价越低，说明状态越符合我们的期望。

延续一维轨道的场景，假设我们的目标是让机器人停靠在位置 $x_{goal}$ 处，并且希望它停下时不要过于剧烈地晃动（即速度 $v$ 应当接近0），同时为了节省电量，我们希望加速度（动作）越小越好。我们可以构建如下的单步代价函数：

$$c(\mathbf{s}_t, \mathbf{u}_t) = w_1 (x_t - x_{goal})^2 + w_2 v_t^2 + w_3 a_t^2$$

其中，$w_1, w_2, w_3$ 是调节各项重要性的权重常数。

然而，具身规划不能只看眼前的一步。我们需要在一个规划视界（Planning Horizon） $H$ 内进行长远考虑。这就引出了规划的最终数学目标——寻找一条长度为 $H$ 的最优动作序列 $\mathbf{U}_{1:H}^* = \{\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_H\}$，使得累积代价 $J$ 最小化：

$$\mathbf{U}_{1:H}^* = \mathop{\arg\min}_{\mathbf{U}_{1:H}} \sum_{k=1}^{H} c(\mathbf{s}_{t+k}, \mathbf{u}_{t+k})$$

约束条件为未来的状态必须由动力学模型 :eqref:eq_neural_dynamics 逐步生成。

## 交叉熵方法（CEM）的严密推导

面对等式 :eqref:eq_trajectory_cost，如果动作空间是离散且有限的（比如只能选“左移”、“右移”、“不动”），我们可以使用广度优先搜索或A*算法穷举所有可能的序列。但是，真实的机器人控制面临的是**高维且连续的动作空间**（例如控制7个机械臂关节的力矩），搜索空间呈指数级爆炸，穷举法彻底失效。

为了在这个非凸、高维的连续空间中找到最小化累积代价的动作序列，我们需要引入一种强大的无导数优化算法：**交叉熵方法（Cross-Entropy Method, CEM）**。

> **类比**：CEM 的迭代优化过程，就如同炮兵团在浓雾中对未知坐标的敌军阵地进行“火力试探与校射”。第一轮，炮兵们基于大致的方位（初始均值）和较大的散布范围（初始方差）盲目地发射一批炮弹（随机采样动作序列）。随后，前线观察哨汇报哪些炮弹落得离目标最近（评估代价并筛选精英样本）。炮兵指挥官立刻根据这些最精准的弹着点，重新计算炮管的瞄准中心（更新均值），并极大地缩小试探的散布范围（减小方差）。如此反复多轮，炮火便会以极快的速度收敛，最终精准覆盖真正的目标。

在严格的数学表述下，CEM 将求解最优序列的问题，转化为估计最优动作序列分布的问题。我们假设在视界 $H$ 内，每个时间步的动作服从高斯分布 $\mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$。CEM 的一次迭代周期包含以下三个严密的步骤：

1. **采样（Sampling）**：从当前的分布 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ 中，独立同分布地采样 $N$ 条完整的动作序列轨迹（即 $N$ 发“炮弹”）：
   $$\mathbf{U}^{(i)} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), \quad i \in \{1, 2, \dots, N\}$$

2. **评估与排序（Evaluation & Sorting）**：利用动力学模型 $f_\theta$ 展开（Rollout）这 $N$ 条序列，计算每一条序列的累积代价 $J(\mathbf{U}^{(i)})$。随后按代价从小到大排序。

3. **更新（Updating）**：选取代价最低的前 $K$ 条序列作为“精英样本”（Elites），记作 $\mathbf{U}_{elite}$。利用这 $K$ 个精英样本的经验统计量，对高斯分布的均值和方差进行极大似然更新：
   $$\boldsymbol{\mu}_{new} = \frac{1}{K} \sum_{j=1}^K \mathbf{U}_{elite}^{(j)}$$
   $$\boldsymbol{\Sigma}_{new} = \frac{1}{K} \sum_{j=1}^K \left( \mathbf{U}_{elite}^{(j)} - \boldsymbol{\mu}_{new} \right) \left( \mathbf{U}_{elite}^{(j)} - \boldsymbol{\mu}_{new} \right)^\top$$

不断重复上述三个步骤 $M$ 次，分布的方差 $\boldsymbol{\Sigma}$ 会不断缩小，均值 $\boldsymbol{\mu}$ 会迅速逼近等式 :eqref:eq_trajectory_cost 的最优解。

## 从零构建具身规划器代码

现在，让我们将上述严密的数学推演转化为具体的张量计算。我们采用模型预测控制（MPC）的范式：在当前时间步，规划出未来 $H$ 步的动作序列，但**仅仅执行序列中的第一个动作**。环境进入新状态后，我们重新进行规划。这种机制能够有效地抵御动力学模型不精确带来的累积误差。

(**首先，我们定义一维环境的代理动力学模型与代价函数。**)

```{.python .input}
#@tab pytorch
import torch
from torch import nn

class SimpleDynamicsModel(nn.Module):
    """一个极为简化的非线性动力学模型代理，替代真实的复杂物理世界"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 实际应用中，这里会是一个通过大量数据预训练的深度残差网络
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim)
        )
        
    def forward(self, state, action):
        # 模型的输入是状态和动作的拼接，输出是状态的变化量 (残差连接)
        # state: [batch_size, state_dim]
        # action: [batch_size, action_dim]
        x = torch.cat([state, action], dim=-1)
        state_delta = self.net(x)
        return state + state_delta

def cost_function(states, actions, target_state):
    """
    计算规划轨迹的累积代价。
    states: [batch_size, horizon, state_dim]
    actions: [batch_size, horizon, action_dim]
    """
    # 状态的L2范数惩罚：距离目标状态越远，代价越大
    state_cost = torch.sum((states - target_state) ** 2, dim=-1) # [batch_size, horizon]
    
    # 动作的L2范数惩罚：约束能量消耗，鼓励平滑控制
    action_cost = torch.sum(actions ** 2, dim=-1) # [batch_size, horizon]
    
    # 将时间视界 H 内的代价进行累加
    # 最终返回形状: [batch_size]
    total_cost = torch.sum(state_cost + 0.1 * action_cost, dim=-1)
    return total_cost
```

(**接下来，我们实现最核心的交叉熵规划器 (CEM Planner)。**)
请务必仔细阅读代码中对张量维度的详细注释。

```{.python .input}
#@tab pytorch
class CEMPlanner:
    def __init__(self, dynamics_model, action_dim, horizon=15, 
                 num_samples=200, num_elites=20, num_iterations=5):
        self.model = dynamics_model
        self.action_dim = action_dim
        self.horizon = horizon          # 规划视界 H
        self.num_samples = num_samples  # 每次采样的序列总数 N
        self.num_elites = num_elites    # 筛选的精英样本数 K
        self.num_iters = num_iterations # CEM的迭代次数 M
        
    def plan(self, initial_state, target_state):
        """
        给定当前状态和目标，计算并返回当前最优的一步动作
        initial_state: [state_dim]
        target_state: [state_dim]
        """
        # 初始化动作序列的高斯分布参数
        # 均值 mu 初始化为0，形状: [horizon, action_dim]
        action_mu = torch.zeros(self.horizon, self.action_dim)
        # 方差 sigma 初始化为1 (标准正态分布)，形状: [horizon, action_dim]
        action_std = torch.ones(self.horizon, self.action_dim)
        
        # 将输入提升为批次维度以便并行推理
        # [state_dim] -> [num_samples, state_dim]
        state_batch = initial_state.unsqueeze(0).repeat(self.num_samples, 1)
        target_batch = target_state.unsqueeze(0).repeat(self.num_samples, 1)
        
        for i in range(self.num_iters):
            # 步骤1：采样 N 条完整的动作轨迹
            # sampled_actions: [num_samples, horizon, action_dim]
            sampled_actions = action_mu.unsqueeze(0) + \
                              action_std.unsqueeze(0) * torch.randn(
                                  self.num_samples, self.horizon, self.action_dim)
            
            # 限制动作的物理边界（如电机的最大输出）假设在 [-1, 1] 之间
            sampled_actions = torch.clamp(sampled_actions, min=-1.0, max=1.0)
            
            # 用于记录模型在 H 步内预测出的所有状态
            predicted_states = []
            current_states = state_batch.clone()
            
            # 在前向动力学模型中展开轨迹
            for t in range(self.horizon):
                # 取出第 t 步的所有采样动作
                curr_actions = sampled_actions[:, t, :] # [num_samples, action_dim]
                # 动力学模型预测下一步状态
                next_states = self.model(current_states, curr_actions)
                predicted_states.append(next_states.unsqueeze(1))
                current_states = next_states
                
            # 拼接得到完整的状态序列预测
            # all_predicted_states: [num_samples, horizon, state_dim]
            all_predicted_states = torch.cat(predicted_states, dim=1)
            
            # 步骤2：评估并排序
            # costs: [num_samples]
            costs = cost_function(all_predicted_states, sampled_actions, target_batch)
            
            # 找到代价最小的前 K 个样本的索引
            _, elite_indices = torch.topk(costs, self.num_elites, largest=False)
            
            # 提取精英样本的动作序列
            # elite_actions: [num_elites, horizon, action_dim]
            elite_actions = sampled_actions[elite_indices]
            
            # 步骤3：更新分布参数
            # 计算新的均值和标准差 (跨 batch_size 维度求均值/方差)
            action_mu = torch.mean(elite_actions, dim=0)   # [horizon, action_dim]
            action_std = torch.std(elite_actions, dim=0)   # [horizon, action_dim]
            
            # 实际工程中，为了防止方差过早收敛导致陷入局部最优，
            # 通常会在标准差上增加一个随时间衰减的极小噪声，此处为保持简洁省略。
            
        # 经过 M 轮迭代后，返回最优均值序列中的第一个动作
        # 这正是模型预测控制（MPC）的核心思想
        return action_mu[0]
```

通过上述严密的设计，我们的机器人不再是盲目试错，而是具备了在“大脑”（世界模型）中预演无数种未来，并沿着代价最低的那条世界线坚定前行的能力。这也正是现代复杂具身智能体能够完成折叠衣服、灵巧抓取等惊艳操作的基石原理。
