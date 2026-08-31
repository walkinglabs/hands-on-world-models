# 模型预测控制（MPC）与交叉熵方法（CEM）

在强化学习与控制理论的交汇处，如何利用一个已知的（或学习到的）动力学模型来进行最优决策，始终是核心问题之一。在前面的章节中，我们已经探讨了如何从数据中学习环境的动力学模型。现在，假设我们手中已经有了一个可以预测未来的“世界模型”，我们应该如何利用它来寻找最优的动作序列？这正是模型预测控制（Model Predictive Control, MPC）的用武之地。而当面临高维、非线性的复杂动作空间时，交叉熵方法（Cross-Entropy Method, CEM）作为一种强大的无梯度优化算法，成为了MPC在深度强化学习时代的黄金搭档。

## 学术背景与历史溯源

模型预测控制的工程发展可以追溯到 20 世纪 70 年代后期的工业过程控制 [[Richalet et al., 1978]](https://doi.org/10.1016/0005-1098(78)90001-8)。MPC 在线求解有限时域控制问题，并只执行当前最优序列的第一个动作，再在下一时刻重新优化；Garcia 等人的综述系统总结了这一类方法的理论与工业实践 [[Garcia et al., 1989]](https://doi.org/10.1016/0005-1098(89)90002-2)。

随着深度学习的发展，研究人员开始用神经网络拟合复杂的非线性动力学。对这类模型进行动作优化时，可以采用不依赖模型梯度的采样方法。Rubinstein 提出的交叉熵方法（CEM）源于稀有事件模拟与优化 [[Rubinstein, 1997]](https://doi.org/10.1016/S0377-2217(96)00385-2)，后续文献系统整理了它在连续与组合优化中的形式 [[Botev et al., 2013]](https://doi.org/10.1016/B978-0-444-53859-8.00003-5)。PETS [[Chua et al., 2018]](https://arxiv.org/abs/1805.12114) 与 PlaNet [[Hafner et al., 2019]](https://arxiv.org/abs/1811.04551) 都把采样规划与学习到的动力学模型结合起来。

## 模型预测控制（MPC）的数学基础

为了透彻理解MPC，我们先抛开复杂的强化学习环境，回到最基础的高中物理运动学。

### 从一维质点运动开始

想象一个质量为 $m$ 的质点在无摩擦的一维直线上运动。在离散时间 $t$ 下，质点的状态可以用其位置 $x_t$ 和速度 $v_t$ 来描述。我们将其状态定义为 $s_t = [x_t, v_t]^\top$。我们可以对质点施加一个力 $F_t$，这个力就是我们的控制输入，即动作 $a_t$。

根据牛顿第二定律和简单的运动学公式，假设时间间隔为 $\Delta t$，下一个时刻的状态可以表示为：
$$x_{t+1} = x_t + v_t \Delta t + \frac{1}{2m} a_t (\Delta t)^2$$
$$v_{t+1} = v_t + \frac{1}{m} a_t \Delta t$$

我们可以将其抽象为一个离散时间动力学函数（Dynamics Function）：
$$s_{t+1} = f(s_t, a_t)$$

我们的目标是让质点在某个未来的时间点停在一个目标位置 $x_{\text{target}}$。为了量化这个目标，我们定义一个代价函数（Cost Function），它在每个时间步评估当前状态和动作的“糟糕程度”。一个简单的一维代价函数可以是距离目标的平方误差加上对巨大推力的惩罚：
$$c(s_t, a_t) = (x_t - x_{\text{target}})^2 + \lambda a_t^2$$

### 有限时域的最优控制问题

在真实的控制场景中，我们不能只看眼下的一步，而必须向前看若干步。假设我们站在时间步 $t$，我们希望规划未来 $H$ 步的动作序列 $\mathbf{a}_{t:t+H-1} = (a_t, a_{t+1}, \dots, a_{t+H-1})$，使得这段时间内的累积代价最小。这个前瞻的步数 $H$ 被称为**预测时域（Prediction Horizon）**。

累积代价函数 $J$ 可以写为：
$$J(\mathbf{a}_{t:t+H-1} \mid s_t) = \sum_{\tau=t}^{t+H-1} c(s_\tau, a_\tau)$$

注意，这里的未来状态 $s_\tau$ 是由初始状态 $s_t$ 和规划的动作序列通过动力学模型 $f$ 递推生成的。因此，寻找最优动作序列的数学本质，是在给定的非线性等式约束（系统动力学）下求解一个多元函数极值问题：

$$\mathbf{a}^*_{t:t+H-1} = \mathop{\mathrm{argmin}}_{\mathbf{a}_{t:t+H-1}} \sum_{\tau=t}^{t+H-1} c(s_\tau, a_\tau)$$
$$\text{subject to } s_{\tau+1} = f(s_\tau, a_\tau), \quad s_t \text{ given}$$

### 滚动时域控制（Receding Horizon）

如果我们能够完美地解出上述优化问题，得到最优序列 $\mathbf{a}^*_{t:t+H-1} = (a^*_t, a^*_{t+1}, \dots, a^*_{t+H-1})$，我们是否应该直接在接下来的 $H$ 步盲目地执行这个序列呢？

在理想世界中，答案是肯定的。但现实世界中，我们的动力学模型 $f$ 往往是不完美的（存在建模误差），且环境可能存在随机扰动。如果盲目执行长序列，微小的误差会随着时间推移呈指数级累积。

为了克服这个问题，MPC采取了**滚动时域（Receding Horizon）**的策略：

1. 在时间步 $t$，基于当前真实状态 $s_t$，求解优化问题该公式，得到未来 $H$ 步的最优动作序列 $\mathbf{a}^*_{t:t+H-1}$。
2. **只执行该序列的第一个动作** $a_t = a^*_t$，并观察环境返回的真实下一个状态 $s_{t+1}$。
3. 时间步推移到 $t+1$，将预测视窗向前滚动一步，重复步骤1。

这种闭环反馈机制极大地增强了MPC对模型误差的鲁棒性。

## 非线性优化的困境与交叉熵方法（CEM）

当动力学函数 $f$ 是线性系统，且代价函数 $c$ 是二次型时（即LQR问题），我们可以通过Riccati方程精确地求出解析解。然而，在基于模型的深度强化学习中，$f$ 通常是一个深度神经网络。此时，优化问题该公式变得极度非凸，高维空间中布满了局部极小值。

如果我们尝试计算目标函数关于动作序列的梯度 $\nabla_{\mathbf{a}} J$，需要通过神经网络模型进行 $H$ 次时间反向传播（BPTT）。这不仅计算量巨大，还会遇到梯度消失或爆炸。

此时，**无梯度优化（Derivative-Free Optimization）**成为了一种极具吸引力的选择。交叉熵方法（CEM）正是其中的翘楚。

### 交叉熵方法的核心直觉

> “与其盲目地在黑暗的山谷中摸索下坡的路（梯度下降），不如在广袤的地形上随机空投成千上万名伞兵（随机采样）。让那些恰好降落在低谷的伞兵发射信号弹（精英选择），然后指导下一批伞兵集中降落在这些信号弹所在的区域周围。通过不断缩小空投范围，我们最终将精准地锁定最深的谷底。”—— 交叉熵方法在黑盒优化中的几何映射

### CEM的数学推导：从重要性采样到KL散度

假设我们有一个定义在实数域上的随机变量 $\mathbf{x} \in \mathbb{R}^d$（在MPC中，$\mathbf{x}$ 就是长为 $H$ 的动作序列），我们需要最小化一个黑盒函数 $S(\mathbf{x})$（对应于MPC中的累积代价 $J$）。

我们可以将寻找最小值的过程，转化为寻找一个概率分布 $p(\mathbf{x})$，使得该分布几乎所有的概率质量都集中在使 $S(\mathbf{x})$ 极小的区域。

假设我们将“优秀的序列”定义为那些使得代价值小于某个极小阈值 $\gamma$ 的样本：$S(\mathbf{x}) \le \gamma$。我们引入一个指示函数 $I_{\{S(\mathbf{x}) \le \gamma\}}$。我们希望估计出满足这个条件的稀有事件的概率。如果直接在整个空间盲目采样，命中的概率微乎其微。

为了高效采样，我们引入一个参数化的分布簇 $p(\mathbf{x}; \theta)$。为了让这个分布 $p(\mathbf{x}; \theta)$ 尽可能贴近“优秀样本”的真实未知分布，我们需要最小化两者之间的Kullback-Leibler (KL) 散度。

在数学上，CEM 可由稀有事件估计与 KL 散度最小化的视角推出 [[Rubinstein, 1997]](https://doi.org/10.1016/S0377-2217(96)00385-2)。实际算法常把更新写成：最大化“精英样本”在参数化分布下的对数似然。

$$\theta_{k+1} = \mathop{\mathrm{argmax}}_{\theta} \frac{1}{N_e} \sum_{i \in \mathcal{E}_k} \log p(\mathbf{x}_i; \theta)$$

其中，$N_e$ 是精英样本的数量，$\mathcal{E}_k$ 是在第 $k$ 次迭代中通过代价函数排序选出的表现最好的样本集合。

### 高斯分布下的CEM更新公式

在连续控制问题中，我们通常假设 $p(\mathbf{x}; \theta)$ 是一个多元高斯分布 $\mathcal{N}(\mu, \Sigma)$。这意味着参数 $\theta = (\mu, \Sigma)$。

将高斯分布的概率密度函数代入该公式并对 $\mu$ 和 $\Sigma$ 求导置零，我们可以得到极其简洁优美的更新公式。对于动作序列，假设各维度独立，我们只需要更新均值和方差：

新的均值是精英样本的经验均值：
$$\mu_{k+1} = \frac{1}{N_e} \sum_{i \in \mathcal{E}_k} \mathbf{x}_i$$

新的方差是精英样本的经验方差：
$$\sigma^2_{k+1} = \frac{1}{N_e} \sum_{i \in \mathcal{E}_k} (\mathbf{x}_i - \mu_{k+1})^2$$

### MPC-CEM的完整算法循环

将CEM嵌入到MPC的每个时间步中，我们就得到了MPC-CEM算法。在每一个环境时间步 $t$，我们需要执行以下CEM规划过程：

1. **初始化**：设定动作分布的初始均值 $\mu_0 = \mathbf{0}$，初始方差 $\sigma_0^2 = \sigma_{\text{init}}^2 \mathbf{I}$。
2. **迭代优化**（循环 $K$ 次）：
   a. **采样**：从 $\mathcal{N}(\mu_k, \sigma_k^2)$ 中采样 $N$ 个动作序列轨迹 $\mathbf{A}^{(i)}_{t:t+H-1}, i=1 \dots N$。
   b. **前向模拟**：利用学到的神经网络动力学模型 $f_{\phi}$，对于每一个采样的动作序列，结合当前真实状态 $s_t$，在模型中虚拟推演未来 $H$ 步的状态轨迹。
   c. **评估**：计算每条轨迹的累积代价 $J^{(i)} = \sum_{\tau=t}^{t+H-1} c(\hat{s}^{(i)}_\tau, a^{(i)}_\tau)$。
   d. **排序与精英选择**：按照代价从低到高对轨迹进行排序，选出前 $N_e$ 个代价最低的动作序列构成精英集 $\mathcal{E}$。
   e. **分布更新**：利用精英集，通过这两个公式计算新的均值 $\mu_{k+1}$ 和方差 $\sigma_{k+1}^2$。
3. **输出控制**：$K$ 次迭代结束后，将最后一次迭代得到的均值序列的首个动作 $\mu_{K, 0}$ 作为当前时刻的真实执行动作 $a_t$。

## 代码实现

现在，我们用代码严谨地实现上述过程。我们将构建一个简单的批处理动力学模型（模拟真实世界模型），并实现纯粹的矩阵化CEM规划器，完全避免低效的Python循环遍历样本。

(**通过张量运算实现高并行的交叉熵方法规划器**)

```python
import torch
import torch.nn as nn

class SimpleDynamicsModel(nn.Module):
    """一个极其简单的一维质点动力学模型，用于演示
       状态空间: [位置, 速度]
       动作空间: [推力]
    """
    def __init__(self, dt=0.1, mass=1.0):
        super().__init__()
        self.dt = dt
        self.mass = mass

    def forward(self, state, action):
        """
        前向传播计算下一状态。支持批量操作。
        state: 形状为 (batch_size, 2) 的张量
        action: 形状为 (batch_size, 1) 的张量
        """
        pos = state[:, 0:1]
        vel = state[:, 1:2]

        # 物理公式: a = F/m
        acc = action / self.mass

        # 欧拉积分更新速度和位置
        next_vel = vel + acc * self.dt
        next_pos = pos + vel * self.dt + 0.5 * acc * (self.dt ** 2)

        return torch.cat([next_pos, next_vel], dim=-1)

def cost_function(state, action, target_pos=5.0):
    """
    计算给定状态和动作的代价。
    代价 = 距离目标的平方误差 + 动作能量惩罚
    """
    pos = state[:, 0:1]
    # L2 正则化项防止动作过大
    action_penalty = 0.01 * (action ** 2)
    distance_error = (pos - target_pos) ** 2
    return distance_error + action_penalty

class CEMPlanner:
    def __init__(self, dynamics, cost_fn, horizon, num_samples, num_elites, num_iters):
        self.dynamics = dynamics
        self.cost_fn = cost_fn
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iters = num_iters
        self.action_dim = 1 # 针对一维问题

    def plan(self, initial_state):
        """
        基于当前真实状态，使用CEM搜索最优动作序列
        initial_state: 形状为 (state_dim,) 的张量
        """
        # 初始化动作分布参数 (均值为0，方差为1)
        # 形状: (horizon, action_dim)
        mu = torch.zeros((self.horizon, self.action_dim))
        sigma = torch.ones((self.horizon, self.action_dim))

        for _ in range(self.num_iters):
            # 1. 采样: (num_samples, horizon, action_dim)
            # 使用重参数化技巧从当前高斯分布中采样
            epsilon = torch.randn((self.num_samples, self.horizon, self.action_dim))
            actions = mu.unsqueeze(0) + sigma.unsqueeze(0) * epsilon

            # 将动作限制在 [-5, 5] 范围内，保证物理可行性
            actions = torch.clamp(actions, min=-5.0, max=5.0)

            # 2. 前向展开与代价评估
            # 拓展初始状态以匹配样本数量: (num_samples, state_dim)
            current_state = initial_state.unsqueeze(0).repeat(self.num_samples, 1)
            total_costs = torch.zeros(self.num_samples)

            # 逐步预测未来 H 步
            for t in range(self.horizon):
                current_action = actions[:, t, :]
                # 状态向前推演
                current_state = self.dynamics(current_state, current_action)
                # 累加当前步代价
                step_cost = self.cost_fn(current_state, current_action).squeeze(-1)
                total_costs += step_cost

            # 3. 排序与精英选择
            # 获取代价最低的 num_elites 个样本的索引
            _, elite_indices = torch.sort(total_costs)
            elite_indices = elite_indices[:self.num_elites]

            # 提取精英动作序列: (num_elites, horizon, action_dim)
            elite_actions = actions[elite_indices]

            # 4. 更新分布参数
            # 沿着样本维度(dim=0)计算新的经验均值和方差
            mu = elite_actions.mean(dim=0)
            sigma = elite_actions.std(dim=0, unbiased=False)

            # 加上极小值防止方差坍缩为0导致无法继续探索
            sigma = torch.clamp(sigma, min=1e-3)

        # 最终返回均值序列的第一个动作作为MPC的当前输出
        return mu[0]
```

## 小结

在这一章中，我们详细拆解了**模型预测控制（MPC）**的核心原理，从基础的一维运动学问题推广到了有限时域最优控制的数学公式。我们解释了由于深度神经网络模型的高度非线性，传统的基于梯度的优化方法难以奏效。为了克服这一瓶颈，我们引入了**交叉熵方法（CEM）**。

CEM巧妙地将复杂的序列优化问题转化为一个**分布匹配问题**，通过不断地在“概率空间”中对“精英样本”进行最大似然估计，能够高效地在巨大的动作空间中锁定最优解。将CEM与MPC结合，赋予了我们在拥有精确世界模型后的强大规划能力。
