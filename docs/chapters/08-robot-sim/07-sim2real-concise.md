# 8.7 Sim2Real 虚实迁移框架的简洁实现
:label:`sec_sim2real_concise`

在深度强化学习（Deep Reinforcement Learning, DRL）在机器人领域的应用中，我们面临一个根本性的矛盾：现代神经网络需要海量的试错数据来收敛，而在真实的物理世界中让机器人进行数百万次的试错不仅成本极其高昂，还伴随着严重的设备损坏风险。因此，在物理仿真环境（Simulation）中进行训练，随后将策略零样本（Zero-shot）部署到真实物理机器人（Reality）上，成为了当前具身智能的主流范式。这种技术路线被称为 **Sim2Real（虚实迁移）**。

然而，无论仿真引擎的物理模型多么精密，它都无法完美复刻真实世界中复杂的非线性摩擦力、电机迟滞、传感器噪声以及质量分布的微小偏差。这种仿真与真实物理之间的分布偏移，在学术界被称为**现实鸿沟（Reality Gap）**。早期的研究发现，在仿真中表现近乎完美的策略，一旦部署到真机上，往往会因为现实鸿沟而瞬间崩溃。

为了跨越这一鸿沟，[Tobin et al., 2017] 提出了著名的**域随机化（Domain Randomization, DR）**技术，通过在仿真中对环境参数进行大规模的随机扰动，迫使策略学习到鲁棒的特征表示；随后，[Peng et al., 2018] 进一步将其扩展到动力学参数领域（Dynamics Randomization），为后续 OpenAI 解决魔方问题 [Akkaya et al., 2019] 以及 ANYmal 四足机器人跨越复杂地形 [Hwangbo et al., 2019] 奠定了坚实的理论基础。

在本节中，我们将从最基础的高中物理定律出发，一步步为你严密推导现实鸿沟的数学本质，并构建一个基于系统辨识（System Identification）与隐状态推断（Latent State Inference）的简洁 Sim2Real 框架。

## 8.7.1 从经典力学到参数化马尔可夫决策过程

为了彻底理解 Sim2Real 的数学机理，我们不能仅仅停留在抽象的马尔可夫决策过程（MDP）概念上。让我们将视角降维，回到高中物理中最基础的单轴滑块模型。

假设我们控制一个质量为 $m$ 的滑块在水平面上滑动，滑块受到连续的控制推力 $u(t)$。真实世界中，滑块还会受到空气阻力与接触面摩擦力的联合作用。为了便于进行严格的数学解析，我们假设阻力与滑块的速度 $v(t)$ 成正比，比例系数为阻尼常数 $c$。

根据牛顿第二定律，我们可以写出该系统在标量空间下的动力学方程：

$$ m \frac{dv(t)}{dt} = u(t) - c v(t) $$
:eqlabel:`eq_sim2real_newton_scalar`

在公式 :eqref:`eq_sim2real_newton_scalar` 中，等式左侧代表物体的惯性力，右侧代表物体所受的合外力。

在数字控制系统中，控制器的时间是离散的。设定控制周期（时间步长）为 $\Delta t$。根据欧拉积分（Euler Integration）方法，我们可以将速度和位置 $p(t)$ 的导数进行离散化近似：

$$ v_{t+1} = v_t + \Delta t \left( \frac{u_t - c v_t}{m} \right) $$
:eqlabel:`eq_sim2real_euler_v`

$$ p_{t+1} = p_t + \Delta t \cdot v_t $$
:eqlabel:`eq_sim2real_euler_p`

这两个极其简单的标量公式，实际上构成了环境状态转移的最基本单元。然而，在现代强化学习和最优控制中，我们需要以矩阵（张量）的形式来统一处理多维状态。令系统在 $t$ 时刻的状态向量为 $\mathbf{s}_t = [p_t, v_t]^\top$。我们可以将上述连续方程映射为经典的状态空间方程（State-Space Representation）：

$$ \dot{\mathbf{s}}(t) = \mathbf{A}(\xi) \mathbf{s}(t) + \mathbf{B}(\xi) u(t) $$
:eqlabel:`eq_sim2real_state_space`

在这里，我们将系统的物理属性提取为一个**环境参数向量** $\xi = [m, c]^\top$。状态转移矩阵 $\mathbf{A}(\xi)$ 和输入矩阵 $\mathbf{B}(\xi)$ 严格依赖于这个参数向量：

$$ \mathbf{A}(\xi) = \begin{bmatrix} 0 & 1 \\ 0 & -\frac{c}{m} \end{bmatrix}, \quad \mathbf{B}(\xi) = \begin{bmatrix} 0 \\ \frac{1}{m} \end{bmatrix} $$
:eqlabel:`eq_sim2real_matrices`

结合欧拉离散化，最终的离散状态转移方程（即强化学习中环境的 `step` 函数的核心数学模型）可以严格表达为：

$$ \mathbf{s}_{t+1} = (\mathbf{I} + \mathbf{A}(\xi) \Delta t) \mathbf{s}_t + (\mathbf{B}(\xi) \Delta t) u_t $$
:eqlabel:`eq_sim2real_discrete_matrix`

至此，我们将一个具体的物理问题，严格地抽象成了一个**参数化马尔可夫决策过程（Parameterized MDP, Param-MDP）**。其状态转移概率分布 $\mathcal{P}(\mathbf{s}_{t+1} | \mathbf{s}_t, u_t; \xi)$ 完全由隐藏的物理参数 $\xi$ 决定。

## 8.7.2 现实鸿沟的泰勒展开分析与域随机化

既然物理模型已经如此清晰，现实鸿沟究竟是从哪里产生的呢？

假设我们在仿真环境中构建了上述模型，并设定了一组标称参数 $\xi_{sim} = [m_{sim}, c_{sim}]^\top$。我们使用强化学习算法（如 PPO）优化策略参数 $\theta$，以最大化累积期望回报 $J(\theta, \xi)$。此时，训练得到的全知策略 $\theta^*$ 是在 $\xi_{sim}$ 上的局部最优解。

当我们把 $\theta^*$ 部署到真实的物理环境中时，真实世界的参数总是存在未知的偏差，即 $\xi_{real} = \xi_{sim} + \Delta \xi$。在标称参数附近，对价值函数 $J(\theta^*, \xi_{real})$ 进行二阶泰勒展开（Taylor Expansion）：

$$ J(\theta^*, \xi_{real}) \approx J(\theta^*, \xi_{sim}) + \nabla_\xi J(\theta^*, \xi)^\top \Delta \xi + \frac{1}{2} \Delta \xi^\top \mathbf{H}_\xi \Delta \xi $$
:eqlabel:`eq_sim2real_taylor`

其中 $\mathbf{H}_\xi$ 是价值函数关于环境参数的海森矩阵（Hessian Matrix）。
在纯粹固定的仿真环境 $\xi_{sim}$ 中训练的神经网络，为了追求极致的回报，往往会过度利用当前环境的特定动力学特性（即发生了过拟合）。在优化景观（Optimization Landscape）上，这就表现为一个极其尖锐的山峰——这意味着海森矩阵 $\mathbf{H}_\xi$ 具有非常大的负特征值。此时，即使 $\Delta \xi$ 极其微小，二次项 $\frac{1}{2} \Delta \xi^\top \mathbf{H}_\xi \Delta \xi$ 也会产生巨大的负面惩罚，导致策略在真实环境中瞬间崩溃。

**域随机化（Domain Randomization, DR）**技术的数学思想，就是改变优化的目标函数。我们不再针对单一的 $\xi_{sim}$ 进行优化，而是设定一个参数分布 $P_\Phi(\xi)$（例如均匀分布或高斯分布），并优化在该分布下的期望回报：

$$ J_{DR}(\theta) = \mathbb{E}_{\xi \sim P_\Phi}[J(\theta, \xi)] = \int P_\Phi(\xi) \mathbb{E}_{\tau \sim \pi_\theta, \mathcal{P}_\xi} \left[ \sum_{t=0}^T \gamma^t r(\mathbf{s}_t, u_t) \right] d\xi $$
:eqlabel:`eq_sim2real_dr_obj`

通过引入对分布的积分，目标函数在参数空间中被强制进行了平滑（类似于数学中的卷积平滑）。优化器为了在宽广的分布上都能获得较高的平均回报，必须寻找一个宽阔而平缓的高原（即减小 $\mathbf{H}_\xi$ 的负特征值）。这样训练出的策略，即使面临 $\Delta \xi$ 的偏差，其性能下降也是平缓且可控的。

## 8.7.3 自适应控制：历史序列与隐状态推断

虽然域随机化显著提升了策略的鲁棒性，但它存在一个致命弱点：过于保守。为了在各种极端的参数下都不至于失败，策略往往会采取极度保守的动作，牺牲了运动的敏捷性和最优性。

为了打破“鲁棒性”与“敏捷性”之间的权衡，我们需要让策略在运行过程中**动态地识别**出当前的真实环境参数 $\xi_{real}$。这在控制论中被称为系统辨识（System Identification）。但真实机器人的传感器通常只能测量位置和速度（状态 $\mathbf{s}_t$），无法直接测量系统的整体质量或接触面的摩擦常数。我们必须通过间接的方式进行推断。

> 💡 **核心机制类比：蒙眼过河的试探者**
> 想象你在黑暗中蒙眼行走（策略网络无法直接观测真实世界的物理参数 $\xi$）。你并不知道地面的摩擦力是冰面还是泥地。但当你迈出第一步（执行动作 $u_{t-1}$）并感受到身体的滑动程度（观测到新的状态 $\mathbf{s}_t$）时，你的大脑会根据“预期位移与实际位移的绝对偏差”瞬间推断出地面的材质（历史编码器提取隐变量 $\mathbf{z}_t$）。随后的每一步，你都基于这个不断修正的内部认知来调整步伐（自适应策略 $\pi(u_t|\mathbf{s}_t, \mathbf{z}_t)$）。这正是 Sim2Real 中基于历史轨迹进行在线系统辨识的本质：通过动作与状态的时序交互，在隐空间中逆向求解物理方程的未知系数。

在数学上，我们将过去 $k$ 步的状态与动作定义为历史轨迹窗口（History Trajectory Window）：

$$ \mathbf{h}_t = (\mathbf{s}_{t-k}, u_{t-k}, \dots, \mathbf{s}_{t-1}, u_{t-1}, \mathbf{s}_t) $$
:eqlabel:`eq_sim2real_history`

我们引入一个历史编码器（通常是 RNN 或 Transformer）$f_\phi$，将高维的时序信息压缩为一个低维的隐变量（Latent Variable）$\mathbf{z}_t$：

$$ \mathbf{z}_t = f_\phi(\mathbf{h}_t) $$
:eqlabel:`eq_sim2real_latent`

在训练时，编码器 $f_\phi$ 与策略网络 $\pi_\theta(u_t | \mathbf{s}_t, \mathbf{z}_t)$ 进行端到端的联合优化。网络被强制要求通过过去的动力学响应 $\mathbf{h}_t$ 去隐式地逆推出环境的真实参数分布，从而实现自适应的闭环控制。

## 8.7.4 简洁框架的代码实现

现在，我们将上述所有的严密数学推导转化为具体的代码实现。我们将构建一个基于域随机化的物理环境，并使用一个带有 GRU（门控循环单元）的自适应策略网络来处理时序信息。

[**定义带有物理参数随机化的环境包装器**]

环境的 `sample_params` 函数体现了分布 $P_\Phi(\xi)$ 的采样过程，而 `step` 函数则严格对应了公式 :eqref:`eq_sim2real_discrete_matrix`。

```{.python .input}
#@tab pytorch
import torch
from torch import nn

class RandomizedLinearEnv(nn.Module):
    def __init__(self, dt=0.05):
        """
        基于标量张量化的极简参数化物理仿真环境。
        dt: 离散化的时间步长。
        """
        super().__init__()
        self.dt = dt

    def sample_params(self, batch_size):
        """在每次回合开始时，从均匀分布中采样质量 m 和阻尼系数 c。"""
        # m ~ U(0.5, 1.5), c ~ U(0.1, 0.5)
        m = torch.rand(batch_size, 1) * 1.0 + 0.5
        c = torch.rand(batch_size, 1) * 0.4 + 0.1
        return m, c

    def step(self, state, action, m, c):
        """
        执行欧拉积分状态转移，对应公式 (8.7.4) 和 (8.7.5)。
        state: 包含位置 p 和速度 v 的张量，形状 [batch_size, 2]
        action: 控制推力 u，形状 [batch_size, 1]
        m, c: 当前采样的物理参数，形状 [batch_size, 1]
        """
        p, v = state[:, 0:1], state[:, 1:2]
        
        # 严格对应连续牛顿定律离散化后的张量计算
        v_next = v + self.dt * (action - c * v) / m
        p_next = p + self.dt * v
        
        return torch.cat([p_next, v_next], dim=-1)
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

class RandomizedLinearEnv(tf.keras.Model):
    def __init__(self, dt=0.05):
        super().__init__()
        self.dt = dt

    def sample_params(self, batch_size):
        """在 TensorFlow 中使用 random.uniform 采样域随机化参数。"""
        m = tf.random.uniform((batch_size, 1), minval=0.5, maxval=1.5)
        c = tf.random.uniform((batch_size, 1), minval=0.1, maxval=0.5)
        return m, c

    def step(self, state, action, m, c):
        p = state[:, 0:1]
        v = state[:, 1:2]
        
        v_next = v + self.dt * (action - c * v) / m
        p_next = p + self.dt * v
        
        return tf.concat([p_next, v_next], axis=-1)
```

[**构建隐状态推断网络与鲁棒策略**]

根据公式 :eqref:`eq_sim2real_latent`，我们将历史序列 $\mathbf{h}_t$ 输入到一个 GRU 编码器中以提取系统辨识特征 $\mathbf{z}_t$。随后，策略网络将当前即时状态 $\mathbf{s}_t$ 与隐变量 $\mathbf{z}_t$ 拼接，决定下一步的动作。

```{.python .input}
#@tab pytorch
class HistoryEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, latent_dim):
        super().__init__()
        # GRU 编码器接收状态与动作的联合输入
        self.rnn = nn.GRU(state_dim + action_dim, hidden_dim, batch_first=True)
        # 将时序隐藏状态映射为物理隐变量 z_t
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, history):
        # history: [batch_size, seq_len, state_dim + action_dim]
        _, h_n = self.rnn(history)
        # h_n 形状: [1, batch_size, hidden_dim]，去除第0维后经过全连接层
        latent = self.fc(h_n.squeeze(0))
        return latent

class RobustPolicy(nn.Module):
    def __init__(self, state_dim, latent_dim, action_dim):
        super().__init__()
        # 策略网络：合并显式观测与隐式系统辨识特征
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, state, latent):
        x = torch.cat([state, latent], dim=-1)
        return self.net(x)
```

```{.python .input}
#@tab tensorflow
class HistoryEncoder(tf.keras.Model):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.rnn = tf.keras.layers.GRU(hidden_dim, return_state=True)
        self.fc = tf.keras.layers.Dense(latent_dim)
        
    def call(self, history):
        # rnn_output 形状为 [batch_size, hidden_dim]，已是最终状态
        _, h_n = self.rnn(history)
        latent = self.fc(h_n)
        return latent

class RobustPolicy(tf.keras.Model):
    def __init__(self, action_dim):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(action_dim)
        ])
        
    def call(self, state, latent):
        x = tf.concat([state, latent], axis=-1)
        return self.net(x)
```

[**定义 Sim2Real 的联合优化过程**]

在此，我们提供一个简化的训练循环骨架。该循环模拟了收集随机化仿真轨迹，并对整个自适应控制图进行联合反向传播的流程。在真实的系统中，这一步通常由 PPO 等强化学习算法的 Critic 误差驱动，此处我们为了简明，将其抽象为迫使滑块移动到目标坐标的均方误差损失。

```{.python .input}
#@tab pytorch
def train_sim2real_step(env, encoder, policy, optimizer, batch_size=32, seq_len=10):
    # 1. 从 P_Phi(xi) 分布中采样环境的随机物理参数（真实但不告诉策略）
    m, c = env.sample_params(batch_size)
    
    # 2. 初始化状态，并创建用于存储历史信息的张量
    state = torch.zeros(batch_size, 2)
    # 每一帧包含 2 维状态和 1 维动作
    history = torch.zeros(batch_size, seq_len, 3) 
    
    loss = 0
    # 在时间步内进行自回归式滚动计算
    for t in range(seq_len):
        # 3. 基于截至目前的历史窗口推断隐变量 z_t
        latent = encoder(history)
        
        # 4. 基于当前状态 s_t 与隐变量 z_t 输出控制命令 u_t
        action = policy(state, latent)
        
        # 5. 环境基于底层的隐藏物理方程进行演化计算出 s_{t+1}
        next_state = env.step(state, action, m, c)
        
        # 更新历史窗口：将当前的 state 和 action 存入
        history_step = torch.cat([state, action], dim=-1)
        history[:, t, :] = history_step
        
        # 目标：将滑块稳定在 p=1.0 的位置并速度归零 (v=0)
        target = torch.tensor([[1.0, 0.0]])
        loss += torch.mean((next_state - target) ** 2)
        
        state = next_state
        
    # 6. 端到端联合优化系统辨识模块与策略网络
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# 实例化环境与网络模型
env = RandomizedLinearEnv()
encoder = HistoryEncoder(state_dim=2, action_dim=1, hidden_dim=32, latent_dim=8)
policy = RobustPolicy(state_dim=2, latent_dim=8, action_dim=1)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(policy.parameters()), lr=0.001)

# 执行单步优化以观察系统运转
print(f"Loss after one step: {train_sim2real_step(env, encoder, policy, optimizer):.4f}")
```

```{.python .input}
#@tab tensorflow
def train_sim2real_step(env, encoder, policy, optimizer, batch_size=32, seq_len=10):
    m, c = env.sample_params(batch_size)
    state = tf.zeros((batch_size, 2))
    
    with tf.GradientTape() as tape:
        loss = 0.0
        # TensorFlow 中我们需要借助 tensor_scatter_nd_update 对历史窗口进行在位更新
        history = tf.zeros((batch_size, seq_len, 3))
        
        for t in range(seq_len):
            latent = encoder(history)
            action = policy(state, latent)
            next_state = env.step(state, action, m, c)
            
            indices = tf.stack([tf.range(batch_size), tf.fill([batch_size], t)], axis=1)
            history_step = tf.concat([state, action], axis=-1)
            history = tf.tensor_scatter_nd_update(history, indices, history_step)
            
            target = tf.constant([[1.0, 0.0]])
            loss += tf.reduce_mean((next_state - target) ** 2)
            state = next_state
            
    # 聚合所有可训练参数，执行统一的梯度下降
    variables = encoder.trainable_variables + policy.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    return loss.numpy()

env = RandomizedLinearEnv()
encoder = HistoryEncoder(hidden_dim=32, latent_dim=8)
policy = RobustPolicy(action_dim=1)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

print(f"Loss after one step: {train_sim2real_step(env, encoder, policy, optimizer):.4f}")
```

## 8.7.5 小结

1. **现实鸿沟的数学根源**在于，针对单一参数 $\xi_{sim}$ 优化的策略网络具有极度狭窄的最优解山峰结构，使得价值函数关于物理参数的海森矩阵存在极大的负特征值，极其微小的现实偏差都会引起性能崩溃。
2. **域随机化（Domain Randomization）**通过引入概率积分项，平滑了参数空间的优化景观。
3. **系统辨识与自适应控制**则进一步打破了鲁棒与敏捷之间的零和博弈。利用循环神经网络（RNN）从状态与动作的历史交织序列中提取隐状态 $\mathbf{z}_t$，我们成功赋予了智能体“蒙眼感知物理材质”的高阶智能，使得 Sim2Real 的零样本迁移成为可能。

## 8.7.6 练习

1. **尝试扩展物理模型**：在当前的滑块模型中，加入静摩擦力的非线性突变效应。提示：你需要使用阶跃函数或 `torch.sign()`，但请注意这可能会导致梯度在原点处不可导。思考在使用 DRL（无梯度黑盒优化）时，这个非线性特性会对策略学习产生什么影响？
2. **探索海森矩阵的几何意义**：在公式 :eqref:`eq_sim2real_taylor` 中，如果海森矩阵 $\mathbf{H}_\xi$ 极其接近于零矩阵，这在优化景观上意味着什么？这种地形对于现实物理部署有什么独特的优势？
3. **分析隐变量的分布**：如果在训练结束后，你将真实环境的质量 $m$ 从 0.5 连续调节至 1.5，并记录编码器输出的隐变量 $\mathbf{z}_t$ 的均值。你预计 $\mathbf{z}_t$ 的几何分布（如使用 PCA 降维）与真实的 $m$ 值会呈现出怎样的数学关联？

:begin_tab:pytorch
[讨论](https://discuss.d2l.ai/t/sim2real-concise-implementation)
:end_tab:
