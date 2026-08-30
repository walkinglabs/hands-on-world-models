# 8.6 遥操作、人类在环与 SERL

在探讨了强化学习的基础以及如何在仿真环境中训练智能体之后，我们面临着一个在真实机器人领域中不可回避的残酷现实：强化学习算法往往需要极其庞大的交互样本（Sample Complexity）。在纯仿真环境中，我们可以通过成千上万个并行环境加速数据的收集；然而，当策略必须在真实物理世界中训练时，环境重置的代价、硬件磨损的风险以及探索过程中的安全隐患，使得传统的试错型（Trial-and-Error）强化学习举步维艰。

为了打破这一僵局，学者们回到了一个最直观的起点：向人类学习。[[Argall et al., 2009]](https://doi.org/10.1016/j.robot.2008.10.024) 对机器人从示范中学习（Learning from Demonstration, LfD）进行了系统性的回顾，指出引入人类先验能够极大地缩小策略搜索空间。近年来，随着高频力反馈设备的普及以及人类在环（Human-in-the-Loop, HIL）控制体系的成熟，研究者们将遥操作（Teleoperation）收集的高质量人类示范与深度强化学习深度融合，诞生了诸如 SERL（Sample-Efficient Robotic Reinforcement Learning）[[Luo et al., 2024]](https://arxiv.org/abs/2401.16013) 等一系列工程与算法相结合的先进框架。

本节先讨论遥操作中的空间映射，再分析 DAgger 如何通过在线聚合数据缓解模仿学习的分布偏移 [[Ross et al., 2011]](https://proceedings.mlr.press/v15/ross11a.html)。随后以 AWAC 为例，说明如何用优势加权把离线数据与在线强化学习结合 [[Nair et al., 2020]](https://arxiv.org/abs/2006.09359)。SERL 是软件与训练流程框架，并不等同于 AWAC；它的公开实现主要围绕示范增强的 off-policy 强化学习 [[Luo et al., 2024]](https://arxiv.org/abs/2401.16013)。

## 8.6.1 遥操作的几何映射：主从机器人的空间同构

遥操作的本质是一个控制论中的主从（Master-Slave）跟随问题。人类操作员控制一个主控设备（例如具有力反馈的机械臂或六自由度手柄），产生状态轨迹；而从动设备（真实机器人）需要实时解析这些状态，并在其自身的工作空间中进行相应的动作。

我们可以从高中最基础的二维平面几何谈起。假设在一个二维笛卡尔坐标系中，主控设备的末端点位置为 $\mathbf{p}_m \in \mathbb{R}^2$。当操作员将设备平移 $\Delta \mathbf{p}_m$ 并旋转角度 $\theta$ 时，我们希望从动设备（机器人末端点） $\mathbf{p}_s$ 能够做出同构的运动。

在最简单的情况下，增量映射可以通过一个标量缩放因子 $\alpha > 0$ 来控制运动的灵敏度：

$$
\Delta \mathbf{p}_s = \alpha \Delta \mathbf{p}_m
$$

然而，机器人的实际姿态不仅仅包含平移。当操作员施加旋转时，二维平面上的旋转可以通过一个 $2 \times 2$ 的正交矩阵 $\mathbf{R}(\theta)$ 来描述：

$$
\mathbf{R}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

为了将旋转和平移统一在一个代数结构中，我们在数学上引入齐次坐标（Homogeneous Coordinates），将 $\mathbf{p} = [x, y]^\top$ 扩展为 $\tilde{\mathbf{p}} = [x, y, 1]^\top$。此时，任何仿射变换都可以被写成一个线性矩阵乘法：

$$
\tilde{\mathbf{p}}'_s = \begin{bmatrix} \mathbf{R}(\theta) & \alpha \Delta \mathbf{p}_m \\ \mathbf{0}^\top & 1 \end{bmatrix} \tilde{\mathbf{p}}_s
$$

在真实的三维物理世界中，状态属于特殊欧几里得群 $SE(3)$。主控设备的位姿（Pose）被表示为变换矩阵 $\mathbf{T}_m \in \mathbb{R}^{4 \times 4}$。为了实现平滑的遥操作映射，我们通常通过计算主控设备在相邻时间步 $t$ 和 $t-1$ 之间的相对变换 $\Delta \mathbf{T}_m = \mathbf{T}_{m, t} \mathbf{T}_{m, t-1}^{-1}$，然后将其作用于机器人的当前位姿 $\mathbf{T}_{s, t-1}$，从而求解出目标位姿 $\mathbf{T}_{s, t}$。这一严谨的同构映射确保了无论操作员身处何种相对朝向，机器人的响应在局部坐标系下都是直观且一致的。

## 8.6.2 人类在环（HIL）：从分布偏移到 DAgger

当通过遥操作收集了大量的人类状态-动作轨迹 $\mathcal{D} = \{(s_i, a_i)\}_{i=1}^N$ 后，最直接的学习方式是行为克隆（Behavioral Cloning, BC）。即通过最小化负对数似然来拟合人类的条件概率分布：

$$
J_{\text{BC}}(\theta) = \mathbb{E}_{(s,a) \sim \mathcal{D}} [-\log \pi_\theta(a|s)]
$$

虽然直观，但 BC 在机器人序列决策中面临着灾难性的分布偏移（Distribution Shift）问题。在训练阶段，策略网络 $\pi_\theta$ 仅仅在人类访问过的状态边缘分布 $p_{\text{data}}(s)$ 上进行了优化；而在部署阶段（闭环控制），机器人依据 $\pi_\theta(a|s)$ 采取动作，环境转移概率 $P(s_{t+1}|s_t, a_t)$ 会导致新的状态分布 $p_{\pi_\theta}(s)$。一旦 $p_{\pi_\theta}(s)$ 偏离了 $p_{\text{data}}(s)$，微小的动作误差就会随时间步累积，导致机器人进入一种它在训练数据中从未见过的状态，进而产生更加荒谬的动作，最终导致任务失败。

Ross 等人提出数据集聚合（DAgger, Dataset Aggregation）算法 [[Ross et al., 2011]](https://proceedings.mlr.press/v15/ross11a.html)。DAgger 反复让当前策略访问状态、由专家给这些状态标注动作，再把新样本聚合进训练集；论文用在线学习中的无悔分析说明其性能界。它属于交互式模仿学习，但不等同于所有形式的人类在环控制。

在 DAgger 的第 $k$ 次迭代中，机器人使用当前的策略 $\pi_{\theta_k}$ 在环境中运行，产生轨迹状态 $\{s_t\}_{t=1}^T$。此时，人类操作员（或专家控制器）被引入反馈循环，为这些已经发生偏离的状态提供最优动作修正 $a_t^* = \pi^*(s_t)$。新产生的数据被聚合到总数据集中 $\mathcal{D} \leftarrow \mathcal{D} \cup \{(s_t, a_t^*)\}$，随后在聚合的数据集上重新训练新的策略 $\pi_{\theta_{k+1}}$。这种方法在严格的数学意义上保证了策略能够在自己诱导的状态分布 $p_{\pi_\theta}(s)$ 下拟合专家行为，从而优雅地解决了分布偏移问题。

## 8.6.3 SERL 的核心机制：受限强化学习与优势加权

DAgger 虽然理论上完备，但它要求人类专家能够实时提供极低延迟且高精度的修正动作，这对人类操作员的认知带宽（Cognitive Bandwidth）提出了极高要求。在更现代的框架（如 SERL）中，我们希望仅在任务初期提供有限的遥操作示范，随后让机器人通过自我强化学习（RL）微调，同时保证它在探索过程中不偏离人类所示范的安全行为流形（Behavioral Manifold）。

这种思想可以被严密地形式化为带有 Kullback-Leibler (KL) 散度约束的策略搜索问题。给定一个由人类示范和部分在线交互混合组成的回放缓冲区（Replay Buffer），设其导出的行为分布为 $p_{\text{data}}(a|s)$。我们希望找到一个策略 $\pi$，在最大化动作价值 $Q^\pi(s,a)$ 的同时，其条件概率分布不偏离 $p_{\text{data}}$ 太远。我们写出受限优化目标：

$$
\max_{\pi} \mathbb{E}_{a \sim \pi(\cdot|s)}[Q(s, a)]
$$

受限于：

$$
D_{\text{KL}}(\pi(\cdot|s) \| p_{\text{data}}(\cdot|s)) \le \epsilon
$$

以及概率归一化约束 $\sum_a \pi(a|s) = 1$。我们将这个带有不等式和等式约束的问题，转化为求解拉格朗日泛函（Lagrangian）的无约束极值问题。引入拉格朗日乘子 $\lambda > 0$（对应 KL 约束）和 $\alpha$（对应概率归一化约束）：

$$
\mathcal{L}(\pi, \lambda, \alpha) = \sum_a \pi(a|s) Q(s,a) - \lambda \left( \sum_a \pi(a|s) \log \frac{\pi(a|s)}{p_{\text{data}}(a|s)} - \epsilon \right) - \alpha \left( \sum_a \pi(a|s) - 1 \right)
$$

对未知的分布变量 $\pi(a|s)$ 进行变分求导，并令导数为零：

$$
\frac{\partial \mathcal{L}}{\partial \pi(a|s)} = Q(s,a) - \lambda \left( \log \frac{\pi(a|s)}{p_{\text{data}}(a|s)} + 1 \right) - \alpha = 0
$$

通过代数变换，我们可以解出最优非参数化策略 $\pi^*(a|s)$：

$$
\pi^*(a|s) = p_{\text{data}}(a|s) \exp \left( \frac{Q(s,a) - \alpha - \lambda}{\lambda} \right)
$$

由于 $\pi^*$ 必须是一个合法的概率分布，我们可以将所有不依赖于动作 $a$ 的项吸收进一个配分函数（Partition Function） $Z(s)$ 中。同时，由于任意一个只依赖于状态 $s$ 的基线函数（Baseline）不会改变相对概率大小，我们将 $Q(s,a)$ 减去状态价值 $V(s)$，从而定义优势函数 $A(s,a) = Q(s,a) - V(s)$。这使得最优策略的解析解变得极其优雅：

$$
\pi^*(a|s) = \frac{1}{Z(s)} p_{\text{data}}(a|s) \exp \left( \frac{A(s,a)}{\lambda} \right)
$$

> 我们可以将这种机制理解为一种“橡皮筋效应”：强化学习的奖励最大化目标（正向的优势函数）试图将策略拉向高价值的动作；而 KL 散度约束就像一根连接在人类示范数据（$p_{\text{data}}$）上的橡皮筋，防止策略走得太远而陷入未知的危险状态。随着乘子 $\lambda$（即温度系数）的调节，橡皮筋的拉力发生改变，从而在“探索高价值动作”与“保守模仿专家”之间达成精妙的平衡。

由于真实应用中我们需要拟合一个参数化的神经网络策略 $\pi_\theta(a|s)$，我们将目标转化为最小化 $\pi_\theta$ 偏离最优解 $\pi^*$ 的 KL 散度，这等价于最大化在 $\pi^*$ 下对数似然的期望。利用重要性采样（Importance Sampling），我们可以将期望的采样分布转回我们的数据集分布 $p_{\text{data}}$：

$$
\max_\theta \mathbb{E}_{a \sim p_{\text{data}}} \left[ \frac{\pi^*(a|s)}{p_{\text{data}}(a|s)} \log \pi_\theta(a|s) \right]
$$

将该公式代入上式（忽略常数项 $Z(s)$），我们终于推导出了 AWAC 的核心更新公式：

$$
\max_\theta \mathbb{E}_{s, a \sim \mathcal{D}} \left[ \exp \left( \frac{A(s,a)}{\lambda} \right) \log \pi_\theta(a|s) \right]
$$

在 AWAC 的推导与近似下，策略更新可写成带有 $\exp(A/\lambda)$ 权重的最大似然：优势更高的动作获得更大权重。这解释了 AWAC 如何复用离线数据，但不应据此断言所有机器人样本高效学习框架（包括 SERL）都以 AWAC 为算法基石。

## 8.6.4 代码实现

在实际的工程落地中，我们需要维护一个缓冲区，并同时训练 Actor（策略网络）和 Critic（价值网络）。下面我们将展示优势加权演员-评论家（AWAC）在核心网络更新步骤的 PyTorch 实现。

(**初始化包含优势加权机制的更新步骤**)

```{.python .input}
#@tab pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class AWACUpdate:
    def __init__(self, actor, critic, actor_lr=3e-4, critic_lr=3e-4, lambda_weight=1.0):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.lambda_weight = lambda_weight

    def update_step(self, states, actions, rewards, next_states, dones):
        """
        执行一次 AWAC 的单步梯度更新。
        所有的输入张量形状为 (batch_size, dim)
        """
        # 1. 更新 Critic (标准 TD-Learning 过程)
        with torch.no_grad():
            # 采样下一个动作并计算目标 Q 值
            next_actions = self.actor(next_states).sample()
            target_q = rewards + (1 - dones) * 0.99 * self.critic(next_states, next_actions)

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. 更新 Actor (带有优势加权机制)
        # 估算状态价值 V(s): 通常通过多次采样动作取 Q 值的平均来实现
        with torch.no_grad():
            num_samples = 4
            sampled_actions = [self.actor(states).sample() for _ in range(num_samples)]
            # 计算多个采样动作的 Q 值并取平均作为基线 V(s)
            q_values = torch.stack([self.critic(states, a) for a in sampled_actions], dim=0)
            v_s = q_values.mean(dim=0)

            # 计算优势函数 A(s,a) = Q(s,a) - V(s)
            advantage = current_q - v_s

            # 计算加权系数 exp(A/lambda) 并截断以防数值爆炸
            weights = torch.clamp(torch.exp(advantage / self.lambda_weight), max=100.0)

        # 获取当前策略对数据集动作的对数概率
        log_probs = self.actor(states).log_prob(actions)

        # 乘以常数权重，实现加权的最大似然估计 (等价于最小化加权负对数似然)
        actor_loss = - (weights * log_probs).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()
```

在这段核心逻辑中，我们在计算 Actor 损失时，通过 `torch.exp(advantage / self.lambda_weight)` 精准地再现了推导出的该公式。并且出于工程稳定性考虑，对权重进行了 `torch.clamp` 处理。

## 8.6.5 小结

- 遥操作（Teleoperation）提供了解决强化学习样本效率低下的关键人类先验，其核心依赖于严格的空间 $SE(3)$ 群几何变换与运动学映射。
- 传统的行为克隆由于忽视了环境转移带来的分布偏移（Distribution Shift）问题，容易在长时间控制中失效。人类在环的 DAgger 算法通过在线学习机制为其提供了强力的理论支撑。
- 现代的样本高效机器人框架（如 SERL）将离线人类示范与在线强化学习有机结合。其底层的 AWAC 算法通过带有 KL 散度约束的拉格朗日变分推导，证明了只需通过指数化优势函数加权的对数似然，即可在“模仿”与“探索”之间取得最优平衡。

## 8.6.6 练习

1. 在二维平面遥操作的该公式中，如果我们希望从动机器人的响应不仅仅包含缩放，还在末端施加一个固定的位置偏置 $\mathbf{b}$，变换矩阵应如何修改？
   - **提示**：回忆齐次坐标系中平移向量在矩阵中的位置。
2. 试证明在推导 DAgger 算法的分布偏移时，如果在每一步策略产生错误的概率为 $\epsilon$，那么在 $T$ 步之后产生偏离状态的总概率上界是 $O(T\epsilon)$。
   - **提示**：可以采用数学归纳法或者直接通过联合概率分解证明。
3. 在 AWAC 算法中，如果 $\lambda \to \infty$，该公式中的最优策略 $\pi^*$ 会退化成什么？对应的 Actor 损失函数代表了哪种经典的模仿学习算法？
   - **提示**：计算 $\lim_{\lambda \to \infty} \exp(A(s,a)/\lambda)$，并回顾该公式。
4. 检查代码实现中计算基线状态价值 $V(s)$ 的方式。为何我们通过对 `sampled_actions` 求平均来逼近 $V(s)$，而不是在代码中再定义一个单独的价值网络去训练它？
   - **提示**：思考 $Q(s,a)$ 和 $V(s)$ 在定义上的数学关系：$V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s,a)]$。
