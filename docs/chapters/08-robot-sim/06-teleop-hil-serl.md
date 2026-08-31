# 8.6 遥操作、人类在环与 SERL

一次抓取失败后，人可能只需把物体放回原位；机器人系统却还要检测失败、复位机械臂、检查碰撞并重新开始。真实交互的瓶颈不只是一秒能执行多少动作，还包括复位、安全监控和硬件维护。

<div align="center">
<img src="/figures/08-robot-sim/source/06-teleop-hil-serl/hilserl-fig1.png" alt="HIL-SERL 在插接、装配和精细操作任务中让人类干预直接修补真实机器人探索。" width="86%">

_图 8.6-1：HIL-SERL 在插接、装配和精细操作任务中让人类干预直接修补真实机器人探索。 出处：Lawrence Yunliang Chen et al.，[HIL-SERL: Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning](https://arxiv.org/abs/2410.21845)（2024），Figure 1。_
</div>

一种做法是先用遥操作收集成功示范，再让策略在线练习；当机器人将要进入危险或无效状态时，人类可以接管并补充纠正数据。[[Argall et al., 2009]](https://doi.org/10.1016/j.robot.2008.10.024) 总结了机器人示范学习，SERL 则把示范增强的 off-policy 强化学习、并行数据采集和训练工具组织为真实机器人学习流程 [[Luo et al., 2024]](https://arxiv.org/abs/2401.16013)。

<div align="center">
<img src="/figures/08-robot-sim/source/06-teleop-hil-serl/haco-fig1.png" alt="HACO 的人机共驾图展示人类在危险动作出现时接管，并把干预反馈用于安全策略学习。" width="86%">

_图 8.6-2：HACO 的人机共驾图展示人类在危险动作出现时接管，并把干预反馈用于安全策略学习。 出处：Zhizheng Liu et al.，[Human-AI Copilot Optimization for Safe Reinforcement Learning](https://arxiv.org/abs/2202.10341)（2022），Figure 1。_
</div>

本节先讨论遥操作中的空间映射，再分析 DAgger 如何通过在线聚合数据缓解模仿学习的分布偏移 [[Ross et al., 2011]](https://proceedings.mlr.press/v15/ross11a.html)。随后以 AWAC 为例，说明如何用优势加权把离线数据与在线强化学习结合 [[Nair et al., 2020]](https://arxiv.org/abs/2006.09359)。SERL 是软件与训练流程框架，并不等同于 AWAC；它的公开实现主要围绕示范增强的 off-policy 强化学习 [[Luo et al., 2024]](https://arxiv.org/abs/2401.16013)。

<div align="center">
<img src="/figures/08-robot-sim/source/06-teleop-hil-serl/serl-fig1.png" alt="SERL 在多种真实机械臂任务中复用示范、奖励分类器与在线强化学习组件。" width="86%">

_图 8.6-3：SERL 在多种真实机械臂任务中复用示范、奖励分类器与在线强化学习组件。 出处：Jianlan Luo et al.，[SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning](https://arxiv.org/abs/2401.16013)（2024），Figure 1。_
</div>

## 8.6.1 遥操作的几何映射：主从机器人的空间同构

遥操作把主控设备的运动转换为机器人目标位姿。这里先用二维增量建立直觉，再扩展到三维刚体变换。

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

在三维空间中，位姿属于特殊欧几里得群 $SE(3)$。主控设备的位姿记为 $\mathbf{T}_m\in\mathbb{R}^{4\times4}$，相邻时刻的相对变换为 $\Delta\mathbf{T}_m=\mathbf{T}_{m,t}\mathbf{T}_{m,t-1}^{-1}$，再把它映射到机器人目标位姿。实际系统还要处理主从坐标系标定、尺度、工作空间限制、逆运动学和延迟；仅有相对变换公式并不能保证操作直观或安全。

## 8.6.2 人类在环（HIL）：从分布偏移到 DAgger

当通过遥操作收集了大量的人类状态-动作轨迹 $\mathcal{D} = \{(s_i, a_i)\}_{i=1}^N$ 后，最直接的学习方式是行为克隆（Behavioral Cloning, BC）。即通过最小化负对数似然来拟合人类的条件概率分布：

$$
J_{\text{BC}}(\theta) = \mathbb{E}_{(s,a) \sim \mathcal{D}} [-\log \pi_\theta(a|s)]
$$

BC 在机器人序列决策中会面临分布偏移（Distribution Shift）。训练时，策略只在专家访问的 $p_{\text{data}}(s)$ 上拟合；部署时，自己的动作会诱导新分布 $p_{\pi_\theta}(s)$。若误差把机器人带到训练集之外，后续预测通常更不可靠，误差便可能继续累积。

Ross 等人提出数据集聚合（DAgger, Dataset Aggregation）算法 [[Ross et al., 2011]](https://proceedings.mlr.press/v15/ross11a.html)。DAgger 反复让当前策略访问状态、由专家给这些状态标注动作，再把新样本聚合进训练集；论文用在线学习中的无悔分析说明其性能界。它属于交互式模仿学习，但不等同于所有形式的人类在环控制。

在 DAgger 的第 $k$ 次迭代中，当前策略 $\pi_{\theta_k}$ 访问状态，专家为这些状态标注动作 $a_t^*=\pi^*(s_t)$，再把样本加入 $\mathcal{D}$ 并重新训练。这样，训练集逐步覆盖当前策略实际访问的状态。DAgger 的理论界依赖专家标注与在线学习假设，有限数据和函数近似下仍可能失败，因此不能理解为彻底消除分布偏移。

<div align="center">
<img src="/figures/08-robot-sim/source/06-teleop-hil-serl/hilserl-fig2.png" alt="HIL-SERL 系统图标出策略执行、人类接管、干预数据回流与离策略更新的闭环。" width="86%">

_图 8.6-4：HIL-SERL 系统图标出策略执行、人类接管、干预数据回流与离策略更新的闭环。 出处：Lawrence Yunliang Chen et al.，[HIL-SERL: Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning](https://arxiv.org/abs/2410.21845)（2024），Figure 2。_
</div>

## 8.6.3 优势加权：连接离线数据与在线更新

DAgger 需要专家为策略访问的状态持续标注。另一条路线是先把示范放入回放缓冲区，再用 off-policy 强化学习复用示范和在线数据。下面用 AWAC 解释优势加权；它不是 SERL 的同义词，也不自动提供安全约束。

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

由于 $\pi^*$ 必须是一个合法的概率分布，可以将所有不依赖动作 $a$ 的项吸收进配分函数 $Z(s)$。只依赖状态 $s$ 的基线不会改变动作之间的相对概率，因此用 $A(s,a) = Q(s,a) - V(s)$ 定义优势函数后，可写成：

$$
\pi^*(a|s) = \frac{1}{Z(s)} p_{\text{data}}(a|s) \exp \left( \frac{A(s,a)}{\lambda} \right)
$$

<div align="center">
<img src="/figures/08-robot-sim/latex/06-teleop-hil-serl/awac-advantage-reweighting.png" alt="数据动作概率乘以指数优势权重，再由配分函数归一化为目标策略" width="86%">

_图 8.6-5：AWAC 用指数优势放大高价值数据动作、压低低价值动作，再通过同一个配分函数把所有权重归一化为概率分布。本文根据上式绘制。_
</div>

温度 $\lambda$ 控制优势权重的尖锐程度。$\lambda$ 较小时，少数高优势动作获得很大权重；较大时，更新更接近对数据动作的普通最大似然。它控制的是更新幅度，不提供形式化的安全保证。

由于真实应用中我们需要拟合一个参数化的神经网络策略 $\pi_\theta(a|s)$，我们将目标转化为最小化 $\pi_\theta$ 偏离最优解 $\pi^*$ 的 KL 散度，这等价于最大化在 $\pi^*$ 下对数似然的期望。利用重要性采样（Importance Sampling），我们可以将期望的采样分布转回我们的数据集分布 $p_{\text{data}}$：

$$
\max_\theta \mathbb{E}_{a \sim p_{\text{data}}} \left[ \frac{\pi^*(a|s)}{p_{\text{data}}(a|s)} \log \pi_\theta(a|s) \right]
$$

将这个策略形式代入上式，并忽略与动作无关的 $Z(s)$，得到 AWAC 使用的加权行为克隆目标：

$$
\max_\theta \mathbb{E}_{s, a \sim \mathcal{D}} \left[ \exp \left( \frac{A(s,a)}{\lambda} \right) \log \pi_\theta(a|s) \right]
$$

在 AWAC 的推导与近似下，策略更新可写成带有 $\exp(A/\lambda)$ 权重的最大似然：优势更高的动作获得更大权重。这解释了 AWAC 如何复用离线数据，但不应据此断言所有机器人样本高效学习框架（包括 SERL）都以 AWAC 为算法基石。

## 8.6.4 代码实现

在实际的工程落地中，我们需要维护一个缓冲区，并同时训练 Actor（策略网络）和 Critic（价值网络）。下面我们将展示优势加权演员-评论家（AWAC）在核心网络更新步骤的 PyTorch 实现。

(**初始化包含优势加权机制的更新步骤**)

```python
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

            # Critic 已更新，因此重新计算数据动作的 Q 值
            data_q = self.critic(states, actions)
            advantage = data_q - v_s

            # 计算加权系数 exp(A/lambda) 并截断以防数值爆炸
            weights = torch.clamp(torch.exp(advantage / self.lambda_weight), max=100.0)

        # 获取当前策略对数据集动作的对数概率
        log_probs = self.actor(states).log_prob(actions).sum(dim=-1, keepdim=True)

        # 乘以常数权重，实现加权的最大似然估计 (等价于最小化加权负对数似然)
        actor_loss = -(weights * log_probs).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()
```

代码用 `exp(advantage / lambda_weight)` 给数据动作加权，并截断过大的权重。这里假定 actor 返回逐动作维的分布，因而先沿动作维求和得到每个样本的对数概率。

## 8.6.5 小结

- 遥操作需要位姿映射，也需要标定、约束、逆运动学和延迟处理。
- 行为克隆只在专家状态分布上训练；DAgger 通过让专家标注当前策略访问的状态来缓解分布偏移。
- AWAC 用指数化优势给回放数据中的动作加权，从而连接离线数据与在线价值估计。
- SERL 是更完整的真实机器人训练流程，不等同于 AWAC，也不因使用示范就自动获得安全保证。

## 8.6.6 练习

1. 在二维平面遥操作的该公式中，如果我们希望从动机器人的响应不仅仅包含缩放，还在末端施加一个固定的位置偏置 $\mathbf{b}$，变换矩阵应如何修改？
   - **提示**：回忆齐次坐标系中平移向量在矩阵中的位置。
2. 试证明在推导 DAgger 算法的分布偏移时，如果在每一步策略产生错误的概率为 $\epsilon$，那么在 $T$ 步之后产生偏离状态的总概率上界是 $O(T\epsilon)$。
   - **提示**：可以采用数学归纳法或者直接通过联合概率分解证明。
3. 在 AWAC 算法中，如果 $\lambda \to \infty$，该公式中的最优策略 $\pi^*$ 会退化成什么？对应的 Actor 损失函数代表了哪种经典的模仿学习算法？
   - **提示**：计算 $\lim_{\lambda \to \infty} \exp(A(s,a)/\lambda)$，并回顾该公式。
4. 检查代码实现中计算基线状态价值 $V(s)$ 的方式。为何我们通过对 `sampled_actions` 求平均来逼近 $V(s)$，而不是在代码中再定义一个单独的价值网络去训练它？
   - **提示**：思考 $Q(s,a)$ 和 $V(s)$ 在定义上的数学关系：$V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s,a)]$。
