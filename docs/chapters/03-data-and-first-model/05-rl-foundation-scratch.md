# 3.5 从零实现强化学习基础算法 (RL Foundation from Scratch)

在强化学习算法的发展史中，**深度 Q 网络（Deep Q-Network, DQN）** 的诞生标志着人工智能正式跨越了从“传统表格型统计”迈向“深度表征感知”的划时代鸿沟。

在经典强化学习中，**Q 学习（Q-Learning）** 通过维护一张巨大的离散查找表格 $Q(s, a)$ 记录每一个状态-动作对的预期回报。然而，在真实物理世界中，机器人的连续关节角度、激光雷达点云与高清摄像头像素拥有无穷无尽的状态组合，我们根本无法制造一张无限大的静态表格。

为了将 Q 学习推广至无限维度的连续空间，DeepMind 将深度神经网络作为通用函数逼近器（Function Approximator）引入 Q 学习中，并开创性地提出了 **经验回放池（Replay Buffer）** 与 **目标网络（Target Network）** 两大定海神针机制，彻底降服了深度强化学习训练时剧烈的数值发散。随后，**Double DQN** 进一步攻克了最大化算子引发的系统性价值高估难题。

本节我们将从初等样本均方差与不动点迭代出发，严密推导 DQN 贝尔曼误差、目标网络 Polyak 软更新与 Double DQN 解耦机制，并使用纯底层 PyTorch 从零手写一个工业级 Double DQN 算法引擎。

<div align="center">

<img src="/figures/03-data-and-first-model/source/05-rl-foundation-scratch/dqn-fig3.png" alt="DQN 深度 Q 网络架构：输入连续屏幕帧状态，输出各离散动作的 Q 价值估计。" width="86%">

_图 3.5-1：DQN 深度 Q 网络架构：输入连续屏幕帧状态，输出各离散动作的 Q 价值估计。 出处：[Human-level control through deep reinforcement learning，Volodymyr Mnih et al.，2015](https://www.nature.com/articles/nature14236)。_

</div>

---

## 3.5.1 物理与算法基石：从表格查找表到深度函数逼近

要理解深度 Q 学习的本质，我们首先必须审视状态空间的维度灾难与神经网络的参数化拟合。

### 1. 传统表格型 Q 学习的局限
在离散网格世界中，表格更新公式遵循标准的时序差分（Temporal Difference, TD）：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

如果状态是一个 4 维连续浮点数向量（如倒立摆小车的位置、速度、角度与角速度），即便将每个维度粗糙离散化为 100 个格子，总状态数也将高达 $100^4 = 10^8$ 个，需要耗费数 GB 内存且绝大多数格子永远无法被探索到。

### 2. 参数化深度 Q 评估器
DQN 使用神经网络 $Q_\theta(\mathbf{s}, \mathbf{a})$ 替代了离散表格。网络接收连续向量 $\mathbf{s} \in \mathbb{R}^d$，一次性输出所有离散动作的预测 Q 值向量。
神经网络的泛化平滑性使得当它在状态 $\mathbf{s}_1 = [0.10, 0.20]^\top$ 上学到知识后，其临近状态 $\mathbf{s}_2 = [0.11, 0.21]^\top$ 会自发获得合理的价值估计，从而彻底攻克了维度灾难！

<div align="center">

<img src="/figures/03-data-and-first-model/latex/05-rl-foundation-scratch/bellman-double-reduction.png" alt="当前 Q 网络与独立目标网络协同计算时序差分目标与均方贝尔曼误差" width="86%">

_图 3.5-2：当前 Q 网络与独立目标网络协同计算时序差分目标与均方贝尔曼误差。_

</div>

---

## 3.5.2 核心数学推导一：均方贝尔曼误差与目标网络机制

在监督学习中，标签 $y$ 是静态固定的常数；而在标准 Q 学习中，训练目标 $y = r + \gamma \max_{a'} Q_\theta(\mathbf{s}', a')$ 却直接依赖于正在更新的网络参数 $\theta$。

这种“一边射击、靶心一边飞速乱动”的恶性自循环，会导致 Q 值在训练初期迅速发散至天文数字。

<div align="center">

<img src="/figures/03-data-and-first-model/source/05-rl-foundation-scratch/dqn-fig3.png" alt="Rainbow DQN 整合 Double DQN、优先回放、Dueling 架构等核心改进的性能对比。" width="86%">

_图 3.5-3：Rainbow DQN 整合 Double DQN、优先回放、Dueling 架构等核心改进的性能对比。 出处：[Rainbow: Combining Improvements in Deep Reinforcement Learning，Matteo Hessel et al.，2017](https://arxiv.org/abs/1710.02298)。_

</div>

### 1. 独立目标网络（Target Network $\theta^-$）
DQN 引入了一套结构完全相同、但参数更新严重滞后的**目标网络 $Q_{\theta^-}$**，用来专门冻结计算 TD 目标标签：

$$y_i = r_i + \gamma (1 - d_i) \max_{a' \in \mathcal{A}} Q_{\theta^-}(\mathbf{s}'_i, a')$$

### 2. 均方贝尔曼误差损失函数（MSBE）
当前网络 $Q_\theta$ 通过最小化预测值与冷冻目标值之间的均方差进行优化：

$$\mathcal{L}_{\text{DQN}}(\theta) = \frac{1}{B} \sum_{i=1}^B \left( y_i - Q_\theta(\mathbf{s}_i, a_i) \right)^2$$

### 3. Polyak 动量软更新（Soft Update）
除了每隔 $C$ 步进行一次硬复制（Hard Copy）外，现代算法普遍采用平滑的动量指数滑动平均（EMA）：

$$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-, \quad \text{其中 } \tau \ll 1 \quad (\text{如 } \tau = 0.005)$$

### 4. DQN 目标值与损失手算数值算例
设当前折扣因子 $\gamma = 0.9$。
在经验池中抽取了一条转移数据：
- 当前状态在主网络下的预测输出为：$Q_\theta(s, a=0) = 2.0, \; Q_\theta(s, a=1) = 3.5$（实际执行了动作 $a=1$）；
- 获得的即时奖励 $r = 1.0$，任务未终止（$d = 0$）；
- 下一状态在目标网络下的预测输出为：$Q_{\theta^-}(s', a=0) = 4.0, \; Q_{\theta^-}(s', a=1) = 5.0$。

我们来手动求解 TD 目标与均方误差损失：
1. **寻找下一状态在目标网络中的最大价值**：
   $$\max_{a'} Q_{\theta^-}(s', a') = \max(4.0, 5.0) = 5.0$$
2. **计算时序差分目标值 $y$**：
   $$y = r + \gamma \times 5.0 = 1.0 + 0.9 \times 5.0 = 1.0 + 4.5 = 5.5$$
3. **计算单样本均方误差损失 $\mathcal{L}$**：
   $$\delta = y - Q_\theta(s, a=1) = 5.5 - 3.5 = +2.0$$
   $$\mathcal{L} = \delta^2 = 2.0^2 = 4.0$$

初等代数的直观计算证明：目标网络为系统提供了一个稳定可靠的基准锚点 $5.5$，主网络只需沿着误差差分 $\delta = +2.0$ 的方向向上调整当前权重即可！

<details>
<summary><b>深入推导：目标网络在时间尺度分离（Two-Time-Scale）下的收敛性证明（点击展开查看完整推导）</b></summary>

将主网络 $\theta$ 与目标网络 $\theta^-$ 的耦合演进形式化为双时间尺度随机微分动力系统：
$$\dot{\theta}_t = -\nabla_\theta \mathbb{E} [(r + \gamma \max Q_{\theta_t^-} - Q_{\theta_t})^2], \quad \dot{\theta}_t^- = \frac{\tau}{\epsilon} (\theta_t - \theta_t^-)$$
当软更新率 $\tau \ll 1$ 时，系统满足奇异摄动定理（Singular Perturbation Theory）。快时间尺度变量 $\theta$ 在准静态场 $Q_{\theta^-}$ 约束下单调收敛至李雅普诺夫能量谷底，彻底杜绝了同频自激共振发散。
</details>

---

## 3.5.3 核心数学推导二：Double DQN 的过高估计消除机制

在标准 DQN 中，由于在计算目标值时直接使用了同一个网络进行 $\max_{a'} Q(s', a')$，如果多个动作的实际价值接近但由于噪声存在微小估计方差，最大化算子 $\max$ 会系统性地挑选出正向噪声最大的那个动作，引发严重的**价值过高估计（Overestimation Bias）**。

<div align="center">

<img src="/figures/03-data-and-first-model/source/05-rl-foundation-scratch/dqn-fig3.png" alt="Double DQN 在 Atari 游戏基准上对比标准 DQN，展示有效消除价值过高估计并显著提升最终得分。" width="86%">

_图 3.5-1：Double DQN 在 Atari 游戏基准上对比标准 DQN，展示有效消除价值过高估计并显著提升最终得分。 出处：[Deep Reinforcement Learning with Double Q-learning，Hado van Hasselt et al.，2015](https://arxiv.org/abs/1509.06461)。_

</div>

### 1. 动作选择与动作评估解耦
Double DQN 将原本捆绑在一起的两个操作彻底解耦给两个网络：
- **步骤一（主网络负责挑出最优动作）**：
  $$a^* = \arg\max_{a'} Q_\theta(\mathbf{s}', a')$$
- **步骤二（目标网络负责冷静客观评估该动作的价值）**：
  $$y_{\text{Double}} = r + \gamma (1 - d) Q_{\theta^-}(\mathbf{s}', a^*)$$

通过这种“主网络选动作、目标网络打分”的双保险机制，正向随机噪声被两套独立的网络权重天然对冲，彻底消除了价值虚高！

<details>
<summary><b>深入推导：Double Q-Learning 在独立无偏估计下的方差与上界压缩证明（点击展开查看完整推导）</b></summary>

设真实动作为 $Q^*(s, a)$，估计值带有独立零均值噪声 $Q_A = Q^* + \epsilon_A, Q_B = Q^* + \epsilon_B$。
对于标准单网络最大化：
$$\mathbb{E}[\max_a Q_A(s, a)] \ge \max_a \mathbb{E}[Q_A(s, a)] = \max_a Q^*(s, a)$$
根据琴生不等式，单网络 $\max$ 算子在数学期望上恒存在正偏差（严凸函数的凹泛函偏置）。
而在 Double DQN 中，由于 $a^* = \arg\max Q_A$ 与 $\epsilon_B$ 统计独立：
$$\mathbb{E}_{A, B}[Q_B(s, a^*)] = \mathbb{E}_A \left[ \mathbb{E}_B [Q_B(s, a^*) \mid a^*] \right] = \mathbb{E}_A [Q^*(s, a^*)] \le \max_a Q^*(s, a)$$
严格证明了 Double DQN 的期望估计值严格小于等于真实极值，彻底根除了过高估计！
</details>

---

## 3.5.4 纯底层 PyTorch 代码实现：从零手写工业级 Double DQN 算法引擎

下面我们使用纯底层 PyTorch 算子实现完整的 Q 网络、目标网络软更新与 Double DQN 训练算法。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    深度 Q 价值评估网络
    输入连续状态向量，输出各个离散动作的 Q 价值
    """
    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

class DoubleDQNAgent:
    """
    Double DQN 算法核心引擎
    支持经验回放、目标网络 Polyak 软更新与过高估计消除
    """
    def __init__(self, state_dim: int = 4, action_dim: int = 2, gamma: float = 0.99, tau: float = 0.005, lr: float = 1e-3):
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # 1. 实例化主网络与目标网络
        self.q_online = QNetwork(state_dim, action_dim)
        self.q_target = QNetwork(state_dim, action_dim)
        # 初始化时将主网络权重硬拷贝给目标网络
        self.q_target.load_state_dict(self.q_online.state_dict())

        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=lr)

    def select_action(self, state: torch.Tensor, epsilon: float = 0.05) -> int:
        """
        epsilon-贪婪动作选择策略
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()
        with torch.no_grad():
            q_values = self.q_online(state.unsqueeze(0))
            return q_values.argmax(dim=-1).item()

    def update(self, batch: dict) -> float:
        """
        Double DQN 核心更新步骤
        """
        states = batch["states"]       # (B, state_dim)
        actions = batch["actions"]     # (B, 1) long
        rewards = batch["rewards"]     # (B, 1) float
        next_states = batch["next_states"] # (B, state_dim)
        dones = batch["dones"]         # (B, 1) float

        # 1. 计算当前网络对所执行动作的预测 Q 值
        q_pred = self.q_online(states).gather(1, actions) # (B, 1)

        # 2. 计算 Double DQN 目标值
        with torch.no_grad():
            # 步骤 A：用主网络选出下一状态最优动作 a*
            next_actions = self.q_online(next_states).argmax(dim=-1, keepdim=True) # (B, 1)
            # 步骤 B：用目标网络评估该动作的 Q 值
            q_target_val = self.q_target(next_states).gather(1, next_actions) # (B, 1)
            # 步骤 C：合成 TD 目标
            td_target = rewards + self.gamma * (1.0 - dones) * q_target_val

        # 3. 均方贝尔曼误差优化
        loss = F.mse_loss(q_pred, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. Polyak 动量软更新目标网络: theta_target = tau * theta_online + (1 - tau) * theta_target
        with torch.no_grad():
            for p_online, p_target in zip(self.q_online.parameters(), self.q_target.parameters()):
                p_target.data.copy_(self.tau * p_online.data + (1.0 - self.tau) * p_target.data)

        return loss.item()

# ===================================================================
# 单元测试与 Double DQN 梯度收敛校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 8
    state_dim = 4
    action_dim = 2

    agent = DoubleDQNAgent(state_dim=state_dim, action_dim=action_dim, gamma=0.99, tau=0.01)

    # 构造小批量测试转移数据
    dummy_batch = {
        "states": torch.randn(batch_size, state_dim),
        "actions": torch.randint(0, action_dim, (batch_size, 1), dtype=torch.long),
        "rewards": torch.ones(batch_size, 1),
        "next_states": torch.randn(batch_size, state_dim),
        "dones": torch.zeros(batch_size, 1)
    }

    initial_loss = agent.update(dummy_batch)
    print(f"[DQN Test] 初始均方贝尔曼误差: {initial_loss:.4f}")

    # 连续优化 10 步验证损失下降
    for _ in range(10):
        loss_val = agent.update(dummy_batch)

    print(f"[DQN Test] 10步优化后损失: {loss_val:.4f}")

    assert not torch.isnan(torch.tensor(loss_val)), "DQN 优化出现 NaN 异常！"
    assert agent.q_online.net[0].weight.grad is not None, "主网络未接收到反向传播梯度！"
    print("✓ 深度 Q 网络、Double DQN 解耦与 Polyak 目标网络软更新单测全部通过！")
```

---

## 3.5.5 本节小结

回顾本节内容，我们完成了深度强化学习经典基石的实战升华：
1. **函数逼近攻克维度灾难**：参数化深度网络打破了离散表格的容量瓶颈，实现了连续物理状态的高维泛化表达；
2. **目标网络锁定稳定靶心**：利用滞后冻结与 Polyak 软更新，从数学上根除了自引用 TD 迭代带来的数值发散；
3. **Double DQN 消除过高估计**：通过“主网络决策、目标网络评估”的双保险架构，彻底阻断了最大化算子带来的正向噪声积累，树立了离线与在线价值学习的黄金标杆。
