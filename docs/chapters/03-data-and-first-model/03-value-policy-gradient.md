# 3.3 价值函数与策略梯度基础 (Value & Policy Gradient)

在强化学习的宏伟版图中，智能体如何从纷繁复杂的物理交互经验中提炼出最优的决策逻辑？

学术界演进出了两大经典流派：
- **基于价值的方法（Value-Based Methods）**：如同一个深思熟虑的棋局裁判，专注于评估当前局面在长远未来能够带来的期望总收益（即价值函数 $V(s)$ 与 $Q(s, a)$），通过贪婪选择价值最大的动作实现决策；
- **基于策略的方法（Policy-Based Methods）**：如同一个凭借直觉肌肉记忆动作的运动员，直接对参数化策略网络 $\pi_\theta(a \mid s)$ 进行微分求导，沿着能够增大预期回报的梯度方向持续优化。

而将两者优势完美结合的 **Actor-Critic（演员-评判家）** 架构，则让 Actor 专注于大胆执行探索，让 Critic 专注于冷静打分纠偏，构成了现代世界模型策略学习的绝对基石。

本节我们将从初等期望代数与对数求导出发，严密推导贝尔曼方程（Bellman Equation）、策略梯度定理（Policy Gradient Theorem）与基线方差削减机制，并使用纯底层 PyTorch 从零手写一个完整的 Actor-Critic 算法。

<div align="center">

<img src="/figures/03-data-and-first-model/source/03-value-policy-gradient/ddpg-fig1.png" alt="Actor-Critic 架构：Actor 策略网络输出动作与环境交互，Critic 价值网络计算 TD 误差指导策略更新。" width="86%">

_图 3.3-1：Actor-Critic 架构：Actor 策略网络输出动作与环境交互，Critic 价值网络计算 TD 误差指导策略更新。 出处：[Reinforcement Learning: An Introduction，Richard S. Sutton & Andrew G. Barto，2018](http://incompleteideas.net/book/the-book-2nd.html)。_

</div>

---

## 3.3.1 物理与数学基石：价值评估与策略分布的形式化

要理解强化学习算法的演进，我们首先必须严密区分状态价值（State Value）与动作价值（Action Value）。

### 1. 状态价值函数（State-Value Function $V(s)$）
定义为智能体在状态 $s$ 下，遵循既定策略 $\pi$ 进行交互所能获得的**未来累积折扣回报的数学期望**：

$$V^\pi(s) = \mathbb{E}_{\pi} \left[ G_t \mid \mathbf{s}_t = s \right] = \mathbb{E}_{\pi} \left[ \sum_{k=0}^\infty \gamma^k r_{t+k} \;\middle|\; \mathbf{s}_t = s \right]$$

### 2. 动作价值函数（Action-Value Function $Q(s, a)$）
定义为智能体在状态 $s$ 下，**先强制执行动作 $a$**，随后继续遵循策略 $\pi$ 所能获得的累积回报期望：

$$Q^\pi(s, a) = \mathbb{E}_{\pi} \left[ G_t \mid \mathbf{s}_t = s, \mathbf{a}_t = a \right] = r(s, a) + \gamma \mathbb{E}_{s' \sim \mathcal{P}} \left[ V^\pi(s') \right]$$

二者的初等代数关系极其纯粹：**状态价值就是所有合法动作价值关于策略概率分布的加权平均值**：

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \cdot Q^\pi(s, a)$$

<div align="center">

<img src="/figures/03-data-and-first-model/latex/03-value-policy-gradient/baseline-zero-expectation.png" alt="贝尔曼期望方程回溯图：当前状态通过动作策略分支与环境状态转移分支的逐层期望汇聚" width="86%">

_图 3.3-2：贝尔曼期望方程回溯图：当前状态通过动作策略分支与环境状态转移分支的逐层期望汇聚。_

</div>

---

## 3.3.2 核心数学推导一：贝尔曼方程与最优性递推

在动态规划（Dynamic Programming）中，Richard Bellman 提出了著名的递归分解定理。

### 1. 贝尔曼期望方程（Bellman Expectation Equation）
将累积回报的展开式代入价值函数定义中：

$$V^\pi(s) = \sum_{a} \pi(a \mid s) \left[ r(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V^\pi(s') \right]$$

### 2. 贝尔曼最优方程（Bellman Optimality Equation）
最优策略 $\pi^*$ 在每个状态下必然选择使得长期价值最大化的最优动作：

$$V^*(s) = \max_{a \in \mathcal{A}} \left[ r(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V^*(s') \right]$$

$$Q^*(s, a) = r(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) \max_{a'} Q^*(s', a')$$

### 3. 贝尔曼最优价值手算数值算例
设机器人处于状态 $s$，拥有两个可选动作：
- **动作 A（保守直行）**：即时奖励 $r_A = 5.0$，转移到下一状态 $s'_1$ 的最优价值为 $V^*(s'_1) = 10.0$；
- **动作 B（激进冲刺）**：即时奖励 $r_B = 8.0$，转移到下一状态 $s'_2$ 的最优价值为 $V^*(s'_2) = 4.0$；
- 折扣因子设为 $\gamma = 0.9$。

我们来手动求解各动作的 Q 价值与状态的最优价值 $V^*(s)$：
1. **计算动作 A 价值**：
   $$Q^*(s, A) = r_A + \gamma V^*(s'_1) = 5.0 + 0.9 \times 10.0 = 5.0 + 9.0 = 14.0$$
2. **计算动作 B 价值**：
   $$Q^*(s, B) = r_B + \gamma V^*(s'_2) = 8.0 + 0.9 \times 4.0 = 8.0 + 3.6 = 11.6$$
3. **取二者极大值得到当前状态最优价值**：
   $$V^*(s) = \max(14.0, \; 11.6) = 14.0 \quad (\text{最优决策为执行动作 A})$$

初等代数的直观计算证明：尽管动作 B 的眼前的即时奖励更高（$8.0 > 5.0$），但贝尔曼方程精确洞悉了动作 A 的长远综合收益更高（$14.0 > 11.6$），从而做出了最具远见的理性决策！

<details>
<summary><b>深入推导：贝尔曼最优算子在巴拿赫空间无穷范数下的 $\gamma$-收缩映射证明（点击展开查看完整推导）</b></summary>

定义贝尔曼最优价值更新算子 $\mathcal{T}^*$：
$$(\mathcal{T}^* V)(s) = \max_{a} \left[ r(s, a) + \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) V(s') \right]$$
对任意两个价值函数 $U, V \in \mathbb{R}^{|\mathcal{S}|}$，在无穷范数 $\|V\|_\infty = \max_s |V(s)|$ 下：
$$|(\mathcal{T}^* U)(s) - (\mathcal{T}^* V)(s)| \le \max_a \left| \gamma \sum_{s'} \mathcal{P}(s' \mid s, a) (U(s') - V(s')) \right| \le \gamma \|U - V\|_\infty \sum_{s'} \mathcal{P}(s' \mid s, a) = \gamma \|U - V\|_\infty$$
由于 $\gamma < 1$，$\mathcal{T}^*$ 是严格的 $\gamma$-压缩映射（Contraction Mapping）。
根据巴拿赫不动点定理（Banach Fixed-Point Theorem），存在且仅存在唯一的驻点 $V^*$ 满足 $\mathcal{T}^* V^* = V^*$，且价值迭代以几何速率 $\mathcal{O}(\gamma^k)$ 绝对全局收敛！
</details>

---

## 3.3.3 核心数学推导二：策略梯度定理与对数似然求导技巧

在连续动作控制（如机械臂关节角度、无人机倾角）中，直接求 $\max_a Q(s, a)$ 是极度困难的。我们转而采用参数化策略网络 $\pi_\theta(a \mid s)$。

<div align="center">

<img src="/figures/03-data-and-first-model/source/03-value-policy-gradient/ddpg-fig1.png" alt="REINFORCE 策略梯度算法学习曲线：展示引入基线函数后方差显著降低与训练加速。" width="86%">

_图 3.3-3：REINFORCE 策略梯度算法学习曲线：展示引入基线函数后方差显著降低与训练加速。 出处：[Reinforcement Learning: An Introduction，Richard S. Sutton & Andrew G. Barto，2018](http://incompleteideas.net/book/the-book-2nd.html)。_

</div>

### 1. 目标函数与对数似然微分技巧（Log-Derivative Trick）
定义策略优化的总目标函数为整条轨迹累积回报的期望值：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)] = \int P(\tau; \theta) R(\tau) d\tau$$

由于环境物理转移概率无法直接求导，利用微积分初等恒等式 $\nabla f(x) = f(x) \nabla \log f(x)$：

$$\nabla_\theta J(\theta) = \int \nabla_\theta P(\tau; \theta) R(\tau) d\tau = \int P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) R(\tau) d\tau = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau; \theta) R(\tau) \right]$$

展开轨迹对数概率，由于环境转移概率与 $\theta$ 无关，其导数项彻底归零：

$$\nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_t \mid \mathbf{s}_t)$$

### 2. 策略梯度通用形式与基线降方差（Baseline Variance Reduction）
为了降低蒙特卡洛采样的巨大方差，引入仅与状态相关的基线函数 $b(s_t) = V_\phi(s_t)$：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_t \mid \mathbf{s}_t) \cdot \underbrace{\left( Q^\pi(\mathbf{s}_t, \mathbf{a}_t) - V_\phi(\mathbf{s}_t) \right)}_{\text{优势函数 } A(\mathbf{s}_t, \mathbf{a}_t)} \right]$$

> **直观物理启示**：
> - 若当前动作取得的实际价值高于平均基准（优势 $A > 0$），梯度将推动网络增大该动作的发生概率；
> - 若当前动作表现劣于平均基准（优势 $A < 0$），梯度将无情打压该动作的生成概率！

<details>
<summary><b>深入推导：引入任意状态基线函数 $b(s)$ 保持策略梯度期望严格无偏的证明（点击展开查看完整推导）</b></summary>

考察基线项对策略梯度的期望贡献：
$$\mathbb{E}_{\mathbf{s}_t, \mathbf{a}_t} \left[ \nabla_\theta \log \pi_\theta(\mathbf{a}_t \mid \mathbf{s}_t) \cdot b(\mathbf{s}_t) \right] = \sum_{\mathbf{s}_t} d^\pi(\mathbf{s}_t) b(\mathbf{s}_t) \sum_{\mathbf{a}_t} \nabla_\theta \pi_\theta(\mathbf{a}_t \mid \mathbf{s}_t)$$
由于策略是合法的概率分布，全概率积分为常数 1：
$$\sum_{\mathbf{a}_t} \pi_\theta(\mathbf{a}_t \mid \mathbf{s}_t) = 1 \implies \nabla_\theta \sum_{\mathbf{a}_t} \pi_\theta(\mathbf{a}_t \mid \mathbf{s}_t) = \sum_{\mathbf{a}_t} \nabla_\theta \pi_\theta(\mathbf{a}_t \mid \mathbf{s}_t) = 0$$
因此基线项期望恒等于 0，证明了减去基线绝不会引入任何梯度偏差，同时大幅缩减了采样方差。
</details>

---

## 3.3.4 纯底层 PyTorch 代码实现：从零手写 Actor-Critic 强化学习算法

下面我们使用纯底层 PyTorch 算子实现一个结构完备的连续动作 Actor-Critic 智能体。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    """
    策略网络 (Actor)
    输出高斯动作分布的均值与方差
    """
    def __init__(self, state_dim: int = 4, action_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(state)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_action_and_log_prob(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

class CriticNetwork(nn.Module):
    """
    价值网络 (Critic)
    输出状态价值标量 V(s)
    """
    def __init__(self, state_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)

class ActorCriticTrainer:
    """
    Actor-Critic 联合优化引擎
    """
    def __init__(self, actor: ActorNetwork, critic: CriticNetwork, gamma: float = 0.99, lr: float = 1e-3):
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lr)
        self.optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr)

    def train_step(
        self, state: torch.Tensor, action: torch.Tensor, reward: float, next_state: torch.Tensor, done: bool
    ) -> tuple[float, float]:
        # 1. 计算 Critic 价值与 TD 误差
        v_current = self.critic(state)
        with torch.no_grad():
            v_next = self.critic(next_state) if not done else 0.0
            td_target = reward + self.gamma * v_next
            advantage = td_target - v_current.item()

        # 2. 更新 Critic 损失: MSE(V(s), TD_Target)
        critic_loss = F.mse_loss(v_current, torch.tensor(td_target, dtype=torch.float32))
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # 3. 更新 Actor 策略梯度损失: - log_pi(a|s) * Advantage
        mean, std = self.actor(state)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum()

        actor_loss = - log_prob * advantage
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        return actor_loss.item(), critic_loss.item()

# ===================================================================
# 单元测试与双网络联合梯度校验
# ===================================================================
if __name__ == "__main__":
    state_dim = 4
    action_dim = 1

    actor = ActorNetwork(state_dim=state_dim, action_dim=action_dim)
    critic = CriticNetwork(state_dim=state_dim)
    trainer = ActorCriticTrainer(actor=actor, critic=critic, gamma=0.95)

    s = torch.randn(state_dim)
    a, _ = actor.get_action_and_log_prob(s)
    s_next = torch.randn(state_dim)
    r = 1.0
    d = False

    actor_l, critic_l = trainer.train_step(s, a, r, s_next, d)
    print(f"[AC Test] 演员策略损失: {actor_l:.4f}, 评判家价值损失: {critic_l:.4f}")

    assert actor.net[0].weight.grad is not None, "Actor 梯度未成功反传！"
    assert critic.net[0].weight.grad is not None, "Critic 梯度未成功反传！"
    print("✓ 连续动作 Actor-Critic 双网络架构与 TD 优势更新单测全部通过！")
```

---

## 3.3.5 本节小结

回顾本节内容，我们建立了强化学习决策优化的双重核心体系：
1. **贝尔曼最优性**：通过将长期期望价值递归分解为即时奖励与下一状态价值，为离散与连续控制提供了扎实的数学标尺；
2. **对数似然微分技巧**：彻底绕开了环境黑盒动力学不可求导的死穴，实现了端到端策略梯度的无偏反向传播；
3. **Actor-Critic 协同进化**：Actor 负责生成多维连续控制，Critic 提供低方差基线引导，构成了后续章节在潜空间世界模型中推演策略的高速引擎。
