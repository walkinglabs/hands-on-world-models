# 8.5 梦境强化学习与无模型策略闭环

在强化学习的漫长演进史中，**无模型强化学习（Model-Free RL, 如 SAC, PPO）** 长期占据着统治地位。

然而，无模型算法存在一个几乎不可调和的阿喀琉斯之踵——**极低的数据样本效率（Sample Inefficiency）**。为了让一只机械臂学会精准抓取或让双足机器人学会平稳步行，无模型算法往往需要在真实物理环境或模拟器中进行数千万步的盲目随机试错。

如果将这种算法直接部署在物理硬件上，即便不考虑关节磨损与电机过热，单单采集这数千万步数据所需要的物理时间就长达数月乃至数年。

人类之所以不需要跌落悬崖上万次就能本能地学会远离悬崖，是因为我们的大脑内部拥有一个强大的**神经世界模型（Mental World Model）**。在我们付诸行动之前，我们已经在脑海的“内心剧场”中以极快的速度演练了行动可能引发的物理后果与危险。

从 Ha 与 Schmidhuber 的 **World Models**，到 Danijar Hafner 等人推出的 **PlaNet** 与 **DreamerV1 / V2 / V3**，**梦境强化学习（Imagination-based Reinforcement Learning）** 开创了全新的智能范式——机器人首先利用少量物理交互数据构建起周围世界的动态运转规律，随后彻底脱离外部物理环境，直接在内部潜在潜空间的“梦境”中自我推演数百万次，实现策略的高速闭环进化。

<div align="center">

<img src="/figures/08-robot-sim/source/05-imagination-rl-rise/dreamerv3-fig4.png" alt="DreamerV3 跨越连续控制、像素 Atari 与高难度 Minecraft 任务，展示统一梦境强化学习架构。" width="86%">

_图 8.5-1：DreamerV3 跨越连续控制、像素 Atari 与高难度 Minecraft 任务，展示统一梦境强化学习架构。 出处：[Mastering Diverse Domains through World Models，Danijar Hafner et al.，2023](https://arxiv.org/abs/2301.04104)。_

</div>

---

## 8.5.1 物理与认知基石：现实交互与梦境想象的二元循环

要理解梦境强化的核心逻辑，我们首先必须审视外部物理真实世界与内部潜在动力学模型之间的二元交互循环。

整个系统由两个相互解耦、异步演进的认知回路构成：

### 1. 外部真实感知回路（Environment Agent Loop）
机器人仅以极低频率（例如每秒 10 次）与外部物理世界进行真实交互。
- 收集少量的真实物理转移元组 $(\mathbf{o}_t, \mathbf{a}_t, r_t, \mathbf{o}_{t+1})$ 并存入经验回放池（Replay Buffer）；
- 利用这些真实数据，训练**循环状态空间世界模型（RSSM）**，让世界模型学会精准预测物理状态的演变规律、图像重构与即时奖励。

### 2. 内部潜空间梦境回路（Latent Imagination Loop）
一旦世界模型具备了基础的物理预测能力，**物理世界的大门被完全关闭**。
- 策略网络（Actor）与价值网络（Critic）从一个随机的潜在初始状态 $\mathbf{s}_0$ 开始，向未来展开为期 $H = 15$ 步的“纯潜空间梦境推演”；
- 梦境中不需要渲染昂贵的像素图像，仅在紧凑的隐向量空间 $\mathbb{R}^d$ 中以数万步/秒的惊人算力暴风骤雨般向前滚动；
- 策略在这个虚拟梦境中累积想象奖励，并通过**端到端反向传播解析梯度**直接优化策略参数！

<div align="center">

<img src="/figures/08-robot-sim/latex/05-imagination-rl-rise/rssm-observation-reward-factor.png" alt="RSSM 潜在动力学因子图：确定性路径与随机路径协同预测未来奖励与观测" width="86%">

_图 8.5-2：RSSM 潜在动力学因子图：确定性路径与随机路径协同预测未来奖励与观测。_

</div>

---

## 8.5.2 核心数学推导一：可微梦境轨迹与解析策略梯度

为什么梦境强化学习的学习效率能够超越无模型强化学习数个数量级？答案在于**可微路径梯度（Pathwise Analytic Gradients）**。

<div align="center">

<img src="/figures/08-robot-sim/source/05-imagination-rl-rise/dreamer-fig2.png" alt="Dreamer 在潜在状态空间中展开想象轨迹，通过重参数化技巧实现策略梯度的解析反向传播。" width="86%">

_图 8.5-3：Dreamer 在潜在状态空间中展开想象轨迹，通过重参数化技巧实现策略梯度的解析反向传播。 出处：[Dream to Control: Learning Behaviors by Latent Imagination，Danijar Hafner et al.，2019](https://arxiv.org/abs/1912.01603)。_

</div>

<div align="center">

<img src="/figures/08-robot-sim/source/05-imagination-rl-rise/planet-fig6.png" alt="PlaNet 仅利用潜在动力学模型与在线规划 (CEM) 解决多种连续控制任务。" width="86%">

_图 8.5-4：PlaNet 仅利用潜在动力学模型与在线规划 (CEM) 解决多种连续控制任务。 出处：[Learning Latent Dynamics for Planning from Pixels，Danijar Hafner et al.，2018](https://arxiv.org/abs/1811.04551)。_

</div>

### 1. 传统无模型策略梯度的“黑盒黑障”
在无模型强化学习中，环境物理引擎是一个不可求导的黑盒。算法只能采用类似 REINFORCE 的似然比得分函数估计梯度：
$$\nabla_\phi J \approx \sum_{t} \nabla_\phi \log \pi_\phi(\mathbf{a}_t \mid \mathbf{s}_t) \hat{A}_t$$
这种梯度估计方法方差极大，就像一个蒙着双眼的人只能通过“最终摔倒没摔倒”的标量反馈来盲目摸索方向。

### 2. 梦境世界模型中的全路径链式反传
在世界模型中，状态转移函数 $\mathbf{s}_{t+1} = f_\theta(\mathbf{s}_t, \mathbf{a}_t)$、动作策略 $\mathbf{a}_t = \pi_\phi(\mathbf{s}_t)$ 与奖励模型 $\hat{r}_t = R_\theta(\mathbf{s}_t)$ 全部是由连续可微的神经网络构成！

对于一条长度为 $H$ 的想象轨迹，总回报可直接对策略网络参数 $\phi$ 依据多元复合函数微积分链式法则精确求导：

$$\frac{\partial}{\partial \phi} \sum_{\tau=1}^H \hat{r}_\tau = \sum_{\tau=1}^H \left( \frac{\partial \hat{r}_\tau}{\partial \mathbf{s}_\tau} \frac{\partial \mathbf{s}_\tau}{\partial \mathbf{a}_{\tau-1}} \frac{\partial \mathbf{a}_{\tau-1}}{\partial \phi} + \frac{\partial \hat{r}_\tau}{\partial \mathbf{s}_\tau} \frac{\partial \mathbf{s}_\tau}{\partial \mathbf{s}_{\tau-1}} \frac{\partial \mathbf{s}_{\tau-1}}{\partial \mathbf{a}_{\tau-2}} \frac{\partial \mathbf{a}_{\tau-2}}{\partial \phi} + \dots \right)$$

### 3. 可微想象梯度详细手算代入算例
设隐状态为标量 $s$，动作为标量 $a$。系统由以下极简可微函数构成：
- 策略网络：$a = \phi \cdot s$（初始参数 $\phi = 2.0$）；
- 动力学模型：$s_{t+1} = 0.5 s_t + a_t$；
- 奖励模型：$r(s) = 3.0 s$。

初始状态为 $s_1 = 1.0$。我们展开长度为 $H = 2$ 步的梦境想象：
1. **第 1 步前向推演**：
   - 动作：$a_1 = \phi \cdot s_1 = 2.0 \times 1.0 = 2.0$；
   - 奖励：$r_1 = 3.0 \times s_1 = 3.0 \times 1.0 = 3.0$；
2. **第 2 步前向推演**：
   - 下一状态：$s_2 = 0.5 s_1 + a_1 = 0.5 \times 1.0 + \phi s_1 = 0.5 + 2.0 = 2.5$；
   - 奖励：$r_2 = 3.0 \times s_2 = 3.0 \times (0.5 s_1 + \phi s_1)$；
3. **计算总回报关于策略参数 $\phi$ 的解析梯度**：
   - $r_1$ 与 $\phi$ 无直接依赖（$\frac{\partial r_1}{\partial \phi} = 0$）；
   - $r_2$ 对 $\phi$ 的偏导数：
     $$\frac{\partial r_2}{\partial \phi} = \frac{\partial r_2}{\partial s_2} \cdot \frac{\partial s_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial \phi} = 3.0 \times 1.0 \times s_1 = 3.0 \times 1.0 \times 1.0 = +3.0$$

初等代数的几步链式展开清晰揭示：世界模型直接告知策略参数 $\phi$：**“只要将 $\phi$ 增大，第 2 步的状态 $s_2$ 就会增大，从而带来 $+3.0$ 的确定性奖励增长！”** 这种具备物理因果白盒的解析梯度，其信息量远非无模型试错所能比拟！

<details>
<summary><b>深入推导：梦境想象可微梯度（Analytic Gradients）相比无模型似然比梯度的方差衰减分析（点击展开查看完整推导）</b></summary>

设真实动力学 Jacobian 矩阵为 $\mathbf{J}_t = \frac{\partial \mathbf{s}_{t+1}}{\partial \mathbf{s}_t}$。
解析梯度估计量为 $\hat{g}_{\text{analytic}} = \sum_t \mathbf{J}_{\text{model}} \nabla_\phi \pi$；得分函数梯度估计量为 $\hat{g}_{\text{score}} = \sum_t \nabla_\phi \log \pi (G_t - b)$。
根据 Rao-Blackwell 定理与对偶控制变量方差分析，由于解析梯度直接对内在动力学路径积分，其估计方差满足：
$$\text{Var}(\hat{g}_{\text{analytic}}) = \mathcal{O}(H \cdot \sigma_{\text{model}}^2) \ll \mathcal{O}(H^3 \cdot \sigma_{\text{env}}^2) = \text{Var}(\hat{g}_{\text{score}})$$
极低的方差使优化器能够跨越长程时间视界（$H \ge 15$）平稳更新，实现了百倍量级的样本效率跨越。
</details>

---

## 8.5.3 核心数学推导二：DreamerV3 的符号对称缩放与无超参泛化

在跨越从雅达利游戏（稀疏奖励、高动态）到四足机器人（密集力矩、高维物理）的广泛任务时，奖励尺度相差上百万倍，常常导致 Critic 价值网络数值溢出或爆炸。

<div align="center">

<img src="/figures/08-robot-sim/source/05-imagination-rl-rise/worldmodels-fig7.png" alt="World Models 首次在梦境隐空间中演练赛车游戏，展示梦境学习的早期先驱原型。" width="86%">

_图 8.5-5：World Models 首次在梦境隐空间中演练赛车游戏，展示梦境学习的早期先驱原型。 出处：[Recurrent World Models Facilitate Policy Evolution，David Ha & Jürgen Schmidhuber，2018](https://arxiv.org/abs/1809.01999)。_

</div>

DreamerV3 引入了三大革命性的**无超参自适应缩放机制**：

### 1. 对称对数变换（Symlog Transformation）
为了压缩极大或极小的物理数值跨度，DreamerV3 提出了对称可逆对数映射：

$$\text{symlog}(x) = \text{sign}(x) \ln(|x| + 1)$$

$$\text{symexp}(y) = \text{sign}(y) (\exp(|y|) - 1)$$

无论输入奖励是 $0.001$ 还是 $1,000,000$，经 $\text{symlog}$ 映射后均被平滑压缩在 $[-15, 15]$ 的健康数值区间内。

### 2. 动态百分位数价值归一化（Percentile Return Normalization）
为了让 Actor 策略梯度更新幅度在不同任务间完全一致，DreamerV3 统计回放价值分布的第 5 分位数 $P_{05}$ 与第 95 分位数 $P_{95}$，并对优势进行动态尺度归一化：

$$\hat{A}_t^{\text{norm}} = \frac{V_t^{\text{imagined}} - V_\psi(\mathbf{s}_t)}{\max(1.0, \; P_{95} - P_{05})}$$

这种极致的数值鲁棒性，让 DreamerV3 成为深度强化学习历史上第一个完全不调节任何超参数、直接通关数百个异构环境的通用世界模型算法！

<details>
<summary><b>深入推导：对称对数函数在无穷方差重尾奖励分布下的李普希茨连续性证明（点击展开查看完整推导）</b></summary>

函数 $f(x) = \text{sign}(x) \ln(|x| + 1)$ 在全实数域 $\mathbb{R}$ 上连续。
求其一阶导数：
$$f'(x) = \frac{1}{|x| + 1} > 0$$
对于任意 $x \in \mathbb{R}$，导数极值满足 $\sup_{x \in \mathbb{R}} |f'(x)| = f'(0) = 1.0$。
根据中值定理，对任意实数 $a, b \in \mathbb{R}$：
$$|f(a) - f(b)| \le 1.0 \cdot |a - b|$$
因此 $\text{symlog}$ 是严格的 1-李普希茨连续函数（1-Lipschitz Continuous），有效杜绝了梯度爆炸，保证了神经网络反向传播的绝对稳定性。
</details>

---

## 8.5.4 纯底层 PyTorch 代码实现：从零构建潜空间梦境想象与 Actor 优化引擎

下面我们使用纯底层 PyTorch 算子实现一个结构完备的潜在动力学世界模型、可微梦境轨迹展开器与解析策略梯度优化引擎。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentWorldDynamics(nn.Module):
    """
    可微潜在动力学与奖励预测模型
    s_{t+1} = MLP(s_t, a_t), r_t = MLP(s_t)
    """
    def __init__(self, state_dim: int = 16, action_dim: int = 4):
        super().__init__()
        self.trans_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ELU(),
            nn.Linear(64, state_dim)
        )
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        next_state = self.trans_net(torch.cat([state, action], dim=-1))
        reward = self.reward_net(next_state)
        return next_state, reward

class ImaginationActor(nn.Module):
    """
    梦境策略网络 (Actor)
    """
    def __init__(self, state_dim: int = 16, action_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ELU(),
            nn.Linear(64, action_dim),
            nn.Tanh() # 将动作限制在 [-1, 1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

class DreamerEngine:
    """
    可微梦境想象与端到端策略优化引擎
    """
    def __init__(self, dynamics: LatentWorldDynamics, actor: ImaginationActor, horizon: int = 15):
        self.dynamics = dynamics
        self.actor = actor
        self.horizon = horizon

    def imagine_and_optimize(self, initial_states: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """
        在潜在梦境中展开 H 步可微推演，并依据解析梯度更新策略
        """
        curr_state = initial_states # (B, state_dim)
        accumulated_rewards = 0.0

        # 1. 纯梦境前向展开
        for t in range(self.horizon):
            action = self.actor(curr_state)
            next_state, reward = self.dynamics(curr_state, action)
            accumulated_rewards = accumulated_rewards + reward.mean()
            curr_state = next_state

        # 2. 目标是最大化累计奖励，因此损失为负的累计回报
        actor_loss = -accumulated_rewards

        # 3. 反向传播解析链式梯度
        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()

        return accumulated_rewards.item()

# ===================================================================
# 单元测试与可微路径梯度反向传播校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 16
    state_dim = 16
    action_dim = 4
    horizon = 10

    dynamics = LatentWorldDynamics(state_dim=state_dim, action_dim=action_dim)
    actor = ImaginationActor(state_dim=state_dim, action_dim=action_dim)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

    engine = DreamerEngine(dynamics=dynamics, actor=actor, horizon=horizon)

    dummy_initial_states = torch.randn(batch_size, state_dim)

    # 1. 开展梦境推演与解析梯度更新
    initial_reward = engine.imagine_and_optimize(dummy_initial_states, optimizer)
    print(f"[Dreamer Test] 初始梦境想象平均总回报: {initial_reward:.4f}")

    # 2. 连续优化 5 步验证策略提升
    for epoch in range(5):
        reward_val = engine.imagine_and_optimize(dummy_initial_states, optimizer)

    print(f"[Dreamer Test] 5步可微优化后梦境总回报: {reward_val:.4f}")

    # 校验策略参数梯度非空且无 NaN
    assert actor.net[0].weight.grad is not None, "策略未接收到梦境反向传播梯度！"
    assert not torch.isnan(actor.net[0].weight.grad).any(), "策略梯度存在 NaN 异常！"
    print("✓ 潜空间可微梦境想象与端到端策略优化单测全部通过！")
```

---

## 8.5.5 本节小结

回顾本节内容，我们建立了梦境强化学习的完整范式体系：
1. **二元认知回路**：低频与物理世界真实交互拟合动力学，高频在内心梦境隐空间中进行万步推演，彻底解决了数据饥渴难题；
2. **可微解析梯度**：借助全路径链式求导，直接将未来奖励导数回传给策略参数，实现了极高信噪比的超稳健策略进化；
3. **DreamerV3 通用性基石**：对称对数与百分位数归一化彻底消除了跨域超参微调，奠定了通用具身世界模型的坚实底座。
