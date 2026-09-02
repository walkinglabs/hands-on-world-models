# 3.6 强化学习基石核心精讲 (RL Foundation Concise)

在探索世界模型与智能决策的整个理论体系中，前五节所介绍的马尔可夫决策过程、经验回放、贝尔曼方程、策略梯度与模型预测控制，共同构成了支撑现代人工智能理解物理世界与自主进化的**五大理论支柱**。

当我们将这些理论碎片拼装为完整的系统时，我们会发现所有决策算法本质上都在回答同一个物理哲学问题：**如何在充满随机扰动与未来不确定性的物理世界中，利用有限的感知与算力资源，求解出长期累积回报最大化的最优控制路径？**

本节我们将以高屋建瓴的全局视角，横向解构无模型强化学习（Model-Free RL）、基于模型的梦境学习（Model-Based World Models）与在线规划（Online MPC）三大主流范式的优劣边界，严密推导连续控制中不可或缺的 **Squashed Gaussian 重参数化策略** 与雅可比变量替换定理，并使用纯底层 PyTorch 实现高精连续控制策略网络。

<div align="center">

<img src="/figures/03-data-and-first-model/source/06-rl-foundation-concise/a3c-fig1.png" alt="Soft Actor-Critic (SAC) 结合最大熵强化学习与双 Q 网络，展示在连续控制基准上的超强表现与样本效率。" width="86%">

_图 3.6-1：Soft Actor-Critic (SAC) 结合最大熵强化学习与双 Q 网络，展示在连续控制基准上的超强表现与样本效率。 出处：[Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor，Tuomas Haarnoja et al.，2018](https://arxiv.org/abs/1801.01290)。_

</div>

---

## 3.6.1 架构与物理全景：三大决策范式的横向深度对比

针对具体的具身控制任务（如机械臂插孔、四足机器人崎岖地形越野），不同算法范式展现出了鲜明的物理特性：

### 1. 无模型强化学习（Model-Free RL, 如 SAC / PPO）
- **核心逻辑**：黑盒试错，直接学习 $Q(s, a)$ 或 $\pi(a \mid s)$，不建立任何环境物理模型；
- **物理优缺点**：最终策略执行延迟极低（单次前向传播 $< 1\text{ ms}$），但数据样本效率极低，需要数千万步真实交互试错。

### 2. 基于世界模型的梦境学习（Model-Based RL, 如 Dreamer / World Models）
- **核心逻辑**：先从少量交互数据中训练一个预测未来物理演变的神经动力学世界模型 $\hat{\mathbf{s}}_{t+1} = f(\hat{\mathbf{s}}_t, \mathbf{a}_t)$，随后在完全脱离真实环境的潜在梦境中推演和训练策略；
- **物理优缺点**：样本效率超越无模型算法百倍以上，但高度依赖世界模型的预测保真度（需防范模型幻觉）。

### 3. 在线轨迹规划与 MPC（Online Planning, 如 CEM / MPPI）
- **核心逻辑**：不训练固化的策略网络，在每个控制瞬间利用物理模型向未来模拟数百条候选路径，挑选最优动作执行；
- **物理优缺点**：对未曾见过的突发障碍物拥有极强的实时闭环自愈能力，但每一步都需要进行高并发数值推演，对边缘端算力要求较高。

<div align="center">

<img src="/figures/03-data-and-first-model/latex/06-rl-foundation-concise/return-recursive-tail.png" alt="三大决策范式（无模型、基于世界模型与在线规划）在样本效率、计算延迟与抗扰动自愈上的全景对比" width="86%">

_图 3.6-2：三大决策范式（无模型、基于世界模型与在线规划）在样本效率、计算延迟与抗扰动自愈上的全景对比。_

</div>

---

## 3.6.2 核心数学推导一：连续动作空间的 Squashed Gaussian 重参数化策略

在机械臂控制中，电机的力矩与转角具有严格的物理上下限 $[-a_{\max}, a_{\max}]$。如果直接使用未经约束的高斯分布采样动作，网络容易输出超出硬件承受能力的危险极限值。

<div align="center">

<img src="/figures/03-data-and-first-model/source/06-rl-foundation-concise/a3c-fig1.png" alt="PETS 在高难度复杂动力学控制任务中对比无模型算法的学习速度与最终性能。" width="86%">

_图 3.6-3：PETS 在高难度复杂动力学控制任务中对比无模型算法的学习速度与最终性能。 出处：[Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models，Kurtland Chua et al.，2018](https://arxiv.org/abs/1805.12114)。_

</div>

Soft Actor-Critic（SAC）提出了**双曲正切挤压高斯分布（Squashed Gaussian Policy）**：

### 1. 两步重参数化动作采样流程
1. **采样无界潜在高斯变量**：
   $$\mathbf{u} = \boldsymbol{\mu}_\theta(\mathbf{s}) + \boldsymbol{\sigma}_\theta(\mathbf{s}) \odot \boldsymbol{\epsilon}, \quad \text{其中 } \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
2. **应用双曲正切函数 $\tanh$ 进行平滑有界映射**：
   $$\mathbf{a} = \tanh(\mathbf{u}) \in (-1, 1)$$

### 2. 雅可比概率密度修正公式（Change of Variables）
根据多维概率密度的变量替换定理，由于 $\mathbf{a} = \tanh(\mathbf{u})$ 是单调非线性双射变换，动作 $\mathbf{a}$ 的对数概率密度等于未挤压变量 $\mathbf{u}$ 的高斯概率减去变换雅可比矩阵行列式的绝对值对数：

$$\log \pi(\mathbf{a} \mid \mathbf{s}) = \log \mu(\mathbf{u} \mid \mathbf{s}) - \log \left| \det\left( \frac{\partial \mathbf{a}}{\partial \mathbf{u}} \right) \right|$$

因为 $\mathbf{a}$ 与 $\mathbf{u}$ 是逐元素独立映射，其雅可比矩阵为对角矩阵：

$$\frac{\partial a_i}{\partial u_i} = 1 - \tanh^2(u_i)$$

展开得到惊人优美的初等代数求和修正项：

$$\log \pi(\mathbf{a} \mid \mathbf{s}) = \sum_{i=1}^{d_a} \left[ \log \mathcal{N}(u_i; \; \mu_i, \sigma_i) - \log(1 - \tanh^2(u_i) + \epsilon) \right]$$

### 3. 概率密度修正手算数值算例
设动作维度为 $1$。网络预测输出高斯均值 $\mu = 0.0$，标准差 $\sigma = 1.0$。
采样到一个标准高斯点 $u = 0.0$：
1. **计算挤压后动作**：$a = \tanh(0.0) = 0.0$；
2. **计算原始高斯对数密度**：
   $$\log \mathcal{N}(0.0; 0, 1) = -\frac{1}{2} \ln(2\pi) - \frac{0^2}{2} = -\frac{1}{2} \ln(6.283) \approx -0.9189$$
3. **计算雅可比修正项**：
   $$1 - \tanh^2(0.0) = 1 - 0 = 1.0 \implies \log(1.0) = 0.0$$
4. **最终动作对数概率**：
   $$\log \pi(a=0 \mid s) = -0.9189 - 0.0 = -0.9189$$

初等代数的几步计算清晰证实：雅可比修正项精确补偿了双曲正切函数在两端边界处的非线性拉伸，使得反向传播能够精准计算最大熵正则化梯度！

<details>
<summary><b>深入推导：重参数化概率密度变换中的雅可比对角行列式展开证明（点击展开查看完整推导）</b></summary>

设双射光滑映射 $\mathbf{g}: \mathbb{R}^n \to \Omega \subset \mathbb{R}^n$ 满足 $\mathbf{y} = \mathbf{g}(\mathbf{x})$。
由测度论概率守恒：$\int_B p_Y(\mathbf{y}) d\mathbf{y} = \int_{\mathbf{g}^{-1}(B)} p_X(\mathbf{x}) d\mathbf{x}$。
利用多元重积分换元公式 $d\mathbf{y} = |\det \mathbf{J}_{\mathbf{g}}(\mathbf{x})| d\mathbf{x}$：
$$p_Y(\mathbf{y}) = p_X(\mathbf{g}^{-1}(\mathbf{y})) \cdot |\det \mathbf{J}_{\mathbf{g}}(\mathbf{x})|^{-1}$$
取对数即得 $\log p_Y(\mathbf{y}) = \log p_X(\mathbf{x}) - \log |\det \mathbf{J}_{\mathbf{g}}(\mathbf{x})|$。
由于 $\mathbf{g}$ 为逐元素独立函数 $y_i = \tanh(x_i)$，雅可比矩阵 $\mathbf{J}$ 严格为对角阵，其行列式等于对角元素乘积 $\det \mathbf{J} = \prod_{i=1}^n (1 - \tanh^2(x_i))$，严格证得换元修正公式。
</details>

---

## 3.6.3 纯底层 PyTorch 代码实现：从零手写 Squashed Gaussian 连续控制策略网络

下面我们使用纯底层 PyTorch 算子手写实现完整的 Squashed Gaussian 策略网络与对数概率雅可比解析修正引擎。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SquashedGaussianPolicy(nn.Module):
    """
    带 Tanh 双曲正切挤压的高精连续控制策略网络 (SAC 核心 Actor)
    输出物理有界动作 a in (-1, 1) 并计算精确修正对数似然 log_pi(a|s)
    """
    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(state)
        mean = self.fc_mean(feat)
        # 将 log_std 裁剪在 [-20, 2] 保证数值稳定性
        log_std = torch.clamp(self.fc_log_std(feat), min=-20.0, max=2.0)
        std = log_std.exp()
        return mean, std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        采样有界动作与对数概率
        :return: (action, log_prob)
        """
        mean, std = self.forward(state)
        normal_dist = torch.distributions.Normal(mean, std)

        # 1. 重参数化高斯采样
        u = normal_dist.rsample()

        # 2. Tanh 双曲正切平滑挤压
        action = torch.tanh(u)

        # 3. 计算未挤压对数高斯似然
        raw_log_prob = normal_dist.log_prob(u).sum(dim=-1, keepdim=True)

        # 4. 雅可比换元对角修正: sum(log(1 - tanh(u)^2 + eps))
        eps = 1e-6
        squash_correction = torch.log(1.0 - action.pow(2) + eps).sum(dim=-1, keepdim=True)

        log_prob = raw_log_prob - squash_correction
        return action, log_prob

# ===================================================================
# 单元测试与动作边界与对数似然校验
# ===================================================================
if __name__ == "__main__":
    batch_size = 4
    state_dim = 4
    action_dim = 2

    policy = SquashedGaussianPolicy(state_dim=state_dim, action_dim=action_dim)
    dummy_state = torch.randn(batch_size, state_dim)

    # 采样动作与修正对数似然
    actions, log_probs = policy.sample(dummy_state)

    print(f"[Policy Test] 采样动作形状: {actions.shape}")
    print(f"[Policy Test] 输出动作范围: [{actions.min().item():.4f}, {actions.max().item():.4f}]")
    print(f"[Policy Test] 修正对数似然形状: {log_probs.shape}")
    print(f"[Policy Test] 各样本 log_prob: {[round(x, 4) for x in log_probs.squeeze().tolist()]}")

    assert actions.shape == (batch_size, action_dim), "动作张量维度不符！"
    assert actions.abs().max().item() < 1.0, "动作输出超出物理有界区间 [-1, 1]！"
    assert not torch.isnan(log_probs).any(), "对数似然计算出现 NaN 异常！"
    print("✓ Squashed Gaussian 连续控制策略与雅可比变量替换单测全部通过！")
```

---

## 3.6.4 本节小结

回顾本节内容，我们完成了强化学习基石的系统性全局总结：
1. **三大决策范式定位**：无模型极速低延迟、世界模型极高样本效率、在线规划极强闭环自愈，构成了智能体在不同任务下的完整工具箱；
2. **Squashed Gaussian 物理安全**：利用 Tanh 挤压将无界高斯映射为物理安全区间，并通过雅可比换元公式严密补偿了概率密度偏置；
3. **世界模型承前启后**：牢固掌握数据流管理、贝尔曼价值估计与连续策略生成，为下一章全面进军循环状态空间模型（RSSM）与 Dreamer 梦境世界模型铺平了通往巅峰的大道！
