# 7.11 世界模型与具身策略闭环

在具身智能的宏大版图中，我们已经探索了从底层浮动基座控制（WBC）到顶层视觉-语言大模型（VLA）的诸多前沿技术。然而，具身智能系统面临着一个最严苛的现实矛盾——**物理世界试错的高昂代价**。

如果像传统的无模型强化学习（Model-Free RL）那样，让重达数十公斤的人形机器人直接在真实水泥地面上通过数百万次跌倒撞击来摸索平衡算法，那么硬件损耗、电机烧毁与实验安全成本将是任何团队都无法承受的天文数字。

人类之所以能够在极少犯错的前提下学会驾驶汽车或操控复杂工具，是因为我们的大脑拥有一座运转不休的“内心物理模拟器”。我们在动手之前，已经在脑海中预演了动作的物理后果。

本节我们将深入探讨具身智能的终极架构之一——**世界模型（World Models）与梦境策略闭环（Latent Imagination Loop）**，剖析机器人如何在其内心构建的世界模型中“做梦”试错，并利用完全可微的计算图实现高效的策略学习。

<div align="center">

<img src="/figures/07-robot-policy/source/11-world-model-body-loop/wm-fig8.png" alt="Ha 与 Schmidhuber 提出的世界模型经典三件套：视觉 V、记忆 M 与控制器 C。" width="86%">

_图 7.11-1：Ha 与 Schmidhuber 提出的世界模型经典三件套：视觉 V、记忆 M 与控制器 C。 出处：[World Models，David Ha; Jürgen Schmidhuber，2018](https://arxiv.org/abs/1803.10122)。_

</div>

---

## 7.11.1 认知与物理基石：人类大脑的心理模拟器与具身闭环演进

要理解世界模型闭环的设计哲学，我们首先需要回顾认知心理学对人类思维机制的奠基性探索。

### 1. 心理模拟器假说（Mental Simulator）
早在 1943 年，认知科学家 Kenneth Craik 就在《解释的本质》中提出了著名的理论：**人类的大脑在内部持续运行着一个关于外部物理现实的微型模拟假说**。
- 当你闭上双眼，思考“如果我现在把手中的咖啡杯松开”时，你无需真正松手，就能在脑海中瞬间预知杯子会垂直加速下落并在地板上摔得粉碎；
- 当你计划在一个狭窄的停车位侧方停车时，你的大脑在转动方向盘之前，已经在内心推演了车身转角与障碍物距离的几何轨迹。

这种在内心展开的“虚拟预演”，使得人类能够以极高的**样本效率（Sample Efficiency）**和近乎零的物理破坏代价，在复杂环境中快速做出最优决策。

### 2. 从 World Models 到 Dreamer 系列的演进
2018 年，David Ha 与神经网络先驱 Jürgen Schmidhuber 提出了开创性的 **World Models** 架构，首次证明了智能体可以完全在由循环神经网络构建的“虚拟梦境”中学会赛车游戏。

随后，Danijar Hafner 等人相继推出了 **DreamerV1**、**DreamerV2** 与 **DreamerV3**，进一步将世界模型推广到了连续动作物理控制与稀疏奖励环境，奠定了现代具身世界模型的技术基石。

<div align="center">

<img src="/figures/07-robot-policy/source/11-world-model-body-loop/wm-fig9.png" alt="确定性历史状态与随机潜状态在时间步间循环流动" width="86%">

_图 7.11-2：确定性历史状态与随机潜状态在时间步间循环流动，预测观察、奖励和折扣因子。_

</div>

---

## 7.11.2 核心数学推导一：循环状态空间模型（RSSM）的双轨动力学

在真实物理环境中，机器人的运动状态演化同时包含两个层面的规律：
1. **宏观确定性物理惯性**：例如机械臂一旦具有了向前的角速度，由于惯性，它在下一时刻必然会继续向前运动（具备强确定性时序依赖）；
2. **微观物理随机不确定性**：例如接触面微小凹凸导致的微小跳动、关节齿轮的微小间隙抖动（具备随机概率波动）。

为了同时精确捕捉这两种物理特性，Danijar Hafner 等人设计了**循环状态空间模型（Recurrent State-Space Model, RSSM）**的双轨状态演化架构。

<div align="center">

<img src="/figures/07-robot-policy/source/11-world-model-body-loop/wm-fig14.png" alt="RSSM 结合确定性循环路径与随机状态路径，分别在过滤与想象中展开。" width="86%">

_图 7.11-3：RSSM 结合确定性循环路径与随机状态路径，分别在过滤与想象中展开。 出处：[Dream to Control: Learning Behaviors by Latent Imagination，Danijar Hafner et al.，2020](https://arxiv.org/abs/1912.01603)。_

</div>

### 1. 双轨状态递推方程
在时刻 $t$，系统的全状态被解耦为一个确定性向量 $\mathbf{h}_t$ 与一个随机性向量 $\mathbf{z}_t$：
1. **确定性状态更新（Deterministic Path）**：采用门控循环单元（GRU），根据上一时刻的确定性状态 $\mathbf{h}_{t-1}$、随机状态 $\mathbf{z}_{t-1}$ 与动作 $\mathbf{a}_{t-1}$，计算当前的宏观惯性状态：
   $$\mathbf{h}_t = \text{GRU}\left(\mathbf{h}_{t-1}, [\mathbf{a}_{t-1}; \mathbf{z}_{t-1}]\right) \in \mathbb{R}^{D_h}$$
2. **随机状态先验预测（Stochastic Prior Path）**：在不依赖当前真实图像的前提下，由前馈神经网络根据宏观状态 $\mathbf{h}_t$ 预测微观状态的均值与方差，并从中采样出随机状态：
   $$\hat{\mathbf{z}}_t \sim p_\theta(\hat{\mathbf{z}}_t \mid \mathbf{h}_t) = \mathcal{N}\left(\boldsymbol{\mu}_\theta(\mathbf{h}_t), \boldsymbol{\sigma}_\theta^2(\mathbf{h}_t)\right) \in \mathbb{R}^{D_z}$$

### 2. 重参数化技巧（Reparameterization Trick）
在计算图中，直接从正态分布中“随机掷骰子采样”会彻底切断梯度的反向传播路径。

为了让反向传播算法能够穿透随机采样层，我们使用经典的**重参数化技巧**：将随机性剥离为一个与网络参数完全无关的独立标准高斯白噪声 $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$：

$$\mathbf{z}_t = \boldsymbol{\mu}_\theta(\mathbf{h}_t) + \boldsymbol{\sigma}_\theta(\mathbf{h}_t) \odot \boldsymbol{\epsilon}$$

> **初等代数直觉**：
> 这一公式将原本不可导的“黑盒采样”，转化为初等代数中一目了然的线性函数：$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \boldsymbol{\epsilon}$。
> 此时，下游的求导梯度可以顺畅地流过均值 $\boldsymbol{\mu}$ 与方差 $\boldsymbol{\sigma}$，直接更新底层的神经网络权重！

<details>
<summary><b>深入推导：世界模型变分自编码器（ELBO）自由能分解与 KL 正则项（点击展开查看完整推导）</b></summary>

世界模型的训练目标是最大化多模态环境轨迹的对数边缘似然 $\log p_\theta(\mathbf{x}_{1:T}, r_{1:T})$。
引入由后验编码器定义的变分分布 $q_\phi(\mathbf{z}_{1:T} \mid \mathbf{x}_{1:T}, \mathbf{a}_{1:T})$，由琴生不等式（Jensen's Inequality）可严格推导其证据下界（ELBO）：
$$\ln p_\theta(\mathbf{x}_{1:T}, r_{1:T}) \ge \sum_{t=1}^T \Big( \underbrace{\mathbb{E}_{q_\phi}[\ln p_\theta(\mathbf{x}_t \mid \mathbf{h}_t, \mathbf{z}_t)]}_{\text{图像观测重构项}} + \underbrace{\mathbb{E}_{q_\phi}[\ln p_\theta(r_t \mid \mathbf{h}_t, \mathbf{z}_t)]}_{\text{物理奖励预测项}} - \underbrace{\mathbb{E}_{q_\phi}[D_{\text{KL}}(q_\phi(\mathbf{z}_t \mid \mathbf{h}_t, \mathbf{x}_t) \parallel p_\theta(\hat{\mathbf{z}}_t \mid \mathbf{h}_t))]}_{\text{先验-后验动力学一致性约束}} \Big)$$
其中 KL 散度项迫使模型内心在无观测时单靠前向推演出的先验分布 $p_\theta(\hat{\mathbf{z}}_t)$ 尽量贴近拥有真实相机输入时计算出的后验分布 $q_\phi(\mathbf{z}_t)$。
</details>

---

## 7.7.3 核心数学推导二：隐空间梦境想象与端到端路径导数

一旦世界模型（RSSM）在少量的真实物理数据上训练成熟，机器人就可以**彻底切断外部传感器**，完全在其内心展开长达数十步的“潜在梦境推演（Latent Imagination）”。

<div align="center">

<img src="/figures/07-robot-policy/source/11-world-model-body-loop/dreamer-fig1.png" alt="Dreamer 在潜在动力学中展开想象轨迹，并用预测价值更新行为。" width="86%">

_图 7.11-4：Dreamer 在潜在动力学中展开想象轨迹，并用预测价值更新行为。 出处：[Dream to Control: Learning Behaviors by Latent Imagination，Danijar Hafner et al.，2020](https://arxiv.org/abs/1912.01603)。_

</div>

### 1. 梦境轨迹的前向展开
在时刻 $t$，智能体从当前真实的物理状态特征 $(\mathbf{h}_t, \mathbf{z}_t)$ 出发，策略网络 $\pi_\psi$ 按照内心推演展开 $H$ 步（例如 $H = 15$）：

$$\begin{aligned}
\mathbf{a}_\tau &= \pi_\psi(\mathbf{h}_\tau, \mathbf{z}_\tau) \\
(\mathbf{h}_{\tau+1}, \mathbf{z}_{\tau+1}) &= \text{RSSM\_Step}(\mathbf{h}_\tau, \mathbf{z}_\tau, \mathbf{a}_\tau) \\
\hat{r}_{\tau+1} &= \text{Reward\_Head}(\mathbf{h}_{\tau+1}, \mathbf{z}_{\tau+1})
\end{aligned}$$

整段想象轨迹的总回报由贴现累积和给出：

$$G_t = \sum_{k=0}^{H-1} \gamma^k \hat{r}_{t+k+1} + \gamma^H v_\xi(\mathbf{h}_{t+H}, \mathbf{z}_{t+H})$$

其中 $\gamma \in (0, 1)$ 为时间贴现因子（通常取 $0.99$），$v_\xi$ 为评估长远未来收益的价值网络（Critic）。

### 2. 解析路径导数（Dynamics Gradients / Pathwise Gradients）
传统的无模型强化学习无法直接对物理环境求导，只能采用高方差的随机策略梯度（如 REINFORCE 或 PPO）。

而在世界模型内部，由于整条梦境轨迹中的状态转移方程和奖励函数全部是由连续可微的神经网络构成的，整个 15 步的推演过程在 PyTorch 中形成了一条**完全打通、处处可导的长计算图**！

策略参数 $\psi$ 的优化目标为最大化梦境期望回报 $J(\psi) = \mathbb{E}[G_t]$。我们可以直接通过微积分链式法则，将回报函数的梯度沿着时间轴反向精准反传回策略网络：

$$\frac{\partial G_t}{\partial \psi} = \sum_{\tau=t}^{t+H} \frac{\partial G_t}{\partial \mathbf{a}_\tau} \frac{\partial \mathbf{a}_\tau}{\partial \psi} = \sum_{\tau=t}^{t+H} \left( \sum_{k=\tau}^{t+H} \frac{\partial \hat{r}_k}{\partial \mathbf{s}_k} \frac{\partial \mathbf{s}_k}{\partial \mathbf{a}_\tau} \right) \frac{\partial \mathbf{a}_\tau}{\partial \psi}$$

这一路径导数赋予了机器人极其惊人的学习速度：它直接指明了“电机在第 3 步往左微调几牛·米，就能使第 15 步的杯子不倾倒”，彻底告别了盲目胡乱尝试的传统探索瓶颈。

<details>
<summary><b>深入推导：时序反向传播（BPTT）沿着潜在动力学链条的链式法则全导数分解（点击展开查看完整推导）</b></summary>

在长达 $H$ 步的计算图中，定义总损失标量 $\mathcal{L}_{\text{actor}} = -G_t$。
根据时序反向传播算法（BPTT），下游状态 $\mathbf{s}_{k+1}$ 对上游动作 $\mathbf{a}_\tau$ 的全雅可比矩阵由多步状态转移矩阵级联连乘得到：
$$\frac{\partial \mathbf{s}_k}{\partial \mathbf{a}_\tau} = \frac{\partial \mathbf{s}_k}{\partial \mathbf{s}_{k-1}} \frac{\partial \mathbf{s}_{k-1}}{\partial \mathbf{s}_{k-2}} \dots \frac{\partial \mathbf{s}_{\tau+1}}{\partial \mathbf{a}_\tau} = \left( \prod_{j=\tau+1}^{k-1} \frac{\partial \mathbf{s}_{j+1}}{\partial \mathbf{s}_j} \right) \frac{\partial \mathbf{s}_{\tau+1}}{\partial \mathbf{a}_\tau}$$
为防止时间视野 $H$ 过长时雅可比矩阵连乘引发的梯度爆炸与弥散，现代世界模型通常引入层归一化（LayerNorm）与 $\lambda$-return 广义优势估计，将方差与偏差维持在最佳平衡点。
</details>

---

## 7.11.4 真实物理世界与虚拟梦境的双闭环体系

现代具身系统通过**双闭环交互体系**实现物理安全与超高智能的兼得：

<div align="center">

<img src="/figures/07-robot-policy/latex/11-world-model-body-loop/rssm-prior-posterior-split.png" alt="世界模型感知-想象-执行-校正完整闭环架构" width="86%">

_图 7.11-5：世界模型闭环：真实观测校正潜在状态；策略在梦境动力学中展开可微想象；最优动作下发至物理本体执行。_

</div>

1. **外闭环（真实物理交互）**：机器人以安全保守的动作在真实环境中采集少量的真机轨迹数据，存入经验回放池；
2. **内闭环（内心虚拟梦境）**：世界模型利用经验数据更新动力学规律；策略网络切断真实传感器，在内心展开每秒数万次的超高速梦境想象，通过端到端路径导数打磨出最优策略；
3. **闭环校准**：打磨好的策略重新部署回真机，若遇到真实世界的意外扰动，世界模型立刻利用新数据修正内心的动力学认知，形成不断自进化的智能闭环。

---

## 7.11.5 纯底层 PyTorch 代码实现：从零构建梦境引擎与端到端闭环

下面我们使用纯底层 PyTorch 算子实现一个完整的循环状态空间模型（RSSMCell）、动作策略网络（Actor）以及在计算图内展开 15 步梦境想象并完成路径导数反向传播的完整流程。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class RSSMCell(nn.Module):
    """
    循环状态空间模型核心单元 (RSSM Dream Engine)
    维护确定性状态 h_t (GRU) 与随机性状态 z_t (Normal) 的双轨演化
    """
    def __init__(self, action_dim: int = 2, hidden_dim: int = 200, latent_dim: int = 30):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 1. 确定性状态递推网络: h_t = GRU(h_{t-1}, [a_{t-1}; z_{t-1}])
        self.gru = nn.GRUCell(action_dim + latent_dim, hidden_dim)

        # 2. 随机状态先验预测网络: p(z_t | h_t) -> (mean, std)
        self.prior_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        self.prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.prior_std = nn.Linear(hidden_dim, latent_dim)

        # 3. 物理奖励预测网络: r_t = R(h_t, z_t)
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward_prior(self, h_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        根据当前确定性状态计算未来随机状态的正态分布参数
        """
        feat = self.prior_mlp(h_t)
        mean = self.prior_mean(feat)
        # softplus 保证方差严格为正，附加 0.1 底噪避免数值崩溃
        std = F.softplus(self.prior_std(feat)) + 0.1
        return mean, std

    def step(
        self, action: torch.Tensor, h_prev: torch.Tensor, z_prev: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        在内心梦境中向前演化一步
        :param action: (B, action_dim) 策略输出的连续动作
        :param h_prev: (B, hidden_dim) 前一步确定性状态
        :param z_prev: (B, latent_dim) 前一步随机状态
        :return: (h_next, z_next, reward_pred)
        """
        # 1. 计算宏观确定性状态
        gru_input = torch.cat([action, z_prev], dim=-1)
        h_t = self.gru(gru_input, h_prev)

        # 2. 计算微观随机状态参数并通过重参数化采样
        mean, std = self.forward_prior(h_t)
        dist = Normal(mean, std)
        # rsample 保证采样操作可导，计算图不被切断
        z_t = dist.rsample()

        # 3. 预测当前物理状态获得的即时奖励
        state_feature = torch.cat([h_t, z_t], dim=-1)
        reward_pred = self.reward_predictor(state_feature)

        return h_t, z_t, reward_pred

class Actor(nn.Module):
    """
    在梦境中输出控制动作的策略网络
    """
    def __init__(self, state_dim: int = 230, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 200),
            nn.ELU(),
            nn.Linear(200, action_dim),
            nn.Tanh() # 严格约束动作输出在 [-1.0, 1.0] 物理有效范围内
        )

    def forward(self, state_feature: torch.Tensor) -> torch.Tensor:
        return self.net(state_feature)

# ===================================================================
# 单元测试：在梦境计算图中展开 15 步推演并反向传播路径导数
# ===================================================================
if __name__ == "__main__":
    batch_size = 4
    action_dim = 2
    hidden_dim = 200
    latent_dim = 30
    horizon = 15 # 梦境想象时间步长度
    gamma = 0.99 # 贴现系数

    # 1. 初始化世界模型与策略网络
    rssm = RSSMCell(action_dim=action_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    actor = Actor(state_dim=hidden_dim + latent_dim, action_dim=action_dim)

    # 策略更新时冻结世界模型参数，但保留状态对动作的求导路径
    for param in rssm.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

    # 2. 模拟真机初始状态特征 (h_0, z_0)
    h_t = torch.zeros(batch_size, hidden_dim)
    z_t = torch.zeros(batch_size, latent_dim)

    # 3. 启动内心梦境推演循环 (Latent Imagination Loop)
    total_imagined_reward = torch.zeros(batch_size, 1)

    for step_idx in range(horizon):
        state_feature = torch.cat([h_t, z_t], dim=-1)

        # 策略网络输出连续动作 (保持可微计算图)
        action = actor(state_feature)

        # 世界模型演化出下一步状态与奖励
        h_t, z_t, reward_pred = rssm.step(action, h_t, z_t)

        # 累加贴现回报
        total_imagined_reward = total_imagined_reward + (gamma ** step_idx) * reward_pred

    # 4. 计算策略损失 (最大化回报即最小化负回报)
    loss = -total_imagined_reward.mean()

    # 5. 端到端反向传播路径导数并更新策略
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"[World Model Loop Test] 梦境推演时间步长: {horizon}")
    print(f"[World Model Loop Test] 15 步想象轨迹预测总回报: {-loss.item():.4f}")
    print(f"[World Model Loop Test] 策略网络第一层权重梯度范数: {actor.net[0].weight.grad.norm().item():.6f}")

    assert actor.net[0].weight.grad is not None, "路径导数反向传播未成功到达策略参数！"
    assert actor.net[0].weight.grad.norm().item() > 0.0, "梯度出现弥散断裂！"
    print("✓ 世界模型 RSSM 双轨推演与端到端路径导数闭环单测全部通过！")
```

---

## 7.11.6 本节小结

回顾本节内容，我们建立了世界模型与具身策略闭环的完整知识图谱：
1. **心理模拟器的物理哲学**：世界模型将高危、昂贵的真实世界试错，转化为在内心神经网络中的低成本、高效率梦境推演；
2. **RSSM 的双轨动力学**：确定性循环状态（GRU）捕捉宏观物理惯性，随机性状态（Normal）表达微观物理不确定性，重参数化技巧保证了采样的处处可微；
3. **端到端路径导数优化**：将原本无法求导的环境黑盒转化为连续可微的神经网络长计算图，通过反向传播直接指引策略的高速迭代；
4. **具身双闭环系统**：外闭环采集真机数据校准认知，内闭环高速做梦迭代策略，构建起安全稳健的自进化智能体系。
