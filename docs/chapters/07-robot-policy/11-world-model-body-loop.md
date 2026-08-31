# 世界模型与具身控制的闭环

真实机器人每执行一次探索动作，都要付出时间、能耗和硬件风险。World Models 先从像素学习潜在动力学，再在模型生成的轨迹中训练控制器，并把控制器放回 CarRacing 环境闭环驾驶。这个思路减少了部分真实试错需求，却把难点转移到了模型误差与分布外想象上。

<div align="center">

<img src="/figures/07-robot-policy/source/11-world-model-body-loop/wm-fig9.png" alt="World Models 控制器在 CarRacing 中沿赛道闭环驾驶，是模型内训练回到真实环境的输出。" width="86%">

_图 7.11-1：World Models 控制器在 CarRacing 中沿赛道闭环驾驶，是模型内训练回到真实环境的输出。 出处：[World Models，David Ha; Jürgen Schmidhuber，2018](https://arxiv.org/abs/1803.10122)。_

</div>

Ha 和 Schmidhuber 展示了从像素数据学习潜在动力学，并在模型生成的轨迹中训练控制器的可行性 [[Ha & Schmidhuber, 2018]](https://arxiv.org/abs/1803.10122)。RSSM 最初由 PlaNet 引入；Dreamer 在此基础上使用潜在想象轨迹训练 actor 与 critic [[Hafner et al., 2020]](https://arxiv.org/abs/1912.01603)。这些结果来自特定游戏与控制基准，构成了“视觉—模型—策略”闭环的实例，而不是对所有复杂具身任务的完整证明。

<div align="center">

<img src="/figures/07-robot-policy/source/11-world-model-body-loop/wm-fig14.png" alt="同一 World Models 方法在真实 VizDoom 环境中躲避火球，展示另一类闭环控制输出。" width="86%">

_图 7.11-2：同一 World Models 方法在真实 VizDoom 环境中躲避火球，展示另一类闭环控制输出。 出处：[World Models，David Ha; Jürgen Schmidhuber，2018](https://arxiv.org/abs/1803.10122)。_

</div>

本节从运动学状态开始，说明 RSSM 如何结合历史与随机隐变量，再解释 Dreamer 类方法如何在潜在轨迹中训练 actor 和 critic。这里的梯度是对学习模型的梯度估计，不是对真实物理世界的精确求解。

<div align="center">

<img src="/figures/07-robot-policy/source/11-world-model-body-loop/wm-fig8.png" alt="V、M、C 三组件把视觉压缩、潜在动力学与控制器连接成闭环。" width="86%">

_图 7.11-3：V、M、C 三组件把视觉压缩、潜在动力学与控制器连接成闭环。 出处：[World Models，David Ha; Jürgen Schmidhuber，2018](https://arxiv.org/abs/1803.10122)。_

</div>

## 从真实物理世界到隐空间投影

在任何控制问题中，我们首先需要理解环境是如何随着时间演化的。假设我们在高中的物理课上研究一个在光滑水平面上做直线运动的滑块。

如果我们知道滑块在时刻 $t$ 的位置 $x_t$ 和速度 $v_t$，并对其施加一个恒定的加速度（控制量）$a_t$，那么经过一个极短的时间间隔 $\Delta t$ 后，滑块的新位置 $x_{t+1}$ 和新速度 $v_{t+1}$ 可以通过最基础的运动学公式精确计算：

$$
\begin{aligned}
v_{t+1} &= v_t + a_t \Delta t \\
x_{t+1} &= x_t + v_t \Delta t + \frac{1}{2} a_t \Delta t^2
\end{aligned}
$$

在这个简单的系统中，滑块的真实物理状态是由位置和速度完全描述的。我们可以将这两个标量组合成一个状态向量 $\mathbf{s}_t = [x_t, v_t]^\top$。此时，上述物理规律可以抽象为一个确定性的转移函数 $f$：

$$
\mathbf{s}_{t+1} = f(\mathbf{s}_t, \mathbf{a}_t)
$$

在四足机器人或多自由度机械臂中，状态维度更高，智能体也通常无法直接获得完整物理状态 $\mathbf{s}_t$，只能使用相机、本体传感器等产生的高维观测 $\mathbf{o}_t$。

直接在像素空间建模转移需要同时处理任务相关结构和大量视觉细节。作为暂时忽略历史的入门写法，可以先用编码器 $E_\phi$ 把高维观测 $\mathbf{o}_t$ 压缩到潜在状态空间：

$$
\mathbf{z}_t \sim q_\phi^{\mathrm{frame}}(\mathbf{z}_t \mid \mathbf{o}_t)
$$

这里，$\mathbf{z}_t$ 是观测的随机隐表示。我们希望它包含物体相对位置、姿态等控制相关信息，但这些语义需要通过下游任务或表征分析确认。

## 循环状态空间模型（RSSM）：梦境的引擎

既然有了隐状态 $\mathbf{z}_t$，我们是否可以直接在隐空间中构建简单的马尔可夫转移函数 $\mathbf{z}_{t+1} = f(\mathbf{z}_t, \mathbf{a}_t)$ 呢？

不一定可以。单帧观测通常不能确定速度或被遮挡物体的状态，因此单帧编码 $\mathbf{z}_t$ 未必满足马尔可夫性。此外，接触、传感器噪声和未观测变量也需要用不确定性表示。

为了解决这个问题，循环状态空间模型（Recurrent State Space Model, RSSM）被提出。RSSM 将世界模型的状态一分为二：确定性隐状态（Deterministic Hidden State）$\mathbf{h}_t$ 和随机性隐状态（Stochastic Latent State）$\mathbf{z}_t$。

确定性状态 $\mathbf{h}_t$ 由循环网络汇总历史，随机状态 $\mathbf{z}_t$ 表示当前时刻在该历史条件下仍需用分布描述的信息。两者是模型的计算分工，不对应“经典”与“量子”两类真实物理量。

RSSM 的内部动力学机制可以通过以下严谨的概率模型定义：

1. **确定性序列推断（Sequence Model）**：利用循环神经网络（如 GRU），基于过去的历史信息和动作，更新确定性状态：

   $$
   \mathbf{h}_t = f_\theta(\mathbf{h}_{t-1}, \mathbf{z}_{t-1}, \mathbf{a}_{t-1})
   $$

2. **先验动态模型（Dynamics Predictor）**：在没有接收当前观测时，根据历史状态预测当前随机状态：

   $$
   \hat{\mathbf{z}}_t \sim p_\theta(\hat{\mathbf{z}}_t \mid \mathbf{h}_t)
   $$

3. **后验表征模型（Representation Model）**：当接收到当前真实的视觉观测 $\mathbf{o}_t$ 后，结合历史记忆，得出对当前真实世界更精准的概率认知（用于修正内部模型和现实对齐）：

   $$
   \mathbf{z}_t \sim q_\phi(\mathbf{z}_t \mid \mathbf{h}_t, \mathbf{o}_t)
   $$

   $$
   $$

<div align="center">

<img src="/figures/07-robot-policy/latex/11-world-model-body-loop/rssm-prior-posterior-split.png" alt="RSSM 同一时刻的无观测先验与有观测后验分支" width="86%">

_图 7.11-4：h_t 由上一状态与动作递推；prior 只看 h_t，posterior 额外读取真实观测 o_t，并在训练时校正同一时刻的随机状态。本文根据上式绘制；TikZ/LaTeX 编译。_

</div>

除此之外，具身控制还必须衡量行为的优劣，世界模型需要预测基于当前状态所能获得的奖励（Reward）：

$$
r_t \sim p_\theta(r_t \mid \mathbf{h}_t, \mathbf{z}_t)
$$

在世界模型的离线训练阶段，目标是使得未见观测的“先验预测”尽可能逼近包含观测事实的“后验认知”。因此，不仅要最小化图像重建和奖励预测的误差，还需要最小化先验分布和后验分布之间的 KL 散度（Kullback-Leibler Divergence）。

## 隐空间中的想象与解析梯度优化

训练世界模型后，智能体可以从真实观测得到的后验状态出发，在隐空间展开有限长度的想象轨迹。它仍需持续用真实数据校正模型，不能永久切断传感器。

<div align="center">

<img src="/figures/07-robot-policy/source/11-world-model-body-loop/dreamer-fig1.png" alt="Dreamer 在潜在动力学中展开想象轨迹，并用预测价值更新行为。" width="86%">

_图 7.11-5：Dreamer 在潜在动力学中展开想象轨迹，并用预测价值更新行为。 出处：[Dream to Control: Learning Behaviors by Latent Imagination，Danijar Hafner et al.，2020](https://arxiv.org/abs/1912.01603)。_

</div>

在时刻 $t$，智能体根据真实观测 $\mathbf{o}_t$ 计算后验状态 $\mathbf{z}_t$ 和历史特征 $\mathbf{h}_t$，再使用 RSSM 的状态转移与奖励模型向前推演。

假设动作策略为 $\mathbf{a}_\tau \sim \pi_\psi(\mathbf{a}_\tau \mid \hat{\mathbf{z}}_\tau, \mathbf{h}_\tau)$。从真实时刻 $t$ 向前展开 $H$ 步，模型生成的潜在轨迹为：

$$
(\mathbf{h}_t, \mathbf{z}_t), \mathbf{a}_t, r_t, (\mathbf{h}_{t+1}, \hat{\mathbf{z}}_{t+1}), \mathbf{a}_{t+1}, r_{t+1}, \dots, (\mathbf{h}_{t+H}, \hat{\mathbf{z}}_{t+H})
$$

策略目标是最大化累积回报。给定折扣因子 $\gamma \in (0,1)$，一段长度为 $H$ 的想象轨迹可以用价值网络补上截断后的回报：

$$
G_\tau = \sum_{k=0}^{H-1} \gamma^k \hat r_{\tau+k}
+ \gamma^H v_\xi(\mathbf{h}_{\tau+H},\hat{\mathbf{z}}_{\tau+H})
$$

$v_\xi$ 是独立训练的价值网络，用来估计截断点之后的回报。Dreamer 实际使用的回报目标还包含 continuation 和 $\lambda$-return 等细节，这里只写最简单的 bootstrap 形式。

真实环境通常不向学习算法提供可直接反向传播的动力学计算图。无模型方法因此使用策略梯度、价值估计等采样式估计器；它们的方差和样本效率取决于具体算法。

学习到的动力学与奖励模型由神经网络构成。对连续动作和可重参数化的随机状态，可以沿想象轨迹使用路径导数，把回报梯度传回 actor。离散变量、不可重参数化采样或停止梯度的位置则需要其他估计器。

actor 的目标可以写成想象回报的期望：

$$
J(\psi)=\mathbb{E}_{p_\theta,\pi_\psi}[G_\tau],
\qquad \nabla_\psi J(\psi)=\nabla_\psi\mathbb{E}[G_\tau]
$$

自动微分会计算动作对所有后续隐状态和奖励的总导数，而不只是单步的 $\partial r/\partial a$。这常称为 dynamics gradients 或 pathwise gradients。它对学习模型而言可以精确计算，但相对于真实环境仍有模型偏差。

## 代码实现：在梦境计算图中推演闭环

下面，我们将通过框架代码构建 RSSM 的核心闭环过程。这段实现精炼地展示了如何维持确定性与随机性状态的双轨更新，并在计算图内展开梦境以实现完全可微的策略优化。

```python
import torch
from torch import nn
from torch.distributions import Normal

class RSSMCell(nn.Module):
    """循环状态空间模型的核心单元 (梦境引擎)"""
    def __init__(self, action_dim, hidden_dim=200, latent_dim=30):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 确定性状态更新网络 (GRU核心)
        # 输入维度: 动作空间 + 前一时刻随机隐状态
        self.gru = nn.GRUCell(action_dim + latent_dim, hidden_dim)

        # 先验网络 (Prior / Dynamics): p(z_t | h_t)
        self.prior_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )
        self.prior_mean = nn.Linear(hidden_dim, latent_dim)
        self.prior_std = nn.Linear(hidden_dim, latent_dim)

        # 奖励预测网络
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward_prior(self, h_t):
        # [计算当前状态下向未来推演的概率分布参数]
        feat = self.prior_mlp(h_t)
        mean = self.prior_mean(feat)
        # 利用 softplus 函数强制标准差严格为正，并附加底噪避免坍缩
        std = nn.functional.softplus(self.prior_std(feat)) + 0.1
        return mean, std

    def step(self, action, h_prev, z_prev):
        # [步骤1: 计算宏观确定性隐状态 h_t]
        # action 维度: (batch, action_dim)
        # z_prev 维度: (batch, latent_dim)
        gru_input = torch.cat([action, z_prev], dim=-1)
        h_t = self.gru(gru_input, h_prev)

        # [步骤2: 利用先验网络预测微观随机状态 z_t]
        mean, std = self.forward_prior(h_t)

        # rsample使用重参数化采样，计算图不会在采样处被切断
        dist = Normal(mean, std)
        z_t = dist.rsample()

        # [步骤3: 预测智能体在当前世界格局下获得的奖赏]
        state_feature = torch.cat([h_t, z_t], dim=-1)
        reward_pred = self.reward_predictor(state_feature)

        return h_t, z_t, reward_pred

class Actor(nn.Module):
    """在梦境中输出控制信号的策略网络"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 200),
            nn.ELU(),
            nn.Linear(200, action_dim),
            nn.Tanh() # 强制动作空间约束在 [-1, 1] 物理有效范围内
        )

    def forward(self, state_feature):
        # 依赖完整的状态特征 (h_t 与 z_t 的拼接) 做出动作决策
        return self.net(state_feature)

# [初始化闭环组件]
batch_size = 4
action_dim = 2
hidden_dim = 200
latent_dim = 30
horizon = 15 # 定义梦境推演的时间视野长度 H

rssm = RSSMCell(action_dim, hidden_dim, latent_dim)
actor = Actor(hidden_dim + latent_dim, action_dim)
# actor更新时冻结世界模型参数，但仍保留状态对动作的梯度
for parameter in rssm.parameters():
    parameter.requires_grad_(False)
optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)

# 假设我们在真实的具身环境中，通过视觉观测刚刚建立起对当前 t 时刻局势的认知
h_t = torch.zeros(batch_size, hidden_dim)
z_t = torch.zeros(batch_size, latent_dim)

# [开始闭合控制环：Latent Imagination Loop]
total_predicted_reward = 0
gamma = 0.99

# 在循环内部，智能体切断传感器，完全靠神经网络在时间轴上向未来延展梦境
for t in range(horizon):
    state_feature = torch.cat([h_t, z_t], dim=-1)

    # 策略网络给出一个连续动作 (注意：此处前向传播维持了完整的微分轨迹)
    action = actor(state_feature)

    # 世界引擎承接动作，演化出下一步的时空状态和奖励
    h_t, z_t, reward = rssm.step(action, h_t, z_t)

    total_predicted_reward = total_predicted_reward + (gamma ** t) * reward

# 计算策略目标，我们期望最大化梦境中的总期望奖励
# 由于需要执行梯度下降，这里直接取负值作为损失函数
loss = -total_predicted_reward.mean()

# 清空梯度
optimizer.zero_grad()
# 梯度沿固定世界模型中的15步想象轨迹传回策略参数
loss.backward()
optimizer.step()

print("想象轨迹与策略反向传播完成")
```

`loss.backward()` 沿想象轨迹把预测回报对动作的影响传回 actor。示例省略了后验编码器、观测解码器、critic、continuation 模型和 $\lambda$-return，也没有展示世界模型训练，因此它只说明路径梯度的计算图，而不是完整 Dreamer 实现。

## 小结

- 世界模型闭环包含真实观测编码、潜在动力学学习、想象轨迹和真实环境再校正。
- **RSSM** 用确定性状态汇总历史，用随机状态表示条件不确定性；这种结构是否合适要由预测与控制结果检验。
- 对连续、可重参数化的想象轨迹，可以用路径梯度训练 actor。它减少真实交互，却仍受模型偏差、价值估计和分布外轨迹影响。
