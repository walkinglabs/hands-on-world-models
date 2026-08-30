# 深度规划网络 (PlaNet)

在强化学习的早期发展中，无模型（Model-Free）方法虽然在诸如雅达利游戏和围棋等复杂任务上取得了令人瞩目的成绩，但其代价是极其高昂的采样复杂性。这意味着智能体需要与环境进行数百万甚至数千万次的交互才能学习到有效的策略。当我们希望将强化学习应用于现实世界的机器人或自动驾驶时，这种对海量数据的依赖成为了不可逾越的鸿沟。

为了解决这一问题，基于模型（Model-Based）的强化学习重新回到了研究者的视野。如果我们能让智能体在脑海中建立一个“世界模型”（World Model），预测其动作可能引发的后果，那么智能体就可以在“想象”中进行反复试错和规划，从而极大地减少在真实物理世界中的试错成本。

然而，当环境的观测是高维度的图像时，准确地预测未来图像（即在像素级别进行规划）被证明是极其困难的。图像中包含了大量与任务无关的冗余信息（例如背景的微小扰动、树叶的飘动）。让神经网络耗费算力去预测每一个像素的精确变化，不仅效率低下，而且容易累积误差。

在这样的背景下，Hafner 等人提出了**深度规划网络**（Deep Planning Network，PlaNet）[[Hafner et al., 2019]](https://arxiv.org/abs/1811.04551)。PlaNet 学习紧凑的潜在动力学，并在潜在空间中规划；训练阶段仍包含观测解码与重构似然，因此不能说它“彻底摒弃”了像素建模。为结合确定性记忆与随机状态，论文引入了**循环状态空间模型**（Recurrent State Space Model, RSSM）。

在本节中，我们将从最基础的序列建模出发，逐步推导 RSSM 的设计逻辑，并详细剖析如何在纯粹的潜在空间中利用交叉熵方法（Cross-Entropy Method, CEM）进行高效规划。

## 潜在空间动力学：为什么我们需要它？

想象你在驾驶汽车。你并不需要预测下一秒视网膜上接收到的每一个光子或每一片树叶的具体位置。相反，你的大脑提取了诸如“前方车辆的速度”、“红绿灯的状态”以及“道路的曲率”等高度抽象的概念，并在这些概念组成的空间中预测如果你踩下刹车会发生什么。

这就是**潜在空间**（Latent Space）的物理直觉。在数学上，我们将高维的观测（例如图像）$o_t \in \mathbb{R}^{H \times W \times C}$ 映射到一个低维的潜在状态变量 $s_t \in \mathbb{R}^{d}$，其中 $d \ll H \times W \times C$。

传统的基于像素的预测模型试图直接拟合映射 $o_{t+1} = f(o_t, a_t)$。这相当于求解一个极其庞大且病态的非线性方程组。而潜在动力学模型则将其分解为以下几个独立的组件：

1. **表征模型**（Representation Model）：将观测压缩为潜在状态 $s_t = \text{Encoder}(o_t)$。
2. **动力学模型**（Dynamics Model/Transition Model）：在潜在空间中预测未来 $s_{t+1} = \text{Transition}(s_t, a_t)$。
3. **奖励模型**（Reward Model）：根据当前的潜在状态预测奖励 $r_t = \text{Reward}(s_t)$。

通过这种分解，规划（Planning）过程只需依赖动力学模型和奖励模型，完全不需要生成任何未来的图像，从而极大地提高了计算效率。

## 循环状态空间模型 (RSSM)

要构建一个能够准确描述环境演化的动力学模型，我们必须处理环境中的两个关键特性：**时间依赖性**（历史决定未来）和**随机性**（未来是不确定的）。

### 确定性路径：RNN 的局限性

如果我们仅仅使用一个标准的循环神经网络（RNN）来建模状态转移，我们可以写出：

$$h_t = f_{\theta}(h_{t-1}, a_{t-1}, o_t)$$

这里 $h_t$ 是 RNN 的隐藏状态。这是一种完全**确定性**（Deterministic）的建模方式。只要给定历史，未来的状态就唯一确定了。

然而，现实世界充满了不可见的部分和固有的随机性（例如掷骰子，或者其他智能体的不可预测行为）。如果强制要求网络预测一个单一的确定性未来，网络通常会输出所有可能未来的“平均值”，导致预测出的潜在状态变得模糊且缺乏物理意义。

### 随机路径：引入状态空间模型

为了捕捉环境的随机性，传统的非线性状态空间模型（State Space Model, SSM）引入了随机变量 $s_t$。其转移过程由概率分布描述：

$$s_t \sim p(s_t \mid s_{t-1}, a_{t-1})$$

观测则是从当前随机状态中生成的：

$$o_t \sim p(o_t \mid s_t)$$

在 SSM 中，$s_t$ 是一个从高斯分布等概率模型中采样得到的随机变量，能够很好地表达不确定性。但纯粹的 SSM 难以记住长期的历史信息，因为马尔可夫假设（Markov Property）限制了它只能依赖前一个状态。

### RSSM：确定性与随机性的完美融合

PlaNet 的核心创新——**循环状态空间模型 (RSSM)**——巧妙地将 RNN 的长期记忆能力与 SSM 的不确定性表达能力结合在了一起。

我们可以将 RSSM 的动力学分解为两部分：一个确定性的隐藏状态 $h_t$，以及一个随机的后验状态 $s_t$。

具体来说，RSSM 包含以下几个核心方程：

1. **确定性状态更新** (Deterministic State Update)：
   $$h_t = f_{\text{RNN}}(h_{t-1}, s_{t-1}, a_{t-1})$$
   这里 $h_t$ 聚合了所有直到时间步 $t$ 的历史信息。

2. **先验模型** (Prior Model / Transition Model)：
   $$p(s_t \mid h_t) = \mathcal{N}(\mu_{\text{prior}}(h_t), \sigma_{\text{prior}}(h_t))$$
   先验模型在**不看**当前观测 $o_t$ 的情况下，仅仅利用历史信息 $h_t$ 来“预测”当前时间步可能发生的随机状态 $s_t$ 的分布。

3. **后验模型** (Posterior Model / Representation Model)：
   $$q(s_t \mid h_t, o_t) = \mathcal{N}(\mu_{\text{post}}(h_t, o_t), \sigma_{\text{post}}(h_t, o_t))$$
   后验模型结合了历史信息 $h_t$ 和当前的真实观测 $o_t$，推断出 $s_t$ 最准确的概率分布。

4. **奖励预测与观测重建**：
   $$r_t \sim p(r_t \mid h_t, s_t)$$
   $$o_t \sim p(o_t \mid h_t, s_t)$$

请注意，在**模型训练**阶段，我们拥有所有的观测数据 $o_t$，因此我们使用后验模型 $q(s_t \mid h_t, o_t)$ 来提取状态；而在**规划（想象未来）**阶段，我们没有未来的观测，只能依靠先验模型 $p(s_t \mid h_t)$ 进行一步步的自回归展开。这正是 RSSM 架构设计的绝妙之处。

## 变分目标函数：优化 RSSM

既然 RSSM 包含了复杂的随机变量，我们如何通过反向传播来训练它？答案是变分推断（Variational Inference）。我们需要最大化观测序列的边际对数似然（Marginal Log-Likelihood）的下界（ELBO）。

对于单个时间步，回忆变分自编码器（VAE）的损失函数，它由“重建误差”和“KL散度（正则化项）”组成。在序列数据中，PlaNet 最大化以下目标函数（相当于最小化损失）：

$$\mathcal{J} = \sum_{t=1}^{T} \mathbb{E}_{q(s_t \mid h_t, o_t)} \left[ \ln p(o_t \mid h_t, s_t) + \ln p(r_t \mid h_t, s_t) \right] - \beta \sum_{t=1}^{T} \text{KL}\left( q(s_t \mid h_t, o_t) \| p(s_t \mid h_t) \right)$$

让我们像拆解精密的机械手表一样，仔细剖析该公式中的每一项：

1. **观测重建项** $\ln p(o_t \mid h_t, s_t)$：迫使潜在状态 $h_t$ 和 $s_t$ 必须保留足够的信息以还原原始图像。通常使用均方误差（MSE）。
2. **奖励预测项** $\ln p(r_t \mid h_t, s_t)$：迫使潜在状态必须包含与任务目标（奖励）相关的信息。这使得我们的潜在空间是任务导向的。
3. **KL 散度项** $\text{KL}(q \| p)$：这是最关键的一项。它迫使**先验分布** $p(s_t \mid h_t)$（仅靠历史预测未来）尽可能地去接近**后验分布** $q(s_t \mid h_t, o_t)$（看到真实结果后的信念）。在规划时，我们只能依靠先验，因此先验预测得越准，规划效果越好。$\beta$ 则是用来控制正则化强度的超参数。

为了使梯度能够反向传播穿过随机采样的 $s_t$，我们必须使用重参数化技巧（Reparameterization Trick），即 $s_t = \mu + \sigma \odot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$。

## 代码实现：构建 RSSM 核心组件

现在，我们将上述数学公式转化为实际的代码。

(**我们首先定义 RSSM 单元，它负责根据方程进行前向推演。**)

```{.python .input}
#@tab pytorch
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

class RSSMCell(nn.Module):
    def __init__(self, action_dim, state_dim, rnn_hidden_dim):
        """
        state_dim: 随机状态 s_t 的维度
        rnn_hidden_dim: 确定性隐状态 h_t 的维度
        """
        super().__init__()
        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        
        # 确定性 RNN 更新: h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
        # 我们使用一个 GRU 单元
        self.gru = nn.GRUCell(state_dim + action_dim, rnn_hidden_dim)
        
        # 先验模型: p(s_t | h_t)
        self.prior_mlp = nn.Sequential(
            nn.Linear(rnn_hidden_dim, 256),
            nn.ELU(),
            nn.Linear(256, 2 * state_dim) # 输出均值和方差对数
        )
        
        # 后验模型: q(s_t | h_t, e_t) (假设 e_t 是观测 o_t 的特征编码)
        self.posterior_mlp = nn.Sequential(
            nn.Linear(rnn_hidden_dim + 256, 256), # 假设观测特征维度为256
            nn.ELU(),
            nn.Linear(256, 2 * state_dim)
        )

    def _split_and_transform(self, x):
        """将输出切割为均值和标准差"""
        mean, log_std = torch.chunk(x, 2, dim=-1)
        # 使用 softplus 保证标准差为正，并加上极小值防止数值不稳定
        std = nn.functional.softplus(log_std) + 0.1 
        return mean, std

    def step_prior(self, prev_state, prev_action, prev_rnn_hidden):
        """仅用先验进行推演（用于规划阶段）"""
        # 1. 拼接 s_{t-1} 和 a_{t-1}
        gru_input = torch.cat([prev_state, prev_action], dim=-1)
        # 2. 更新确定性状态 h_t
        rnn_hidden = self.gru(gru_input, prev_rnn_hidden)
        # 3. 计算先验分布 p(s_t | h_t)
        prior_stats = self.prior_mlp(rnn_hidden)
        prior_mean, prior_std = self._split_and_transform(prior_stats)
        
        # 重参数化采样
        prior_dist = Normal(prior_mean, prior_std)
        state = prior_dist.rsample()
        
        return state, rnn_hidden, prior_dist

    def step_posterior(self, prev_state, prev_action, prev_rnn_hidden, obs_embed):
        """结合观测进行推断（用于训练阶段）"""
        # 1. 同样需要先计算确定性状态 h_t
        gru_input = torch.cat([prev_state, prev_action], dim=-1)
        rnn_hidden = self.gru(gru_input, prev_rnn_hidden)
        
        # 2. 计算先验分布（计算 KL 损失时需要）
        prior_stats = self.prior_mlp(rnn_hidden)
        prior_mean, prior_std = self._split_and_transform(prior_stats)
        prior_dist = Normal(prior_mean, prior_std)
        
        # 3. 计算后验分布 q(s_t | h_t, o_t)
        post_input = torch.cat([rnn_hidden, obs_embed], dim=-1)
        post_stats = self.posterior_mlp(post_input)
        post_mean, post_std = self._split_and_transform(post_stats)
        post_dist = Normal(post_mean, post_std)
        
        # 从后验中重参数化采样得到实际使用的 s_t
        state = post_dist.rsample()
        
        return state, rnn_hidden, prior_dist, post_dist
```

这段代码精准地映射了我们之前推导的数学公式。在 `step_posterior` 中，我们同时计算了先验和后验分布，为下一步计算 KL 散度做好了准备。

## 潜在空间中的决策：交叉熵方法 (CEM)

拥有了训练好的 RSSM，我们的智能体便具备了“想象未来”的能力。接下来，我们必须回答：**如何利用这些想象来选择当前的最佳动作？**

PlaNet 采用了一种名为**交叉熵方法**（Cross-Entropy Method, CEM）的无导数优化算法。尽管它的名字带有“交叉熵”，但在这里它的核心思想更接近于进化算法或粒子滤波。

假设我们要在潜在空间中规划未来 $H$ 步。CEM 的算法流程极其优雅直观：

1. **初始化**：假设每一步的动作服从正态分布，初始时，在规划区间 $H$ 上的动作序列分布的均值为 $\mu_a = 0$，方差为 $\sigma_a^2 = I$。
2. **采样轨迹**：从当前的动作分布 $\mathcal{N}(\mu_a, \sigma_a^2)$ 中采样 $N$ 条可能的未来动作序列（每条序列长为 $H$）。
3. **在脑海中模拟**：对于每一条采样的动作序列，利用 RSSM 的**先验模型**（即前文的确定性状态更新方程和先验转移分布方程），自回归地推演未来的潜在状态 $s_{t+1}, \dots, s_{t+H}$。
4. **评估轨迹**：使用奖励模型 $r_{\tau} = \text{Reward}(s_{\tau})$ 评估这 $N$ 条轨迹的累积奖励（Return）。
5. **精英筛选与分布更新**：挑出累积奖励最高的 $K$ 条轨迹（称为“精英集”，通常 $K \ll N$）。用这 $K$ 条精英动作序列的均值和方差，来更新我们的动作分布参数 $\mu_a$ 和 $\sigma_a^2$。
6. **迭代**：重复步骤 2-5 若干次（如 10 次）。最终，最优动作分布 $\mu_a$ 的第一个时间步动作，即为智能体当前要执行的真实物理动作。执行该动作，接收新观测，重新开始规划（这是模型预测控制 MPC 的思想）。

由于我们在模拟中根本不需要生成高维图像，仅在低维潜在状态 $s_t$ 和 $h_t$ 之间进行极轻量级的矩阵乘法，CEM 可以极其快速地在一秒钟内模拟成千上万条轨迹。

(**以下是 CEM 规划算法在张量级别的大致实现框架。**)

```{.python .input}
#@tab pytorch
def cem_planning(current_state, current_rnn_hidden, rssm, reward_model, 
                 plan_horizon=12, num_samples=1000, num_elites=100, iterations=10, action_dim=6):
    """
    基于 CEM 的潜在空间规划器
    """
    # 初始化动作序列分布 (plan_horizon, action_dim)
    action_mean = torch.zeros(plan_horizon, action_dim)
    action_std = torch.ones(plan_horizon, action_dim)
    
    for opt_step in range(iterations):
        # 1. 采样 N 条动作序列 (num_samples, plan_horizon, action_dim)
        actions = Normal(action_mean, action_std).sample((num_samples,))
        
        # 约束动作到合法范围，例如 [-1, 1]
        actions = torch.clamp(actions, -1.0, 1.0)
        
        # 2. 在潜在空间中模拟推演
        returns = torch.zeros(num_samples)
        state = current_state.expand(num_samples, -1)
        rnn_hidden = current_rnn_hidden.expand(num_samples, -1)
        
        # 沿规划视界滚动
        for t in range(plan_horizon):
            action_t = actions[:, t, :]
            # 仅使用先验模型推演
            state, rnn_hidden, _ = rssm.step_prior(state, action_t, rnn_hidden)
            # 预测奖励
            reward_t = reward_model(torch.cat([state, rnn_hidden], dim=-1)).squeeze(-1)
            returns += reward_t
            
        # 3. 筛选精英并更新分布
        # 获取 top-K 轨迹的索引
        _, elite_indices = torch.topk(returns, num_elites)
        
        # 提取精英动作序列 (num_elites, plan_horizon, action_dim)
        elite_actions = actions[elite_indices]
        
        # 更新均值和标准差以用于下一轮迭代
        action_mean = elite_actions.mean(dim=0)
        action_std = elite_actions.std(dim=0)
        
    # 返回规划出的最优序列的第一步动作
    return action_mean[0]
```

正如你所见，整个规划过程是一个纯粹的张量运算流水线，这也是 PlaNet 能够打破传统模型计算瓶颈的核心秘诀。

## 小结

PlaNet 代表了基于模型强化学习范式的一次重要飞跃。它告诉我们，智能体不需要在复杂的现实细节中泥足深陷。通过构建**循环状态空间模型 (RSSM)**，智能体学会了提取一个同时具备确定性记忆和随机表达能力的抽象潜在空间。基于变分推断的训练框架保证了该空间的有效性，而无梯度的 CEM 方法则在这个紧凑的空间中提供了高效的远景规划能力。

在后续的发展中，研究者们发现 CEM 虽然灵活，但在处理高维度长周期的规划时依然存在局限。这为后来 Dreamer 算法直接在潜在空间中训练策略网络埋下了伏笔（我们将在下一节详细探讨）。
