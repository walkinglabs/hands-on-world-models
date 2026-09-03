# 4.3 PlaNet：在潜在空间中规划

许多无模型（Model-Free）强化学习方法需要大量环境交互才能学到有效策略。在模拟器中这主要消耗计算时间；在机器人等真实系统中，还会带来采集成本、设备磨损和安全约束。

基于模型（Model-Based）强化学习先从交互数据学习动力学，再用模型评估候选动作。这样可以复用已有数据进行多次规划，但收益取决于模型误差是否足够小。

当观测是图像时，逐像素滚动预测需要处理大量细节，其中一部分与奖励和控制无关。潜在模型希望把规划放到较小的表示空间中，同时在训练时仍通过观测与奖励目标约束表示。

在这样的背景下，Hafner 等人提出了**深度规划网络**（Deep Planning Network，PlaNet）[[Hafner et al., 2019]](https://arxiv.org/abs/1811.04551)。PlaNet 学习紧凑的潜在动力学，并在潜在空间中规划；训练阶段仍包含观测解码与重构似然，因此不能说它“彻底摒弃”了像素建模。为结合确定性记忆与随机状态，论文引入了**循环状态空间模型**（Recurrent State Space Model, RSSM）。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/03-planet/planet-fig10.png" alt="PlaNet 在六类视觉控制任务中从五帧上下文继续开环预测，展示学得潜在动力学实际维持运动的能力。" width="86%">

_图 4.3-1：PlaNet 在六类视觉控制任务中从五帧上下文继续开环预测，展示学得潜在动力学实际维持运动的能力。 出处：Danijar Hafner；Timothy Lillicrap；Ian Fischer；Ruben Villegas；David Ha；Honglak Lee；James Davidson，[Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)（2019），Figure 10。_

</div>

本节先回顾 PlaNet 中的 RSSM，再说明交叉熵方法（Cross-Entropy Method, CEM）如何在学到的潜在动力学中优化动作序列。

## 潜在空间动力学：为什么我们需要它？

想象你在驾驶汽车。你并不需要预测下一秒视网膜上接收到的每一个光子或每一片树叶的具体位置。相反，你的大脑提取了诸如“前方车辆的速度”、“红绿灯的状态”以及“道路的曲率”等高度抽象的概念，并在这些概念组成的空间中预测如果你踩下刹车会发生什么。

这就是**潜在空间**（Latent Space）的物理直觉。在数学上，我们将高维的观测（例如图像）$o_t \in \mathbb{R}^{H \times W \times C}$ 映射到一个低维的潜在状态变量 $s_t \in \mathbb{R}^{d}$，其中 $d \ll H \times W \times C$。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/03-planet/e2c-fig10.png" alt="E2C 三连杆机械臂的真实帧与模型预测并排，说明控制所需转移可以在低维表征中学习。" width="86%">

_图 4.3-2：E2C 三连杆机械臂的真实帧与模型预测并排，说明控制所需转移可以在低维表征中学习。 出处：Manuel Watter；Jost Tobias Springenberg；Joschka Boedecker；Martin Riedmiller，[Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images](https://arxiv.org/abs/1506.07365)（2015），Figure 10。_

</div>

直接预测 $o_{t+1}=f(o_t,a_t)$ 需要输出高维观测。潜在动力学把问题拆成几个组件：

1. **表征模型**（Representation Model）：将观测压缩为潜在状态 $s_t = \text{Encoder}(o_t)$。
2. **动力学模型**（Dynamics Model/Transition Model）：在潜在空间中预测未来 $s_{t+1} = \text{Transition}(s_t, a_t)$。
3. **奖励模型**（Reward Model）：根据当前的潜在状态预测奖励 $r_t = \text{Reward}(s_t)$。

训练时仍使用观测模型提供学习信号；规划时只需滚动动力学与奖励模型，无需把每个候选轨迹都解码成图像。

## 循环状态空间模型 (RSSM)

潜在动力学需要同时利用历史，并表达给定历史后的不确定性。

### 确定性路径：RNN 的局限性

如果我们仅仅使用一个标准的循环神经网络（RNN）来建模状态转移，我们可以写出：

$$h_t = f_{\theta}(h_{t-1}, a_{t-1}, o_t)$$

这里 $h_t$ 是 RNN 的隐藏状态。这是一种完全**确定性**（Deterministic）的建模方式。只要给定历史，未来的状态就唯一确定了。

若确定性模型配合单峰像素损失训练，多模态未来容易被折中成平均预测。也可以用更丰富的输出分布缓解这一问题，因此关键不只是“有没有 RNN”，还包括状态与似然如何参数化。

### 随机路径：引入状态空间模型

为了捕捉环境的随机性，传统的非线性状态空间模型（State Space Model, SSM）引入了随机变量 $s_t$。其转移过程由概率分布描述：

$$s_t \sim p(s_t \mid s_{t-1}, a_{t-1})$$

观测则是从当前随机状态中生成的：

$$o_t \sim p(o_t \mid s_t)$$

在 SSM 中，$s_t$ 可由高斯等分布参数化以表达不确定性。马尔可夫假设并不意味着模型只能记住一步：若状态本身足够充分，它可以携带过去信息。实践困难在于，所有长期信息都要经过随机状态逐步传递，训练与优化可能更脆弱。

### RSSM：组合确定性与随机状态

PlaNet 的 **RSSM** 把确定性循环状态与随机状态放在同一转移模型中。

我们可以将 RSSM 的动力学分解为两部分：一个确定性的隐藏状态 $h_t$，以及一个随机的后验状态 $s_t$。

具体来说，RSSM 包含以下几个核心方程：

1. **确定性状态更新** (Deterministic State Update)：
   $$h_t = f_{\text{RNN}}(h_{t-1}, s_{t-1}, a_{t-1})$$
   这里 $h_t$ 汇总截至时间步 $t$、对预测有用的历史信息。

2. **先验模型** (Prior Model / Transition Model)：
   $$p(s_t \mid h_t) = \mathcal{N}(\mu_{\text{prior}}(h_t), \sigma_{\text{prior}}(h_t))$$
   先验模型在**不看**当前观测 $o_t$ 的情况下，仅仅利用历史信息 $h_t$ 来“预测”当前时间步可能发生的随机状态 $s_t$ 的分布。

3. **后验模型** (Posterior Model / Representation Model)：
   $$q(s_t \mid h_t, o_t) = \mathcal{N}(\mu_{\text{post}}(h_t, o_t), \sigma_{\text{post}}(h_t, o_t))$$
   后验模型结合历史状态与当前观测，构造训练时使用的近似后验。

4. **奖励预测与观测重建**：
   $$r_t \sim p(r_t \mid h_t, s_t)$$
   $$o_t \sim p(o_t \mid h_t, s_t)$$

训练阶段使用观测后验估计当前状态；规划阶段没有未来观测，只能从先验逐步采样。KL 项负责缩小两种状态分布之间的差异。

## 变分目标函数：优化 RSSM

既然 RSSM 包含了复杂的随机变量，我们如何通过反向传播来训练它？答案是变分推断（Variational Inference）。我们需要最大化观测序列的边际对数似然（Marginal Log-Likelihood）的下界（ELBO）。

对于单个时间步，回忆变分自编码器（VAE）的损失函数，它由“重建误差”和“KL散度（正则化项）”组成。在序列数据中，PlaNet 最大化以下目标函数（相当于最小化损失）：

$$\mathcal{J} = \sum_{t=1}^{T} \mathbb{E}_{q(s_t \mid h_t, o_t)} \left[ \ln p(o_t \mid h_t, s_t) + \ln p(r_t \mid h_t, s_t) \right] - \beta \sum_{t=1}^{T} \text{KL}\left( q(s_t \mid h_t, o_t) \| p(s_t \mid h_t) \right)$$

这个示意目标包含三类项：

1. **观测对数似然** $\ln p(o_t\mid h_t,s_t)$：鼓励潜在状态保留解释观测所需的信息；高斯像素似然对应按尺度加权的平方误差。
2. **奖励对数似然** $\ln p(r_t\mid h_t,s_t)$：鼓励状态保留与控制目标相关的信息。
3. **KL 散度** $\mathrm{KL}(q\|p)$：让历史先验接近观测辅助后验。式中的 $\beta$ 是教学中常见的权重写法；具体 PlaNet 实现与后续工作还会使用不同的平衡和正则策略。

连续高斯状态通常使用重参数化 $s_t=\mu+\sigma\odot\epsilon$，其中 $\epsilon\sim\mathcal{N}(0,I)$，以获得路径梯度。

## 代码实现：构建 RSSM 核心组件

下面把状态更新写成代码。

先定义 RSSM 单元。

```python
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

`step_posterior` 同时返回先验与后验，便于外部计算 KL。完整 PlaNet 还需要观测编码器、解码器、奖励模型和序列损失，这里只保留状态核心。

## 潜在空间中的决策：交叉熵方法 (CEM)

训练好 RSSM 后，可以用它评估候选动作序列。PlaNet 采用**交叉熵方法**（Cross-Entropy Method, CEM）做无导数优化。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/03-planet/planet-fig12.png" alt="PlaNet 用规划地平线、候选数、精英比例和迭代次数的网格实验显示 CEM 搜索预算如何改变控制性能。" width="86%">

_图 4.3-3：PlaNet 用规划地平线、候选数、精英比例和迭代次数的网格实验显示 CEM 搜索预算如何改变控制性能。 出处：Danijar Hafner；Timothy Lillicrap；Ian Fischer；Ruben Villegas；David Ha；Honglak Lee；James Davidson，[Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)（2019），Figure 12。_

</div>

CEM 反复采样、筛选高回报样本并重估采样分布。它与粒子滤波都使用样本，但目标与更新规则不同，不应把两者视为同一种算法。

假设规划未来 $H$ 步，流程如下：

1. **初始化**：假设每一步的动作服从正态分布，初始时，在规划区间 $H$ 上的动作序列分布的均值为 $\mu_a = 0$，方差为 $\sigma_a^2 = I$。
2. **采样轨迹**：从当前的动作分布 $\mathcal{N}(\mu_a, \sigma_a^2)$ 中采样 $N$ 条可能的未来动作序列（每条序列长为 $H$）。
3. **在脑海中模拟**：对于每一条采样的动作序列，利用 RSSM 的**先验模型**（即前文的确定性状态更新方程和先验转移分布方程），自回归地推演未来的潜在状态 $s_{t+1}, \dots, s_{t+H}$。
4. **评估轨迹**：使用奖励模型 $r_{\tau} = \text{Reward}(s_{\tau})$ 评估这 $N$ 条轨迹的累积奖励（Return）。
5. **精英筛选与分布更新**：挑出累积奖励最高的 $K$ 条轨迹（称为“精英集”，通常 $K \ll N$）。用这 $K$ 条精英动作序列的均值和方差，来更新我们的动作分布参数 $\mu_a$ 和 $\sigma_a^2$。

<div align="center"><img src="/figures/04-latent-dynamics/latex/03-planet/cem-elite-refit-loop.png" alt="从动作序列分布采样并在潜空间评分，选出高回报精英后用其均值和方差更新下一轮分布" width="86%">

_图 4.3-4：CEM 用当前分布采样候选序列，按想象回报选出 top-K 精英，再以精英样本矩重估下一轮分布；这一更新不需要奖励梯度。本文根据上述步骤绘制。_

</div>

6. **迭代**：重复步骤 2-5 若干次（如 10 次）。最终，最优动作分布 $\mu_a$ 的第一个时间步动作，即为智能体当前要执行的真实物理动作。执行该动作，接收新观测，重新开始规划（这是模型预测控制 MPC 的思想）。

因为候选轨迹无需解码成图像，单次滚动通常比像素模型便宜。但实际速度取决于样本数、规划视界、模型大小与硬件，不能统一承诺“一秒钟内”完成多少轨迹。

下面给出张量级教学实现。

```python
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
        action_std = elite_actions.std(dim=0).clamp_min(1e-4)

    # 返回规划出的最优序列的第一步动作
    return action_mean[0]
```

实际 MPC 还会处理动作边界、折扣、终止信号与分布平滑；这段代码只展示 CEM 的采样—筛选—重估主循环。

## 小结

PlaNet 用 RSSM 学习确定性记忆与随机状态，并在潜在空间用 CEM 规划。变分目标提供观测与奖励学习信号，但模型误差仍会随规划视界累积；CEM 也需要大量候选轨迹。因此，这套方法是在样本复用、模型偏差和在线规划开销之间做权衡。

<div align="center">
  <img src="/figures/04-latent-dynamics/source/03-planet/pilco-fig11.png" alt="PILCO 在真实小车倒立摆上的连续快照展示早期数据高效模型式控制如何落到物理系统。" width="86%">

_图 4.3-5：PILCO 在真实小车倒立摆上的连续快照展示早期数据高效模型式控制如何落到物理系统。 出处：Marc Peter Deisenroth；Diether Fox；Carl Edward Rasmussen，[PILCO: A Model-Based and Data-Efficient Approach to Policy Search](https://doi.org/10.1109/TPAMI.2013.218)（2015），Figure 11。_

</div>

在后续的发展中，研究者们发现 CEM 虽然灵活，但在处理高维度长周期的规划时依然存在局限。这为后来 Dreamer 算法直接在潜在空间中训练策略网络埋下了伏笔（我们将在下一节详细探讨）。
